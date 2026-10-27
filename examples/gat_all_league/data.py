"""Rebuild clocks and populations from the preserved full archived schedule."""
from dataclasses import asdict
from pathlib import Path
import json,shutil
import numpy as np
import pandas as pd
from . import core
from .stages import repair_schedule,attach_schedule,stage_inventory,BELGIAN_STAGES
from gat_cross_league import core as previous_core
from gat_cross_league.data import within_window
from gat_cross_league.bootstrap import week_groups,calendar_samples
from gat_cross_league.initialization import feature_names
from xdiyo_analytics.graphs.preprocessing import GraphPreprocessor

EXPECTED={'fold_1':dict(fit=9177,inner=4593,outer=4315),'fold_2':dict(fit=13974,inner=4315,outer=4275)}

def prepare(run,handoff):
    run=Path(run)
    if (run/'FREEZE.json').exists():raise ValueError('Cannot prepare a frozen run')
    run.mkdir(parents=True,exist_ok=True)
    if (run/'HANDOFF.md').exists():assert core.digest(run/'HANDOFF.md')==core.digest(handoff)
    else:
        assert not list(run.iterdir()),'Output directory is not empty'
        shutil.copyfile(handoff,run/'HANDOFF.md')
    for name in ('shared/data','shared/populations','shared/preprocessing','shared/initialization','shared/bootstrap',
                 'shared/checks','01_pooled','02_baselines','03_comparisons'):(run/name).mkdir(parents=True,exist_ok=True)
    core.status(run,stage='preparing',completed=0,planned=18,worker=None)
    parent,old_store=previous_core.load(core.PARENT,frozen=True)
    old_receipt=json.loads((core.PARENT/'COMPLETE.json').read_text())
    for relative,expected in old_receipt['files'].items():assert core.digest(core.PARENT/relative)==expected,relative
    protected=json.loads((core.PARENT/'shared/checks/protected_artifacts.json').read_text())
    protected.update({f.relative_to(core.ROOT).as_posix():core.digest(f) for f in core.PARENT.rglob('*') if f.is_file()})
    core.write_json(run/'shared/checks/protected_artifacts.json',protected)
    before=old_store.schedule;after=repair_schedule(before)
    stage_inventory(after).to_csv(run/'shared/checks/stage_mapping.csv',index=False)
    changed=before.round_issue_at.ne(after.round_issue_at)
    assert after.loc[changed,'competition_id'].eq(38).all()
    clock_changes=after.loc[changed,[*core.MATCH_KEYS,'tournament_id','round_stage','round','kickoff_at','round_issue_at']].copy()
    clock_changes['previous_round_issue_at']=before.loc[changed,'round_issue_at'].to_numpy()
    clock_changes.to_csv(run/'shared/checks/clock_changes.csv',index=False)
    history=attach_schedule(old_store.history,after)
    memberships=old_store.team_seasons
    teams=pd.concat([after[['home_id','competition_id','season_id','round_issue_at']].rename(columns={'home_id':'team_id'}),
                     after[['away_id','competition_id','season_id','round_issue_at']].rename(columns={'away_id':'team_id'})])
    first=teams.groupby(['team_id','competition_id','season_id']).round_issue_at.min()
    member_changes=[]
    for i,row in memberships.iterrows():
        if row.competition_id in (35,44):continue
        issue=first.loc[(row.team_id,row.competition_id,row.season_id)]
        if issue!=row.entry_at:
            assert row.competition_id==38,'Unexpected non-Belgian membership change'
            member_changes.append(dict(team_id=int(row.team_id),competition_id=int(row.competition_id),season_id=int(row.season_id),
                previous_entry=row.entry_at,corrected_entry=issue))
            memberships.loc[i,'entry_at']=issue
    pd.DataFrame(member_changes,columns=['team_id','competition_id','season_id','previous_entry','corrected_entry']).to_csv(
        run/'shared/checks/membership_changes.csv',index=False)
    provenance=dict(parent=parent['provenance'],parent_archive=str(core.PARENT),parent_completion_sha256=core.digest(core.PARENT/'COMPLETE.json'),
        round_stage_policy='Belgium: validated tournament identity; all other competitions: constant zero',
        anchor_policy='Rebuild Belgian anchors from full archived schedule before target filtering; retain other explicit anchors')
    store=core.ComponentStore(history,memberships,schedule=after,provenance=provenance)
    assert store.version!=old_store.version,'Prepared inputs must not reuse the old context identity'
    for name,frame in [('history',store.history),('schedule',store.schedule),('memberships',store.team_seasons)]:
        frame.to_parquet(run/'shared/data'/f'{name}.parquet',index=False)
    receipts={};changes=[];inventory=[]
    for fold in core.FOLDS:
        receipts[fold]={};last_boundary=None
        for phase in ('fit','inner','outer'):
            window=parent['folds'][fold][phase]
            start,end=pd.Timestamp(window['start']),pd.Timestamp(window['end'])
            local=attach_schedule(pd.read_parquet(core.PARENT/'shared/populations'/f'{fold}_local_{phase}.parquet'),after)
            take=within_window(store,after,start,end,core.config()) & after.competition_id.ne(35).to_numpy()
            population=pd.concat([local,after.loc[take]],ignore_index=True).sort_values(['kickoff_at',*core.MATCH_KEYS],kind='stable').reset_index(drop=True)
            assert core.keys(population).is_unique
            old=pd.read_parquet(core.PARENT/'shared/populations'/f'{fold}_pooled_{phase}.parquet')
            added=core.keys(population).difference(core.keys(old));removed=core.keys(old).difference(core.keys(population))
            assert not len(removed),'Repair unexpectedly removed eligible fixtures'
            for key in added:
                row=population.set_index(list(core.MATCH_KEYS)).loc[key]
                assert int(key[0])==38 and int(row.round_stage)!=38,'Addition is not a Belgian playoff fixture'
                changes.append(dict(fold=fold,phase=phase,change='added',**dict(zip(core.MATCH_KEYS,map(int,key))),
                    tournament_id=int(row.tournament_id),round_stage=int(row.round_stage),round=int(row['round']),
                    kickoff_at=row.kickoff_at,round_issue_at=row.round_issue_at))
            assert len(population)==EXPECTED[fold][phase],(fold,phase,len(population),EXPECTED[fold][phase])
            issue=store.issue_times(population,core.config());known=store.label_availability(population,core.config())
            store.assert_targets_unobserved(population,issue,core.config())
            first_cutoff=min(core.config().history_cutoff(t) for t in issue)
            assert last_boundary is None or last_boundary<=first_cutoff
            last_boundary=known.max()
            population.to_parquet(run/'shared/populations'/f'{fold}_{phase}.parquet',index=False)
            receipts[fold][phase]=dict(start=start.isoformat(),end=end.isoformat(),first_cutoff=first_cutoff.isoformat(),
                count=len(population),previous_count=len(old),added=len(added),removed=0,identity=core.frame_hash(population),
                issuance_groups=issue.nunique(),label_boundary=known.max().isoformat())
            for label,item in parent['catalog'].items():
                part=population[population.competition_id.eq(item['competition_id'])];y=core.truth(part)
                inventory.append(dict(fold=fold,phase=phase,league=label,competition_id=item['competition_id'],count=len(part),
                    share=len(part)/len(population),home_fraction=float((y==0).mean()),draw_fraction=float((y==1).mean()),away_fraction=float((y==2).mean()),
                    stage_count=part.round_stage.nunique(),playoff_count=int((part.competition_id.eq(38)&part.round_stage.ne(38)).sum()),
                    first_kickoff=str(part.kickoff_at.min()),last_kickoff=str(part.kickoff_at.max())))
    pd.DataFrame(changes).to_csv(run/'shared/checks/population_changes.csv',index=False)
    pd.DataFrame(inventory).to_csv(run/'league_inventory.csv',index=False)
    initialization=[]
    for fold in core.FOLDS:
        source=core.PARENT/'shared/preprocessing'/f'{fold}.json'
        shutil.copyfile(source,run/'shared/preprocessing'/f'{fold}.json')
        state=json.loads(source.read_text())['state']
        for name in core.CONFIGS:
            before_pre=GraphPreprocessor.from_state(previous_core.config(name),state)
            after_pre=GraphPreprocessor.from_state(core.config(name),state)
            assert feature_names(before_pre)==feature_names(after_pre),'Grouping entered a predictive feature'
        for seed in core.SEEDS:
            source=core.PARENT/'shared/initialization'/fold/f'seed_{seed}'
            destination=run/'shared/initialization'/fold/f'seed_{seed}';destination.mkdir(parents=True,exist_ok=True)
            for mode in ('snapshot','time_features'):
                shutil.copyfile(source/f'{mode}.pt',destination/f'{mode}.pt')
                initialization.append(dict(fold=fold,seed=seed,mode=mode,source=str(source/f'{mode}.pt'),sha256=core.digest(destination/f'{mode}.pt')))
    core.write_json(run/'shared/checks/initialization_reuse.json',initialization)
    protocol=dict(schema=1,assignment_sha256=core.digest(run/'HANDOFF.md'),parent=str(core.PARENT),matrix=core.matrix(),
        catalog=parent['catalog'],competitions=parent['competitions'],folds=receipts,store_version=store.version,previous_store_version=old_store.version,
        provenance=provenance,history_start=parent['history_start'],stage_mapping=BELGIAN_STAGES,
        configurations={name:asdict(core.config(name)) for name in core.CONFIGS},resources=parent['resources'],
        training=dict(max_steps=200,patience=15,min_delta=0.,monitor=core.MONITOR,restore_best=True,scheduler=None,
            objective='Global per-fixture outcome CE + beta times summed marginal goal loss + native prior penalty'),
        configuration_selection=dict(population='own fold all13 inner',score='Mean selected raw inner micro CE across three seeds',
            exact_tie_order=list(core.CONFIGS),outer_gate='Seal all fold choices before new outer scoring',single_recommendation='Latest fold inner winner'),
        bootstrap=dict(seed=20261027,replicates=20000,blocks=[2,4,8],primary_block=4,week='Monday Europe/Berlin',
            rejection_cap=1000000,original_fixed_mixture=True),
        metrics=dict(primary='raw all13 outer micro H/D/A CE',RPS=dict(role='secondary',normalization='1/(3-1)',
            formula='0.5 * sum_{k=0,1} (sum_{j<=k} p_j - 1[y<=k])**2',class_order=['home','draw','away'])),
        limitations=['Previously inspected development periods','Timing and checkpoint selection change jointly; their separate effects are not identified'])
    core.write_json(run/'PROTOCOL.json',protocol)
    core.write_json(run/'shared/checks/stage_repair.json',dict(changed_clocks=int(changed.sum()),non_belgian_changed_clocks=0,
        membership_changes=len(member_changes),old_store=old_store.version,new_store=store.version,counts=receipts,
        examples=after[after.event_id.isin([11234494,13631351])][[*core.MATCH_KEYS,'tournament_id','round_stage','round','kickoff_at','round_issue_at']].to_dict('records')))
    rng=np.random.default_rng(20261027);bootstrap_receipts=[]
    for fold in core.FOLDS:
        frame=core.populations(run,fold)['outer'];issue=store.issue_times(frame,core.config());weeks,positions=week_groups(issue)
        counts=np.column_stack([np.bincount(positions,weights=frame.competition_id.eq(league),minlength=len(weeks)) for league in protocol['competitions']]).astype(np.int64)
        for _,indices in frame.groupby(list(core.ROUND_KEYS)).indices.items():assert len(set(positions[indices]))==1
        folder=run/'shared/bootstrap'/fold;folder.mkdir(exist_ok=True)
        core.write_json(folder/'weeks.json',[v.isoformat() for v in weeks]);np.save(folder/'fixture_week.npy',positions);np.save(folder/'counts_by_league.npy',counts)
        for length in (2,4,8):
            weights,receipt=calendar_samples(counts,length,rng)
            assert weights is not None,'Supported original cohort could not be resampled'
            np.save(folder/f'blocks_{length}.npy',weights);bootstrap_receipts.append(dict(fold=fold,block_weeks=length,**receipt))
    core.write_json(run/'shared/checks/bootstrap.json',bootstrap_receipts)
    pd.DataFrame(core.matrix()).assign(status='pending').to_csv(run/'RUN_MANIFEST.csv',index=False)
    core.status(run,stage='prepared')
    print(json.dumps(dict(stage='prepared',run=str(run),changed_clocks=int(changed.sum()),membership_changes=len(member_changes),counts=EXPECTED)))
