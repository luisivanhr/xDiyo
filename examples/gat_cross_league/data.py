"""Resolve the catalog and construct causal populations without extending dates."""
from dataclasses import asdict
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
from . import core
from compare_gat_temporal import digest, write_json, frame_hash
from xdiyo_analytics.data import load_season
from xdiyo_analytics.histories import build_team_history
from xdiyo_analytics.graphs import TemporalEventStore
from xdiyo_analytics.graphs.config import MATCH_KEYS
from xdiyo_analytics.graphs.goals import goal_labels


def eligible(frame):
    y = frame[['goals_for','goals_against']].to_numpy(dtype=float, na_value=np.nan)
    return (frame.status.eq('finished').fillna(False).to_numpy(bool)
            & ~frame.is_awarded.eq(True).fillna(False).to_numpy(bool)
            & np.isfinite(y).all(1) & (y >= 0).all(1) & (y == np.floor(y)).all(1))


def within_window(store, frame, start, end, cfg):
    issuance = store.issue_times(frame, cfg)
    available = store.label_availability(frame, cfg)
    return eligible(frame) & issuance.ge(start).to_numpy() & available.le(end).to_numpy()


def prepare(run, handoff):
    run = Path(run)
    if (run/'FREEZE.json').exists():
        raise ValueError('Frozen experiment cannot be prepared again')
    run.mkdir(parents=True, exist_ok=True)
    if (run/'HANDOFF.md').exists():
        assert digest(run/'HANDOFF.md') == digest(handoff), 'Different assignment in output directory'
    else:
        assert not list(run.iterdir()), 'Output is not empty'
        shutil.copyfile(handoff,run/'HANDOFF.md')
    core.status(run, stage='inventory', completed=0, planned=36, worker=None)
    for folder in ('shared/data','shared/populations','shared/checks','shared/preprocessing',
                   'shared/initialization','shared/bootstrap','03_baselines','04_comparisons'):
        (run/folder).mkdir(parents=True, exist_ok=True)
    # Preserve all previous evidence, including the parent completion receipt.
    previous = dict(json.loads((core.PARENT/'preflight/preserved_artifacts.json').read_text()))
    previous.update({p.relative_to(core.ROOT).as_posix():digest(p)
                     for p in core.PARENT.rglob('*') if p.is_file()})
    write_json(run/'shared/checks/protected_artifacts.json', previous)
    catalog_path = core.ROOT/'scraping/sofascore/competition_registry.json'
    catalog = json.loads(catalog_path.read_text())
    assert len(catalog) == 13 and catalog['Bundesliga']['competition_id'] == 35
    registry = sorted(x['competition_id'] for x in catalog.values())
    cfg = core.config()
    parent_protocol = json.loads((core.PARENT/'MASTER_PROTOCOL.json').read_text())
    parent_data = core.PARENT/'shared/data'
    old_history = pd.read_parquet(parent_data/'history.parquet')
    old_memberships = pd.read_parquet(parent_data/'memberships.parquet')
    old_schedule = pd.read_parquet(parent_data/'schedule.parquet')
    parent_store = TemporalEventStore(old_history,old_memberships,schedule=old_schedule,
                                     provenance=parent_protocol['provenance'])
    lower = old_history.kickoff_at.min()
    upper = max(pd.read_parquet(parent_data/f'{fold}_outer.parquet').kickoff_at.max() for fold in core.FOLDS)
    histories, schedules, memberships = [old_history], [old_schedule], [old_memberships]
    sources, membership_evidence, exclusions = [], [], []
    data_root = core.ROOT/'data/xDiyo_data'
    countries = {v['competition_id']:v['system'] for v in catalog.values()}
    for league, item in catalog.items():
        if item['competition_id'] in (35,44):
            continue
        for year in range(19,25):
            stem = f'{league}_{year:02}_{year+1:02}'
            data = load_season(data_root,stem,tables=('matches','team_seasons'),
                               verify_hashes=True,include_awarded=True)
            h = build_team_history(data)
            h.attrs = {}
            h['source_league'],h['source_season'] = league,f'{year:02}_{year+1:02}'
            if h.kickoff_at.isna().any() or h['round'].isna().any():
                raise ValueError(f'Missing original schedule timestamp/round: {stem}')
            s = h[h.side.eq('home')].rename(columns={'team_id':'home_id','opponent_id':'away_id'}).copy()
            # Original anchors use the full publication, including excluded targets.
            anchors = s.groupby(list(cfg.round_keys)).kickoff_at.transform('min')
            s['round_issue_at'] = anchors
            member = data['team_seasons'].copy()
            member.attrs = {}
            issue_map = dict(zip(core.keys(s),anchors))
            team_first = h.assign(_issue=[issue_map[k] for k in core.keys(h)]).groupby(
                ['team_id','competition_id','season_id'])._issue.min()
            for row in member.itertuples():
                key = (row.team_id,row.competition_id,row.season_id)
                if key not in team_first:
                    continue
                # Conservative known-by-issuance boundary: never infer a July 1
                # switch while a delayed prior season is still being played.
                membership_evidence.append(dict(team_id=int(row.team_id),competition_id=int(row.competition_id),
                    season_id=int(row.season_id),entry_at=team_first[key],movement=row.movement,
                    source=stem,evidence=getattr(row,'evidence',''),
                    rule='First original full-round issuance containing this team in this season; no earlier backfill'))
            keep = h.kickoff_at.ge(lower) & ~h.is_awarded.eq(True).fillna(False)
            histories.append(h[keep].copy())
            schedules.append(s)
            exclusions.append(dict(source=stem,awarded=int(s.is_awarded.eq(True).sum()),
                before_history_start=int((h.side.eq('home') & h.kickoff_at.lt(lower)).sum()),
                unfinished_or_invalid=int((~eligible(s)).sum())))
            source = dict(data.provenance)
            source.update(manifest_sha256=digest(source['manifest_path']))
            sources.append(source)
    foreign_memberships = pd.DataFrame(membership_evidence)
    foreign_memberships.to_csv(run/'shared/checks/membership_boundaries.csv',index=False)
    memberships.append(foreign_memberships[list(old_memberships.columns)].copy())
    # Preserve the German schedule exactly; explicit anchor values are its old
    # full-schedule anchors, not recomputed from an eventual target subset.
    schedules[0] = old_schedule.assign(round_issue_at=parent_store.issue_times(old_schedule,cfg).to_numpy())
    for frames in (histories,schedules,memberships):
        for frame in frames:
            for column in (*MATCH_KEYS,'team_id','opponent_id','home_id','away_id'):
                if column in frame:
                    frame[column] = pd.array(frame[column].tolist(),dtype='UInt64')
    history = pd.concat(histories,ignore_index=True)
    schedule = pd.concat(schedules,ignore_index=True)
    member = pd.concat(memberships,ignore_index=True)
    teams = history[['team_id','competition_id']].drop_duplicates()
    teams['country'] = teams.competition_id.map(countries)
    collisions = teams.groupby('team_id').country.nunique()
    assert not collisions.gt(1).any(), 'Cross-country provider identity collision requires review'
    assert not member.duplicated(['team_id','entry_at']).any(), 'Ambiguous simultaneous membership boundaries'
    provenance = dict(parent=parent_protocol['provenance'],foreign_sources=sources,
        foreign_membership_policy='Reviewed movement metadata at first original team-round issuance; retrospective schedule assumption',
        catalog_sha256=digest(catalog_path),history_lower_bound=lower.isoformat())
    store = TemporalEventStore(history,member,schedule=schedule,provenance=provenance)
    for name,frame in [('history',store.history),('memberships',store.team_seasons),('schedule',store.schedule)]:
        frame.to_parquet(run/'shared/data'/f'{name}.parquet',index=False)
    pd.DataFrame(exclusions).to_csv(run/'shared/checks/source_exclusions.csv',index=False)
    write_json(run/'shared/checks/source_versions.json',sources)
    schedule = store.schedule
    issue = store.issue_times(schedule,cfg)
    available = store.label_availability(schedule,cfg)
    receipts, counts = {}, []
    for fold in core.FOLDS:
        receipts[fold] = {}
        previous = None
        for phase,parent_phase in [('fit','expanded'),('inner','inner'),('outer','outer')]:
            local = pd.read_parquet(parent_data/f'{fold}_{parent_phase}.parquet')
            original_issue = parent_store.issue_times(local,cfg)
            pd.testing.assert_series_equal(store.issue_times(local,cfg),original_issue)
            original_available = parent_store.label_availability(local,cfg)
            pd.testing.assert_series_equal(store.label_availability(local,cfg),original_available)
            start,end = original_issue.min(),original_available.max()
            take = within_window(store,schedule,start,end,cfg) & schedule.competition_id.ne(35).to_numpy()
            foreign = schedule[take].copy()
            # Keep covariates and row ordering in the local set byte-for-byte.
            shutil.copyfile(parent_data/f'{fold}_{parent_phase}.parquet',run/'shared/populations'/f'{fold}_local_{phase}.parquet')
            pooled = pd.concat([local,foreign],ignore_index=True).sort_values(['kickoff_at',*MATCH_KEYS],kind='stable').reset_index(drop=True)
            assert not core.keys(pooled).duplicated().any()
            keep_bl = pooled[pooled.competition_id.eq(35)].set_index(list(MATCH_KEYS)).reindex(core.keys(local)).reset_index()
            for column in local:
                pd.testing.assert_series_equal(keep_bl[column].reset_index(drop=True),local[column].reset_index(drop=True),
                                               check_dtype=False,check_names=False)
            pooled.to_parquet(run/'shared/populations'/f'{fold}_pooled_{phase}.parquet',index=False)
            pooled_issue = store.issue_times(pooled,cfg)
            store.assert_targets_unobserved(pooled,pooled_issue,cfg)
            known = store.label_availability(pooled,cfg)
            first_cutoff = min(cfg.history_cutoff(t) for t in pooled_issue)
            assert previous is None or previous <= first_cutoff, (fold,phase,previous,first_cutoff)
            previous = known.max()
            goal_labels(store,pooled,known.max())
            receipts[fold][phase] = dict(start=start.isoformat(),end=end.isoformat(),
                first_cutoff=first_cutoff.isoformat(),local_count=len(local),pooled_count=len(pooled),
                local_hash=frame_hash(local),pooled_hash=frame_hash(pooled),issuance_groups=pooled_issue.nunique())
            for league,item in catalog.items():
                part=pooled[pooled.competition_id.eq(item['competition_id'])]
                counts.append(dict(fold=fold,phase=phase,competition_id=item['competition_id'],league=league,
                    country=item['system'],division=item['tier'],count=len(part),
                    share=len(part)/len(pooled),home_wins=int((core.truth(part)==0).sum()),
                    draws=int((core.truth(part)==1).sum()),away_wins=int((core.truth(part)==2).sum()),
                    first_kickoff=str(part.kickoff_at.min()),last_kickoff=str(part.kickoff_at.max()),
                    neutral_known=int(part.get('neutral_venue',pd.Series(np.nan,index=part.index)).notna().sum()),
                    before_issuance_boundary=int((schedule.competition_id.eq(item['competition_id']) & eligible(schedule) & issue.lt(start)).sum()),
                    after_label_boundary=int((schedule.competition_id.eq(item['competition_id']) & eligible(schedule) & issue.ge(start) & available.gt(end)).sum())))
        for arm in core.ARMS:
            (run/core.DIRECTORIES[arm]).mkdir(exist_ok=True)
    inventory = pd.DataFrame(counts)
    inventory.to_csv(run/'league_inventory.csv',index=False)
    assert inventory[inventory.phase.eq('fit')]['count'].gt(0).all(), 'An all-13 fitting league has no targets'
    # Primary resamples are inherited unchanged, not regenerated.
    for fold in core.FOLDS:
        shutil.copytree(core.PARENT/'shared/bootstrap'/fold,run/'shared/bootstrap/bundesliga'/fold,dirs_exist_ok=True)
    protocol = dict(schema=1,assignment_sha256=digest(run/'HANDOFF.md'),matrix=core.matrix(),
        parent=str(core.PARENT),parent_completion_sha256=digest(core.PARENT/'COMPLETE.json'),
        catalog=catalog,catalog_sha256=digest(catalog_path),competitions=registry,folds=receipts,
        provenance=provenance,store_version=store.version,history_start=lower.isoformat(),
        configurations={k:asdict(core.config(k)) for k in core.CONFIGS},
        resources=dict(device='cuda:0',threads=1,input_cache_bytes=4*2**30,headroom_bytes=2**30,
                       rss_limit_gib=8.,cuda_fraction=.8,complete_components=False),
        training=dict(max_steps=200,patience=15,min_delta=0.,monitor='raw Bundesliga inner CE',
                      restore_best=True,scheduler=None,full_population_denominator=True),
        bootstrap=dict(primary=dict(seed=20261026,replicates=20000,blocks=[2,4,8]),
                       secondary=dict(seed=20261027,replicates=20000,blocks=[2,4,8],week='Monday Europe/Berlin',
                                      rejection_cap=1000000,original_mixture_weights=True)),
        limits=['Previously inspected development periods', 'Retrospective full schedules and reviewed membership metadata',
                'German membership and archive rows preserved; foreign effective boundaries conservatively delayed to first team-round issuance',
                'No artificial graph edges; shared parameters transfer across disconnected domestic competitions',
                'Bundesliga-selected checkpoints applied to secondary leagues'])
    write_json(run/'PROTOCOL.json',protocol)
    pd.DataFrame([dict(**row,status='planned',provenance='new') for row in core.matrix()]).to_csv(run/'RUN_MANIFEST.csv',index=False)
    core.status(run,stage='populations_prepared')
    print(json.dumps(dict(stage='populations_prepared',history=len(history)//2,cohort=catalog,
                         folds=receipts)),flush=True)
