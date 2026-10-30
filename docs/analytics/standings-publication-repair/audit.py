"""Read-only saved-bank attribution and independent standings reconstruction."""
import argparse, hashlib, json, subprocess, sys, types
from pathlib import Path
import numpy as np
import pandas as pd
from xdiyo_analytics.ratings import build_standings
from xdiyo_analytics.ratings.standings import TOTALS, POLICY
from xdiyo_analytics.features import DerivedStandings
from xdiyo_analytics.features.history import availability_times
BASE='4266575f62c899f3c6b3c7ca95a76fd633f93ece'
CASES=['La_Liga_20_21','Premier_League_20_21','Eredivisie_20_21','Serie_B_23_24']
WINDOWS=[1,5,10,20]

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def independent(h,m,run,release):
    """Rebuild each table directly; no production ranks, accumulators or windows."""
    allowed_release=pd.to_datetime(release,utc=True)
    rows=0
    for q,table in run.snapshots.groupby('recorded_at'):
        good=h.status.eq('finished')&h.goals_for.notna()&h.goals_against.notna()&h.kickoff_at.lt(q)&allowed_release.le(q)
        totals={}
        for t in m.team_id:
            games=h.loc[good&h.team_id.eq(t)].sort_values(['kickoff_at','event_id'])
            for w in [0,*WINDOWS]:
                g=games.tail(w) if w else games
                gf=g.goals_for.to_numpy(dtype=int);ga=g.goals_against.to_numpy(dtype=int)
                wins=int((gf>ga).sum());draws=int((gf==ga).sum())
                totals[t,w]=[len(g),wins,draws,len(g)-wins-draws,3*wins+draws,int(gf.sum()),int(ga.sum()),int((gf-ga).sum())]
        for r in table.itertuples():
            v=totals[r.team_id,r.window];key=(v[4],v[7],v[5],-v[6])
            rank=1+sum((z[4],z[7],z[5],-z[6])>key for (t,w),z in totals.items() if w==r.window)
            assert r.rank==rank and [getattr(r,c) for c in TOTALS]==v,(q,r.team_id,r.window)
            rows+=1
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=True,parents=True);root=a.experiment_root
    watched={}
    def read_json(path):
        watched[str(path)]=digest(path);return json.loads(path.read_text())
    def read_table(path):
        watched[str(path)]=digest(path);return pd.read_parquet(path)
    raw=subprocess.check_output(['git','show',BASE+':src/xdiyo_analytics/ratings/standings.py'])
    baseline=types.ModuleType('xdiyo_analytics.ratings._old_standings');baseline.__package__='xdiyo_analytics.ratings';exec(compile(raw,'frozen_standings.py','exec'),baseline.__dict__)
    bank=read_table(root/'shared/features/phase_context.parquet');bank_meta=read_json(root/'shared/features/phase_context.json')
    summaries=[];cutoff_rows=[];changes=[];samples=[];bank_deltas=[]
    for name in CASES:
        h=read_table(root/f'shared/native_sources/{name}_history.parquet');m=read_table(root/f'shared/native_sources/{name}_memberships.parquet')
        saved=read_table(root/f'shared/standings/{name}/snapshots.parquet');meta=read_json(root/f'shared/standings/{name}/metadata.json')
        group=bank.loc[bank.competition_id.eq(h.competition_id.iloc[0])&bank.season_id.eq(h.season_id.iloc[0])]
        assert not group.event_id.duplicated().any()
        cutoffs=h.event_id.map(group.set_index('event_id').cutoff_at)
        assert cutoffs.notna().all()
        # Reproduce the saved experiment's explicitly labelled legacy proxy ONLY
        # to isolate the publication-gate effect. This is not release-time proof.
        release=availability_times(h,h.kickoff_at)
        old=baseline.build_standings(h,team_seasons=m,normalized_ranks=WINDOWS,cutoffs=cutoffs,available_at=release)
        keys=['competition_id','season_id','recorded_at','window','team_id']
        sort=lambda f:f.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(sort(old.snapshots)[saved.columns],sort(saved),check_dtype=False)
        assert old.metadata['source_sha256']==meta['source_sha256']
        new=build_standings(h,team_seasons=m,normalized_ranks=WINDOWS,cutoffs=cutoffs,available_at=release)
        b,c=sort(old.snapshots),sort(new.snapshots)
        assert b[keys].equals(c[keys])
        rank_changed=b['rank'].ne(c['rank']);total_changed=b[list(TOTALS)].ne(c[list(TOTALS)]).any(axis=1)
        d=c[keys+['rank',*TOTALS]].copy();d['old_rank']=b['rank'];d['rank_changed']=rank_changed;d['totals_changed']=total_changed;d['case']=name
        changes.append(d.loc[rank_changed|total_changed])
        checks=independent(h,m,new,release)
        # Separate conservative explicit replay; no automatic kickoff+3h policy
        # is installed. Reviewed completion bounds remain enforced.
        conservative_release=availability_times(h,h.kickoff_at+pd.Timedelta(hours=3))
        bounded=build_standings(h,team_seasons=m,normalized_ranks=WINDOWS,cutoffs=cutoffs,available_at=conservative_release)
        checks+=independent(h,m,bounded,conservative_release)
        bounded_table=sort(bounded.snapshots)
        first=set(h.loc[h['round'].eq(1),'event_id'])
        for pub,oldpub in zip(new.publications,old.publications):
            q=pd.Timestamp(pub['cutoff']);available=h.loc[h.event_id.isin(pub['contributing_events'])]
            cutoff_rows.append(dict(case=name,cutoff=pub['cutoff'],old_gate_open=oldpub['gate_open'],eligible_results=len(pub['contributing_events']),pending_original_openers=json.dumps(sorted(map(int,first-set(pub['contributing_events'])))),latest_source_kickoff=None if available.empty else str(available.kickoff_at.max()),latest_assumed_release=None if available.empty else str(release.loc[available.index].max()),contributing_events=json.dumps(list(map(int,pub['contributing_events']))),availability='legacy_proxy_attribution_only'))
        affected_q=c.loc[rank_changed,'recorded_at'].unique()
        for q in list(affected_q[:1])+list(affected_q[-1:]):
            sample=d.loc[d.recorded_at.eq(q)&d.window.isin([0,5])].copy();sample['team_name']=sample.team_id.map(h.drop_duplicates('team_id').set_index('team_id').team_name);samples.append(sample)
        # Compare actual saved bank coordinates and exact original cutoffs.
        values=new.features(h,DerivedStandings(WINDOWS),cutoffs=cutoffs)
        for i,row in enumerate(h.itertuples(index=False)):
            original=group.loc[group.event_id.eq(row.event_id)].iloc[0]
            for w in WINDOWS:
                before=float(original[f'{row.side}_rank_{w}']);after=float(values.iloc[i][f'team::form_rank_{w}'])
                if before!=after:bank_deltas.append(dict(case=name,competition_id=int(row.competition_id),season_id=int(row.season_id),event_id=int(row.event_id),side=row.side,window=w,cutoff=str(cutoffs.iloc[i]),old=before,new=after))
        summaries.append(dict(case=name,fixtures=h.event_id.nunique(),teams=len(m),cutoffs=len(new.publications),snapshot_rows=len(c),rank_changed_rows=int(rank_changed.sum()),totals_changed_rows=int(total_changed.sum()),changed_publications=int(c.loc[rank_changed,'recorded_at'].nunique()),independent_rows_checked=checks,saved_snapshot_reproduction=True,legacy_proxy_source_hash=old.metadata['source_sha256'],saved_source_hash=meta['source_sha256'],conservative_3h_additional_rank_changes=int(c['rank'].ne(bounded_table['rank']).sum()),availability_limitation='Legacy proxy reproduced only for attribution; +3h replay is an explicit conservative assumption, not measured publication evidence.'))
        print(name,summaries[-1],flush=True)
    pd.DataFrame(summaries).to_csv(a.output/'season-summary.csv',index=False)
    pd.DataFrame(cutoff_rows).to_csv(a.output/'cutoff-evidence.csv',index=False)
    pd.concat(changes).to_csv(a.output/'changed-snapshots.csv',index=False)
    pd.concat(samples).to_csv(a.output/'independently-checked-sample-tables.csv',index=False)
    delta=pd.DataFrame(bank_deltas);delta.to_csv(a.output/'changed-bank-coordinates.csv',index=False)
    inventory=[]
    for path in sorted(root.glob('*/**/model/graph.json')):
        if 'source_frozen' in path.parts or 'reporting_source' in path.parts:continue
        body=read_json(path);schema=body.get('phase_rank_schema') or {};cols=schema.get('fixture_columns',[])+schema.get('edge_columns',[])
        rank_cols=[x for x in cols if 'rank' in x]
        cfg=body.get('config',{}).get('phase_rank_features') or {}
        inventory.append(dict(path=str(path.relative_to(root)),route=cfg.get('route','N'),window=cfg.get('window'),source_id=cfg.get('source_id'),rank_coordinates=json.dumps(rank_cols),action='regenerate_features_and_retrain_before_corrected_comparison' if rank_cols else 'no_direct_rank_dependency_preserve_control',graph_sha256=digest(path)))
    pd.DataFrame(inventory).to_csv(a.output/'model-inventory.csv',index=False)
    artifacts=[]
    for path in sorted((root/'shared/standings').glob('*/metadata.json')):
        meta=read_json(path)
        if meta.get('policy')==baseline.POLICY:artifacts.append(dict(path=str(path.parent.relative_to(root)),policy=meta['policy'],action='rebuild_from_source_before_new_policy_use'))
    for filename in ['phase_context.json','phase_context.parquet','rank_diagnostics.parquet']:
        artifacts.append(dict(path='shared/features/'+filename,action='regenerate_rank_bank_and_diagnostics_with_new_source_id'))
    pd.DataFrame(artifacts).to_csv(a.output/'feature-artifact-inventory.csv',index=False)
    # Exact fold membership impact for each rank window; no labels/weights opened.
    impacts=[]
    for path in sorted((root/'shared/populations').glob('*.parquet')):
        f=read_table(path);keys=['competition_id','season_id','event_id']
        for w,g in delta.groupby('window'):
            impacted=f[keys].merge(g[keys].drop_duplicates(),on=keys,how='inner')
            impacts.append(dict(population=path.stem,window=int(w),affected_fixtures=len(impacted),scope='four cited seasons; lower bound outside these seasons'))
    pd.DataFrame(impacts).to_csv(a.output/'fold-impact.csv',index=False)
    assert all(digest(Path(p))==sha for p,sha in watched.items()),'Read-only input changed during audit'
    receipt=dict(base_commit=BASE,baseline_source_sha256=hashlib.sha256(raw).hexdigest(),policy=POLICY,python=sys.version,pandas=pd.__version__,command=sys.argv,source_hashes={str(p):digest(p) for p in [Path(__file__),Path('src/xdiyo_analytics/ratings/standings.py'),Path('src/xdiyo_analytics/features/standings.py')]},inputs=watched,input_hashes_unchanged=True,bank_source_id=bank_meta['source_id'],summaries=summaries,affected_models=sum(bool(json.loads(r['rank_coordinates'])) for r in inventory),models_inspected=len(inventory),legacy_snapshot_sets=len(artifacts)-3,rank_feature_coordinates_changed=len(delta),output_hashes={p.name:digest(p) for p in a.output.glob('*.csv')})
    (a.output/'audit-receipt.json').write_text(json.dumps(receipt,indent=2,default=str)+'\n')
if __name__=='__main__':main()
