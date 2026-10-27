"""Read-only bounded source/oracle/provider comparison; never fit a model."""
import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter
import tracemalloc
import numpy as np
import pandas as pd
from xdiyo_analytics.data import load_seasons
from xdiyo_analytics.histories import build_team_history
from xdiyo_analytics.features import DerivedStandings
from xdiyo_analytics.features.history import availability_times
from xdiyo_analytics.ratings import build_standings


def measured(fn):
    tracemalloc.start();start=perf_counter();value=fn();seconds=perf_counter()-start
    _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    return value,dict(seconds=seconds,peak_python_bytes=peak)


def validate_oracle(h,m,run,oracle):
    count=0
    for (_,_,q,w),table in run.snapshots.groupby(['competition_id','season_id','recorded_at','window']):
        query=h.loc[h.kickoff_at.ge(q)].iloc[0]
        totals,ranks,gate=oracle(h,m,query,q,w)
        for row in table.itertuples(index=False):
            assert row.rank==ranks[row.team_id] and row.gate_open==gate
            assert [getattr(row,c) for c in ('played','wins','draws','losses','points','gf','ga','gd')]==totals[row.team_id]
            count+=1
    return count


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('standings_oracle','tests/analytics/test_derived_standings.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    files=['src/xdiyo_analytics/ratings/standings.py','src/xdiyo_analytics/features/standings.py','src/xdiyo_analytics/features/evaluation.py','tests/analytics/test_derived_standings.py',__file__]
    report=dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        python=platform.python_version(),pandas=pd.__version__,numpy=np.__version__,
        command=subprocess.list2cmdline(__import__('sys').argv),source_hashes={f:sha256(Path(f).read_bytes()).hexdigest() for f in files},populations=[])
    for league,season in [('Premier_League','22_23'),('Premier_League','23_24'),('Bundesliga','22_23')]:
        data=load_seasons(args.data_root,[season],leagues=[league],tables=['matches','pregame','team_seasons'],include_awarded=True,verify_hashes=True)
        h=build_team_history(data);h['release']=availability_times(h,h.kickoff_at+pd.Timedelta(hours=3));m=data['team_seasons']
        windows=(3,6,12,30)
        run,timing=measured(lambda:build_standings(h,team_seasons=m,normalized_ranks=windows,available_at='release'))
        count,slow=measured(lambda:validate_oracle(h,m,run,module.oracle))
        values,lookup=measured(lambda:run.features(h,DerivedStandings([],True)))
        _,repeat=measured(lambda:[run.features(h,DerivedStandings([3,6])) for _ in range(3)])
        comparisons=[];snap=run.snapshots;pubs={pd.Timestamp(x['cutoff']):x for x in run.publications}
        for i,row in enumerate(h.itertuples(index=False)):
            if pd.isna(row.team_position):continue
            q=h.loc[h['round'].eq(row.round),'kickoff_at'].min();pub=pubs[q]
            intervening=bool((h.release.gt(q)&h.release.le(row.kickoff_at)&h.kickoff_at.lt(row.kickoff_at)).any())
            at=snap.loc[snap.recorded_at.eq(q)&snap.window.eq(0)]
            tied=at.loc[at['rank'].eq(values.iloc[i,0]),'rank'].count()>1
            category='cutoff_mismatch' if intervening else 'cold_start_gate' if not pub['gate_open'] else 'shared_ties' if tied else 'unexplained_official_rules_or_source'
            error=abs(values.iloc[i,0]-row.team_position)
            comparisons.append(dict(event_id=int(row.event_id),team_id=int(row.team_id),team_name=row.team_name,
                cutoff=q.isoformat(),kickoff=row.kickoff_at.isoformat(),derived=float(values.iloc[i,0]),provider=float(row.team_position),
                error=float(error),boundary_aligned=not intervening,category='exact' if error==0 else category))
        frame=pd.DataFrame(comparisons)
        def metrics(f):return dict(n=len(f),exact_match_rate=float(f.error.eq(0).mean()),mean_absolute_error=float(f.error.mean()),max_absolute_error=float(f.error.max()))
        frame.to_csv(out/f'{league}_{season}_positions.csv',index=False)
        late=h.copy();late.loc[late.event_id.eq(late.event_id.iloc[0]),'release']+=pd.Timedelta(days=30)
        late_run,late_time=measured(lambda:build_standings(late,team_seasons=m,normalized_ranks=windows,available_at='release'))
        validate_oracle(late,m,late_run,module.oracle)
        report['populations'].append(dict(league=league,season=season,matches=len(h)//2,teams=len(m),source=data.provenance,
            oracle_rows=count,exact_oracle_parity=True,delayed_oracle_parity=True,replay=timing,brute=slow,lookup=lookup,
            three_repeated_lookups=repeat,delayed_replay=late_time,comparison=metrics(frame),aligned=metrics(frame.loc[frame.boundary_aligned]),
            categories=frame.category.value_counts().to_dict(),representative_mismatches=frame.loc[frame.error.gt(0)].head(12).to_dict('records'),
            publication_count=len(run.publications),gate_closed_publications=sum(not x['gate_open'] for x in run.publications)))
        (out/'validation.json').write_text(json.dumps(report,indent=2,default=str)+'\n',encoding='utf-8')
        print(league,season,metrics(frame),timing,flush=True)
    (out/'validation.json').write_text(json.dumps(report,indent=2,default=str)+'\n',encoding='utf-8')


if __name__=='__main__':main()
