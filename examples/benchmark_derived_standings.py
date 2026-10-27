"""Bounded replay versus grouped reconstruction, including delayed releases.

Both candidates use the SAME validated canonical source. Native canonicalization
is timed separately; the injected already-validated structure is benchmark-only.
"""
import argparse
import json
from pathlib import Path
from hashlib import sha256
from unittest.mock import patch
import pandas as pd
from xdiyo_analytics.data import load_seasons
from xdiyo_analytics.histories import build_team_history
from xdiyo_analytics.ratings import build_standings
from xdiyo_analytics.ratings.standings import _canonical
from validate_derived_standings import measured


def grouped(h,roster,times,windows):
    source=h.copy()
    source['played']=1
    source['wins']=source.goals_for.gt(source.goals_against).astype(int)
    source['draws']=source.goals_for.eq(source.goals_against).astype(int)
    source['points']=3*source.wins+source.draws
    source['gf']=source.goals_for.astype(int);source['ga']=source.goals_against.astype(int)
    teams=roster.team_id.tolist();records=[]
    for q in times:
        active=source.loc[source.kickoff_at.lt(q)&source.release.le(q)].sort_values(['kickoff_at','event_id'])
        gate=set(source.loc[source['round'].eq(1),'event_id'])<=set(active.event_id)
        for w in (0,*windows):
            sample=active.groupby('team_id',sort=False).tail(w) if w else active
            t=sample.groupby('team_id')[['points','gf','ga']].sum().reindex(teams,fill_value=0)
            if not gate:t[:]=0
            t['gd']=t.gf-t.ga
            t=t.sort_values(['points','gd','gf','ga'],ascending=[False,False,False,True])
            keys=list(t[['points','gd','gf','ga']].itertuples(index=False,name=None))
            ranks={}
            for i,k in enumerate(keys):ranks.setdefault(k,i+1)
            for team,k in zip(t.index,keys):records.append((q,w,team,ranks[k]))
    return pd.DataFrame(records,columns=['recorded_at','window','team_id','rank'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    data=load_seasons(a.data_root,['22_23'],leagues=['Premier_League'],tables=['matches','team_seasons'],include_awarded=True,verify_hashes=True)
    h=build_team_history(data);m=data['team_seasons'];windows=(1,3,6,12,30)
    results=[]
    for delay in (False,True):
        source=h.copy();source['release']=source.kickoff_at+pd.Timedelta(hours=3)
        if delay:source.loc[source.event_id.eq(source.event_id.iloc[0]),'release']+=pd.Timedelta(days=35)
        canonical,prep=measured(lambda:_canonical(source,m,'release'))
        with patch('xdiyo_analytics.ratings.standings._canonical',return_value=canonical):
            native,incremental=measured(lambda:build_standings(source,team_seasons=m,normalized_ranks=windows,available_at='release'))
        times=sorted(native.snapshots.recorded_at.unique())
        other,reconstruction=measured(lambda:grouped(source,m,times,windows))
        keys=['recorded_at','window','team_id']
        actual=native.snapshots[keys+['rank']].sort_values(keys).reset_index(drop=True)
        expected=other.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(actual,expected,check_dtype=False)
        results.append(dict(delayed=delay,fixtures=len(h)//2,cutoffs=len(times),windows=list(windows),
            preparation=prep,native_replay=incremental,pandas_grouped=reconstruction,exact_rank_parity=True,
            grouped_over_native_seconds=reconstruction['seconds']/incremental['seconds'],
            memory_measure='tracemalloc peak Python allocations; not process RSS'))
    Path(a.output).write_text(json.dumps(dict(results=results,source_hashes={f:sha256(Path(f).read_bytes()).hexdigest() for f in [__file__,'src/xdiyo_analytics/ratings/standings.py']}),indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
