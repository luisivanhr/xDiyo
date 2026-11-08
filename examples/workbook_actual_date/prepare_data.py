"""Reproducible workbook-only preparation; refuses to replace an existing run."""
from collections import defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'src'))
from xdiyo_analytics.data.fixture_rounds import assign_fixture_rounds, SELECTED
from xdiyo_analytics.data.native_fixture_rounds import load_combined_fixture_rounds, LEAGUES
from xdiyo_analytics.data.catalogue.registry import name_key
from xdiyo_analytics.graphs.actual_date_context import build_actual_date_context

TZ='Europe/Berlin'
ORIGINAL={row[0]:row[4] for row in LEAGUES}
STATMAP={'total_shots':'Match overview::totalShotsOnGoal',
         'shots_off_target':'Shots::shotsOffGoal','corners':'Match overview::cornerKicks'}

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def write(frame,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    frame.reset_index(drop=True).to_parquet(path,index=False)

def prepare(matches,statistics,first,second,output):
    output=Path(output)
    if output.exists():raise ValueError('Output already exists; use a fresh run directory.')
    raw=pd.read_parquet(matches); stats=pd.read_parquet(statistics)
    rounds=load_combined_fixture_rounds(REPO/'data/fixture_rounds/v59_20261108',
                                      REPO/'data/fixture_rounds/original13_native_20261108')
    raw=assign_fixture_rounds(raw,rounds)
    for frame in (raw,stats):
        frame['match_id']=pd.to_numeric(frame.match_id,errors='raise').astype('int64')
        if not frame.match_id.is_unique:raise ValueError('Duplicate workbook identity.')
    if set(raw.match_id)!=set(stats.match_id):raise ValueError('Statistics match population mismatch.')
    for column in ('start_datetime','country','league','season','home_team','away_team'):
        pd.testing.assert_series_equal(raw.set_index('match_id')[column].sort_index(),
                                       stats.set_index('match_id')[column].sort_index(),check_dtype=False)
    f=pd.read_csv(first); s=pd.read_csv(second)
    ids=set(f.league_id)|set(s.league_id)
    if len(ids)!=47 or not set(ORIGINAL)<=ids:raise ValueError('Inclusion lists changed.')
    raw['league_id']=pd.to_numeric(raw.league_id,errors='raise').astype('int64')
    civil=pd.to_datetime(raw.start_datetime)
    selected=raw.loc[raw.league_id.isin(ids)&civil.ge('2019-07-01')&civil.lt('2026-07-01')].copy()
    selected=selected.merge(stats[['match_id']+[c for c in stats if c.endswith('_ft')]],on='match_id',validate='one_to_one')
    selected.attrs={}
    directory_path=REPO/'data/fixture_rounds/original13_native_20261108/provenance/reviewed_team_directory.json'
    directory=json.loads(directory_path.read_text(encoding='utf-8'))
    aliases=defaultdict(list)
    for team in directory['teams']:
        for alias in team['names']:
            aliases[(team['country'],name_key(alias['name']))].append((alias['start'],alias['end'],int(team['team_id'])))
    for side in ('home','away'):
        mapped=[]
        for country,name,day in zip(selected.country,selected[side+'_team'],selected.start_datetime.str[:10]):
            choices={tid for start,end,tid in aliases[(country,name_key(name))] if start<=day<=end}
            if len(choices)!=1:raise ValueError(f'Unresolved reviewed team: {country}/{name}/{day}/{choices}')
            mapped.append(next(iter(choices)))
        selected[side+'_id']=np.array(mapped,dtype=np.int64)
    if selected.home_id.eq(selected.away_id).any():raise ValueError('Self match.')
    if selected.duplicated(['league_id','season','start_datetime','home_id','away_id']).any():
        raise ValueError('Duplicate physical fixture.')
    selected['competition_id']=[ORIGINAL.get(int(i),1_000_000_000+int(i)) for i in selected.league_id]
    selected['season_id']=[2_000_000_000+int(lid)*10000+int(str(season)[:4]) for lid,season in zip(selected.league_id,selected.season)]
    selected['event_id']=3_000_000_000+selected.match_id
    # Berlin is an explicit study convention, not verified source timezone.
    # Repeated fall-back hours use the later feasible instant as a conservative
    # proxy; retain both bounds in the audit. This never changes the playing day.
    civil=pd.to_datetime(selected.start_datetime)
    early=civil.dt.tz_localize(TZ,ambiguous=True,nonexistent='raise').dt.tz_convert('UTC')
    late=civil.dt.tz_localize(TZ,ambiguous=False,nonexistent='raise').dt.tz_convert('UTC')
    clock_audit=selected.loc[early.ne(late),['match_id','league_id','start_datetime']].copy()
    clock_audit['earliest_utc']=early.loc[clock_audit.index];clock_audit['latest_utc']=late.loc[clock_audit.index]
    selected['kickoff_at']=late.astype('datetime64[ns, UTC]')
    selected['issuance_at']=selected.kickoff_at.dt.tz_convert(TZ).dt.normalize().dt.tz_convert('UTC')
    selected['result_available_at']=selected.kickoff_at
    selected['round']=selected.season_round
    selected['round_stage']=selected.phase_key
    selected['is_original13']=selected.league_id.isin(ORIGINAL)
    selected['source_dataset']='international_workbook'
    selected['source_league']=selected.country+' :: '+selected.league
    selected['season_year']=selected.season
    selected['status']='finished'
    selected['goals_for']=pd.to_numeric(selected.home_goals_ft,errors='raise').astype(float)
    selected['goals_against']=pd.to_numeric(selected.away_goals_ft,errors='raise').astype(float)
    if not np.isfinite(selected[['goals_for','goals_against']]).all().all():raise ValueError('Missing target score.')
    keep=['competition_id','season_id','event_id','match_id','league_id','home_id','away_id','kickoff_at','issuance_at',
          'result_available_at','round','round_stage',*SELECTED,'is_original13','source_dataset','source_league','season_year',
          'status','goals_for','goals_against']
    schedule=selected[keep].sort_values(['kickoff_at','competition_id','event_id'],kind='stable').reset_index(drop=True)
    perspectives=[]
    for side,other in (('home','away'),('away','home')):
        h=selected[keep].copy();h['side']=side;h['team_id']=selected[side+'_id'];h['opponent_id']=selected[other+'_id']
        h['goals_for']=pd.to_numeric(selected[side+'_goals_ft'],errors='raise').astype(float)
        h['goals_against']=pd.to_numeric(selected[other+'_goals_ft'],errors='raise').astype(float)
        for role,position in (('team',side),('opponent',other)):
            for field,native in STATMAP.items():
                column=role+'::ALL::'+native+'::value'
                values=selected[position+'_'+field+'_ft'].replace({'nan':np.nan,'':np.nan})
                h[column]=pd.to_numeric(values,errors='raise').astype(float)
                if ((h[column].dropna()<0)|(h[column].dropna()%1!=0)).any():raise ValueError('Invalid statistic.')
                h['available_at::'+column]=h.kickoff_at
        perspectives.append(h)
    history=pd.concat(perspectives,ignore_index=True)
    members=history.groupby(['team_id','competition_id','season_id'],as_index=False).issuance_at.min().rename(columns={'issuance_at':'entry_at'})
    members['movement']='unknown'
    if members.duplicated(['team_id','entry_at']).any():raise ValueError('Simultaneous membership conflict.')
    # Identical original-13 context across stages, independent of stage size.
    phase=build_actual_date_context(schedule,timezone=TZ,provenance=dict(source='workbook_only',rounds=rounds.provenance))
    output.mkdir(parents=True)
    for name in ('src','examples'):
        shutil.copytree(REPO/name,output/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','bundles'))
    for path in Path(__file__).parent.glob('*.py'):shutil.copy2(path,output/path.name)
    shutil.copy2(Path(__file__).parent/'EXPERIMENT_PROTOCOL.md',output/'EXPERIMENT_PROTOCOL.md')
    data=output/'data';data.mkdir()
    write(clock_audit,data/'ambiguous_civil_clocks.parquet')
    inventory=[dict(path=str(Path(p).resolve()),sha256=sha(p),bytes=Path(p).stat().st_size)
               for p in (matches,statistics,first,second,directory_path)]
    manifest={}
    for stage,leagues in [('original13',set(ORIGINAL)),('first_divisions',set(ORIGINAL)|set(f.league_id)),('first_and_second_divisions',ids)]:
        root=data/stage;ss=schedule.loc[schedule.league_id.isin(leagues)].reset_index(drop=True)
        hh=history.loc[history.league_id.isin(leagues)].reset_index(drop=True)
        mm=members.loc[members.competition_id.isin(ss.competition_id.unique())].reset_index(drop=True)
        for frame,name in ((ss,'schedule'),(hh,'history'),(mm,'memberships')):write(frame,root/(name+'.parquet'))
        from xdiyo_analytics.graphs.phase_context import PhaseRankLookup
        pf=phase.frame.loc[phase.frame.event_id.isin(ss.event_id)].reset_index(drop=True)
        stage_phase=PhaseRankLookup(pf,provenance=phase.state()['provenance'],policy=phase.policy);stage_phase.save(root/'features')
        info={}
        def population(name,start,split,end):
            dates=ss.kickoff_at.dt.tz_convert(TZ)
            test=ss.loc[dates.ge(pd.Timestamp(split,tz=TZ))&dates.lt(pd.Timestamp(end,tz=TZ))].copy()
            boundary=test.issuance_at.min()
            train=ss.loc[dates.ge(pd.Timestamp(start,tz=TZ))&dates.lt(pd.Timestamp(split,tz=TZ))&ss.result_available_at.lt(boundary)].copy()
            write(train,root/'populations'/(name+'_train.parquet'));write(test,root/'populations'/(name+'_test.parquet'))
            info[name]=dict(train=len(train),test=len(test),original13_test=int(test.is_original13.sum()),cutoff=str(boundary))
            return train,test
        train,test=population('dev','2020-07-01','2022-02-11','2022-07-01')
        write(train,root/'populations/dev_fit.parquet');write(test,root/'populations/dev_validation.parquet')
        for n,year in enumerate(range(2022,2026),1):population('fold_'+str(n),'2020-07-01',f'{year}-07-01',f'{year+1}-07-01')
        prov=dict(stage=stage,source_policy='workbook-only for every stage, same identity/statistics definitions',
            forecast_policy='actual_playing_date_v1',timezone=TZ,source_timezone='unknown; Berlin civil-clock assumption; repeated hour uses later feasible UTC bound',
            ambiguous_clocks=int(clock_audit.league_id.isin(leagues).sum()),
            availability='actual kickoff retrospective proxy; strict-before',membership='first fixture playing-date cutoff; movement unknown',
            source_inventory=inventory,round_provenance=rounds.provenance,phase_policy=phase.policy,
            phase_coverage={c:int(pf[c].notna().sum()) for c in pf if c=='countdown' or '_rank_' in c},
            history_rows=len(hh),fixtures=len(ss),teams=int(hh.team_id.nunique()),leagues=int(ss.league_id.nunique()),populations=info)
        (root/'provenance.json').write_text(json.dumps(prov,indent=2),encoding='utf-8')
        manifest[stage]=prov
        print(stage,info,flush=True)
    (data/'MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (data/'source_inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--matches',required=True);p.add_argument('--statistics',required=True)
    p.add_argument('--first',required=True);p.add_argument('--second',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();prepare(**vars(args))
