"""Paired observed outcomes for independent label/settlement checks."""
import pandas as pd

from xdiyo_analytics.features import Stat

STAT = Stat('ALL', 'Match overview', 'cornerKicks')
ALL_PERIODS = Stat(None, 'Match overview', 'cornerKicks')
OWN = 'team::ALL::Match overview::cornerKicks::value'
OTHER = 'opponent::ALL::Match overview::cornerKicks::value'
TEAM = 2**63 + 31

def label_history():
    games = [
        (3., 8., 'W', 'finished'),
        (5., 5., 'D', 'finished'),
        (11., 2., 'L', 'finished'),
        (7., 4., 'W', 'notstarted'),
        (None, None, None, 'cancelled'),
        (float('inf'), 3., None, 'finished'),
    ]
    rows, stat_columns = [], {}
    for event, (home_value, away_value, result, status) in enumerate(games):
        for side, team, opponent, own, other in [
            ('home', TEAM, TEAM+1, home_value, away_value),
            ('away', TEAM+1, TEAM, away_value, home_value),
        ]:
            row = dict(event_id=2**63+101+event, team_id=team, opponent_id=opponent,
                       competition_id=2**53+19, season_id=2**53+23,
                       source_league='Synthetic', source_season='24_25',
                       kickoff_at=pd.Timestamp('2025-01-01', tz='UTC')+pd.Timedelta(days=event),
                       side=side, status=status, round=event+1, stage='regular',
                       result=result if side=='home' else {'W':'L','D':'D','L':'W',None:None}[result])
            for period, field, group, pair in [
                ('ALL','value','Match overview',(own,other)),
                ('1ST','value','Match overview',(1. if side=='home' else 2.,2. if side=='home' else 1.)),
                ('ALL','total','Match overview',(100. if side=='home' else 200.,200. if side=='home' else 100.)),
                ('ALL','value','Other group',(99.,99.)),
            ]:
                for role, value in zip(('team','opponent'),pair):
                    column=f'{role}::{period}::{group}::cornerKicks::{field}'
                    row[column]=value
                    stat_columns[column]=dict(role=role,period=period,group_name=group,key='cornerKicks',field=field)
            rows.append(row)
    history=pd.DataFrame(rows)
    for column in ('event_id','team_id','opponent_id','competition_id','season_id'):
        history[column]=pd.array(history[column],dtype='uint64[pyarrow]')
    for column in stat_columns:
        history[column]=pd.array(history[column],dtype='float64[pyarrow]')
    history['status']=pd.array(history.status,dtype='string')
    history['result']=pd.array(history.result,dtype='string')
    history.index=pd.Index([5,5,2,8,2,9,1,1,5,7,7,3],name='stored_row')
    history.attrs={'stat_columns':stat_columns,'source':{'version':'independent_labels_fixture'}}
    return history
