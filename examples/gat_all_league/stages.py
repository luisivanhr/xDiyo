"""Canonical grouping metadata; preserve valid clocks outside Belgium."""
import numpy as np
import pandas as pd
from . import core

BELGIAN_STAGES={38:'regular',12656:'championship_playoff',15545:'relegation_playoff',92614:'european_playoff'}

def repair_schedule(schedule):
    result=schedule.copy(deep=True)
    belgium=result.competition_id.eq(38)
    if belgium.any() and ('tournament_id' not in result or result.loc[belgium,'tournament_id'].isna().any()):raise ValueError('Belgian stage identity is missing')
    stages=result.loc[belgium,'tournament_id'].astype(np.int64) if belgium.any() else pd.Series(dtype=np.int64)
    if not set(stages)<=set(BELGIAN_STAGES):raise ValueError('Unreviewed Belgian tournament identity')
    result['round_stage']=np.zeros(len(result),dtype=np.int64)
    result.loc[belgium,'round_stage']=stages.to_numpy()
    old_keys=['competition_id','season_id','round']
    if 'round_issue_at' not in result:
        result['round_issue_at']=result.groupby(old_keys,sort=False).kickoff_at.transform('min')
    before=pd.to_datetime(result.round_issue_at,utc=True)
    rebuilt=result.loc[belgium].groupby(list(core.ROUND_KEYS),sort=False).kickoff_at.transform('min')
    result['round_issue_at']=before.copy()
    result.loc[belgium,'round_issue_at']=rebuilt
    np.testing.assert_array_equal(result.loc[~belgium,'round_issue_at'].to_numpy(),before[~belgium].to_numpy())
    anchors=result.groupby(list(core.ROUND_KEYS),sort=False).round_issue_at.transform('min')
    if not anchors.equals(result.round_issue_at):raise ValueError('Inconsistent corrected full-stage anchors')
    if (result.round_issue_at>result.kickoff_at).any():raise ValueError('Anchor follows kickoff')
    return result

def attach_schedule(frame,schedule):
    """Use authoritative full-schedule keys and replace stale explicit clocks."""
    result=frame.copy(deep=True)
    lookup=schedule.set_index(list(core.MATCH_KEYS)).loc[core.keys(frame)]
    for column in ('round_stage','round_issue_at'):
        result[column]=lookup[column].to_numpy()
    if 'issuance_at' in result:result['issuance_at']=lookup.round_issue_at.to_numpy()
    return result

def stage_inventory(schedule):
    result=schedule[schedule.competition_id.eq(38)].groupby(['season_id','round_stage','tournament_id']).agg(
        count=('event_id','size'),first_kickoff=('kickoff_at','min'),last_kickoff=('kickoff_at','max'),
        first_round=('round','min'),last_round=('round','max')).reset_index()
    result['stage_name']=result.round_stage.map(BELGIAN_STAGES)
    return result
