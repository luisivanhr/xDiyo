"""Reviewed curtailed calendars must not require future unplayed fixtures."""
import json
import numpy as np
import pandas as pd
import pytest
from xdiyo_analytics.features import DerivedStandings,SeasonCountdown
from xdiyo_analytics.features.countdown import season_countdown
from xdiyo_analytics.ratings import build_standings,StandingsRun
from xdiyo_analytics.graphs import build_phase_context
from test_derived_standings import population


def regular(meetings=2):
    h,m=population(6)
    if meetings==3:
        third=h.loc[h['round'].le(5)].copy()
        third['event_id']+=500000;third['round']+=10;third['kickoff_at']+=pd.Timedelta(days=30)
        h=pd.concat([h,third],ignore_index=True)
    h['standings_table_id']='regular';m['standings_table_id']='regular'
    h['standings_rule']='independent_3_1_0'
    h['source_league']='Premier_League';h['source_season']='19_20'
    h['release']=h.kickoff_at+pd.Timedelta(hours=1)
    return h,m


def build(h,m,meetings=2,**kwargs):
    return build_standings(h,team_seasons=m,normalized_ranks=[1,5,10,20],
        timing='kickoff_day',meetings_per_pair=meetings,allow_incomplete_schedule=True,**kwargs)


@pytest.mark.parametrize('meetings',[2,3])
def test_incomplete_values_equal_complete_calendar_with_unplayed_future(meetings,tmp_path):
    h,m=regular(meetings);last=int(h['round'].max())-2
    observed=h.loc[h['round'].le(last)].copy();queries=observed.loc[observed.side.eq('home')].copy()
    full=h.copy();full.loc[full['round'].gt(last),['goals_for','goals_against']]=np.nan
    full.loc[full['round'].gt(last),'status']='canceled'
    expected=build_standings(full,team_seasons=m,normalized_ranks=[1,5,10,20],queries=queries,
        timing='kickoff_day',**({'meetings_per_pair':3} if meetings==3 else {}))
    actual=build(observed,m,meetings,queries=queries)
    expression=DerivedStandings([1,5,10,20],side='both')
    np.testing.assert_array_equal(actual.features(queries,expression),expected.features(queries,expression))
    pd.testing.assert_frame_equal(actual.snapshots,expected.snapshots)
    assert actual.metadata['allow_incomplete_schedule'] is True
    assert actual.metadata['schedule']=='explicit_independent_regular_incomplete_schedule'
    assert actual.metadata['state_sha256']!=expected.metadata['state_sha256']
    actual.save(tmp_path/'saved');loaded=StandingsRun.load(tmp_path/'saved')
    pd.testing.assert_frame_equal(loaded.features(queries,expression),actual.features(queries,expression))
    with pytest.raises(ValueError,match='Incomplete'):
        build_standings(observed,team_seasons=m,timing='kickoff_day',**({'meetings_per_pair':3} if meetings==3 else {}))


def test_earlier_tables_ignore_current_and_future_results_in_partial_calendar():
    h,m=regular();h=h.loc[h['round'].le(7)].copy()
    day=h.loc[h['round'].eq(5),'kickoff_at'].min().normalize()
    queries=h.loc[h.kickoff_at.dt.normalize().eq(day)]
    expected=build(h,m,queries=queries)
    changed=h.copy();changed.loc[changed.kickoff_at.ge(day),['goals_for','goals_against']]+=100
    actual=build(changed,m,queries=queries)
    pd.testing.assert_frame_equal(expected.snapshots,actual.snapshots)
    for p in actual.publications:
        used=h.loc[h.event_id.isin(p['contributing_events'])]
        assert used.kickoff_at.lt(day).all()


def test_timestamp_availability_still_gates_partial_calendar_results():
    h,m=regular();h=h.loc[h['round'].le(7)].copy();q=h.loc[h['round'].eq(5)]
    options=dict(team_seasons=m,normalized_ranks=[1,5],queries=q,cutoffs=q.kickoff_at,
        available_at='release',meetings_per_pair=2,allow_incomplete_schedule=True)
    before=build_standings(h,**options)
    h['release']=pd.Timestamp('2099-01-01',tz='UTC')
    after=build_standings(h,**options)
    assert before.snapshots.points.sum()>0 and after.snapshots.points.sum()==0


@pytest.mark.parametrize('failure',['no_table','no_format','bad_rule','mixed_stage','stage_json','repeat','missing_roster','first_round','round_limit','awarded'])
def test_partial_opt_in_retains_identity_format_membership_and_review_guards(failure):
    h,m=regular();h=h.loc[h['round'].le(7)].copy();kwargs=dict(meetings_per_pair=2)
    if failure=='no_table':h=h.drop(columns='standings_table_id');m=m.drop(columns='standings_table_id')
    elif failure=='no_format':kwargs={}
    elif failure=='bad_rule':h['standings_rule']='carry_half_points'
    elif failure=='mixed_stage':h['tournament_id']=1;h.loc[h['round'].eq(7),'tournament_id']=2
    elif failure=='stage_json':h['stage_json']=json.dumps({'round':1,'name':'Playoffs'})
    elif failure=='repeat':
        duplicate=h.loc[h.event_id.eq(h.event_id.iloc[0])].copy();duplicate['event_id']+=100000;duplicate['round']=8
        h=pd.concat([h,duplicate],ignore_index=True)
    elif failure=='missing_roster':m=m.loc[m.team_id.ne(m.team_id.iloc[0])]
    elif failure=='first_round':h=h.loc[h.event_id.ne(h.event_id.iloc[0])]
    elif failure=='round_limit':h.loc[h['round'].eq(7),'round']=11
    elif failure=='awarded':h['is_awarded']=h.event_id.eq(h.event_id.iloc[0])
    with pytest.raises(ValueError):
        build_standings(h,team_seasons=m,timing='kickoff_day',allow_incomplete_schedule=True,**kwargs)


def test_partial_phase_keeps_original_countdown_definition_and_complete_rank_values():
    h,m=regular();h=h.loc[h['round'].le(7)].copy()
    fixtures=h.loc[h.side.eq('home')].rename(columns={'team_id':'home_id','opponent_id':'away_id'}).reset_index(drop=True)
    cuts=fixtures.kickoff_at.dt.normalize()
    lookup=build_phase_context(h,m,fixtures,cutoffs=cuts,provenance={'review':'test independent regular season'},meetings_per_pair=2,allow_incomplete_schedule=True)
    clock=season_countdown(fixtures,SeasonCountdown(),schedule_history=h,team_seasons=m,cutoffs=cuts)
    np.testing.assert_array_equal(lookup.frame.set_index('event_id').loc[fixtures.event_id,'countdown'],clock)
    assert lookup.frame.filter(regex='rank_|countdown').notna().all().all()
    assert lookup.frame.countdown.min()>0  # planned 10 rounds, not observed 7
    run=build(h,m,queries=h.loc[h.side.eq('home')])
    expected=run.features(h,DerivedStandings([1],side='both'))
    np.testing.assert_array_equal(lookup.edges(h)[:,:2],expected[['team::form_rank_1','opponent::form_rank_1']])
    assert lookup.edges(h)[:,2:].sum()==0


def test_complete_default_identity_and_values_unchanged():
    h,m=regular()
    a=build_standings(h,team_seasons=m,timing='kickoff_day')
    b=build_standings(h,team_seasons=m,timing='kickoff_day',allow_incomplete_schedule=False)
    pd.testing.assert_frame_equal(a.snapshots,b.snapshots)
    assert a.metadata==b.metadata
