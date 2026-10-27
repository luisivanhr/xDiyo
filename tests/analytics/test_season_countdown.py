"""Native remaining-round clock: schedules, special formats and integration."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.features import SeasonCountdown, evaluate_features
from xdiyo_analytics.features.countdown import season_countdown
from xdiyo_analytics.ui.recipe import catalog_for_ui, node, prepare_recipe, export_python
from test_ui_workflow import ui_recipe


def schedule(rounds=(1, 2, 3), days=(0, 7, 14), **extra):
    return pd.DataFrame(dict(competition_id=1, season_id=1,
        event_id=np.arange(len(rounds)), round=rounds,
        kickoff_at=pd.Timestamp('2020-01-01', tz='UTC') + pd.to_timedelta(days, unit='D'), **extra))


@pytest.mark.parametrize('n,total', [(18,34),(19,38),(20,38),(22,42),(24,46)])
def test_double_round_robin_even_odd_and_partial(n,total):
    h=schedule()
    x=season_countdown(h,SeasonCountdown(format='double',team_count=n))
    np.testing.assert_allclose(x, (total-np.arange(1,4))/(total-1))
    last=schedule((total,), (100,))
    assert season_countdown(last,SeasonCountdown(format='double',team_count=n)).iloc[0]==0


@pytest.mark.parametrize('value',[0,1,-1,True,2.5,'38',float('inf')])
@pytest.mark.parametrize('parameter',['total_rounds','team_count'])
def test_invalid_configuration(value,parameter):
    with pytest.raises(ValueError,match='at least 2'):
        SeasonCountdown(**{parameter:value})


def test_postponed_round_uses_active_round_and_isolated_advanced_match_not_anchor():
    # Round 3 has an advanced singleton on day 1; its main cluster starts day 14.
    h=schedule((1,2,2,2,3,3,3,1),(0,7,7,8,1,14,15,16))
    h.index=[8]*len(h)
    x=season_countdown(h,SeasonCountdown(3))
    np.testing.assert_array_equal(x,[1,.5,.5,.5,0,0,0,0])
    # Reverse input and duplicate team rows cannot change anchors or weighting.
    paired=pd.concat([h,h]).iloc[::-1]
    got=season_countdown(paired,SeasonCountdown(3))
    np.testing.assert_array_equal(got,np.tile(x,2)[::-1])


def test_between_rounds_and_after_final_round():
    h=schedule((1,2,3,1,1,1),(0,7,14,10,14,30))
    np.testing.assert_array_equal(season_countdown(h,SeasonCountdown(3)),[1,.5,0,.5,0,0])


def test_missing_schedule_values_and_conflicting_copies():
    h=schedule((1,None,3),(0,7,14));h.loc[2,'kickoff_at']=pd.NaT
    np.testing.assert_allclose(season_countdown(h,SeasonCountdown(3)),[1,np.nan,np.nan],equal_nan=True)
    broken=pd.concat([h,h.iloc[[0]].assign(round=2)])
    with pytest.raises(ValueError,match='inconsistent fixture'):
        season_countdown(broken,SeasonCountdown(3))
    with pytest.raises(ValueError,match='conflicts'):
        season_countdown(h.assign(round=[2,None,3]),SeasonCountdown(3),schedule_history=h)
    with pytest.raises(ValueError,match='absent'):
        season_countdown(h.assign(event_id=[3,4,5]),SeasonCountdown(3),schedule_history=h)


@pytest.mark.parametrize('bad',[0,-1,1.5,float('inf'),4])
def test_invalid_rounds(bad):
    with pytest.raises(ValueError,match='round'):
        season_countdown(schedule((bad,),(0,)),SeasonCountdown(3))


def test_membership_required_unknown_format_and_season_isolation():
    h=schedule(source_league='Premier_League')
    with pytest.raises(ValueError,match='team_seasons'):
        season_countdown(h,SeasonCountdown())
    roster=pd.DataFrame(dict(competition_id=1,season_id=1,team_id=range(20)))
    np.testing.assert_allclose(season_countdown(h,SeasonCountdown(),team_seasons=roster),[1,36/37,35/37])
    with pytest.raises(ValueError,match='verified format'):
        season_countdown(h.assign(source_league='Unknown'),SeasonCountdown(),team_seasons=roster)
    other=h.assign(season_id=2,kickoff_at=h.kickoff_at+pd.Timedelta(days=365))
    np.testing.assert_array_equal(season_countdown(pd.concat([h,other]),SeasonCountdown(3)),[1,.5,0,1,.5,0])


@pytest.mark.parametrize('league,n,total,tournament',[
    ('Liga_MX_Apertura',18,17,28),('Liga_MX_Apertura',19,19,28),
    ('Liga_MX_Clausura',18,17,16753),('Premiership',12,38,54),('Pro_League',18,34,38)])
def test_native_formats(league,n,total,tournament):
    h=schedule((1,total),(0,100),source_league=league,tournament_id=tournament)
    np.testing.assert_array_equal(season_countdown(h,SeasonCountdown(team_count=n)),[1,0])
    if league!='Premiership':
        h.loc[1,'tournament_id']=999
        assert np.isnan(season_countdown(h,SeasonCountdown(team_count=n)).iloc[1])


def test_scotland_split_continues_without_reset():
    h=schedule((33,34,38),(0,7,35),source_league='Premiership',tournament_id=[54,92066,92067])
    np.testing.assert_allclose(season_countdown(h,SeasonCountdown()),[5/37,4/37,0])


def test_japan_two_stage_labels_and_regional_transition():
    h=schedule((1,17,1,17),(0,7,14,21),source_league='J1',source_season='16_16')
    h['kickoff_at']=pd.to_datetime(['2016-02-27','2016-06-25','2016-07-02','2016-11-03'],utc=True)
    np.testing.assert_allclose(season_countdown(h,SeasonCountdown(team_count=18)),[1,17/33,16/33,0])
    h=schedule((1,18,1,18),(0,7,14,21),source_league='J1_Transition',tournament_id=[171852,171852,171853,171853])
    np.testing.assert_array_equal(season_countdown(h,SeasonCountdown()),[1,0,1,0])


def test_results_do_not_enter_clock_and_filtered_queries_keep_full_schedule():
    h=schedule((1,2,3,1),(0,7,14,15))
    baseline=season_countdown(h,SeasonCountdown(3))
    changed=h.assign(goals_for=[1000]*4,status='notstarted',result='X',available_at='2100-01-01')
    pd.testing.assert_series_equal(baseline,season_countdown(changed,SeasonCountdown(3)))
    pd.testing.assert_series_equal(baseline.iloc[[3,0]],season_countdown(h.iloc[[3,0]],SeasonCountdown(3),schedule_history=h))


def test_ui_json_python_preparation_fixture_scope_and_context(ui_recipe):
    catalog=catalog_for_ui()
    expression=SeasonCountdown(8)
    assert catalog.build(json.loads(json.dumps(catalog.encode(expression))))==expression
    fields=next(s for s in catalog.schema() if s['id']=='features.SeasonCountdown')['fields']
    assert {f['name'] for f in fields}=={'total_rounds','format','team_count'}
    ui_recipe['features']={'season_countdown':node('features.SeasonCountdown',total_rounds=8)}
    prepared=prepare_recipe(ui_recipe)
    assert prepared.dataset.X.columns.tolist()==['fixture::season_countdown']
    np.testing.assert_allclose(prepared.dataset.X['fixture::season_countdown'],(8-prepared.dataset.metadata['round'])/7)
    namespace={}
    exec(export_python(ui_recipe).split('result = run_recipe')[0],namespace)
    pd.testing.assert_frame_equal(prepared.dataset.X,prepare_recipe(namespace['recipe']).dataset.X)
    context=deepcopy(ui_recipe);context['features']={'home':node('features.IsHome')};context['context_features']=ui_recipe['features']
    np.testing.assert_array_equal(prepared.dataset.X.to_numpy(),prepare_recipe(context).dataset.X[['season_countdown']].to_numpy())


def test_historical_mexico_tournament_identifier():
    h=schedule((1,17),(0,100),source_league='Liga_MX_Clausura',source_season='10_11',tournament_id=28)
    np.testing.assert_array_equal(season_countdown(h,SeasonCountdown(team_count=18)),[1,0])


def test_single_leg_with_only_one_round_is_undefined():
    with pytest.raises(ValueError,match='two planned rounds'):
        season_countdown(schedule((1,),(0,)),SeasonCountdown(format='single',team_count=2))


def test_roster_cannot_omit_an_observed_team():
    h=schedule(source_league='Premier_League',home_id=1,away_id=3)
    roster=pd.DataFrame(dict(competition_id=1,season_id=1,team_id=[1,2]))
    with pytest.raises(ValueError,match='omits scheduled'):
        season_countdown(h,SeasonCountdown(),team_seasons=roster)


def test_clock_is_fixture_scoped_through_arithmetic_and_cutoffs():
    from xdiyo_analytics.features import Difference,Constant
    from transition_samples import history_from_games
    h=history_from_games([dict(round=1,day=0),dict(round=2,day=7),dict(round=3,day=14)])
    feature=Difference(Constant(1),SeasonCountdown(3))
    a=evaluate_features(h,{'elapsed':feature},cutoffs=h.kickoff_at-pd.Timedelta(days=1))
    b=evaluate_features(h,{'elapsed':feature},cutoffs=h.kickoff_at-pd.Timedelta(days=2))
    np.testing.assert_array_equal(a.elapsed,b.elapsed)
    assert a.attrs['feature_scopes']=={'elapsed':'fixture'}


def test_real_postponement_and_two_stage_source_conventions():
    from pathlib import Path
    from xdiyo_analytics.data import load_seasons
    root=Path(__file__).resolve().parents[2]/'data/xDiyo_data'
    d=load_seasons(root,'23_24',leagues=['Premier_League'],tables=['matches','team_seasons'],include_awarded=True)
    x=season_countdown(d.matches,SeasonCountdown(),team_seasons=d['team_seasons'])
    # Chelsea-Tottenham: assigned round 26; actual kickoff during round 35.
    mask=d.matches.event_id.eq(12226774)
    assert mask.sum()==1
    assert d.matches.loc[mask,'round'].iloc[0]==26
    assert x[mask].iloc[0]==3/37
    h=schedule((1,18,34),(0,100,200),source_league='J1',source_season='15_15',tournament_id=82)
    np.testing.assert_allclose(season_countdown(h,SeasonCountdown(team_count=18)),[1,16/33,0])


def test_explicit_countdown_does_not_shadow_standings_membership():
    from pathlib import Path
    from examples.derived_standings import recipe
    root=Path(__file__).resolve().parents[2]/'data/xDiyo_data'
    spec=recipe(root)
    spec['features']={'standing':node('features.DerivedStandings',normalized_ranks=[],include_raw=True),
                      'clock':node('features.SeasonCountdown',total_rounds=38)}
    prepared=prepare_recipe(spec)
    assert len(prepared.dataset.X)==380
    assert prepared.dataset.X['fixture::clock'].notna().all()
    assert any('standing' in c for c in prepared.dataset.X)
