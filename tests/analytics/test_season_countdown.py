"""Fixed causal majority-started countdown and native integration."""
from copy import deepcopy
from importlib.resources import files
import json

import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.features import SeasonCountdown, evaluate_features
from xdiyo_analytics.features.countdown import season_countdown
from xdiyo_analytics.ui.recipe import catalog_for_ui, node, prepare_recipe, export_python, validate_recipe
from xdiyo_analytics.ui import migrate_season_countdown_recipe
from test_ui_workflow import ui_recipe


def schedule(rounds=(1,2,3),days=(0,7,14),**extra):
    return pd.DataFrame(dict(competition_id=1,season_id=1,event_id=np.arange(len(rounds)),
        round=rounds,kickoff_at=pd.Timestamp('2020-01-01',tz='UTC')+pd.to_timedelta(days,unit='D'),
        **{'status':'finished',**extra}))


def clock(total=5,n=8,**kwargs):
    return SeasonCountdown(total_rounds=total,team_count=n,**kwargs)


@pytest.mark.parametrize('n,total',[(18,34),(19,38),(20,38),(22,42),(24,46)])
def test_planned_length_and_partial_download(n,total):
    h=schedule((1,2,total),(0,7,14))
    np.testing.assert_allclose(season_countdown(h,SeasonCountdown(format='double',team_count=n)),[1,(total-2)/(total-1),0])


@pytest.mark.parametrize('value',[0,1,-1,True,2.5,'38',float('inf')])
@pytest.mark.parametrize('parameter',['total_rounds','team_count'])
def test_invalid_configuration(value,parameter):
    with pytest.raises(ValueError,match='at least 2'):
        SeasonCountdown(**{parameter:value})


@pytest.mark.parametrize('bad',[0,-1,1.5,float('inf'),6])
def test_invalid_rounds(bad):
    with pytest.raises(ValueError,match='round'):
        season_countdown(schedule((bad,),(0,)),clock())


def test_majority_boundary_postponements_and_advanced_singleton():
    h=schedule((1,5,3,3,3,1,1,1),(0,1,2,3,4,2,4,5))
    # Four expected games per round, strict majority is three. Round 5 singleton
    # never moves other games. At the third r3 kickoff itself, only two count.
    np.testing.assert_allclose(season_countdown(h,clock()),[1,0,.5,.5,.5,1,1,.5])
    h.index=[8]*len(h)
    paired=pd.concat([h,h]).iloc[::-1]
    np.testing.assert_allclose(season_countdown(paired,clock()),np.tile([1,0,.5,.5,.5,1,1,.5],2)[::-1])


def test_simultaneous_starts_and_nanosecond_strict_boundary():
    h=schedule((3,3,3,1),(7,7,7,8))
    query=h.iloc[[3]]
    boundary=h.kickoff_at.iloc[0]
    assert season_countdown(query,clock(),schedule_history=h,cutoffs=boundary).iloc[0]==1
    assert season_countdown(query,clock(),schedule_history=h,cutoffs=boundary+pd.Timedelta(nanoseconds=1)).iloc[0]==.5
    assert season_countdown(query,clock(),schedule_history=h,cutoffs=boundary-pd.Timedelta(nanoseconds=1)).iloc[0]==1


def test_future_mutation_outcomes_and_input_permutation_safety():
    h=schedule((1,2,2,2,1,4,4,4),(0,7,7,7,8,20,20,20))
    target=h.iloc[[4]];x=season_countdown(target,clock(),schedule_history=h)
    assert x.iloc[0]==.75
    changed=h.copy();changed.loc[[5,6,7],'kickoff_at']+=pd.Timedelta(days=100)
    changed=changed.assign(goals_for=999,status_code=999,available_at='2100-01-01')
    pd.testing.assert_series_equal(x,season_countdown(target,clock(),schedule_history=changed.iloc[::-1]))
    # Losing or adding future downloaded fixtures cannot lower the threshold.
    pd.testing.assert_series_equal(x,season_countdown(target,clock(),schedule_history=h.iloc[:5]))


def test_partial_downloads_do_not_define_threshold_and_total_override_needs_membership():
    h=schedule((4,4,1),(1,2,3))
    assert season_countdown(h,clock()).iloc[-1]==1  # 2 of expected 4 is not a majority
    with pytest.raises(ValueError,match='majority threshold'):
        season_countdown(h,SeasonCountdown(total_rounds=5))
    roster=pd.DataFrame(dict(competition_id=1,season_id=1,team_id=range(8)))
    assert season_countdown(h,SeasonCountdown(5),team_seasons=roster).iloc[-1]==1


@pytest.mark.parametrize('status',['notstarted','postponed','cancelled','unknown',None])
def test_scheduled_timestamps_do_not_establish_start(status):
    h=schedule((3,3,3,1),(0,0,0,1),status=[status]*3+['notstarted'])
    assert season_countdown(h,clock()).iloc[-1]==1


@pytest.mark.parametrize('play,awarded,expected',[
    ('not_played',False,1),('unplayed',True,1),(None,True,1),
    ('abandoned',True,.5),('resumed',False,.5),('played',True,.5)])
def test_reviewed_play_evidence_and_awards(play,awarded,expected):
    h=schedule((3,3,3,1),(0,0,0,1),play_status=[play]*3+[None],is_awarded=[awarded]*3+[False])
    h['result_available_at']='2100-01-01' # completion/result release is irrelevant to actual start
    assert season_countdown(h,clock()).iloc[-1]==expected


def test_missing_values_and_conflicting_fixture_copies():
    h=schedule((1,None,3),(0,7,14));h.loc[2,'kickoff_at']=pd.NaT
    np.testing.assert_allclose(season_countdown(h,clock()),[1,np.nan,np.nan],equal_nan=True)
    for field,value in [('round',2),('status','notstarted'),('home_id',99)]:
        base=h.assign(home_id=1,away_id=2)
        broken=pd.concat([base,base.iloc[[0]].assign(**{field:value})])
        with pytest.raises(ValueError,match='inconsistent fixture'):
            season_countdown(broken,clock())
    with pytest.raises(ValueError,match='conflicts'):
        season_countdown(h.assign(round=[2,None,3]),clock(),schedule_history=h)
    with pytest.raises(ValueError,match='absent'):
        season_countdown(h.assign(event_id=[9,10,11]),clock(),schedule_history=h)
    with pytest.raises(ValueError,match='no later'):
        season_countdown(h,clock(),cutoffs=pd.Timestamp('2100-01-01',tz='UTC'))


@pytest.mark.parametrize('league,n,total,tournament',[
 ('Liga_MX_Apertura',18,17,28),('Liga_MX_Apertura',19,19,28),
 ('Liga_MX_Clausura',18,17,16753),('Premiership',12,38,54),('Pro_League',18,34,38)])
def test_native_formats_and_exclusions(league,n,total,tournament):
    h=schedule((1,total),(0,100),source_league=league,tournament_id=tournament)
    np.testing.assert_array_equal(season_countdown(h,SeasonCountdown(team_count=n)),[1,0])
    h.loc[1,'tournament_id']=999
    assert np.isnan(season_countdown(h,SeasonCountdown(team_count=n)).iloc[1])


def test_scotland_both_split_groups_share_six_fixture_threshold():
    h=schedule((34,34,34,34,1,38),(0,0,0,1,2,0),source_league='Premiership',tournament_id=[92066,92066,92067,92067,54,999])
    assert season_countdown(h,SeasonCountdown()).iloc[4]==4/37
    before=h.iloc[[4]]
    assert season_countdown(before,SeasonCountdown(),schedule_history=h,cutoffs=h.kickoff_at.iloc[3]).iloc[0]==1
    with pytest.raises(ValueError,match='tournament_id'):
        season_countdown(h.drop(columns='tournament_id'),SeasonCountdown())


def test_japan_phase_identity_and_independent_transition_calendars():
    h=schedule((10,2,2,2,2,2,1),(20,10,10,10,10,10,21),source_league='J1',source_season='16_16',season_phase=[1,2,2,2,2,2,1],tournament_id=82)
    # 18 teams: 9 games, majority 5. Season round 19 is activated, not false 27.
    assert season_countdown(h,SeasonCountdown(team_count=18)).iloc[0]==15/33
    with pytest.raises(ValueError,match='original season_phase'):
        season_countdown(h.drop(columns='season_phase'),SeasonCountdown(team_count=18))
    g=schedule((18,18,18,1,1),(0,0,0,1,1),source_league='J1_Transition',tournament_id=[171852]*4+[171853])
    np.testing.assert_array_equal(season_countdown(g,SeasonCountdown()),[0,0,0,0,1])


def test_verified_japan_resource_is_date_independent():
    from xdiyo_analytics.features.countdown import _japan_phases
    with files('xdiyo_analytics.features').joinpath('data/j1_2016_phases.csv').open('rb') as f:
        reference=pd.read_csv(f)
    assert len(reference)==306 and reference.season_phase.value_counts().to_dict()=={1:153,2:153}
    reference['kickoff_at']=pd.Timestamp('2099-01-01',tz='UTC')
    np.testing.assert_array_equal(_japan_phases(reference.drop(columns='season_phase')),reference.season_phase)
    with pytest.raises(ValueError,match='conflicts with verified'):
        _japan_phases(reference.assign(season_phase=3-reference.season_phase))


def test_season_reset_and_odd_team_byes():
    h=schedule((3,3,3,1),(0,0,0,1))
    later=schedule((1,),(370,)).assign(season_id=2)
    x=season_countdown(pd.concat([h,later]),clock(n=9)) # 9 teams still four games, majority three
    assert x.iloc[-2]==.5 and x.iloc[-1]==1


def test_activations_built_once_per_calendar_not_per_target(monkeypatch):
    import xdiyo_analytics.features.countdown as module
    original=module._activations;calls=[]
    def tracked(*args):
        calls.append(len(args[0]));return original(*args)
    monkeypatch.setattr(module,'_activations',tracked)
    h=schedule(tuple(np.repeat(range(1,6),4)),tuple(np.repeat(range(5),4)))
    season_countdown(h,clock())
    assert calls==[20]


def test_recipe_migration_is_explicit_copy_only_and_recursive(ui_recipe):
    old=node('features.SeasonCountdown',total_rounds=8,team_count=2,schedule_mode='as_of')
    ui_recipe['features']={'clock':old};ui_recipe['context_features']={'other':deepcopy(old)}
    with pytest.raises(ValueError,match='migrate_season_countdown_recipe'):
        validate_recipe(ui_recipe)
    before=deepcopy(ui_recipe)
    migrated,changes=migrate_season_countdown_recipe(ui_recipe)
    assert ui_recipe==before and len(changes)==2
    assert all(c['values_may_change'] for c in changes)
    assert 'schedule_mode' not in migrated['features']['clock']['params']
    assert migrate_season_countdown_recipe(migrated)==(migrated,[])
    prepare_recipe(migrated)
    with pytest.raises(TypeError):
        SeasonCountdown(schedule_mode='retrospective')


def test_ui_json_python_team_fixture_context_and_cutoff_alignment(ui_recipe):
    catalog=catalog_for_ui();expr=clock(8,n=2)
    assert catalog.build(json.loads(json.dumps(catalog.encode(expr))))==expr
    fields=next(s for s in catalog.schema() if s['id']=='features.SeasonCountdown')['fields']
    assert {f['name'] for f in fields}=={'total_rounds','format','team_count'}
    ui_recipe['features']={'clock':node('features.SeasonCountdown',total_rounds=8,team_count=2)}
    ui_recipe['cutoff_hours']=24
    a=prepare_recipe(ui_recipe)
    assert a.dataset.X.columns.tolist()==['fixture::clock']
    context=deepcopy(ui_recipe);context['features']={'home':node('features.IsHome')};context['context_features']=ui_recipe['features']
    b=prepare_recipe(context)
    np.testing.assert_array_equal(a.dataset.X['fixture::clock'],b.dataset.X.clock)
    namespace={};exec(export_python(context).split('result = run_recipe')[0],namespace)
    pd.testing.assert_frame_equal(b.dataset.X,prepare_recipe(namespace['recipe']).dataset.X)
    from transition_samples import history_from_games
    h=history_from_games([dict(round=3,day=0),dict(round=3,day=1),dict(round=3,day=2),dict(round=1,day=3)])
    h.index=np.arange(100,108)
    times=h.kickoff_at-pd.Timedelta(days=1)
    features=evaluate_features(h,{'clock':clock()},cutoffs=times)
    assert features.clock.iloc[-1]==1 # third kickoff equals cutoff
    from xdiyo_analytics.features.contextual import evaluate_context_features
    extra=evaluate_context_features(h,{'clock':clock()},cutoffs=times)
    np.testing.assert_array_equal(features.clock,extra.clock)
    assert features.attrs['feature_scopes']=={'clock':'fixture'}


def test_roster_validation_and_historical_mexico_id():
    h=schedule(source_league='Premier_League',home_id=1,away_id=3)
    roster=pd.DataFrame(dict(competition_id=1,season_id=1,team_id=[1,2]))
    with pytest.raises(ValueError,match='omits scheduled'):
        season_countdown(h,SeasonCountdown(),team_seasons=roster)
    h=schedule((1,17),(0,100),source_league='Liga_MX_Clausura',source_season='10_11',tournament_id=28)
    np.testing.assert_array_equal(season_countdown(h,SeasonCountdown(team_count=18)),[1,0])


def test_audit_rejects_unexplained_missing_outputs():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[2]/'docs/analytics/season-countdown/audit.py'
    spec=importlib.util.spec_from_file_location('majority_audit',path)
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    h=schedule((1,2,1),(0,7,14),tournament_id=[38,38,999])
    expected={'matches':3,'expected_eligible':2,'expected_excluded':1}
    values=pd.Series([1.,.5,np.nan])
    checks=[dict(publication='Pro_League_20_21',event_id=1,total_rounds=3,effective_round=2)]
    audit.validate_publication('Pro_League_20_21',h,values,expected,checks)
    for broken in (pd.Series([np.nan]*3),pd.Series([1.,np.nan,np.nan]),pd.Series([1.,.5,0.])):
        with pytest.raises(AssertionError,match='missingness'):
            audit.validate_publication('Pro_League_20_21',h,broken,expected,checks)
    with pytest.raises(AssertionError,match='eligible count'):
        audit.validate_publication('Pro_League_20_21',h,values,{**expected,'expected_eligible':3},checks)
    with pytest.raises(AssertionError,match='effective round'):
        audit.validate_publication('Pro_League_20_21',h,pd.Series([1.,0.,np.nan]),expected,checks)


def test_missing_status_never_counts_and_live_start_does():
    h=schedule((3,3,3,1),(0,0,0,1))
    assert season_countdown(h.drop(columns='status'),clock()).iloc[-1]==1
    assert season_countdown(h.assign(status='inprogress'),clock()).iloc[-1]==.5


def test_migration_preserves_auxiliary_files_and_refuses_unknown_modes(ui_recipe):
    ui_recipe['features']={'clock':node('features.SeasonCountdown',schedule_mode='retrospective')}
    ui_recipe['feature_options']={'schedule_history':node('input.Table',path='archive.parquet')}
    migrated,changes=migrate_season_countdown_recipe(ui_recipe)
    assert changes and migrated['feature_options']==ui_recipe['feature_options']
    ui_recipe['features']['clock']['params']['schedule_mode']='invented'
    with pytest.raises(ValueError,match='unrecognized legacy'):
        migrate_season_countdown_recipe(ui_recipe)


def test_fixed_format_membership_and_unknown_formats_fail_clearly():
    h=schedule(source_league='Unknown')
    with pytest.raises(ValueError,match='verified format'):
        season_countdown(h,SeasonCountdown(team_count=20))
    for fmt in ('scotland','japan_transition'):
        with pytest.raises(ValueError,match='requires'):
            season_countdown(h.assign(tournament_id=54 if fmt=='scotland' else 171852),SeasonCountdown(format=fmt,team_count=18))
