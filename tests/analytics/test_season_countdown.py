"""Nearest scheduled round-cluster countdown and native integration."""
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


def delayed_schedule():
    # Round 1 main cluster is day 0 (three fixtures); its catch-up is day 30.
    return schedule((1,1,1,1,35,35,35),(0,0,0,30,29,30,31))


def test_pre_match_round_and_postponed_fixture_before_any_start():
    h=delayed_schedule().assign(status='notstarted')
    early=pd.Timestamp('2019-12-01',tz='UTC')
    got=season_countdown(h,clock(38,20),cutoffs=early)
    np.testing.assert_array_equal(got,[1,1,1,3/37,3/37,3/37,3/37])
    # A shared early cutoff never delays round assignment until actual play.
    pd.testing.assert_series_equal(got,season_countdown(h,clock(38,20)))


@pytest.mark.parametrize('status',['notstarted','postponed','cancelled','finished','inprogress','unknown',None])
def test_played_status_results_and_unresolved_batch_do_not_change_schedule(status):
    h=delayed_schedule()
    expected=season_countdown(h,clock(38,20))
    changed=h.assign(status=status,play_status='unplayed',is_awarded=True,
        goals_for=999,goals_against=0,result_available_at='2100-01-01',
        prediction_batch_round=38,prediction_batch_id='mixed')
    pd.testing.assert_series_equal(expected,season_countdown(changed,clock(38,20)))
    pd.testing.assert_series_equal(expected,season_countdown(h.drop(columns='status'),clock(38,20)))


def test_nearest_interval_and_later_ordinal_ties_with_original_lower_bound():
    h=schedule((1,1,1,1,1,1,3,3,5,5),(0,0,0,12,14,40,10,12,16,18))
    got=season_countdown(h,clock())
    # Day 12 is inside round 3; day 14 equidistant from r3/r5 => r5;
    # day 40 falls after both and is closest to r5, without any ambiguity NaN.
    np.testing.assert_array_equal(got,[1,1,1,.5,0,0,.5,.5,0,0])
    advanced=schedule((5,5,5,5,2,2),(30,30,30,8,7,9))
    assert season_countdown(advanced,clock()).iloc[3]==0 # never below assigned 5


def test_main_cluster_largest_earliest_tie_and_exact_96_hour_span():
    from xdiyo_analytics.features.countdown import _round_clusters
    h=schedule((3,3,3,3,3),(20,21,1,5,40)).assign(_ordinal=3.,_eligible=True)
    rounds,starts,ends=_round_clusters(h)
    assert rounds.tolist()==[3]
    assert pd.Timestamp(starts[0],tz='UTC')==h.kickoff_at.iloc[2]
    assert pd.Timestamp(ends[0],tz='UTC')==h.kickoff_at.iloc[3]
    # One nanosecond beyond 96 hours breaks the earlier tied cluster.
    h['kickoff_at']=h.kickoff_at.astype('datetime64[ns, UTC]')
    h.loc[3,'kickoff_at']+=pd.Timedelta(nanoseconds=1)
    _,starts,_=_round_clusters(h)
    assert pd.Timestamp(starts[0],tz='UTC')==h.kickoff_at.iloc[0]


def test_own_cluster_wins_over_overlapping_later_round_and_advanced_singleton():
    h=schedule((1,1,1,5,5,5,5),(0,0,0,0,30,31,32))
    np.testing.assert_array_equal(season_countdown(h,clock()),[1,1,1,0,0,0,0])
    # Even overlapping main clusters do not reassign ordinary round-1 games.
    h=schedule((1,1,3,3),(0,1,0,1))
    np.testing.assert_array_equal(season_countdown(h,clock()),[1,1,.5,.5])


def test_row_order_duplicate_team_rows_and_queries_share_full_schedule():
    h=delayed_schedule()
    expected=season_countdown(h,clock(38,20))
    shuffled=h.sample(frac=1,random_state=7)
    pd.testing.assert_series_equal(expected,season_countdown(shuffled,clock(38,20)).sort_index())
    subset=h.iloc[[6,3,0]]
    pd.testing.assert_series_equal(expected.loc[subset.index],season_countdown(subset,clock(38,20),schedule_history=shuffled))
    paired=pd.concat([h,h]).iloc[::-1];paired.index=[8]*len(paired)
    np.testing.assert_array_equal(season_countdown(paired,clock(38,20),schedule_history=h),np.tile(expected,2)[::-1])
    # Repeated copies never add votes to another main cluster.
    repeat=pd.concat([h,h.iloc[[3]*8]])
    assert season_countdown(repeat,clock(38,20)).iloc[-1]==3/37


def test_calendar_uses_scheduled_context_not_cutoff_or_batch_override():
    h=delayed_schedule()
    query=h.iloc[[3]].assign(prediction_batch_round=37)
    assert season_countdown(query,clock(38,20),schedule_history=h,cutoffs=h.kickoff_at.iloc[0]).iloc[0]==3/37
    # Future scheduled context may change a historical export; no claim that
    # unavailable archived schedule knowledge has been reconstructed.
    changed=h.copy();changed.loc[4:,'kickoff_at']+=pd.Timedelta(days=100)
    assert season_countdown(query,clock(38,20),schedule_history=changed).iloc[0]==1


def test_total_override_needs_no_threshold_or_membership():
    h=delayed_schedule()
    pd.testing.assert_series_equal(season_countdown(h,SeasonCountdown(38)),season_countdown(h,clock(38,20)))
    with pytest.raises(ValueError,match='infer total_rounds'):
        season_countdown(h,SeasonCountdown(format='double'))


def test_no_clusters_retain_original_ordinal_and_extreme_timestamps():
    from xdiyo_analytics.features.countdown import _nearest_rounds
    assigned=np.array([1.,3.]);times=np.array([-9_000_000_000_000_000_000,9_000_000_000_000_000_000],dtype='int64')
    empty=(np.array([]),np.array([],dtype='int64'),np.array([],dtype='int64'))
    np.testing.assert_array_equal(_nearest_rounds(assigned,times,empty),assigned)
    clusters=(np.array([2.,4.]),times.copy(),times.copy())
    np.testing.assert_array_equal(_nearest_rounds(assigned,times,clusters),[2,4])


def test_missing_values_and_conflicting_fixture_copies():
    h=schedule((1,None,3),(0,7,14));h.loc[2,'kickoff_at']=pd.NaT
    np.testing.assert_allclose(season_countdown(h,clock()),[1,np.nan,np.nan],equal_nan=True)
    for field,value in [('round',2),('home_id',99)]:
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


def test_scotland_split_stages_share_continuous_calendar():
    h=schedule((1,1,1,1,34,34,38),(0,0,0,32,30,32,50),
        source_league='Premiership',tournament_id=[54]*4+[92066,92067,999])
    assert season_countdown(h,SeasonCountdown()).iloc[3]==4/37
    assert season_countdown(h.iloc[[3]],SeasonCountdown(),schedule_history=h,cutoffs=h.kickoff_at.iloc[0]).iloc[0]==4/37
    assert np.isnan(season_countdown(h,SeasonCountdown()).iloc[-1])
    with pytest.raises(ValueError,match='tournament_id'):
        season_countdown(h.drop(columns='tournament_id'),SeasonCountdown())


def test_japan_phase_identity_and_independent_transition_calendars():
    h=schedule((10,10,10,10,2,2),(0,0,0,20,19,21),source_league='J1',source_season='16_16',season_phase=[1,1,1,1,2,2],tournament_id=82)
    assert season_countdown(h,SeasonCountdown(team_count=18)).iloc[3]==15/33 # ordinal19, not false27
    with pytest.raises(ValueError,match='original season_phase'):
        season_countdown(h.drop(columns='season_phase'),SeasonCountdown(team_count=18))
    g=schedule((1,1,1,1,18,18,1),(0,0,0,30,30,30,30),source_league='J1_Transition',tournament_id=[171852]*6+[171853])
    np.testing.assert_array_equal(season_countdown(g,SeasonCountdown()),[1,1,1,0,0,0,1])


def test_verified_japan_resource_is_date_independent():
    from xdiyo_analytics.features.countdown import _japan_phases
    with files('xdiyo_analytics.features').joinpath('data/j1_2016_phases.csv').open('rb') as f:
        reference=pd.read_csv(f)
    assert len(reference)==306 and reference.season_phase.value_counts().to_dict()=={1:153,2:153}
    reference['kickoff_at']=pd.Timestamp('2099-01-01',tz='UTC')
    np.testing.assert_array_equal(_japan_phases(reference.drop(columns='season_phase')),reference.season_phase)
    with pytest.raises(ValueError,match='conflicts with verified'):
        _japan_phases(reference.assign(season_phase=3-reference.season_phase))


def test_season_and_competition_reset_and_odd_team_byes():
    h=schedule((1,1,1,1,3,3),(0,0,0,30,30,30))
    later=schedule((1,),(370,)).assign(season_id=2)
    other=schedule((1,),(30,)).assign(competition_id=2)
    x=season_countdown(pd.concat([h,later,other]),clock(n=9))
    assert x.iloc[3]==.5 and x.iloc[-2]==1 and x.iloc[-1]==1


def test_clusters_built_once_per_calendar_not_per_target(monkeypatch):
    import xdiyo_analytics.features.countdown as module
    original=module._round_clusters;calls=[]
    def tracked(*args):
        calls.append(len(args[0]));return original(*args)
    monkeypatch.setattr(module,'_round_clusters',tracked)
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
    assert features.clock.iloc[-1]==1 # a singleton is inside its own main cluster
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
    path=Path(__file__).resolve().parents[2]/'docs/analytics/season-countdown/audit_scheduled.py'
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


def test_status_differences_in_copies_do_not_change_schedule():
    h=delayed_schedule()
    copy=h.assign(status='notstarted')
    np.testing.assert_array_equal(season_countdown(pd.concat([h,copy]),clock(38,20)),np.tile(season_countdown(h,clock(38,20)),2))


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


def test_real_pipeline_preserves_postponement_and_explicit_japan_phase():
    from xdiyo_analytics.data.loading import SeasonData
    from xdiyo_analytics.histories import build_team_history
    from xdiyo_analytics.labels import create_labels, Outcome
    from xdiyo_analytics.datasets import assemble_dataset
    from xdiyo_analytics.features.contextual import evaluate_context_features
    h=schedule((10,10,10,10,2,2),(0,0,0,20,19,21),source_league='J1',source_season='16_16',season_phase=[1,1,1,1,2,2],tournament_id=82)
    matches=h.assign(home_id=np.arange(6)*2+1,away_id=np.arange(6)*2+2,
        home_score_current=1,away_score_current=0,
        kickoff_utc=h.kickoff_at.astype('datetime64[ns, UTC]').astype('int64')//10**9).drop(columns='kickoff_at')
    history=build_team_history(SeasonData({'matches':matches},{}))
    expr=SeasonCountdown(team_count=18)
    features=evaluate_features(history,{'clock':expr},schedule_history=matches,keyed=True,
        cutoffs=pd.Timestamp('2019-12-01',tz='UTC'))
    labels=create_labels(history,{'outcome':Outcome(perspective='home')})['outcome']
    dataset=assemble_dataset(features,labels,layout='match')
    context=evaluate_context_features(dataset.metadata,{'clock':expr},schedule_history=matches,
        cutoffs=pd.Timestamp('2019-12-01',tz='UTC'))
    np.testing.assert_array_equal(dataset.X['fixture::clock'],context.clock)
    assert dataset.X['fixture::clock'].iloc[3]==15/33
    assert 'season_phase' in dataset.metadata
    assert dataset.X.columns.tolist()==['fixture::clock']


def test_chunked_queries_preserve_order_and_exact_nanosecond_ties():
    from xdiyo_analytics.features.countdown import _nearest_rounds
    n=8201
    assigned=np.ones(n)
    instants=np.tile(np.array([9,10,11,12,13,14],dtype='int64'),1367)[:n]
    clusters=(np.array([2.,3.]),np.array([8,14],dtype='int64'),np.array([10,16],dtype='int64'))
    want=np.array([2 if t<12 else 3 for t in instants])
    np.testing.assert_array_equal(_nearest_rounds(assigned,instants,clusters),want)
