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


def retrospective(*args, **kwargs):
    return SeasonCountdown(*args, schedule_mode='retrospective', **kwargs)


def schedule(rounds=(1, 2, 3), days=(0, 7, 14), **extra):
    return pd.DataFrame(dict(competition_id=1, season_id=1,
        event_id=np.arange(len(rounds)), round=rounds,
        kickoff_at=pd.Timestamp('2020-01-01', tz='UTC') + pd.to_timedelta(days, unit='D'), **extra))


@pytest.mark.parametrize('n,total', [(18,34),(19,38),(20,38),(22,42),(24,46)])
def test_double_round_robin_even_odd_and_partial(n,total):
    h=schedule()
    x=season_countdown(h,retrospective(format='double',team_count=n))
    np.testing.assert_allclose(x, (total-np.arange(1,4))/(total-1))
    last=schedule((total,), (100,))
    assert season_countdown(last,retrospective(format='double',team_count=n)).iloc[0]==0


@pytest.mark.parametrize('value',[0,1,-1,True,2.5,'38',float('inf')])
@pytest.mark.parametrize('parameter',['total_rounds','team_count'])
def test_invalid_configuration(value,parameter):
    with pytest.raises(ValueError,match='at least 2'):
        retrospective(**{parameter:value})


def test_postponed_round_uses_active_round_and_isolated_advanced_match_not_anchor():
    # Round 3 has an advanced singleton on day 1; its main cluster starts day 14.
    h=schedule((1,2,2,2,3,3,3,1),(0,7,7,8,1,14,15,16))
    h.index=[8]*len(h)
    x=season_countdown(h,retrospective(3))
    np.testing.assert_array_equal(x,[1,.5,.5,.5,0,0,0,0])
    # Reverse input and duplicate team rows cannot change anchors or weighting.
    paired=pd.concat([h,h]).iloc[::-1]
    got=season_countdown(paired,retrospective(3))
    np.testing.assert_array_equal(got,np.tile(x,2)[::-1])


def test_between_rounds_and_after_final_round():
    h=schedule((1,2,3,1,1,1),(0,7,14,10,14,30))
    np.testing.assert_array_equal(season_countdown(h,retrospective(3)),[1,.5,0,.5,0,0])


def test_missing_schedule_values_and_conflicting_copies():
    h=schedule((1,None,3),(0,7,14));h.loc[2,'kickoff_at']=pd.NaT
    np.testing.assert_allclose(season_countdown(h,retrospective(3)),[1,np.nan,np.nan],equal_nan=True)
    broken=pd.concat([h,h.iloc[[0]].assign(round=2)])
    with pytest.raises(ValueError,match='inconsistent fixture'):
        season_countdown(broken,retrospective(3))
    with pytest.raises(ValueError,match='conflicts'):
        season_countdown(h.assign(round=[2,None,3]),retrospective(3),schedule_history=h)
    with pytest.raises(ValueError,match='absent'):
        season_countdown(h.assign(event_id=[3,4,5]),retrospective(3),schedule_history=h)


@pytest.mark.parametrize('bad',[0,-1,1.5,float('inf'),4])
def test_invalid_rounds(bad):
    with pytest.raises(ValueError,match='round'):
        season_countdown(schedule((bad,),(0,)),retrospective(3))


def test_membership_required_unknown_format_and_season_isolation():
    h=schedule(source_league='Premier_League')
    with pytest.raises(ValueError,match='team_seasons'):
        season_countdown(h,retrospective())
    roster=pd.DataFrame(dict(competition_id=1,season_id=1,team_id=range(20)))
    np.testing.assert_allclose(season_countdown(h,retrospective(),team_seasons=roster),[1,36/37,35/37])
    with pytest.raises(ValueError,match='verified format'):
        season_countdown(h.assign(source_league='Unknown'),retrospective(),team_seasons=roster)
    other=h.assign(season_id=2,kickoff_at=h.kickoff_at+pd.Timedelta(days=365))
    np.testing.assert_array_equal(season_countdown(pd.concat([h,other]),retrospective(3)),[1,.5,0,1,.5,0])


@pytest.mark.parametrize('league,n,total,tournament',[
    ('Liga_MX_Apertura',18,17,28),('Liga_MX_Apertura',19,19,28),
    ('Liga_MX_Clausura',18,17,16753),('Premiership',12,38,54),('Pro_League',18,34,38)])
def test_native_formats(league,n,total,tournament):
    h=schedule((1,total),(0,100),source_league=league,tournament_id=tournament)
    np.testing.assert_array_equal(season_countdown(h,retrospective(team_count=n)),[1,0])
    if league!='Premiership':
        h.loc[1,'tournament_id']=999
        assert np.isnan(season_countdown(h,retrospective(team_count=n)).iloc[1])


def test_scotland_split_continues_without_reset():
    h=schedule((33,34,38),(0,7,35),source_league='Premiership',tournament_id=[54,92066,92067])
    np.testing.assert_allclose(season_countdown(h,retrospective()),[5/37,4/37,0])


def test_japan_two_stage_labels_and_regional_transition():
    h=schedule((1,17,1,17),(0,7,14,21),source_league='J1',source_season='16_16',season_phase=[1,1,2,2])
    h['kickoff_at']=pd.to_datetime(['2016-02-27','2016-06-25','2016-07-02','2016-11-03'],utc=True)
    np.testing.assert_allclose(season_countdown(h,retrospective(team_count=18)),[1,17/33,16/33,0])
    h=schedule((1,18,1,18),(0,7,14,21),source_league='J1_Transition',tournament_id=[171852,171852,171853,171853])
    np.testing.assert_array_equal(season_countdown(h,retrospective()),[1,0,1,0])


def test_results_do_not_enter_clock_and_filtered_queries_keep_full_schedule():
    h=schedule((1,2,3,1),(0,7,14,15))
    baseline=season_countdown(h,retrospective(3))
    changed=h.assign(goals_for=[1000]*4,status='notstarted',result='X',available_at='2100-01-01')
    pd.testing.assert_series_equal(baseline,season_countdown(changed,retrospective(3)))
    pd.testing.assert_series_equal(baseline.iloc[[3,0]],season_countdown(h.iloc[[3,0]],retrospective(3),schedule_history=h))


def test_ui_json_python_preparation_fixture_scope_and_context(ui_recipe):
    catalog=catalog_for_ui()
    expression=retrospective(8)
    assert catalog.build(json.loads(json.dumps(catalog.encode(expression))))==expression
    fields=next(s for s in catalog.schema() if s['id']=='features.SeasonCountdown')['fields']
    assert {f['name'] for f in fields}=={'total_rounds','format','team_count','schedule_mode'}
    ui_recipe['features']={'season_countdown':node('features.SeasonCountdown',schedule_mode='retrospective',total_rounds=8)}
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
    np.testing.assert_array_equal(season_countdown(h,retrospective(team_count=18)),[1,0])


def test_single_leg_with_only_one_round_is_undefined():
    with pytest.raises(ValueError,match='two planned rounds'):
        season_countdown(schedule((1,),(0,)),retrospective(format='single',team_count=2))


def test_roster_cannot_omit_an_observed_team():
    h=schedule(source_league='Premier_League',home_id=1,away_id=3)
    roster=pd.DataFrame(dict(competition_id=1,season_id=1,team_id=[1,2]))
    with pytest.raises(ValueError,match='omits scheduled'):
        season_countdown(h,retrospective(),team_seasons=roster)


def test_clock_is_fixture_scoped_through_arithmetic_and_cutoffs():
    from xdiyo_analytics.features import Difference,Constant
    from transition_samples import history_from_games
    h=history_from_games([dict(round=1,day=0),dict(round=2,day=7),dict(round=3,day=14)])
    feature=Difference(Constant(1),retrospective(3))
    a=evaluate_features(h,{'elapsed':feature},cutoffs=h.kickoff_at-pd.Timedelta(days=1))
    b=evaluate_features(h,{'elapsed':feature},cutoffs=h.kickoff_at-pd.Timedelta(days=2))
    np.testing.assert_array_equal(a.elapsed,b.elapsed)
    assert a.attrs['feature_scopes']=={'elapsed':'fixture'}


def test_real_postponement_and_two_stage_source_conventions():
    from pathlib import Path
    from xdiyo_analytics.data import load_seasons
    root=Path(__file__).resolve().parents[2]/'data/xDiyo_data'
    d=load_seasons(root,'23_24',leagues=['Premier_League'],tables=['matches','team_seasons'],include_awarded=True)
    x=season_countdown(d.matches,retrospective(),team_seasons=d['team_seasons'])
    # Chelsea-Tottenham: assigned round 26; actual kickoff during round 35.
    mask=d.matches.event_id.eq(12226774)
    assert mask.sum()==1
    assert d.matches.loc[mask,'round'].iloc[0]==26
    assert x[mask].iloc[0]==3/37
    h=schedule((1,18,34),(0,100,200),source_league='J1',source_season='15_15',tournament_id=82)
    np.testing.assert_allclose(season_countdown(h,retrospective(team_count=18)),[1,16/33,0])


def test_explicit_countdown_does_not_shadow_standings_membership():
    from pathlib import Path
    from examples.derived_standings import recipe
    root=Path(__file__).resolve().parents[2]/'data/xDiyo_data'
    spec=recipe(root)
    spec['features']={'standing':node('features.DerivedStandings',normalized_ranks=[],include_raw=True),
                      'clock':node('features.SeasonCountdown',schedule_mode='retrospective',total_rounds=38)}
    prepared=prepare_recipe(spec)
    assert len(prepared.dataset.X)==380
    assert prepared.dataset.X['fixture::clock'].notna().all()
    assert any('standing' in c for c in prepared.dataset.X)


def test_japan_original_phase_survives_postponement_across_stage_boundary():
    h=schedule((10,1,2,2),(190,182,189,189),source_league='J1',source_season='16_16',
               season_phase=[1,2,2,2],tournament_id=82)
    h['kickoff_at']=pd.to_datetime(['2016-07-10','2016-07-02','2016-07-09','2016-07-09'],utc=True)
    # Delayed first-stage round 10 is played during season round 19, not 27.
    np.testing.assert_allclose(season_countdown(h,retrospective(team_count=18)),[15/33,16/33,15/33,15/33])
    with pytest.raises(ValueError,match='original season_phase'):
        season_countdown(h.drop(columns='season_phase'),retrospective(team_count=18))


def test_packaged_japan_phase_uses_identity_never_dates():
    from importlib.resources import files
    from xdiyo_analytics.features.countdown import _japan_phases
    with files('xdiyo_analytics.features').joinpath('data/j1_2016_phases.csv').open('rb') as f:
        reference=pd.read_csv(f)
    assert len(reference)==306
    assert reference.season_phase.value_counts().to_dict()=={1:153,2:153}
    reference['kickoff_at']=pd.Timestamp('2099-01-01',tz='UTC')
    np.testing.assert_array_equal(_japan_phases(reference.drop(columns='season_phase')),reference.season_phase)
    with pytest.raises(ValueError,match='conflicts with verified'):
        _japan_phases(reference.assign(season_phase=3-reference.season_phase))
    with pytest.raises(ValueError,match='original season_phase'):
        _japan_phases(reference.drop(columns='season_phase').assign(home_id=-1))


def test_scotland_unrecognized_stages_are_excluded_and_cannot_move_anchors():
    h=schedule((1,2,38,38),(0,7,0,0),source_league='Premiership',tournament_id=[54,54,999,None])
    np.testing.assert_allclose(season_countdown(h,retrospective()),[1,36/37,np.nan,np.nan],equal_nan=True)
    with pytest.raises(ValueError,match='tournament_id'):
        season_countdown(h.drop(columns='tournament_id'),retrospective())


def test_schedule_assumption_is_explicit_and_serialized():
    h=schedule()
    with pytest.raises(ValueError,match='schedule_snapshot_at'):
        season_countdown(h,SeasonCountdown(3))
    with pytest.raises(ValueError,match='schedule_mode'):
        SeasonCountdown(schedule_mode='guess')
    catalog=catalog_for_ui()
    for mode in ('as_of','retrospective'):
        expr=SeasonCountdown(3,schedule_mode=mode)
        assert catalog.build(json.loads(json.dumps(catalog.encode(expr))))==expr


def test_asof_snapshots_ignore_future_revision_and_respect_cutoff_alignment():
    h=schedule((1,1,2,2,3),(0,9,7,7,14))
    original=h.assign(schedule_snapshot_at='2019-12-31')
    revised=h.copy()
    revised.loc[[2,3],'kickoff_at']=pd.Timestamp('2020-02-01',tz='UTC')
    revised['schedule_snapshot_at']='2020-01-11'
    snapshots=pd.concat([original,revised],ignore_index=True)
    query=h.iloc[[1]].copy(); query.index=[123]
    cutoff=pd.Series(pd.to_datetime(['2020-01-10'],utc=True),index=query.index)
    expr=SeasonCountdown(3)
    assert season_countdown(query,expr,schedule_history=snapshots,cutoffs=cutoff).iloc[0]==.5
    # Final schedule clustering really would change this historical value.
    assert season_countdown(query,retrospective(3),schedule_history=revised).iloc[0]==1
    changed=snapshots.copy(); changed.loc[changed.schedule_snapshot_at.eq('2020-01-11'),'kickoff_at']+=pd.Timedelta(days=100)
    pd.testing.assert_series_equal(season_countdown(query,expr,schedule_history=snapshots,cutoffs=cutoff),
                                  season_countdown(query,expr,schedule_history=changed,cutoffs=cutoff))
    paired=pd.concat([query,query])
    assert season_countdown(paired,expr,schedule_history=snapshots).tolist()==[.5,.5]
    with pytest.raises(ValueError,match='align'):
        season_countdown(query,expr,schedule_history=snapshots,cutoffs=cutoff.reset_index(drop=True))


def test_asof_latest_inclusive_snapshot_and_fail_closed_inputs():
    h=schedule((1,2,3),(0,7,14))
    first=h.assign(schedule_snapshot_at='2019-12-30')
    second=h.assign(schedule_snapshot_at='2020-01-01')
    x=season_countdown(h,SeasonCountdown(3),schedule_history=pd.concat([first,second]))
    np.testing.assert_array_equal(x,[1,.5,0])
    with pytest.raises(ValueError,match='No schedule snapshot known'):
        season_countdown(h,SeasonCountdown(3),schedule_history=first,cutoffs=pd.Timestamp('2019-12-01',tz='UTC'))
    with pytest.raises(ValueError,match='no later'):
        season_countdown(h,SeasonCountdown(3),schedule_history=first,cutoffs=h.kickoff_at+pd.Timedelta(days=1))
    with pytest.raises(ValueError,match='must be known'):
        season_countdown(h,SeasonCountdown(3),schedule_history=first.assign(schedule_snapshot_at=pd.NaT))
    with pytest.raises(ValueError,match='conflicts'):
        season_countdown(h,SeasonCountdown(3),schedule_history=first.assign(round=1))


def test_asof_evaluator_uses_cutoffs_for_both_team_rows():
    from transition_samples import history_from_games
    h=history_from_games([dict(round=1,day=0),dict(round=2,day=7),dict(round=3,day=14)])
    snapshots=h.assign(schedule_snapshot_at=h.kickoff_at.min()-pd.Timedelta(days=2))
    cutoffs=h.kickoff_at-pd.Timedelta(days=1)
    features=evaluate_features(h,{'clock':SeasonCountdown(3)},cutoffs=cutoffs,schedule_history=snapshots)
    assert features.attrs['feature_scopes']=={'clock':'fixture'}
    assert features.clock.tolist()==[1,1,.5,.5,0,0]
    from xdiyo_analytics.features.contextual import evaluate_context_features
    extra=evaluate_context_features(h,{'clock':SeasonCountdown(3)},cutoffs=cutoffs,schedule_history=snapshots)
    np.testing.assert_array_equal(features.clock,extra.clock)


def test_asof_recipe_needs_snapshots(ui_recipe):
    ui_recipe['features']={'clock':node('features.SeasonCountdown',total_rounds=8)}
    with pytest.raises(ValueError,match='explicit schedule_history'):
        prepare_recipe(ui_recipe)


def test_asof_recipe_context_export_and_cutoff_hours(ui_recipe,tmp_path):
    from xdiyo_analytics.data import load_seasons
    data=load_seasons(**{**ui_recipe['data'],'tables':['matches'],'include_awarded':True})
    snapshots=data.matches.assign(schedule_snapshot_at='2000-01-01')
    path=tmp_path/'snapshots.parquet'; snapshots.to_parquet(path,index=False)
    ui_recipe['feature_options']={'schedule_history':node('input.Table',path=str(path))}
    ui_recipe['features']={'clock':node('features.SeasonCountdown',total_rounds=8)}
    ui_recipe['cutoff_hours']=24
    a=prepare_recipe(ui_recipe)
    context=deepcopy(ui_recipe);context['features']={'home':node('features.IsHome')}
    context['context_features']=ui_recipe['features']
    b=prepare_recipe(context)
    np.testing.assert_array_equal(a.dataset.X['fixture::clock'],b.dataset.X.clock)
    namespace={};exec(export_python(context).split('result = run_recipe')[0],namespace)
    pd.testing.assert_frame_equal(b.dataset.X,prepare_recipe(namespace['recipe']).dataset.X)
    # Earliest snapshot is now after the first issuance but still before kickoff.
    snapshots['schedule_snapshot_at']=pd.to_datetime(snapshots.kickoff_utc,unit='s',utc=True).min()-pd.Timedelta(hours=12)
    snapshots.to_parquet(path,index=False)
    with pytest.raises(ValueError,match='No schedule snapshot known'):
        prepare_recipe(context)


def test_publication_audit_detects_all_nan_missing_exclusions_counts_and_rounds():
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[2]/'docs/analytics/season-countdown/audit.py'
    spec=importlib.util.spec_from_file_location('countdown_audit',path)
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    h=schedule((1,2,1),(0,7,14),tournament_id=[38,38,999])
    expected={'matches':3,'expected_eligible':2,'expected_excluded':1}
    values=pd.Series([1.,.5,np.nan])
    check=[dict(publication='Pro_League_20_21',event_id=1,total_rounds=3,effective_round=2)]
    audit.validate_publication('Pro_League_20_21',h,values,expected,check)
    for broken in (pd.Series([np.nan]*3),pd.Series([1.,np.nan,np.nan]),pd.Series([1.,.5,0.])):
        with pytest.raises(AssertionError,match='missingness'):
            audit.validate_publication('Pro_League_20_21',h,broken,expected,check)
    with pytest.raises(AssertionError,match='eligible count'):
        audit.validate_publication('Pro_League_20_21',h,values,{**expected,'expected_eligible':3},check)
    with pytest.raises(AssertionError,match='effective round'):
        audit.validate_publication('Pro_League_20_21',h,pd.Series([1.,0.,np.nan]),expected,check)
    with pytest.raises(AssertionError,match='fixture count'):
        audit.validate_publication('Pro_League_20_21',h.iloc[:2],values.iloc[:2],expected,check)


def test_latest_snapshot_is_selected_inclusively():
    h=schedule((1,1,2,2,3),(0,9,7,7,14))
    old=h.assign(schedule_snapshot_at='2019-12-31')
    new=h.copy();new.loc[[2,3],'kickoff_at']=pd.Timestamp('2020-02-01',tz='UTC')
    new['schedule_snapshot_at']='2020-01-09'
    snapshots=pd.concat([old,new],ignore_index=True)
    target=h.iloc[[1]]
    before=pd.Timestamp('2020-01-09',tz='UTC')-pd.Timedelta(nanoseconds=1)
    at=pd.Timestamp('2020-01-09',tz='UTC')
    assert season_countdown(target,SeasonCountdown(3),schedule_history=snapshots,cutoffs=before).iloc[0]==.5
    assert season_countdown(target,SeasonCountdown(3),schedule_history=snapshots,cutoffs=at).iloc[0]==1
