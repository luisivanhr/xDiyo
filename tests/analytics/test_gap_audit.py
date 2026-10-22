"""Adversarial regressions from the paper/prompt audit; all data are synthetic."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.features import GAPFixture, Rating, evaluate_features
from xdiyo_analytics.ratings import GAPParameters, GAPRatingRun, GAPSeason, tune_gap
from gap_fixtures import fixture, statistic, season_plan, vintage, build, PARAMS


def test_query_requires_explicit_production_cutoffs_on_all_feature_paths():
    h = pd.DataFrame(fixture(1,'2020-01-02') + fixture(2,'2020-01-03'))
    h['round'] = 1
    run = build(h)
    for method in (run.features, run.fixture_features, run.diagnostics):
        with pytest.raises(ValueError, match='explicit.*cutoff'):
            method(h)
    for expression in (GAPFixture('g'), Rating('g')):
        with pytest.raises(ValueError, match='explicit.*cutoff'):
            evaluate_features(h, {'gap':expression}, ratings={'g':run})
    frozen = pd.Series(pd.Timestamp('2020-01-02',tz='UTC'), index=h.index)
    assert run.fixture_features(h,cutoffs=frozen).eq(0).all().all()


def test_fixture_perspectives_cannot_use_different_cutoffs():
    h = pd.DataFrame(fixture(1,'2020-01-02') + fixture(2,'2020-01-04'))
    query = h.iloc[2:]
    cutoffs = pd.Series(pd.to_datetime(['2020-01-02','2020-01-04'],utc=True),index=query.index)
    run = build(h)
    with pytest.raises(ValueError, match='perspective.*cutoff'):
        run.fixture_features(query, cutoffs=cutoffs)


def test_declared_prediction_batch_must_have_one_frozen_cutoff():
    h = pd.DataFrame(fixture(1,'2020-01-02') + fixture(2,'2020-01-04'))
    h['prediction_batch_id'] = 'same-production-round'
    with pytest.raises(ValueError, match='batch.*cutoff'):
        build(h).fixture_features(h, cutoffs=h.kickoff_at)


def test_postponed_previous_season_retains_its_parameter_vintage(tmp_path):
    h = pd.DataFrame(fixture(1,'2020-01-02') + fixture(2,'2021-01-02',season=2021) +
                     fixture(3,'2021-01-04',season=2020))
    plans = [season_plan(),season_plan(season=2021,entry='2021-01-01',previous=2020,
                                      members=((1,'retained'),(2,'retained')))]
    old = vintage(end='2021-01-01')
    new = vintage(start='2021-01-01',fit='2021-01-01',parameters=GAPParameters(.9,.1,.9))
    run = build(h,plans=plans,vintages=[old,new])
    # Previous-season match remains a .2/.75/.6 fixture after the calendar boundary.
    result = run.fixture_features(h.iloc[4:],cutoffs=h.iloc[4:].kickoff_at)
    assert result.iloc[0,0] == pytest.approx(.27)
    run.save(tmp_path/'postponed')
    pd.testing.assert_frame_equal(result,GAPRatingRun.load(tmp_path/'postponed').fixture_features(h.iloc[4:],cutoffs=h.iloc[4:].kickoff_at))


def test_unselected_future_season_metadata_does_not_block_past_build():
    h = pd.DataFrame(fixture(1,'2020-01-02'))
    future = season_plan(season=2025,entry='2025-01-01',previous=2024,
                         members=((1,'retained'),(2,'retained')))
    # Unknown future predecessor must be checked when that season becomes active,
    # not require information about 2024 to produce a January 2020 snapshot.
    expected=build(h,cutoff='2020-01-03').fixture_features(h,cutoffs=h.kickoff_at)
    actual=build(h,plans=[season_plan(),future],cutoff='2020-01-03').fixture_features(h,cutoffs=h.kickoff_at)
    pd.testing.assert_frame_equal(expected,actual)


def test_tuning_cannot_infer_closed_training_seasons_from_target_date(monkeypatch):
    import scipy.optimize
    h = pd.DataFrame(fixture(1,'2020-01-02') + fixture(2,'2020-01-04'))
    def forbidden(*args, **kwargs):
        pytest.fail('Optimizer reached with an undeclared target season and no season closure evidence')
    monkeypatch.setattr(scipy.optimize,'minimize',forbidden)
    with pytest.raises(ValueError,match='target season|closed'):
        tune_gap(h,statistic=statistic(),seasons=[season_plan()],fit_cutoff='2020-01-31',target_start='2020-02-01',
                 initial=PARAMS,cutoffs=h.kickoff_at,objective_mask=[True]*4,
                 maxiter=1,maxfev=4,xatol=.001,fatol=.001)


def test_context_adapter_must_not_discard_explicit_unknown_predecessor():
    from xdiyo_analytics.features import TransitionContext
    from xdiyo_analytics.ratings import gap_season_from_context
    h = pd.DataFrame(fixture(1,'2020-01-02') + fixture(2,'2021-01-02',season=2021))
    h['season_year'] = h.season_id.astype(str)
    context = TransitionContext(h,season_starts={(10,2020):'2020-01-01',(10,2021):'2021-01-01'},
                                team_seasons=[dict(competition_id=10,season_id=2021,team_id=t,movement='retained',
                                                   previous_competition_id=None,previous_season_id=None) for t in (1,2)])
    with pytest.raises(ValueError,match='predecessor'):
        gap_season_from_context(context,competition_id=10,season_id=2021,statistic='goals',
                                known_at='2020-12-31',evidence='Synthetic membership with unknown predecessor')


def _tuning_case():
    h = pd.DataFrame(fixture(1,'2020-01-02') + fixture(2,'2020-01-04'))
    old = season_plan(closed_at='2020-12-30')
    target = season_plan(season=2021,entry='2021-01-01',previous=2020,members=((1,'retained'),(2,'retained')))
    options = dict(statistic=statistic(),seasons=[old],target_seasons=[target],fit_cutoff='2020-12-31',
                   target_start='2021-01-01',initial=PARAMS,cutoffs=h.kickoff_at,objective_mask=[True]*4,
                   maxiter=1,maxfev=4,xatol=.001,fatol=.001)
    return h, old, target, options


@pytest.mark.parametrize('fault, message', [('target_overlap','Held-out'),('unclosed','closed_at'),
                                           ('closure_after_fit','closed_at'),('target_date','target_start'),
                                           ('observation_after_closure','closure'),('unknown_fixture_season','closed training')])
def test_bad_season_folds_fail_before_optimizer(monkeypatch,fault,message):
    import scipy.optimize
    h, old, target, options = _tuning_case()
    monkeypatch.setattr(scipy.optimize,'minimize',lambda *a,**k: pytest.fail('Unsafe fold reached optimizer'))
    if fault == 'target_overlap': options['target_seasons']=[old]
    elif fault == 'unclosed': options['seasons']=[replace(old,closed_at=None)]
    elif fault == 'closure_after_fit': options['seasons']=[replace(old,closed_at='2021-01-01')]
    elif fault == 'target_date': options['target_start']='2021-01-02'
    elif fault == 'observation_after_closure': options['seasons']=[replace(old,closed_at='2020-01-03')]
    elif fault == 'unknown_fixture_season': h['season_id']=2019
    with pytest.raises(ValueError,match=message):
        tune_gap(h,**options)


def test_released_after_fit_data_cannot_enter_optimizer(monkeypatch):
    import scipy.optimize
    h, old, target, options = _tuning_case()
    h.loc[h.event_id.eq(1),'result_available_at']=pd.Timestamp('2021-01-02',tz='UTC')
    monkeypatch.setattr(scipy.optimize,'minimize',lambda *a,**k: pytest.fail('Future release reached optimizer'))
    with pytest.raises(ValueError,match='after the parameter fit cutoff'):
        tune_gap(h,**options)


def test_all_target_outcomes_are_forbidden_even_if_masked_out(monkeypatch):
    import scipy.optimize
    h, old, target, options = _tuning_case()
    h=pd.concat([h,pd.DataFrame(fixture(3,'2021-01-02',season=2021))],ignore_index=True)
    options.update(cutoffs=h.kickoff_at,objective_mask=[True]*4+[False]*2)
    monkeypatch.setattr(scipy.optimize,'minimize',lambda *a,**k: pytest.fail('Held-out outcomes reached optimizer'))
    with pytest.raises(ValueError,match='after the parameter fit cutoff'):
        tune_gap(h,**options)


def test_fixture_half_missing_cutoff_is_inconsistent_but_whole_missing_is_unavailable():
    h=pd.DataFrame(fixture(1,'2020-01-02'))
    run=build(h)
    times=h.kickoff_at.copy()
    times.iloc[0]=pd.NaT
    with pytest.raises(ValueError,match='perspective'):
        run.features(h,cutoffs=times)
    times.iloc[:]=pd.NaT
    assert run.features(h,cutoffs=times).isna().all().all()


def test_legacy_artifact_requires_rebuild_and_new_artifact_enforces_cutoffs(tmp_path):
    import json
    h=pd.DataFrame(fixture(1,'2020-01-02'))
    run=build(h)
    run.save(tmp_path/'run')
    loaded=GAPRatingRun.load(tmp_path/'run')
    with pytest.raises(ValueError,match='explicit'):
        loaded.fixture_features(h)
    assert loaded.metadata['vintage_selection']=='season_entry'
    manifest=tmp_path/'run'/'gap.json'
    data=json.loads(manifest.read_text())
    data['schema']=1
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='rebuild schema-1'):
        GAPRatingRun.load(tmp_path/'run')


def test_cross_stat_season_entry_must_agree():
    h=pd.DataFrame(fixture(1,'2020-01-02'))
    with pytest.raises(ValueError,match='agree.*entry'):
        build(h,stats=('goals','corners'),plans=[season_plan(),season_plan(stat='corners',entry='2020-01-02')])


@pytest.mark.parametrize('value',[1,np.int64(1),True])
def test_numeric_season_and_vintage_dates_are_not_interpreted_as_nanoseconds(value):
    with pytest.raises(ValueError,match='datetime'):
        replace(season_plan(),known_at=value)
    with pytest.raises(ValueError,match='datetime'):
        replace(vintage(),fit_cutoff=value)


def test_fixture_perspectives_cannot_claim_different_prediction_batches():
    h = pd.DataFrame(fixture(1, '2020-01-02'))
    h['prediction_batch_id'] = ['batch-a', 'batch-b']
    with pytest.raises(ValueError, match='perspectives.*batch'):
        build(h).fixture_features(h, cutoffs=h.kickoff_at)
