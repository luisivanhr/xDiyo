"""Fit-local weights, original class identities and supported adapter routing."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, SGDClassifier, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xdiyo_analytics.weighting import ClassWeightPolicy
from xdiyo_analytics.reporting import ClassWeightReporter
from xdiyo_analytics.analysis import PreTrainingAnalysis
from xdiyo_analytics.selection import Candidate, ModelSelection
from xdiyo_analytics.splits import Fold, SplitPlan
from xdiyo_analytics.training import (EstimatorAdapter, TrainingRunner, ProbabilityCalibrator,
    ValidationTail, PartialFitBackend, FitContext, save_model, load_model, refit_model)
from test_probability_calibration import classification_data, ScopedProbabilities
from model_selection_samples import outer
from split_samples import rows_for
from test_ui_workflow import ui_recipe
from execution_samples import TraceAdapter, ParentObserver


def weight_context(labels=(2, 2, 2, 7, 7, 12)):
    index = pd.Index([50, 10, 80, 20, 90, 30], name='row_position')
    X = pd.DataFrame({'signal': [0., 1., 2., 1., 3., 4.]}, index=index)
    return FitContext(X=X, y=pd.DataFrame({'corners': labels}, index=index),
        metadata=pd.DataFrame({'event_id': index}, index=index), layout='match', match_columns=('event_id',), fold_id=0)


@pytest.mark.parametrize('mode,power', [('none', 1.), ('balanced', 1.), ('power', 0.), ('power', .5), ('power', 2.)])
def test_formula_and_original_labels(mode, power):
    ctx = weight_context()
    actual = ClassWeightPolicy(mode=mode, power=power).compute(ctx)
    counts = ctx.y.corners.value_counts()
    exponent = 0. if mode == 'none' else 1. if mode == 'balanced' else power
    expected = (len(ctx.y)/(3*ctx.y.corners.map(counts)))**exponent
    expected /= expected.mean()
    pd.testing.assert_index_equal(actual.values.index, ctx.y.index)
    np.testing.assert_allclose(actual.values, expected)
    assert actual.target == 'corners' and actual.values.mean() == pytest.approx(1.)
    table = actual.classes.set_index('class')
    assert table['count'].to_dict() == {2: 3, 7: 2, 12: 1}
    assert table.weighted_proportion.sum() == pytest.approx(1.)
    if mode == 'balanced':
        np.testing.assert_allclose(table.weighted_proportion, 1/3)


def test_custom_maps_original_count_labels_and_json_keys():
    ctx = weight_context()
    a = ClassWeightPolicy(mode='custom', class_weights={2: 1., 7: 3., 12: 5.}).compute(ctx)
    b = ClassWeightPolicy(mode='custom', class_weights={'2': 1., '7': 3., '12': 5.}).compute(ctx)
    expected = np.array([1, 1, 1, 3, 3, 5], dtype=float)
    expected /= expected.mean()
    np.testing.assert_allclose(a.values, expected)
    pd.testing.assert_series_equal(a.values, b.values)


def test_callback_reorders_exact_identities_and_normalizes_per_observation():
    ctx = weight_context()
    seen = []
    def calculate(supplied):
        seen.append(supplied)
        return pd.Series([1., 2., 3., 4., 5., 6.], index=ctx.y.index).iloc[::-1]
    actual = ClassWeightPolicy(mode='callable', calculator=calculate).compute(ctx)
    assert seen == [ctx]
    np.testing.assert_allclose(actual.values, np.arange(1, 7)/3.5)
    # Per-observation callbacks can vary inside one class; class table reports their mean.
    assert actual.classes.set_index('class').loc[2, 'weight'] == pytest.approx(2/3.5)


@pytest.mark.parametrize('kind', ['missing', 'extra', 'duplicate', 'negative', 'nan', 'all_zero', 'array'])
def test_callback_malformed_weights_rejected(kind):
    ctx = weight_context()
    values = pd.Series(1., index=ctx.y.index)
    if kind == 'missing': values = values.iloc[:-1]
    if kind == 'extra': values.loc[100] = 1.
    if kind == 'duplicate': values.index = [50]*len(values)
    if kind == 'negative': values.iloc[0] = -1.
    if kind == 'nan': values.iloc[0] = np.nan
    if kind == 'all_zero': values[:] = 0.
    if kind == 'array': values = values.to_numpy()
    with pytest.raises(ValueError):
        ClassWeightPolicy(mode='callable', calculator=lambda _: values).compute(ctx)


def test_weight_report_is_descriptive_until_consumed_and_wrong_rows_rejected():
    data = classification_data()
    fold = outer(data)
    plan = SplitPlan([fold], len(data.X), np.arange(len(data.X)))
    report = PreTrainingAnalysis({'weights': ClassWeightReporter()}).run(data, split_plan=plan)
    result = report.studies[0].result
    assert result.weights is not None and 'class_balance' in result.tables
    runner = TrainingRunner(lambda: EstimatorAdapter(LogisticRegression()))
    with pytest.raises(ValueError, match='explicitly'):
        runner.run(data, plan, analysis_report=report)
    # Excluding a validation tail makes the original whole-train report ineligible.
    with pytest.raises(ValueError, match='exactly'):
        TrainingRunner(lambda: EstimatorAdapter(LogisticRegression()), validation=ValidationTail(.25)).run(
            data, plan, analysis_report=report, weights_from='weights')


@pytest.mark.parametrize('pipeline', [False, True])
def test_estimator_routes_weights_once_and_matches_native_fit(pipeline):
    ctx = weight_context((0, 0, 0, 0, 1, 1))
    ctx.sample_weight = ClassWeightPolicy(mode='balanced').compute(ctx).values
    def make():
        model = LogisticRegression(C=.7, tol=1e-10)
        return Pipeline([('scale', StandardScaler()), ('classifier', model)]) if pipeline else model
    actual = EstimatorAdapter(make())
    actual.fit(ctx)
    reference = make()
    key = 'classifier__sample_weight' if pipeline else 'sample_weight'
    reference.fit(ctx.X, ctx.y.corners, **{key: ctx.sample_weight.to_numpy()})
    np.testing.assert_allclose(actual.estimator.predict_proba(ctx.X), reference.predict_proba(ctx.X), atol=1e-12)
    if pipeline:
        np.testing.assert_array_equal(actual.estimator[0].mean_, ctx.X.mean().to_numpy())


@pytest.mark.parametrize('estimator,error', [(Ridge(), TypeError), (KNeighborsClassifier(), TypeError),
    (LogisticRegression(class_weight='balanced'), ValueError)])
def test_unsupported_or_native_duplicate_weighting_is_explicit(estimator, error):
    ctx = weight_context((0, 0, 0, 0, 1, 1))
    ctx.sample_weight = ClassWeightPolicy().compute(ctx).values
    with pytest.raises(error):
        EstimatorAdapter(estimator).fit(ctx)


def test_partial_fit_routes_normalized_weights_and_rejects_duplicate_kwargs():
    ctx = weight_context((0, 0, 0, 0, 1, 1))
    ctx.sample_weight = ClassWeightPolicy(mode='balanced').compute(ctx).values
    settings = dict(loss='log_loss', random_state=23, shuffle=False)
    backend = PartialFitBackend(SGDClassifier(**settings), loss='log_loss', prediction_methods=('predict_proba',))
    backend.initialize(ctx)
    backend.step()
    reference = SGDClassifier(**settings).partial_fit(ctx.X, ctx.y.corners, classes=[0, 1], sample_weight=ctx.sample_weight.to_numpy())
    np.testing.assert_array_equal(backend.estimator.coef_, reference.coef_)
    with pytest.raises(ValueError, match='not both'):
        PartialFitBackend(SGDClassifier(**settings), partial_fit_kwargs={'sample_weight': np.ones(6)}).initialize(ctx)


class WeightedSpy(ScopedProbabilities):
    supports_sample_weight = True
    def fit(self, context):
        super().fit(context)
        self.weighted_class_mass_ = context.sample_weight.groupby(context.y.iloc[:, 0]).sum()
        return self


def weighted_candidate(**kwargs):
    return Candidate('weighted', WeightedSpy, pre_analysis=PreTrainingAnalysis({'weights': ClassWeightReporter()}),
        weights_from='weights', **kwargs)


def test_candidate_weights_exclude_validation_calibration_and_heldout_labels():
    from xdiyo_analytics.selection.core import _fit_candidate, _FreshModels
    data = classification_data()
    # Deliberately different proportions in the fitting and withheld populations.
    data.y['target'] = (data.metadata.case % 3 == 0).astype(int)
    folds = SplitPlan([outer(data)], len(data.X), np.arange(len(data.X)))
    candidate = weighted_candidate(validation=ValidationTail(.25), calibration=ProbabilityCalibrator(fraction=.25))
    first = _fit_candidate(data, folds, candidate, _FreshModels(), None).folds[0]
    ctx = first.model.estimator.context
    expected = ClassWeightPolicy().compute(ctx)
    pd.testing.assert_series_equal(ctx.sample_weight, expected.values)
    fit_rows = set(ctx.X.index)
    assert fit_rows.isdisjoint(first.validation_positions)
    assert fit_rows.isdisjoint(first.calibration_positions)
    assert fit_rows.isdisjoint(first.test_positions)
    changed = deepcopy(data)
    changed.y.iloc[first.test_positions] = 100
    second = _fit_candidate(changed, folds, candidate, _FreshModels(), None).folds[0]
    pd.testing.assert_series_equal(second.model.estimator.context.sample_weight, ctx.sample_weight)


def test_nested_weight_reports_recompute_on_each_inner_fit():
    data = classification_data()
    data.y['target'] = (data.metadata.case % 3 == 0).astype(int)
    outer_plan = SplitPlan([outer(data)], len(data.X), np.arange(len(data.X)))
    def inner(local):
        return SplitPlan([Fold(rows_for(local, range(6)), rows_for(local, range(6, 9)), rows_for(local, range(6, 9)))],
            len(local.X), np.arange(len(local.X)))
    result = ModelSelection([weighted_candidate()], metrics='mse').run_nested(data, outer_plan, inner)
    trial = result.selections[0].trials[0].training.folds[0]
    ctx = trial.model.context
    assert set(ctx.metadata.case) == set(range(6))
    pd.testing.assert_series_equal(ctx.sample_weight, ClassWeightPolicy().compute(ctx).values)


def test_direct_runner_weighting_and_refit_persist_predictions(tmp_path):
    data = classification_data()
    data.y['target'] = (data.metadata.case % 3 == 0).astype(int)
    factory = lambda: EstimatorAdapter(LogisticRegression(C=.4), ('predict', 'predict_proba'))
    policy = ClassWeightPolicy(mode='power', power=.5)
    plan = SplitPlan([outer(data)], len(data.X), np.arange(len(data.X)))
    result = TrainingRunner(factory, weighting=policy).run(data, plan)
    saved = load_model(save_model(result, tmp_path/'weighted-fold'))
    for key, frame in result.folds[0].predictions.items():
        pd.testing.assert_frame_equal(saved.predict(data, positions=plan.folds[0].test)[key], frame)
    refit = refit_model(data, factory, train_positions=rows_for(data, range(16)), weighting=policy)
    restored = load_model(save_model(refit, tmp_path/'weighted-refit'))
    for key, frame in refit.predict(data).items():
        pd.testing.assert_frame_equal(restored.predict(data)[key], frame)


def test_boosting_routes_original_label_weights_and_keeps_validation_unweighted(monkeypatch):
    from xgboost import XGBClassifier
    from xdiyo_analytics.ui.adapters import BoostingAdapter
    ctx = weight_context()
    ctx.sample_weight = ClassWeightPolicy(mode='custom', class_weights={2: 1., 7: 2., 12: 4.}).compute(ctx).values
    ctx.validation = replace(ctx, X=ctx.X.iloc[:2], y=ctx.y.iloc[:2], metadata=ctx.metadata.iloc[:2], sample_weight=None)
    captured = []
    original = XGBClassifier.fit
    def inspect(self, X, y, *, sample_weight=None, **kwargs):
        captured.append((np.asarray(y), np.asarray(sample_weight), kwargs.copy()))
        return original(self, X, y, sample_weight=sample_weight, **kwargs)
    monkeypatch.setattr(XGBClassifier, 'fit', inspect)
    model = XGBClassifier(n_estimators=2, max_depth=1, n_jobs=1, tree_method='hist', device='cpu')
    adapter = BoostingAdapter(model, ('predict', 'predict_proba'), fit_kwargs={'verbose': False})
    adapter.fit(ctx)
    np.testing.assert_array_equal(captured[0][0], [0, 0, 0, 1, 1, 2])
    np.testing.assert_allclose(captured[0][1], ctx.sample_weight)
    assert 'sample_weight_eval_set' not in captured[0][2]
    outputs = adapter.predict(ctx)
    assert outputs['predict_proba'].columns.get_level_values(1).tolist() == [2, 7, 12]
    with pytest.raises(ValueError, match='not both'):
        BoostingAdapter(model, fit_kwargs={'sample_weight': np.ones(6)}).fit(ctx)


def test_candidate_refit_recomputes_weights_and_restores_without_reweighting(tmp_path, monkeypatch):
    from xdiyo_analytics.experiments import FootballExperiment, PreparedExperiment, RefitPolicy
    data = classification_data()
    data.y['target'] = (data.metadata.case % 3 == 0).astype(int)
    factory = lambda: EstimatorAdapter(LogisticRegression(C=.4), ('predict', 'predict_proba'))
    candidate = Candidate('weighted logistic', factory, pre_analysis=PreTrainingAnalysis({'weights': ClassWeightReporter()}), weights_from='weights')
    plan = SplitPlan([outer(data)], len(data.X), np.arange(len(data.X)))
    experiment = FootballExperiment('class weighting recovery', output_dir=tmp_path)
    seen = []
    original = ClassWeightPolicy.compute
    def inspect(self, context):
        seen.append(set(context.metadata.case))
        return original(self, context)
    monkeypatch.setattr(ClassWeightPolicy, 'compute', inspect)
    first = experiment.run(PreparedExperiment(data, plan), model=candidate,
        refit_policy=RefitPolicy(rows_for(data, range(16))))
    assert seen == [set(range(12)), set(range(16))]
    seen.clear()
    def forbidden(*args, **kwargs):
        raise AssertionError('Recovered experiment must not fit')
    monkeypatch.setattr(LogisticRegression, 'fit', forbidden)
    cached = experiment.run(PreparedExperiment(data, plan), model=candidate,
        refit_policy=RefitPolicy(rows_for(data, range(16))))
    assert cached.reused and seen == []
    for key, frame in first.refit.predict(data).items():
        pd.testing.assert_frame_equal(cached.refit.predict(data)[key], frame)


def test_iterative_runner_passes_weights_to_backend_after_validation_split():
    from xdiyo_analytics.training import IterativeAdapter, TrainingControl
    data = classification_data()
    data.y['target'] = (data.metadata.case % 3 == 0).astype(int)
    factory = lambda: IterativeAdapter(lambda seed: PartialFitBackend(
        SGDClassifier(loss='log_loss', random_state=seed), loss='log_loss',
        prediction_methods=('predict', 'predict_proba')))
    plan = SplitPlan([outer(data)], len(data.X), np.arange(len(data.X)))
    result = TrainingRunner(factory, weighting=ClassWeightPolicy(), validation=ValidationTail(.25),
        control=TrainingControl(max_steps=2)).run(data, plan).folds[0]
    backend = result.model.backend_
    expected = ClassWeightPolicy().compute(backend.context_)
    np.testing.assert_allclose(backend.kwargs_['sample_weight'], expected.values)
    assert backend.validation_.sample_weight is None
    assert set(backend.context_.X.index).isdisjoint(backend.validation_.X.index)


def test_no_weighting_preserves_unweighted_estimator_behavior():
    data = classification_data()
    factory = lambda: EstimatorAdapter(LogisticRegression(C=.4), ('predict_proba',))
    plan = SplitPlan([outer(data)], len(data.X), np.arange(len(data.X)))
    result = TrainingRunner(factory).run(data, plan).folds[0]
    native = LogisticRegression(C=.4).fit(data.X.iloc[plan.folds[0].train], data.y.iloc[plan.folds[0].train, 0])
    np.testing.assert_allclose(result.predictions['predict_proba'], native.predict_proba(data.X.iloc[plan.folds[0].test]))


def test_ui_consumed_weight_report_and_changed_power_invalidates_fit(ui_recipe, monkeypatch):
    from xdiyo_analytics.ui import run_recipe
    from xdiyo_analytics.ui.recipe import node
    ui_recipe['model'] = node('sklearn.linear_model.LogisticRegression', C=.5)
    ui_recipe['fitted_reporters'] = {'weights': node('reporting.ClassWeightReporter', mode='power', power=.5)}
    ui_recipe['candidate']['weights_from'] = 'weights'
    first = run_recipe(ui_recipe)
    summary = first.training.folds[0].training_summary
    assert summary['weight_mean'] == pytest.approx(1.) and summary['class_balance']
    calls = []
    original = LogisticRegression.fit
    def inspect(self, X, y, sample_weight=None):
        calls.append(np.asarray(sample_weight))
        return original(self, X, y, sample_weight=sample_weight)
    monkeypatch.setattr(LogisticRegression, 'fit', inspect)
    reused = run_recipe(ui_recipe)
    assert reused.reused and calls == []
    ui_recipe['fitted_reporters']['weights']['params']['power'] = 1.
    changed = run_recipe(ui_recipe)
    assert not changed.reused and len(calls) == 1
    assert changed.record['run_id'] != first.record['run_id']
    assert changed.training.folds[0].training_summary['class_balance'] != summary['class_balance']


class WeightedTrace(TraceAdapter):
    supports_sample_weight = True


def test_two_weighted_process_folds_do_not_serialize_parent_observer(tmp_path):
    import os
    from functools import partial
    from model_selection_samples import plan
    from xdiyo_analytics.training import ExecutionPolicy
    data = classification_data()
    data.y['target'] = (data.metadata.case % 3 == 0).astype(int)
    observer = ParentObserver()
    result = TrainingRunner(partial(WeightedTrace, folder=tmp_path, barrier=True, delay=.2),
        weighting=ClassWeightPolicy(mode='power', power=.5), observer=observer,
        execution=ExecutionPolicy(2)).run(data, plan(data))
    assert len({fold.model.pid for fold in result.folds}) == 2
    assert os.getpid() not in {fold.model.pid for fold in result.folds}
    assert observer.calls and all(pid == os.getpid() for pid, _, _ in observer.calls)
    for fold in result.folds:
        ctx = fold.model.context
        pd.testing.assert_series_equal(ctx.sample_weight, ClassWeightPolicy(mode='power', power=.5).compute(ctx).values)


@pytest.mark.parametrize('options', [{'is_unbalance': True}, {'scale_pos_weight': 2.}, {'class_weight': 'balanced'}])
def test_lightgbm_native_weighting_conflicts_are_rejected(options):
    from lightgbm import LGBMClassifier
    from xdiyo_analytics.ui.adapters import BoostingAdapter
    ctx = weight_context((0, 0, 0, 0, 1, 1))
    ctx.sample_weight = ClassWeightPolicy().compute(ctx).values
    with pytest.raises(ValueError, match='native class_weight'):
        BoostingAdapter(LGBMClassifier(n_estimators=2, verbosity=-1, **options)).fit(ctx)
