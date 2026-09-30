"""Independent numeric calibration and training-population boundary checks."""
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from xdiyo_analytics.training import (ProbabilityCalibrator, CalibratedAdapter, TrainingRunner,
    ValidationTail, PredictionContext, save_model, load_model, refit_model)
from xdiyo_analytics.analysis import PreTrainingAnalysis
from xdiyo_analytics.reporting import TopKCorrelationSelector
from xdiyo_analytics.selection import Candidate, ModelSelection
from xdiyo_analytics.splits import Fold, SplitPlan
from model_selection_samples import sample, outer, plan, development
from split_samples import rows_for
from test_ui_workflow import ui_recipe


def probabilities(values, target='corners', classes=(0, 1), index=None):
    return pd.DataFrame(values, index=index, columns=pd.MultiIndex.from_product([[target], classes], names=['target', 'class']))


def test_temperature_reduces_overconfident_loss_and_preserves_identity():
    index = pd.Index(np.arange(100)[::-1], name='match')
    p = probabilities(np.tile([.01, .99], (100, 1)), index=index)
    y = pd.DataFrame({'corners': [1]*80+[0]*20}, index=index)
    before = p.copy(deep=True)
    calibrator = ProbabilityCalibrator().fit(p, y)
    q = calibrator.transform(p)
    assert calibrator.models_['corners'] > 1
    np.testing.assert_allclose(q[('corners', 1)], .8, atol=1e-5)
    idx = np.arange(100), y.corners.to_numpy()
    assert -np.log(q.to_numpy()[idx]).mean() < -np.log(p.to_numpy()[idx]).mean()
    pd.testing.assert_frame_equal(p, before)
    pd.testing.assert_index_equal(q.index, p.index)
    pd.testing.assert_index_equal(q.columns, p.columns)


@pytest.mark.parametrize('method', ['temperature', 'sigmoid', 'isotonic'])
def test_multitarget_calibration_has_independent_simplexes_and_original_labels(method):
    rng = np.random.default_rng(24)
    index = pd.Index([11, 4, 90, 3, 42, 19]*10)
    p = pd.concat([probabilities(rng.dirichlet([2, 3, 4], 60), classes=(3, 7, 12), index=index),
                   probabilities(rng.dirichlet([2, 2], 60), target='outcome', classes=('away', 'home'), index=index)], axis=1)
    y = pd.DataFrame({'corners': [3, 7, 12]*20, 'outcome': ['home', 'away']*30}, index=index)
    calibrator = ProbabilityCalibrator(method=method).fit(p, y)
    q = calibrator.transform(p)
    assert calibrator.n_samples_ == 60
    pd.testing.assert_index_equal(q.columns, p.columns)
    pd.testing.assert_index_equal(q.index, p.index)
    for target in ['corners', 'outcome']:
        actual = q.xs(target, axis=1, level=0)
        assert np.isfinite(actual.to_numpy()).all() and actual.ge(0).all().all()
        np.testing.assert_allclose(actual.sum(axis=1), 1, atol=1e-12)


@pytest.mark.parametrize('method', ['sigmoid', 'isotonic'])
def test_binary_calibration_matches_independent_reference(method):
    rng = np.random.default_rng(881)
    logits = np.linspace(-3, 3, 180)
    raw = expit(logits)
    truth = (rng.random(180) < expit(.6*logits-.3)).astype(int)
    p = probabilities(np.column_stack([1-raw, raw]))
    y = pd.DataFrame({'corners': truth})
    calibrated = ProbabilityCalibrator(method=method).fit(p, y).transform(p)
    columns = []
    for i in range(2):
        x = p.iloc[:, i].to_numpy()
        binary = (truth == i).astype(int)
        if method == 'isotonic':
            expected = IsotonicRegression(out_of_bounds='clip').fit(x, binary).predict(x)
        else:
            design = (np.log(x)-np.log1p(-x))[:, None]
            expected = LogisticRegression(C=np.inf, solver='lbfgs', tol=1e-11, max_iter=500).fit(design, binary).predict_proba(design)[:, 1]
        columns.append(expected)
    expected = np.column_stack(columns)
    expected /= expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(calibrated, expected, atol=2e-4)


@pytest.mark.parametrize('problem', ['sum', 'negative', 'nan', 'columns', 'order', 'unknown_label', 'missing_label', 'empty'])
def test_calibration_rejects_malformed_or_misaligned_inputs(problem):
    p = probabilities([[.2, .8], [.7, .3]], index=[7, 3])
    y = pd.DataFrame({'corners': [1, 0]}, index=[7, 3])
    if problem == 'sum': p.iloc[0, 0] = .3
    if problem == 'negative': p.iloc[0] = [-.1, 1.1]
    if problem == 'nan': p.iloc[0, 0] = np.nan
    if problem == 'columns': p.columns = ['a', 'b']
    if problem == 'order': y = y.iloc[::-1]
    if problem == 'unknown_label': y.iloc[0, 0] = 8
    if problem == 'missing_label': y.iloc[0, 0] = np.nan
    if problem == 'empty': p, y = p.iloc[:0], y.iloc[:0]
    with pytest.raises(ValueError): ProbabilityCalibrator().fit(p, y)


class ScopedProbabilities:
    """Deterministic miscalibrated probabilities; no access to prediction targets."""
    def fit(self, context):
        self.context = deepcopy(context)
        self.targets = list(context.y)
        return self
    def fit_controlled(self, context, control):
        return self.fit(context)
    def predict(self, context):
        assert not hasattr(context, 'y')
        signal = context.X.iloc[:, 0].to_numpy()
        p = expit(2 * ((signal % 4)-1.5))
        proba = pd.concat([probabilities(np.column_stack([1-p, p]), target=target, index=context.X.index)
                           for target in self.targets], axis=1)
        return {'predict': pd.DataFrame({target: (p > .5).astype(int) for target in self.targets}, index=context.X.index),
                'predict_proba': proba}


def classification_data(layout='match'):
    data = sample(layout=layout, shuffle=True)
    data.y['target'] = (data.metadata.case % 2).astype(int)
    return data


@pytest.mark.parametrize('layout', ['match', 'team_match'])
def test_fit_validation_calibration_selection_and_test_scopes_disjoint(layout, monkeypatch):
    data = classification_data(layout)
    seen = []
    original = TopKCorrelationSelector.select
    def inspect(self, context):
        seen.append(set(context.metadata.case))
        return original(self, context)
    monkeypatch.setattr(TopKCorrelationSelector, 'select', inspect)
    candidate = Candidate('Calibrated', ScopedProbabilities,
        calibration=ProbabilityCalibrator(fraction=.25), validation=ValidationTail(.25),
        pre_analysis=PreTrainingAnalysis({'pick': TopKCorrelationSelector(type='per_fold', partition='train', k=1)}),
        features_from='pick')
    from xdiyo_analytics.selection.core import _fit_candidate, _FreshModels
    folds = SplitPlan([outer(data)], len(data.X), np.arange(len(data.X)))
    result = _fit_candidate(data, folds, candidate, _FreshModels(), None)
    fold = result.folds[0]
    populations = [set(fold.fit_positions), set(fold.validation_positions), set(fold.calibration_positions), set(fold.test_positions)]
    for i, a in enumerate(populations):
        assert a
        for b in populations[i+1:]: assert not a & b
    assert set.union(*populations[:3]) == set(fold.train_positions)
    assert seen == [set(data.metadata.iloc[fold.fit_positions].case)]
    assert set(data.metadata.iloc[fold.calibration_positions].case) == {9, 10, 11}
    assert set(fold.model.estimator.context.metadata.case) == seen[0]
    assert set(fold.model.estimator.context.validation.metadata.case) == set(data.metadata.iloc[fold.validation_positions].case)
    assert fold.model.calibrator.n_samples_ == len(fold.calibration_positions)
    assert {'predict', 'predict_proba', 'predict_proba_raw'} == set(fold.predictions)


def test_test_target_changes_do_not_change_calibration_or_prediction():
    data = classification_data()
    folds = SplitPlan([outer(data)], len(data.X), np.arange(len(data.X)))
    runner = TrainingRunner(ScopedProbabilities, calibration=ProbabilityCalibrator(fraction=.25))
    first = runner.run(data, folds)
    altered = deepcopy(data)
    altered.y.iloc[folds.folds[0].test, 0] = 99
    second = runner.run(altered, folds)
    assert first.folds[0].model.calibrator.models_ == second.folds[0].model.calibrator.models_
    for method in first.folds[0].predictions:
        pd.testing.assert_frame_equal(first.folds[0].predictions[method], second.folds[0].predictions[method])


def test_disabled_and_update_predict_false_preserve_original_point_outputs():
    data = classification_data()
    folds = SplitPlan([outer(data)], len(data.X), np.arange(len(data.X)))
    plain = TrainingRunner(ScopedProbabilities).run(data, folds).folds[0]
    explicit = TrainingRunner(ScopedProbabilities, calibration=None).run(data, folds).folds[0]
    assert set(plain.predictions) == {'predict', 'predict_proba'}
    pd.testing.assert_frame_equal(plain.predictions['predict_proba'], explicit.predictions['predict_proba'])
    calibrated = TrainingRunner(ScopedProbabilities, calibration=ProbabilityCalibrator(update_predict=False)).run(data, folds).folds[0]
    pd.testing.assert_frame_equal(calibrated.predictions['predict'], plain.predictions['predict'])
    pd.testing.assert_frame_equal(calibrated.predictions['predict_proba_raw'], plain.predictions['predict_proba'])


def test_calibrated_fitted_and_refit_model_save_load_prediction_equivalence(tmp_path):
    data = classification_data()
    policy = ProbabilityCalibrator(fraction=.25)
    fitted = refit_model(data, ScopedProbabilities, train_positions=rows_for(data, range(12)), calibration=policy)
    expected = fitted.predict(data)
    path = save_model(fitted, tmp_path/'refit')
    restored = load_model(path)
    assert isinstance(restored.model, CalibratedAdapter)
    np.testing.assert_array_equal(restored.calibration_positions, fitted.calibration_positions)
    np.testing.assert_array_equal(restored.model.calibration_positions_, fitted.calibration_positions)
    for key, frame in expected.items(): pd.testing.assert_frame_equal(restored.predict(data)[key], frame)
    folds = SplitPlan([outer(data)], len(data.X), np.arange(len(data.X)))
    training = TrainingRunner(ScopedProbabilities, calibration=policy).run(data, folds)
    saved = load_model(save_model(training, tmp_path/'fold'))
    for method, frame in training.folds[0].predictions.items():
        pd.testing.assert_frame_equal(saved.predict(data, positions=folds.folds[0].test)[method], frame)


def test_nested_calibration_positions_are_original_dataset_positions():
    data = classification_data()
    outer_folds = SplitPlan([Fold(rows_for(data, range(12)), rows_for(data, range(12, 16)), rows_for(data, range(12, 16)))], len(data.X), np.arange(len(data.X)))
    def inner(local):
        return SplitPlan([Fold(rows_for(local, range(8)), rows_for(local, range(8, 12)), rows_for(local, range(8, 12)))], len(local.X), np.arange(len(local.X)))
    search = ModelSelection([Candidate('Calibrated', ScopedProbabilities, calibration=ProbabilityCalibrator(fraction=.25))], metrics='mse')
    result = search.run_nested(data, outer_folds, inner)
    trial_fold = result.selections[0].trials[0].training.folds[0]
    expected = set(rows_for(data, [6, 7]))
    assert set(trial_fold.calibration_positions) == expected
    assert set(trial_fold.model.calibration_positions_) == expected


def test_calibration_remains_inside_refit_population_and_refits_selector(tmp_path, monkeypatch):
    from xdiyo_analytics.experiments import FootballExperiment, PreparedExperiment, RefitPolicy
    data = classification_data()
    selected_rows = []
    original = TopKCorrelationSelector.select
    def inspect(self, context):
        selected_rows.append(set(context.metadata.case))
        return original(self, context)
    monkeypatch.setattr(TopKCorrelationSelector, 'select', inspect)
    candidate = Candidate('Calibrated', ScopedProbabilities,
        calibration=ProbabilityCalibrator(fraction=.25),
        pre_analysis=PreTrainingAnalysis({'pick': TopKCorrelationSelector(type='per_fold', partition='train', k=1)}), features_from='pick')
    folds = SplitPlan([outer(data)], len(data.X), np.arange(len(data.X)))
    experiment = FootballExperiment('Calibrated refit', output_dir=tmp_path)
    result = experiment.run(PreparedExperiment(data, folds), model=candidate,
        refit_policy=RefitPolicy(rows_for(data, range(16)), validation=ValidationTail(.25)))
    fit = result.refit
    assert set(data.metadata.iloc[fit.calibration_positions].case) == {12, 13, 14, 15}
    assert set(fit.fit_positions).isdisjoint(fit.calibration_positions)
    assert set(fit.validation_positions).isdisjoint(fit.calibration_positions)
    assert selected_rows == [set(range(9)), set(range(9))]
    restored = experiment.load(result.record['run_id'])
    for method, frame in fit.predict(data).items():
        pd.testing.assert_frame_equal(restored.refit.predict(data)[method], frame)


@pytest.mark.parametrize('method', ['temperature', 'sigmoid', 'isotonic'])
def test_ui_optional_calibration_requests_probabilities_and_survives_reuse(ui_recipe, method):
    from xdiyo_analytics.ui import run_recipe
    from xdiyo_analytics.ui.recipe import node
    ui_recipe['model'] = node('sklearn.linear_model.LogisticRegression', C=.2)
    ui_recipe['labels']['corners'] = node('labels.Above', source=ui_recipe['labels']['corners'], threshold=7)
    ui_recipe['candidate']['calibration'] = node('training.ProbabilityCalibrator', method=method, fraction=.25)
    assert ui_recipe['prediction_methods'] == ['predict']
    first = run_recipe(ui_recipe)
    fold = first.training.folds[0]
    assert isinstance(fold.model, CalibratedAdapter)
    assert {'predict_proba_raw', 'predict_proba'} <= set(fold.predictions)
    assert len(fold.calibration_positions) > 0
    second = run_recipe(ui_recipe)
    assert second.reused and isinstance(second.training.folds[0].model, CalibratedAdapter)
    for key in fold.predictions:
        pd.testing.assert_frame_equal(second.training.folds[0].predictions[key], fold.predictions[key])
