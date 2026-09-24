"""Validation memberships and independent SGD arithmetic on synthetic rows."""
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import SGDClassifier, SGDRegressor, Ridge
from sklearn.preprocessing import StandardScaler
from xdiyo_analytics.analysis import PreTrainingAnalysis
from xdiyo_analytics.reporting import TopKCorrelationSelector
from xdiyo_analytics.splits import Fold, SplitPlan
from xdiyo_analytics.training import (EstimatorAdapter, TrainingRunner, TrainingControl, ValidationTail,
    IterativeAdapter, PartialFitBackend, ReduceOnPlateau, fit_predict, split_training_rows)
from split_samples import dataset, rows_for, cases, unchanged
from training_samples import MeanAdapter, sample, plan


class ControlledMean(MeanAdapter):
    def fit_controlled(self, context, control):
        self.control = control
        self.fit(context)


@pytest.mark.parametrize('layout', ['match', 'team_match'])
@pytest.mark.parametrize('fraction,valid_cases', [(.2, {4}), (.34, {2, 3, 4})])
def test_temporal_validation_uses_distinct_tied_batches_and_whole_matches(layout, fraction, valid_cases):
    data = dataset(layout=layout, shuffle=True, specs=[{'day': d} for d in [0, 0, 1, 1, 2, 3]])
    train = rows_for(data, range(5))[::-1]; before = deepcopy(data)
    fit, valid = split_training_rows(data, train, ValidationTail(fraction))
    assert cases(data, valid) == valid_cases and cases(data, fit) == set(range(5)) - valid_cases
    assert valid.tolist() == [int(i) for i in train if data.metadata.iloc[i]['case'] in valid_cases]
    assert fit.tolist() == [int(i) for i in train if data.metadata.iloc[i]['case'] not in valid_cases]
    unchanged(data, before)


@pytest.mark.parametrize('validation', [[4], [1, 1], [True], [1.5], [0, 1, 2, 3]])
def test_invalid_validation_memberships_rejected(validation):
    with pytest.raises(ValueError): split_training_rows(dataset(6), [0, 1, 2, 3], validation)


def test_partial_match_validation_is_rejected():
    with pytest.raises(ValueError, match='whole matches'):
        split_training_rows(dataset(6, layout='team_match'), np.arange(8), [7])


@pytest.mark.parametrize('fraction', [0, 1, -.1, True, np.nan])
def test_validation_tail_rejects_invalid_fraction(fraction):
    with pytest.raises(ValueError): split_training_rows(dataset(6), [0, 1, 2, 3], ValidationTail(fraction))


@pytest.mark.parametrize('problem', ['one_batch', 'missing_time'])
def test_validation_tail_cannot_reserve_every_batch_or_use_missing_times(problem):
    data = dataset(4)
    if problem == 'one_batch': data.metadata['kickoff_at'] = data.metadata.kickoff_at.iloc[0]
    else: data.metadata.loc[1, 'kickoff_at'] = pd.NaT
    with pytest.raises(ValueError): split_training_rows(data, [0, 1, 2], ValidationTail())


@pytest.mark.parametrize('layout', ['match', 'team_match'])
def test_runner_retains_development_scope_but_fits_and_selects_on_fit_rows_only(layout):
    data = sample(layout=layout, shuffle=True); folds = plan(data); before = deepcopy(folds)
    runner = TrainingRunner(ControlledMean, validation=ValidationTail(.25))
    selection_plan = runner.selection_plan(data, folds)
    report = PreTrainingAnalysis({'choice': TopKCorrelationSelector(type='per_fold', partition='train', k=1)}).run(data, split_plan=selection_plan)
    result = runner.run(data, folds, analysis_report=report, features_from='choice')
    for f, original, scoped, study in zip(result.folds, folds.folds, selection_plan.folds, report.studies):
        assert f.train_positions.tolist() == original.train.tolist()
        assert f.fit_positions.tolist() == scoped.train.tolist() == list(f.model.fit_context.X.index)
        assert set(f.validation_positions) == set(original.train) - set(scoped.train)
        assert list(f.model.fit_context.validation.X.index) == f.validation_positions.tolist()
        assert set(study.row_positions) == set(f.fit_positions)
        assert f.test_positions.tolist() == original.test.tolist() and f.score_positions.tolist() == original.score.tolist()
    for a, b in zip(folds.folds, before.folds):
        assert a.train.tolist() == b.train.tolist() and a.metadata == b.metadata
    leaking = PreTrainingAnalysis({'choice': TopKCorrelationSelector(type='per_fold', partition='train', k=1)}).run(data, split_plan=folds)
    with pytest.raises(ValueError, match='fitting training rows'):
        runner.run(data, folds, analysis_report=leaking, features_from='choice')


def test_fold_mapping_preserves_selected_original_fold_id():
    data = sample(); folds = plan(data)
    result = TrainingRunner(ControlledMean, validation={1: [6]}).run(data, folds, fold_ids=[1])
    assert result.folds[0].fold_id == 1 and result.folds[0].validation_positions.tolist() == [6]
    assert result.folds[0].fit_positions.tolist() == list(range(6))


@pytest.mark.parametrize('option', ['control', 'validation'])
def test_plain_estimator_rejects_unsupported_controls_without_fitting(option):
    data = dataset(6); estimator = Ridge()
    kwargs = {'control': TrainingControl(max_steps=1)} if option == 'control' else {'validation': [2]}
    with pytest.raises(TypeError, match='does not support'):
        fit_predict(data, Fold(np.arange(3), np.array([4]), np.array([4])), lambda: EstimatorAdapter(estimator), **kwargs)
    assert not hasattr(estimator, 'coef_')


def test_actual_sgd_updates_and_losses_match_hand_computation_with_fit_only_scaling():
    data = dataset(8); original = deepcopy(data)
    def factory():
        return IterativeAdapter(lambda seed: PartialFitBackend(SGDRegressor(
            loss='squared_error', penalty=None, learning_rate='constant', eta0=.05,
            shuffle=False, random_state=seed), preprocessor=StandardScaler()))
    fitted = fit_predict(data, Fold(np.arange(4), np.array([5, 6]), np.array([6])), factory,
        validation=[3], control=TrainingControl(max_steps=3, restore_best=False))
    mean, scale = 1., np.sqrt(2/3)
    z = (np.arange(3.) - mean) / scale; truth = np.array([1., 2., 3.])
    w = b = 0.; expected = []
    for _ in range(3):
        for x, y in zip(z, truth):
            error = y - (w*x+b)
            w += .05*error*x; b += .05*error
        expected += [np.mean((truth - (w*z+b))**2), (4 - (w*(3-mean)/scale+b))**2]
    backend = fitted.model.backend_
    np.testing.assert_allclose(backend.estimator.coef_, [w], atol=1e-14)
    np.testing.assert_allclose(backend.estimator.intercept_, [b], atol=1e-14)
    np.testing.assert_allclose(backend.preprocessor.mean_, [mean])
    np.testing.assert_allclose(backend.preprocessor.scale_, [scale])
    np.testing.assert_allclose(fitted.training_history.value, expected, atol=1e-14)
    np.testing.assert_allclose(fitted.predictions['predict'].outcome, w*(np.array([5., 6.])-mean)/scale+b)
    assert fitted.training_summary['monitor'] == 'validation_loss'
    unchanged(data, original)


def test_actual_logistic_sgd_matches_hand_gradients_and_fitting_class_vocabulary():
    data = dataset(8); data.X['form'] = np.array([-.5, .5, 1., 1.5, 2., 3., 4., 5.])
    data.y['outcome'] = [0, 1, 0, 1, 2, 2, 2, 2]
    def factory():
        return IterativeAdapter(lambda seed: PartialFitBackend(SGDClassifier(
            loss='log_loss', penalty=None, learning_rate='constant', eta0=.1, shuffle=False, random_state=seed),
            loss='log_loss', prediction_methods=('predict', 'predict_proba')))
    fitted = fit_predict(data, Fold(np.arange(4), np.array([4, 5]), np.array([5])), factory,
        validation=[3], control=TrainingControl(max_steps=2, restore_best=False))
    w = b = 0.; expected = []
    def probability(x): return 1/(1+np.exp(-(w*x+b)))
    xfit, yfit = np.array([-.5, .5, 1.]), np.array([0, 1, 0])
    for _ in range(2):
        for x, y in zip(xfit, yfit):
            error = y-probability(x); w += .1*error*x; b += .1*error
        p = probability(xfit)
        expected += [-np.mean(yfit*np.log(p)+(1-yfit)*np.log(1-p)), -np.log(probability(1.5))]
    model = fitted.model.backend_.estimator
    np.testing.assert_allclose(model.coef_, [[w]], atol=1e-14)
    np.testing.assert_allclose(model.intercept_, [b], atol=1e-14)
    np.testing.assert_allclose(fitted.training_history.value, expected, atol=1e-14)
    assert model.classes_.tolist() == [0, 1]
    probabilities = fitted.predictions['predict_proba']
    assert probabilities.columns.tolist() == [('outcome', 0), ('outcome', 1)]
    np.testing.assert_allclose(probabilities[('outcome', 1)], probability(np.array([2., 3.])))


def test_explicit_classes_can_cover_a_class_absent_from_fit_rows():
    data = dataset(7); data.y['outcome'] = [0, 0, 0, 1, 1, 1, 1]
    def factory(explicit):
        return lambda: IterativeAdapter(lambda seed: PartialFitBackend(SGDClassifier(
            loss='log_loss', random_state=seed), loss='log_loss',
            partial_fit_kwargs={'classes': [0, 1]} if explicit else None))
    fold = Fold(np.arange(4), np.array([5]), np.array([5]))
    with pytest.raises(ValueError):
        fit_predict(data, fold, factory(False), validation=[3], control=TrainingControl(max_steps=1))
    fitted = fit_predict(data, fold, factory(True), validation=[3], control=TrainingControl(max_steps=1))
    assert fitted.model.backend_.estimator.classes_.tolist() == [0, 1]


def test_builtin_scheduler_does_not_override_an_estimators_native_schedule():
    data = dataset(6)
    factory = lambda: IterativeAdapter(lambda seed: PartialFitBackend(SGDRegressor(learning_rate='invscaling')))
    with pytest.raises(ValueError, match='constant'):
        fit_predict(data, Fold(np.arange(3), np.array([4]), np.array([4])), factory,
                    control=TrainingControl(max_steps=1, scheduler=ReduceOnPlateau()))


def test_partial_fit_backend_rejects_multiple_targets():
    data = sample()
    factory = lambda: IterativeAdapter(lambda seed: PartialFitBackend(SGDRegressor()))
    with pytest.raises(ValueError, match='one selected target'):
        TrainingRunner(factory, control=TrainingControl(max_steps=1)).run(data, plan(data), fold_ids=[0])
