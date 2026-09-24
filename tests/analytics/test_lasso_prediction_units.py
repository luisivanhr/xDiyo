"""Separate input scaling, target inversion and a suppressed Lasso intercept."""

from dataclasses import replace
from functools import partial

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_selection_samples import sample
from xdiyo_analytics.experiments import FootballExperiment, PreparedExperiment
from xdiyo_analytics.selection import Candidate, ModelSelection
from xdiyo_analytics.splits import Fold, SplitPlan
from xdiyo_analytics.training import EstimatorAdapter, TrainingRunner
from xdiyo_analytics.ui.recipe import ModelFactory, catalog_for_ui, node


@pytest.mark.parametrize('fit_intercept', [False, True])
def test_lasso_feature_scaling_matches_closed_form_without_any_target_inverse(fit_intercept):
    data = sample()
    x = np.r_[np.arange(98., 103.), np.arange(103., 114.)]
    data.X = pd.DataFrame({'signal': x, 'constant_home_flag': 1.})
    data.y = pd.DataFrame({'target': 1000. + 7. * (x - 100.)})
    rows = np.arange(len(x))
    plan = SplitPlan([Fold(rows[:5], rows[5:7], rows[5:7])], len(x), rows)
    recipe = dict(
        model=node('sklearn.linear_model.Lasso', alpha=.1, fit_intercept=fit_intercept,
                   tol=1e-12, max_iter=10000),
        preprocessors=[node('sklearn.impute.SimpleImputer', strategy='median'),
                       node('sklearn.preprocessing.StandardScaler')],
    )
    fitted = TrainingRunner(ModelFactory(recipe, catalog_for_ui())).run(data, plan).folds[0]
    scaler = fitted.model.estimator.named_steps['prepare_1']
    np.testing.assert_allclose(scaler.mean_, [100., 1.])
    np.testing.assert_allclose(scaler.scale_, [np.sqrt(2), 1.])
    assert scaler.n_samples_seen_ == 5
    assert not np.isclose(scaler.mean_[0], data.X.signal.mean())
    # One standardized nonconstant feature: soft-threshold covariance by alpha.
    coefficient = 7 * np.sqrt(2) - .1
    expected = coefficient * (x[5:7] - 100) / np.sqrt(2)
    if fit_intercept:
        expected += 1000.
    np.testing.assert_allclose(fitted.predictions['predict'].target, expected, atol=1e-9)
    manual = Pipeline([('impute', SimpleImputer(strategy='median')),
                       ('scale', StandardScaler()),
                       ('model', Lasso(alpha=.1, fit_intercept=fit_intercept,
                                       tol=1e-12, max_iter=10000))])
    manual.fit(data.X.iloc[:5], data.y.iloc[:5, 0])
    np.testing.assert_allclose(fitted.predictions['predict'].target,
                               manual.predict(data.X.iloc[5:7]), atol=1e-9)
    np.testing.assert_array_equal(fitted.y_true.target, [1021., 1028.])
    assert fitted.model.estimator.named_steps['model'].intercept_ == (1000. if fit_intercept else 0.)


def target_adapter():
    return EstimatorAdapter(Pipeline([
        ('input_scale', StandardScaler()),
        ('target_model', TransformedTargetRegressor(
            regressor=LinearRegression(), transformer=StandardScaler())),
    ]))


def inner_plan(data):
    rows = np.arange(len(data.X))
    return SplitPlan([Fold(rows[:4], rows[4:6], rows[4:6])], len(rows), rows)


def test_nested_selection_and_data_only_recovery_preserve_one_target_inverse(tmp_path):
    data = sample()
    x = np.arange(len(data.X), dtype=float)
    data.X = pd.DataFrame({'signal': 100. + x})
    data.y = pd.DataFrame({'target': 1000. + 7. * x})
    rows = np.arange(len(x))
    outer = SplitPlan([Fold(rows[:8], rows[8:10], rows[8:10]),
                       Fold(rows[:12], rows[12:], rows[12:])], len(rows), rows)
    candidates = [Candidate('Explicit standardized target', target_adapter,
                            config={'target_transform': 'StandardScaler'})]
    experiment = FootballExperiment('Synthetic target inversion audit', output_dir=tmp_path)
    result = experiment.run(PreparedExperiment(data, outer),
                            model_selection=ModelSelection(candidates, metrics='mse'),
                            inner_plan_factory=inner_plan)
    for fold in result.training.folds:
        model = fold.model.estimator
        x_scaler = model.named_steps['input_scale']
        wrapper = model.named_steps['target_model']
        train, test = fold.fit_positions, fold.test_positions
        np.testing.assert_allclose(x_scaler.mean_, data.X.iloc[train].mean())
        np.testing.assert_allclose(wrapper.transformer_.mean_, data.y.iloc[train].mean())
        transformed = x_scaler.transform(data.X.iloc[test])
        internal = wrapper.regressor_.predict(transformed)
        manual_inverse = internal * data.y.iloc[train, 0].std(ddof=0) + data.y.iloc[train, 0].mean()
        np.testing.assert_allclose(fold.predictions['predict'].target, manual_inverse, atol=1e-9)
        np.testing.assert_allclose(fold.predictions['predict'].target, data.y.iloc[test, 0], atol=1e-9)
        assert not np.allclose(internal, fold.predictions['predict'].target)
    loaded = experiment.load(result.record['run_id'])
    for original, restored in zip(result.training.folds, loaded.training.folds):
        assert restored.model is None
        pd.testing.assert_frame_equal(original.predictions['predict'], restored.predictions['predict'])
        pd.testing.assert_frame_equal(original.y_true, restored.y_true)
