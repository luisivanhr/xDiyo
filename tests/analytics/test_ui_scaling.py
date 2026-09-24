"""Fit-only preprocessing and original-unit predictions through UI model factories."""

from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from model_selection_samples import sample
from xdiyo_analytics.splits import Fold, SplitPlan
from xdiyo_analytics.training import TrainingRunner, load_model, save_model
from xdiyo_analytics.ui.catalog import Catalog
from xdiyo_analytics.ui.recipe import ModelFactory, node


class AuditedXScaler(StandardScaler):
    """Retain numeric fit/transform inputs; X inversion must never be requested."""

    def fit(self, X, y=None, **kwargs):
        self.fit_values_ = np.asarray(X, dtype=float).copy()
        self.transform_values_ = []
        return super().fit(X, y, **kwargs)

    def transform(self, X, copy=None):
        self.transform_values_.append(np.asarray(X, dtype=float).copy())
        return super().transform(X, copy=copy)

    def inverse_transform(self, X, copy=None):
        raise AssertionError('Predicting must not inverse-transform X.')


def scaling_data():
    data = sample()
    # Deliberately displaced validation/test values make leakage observable.
    values = np.r_[np.arange(8.), [500., 700.], [1000., 2000., 3000., 4000., 5000., 6000.]]
    data.X = pd.DataFrame({'signal': values}, index=data.X.index)
    data.y = pd.DataFrame({'target': 1000. + 7. * values}, index=data.X.index)
    fold = Fold(np.arange(10), np.arange(10, 16), np.arange(10, 16))
    return data, SplitPlan([fold], len(data.X), np.arange(len(data.X)))


def scaling_catalog():
    catalog = Catalog()
    for key, constructor, category in (
        ('test.AuditedXScaler', AuditedXScaler, 'preprocessor'),
        ('sklearn.preprocessing.StandardScaler', StandardScaler, 'preprocessor'),
        ('sklearn.linear_model.LinearRegression', LinearRegression, 'model'),
        ('sklearn.compose.TransformedTargetRegressor', TransformedTargetRegressor, 'model'),
    ):
        catalog.register(key, constructor, category=category)
    return catalog


def test_ui_estimator_pipeline_fits_x_only_on_training_and_reuses_saved_state(tmp_path):
    data, plan = scaling_data()
    recipe = dict(model=node('sklearn.linear_model.LinearRegression'),
                  preprocessors=[node('test.AuditedXScaler')])
    training = TrainingRunner(ModelFactory(recipe, scaling_catalog())).run(data, plan)
    fold = training.folds[0]
    scaler = fold.model.estimator[0]
    expected = data.X.iloc[:10].to_numpy()
    np.testing.assert_array_equal(scaler.fit_values_, expected)
    np.testing.assert_allclose(scaler.mean_, expected.mean(axis=0))
    assert scaler.n_samples_seen_ == 10
    assert not np.allclose(scaler.mean_, data.X.mean(axis=0))
    np.testing.assert_array_equal(scaler.transform_values_[-1], data.X.iloc[10:].to_numpy())
    np.testing.assert_allclose(fold.predictions['predict'].target, data.y.iloc[10:].target, atol=1e-8)
    save_model(training, tmp_path / 'scaled-model')
    restored = load_model(tmp_path / 'scaled-model')
    unseen = replace(data, X=data.X.iloc[10:].copy(), y=data.y.iloc[10:].copy(), metadata=data.metadata.iloc[10:].copy())
    mean_before = restored.model.estimator[0].mean_.copy()
    predictions = restored.predict(unseen)['predict']
    np.testing.assert_allclose(predictions, fold.predictions['predict'])
    np.testing.assert_array_equal(restored.model.estimator[0].mean_, mean_before)
    assert restored.model.estimator[0].n_samples_seen_ == 10


@pytest.mark.parametrize('family', ['lightgbm', 'xgboost'])
def test_native_validation_and_test_use_fit_only_x_transform(family, monkeypatch, tmp_path):
    native = pytest.importorskip(family)
    model_class = native.LGBMRegressor if family == 'lightgbm' else native.XGBRegressor
    original_fit = model_class.fit
    seen = {}

    def recording_fit(self, X, y, **kwargs):
        seen['fit_X'] = np.asarray(X, dtype=float).copy()
        seen['fit_y'] = np.asarray(y, dtype=float).copy()
        seen['validation_X'] = np.asarray(kwargs['eval_set'][0][0], dtype=float).copy()
        seen['validation_y'] = np.asarray(kwargs['eval_set'][0][1], dtype=float).copy()
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(model_class, 'fit', recording_fit)
    catalog = scaling_catalog()
    component = family + '.' + model_class.__name__
    catalog.register(component, model_class, category='model')
    params = dict(n_estimators=2, max_depth=2, n_jobs=1)
    params.update(dict(verbosity=-1, min_child_samples=1) if family == 'lightgbm'
                  else dict(verbosity=0, tree_method='hist'))
    recipe = dict(model=node(component, **params), preprocessors=[node('test.AuditedXScaler')])
    data, plan = scaling_data()
    training = TrainingRunner(ModelFactory(recipe, catalog), validation=[8, 9]).run(data, plan)
    fold = training.folds[0]
    scaler = fold.model.preprocessing_[0]
    mean = np.arange(8.).mean()
    scale = np.arange(8.).std()
    assert scaler.n_samples_seen_ == 8
    np.testing.assert_allclose(scaler.mean_, [mean])
    np.testing.assert_allclose(scaler.scale_, [scale])
    np.testing.assert_allclose(seen['fit_X'][:, 0], (np.arange(8.) - mean) / scale)
    np.testing.assert_allclose(seen['validation_X'][:, 0], (np.array([500., 700.]) - mean) / scale)
    np.testing.assert_array_equal(seen['fit_y'], data.y.iloc[:8, 0])
    np.testing.assert_array_equal(seen['validation_y'], data.y.iloc[8:10, 0])
    np.testing.assert_array_equal(scaler.transform_values_[-1], data.X.iloc[10:].to_numpy())
    manual = fold.model.native_model_.predict(pd.DataFrame(
        (data.X.iloc[10:].to_numpy() - mean) / scale, columns=['f0']))
    np.testing.assert_allclose(fold.predictions['predict'].target, manual)
    save_model(training, tmp_path / (family + '-scaled'))
    restored = load_model(tmp_path / (family + '-scaled'))
    heldout = replace(data, X=data.X.iloc[10:].copy(), y=data.y.iloc[10:].copy(), metadata=data.metadata.iloc[10:].copy())
    np.testing.assert_allclose(restored.predict(heldout)['predict'], fold.predictions['predict'])
    assert restored.model.preprocessing_[0].n_samples_seen_ == 8


def test_explicit_target_transform_is_inverted_after_prediction_before_outputs(tmp_path):
    data, plan = scaling_data()
    recipe = dict(model=node('sklearn.compose.TransformedTargetRegressor',
                             regressor=node('sklearn.linear_model.LinearRegression'),
                             transformer=node('sklearn.preprocessing.StandardScaler')),
                  preprocessors=[node('test.AuditedXScaler')])
    training = TrainingRunner(ModelFactory(recipe, scaling_catalog())).run(data, plan)
    fold = training.folds[0]
    x_scaler, target_model = fold.model.estimator.steps[0][1], fold.model.estimator.steps[-1][1]
    y_train = data.y.iloc[:10, 0].to_numpy()
    np.testing.assert_allclose(target_model.transformer_.mean_, [y_train.mean()])
    np.testing.assert_allclose(target_model.transformer_.scale_, [y_train.std()])
    assert target_model.transformer_.n_samples_seen_ == 10
    transformed_X = x_scaler.transform(data.X.iloc[10:])
    standardized_prediction = target_model.regressor_.predict(transformed_X)
    expected_original_units = standardized_prediction * y_train.std() + y_train.mean()
    actual = fold.predictions['predict'].target.to_numpy()
    np.testing.assert_allclose(actual, expected_original_units, atol=1e-8)
    np.testing.assert_allclose(actual, data.y.iloc[10:, 0], atol=1e-8)
    assert not np.allclose(actual, standardized_prediction)
    # Truth is never replaced by transformed y; reporters compare original units.
    np.testing.assert_array_equal(fold.y_true.target, data.y.iloc[10:, 0])
    save_model(training, tmp_path / 'target-transformed')
    restored = load_model(tmp_path / 'target-transformed')
    heldout = replace(data, X=data.X.iloc[10:].copy(), y=data.y.iloc[10:].copy(), metadata=data.metadata.iloc[10:].copy())
    np.testing.assert_allclose(restored.predict(heldout)['predict'].target, actual)
