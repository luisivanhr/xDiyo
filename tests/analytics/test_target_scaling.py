"""Optional label scaling through actual factories, folds and persistence."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_selection_samples import sample
from xdiyo_analytics.experiments import FootballExperiment, PreparedExperiment, RefitPolicy
from xdiyo_analytics.selection import ModelSelection
from xdiyo_analytics.splits import Fold, SplitPlan
from xdiyo_analytics.training import EstimatorAdapter, TargetTransformAdapter, TrainingRunner, save_model, load_model
from xdiyo_analytics.ui.recipe import ModelFactory, catalog_for_ui, default_recipe, make_candidate, node, validate_recipe


def scaling_data():
    data = sample()
    values = np.r_[np.arange(8.), [500., 700.], [1000., 2000., 3000., 4000., 5000., 6000.]]
    data.X = pd.DataFrame({'signal': values}, index=data.X.index)
    data.y = pd.DataFrame({'target': 1000. + 7. * values}, index=data.X.index)
    rows = np.arange(len(values))
    return data, SplitPlan([Fold(rows[:10], rows[10:], rows[10:])], len(rows), rows)


def recipe_for(model=None, *, alpha=.01, scaler=None):
    return dict(model=model or node('sklearn.linear_model.Lasso', alpha=alpha, fit_intercept=False,
                                   tol=1e-12, max_iter=10000),
                preprocessors=[node('sklearn.impute.SimpleImputer', strategy='median'),
                               node('sklearn.preprocessing.StandardScaler')],
                target_transformer=node('targets.StandardScaler', **(scaler or {})))


def test_optional_recipe_catalog_and_none_default():
    recipe = default_recipe()
    assert recipe['target_transformer'] is None
    catalog = catalog_for_ui()
    assert any(entry['id'] == 'targets.StandardScaler' and entry['category'] == 'target_transformer'
               for entry in catalog.schema())
    assert type(ModelFactory(recipe, catalog)()) is EstimatorAdapter
    recipe['target_transformer'] = node('targets.StandardScaler')
    assert isinstance(ModelFactory(validate_recipe(recipe), catalog)(), TargetTransformAdapter)


@pytest.mark.parametrize('scaler_options', [{}, {'with_mean': False}, {'with_std': False}])
def test_lasso_target_scaler_matches_manual_ttr_and_original_unit_coefficients(scaler_options, tmp_path):
    data, plan = scaling_data()
    fitted = TrainingRunner(ModelFactory(recipe_for(scaler=scaler_options), catalog_for_ui())).run(data, plan)
    fold = fitted.folds[0]
    adapter = fold.model
    assert adapter.transformer_.n_samples_seen_ == 10
    np.testing.assert_allclose(adapter.transformer_.mean_, data.y.iloc[:10].mean())
    assert not np.allclose(adapter.transformer_.mean_, data.y.mean())
    manual = TransformedTargetRegressor(regressor=Pipeline([
        ('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler()),
        ('model', Lasso(alpha=.01, fit_intercept=False, tol=1e-12, max_iter=10000))]),
        transformer=StandardScaler(**scaler_options))
    manual.fit(data.X.iloc[:10], data.y.iloc[:10, 0])
    np.testing.assert_allclose(fold.predictions['predict'].target, manual.predict(data.X.iloc[10:]), rtol=1e-12)
    pd.testing.assert_series_equal(fold.y_true.target.reset_index(drop=True), data.y.iloc[10:, 0].reset_index(drop=True))
    coefficients = adapter.coefficient_table()
    transformed_X = adapter.estimator.estimator[:-1].transform(data.X.iloc[10:])
    slope = coefficients.loc[coefficients.term.eq('feature'), 'coefficient'].to_numpy()
    intercept = coefficients.loc[coefficients.term.eq('intercept'), 'coefficient'].iloc[0]
    np.testing.assert_allclose(transformed_X @ slope + intercept, fold.predictions['predict'].target)
    path = save_model(fitted, tmp_path / 'target-scaled-model')
    restored = load_model(path)
    heldout = replace(data, X=data.X.iloc[10:], y=data.y.iloc[10:], metadata=data.metadata.iloc[10:])
    np.testing.assert_allclose(restored.predict(heldout)['predict'], fold.predictions['predict'])
    assert restored.model.transformer_.n_samples_seen_ == 10


@pytest.mark.parametrize('family', ['lightgbm', 'xgboost'])
def test_native_validation_targets_use_fit_only_scaler_and_inverse_once(family, monkeypatch):
    native = pytest.importorskip(family)
    constructor = native.LGBMRegressor if family == 'lightgbm' else native.XGBRegressor
    original = constructor.fit
    seen = {}

    def capture(self, X, y, **kwargs):
        seen['fit_y'] = np.asarray(y).copy()
        seen['validation_y'] = np.asarray(kwargs['eval_set'][0][1]).copy()
        return original(self, X, y, **kwargs)

    monkeypatch.setattr(constructor, 'fit', capture)
    parameters = dict(n_estimators=3, max_depth=2, n_jobs=1)
    parameters.update(dict(verbosity=-1, min_child_samples=1) if family == 'lightgbm'
                      else dict(tree_method='hist', objective='reg:quantileerror', quantile_alpha=.9))
    data, plan = scaling_data()
    recipe = recipe_for(node(family + '.' + constructor.__name__, **parameters))
    result = TrainingRunner(ModelFactory(recipe, catalog_for_ui()), validation=[8, 9]).run(data, plan)
    fold = result.folds[0]
    adapter = fold.model
    mean, scale = data.y.iloc[:8, 0].mean(), data.y.iloc[:8, 0].std(ddof=0)
    np.testing.assert_allclose(adapter.transformer_.mean_, [mean])
    np.testing.assert_allclose(adapter.transformer_.scale_, [scale])
    assert adapter.transformer_.n_samples_seen_ == 8
    np.testing.assert_allclose(seen['fit_y'], (data.y.iloc[:8, 0] - mean) / scale)
    np.testing.assert_allclose(seen['validation_y'], (data.y.iloc[8:10, 0] - mean) / scale)
    native_adapter = adapter.estimator
    transformed_X = native_adapter.preprocessing_.transform(data.X.iloc[10:])
    native_prediction = native_adapter.native_model_.predict(pd.DataFrame(transformed_X, columns=['f0']))
    np.testing.assert_allclose(fold.predictions['predict'].target, native_prediction * scale + mean, rtol=1e-6)
    assert fold.training_history.metric.str.endswith('[transformed target]').all()
    np.testing.assert_array_equal(fold.y_true.target, data.y.iloc[10:, 0])


def inner_plan(data):
    rows = np.arange(len(data.X))
    return SplitPlan([Fold(rows[:4], rows[4:6], rows[4:6])], len(rows), rows)


def test_nested_grid_refit_recovery_and_future_predictions_keep_original_units(tmp_path):
    data, _ = scaling_data()
    rows = np.arange(len(data.X))
    outer = SplitPlan([Fold(rows[:8], rows[8:10], rows[8:10]),
                       Fold(rows[:10], rows[10:], rows[10:])], len(rows), rows)
    catalog = catalog_for_ui()
    candidates = [make_candidate(dict(recipe_for(alpha=alpha), candidate={'name': f'Lasso alpha={alpha}'}), catalog)
                  for alpha in (.001, .1)]
    experiment = FootballExperiment('Synthetic target scaling', output_dir=tmp_path)
    result = experiment.run(PreparedExperiment(data, outer),
        model_selection=ModelSelection(candidates, metrics='mse'), inner_plan_factory=inner_plan,
        refit_policy=RefitPolicy(train_positions=rows, candidate=candidates[0]))
    scalers = [fold.model.transformer_ for fold in result.training.folds]
    assert scalers[0] is not scalers[1]
    for fold, scaler in zip(result.training.folds, scalers):
        np.testing.assert_allclose(scaler.mean_, data.y.iloc[fold.fit_positions].mean())
        native = fold.model.estimator.estimator
        z = native.predict(data.X.iloc[fold.test_positions])
        expected = z * scaler.scale_[0] + scaler.mean_[0]
        np.testing.assert_allclose(fold.predictions['predict'].target, expected)
    assert result.refit.model.transformer_.n_samples_seen_ == len(data.X)
    path = save_model(result.refit, tmp_path / 'deployment')
    restored = load_model(path)
    np.testing.assert_allclose(restored.predict(data)['predict'], result.refit.predict(data)['predict'])
    loaded = experiment.load(result.record['run_id'])
    for live, recovered in zip(result.training.folds, loaded.training.folds):
        pd.testing.assert_frame_equal(live.predictions['predict'], recovered.predictions['predict'])


def test_reusing_wrapped_estimator_across_folds_is_rejected():
    data, _ = scaling_data()
    rows = np.arange(len(data.X))
    plan = SplitPlan([Fold(rows[:6], rows[6:8], rows[6:8]),
                      Fold(rows[:8], rows[8:], rows[8:])], len(rows), rows)
    shared = EstimatorAdapter(Lasso())
    with pytest.raises(ValueError, match='reused an estimator'):
        TrainingRunner(lambda: TargetTransformAdapter(shared, StandardScaler())).run(data, plan)


def test_unsupported_prediction_contracts_are_explicit():
    with pytest.raises(TypeError, match='classifiers or rankers'):
        TargetTransformAdapter(EstimatorAdapter(LogisticRegression()), StandardScaler())
    with pytest.raises(TypeError, match='currently supports'):
        TargetTransformAdapter(object(), StandardScaler())
    with pytest.raises(ValueError, match='numeric predict'):
        TargetTransformAdapter(EstimatorAdapter(Lasso(), ('predict', 'predict_proba')), StandardScaler())
