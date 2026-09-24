"""NB2 modal predictions match the probability mass, without changing fitting."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.stats import nbinom
from sklearn.base import clone, get_tags
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from test_negative_binomial import population
from test_ui_workflow import ui_recipe
from xdiyo_analytics.reporting import CoefficientReporter
from xdiyo_analytics.training import NegativeBinomialRegressor, save_model, load_model
from xdiyo_analytics.ui import prepare_recipe, run_recipe
from xdiyo_analytics.ui.recipe import node


@pytest.mark.parametrize('dispersion', [.1, .2, .6, 1., 1.8])
def test_mode_is_a_maximum_of_independent_scipy_mass(dispersion, monkeypatch):
    X, y = population()
    model = NegativeBinomialRegressor(dispersion=dispersion, prediction='mode').fit(X, y)
    means = np.array([.25, 1., 4.7, 10., 25.])
    monkeypatch.setattr(model, 'predict_mean', lambda _: means.copy())
    predicted = model.predict(X.iloc[:len(means)])
    assert predicted.shape == means.shape
    np.testing.assert_array_equal(predicted, np.floor(predicted))
    assert (predicted >= 0).all()
    for mean, mode in zip(means, predicted):
        distribution = nbinom(1 / dispersion, 1 / (1 + dispersion * mean))
        largest_mass = distribution.pmf(np.arange(300)).max()
        assert distribution.pmf(mode) == pytest.approx(largest_mass, rel=1e-12, abs=1e-14)
    if dispersion >= 1:
        np.testing.assert_array_equal(predicted, 0.)


def test_exact_tie_selects_upper_mode_and_zero_mean_is_zero(monkeypatch):
    X, y = population()
    model = NegativeBinomialRegressor(dispersion=.2, prediction='mode').fit(X, y)
    monkeypatch.setattr(model, 'predict_mean', lambda _: np.array([10., 0.]))
    np.testing.assert_array_equal(model.predict(X.iloc[:2]), [8., 0.])
    distribution = nbinom(5., 1 / 3)
    assert distribution.pmf(7) == pytest.approx(distribution.pmf(8), rel=1e-12)


@pytest.mark.parametrize('penalty', ['none', 'l1', 'l2', 'elasticnet'])
def test_mode_changes_only_output_not_fitted_likelihood(penalty):
    X, y = population()
    options = dict(dispersion=.3, penalty=penalty, alpha=.1, max_iter=500)
    mean = NegativeBinomialRegressor(**options).fit(X, y)
    mode = NegativeBinomialRegressor(prediction='mode', **options).fit(X, y)
    np.testing.assert_array_equal(mean.coef_, mode.coef_)
    assert mean.intercept_ == mode.intercept_
    np.testing.assert_array_equal(mean.predict(X), mode.predict_mean(X))
    assert mode.coefficient_units_ == 'log_mean'
    assert mode.prediction_ == 'mode'
    assert get_tags(mode).regressor_tags.poor_score
    assert not get_tags(mean).regressor_tags.poor_score


def test_fitted_choice_and_dispersion_stay_frozen_and_old_model_defaults_to_mean():
    X, y = population()
    model = NegativeBinomialRegressor(dispersion=.2, prediction='mode').fit(X, y)
    expected = model.predict(X)
    model.set_params(prediction='mean', dispersion=2.)
    np.testing.assert_array_equal(model.predict(X), expected)
    del model.prediction_  # Pre-option persisted models have no fitted output choice.
    del model.prediction  # Nor the new constructor parameter on old serialized objects.
    np.testing.assert_array_equal(model.predict(X), model.predict_mean(X))


def test_modal_zero_after_log_mean_underflow():
    X, y = population()
    model = NegativeBinomialRegressor(dispersion=.2, prediction='mode').fit(X, y)
    model.coef_[:] = 0
    model.intercept_ = -1000.
    np.testing.assert_array_equal(model.predict_mean(X), 0.)
    np.testing.assert_array_equal(model.predict(X), 0.)


def test_invalid_prediction_choice_rejected_on_fit():
    X, y = population()
    with pytest.raises(ValueError, match='prediction'):
        NegativeBinomialRegressor(prediction='median').fit(X, y)


def test_mode_pipeline_grid_and_actual_ui_persistence(ui_recipe, tmp_path):
    X, y = population()
    pipe = Pipeline([('scale', StandardScaler()),
                     ('model', NegativeBinomialRegressor(dispersion=.3, max_iter=500))])
    grid = GridSearchCV(pipe, {'model__prediction': ['mean', 'mode']}, cv=2,
                        scoring='neg_mean_absolute_error', error_score='raise').fit(X, y)
    assert np.isfinite(grid.cv_results_['mean_test_score']).all()
    assert not hasattr(clone(grid.best_estimator_).named_steps['model'], 'prediction_')
    ui_recipe['model'] = node('training.NegativeBinomialRegressor', dispersion=.3, prediction='mode')
    prepared = prepare_recipe(ui_recipe)
    result = run_recipe(ui_recipe, prepared=prepared)
    fold = result.training.folds[0]
    values = fold.predictions['predict'].to_numpy()
    np.testing.assert_array_equal(values, np.floor(values))
    assert fold.training_summary['prediction'] == 'mode'
    report = CoefficientReporter(type='overall').run(SimpleNamespace(partition='model', models={0: fold.model}))
    assert set(report.tables['coefficients'].coefficient_units) == {'log_mean'}
    assert any('mode' in note.lower() and 'mean' in note.lower() for note in report.notes)
    saved = load_model(save_model(result.training, tmp_path / 'modal-model'))
    pipeline = fold.model.estimator
    np.testing.assert_array_equal(saved.predict(prepared.dataset)['predict'],
                                  pipeline.predict(prepared.dataset.X).reshape(-1, 1))
    assert saved.model.estimator[-1].prediction_ == 'mode'


@parametrize_with_checks([NegativeBinomialRegressor(prediction='mode')], xfail_strict=True)
def test_mode_sklearn_contract(estimator, check):
    check(estimator)
