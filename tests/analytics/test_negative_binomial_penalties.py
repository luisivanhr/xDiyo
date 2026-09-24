"""Penalized NB2 fits agree with native likelihood fits and retain count units."""

import numpy as np
import pytest
import statsmodels.api as sm
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from test_negative_binomial import population
from test_ui_workflow import ui_recipe
from xdiyo_analytics.training import NegativeBinomialRegressor, save_model, load_model
from xdiyo_analytics.ui import prepare_recipe, run_recipe
from xdiyo_analytics.ui.recipe import node


@pytest.mark.parametrize('penalty,ratio', [('l1', 1.), ('elasticnet', .35), ('l2', 0.), ('elasticnet', 0.)])
@pytest.mark.parametrize('intercept', [True, False])
def test_penalty_matches_independent_statsmodels_regularized_fit(penalty, ratio, intercept):
    X, y = population()
    alpha, dispersion = .08, .7
    actual = NegativeBinomialRegressor(dispersion=dispersion, penalty=penalty, alpha=alpha,
        l1_ratio=ratio, fit_intercept=intercept, max_iter=1000, tol=1e-9).fit(X, y)
    design = sm.add_constant(X.to_numpy(), has_constant='add') if intercept else X.to_numpy()
    weights = np.full(design.shape[1], alpha)
    if intercept:
        weights[0] = 0.
    model = sm.GLM(y, design, family=sm.families.NegativeBinomial(alpha=dispersion))
    reference = model.fit_regularized(alpha=weights, L1_wt=ratio, maxiter=1000,
        cnvrg_tol=1e-10, zero_tol=1e-10, refit=False, opt_method='bfgs')
    expected = np.asarray(reference.params)
    np.testing.assert_allclose(actual.coef_, expected[1:] if intercept else expected, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(actual.predict(X), reference.predict(design), atol=5e-5, rtol=2e-5)
    actual_params = np.r_[actual.intercept_, actual.coef_] if intercept else actual.coef_
    gradient = -model.score(actual_params) / len(y)
    if ratio == 0:
        np.testing.assert_allclose(gradient + weights * actual_params, 0., atol=2e-7)
        assert actual.n_iter_ is not None
    else:
        active = actual_params != 0
        np.testing.assert_allclose(gradient[active] + weights[active] * (
            (1 - ratio) * actual_params[active] + ratio * np.sign(actual_params[active])), 0., atol=2e-7)
        assert np.all(np.abs(gradient[~active]) <= weights[~active] * ratio + 2e-7)
        assert actual.n_iter_ is None
    assert np.all(actual.predict(X) > 0)


@pytest.mark.parametrize('penalty', ['l1', 'l2', 'elasticnet'])
def test_zero_strength_is_exact_unpenalized_behavior(penalty):
    X, y = population()
    plain = NegativeBinomialRegressor().fit(X, y)
    zero = NegativeBinomialRegressor(penalty=penalty, alpha=0.).fit(X, y)
    np.testing.assert_array_equal(zero.coef_, plain.coef_)
    np.testing.assert_array_equal(zero.predict(X), plain.predict(X))
    assert zero.intercept_ == plain.intercept_
    assert zero.n_iter_ == plain.n_iter_


def test_none_ignores_strength_and_l1_leaves_intercept_unpenalized():
    X, y = population()
    baseline = NegativeBinomialRegressor().fit(X, y)
    ignored = NegativeBinomialRegressor(penalty='none', alpha=1e6).fit(X, y)
    np.testing.assert_array_equal(baseline.predict(X), ignored.predict(X))
    sparse = NegativeBinomialRegressor(penalty='l1', alpha=100., max_iter=500).fit(X, y)
    np.testing.assert_array_equal(sparse.coef_, np.zeros(X.shape[1]))
    assert np.exp(sparse.intercept_) == pytest.approx(np.mean(y), rel=1e-7)
    np.testing.assert_allclose(sparse.predict(X), np.mean(y), rtol=1e-7)


def test_ridge_iteration_budget_and_nonconvergence_are_retained():
    X, y = population()
    with pytest.warns(ConvergenceWarning, match='did not converge'):
        model = NegativeBinomialRegressor(penalty='l2', alpha=.05, max_iter=1, tol=1e-12).fit(X, y)
    assert model.n_iter_ == 1
    assert not model.converged_


@pytest.mark.parametrize('changes', [{'penalty': 'unknown'}, {'alpha': -1}, {'alpha': np.inf},
                                   {'alpha': True}, {'l1_ratio': -.1}, {'l1_ratio': 1.1}, {'l1_ratio': np.nan}])
def test_invalid_regularization_parameters(changes):
    model = NegativeBinomialRegressor(**changes)
    X, y = population()
    with pytest.raises((ValueError, TypeError)):
        model.fit(X, y)


def test_penalty_grid_pipeline_and_saved_ui_run(ui_recipe, tmp_path):
    X, y = population()
    pipeline = Pipeline([('scale', StandardScaler()), ('model', NegativeBinomialRegressor(max_iter=500))])
    search = GridSearchCV(pipeline, {'model__penalty': ['l1', 'l2'], 'model__alpha': [.01, .1]},
                          cv=2, scoring='neg_mean_squared_error', error_score='raise').fit(X, y)
    assert np.isfinite(search.cv_results_['mean_test_score']).all()
    copied = clone(search.best_estimator_)
    assert copied.get_params()['model__penalty'] in ['l1', 'l2']
    assert not hasattr(copied.named_steps['model'], 'coef_')
    ui_recipe['model'] = node('training.NegativeBinomialRegressor', penalty='elasticnet', alpha=.1,
                              l1_ratio=.4, max_iter=500)
    prepared = prepare_recipe(ui_recipe)
    result = run_recipe(ui_recipe, prepared=prepared)
    saved = load_model(save_model(result.training, tmp_path / 'penalized-nb'))
    native_pipeline = result.training.folds[0].model.estimator
    np.testing.assert_allclose(saved.predict(prepared.dataset)['predict'],
                               native_pipeline.predict(prepared.dataset.X).reshape(-1, 1))
    assert saved.model.estimator[-1].get_params()['penalty'] == 'elasticnet'
