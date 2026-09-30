"""Learned NB2 dispersion: independent likelihood, sklearn and pipeline evidence."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from scipy.stats import nbinom
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks
from statsmodels.discrete.discrete_model import NegativeBinomial

from test_ui_workflow import ui_recipe
from xdiyo_analytics.training import NegativeBinomialRegressor, save_model, load_model
from xdiyo_analytics.ui import prepare_recipe, run_recipe
from xdiyo_analytics.ui.recipe import node
from xdiyo_analytics.ui.inventory import inventory


def population(intercept=True):
    rng = np.random.default_rng(428)
    X = pd.DataFrame(rng.normal(size=(450, 3)), columns=['attack', 'defense', 'home'])
    mean = np.exp((1.1 if intercept else 0.) + X.to_numpy() @ [.5, -.3, .2])
    dispersion = .65
    y = rng.negative_binomial(1/dispersion, 1/(1+dispersion*mean))
    return X, y


@pytest.mark.parametrize('intercept', [True, False])
@pytest.mark.parametrize('initial', [.15, 2.])
def test_joint_mle_matches_independent_statsmodels_nb2(intercept, initial):
    X, y = population(intercept)
    original = X.copy(deep=True), y.copy()
    actual = NegativeBinomialRegressor(learn_dispersion=True, dispersion=initial,
        fit_intercept=intercept, max_iter=1000, tol=1e-9).fit(X, y)
    design = np.column_stack([np.ones(len(y)), X]) if intercept else X.to_numpy()
    reference = NegativeBinomial(y, design, loglike_method='nb2').fit(method='bfgs',
        maxiter=1000, gtol=1e-9, disp=False)
    params = np.r_[actual.intercept_, actual.coef_, actual.dispersion_] if intercept else np.r_[actual.coef_, actual.dispersion_]
    np.testing.assert_allclose(params, reference.params, rtol=2e-4, atol=2e-5)
    np.testing.assert_allclose(actual.predict_mean(X), reference.predict(design), rtol=3e-4, atol=1e-4)
    assert actual.dispersion == initial and actual.dispersion_ > 0
    assert actual.dispersion_estimated_ and not actual.dispersion_at_boundary_
    assert actual.converged_ and actual.n_iter_ > 0
    pd.testing.assert_frame_equal(X, original[0])
    np.testing.assert_array_equal(y, original[1])


@pytest.mark.parametrize('penalty,ratio', [('l1', 1.), ('l2', 0.), ('elasticnet', .4)])
def test_joint_penalty_shrinks_slopes_and_satisfies_unpenalized_dispersion_score(penalty, ratio):
    X, y = population()
    baseline = NegativeBinomialRegressor(learn_dispersion=True, max_iter=1000, tol=1e-9).fit(X, y)
    strength = .08
    actual = NegativeBinomialRegressor(learn_dispersion=True, penalty=penalty, alpha=strength,
        l1_ratio=ratio, max_iter=1500, tol=1e-9).fit(X, y)
    assert np.linalg.norm(actual.coef_) < np.linalg.norm(baseline.coef_)
    design = np.column_stack([np.ones(len(y)), X])
    params = np.r_[actual.intercept_, actual.coef_, actual.dispersion_]
    def loss(theta):
        mu, dispersion = np.exp(design @ theta[:-1]), theta[-1]
        return -nbinom.logpmf(y, 1/dispersion, 1/(1+dispersion*mu)).mean()
    perturbations = np.eye(len(params))*1e-5
    gradient = np.array([(loss(params+step)-loss(params-step))/2e-5 for step in perturbations])
    np.testing.assert_allclose(gradient[[0, -1]], 0., atol=2e-4)
    active = np.abs(actual.coef_) > 1e-6
    np.testing.assert_allclose(gradient[1:-1][active] + strength*((1-ratio)*actual.coef_[active]
        + ratio*np.sign(actual.coef_[active])), 0., atol=2e-4)
    assert np.all(np.abs(gradient[1:-1][~active]) <= strength*ratio+2e-4)


def test_large_l1_penalty_does_not_shrink_intercept_or_dispersion():
    X, y = population()
    actual = NegativeBinomialRegressor(learn_dispersion=True, penalty='l1', alpha=10.,
        max_iter=1500, tol=1e-9).fit(X, y)
    reference = NegativeBinomial(y, np.ones((len(y), 1)), loglike_method='nb2').fit(disp=False, maxiter=1000)
    np.testing.assert_allclose(actual.coef_, 0., atol=1e-6)
    assert np.exp(actual.intercept_) == pytest.approx(np.mean(y), rel=2e-5)
    assert actual.dispersion_ == pytest.approx(reference.params[-1], rel=2e-4)


@pytest.mark.parametrize('changes', [{'learn_dispersion': 'yes'}, {'learn_dispersion': 1},
    {'learn_dispersion': None}, {'learn_dispersion': True, 'dispersion': 0},
    {'learn_dispersion': True, 'dispersion': -1}, {'learn_dispersion': True, 'dispersion': np.inf}])
def test_learn_switch_and_positive_initial_value_are_validated_at_fit(changes):
    model = NegativeBinomialRegressor(**changes)
    X, y = population()
    with pytest.raises(ValueError):
        model.fit(X, y)


def test_learned_mode_uses_fitted_dispersion_and_refit_resets_it():
    X, y = population()
    model = NegativeBinomialRegressor(learn_dispersion=True, dispersion=.1, prediction='mode', max_iter=1000).fit(X, y)
    mean, fitted = model.predict_mean(X), model.dispersion_
    np.testing.assert_array_equal(model.predict(X), np.floor(mean*max(0, 1-fitted)))
    for mu, mode in zip(mean[:10], model.predict(X)[:10]):
        rv = nbinom(1/fitted, 1/(1+fitted*mu))
        assert rv.pmf(mode) >= rv.pmf(max(0, mode-1))-1e-12
        assert rv.pmf(mode) >= rv.pmf(mode+1)-1e-12
    before = model.predict(X)
    model.set_params(dispersion=2.)
    np.testing.assert_array_equal(model.predict(X), before)
    model.set_params(learn_dispersion=False, dispersion=.4).fit(X, y)
    assert model.dispersion_ == .4
    fresh = clone(model)
    assert fresh.get_params() == model.get_params() and not hasattr(fresh, 'dispersion_')


def test_joint_pipeline_and_small_grid():
    X, y = population()
    pipeline = Pipeline([('scale', StandardScaler()), ('model', NegativeBinomialRegressor(max_iter=1000))])
    grid = GridSearchCV(pipeline, {'model__learn_dispersion': [False, True], 'model__dispersion': [.3, 1.]},
        cv=2, scoring='neg_mean_squared_error', error_score='raise').fit(X, y)
    assert np.isfinite(grid.cv_results_['mean_test_score']).all()
    assert grid.predict(X[:4]).shape == (4,)
    assert 'model__learn_dispersion' in clone(grid.best_estimator_).get_params(deep=True)


@pytest.mark.parametrize('penalty', ['l1', 'l2', 'elasticnet'])
def test_zero_penalty_matches_unregularized_joint_fit(penalty):
    X, y = population()
    baseline = NegativeBinomialRegressor(learn_dispersion=True, max_iter=500).fit(X, y)
    zero = NegativeBinomialRegressor(learn_dispersion=True, penalty=penalty, alpha=0., max_iter=500).fit(X, y)
    np.testing.assert_array_equal(zero.coef_, baseline.coef_)
    assert zero.dispersion_ == baseline.dispersion_


def test_joint_iteration_limit_warns_and_retains_convergence_state():
    X, y = population()
    with pytest.warns(ConvergenceWarning, match='did not converge'):
        model = NegativeBinomialRegressor(learn_dispersion=True, max_iter=1, tol=1e-12).fit(X, y)
    assert model.n_iter_ == 1 and not model.converged_


def test_learned_ui_training_only_native_distribution_and_persistence(ui_recipe, tmp_path):
    fields = {f['name']: f for f in inventory()['components']['training.NegativeBinomialRegressor']['fields']}
    assert fields['learn_dispersion']['kind'] == 'boolean'
    ui_recipe['model'] = node('training.NegativeBinomialRegressor', learn_dispersion=True,
        dispersion=.3, prediction='mode', penalty='l2', alpha=.1, max_iter=1000)
    prepared = prepare_recipe(ui_recipe)
    first = run_recipe(ui_recipe, prepared=prepared)
    fold = first.training.folds[0]
    native = fold.model.estimator[-1]
    assert native.dispersion_estimated_
    assert fold.training_summary['dispersion_estimated'] is True
    assert fold.training_summary['dispersion'] == native.dispersion_
    distribution = fold.predictions['count_distribution']['corners']
    np.testing.assert_allclose(distribution['dispersion'], native.dispersion_)
    transformed = fold.model.estimator[:-1].transform(prepared.dataset.X.iloc[fold.test_positions])
    np.testing.assert_allclose(distribution['mean'], np.exp(native.intercept_ + transformed @ native.coef_))
    saved = load_model(save_model(first.training, tmp_path/'learned-nb'))
    assert saved.model.estimator[-1].dispersion_ == native.dispersion_
    predictions = saved.predict(prepared.dataset)
    np.testing.assert_array_equal(predictions['predict'].to_numpy().ravel(), fold.model.estimator.predict(prepared.dataset.X))
    np.testing.assert_allclose(predictions['count_distribution']['corners']['dispersion'], native.dispersion_)
    changed = prepare_recipe(ui_recipe)
    changed.dataset.y.iloc[fold.test_positions] = 1000
    recipe = deepcopy(ui_recipe)
    recipe['output_dir'] = str(tmp_path/'different-outcomes')
    second = run_recipe(recipe, prepared=changed)
    other = second.training.folds[0].model.estimator[-1]
    assert other.dispersion_ == native.dispersion_
    np.testing.assert_array_equal(other.coef_, native.coef_)


@parametrize_with_checks([NegativeBinomialRegressor(learn_dispersion=True, max_iter=500)], xfail_strict=True)
def test_learned_sklearn_contract(estimator, check):
    check(estimator)
