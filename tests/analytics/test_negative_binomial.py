"""Independent NB2 numerics, sklearn contracts and actual recipe integration."""

from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from sklearn.base import clone, get_tags
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from test_ui_workflow import ui_recipe
from xdiyo_analytics.reporting import CoefficientReporter
from xdiyo_analytics.training import NegativeBinomialRegressor, save_model, load_model
from xdiyo_analytics.ui import prepare_recipe, run_recipe
from xdiyo_analytics.ui.recipe import ModelFactory, catalog_for_ui, node


def population():
    rng = np.random.default_rng(193)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=['attack', 'defense', 'home'])
    mu = np.exp(.7 + X.to_numpy() @ [.4, -.3, .15])
    dispersion = .6
    y = rng.negative_binomial(1 / dispersion, 1 / (1 + dispersion * mu))
    return X, y


@pytest.mark.parametrize('intercept', [True, False])
@pytest.mark.parametrize('dispersion', [.2, 1.4])
def test_parameters_and_prediction_match_direct_statsmodels_glm(intercept, dispersion):
    X, y = population()
    before_X, before_y = X.copy(deep=True), y.copy()
    estimator = NegativeBinomialRegressor(dispersion=dispersion, fit_intercept=intercept, max_iter=200, tol=1e-10)
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter('always', ConvergenceWarning)
        assert estimator.fit(X, y) is estimator
    design = sm.add_constant(X.to_numpy(), has_constant='add') if intercept else X.to_numpy()
    reference = sm.GLM(y, design, family=sm.families.NegativeBinomial(alpha=dispersion)).fit(maxiter=200, tol=1e-10)
    np.testing.assert_allclose(estimator.coef_, reference.params[1:] if intercept else reference.params, atol=1e-9)
    assert estimator.intercept_ == pytest.approx(reference.params[0] if intercept else 0., abs=1e-9)
    np.testing.assert_allclose(estimator.predict(X), reference.predict(design), rtol=1e-10)
    assert estimator.predict(X).shape == (len(X),)
    assert np.all(estimator.predict(X) > 0)
    assert estimator.dispersion_ == dispersion
    assert estimator.converged_ == bool(reference.converged)
    assert any(issubclass(item.category, ConvergenceWarning) for item in emitted) == (not estimator.converged_)
    assert estimator.n_iter_ > 0
    np.testing.assert_array_equal(estimator.feature_names_in_, X.columns)
    pd.testing.assert_frame_equal(X, before_X)
    np.testing.assert_array_equal(y, before_y)


def test_clone_pipeline_grid_and_refit_reset():
    X, y = population()
    model = NegativeBinomialRegressor(dispersion=.5, max_iter=150).fit(X, y)
    fresh = clone(model)
    assert fresh.get_params() == model.get_params()
    assert not hasattr(fresh, 'coef_')
    assert get_tags(model).target_tags.positive_only
    assert not get_tags(model).target_tags.multi_output
    pipeline = Pipeline([('scale', StandardScaler()), ('model', fresh)])
    pipeline.set_params(model__dispersion=.2)
    assert pipeline.get_params()['model__dispersion'] == .2
    search = GridSearchCV(pipeline, {'model__dispersion': [.2, .8]}, cv=3,
                          scoring='neg_mean_squared_error', error_score='raise').fit(X, y)
    assert np.isfinite(search.cv_results_['mean_test_score']).all()
    assert search.predict(X.iloc[:3]).shape == (3,)
    model.fit(X[['attack']], y)
    assert model.coef_.shape == (1,)
    np.testing.assert_array_equal(model.feature_names_in_, ['attack'])


@pytest.mark.parametrize('changes', [{'dispersion': 0}, {'dispersion': -1}, {'dispersion': np.inf},
                                   {'max_iter': 0}, {'max_iter': 1.5}, {'tol': 0}, {'tol': np.nan}])
def test_invalid_parameters_are_rejected_on_fit(changes):
    model = NegativeBinomialRegressor(**changes)
    X, y = population()
    with pytest.raises((ValueError, TypeError)):
        model.fit(X, y)


def test_target_domain_fitted_state_and_feature_validation():
    X, y = population()
    with pytest.raises(NotFittedError):
        NegativeBinomialRegressor().predict(X)
    for invalid in [-np.ones(len(y)), np.zeros(len(y)), np.full(len(y), np.nan), np.c_[y, y]]:
        with pytest.raises(ValueError):
            NegativeBinomialRegressor().fit(X, invalid)
    model = NegativeBinomialRegressor().fit(X, y)
    with pytest.raises(ValueError):
        model.predict(X[['home', 'defense', 'attack']])
    with pytest.raises(ValueError):
        model.predict(X[['attack']])
    # Native GLM accepts nonnegative fractional outcomes; predictions remain means.
    assert np.isfinite(NegativeBinomialRegressor().fit(X, y + .25).predict(X)).all()


def test_actual_ui_recipe_persistence_and_log_coefficient_report(ui_recipe, tmp_path):
    ui_recipe['model'] = node('training.NegativeBinomialRegressor', dispersion=.7)
    prepared = prepare_recipe(ui_recipe)
    result = run_recipe(ui_recipe, prepared=prepared)
    fold = result.training.folds[0]
    assert np.isfinite(fold.predictions['predict'].to_numpy()).all()
    assert (fold.predictions['predict'].to_numpy() > 0).all()
    report = CoefficientReporter(type='overall').run(SimpleNamespace(partition='model', models={0: fold.model}))
    terms = report.tables['coefficients']
    assert set(terms.coefficient_units) == {'log_mean'}
    assert any('log' in note.lower() for note in report.notes)
    model = fold.model.estimator
    transformed_X = model[:-1].transform(prepared.dataset.X.iloc[fold.test_positions])
    slopes = terms.loc[terms.term.eq('feature'), 'coefficient'].to_numpy()
    intercept = terms.loc[terms.term.eq('intercept'), 'coefficient'].iloc[0]
    np.testing.assert_allclose(np.exp(transformed_X @ slopes + intercept), fold.predictions['predict'].iloc[:, 0])
    location = save_model(result.training, tmp_path / 'saved-nb')
    loaded = load_model(location)
    np.testing.assert_allclose(loaded.predict(prepared.dataset)['predict'],
                               model.predict(prepared.dataset.X).reshape(-1, 1))
    ui_recipe['target_transformer'] = node('targets.StandardScaler')
    with pytest.raises((ValueError, TypeError), match='(?i)(target|count|untransformed)'):
        ModelFactory(ui_recipe, catalog_for_ui())()


@parametrize_with_checks([NegativeBinomialRegressor()], xfail_strict=True)
def test_public_sklearn_checks(estimator, check):
    check(estimator)
