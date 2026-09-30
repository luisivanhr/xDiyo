"""Count regression estimators compatible with sklearn pipelines and search."""

from numbers import Integral, Real
import warnings

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_is_fitted, validate_data


class NegativeBinomialRegressor(RegressorMixin, BaseEstimator):
    """NB2 regression with a log link and fixed or learned dispersion.

    By default predicts the conditional mean, exp(intercept + X @ coef), in
    count units. prediction='mode' returns the distribution's modal count;
    predict_mean always returns the mean and fitted coefficients remain log means.
    The conditional variance is mean + dispersion * mean**2. Dispersion is
    fixed by default. learn_dispersion=True jointly estimates it and the mean
    coefficients on training rows; dispersion then supplies its initial value.
    One dispersion is learned per fit, not one per observation. Dense finite inputs
    and one nonnegative numeric target are supported; keep labels unscaled.
    Optional L1/L2/Elastic Net penalties shrink slopes, never the intercept.
    alpha is penalty strength; dispersion controls the NB2 variance separately.
    """

    requires_untransformed_target = True

    def __init__(self, *, dispersion=1.0, fit_intercept=True, max_iter=100, tol=1e-8,
                 penalty='none', alpha=0.1, l1_ratio=0.5, prediction='mean', learn_dispersion=False):
        self.dispersion = dispersion
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.prediction = prediction
        self.learn_dispersion = learn_dispersion

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.positive_only = True
        # The conditional mode is not the squared-error/R2 optimal forecast.
        tags.regressor_tags.poor_score = getattr(self, 'prediction', 'mean') == 'mode'
        return tags

    def fit(self, X, y):
        """Fit the NB2 likelihood; optionally estimate its dispersion jointly."""
        from statsmodels.genmod.generalized_linear_model import GLM
        from statsmodels.genmod.families import NegativeBinomial

        for name in ('dispersion', 'tol'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be a finite positive number.')
        if isinstance(self.max_iter, bool) or not isinstance(self.max_iter, Integral) or self.max_iter < 1:
            raise ValueError('max_iter must be a positive integer.')
        if not isinstance(self.fit_intercept, (bool, np.bool_)):
            raise ValueError('fit_intercept must be a boolean.')
        if not isinstance(self.learn_dispersion, (bool, np.bool_)):
            raise ValueError('learn_dispersion must be a boolean.')
        if self.penalty not in ('none', 'l1', 'l2', 'elasticnet'):
            raise ValueError('penalty must be none, l1, l2 or elasticnet.')
        if self.prediction not in ('mean', 'mode'):
            raise ValueError('prediction must be mean or mode.')
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, Real) or not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError('alpha must be a finite nonnegative penalty strength.')
        if isinstance(self.l1_ratio, bool) or not isinstance(self.l1_ratio, Real) or not np.isfinite(self.l1_ratio) or not 0 <= self.l1_ratio <= 1:
            raise ValueError('l1_ratio must be between 0 and 1.')
        for name in tuple(vars(self)):
            if name.endswith('_'):
                delattr(self, name)
        X, y = validate_data(self, X, y, y_numeric=True, dtype=np.float64)
        if np.any(y < 0):
            raise ValueError('NegativeBinomialRegressor requires nonnegative target values.')
        if not np.any(y > 0):
            raise ValueError('An all-zero target has no finite log-link intercept estimate.')
        design = np.column_stack((np.ones(len(X)), X)) if self.fit_intercept else X
        fitted_dispersion = float(self.dispersion)
        if self.learn_dispersion:
            params, fitted_dispersion, iterations, converged = self._fit_joint(design, y)
        else:
            model = GLM(y, design, family=NegativeBinomial(alpha=self.dispersion))
            if self.penalty == 'none' or self.alpha == 0:
                result = model.fit(maxiter=self.max_iter, tol=self.tol)
                params = np.asarray(result.params)
                iterations, converged = int(result.fit_history['iteration']), bool(result.converged)
            else:
                params, iterations, converged = self._fit_penalized(model, design.shape[1])
        if not np.isfinite(params).all():
            raise ValueError('Negative Binomial fitting produced non-finite coefficients.')
        self.coef_ = params[1:].copy() if self.fit_intercept else params.copy()
        self.intercept_ = float(params[0]) if self.fit_intercept else 0.0
        self.dispersion_ = fitted_dispersion
        self.dispersion_estimated_ = bool(self.learn_dispersion)
        self.dispersion_at_boundary_ = bool(self.learn_dispersion and
            (fitted_dispersion <= 1e-6*(1+1e-5) or fitted_dispersion >= 1e6/(1+1e-5)))
        self.n_iter_ = iterations
        self.converged_ = converged
        self.coefficient_units_ = 'log_mean'
        self.prediction_ = self.prediction
        if not self.converged_:
            warnings.warn('Negative Binomial regression did not converge; increase max_iter or inspect the predictors.',
                          ConvergenceWarning, stacklevel=2)
        return self

    def _fit_joint(self, design, y):
        """Joint likelihood in beta/log-dispersion; penalize only the slopes.

        Positive/negative slope parts implement exact L1 (not a smoothed proxy).
        Log dispersion is bounded to [log(1e-6), log(1e6)] for numerical stability;
        the lower endpoint approximates Poisson and is recorded in fitted state.
        """
        from scipy.optimize import minimize
        from scipy.special import gammaln, digamma, expit

        width, offset = design.shape[1], int(self.fit_intercept)
        strength = self.alpha if self.penalty != 'none' else 0.0
        ratio = {'l1':1.0, 'l2':0.0, 'none':0.0}.get(self.penalty, self.l1_ratio)
        split = strength > 0 and ratio > 0
        slopes = width-offset
        start = np.zeros(width + (slopes if split else 0) + 1)
        if self.fit_intercept:
            start[0] = np.log(y.mean())
        bounds = [(None, None)]*offset + ([(0, None)]*(2*slopes) if split else [(None, None)]*slopes)
        bounds.append((np.log(1e-6), np.log(1e6)))
        start[-1] = np.clip(np.log(self.dispersion), *bounds[-1])
        def unpack(theta):
            beta = theta[:width].copy()
            if split:
                beta[offset:] -= theta[width:-1]
            return beta

        def objective(theta):
            beta = unpack(theta)
            eta = design @ beta
            log_a = theta[-1]
            r = np.exp(-log_a)
            log_denom = np.logaddexp(0, eta + log_a)
            ll = gammaln(y+r)-gammaln(r)-gammaln(y+1) + y*(eta+log_a) - (y+r)*log_denom
            score_eta = y-(y+r)*expit(eta+log_a)
            score_log_a = r*(digamma(r)-digamma(y+r)+log_denom) + score_eta
            value = -ll.mean() + strength*(1-ratio)*np.dot(beta[offset:], beta[offset:])/2
            gradient = -(design.T @ score_eta)/len(y)
            gradient[offset:] += strength*(1-ratio)*beta[offset:]
            if split:
                value += strength*ratio*(theta[offset:width].sum()+theta[width:-1].sum())
                gradient = np.r_[gradient[:offset], gradient[offset:]+strength*ratio,
                                 -gradient[offset:]+strength*ratio]
            return value, np.r_[gradient, -score_log_a.mean()]

        result = minimize(objective, start, jac=True, method='L-BFGS-B', bounds=bounds,
                          options={'maxiter':self.max_iter, 'gtol':self.tol,
                                   'ftol':min(self.tol, 1e-6)**2, 'maxls':50})
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            raise ValueError('Joint Negative Binomial fitting produced a non-finite solution.')
        return unpack(result.x), float(np.exp(result.x[-1])), int(result.nit), bool(result.success)

    def _fit_penalized(self, model, width):
        """Minimize average negative log likelihood plus a slope-only penalty."""
        weights = np.full(width, self.alpha, dtype=float)
        if self.fit_intercept:
            weights[0] = 0.0
        ratio = {'l1': 1.0, 'l2': 0.0}.get(self.penalty, self.l1_ratio)
        if ratio == 0:
            # statsmodels 0.14's ridge shortcut ignores maxiter/tolerance and
            # discards optimizer status. Use its public likelihood and score
            # with scipy directly so the declared fitting controls are honored.
            from scipy.optimize import minimize

            def objective(params):
                return -model.loglike(params, scale=1) / model.nobs + np.dot(weights, params ** 2) / 2

            def gradient(params):
                return -model.score(params, scale=1) / model.nobs + weights * params

            result = minimize(objective, np.zeros(width), jac=gradient, method='BFGS',
                              options={'maxiter': self.max_iter, 'gtol': self.tol})
            return result.x, int(result.nit), bool(result.success)
        result = model.fit_regularized(alpha=weights, L1_wt=ratio, refit=False,
                                       maxiter=self.max_iter, cnvrg_tol=self.tol, zero_tol=1e-10)
        # Native coordinate descent reports convergence but no iteration count.
        return np.asarray(result.params), None, bool(result.converged)

    def predict(self, X):
        """Return the fitted mean or the most probable integer count.

        Mode uses the upper maximizer when two adjacent counts tie. Saved
        models predating this option retain mean prediction.
        """
        mean = self.predict_mean(X)
        if getattr(self, 'prediction_', 'mean') == 'mean':
            return mean
        if self.dispersion_ >= 1:
            return np.zeros_like(mean)
        return np.floor(mean * (1 - self.dispersion_))

    def predict_mean(self, X):
        """Return the fitted expected count irrespective of prediction choice."""
        check_is_fitted(self, ('coef_', 'intercept_'))
        X = validate_data(self, X, reset=False, dtype=np.float64)
        with np.errstate(over='ignore', invalid='ignore'):
            prediction = np.exp(self.intercept_ + X @ self.coef_)
        if not np.isfinite(prediction).all():
            raise ValueError('Negative Binomial prediction overflowed; inspect feature ranges and fitted coefficients.')
        return prediction
