"""Binary margin response contract and single Platt-smoothed sigmoid."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit


def validate_margins(frame):
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique or not len(frame.columns):
        raise ValueError('Decision margins require distinct nonempty target columns.')
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError('Decision margins must be finite.')
    schema = frame.attrs.get('response_schema', {})
    if list(schema) != list(frame.columns):
        raise ValueError('Decision margins need fitted binary class metadata for every target.')
    for info in schema.values():
        classes = info.get('classes', [])
        if len(classes) != 2 or len(pd.Index(classes).unique()) != 2 or pd.isna(classes).any():
            raise ValueError('Decision margins require two distinct nonmissing fitted classes.')
        if info.get('positive_class') != classes[1]:
            raise ValueError('Positive margins must identify fitted classes_[1].')
    return schema


def fit_sigmoid(margins, positive):
    """Mean binary log loss with Platt targets, no regularizer or sample weights."""
    scale = float(np.max(np.abs(margins)))
    if not scale or np.ptp(margins / scale) == 0:
        raise ValueError('Constant/degenerate decision margins cannot be calibrated.')
    x = margins / scale
    n1 = int(positive.sum())
    n0 = len(positive) - n1
    y = np.where(positive == 1, (n1+1)/(n1+2), 1/(n0+2))
    def objective(theta):
        z = theta[0]*x + theta[1]
        loss = np.mean(y*np.logaddexp(0, -z) + (1-y)*np.logaddexp(0, z))
        residual = expit(z) - y
        return loss, np.array([np.mean(residual*x), np.mean(residual)])
    options = dict(maxiter=1000, ftol=1e-12, gtol=1e-10)
    fitted = minimize(objective, [0., 0.], jac=True, method='L-BFGS-B', options=options)
    if not fitted.success or not np.isfinite(fitted.x).all() or not np.isfinite(fitted.fun):
        raise RuntimeError(f'Margin sigmoid optimization failed: {fitted.message}')
    a, b = map(float, fitted.x)
    if not np.isfinite(a / scale):
        raise ValueError('Degenerate margin scale produces a nonfinite sigmoid slope.')
    return (a, b, scale), dict(objective='platt_smoothed_binary_log_loss_v1',
                              optimizer='L-BFGS-B', options=options, converged=True,
                              iterations=int(fitted.nit), loss=float(fitted.fun),
                              a=a/scale, b=b, scaled_a=a, margin_scale=scale,
                              positive_count=n1, negative_count=n0)
