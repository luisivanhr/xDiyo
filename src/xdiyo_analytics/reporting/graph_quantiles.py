"""Held-out quantile diagnostics, with no interpolated or invented summaries."""
import numpy as np
import pandas as pd
from ..graphs.extensions import quantile_column, NumericStatisticTarget
from ..graphs.statistics import transform_target
from .studies import _plotting, _style


def diagnose_quantiles(target, predicted, y, fold, index, scale):
    factor = 100 if target.proportion else 1
    tau = np.asarray(target.quantiles)
    columns = [quantile_column(target, q) for q in tau]
    q = predicted[columns].to_numpy(dtype=float)
    if not np.isfinite(q).all() or (np.diff(q, axis=1) < 0).any():
        raise ValueError('Retained quantile forecasts must be finite and nondecreasing.')
    generic = isinstance(target, NumericStatisticTarget)
    invalid = ((q < 0).any() or (target.proportion and (q > 100).any())) if not generic else (
        (target.support in ('count','nonnegative_continuous') and (q < 0).any()) or
        (target.support == 'bounded_continuous' and ((q < target.lower).any() or (q > target.upper).any())))
    if invalid:
        raise ValueError('Retained quantiles violate target support.')
    valid = np.isfinite(y)
    observed = y*factor
    residual = observed[valid, None]-q[valid]
    pinball = np.maximum(tau*residual, (tau-1)*residual)
    transformed_y = transform_target(target, y[valid])
    transformed_q = transform_target(target, q[valid]/factor)
    error = (transformed_y[:, None]-transformed_q)/scale
    normalized = np.maximum(tau*error, (tau-1)*error)
    table = pd.DataFrame(dict(fold_id=fold, target=target.name, observed=observed), index=index)
    levels = []
    for j, level in enumerate(tau):
        table['quantile::'+repr(float(level))] = q[:, j]
        levels.append(dict(fold_id=fold,target=target.name,level=level,n=int(valid.sum()),
            pinball=float(pinball[:,j].mean()) if valid.any() else np.nan,
            training_space_pinball=float(normalized[:,j].mean()) if valid.any() else np.nan,
            empirical_coverage=float((observed[valid]<=q[valid,j]).mean()) if valid.any() else np.nan))
    score = dict(fold_id=fold,target=target.name,family='quantile',n=int(valid.sum()),missing=int((~valid).sum()),
        pinball=float(pinball.mean()) if valid.any() else np.nan,
        training_space_pinball=float(normalized.mean()) if valid.any() else np.nan)
    if .5 in target.quantiles:
        table['median'] = q[:, target.quantiles.index(.5)]
    title = target.name + f' · Fold {fold}'
    if len(tau) > 1:
        central = 'central ' if np.isclose(tau[0]+tau[-1],1,rtol=0,atol=1e-12) else ''
        label = f'Nominal {central}{100*(tau[-1]-tau[0]):g}% quantile interval'
        table['interval_label'] = label
        table['interval_lower'],table['interval_upper'] = q[:,0],q[:,-1]
        score.update(interval_label=label,interval_coverage=float(((observed[valid]>=q[valid,0]) & (observed[valid]<=q[valid,-1])).mean()) if valid.any() else np.nan)
        title += ' · '+label
    go = _plotting(); fig = go.Figure()
    x = np.arange(len(y))
    for j, level in enumerate(tau):
        fig.add_trace(go.Scatter(x=x,y=q[:,j],mode='lines',name=f'Quantile {level:g}'))
    fig.add_trace(go.Scatter(x=x[valid],y=observed[valid],mode='markers',name='Observed',marker=dict(color='#268bd2')))
    return table, score, levels, _style(fig,title,'Fixture order','Native statistic units')
