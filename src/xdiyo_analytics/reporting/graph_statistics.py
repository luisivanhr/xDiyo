"""Native held-out marginal statistic diagnostics, using retained predictions."""
from dataclasses import dataclass
from typing import ClassVar
import numpy as np
import pandas as pd
from .post_training import PredictionReporter
from .contracts import StudyResult, Artifact
from .graphs import _graph
from .studies import _plotting, _style


@dataclass(kw_only=True)
class GraphStatisticsReporter(PredictionReporter):
    """Forecast accuracy, marginal scores, intervals and quantile coverage.

    Actual statistics are read only for held-out scoring, never for prediction.
    No recalibration, sampling, fitting or joint-distribution claim is made.
    """
    supported_types: ClassVar[tuple] = ('per_fold', 'overall')

    def run(self, context):
        from scipy import stats
        from ..graphs.statistics import labels
        if context.pooling == 'mean':
            raise ValueError('Statistics distributions cannot be pooled by averaging their parameters.')
        result = StudyResult('Upcoming match-statistics forecasts', notes=[
            'Forecasts use pre-match inputs. Observed statistics are used only for held-out diagnostics.',
            'Each probabilistic target is marginal. Cross-target dependence and automatic calibration are not claimed. Parametric central intervals have nominal 80% coverage; quantile intervals use their configured endpoint levels. Discrete intervals can overcover.',
            'Possession is displayed in native percentage units. Its likelihood uses a mixed measure: point masses at 0/100 and a density on fractions in (0,1). NLL magnitudes across target types are not directly comparable.',
        ])
        forecasts = context.predictions.get('statistics')
        if forecasts is None:
            result.notes.append('No retained statistics predictions; enable StatisticsForecast when fitting.')
            return result
        scores, exports, quantile_scores, quantile_objectives = [], [], [], []
        go = _plotting()
        for fold, meta in context.metadata.groupby(level='fold_id', sort=False):
            model = _graph(context.models.get(fold))
            if model is None or model.config.statistics_forecast is None:
                continue
            cfg = model.config.statistics_forecast
            actual = labels(model.event_store, meta, cfg, pd.Timestamp.max.tz_localize('UTC'))
            predicted = forecasts.loc[meta.index]
            for i, target in enumerate(cfg.targets):
                name = target.name
                y = actual[:, i]; valid = np.isfinite(y)
                factor = 100 if target.proportion else 1
                if target.family == 'quantile':
                    from .graph_quantiles import diagnose_quantiles
                    table, score, levels, fig = diagnose_quantiles(target, predicted, y, fold, meta.index, model.backend_.statistics_scale.scale[i])
                    largest = max(t.weight for t in cfg.targets)
                    relative = (target.weight/largest)/sum(t.weight/largest for t in cfg.targets)
                    quantile_objectives.append(dict(fold_id=fold,target=target.name,weighted_component=cfg.weight*relative*(score['training_space_pinball'] if score['n'] else np.nan)))
                    exports.append(table); scores.append(score); quantile_scores.extend(levels)
                    result.artifacts.append(Artifact('plotly', fig, name))
                    continue
                mu = predicted[name+'::mean'].to_numpy()/factor
                if not np.isfinite(mu).all():
                    raise ValueError('Nonfinite retained statistics forecast.')
                lo = np.full(len(y), np.nan); hi = lo.copy(); nll = lo.copy()
                if target.family in ('poisson', 'negative_binomial'):
                    if target.family == 'poisson':
                        distribution = stats.poisson(mu)
                    else:
                        d = predicted[name+'::dispersion'].to_numpy()
                        distribution = stats.nbinom(d, d/(d+mu))
                    lo,hi = distribution.ppf(.1),distribution.ppf(.9)
                    nll[valid] = -distribution.logpmf(np.where(valid,y,0))[valid]
                elif target.family == 'inflated_beta':
                    a,b = (predicted[name+'::'+field].to_numpy() for field in ('a','b'))
                    p0,p1,pi = (predicted[name+'::'+field].to_numpy() for field in ('p_zero','p_one','p_interior'))
                    for q,out in ((.1,lo),(.9,hi)):
                        out[:] = np.where(q<=p0,0,np.where(q>=1-p1,1,stats.beta.ppf(np.clip((q-p0)/np.maximum(pi,np.finfo(float).tiny),0,1),a,b)))
                    zero,one = valid & (y==0),valid & (y==1)
                    interior = valid & (y>0) & (y<1)
                    logkeys = [name+'::log_p_'+field for field in ('zero','one','interior')]
                    if all(key in predicted for key in logkeys):
                        l0,l1,li = (predicted[key].to_numpy() for key in logkeys)
                    else:  # Legacy retained predictions: do not invent a clipped likelihood.
                        with np.errstate(divide='ignore'):
                            l0,l1,li = np.log(p0),np.log(p1),np.log(pi)
                        result.notes.append('Legacy mixture outputs lack log masses; an underflowed probability cannot recover its finite proper score.')
                    nll[zero]=-l0[zero];nll[one]=-l1[one]
                    nll[interior]=-li[interior]-stats.beta.logpdf(y[interior],a[interior],b[interior])
                table = pd.DataFrame(dict(fold_id=fold,target=name,observed=y*factor,predicted=mu*factor,
                    lower_10=lo*factor,upper_90=hi*factor,nll=nll),index=meta.index)
                exports.append(table)
                scores.append(dict(fold_id=fold,target=name,family=target.family,n=int(valid.sum()),missing=int((~valid).sum()),
                    mae=float(np.mean(np.abs(y[valid]-mu[valid]))*factor) if valid.any() else np.nan,
                    nll=float(nll[valid].mean()) if valid.any() and target.family!='point' else np.nan,
                    coverage_80=float(((y[valid]>=lo[valid]) & (y[valid]<=hi[valid])).mean()) if valid.any() and target.family!='point' else np.nan))
                fig=go.Figure(go.Scatter(x=table.predicted[valid],y=table.observed[valid],mode='markers',name=name,
                                       marker=dict(color='#268bd2')))
                result.artifacts.append(Artifact('plotly',_style(fig,f'{name} · Fold {fold}','Forecast mean','Observed'),name))
        result.tables['forecasts']=pd.concat(exports) if exports else pd.DataFrame()
        result.tables['scores']=pd.DataFrame(scores)
        if quantile_scores:
            result.tables['quantile_scores'] = pd.DataFrame(quantile_scores)
            result.tables['quantile_objective'] = pd.DataFrame(quantile_objectives).groupby('fold_id',sort=False).weighted_component.sum(min_count=1).rename('heldout_equal_row_quantile_objective').reset_index()
            result.artifacts.append(Artifact('table',result.tables['quantile_objective'],'Held-out quantile objective component (equal observed-row weights)'))
            result.artifacts.append(Artifact('table',result.tables['quantile_scores'],'Marginal quantile scores and coverage'))
            result.notes.append('Quantile pinball scores use original units and equal observed-row weights; training-space scores use the saved training-only transform/scale. Training curves retain the actual weighted combined objective. Discrete ties can make P(Y <= quantile) differ from its level. Quantiles are marginal, not a joint distribution. A median is shown only if 0.5 was requested; it is not a mean.')
        result.artifacts.insert(0,Artifact('table',result.tables['scores'],'Accuracy and marginal uncertainty'))
        return result
