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
    """Forecast accuracy, proper marginal scores and central 80% coverage.

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
            'Each probabilistic target is a marginal forecast. Cross-target dependence and automatic calibration are not claimed. Central intervals have nominal 80% coverage; discrete intervals can overcover.',
            'Possession is displayed in native percentage units. Its likelihood uses a mixed measure: point masses at 0/100 and a density on fractions in (0,1). NLL magnitudes across target types are not directly comparable.',
        ])
        forecasts = context.predictions.get('statistics')
        if forecasts is None:
            result.notes.append('No retained statistics predictions; enable StatisticsForecast when fitting.')
            return result
        scores, exports = [], []
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
                    nll[zero]=-np.log(p0[zero]);nll[one]=-np.log(p1[one])
                    nll[interior]=-np.log(pi[interior])-stats.beta.logpdf(y[interior],a[interior],b[interior])
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
        result.artifacts.insert(0,Artifact('table',result.tables['scores'],'Accuracy and marginal uncertainty'))
        return result
