"""Differentiable marginal statistics forecasts and strictly masked objectives."""
import torch
from torch import nn
from torch.nn import functional as F
from .extensions import NumericStatisticTarget
from .statistics import transform_target


class StatisticsHead(nn.Module):
    def __init__(self, config, input_dim, width):
        super().__init__()
        self.config = config
        self.sizes = [len(t.quantiles) if t.family == 'quantile' else 5 if t.family == 'inflated_beta' else 2 if t.family == 'negative_binomial' else 1 for t in config.targets]
        self.head = nn.Sequential(nn.Linear(input_dim, width), nn.ReLU(), nn.Linear(width, sum(self.sizes)))
        self.register_buffer('center', torch.zeros(len(config.targets)))
        self.register_buffer('scale', torch.ones(len(config.targets)))
        for i, target in enumerate(config.targets):
            if target.family == 'quantile':
                self.register_buffer(f'levels_{i}', torch.tensor(target.quantiles), persistent=False)

    @property
    def summary_dim(self):
        return sum(len(t.quantiles) if t.family == 'quantile' else 1 if t.family == 'point' else 2 for t in self.config.targets)

    def forward(self, x):
        raw = self.head(x).split(self.sizes, dim=1)
        forecasts, summaries = [], []
        for i, (t, r) in enumerate(zip(self.config.targets, raw)):
            if t.family == 'quantile':
                # Order-statistic parameterization is used during TRAINING too.
                # Ties are permitted; no forced positive increments or rounding.
                ordered = r.sort(dim=1).values
                q = ordered.sigmoid() if t.proportion else F.softplus(ordered)
                if isinstance(t, NumericStatisticTarget):
                    if t.support == 'signed_continuous':
                        q = ordered
                    elif t.support == 'bounded_continuous':
                        q = t.lower+(t.upper-t.lower)*ordered.sigmoid()
                transformed = transform_target(t, q)
                summaries.extend(((transformed-self.center[i])/self.scale[i]).unbind(1))
                forecasts.append(dict(quantiles=q))
                continue
            if t.family == 'inflated_beta':
                mixture = r[:, :3].softmax(1)
                a, b = F.softplus(r[:, 3])+1e-4, F.softplus(r[:, 4])+1e-4
                interior = a/(a+b)
                mean = mixture[:, 1] + mixture[:, 2]*interior
                second = mixture[:, 1] + mixture[:, 2]*(a*(a+1)/((a+b)*(a+b+1)))
                std = (second-mean.square()).clamp_min(1e-12).sqrt()
                params = dict(log_mixture=r[:, :3].log_softmax(1), a=a, b=b)
            else:
                mean = r[:, 0].sigmoid() if t.proportion else F.softplus(r[:, 0])+1e-6
                params = {}
                if t.family == 'negative_binomial':
                    dispersion = F.softplus(r[:, 1])+1e-4
                    params['dispersion'] = dispersion
                    std = (mean+mean.square()/dispersion).sqrt()
                elif t.family == 'poisson':
                    std = mean.sqrt()
                else:
                    std = None
            transformed = mean if t.proportion else mean.log1p()
            summaries.append((transformed-self.center[i])/self.scale[i])
            if std is not None:
                summaries.append(std/self.scale[i] if t.proportion else std.log1p()/self.scale[i])
            forecasts.append(dict(mean=mean, std=std, **params))
        return forecasts, torch.stack(summaries, dim=1)

    def losses(self, forecasts, targets, weights, denominators, *, prepared=None):
        # Each target: weighted sum over observed rows / its full-population
        # observed weight. Then a convex weighted average across targets.
        result = next(iter(forecasts[0].values()))[:0].sum()
        largest = max(t.weight for t in self.config.targets)
        total = sum(t.weight/largest for t in self.config.targets)
        for i, (t, f) in enumerate(zip(self.config.targets, forecasts)):
            if prepared is None:
                valid = torch.isfinite(targets[:, i]) & (weights > 0)
                y, w = targets[valid, i], weights[valid]
            else:
                valid, y, w = prepared[i]
            if t.family == 'quantile':
                q = f['quantiles'][valid]
                observed = transform_target(t, y)
                predicted = transform_target(t, q)
                residual = (observed[:, None]-predicted)/self.scale[i]
                tau = getattr(self, f'levels_{i}')
                loss = torch.maximum(tau*residual, (tau-1)*residual).mean(1)
            elif t.family == 'point':
                mu = f['mean'][valid]
                observed = transform_target(t, y)
                predicted = mu if t.proportion else mu.log1p()
                observed = (observed-self.center[i])/self.scale[i]
                predicted = (predicted-self.center[i])/self.scale[i]
                loss = F.huber_loss(predicted, observed, reduction='none') if self.config.point_loss == 'huber' else (predicted-observed).square()
            elif t.family == 'poisson':
                loss = -torch.distributions.Poisson(f['mean'][valid], validate_args=False).log_prob(y)
            elif t.family == 'negative_binomial':
                d = f['dispersion'][valid]
                loss = -torch.distributions.NegativeBinomial(d, logits=f['mean'][valid].log()-d.log(), validate_args=False).log_prob(y)
            else:
                logmix = f['log_mixture'][valid]
                # Select interior before logarithms: endpoints never enter Beta.
                loss = torch.empty_like(y)
                zero, one = y == 0, y == 1
                interior = ~(zero | one)
                loss[zero] = -logmix[zero, 0]
                loss[one] = -logmix[one, 1]
                beta = torch.distributions.Beta(f['a'][valid][interior], f['b'][valid][interior], validate_args=False)
                loss[interior] = -logmix[interior, 2]-beta.log_prob(y[interior])
            result = result + (t.weight/largest)/total * (loss*w).sum()/denominators[i].clamp_min(torch.finfo(w.dtype).tiny)
        return result
