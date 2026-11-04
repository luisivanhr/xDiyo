"""Structured auxiliary count models. No count output enters the classifier."""
import torch
from torch import nn


class AttackDefenceHead(nn.Module):
    def __init__(self, config, dimensions):
        super().__init__()
        self.config = config
        self.attack = nn.Linear(dimensions, 1, bias=False)
        self.defence = nn.Linear(dimensions, 1, bias=False)
        nn.init.zeros_(self.attack.weight)
        nn.init.zeros_(self.defence.weight)
        self.scoring = nn.Parameter(torch.zeros(len(config.competitions)))
        self.home_effect = nn.Parameter(torch.zeros(len(config.competitions)))
        self.register_buffer('centering_offsets', torch.zeros(2, dtype=torch.float64))
        if config.likelihood != 'poisson':
            # float64 nuisance likelihood arithmetic, including its scalar parameter.
            alpha = torch.tensor(config.initial_alpha, dtype=torch.float64)
            self.raw_alpha = nn.Parameter(torch.log(torch.expm1(alpha)))
        else:
            self.register_parameter('raw_alpha', None)

    def initialize_means(self, competitions, counts):
        events = []
        with torch.no_grad():
            for i, competition in enumerate(self.config.competitions):
                selected = counts[competitions == competition].double()
                if not len(selected):
                    raise ValueError('A supported goal competition has no fitting labels.')
                means = selected.mean(0)
                for side in range(2):
                    if means[side] == 0:
                        means[side] = 1e-6
                        events.append(dict(competition=competition, side=side, initialization_floor=1e-6))
                self.scoring[i] = means[1].log()
                self.home_effect[i] = means[0].log() - means[1].log()
        return events

    def forward(self, home, away, competition, neutral):
        indices = torch.full_like(competition, -1)
        for index, identity in enumerate(self.config.competitions):
            indices = torch.where(competition == identity, index, indices)
        if (indices < 0).any():
            raise ValueError('Unsupported target competition for structured goal predictions.')
        a = torch.cat((self.attack(home), self.attack(away)), dim=1)
        d = torch.cat((self.defence(home), self.defence(away)), dim=1)
        mu, gamma = self.scoring[indices], self.home_effect[indices]
        # Unknown neutrality uses the ordinary home term, as declared by the suite.
        home_term = (neutral != 1).to(gamma.dtype) * gamma
        eta = torch.stack((mu + home_term + a[:, 0] - d[:, 1],
                           mu + a[:, 1] - d[:, 0]), dim=1)
        rates = eta.double().exp()
        if not torch.isfinite(rates).all() or (rates <= 0).any():
            raise FloatingPointError('Structured rates are not finite and positive; no clipping is applied.')
        result = dict(rates=rates, log_rates=eta.double(), attack=a, defence=d,
                      scoring=mu, home_effect=gamma, neutral=neutral)
        if self.raw_alpha is not None:
            alpha = torch.nn.functional.softplus(self.raw_alpha)
            if not torch.isfinite(alpha) or alpha <= 0:
                raise FloatingPointError('Dispersion is not finite and positive.')
            result.update(alpha=alpha, kappa=alpha.reciprocal())
        return result


def optimizer_groups(model, config):
    """Retain the native common AdamW group; intercept/dispersion exceptions only."""
    goal = config.goal_auxiliary
    if goal is None or goal.architecture == 'linear':
        return model.parameters()
    intercepts = ('goal_head.scoring', 'goal_head.home_effect')
    parameters = dict(model.named_parameters())
    common = [p for name, p in parameters.items() if name not in (*intercepts, 'goal_head.raw_alpha')]
    groups = [dict(params=common, lr=config.learning_rate, weight_decay=config.weight_decay),
              dict(params=[parameters[n] for n in intercepts], lr=config.learning_rate, weight_decay=0.)]
    if model.goal_head.raw_alpha is not None:
        groups.append(dict(params=[model.goal_head.raw_alpha], lr=goal.dispersion_learning_rate, weight_decay=0.))
    return groups
