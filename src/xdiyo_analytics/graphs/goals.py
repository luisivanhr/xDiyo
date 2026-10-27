"""Strict fixture-level goal labels and optional marginal Poisson supervision."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .config import MATCH_KEYS
from .extensions import StatisticTarget
from .statistics import labels


def goal_labels(store, metadata, known_at, outcomes=None):
    """One row per fixture, home then away; reuse the availability-aware lookup."""
    if metadata.duplicated(list(MATCH_KEYS)).any():
        raise ValueError('Goal supervision requires unique fixture identities.')
    schema = SimpleNamespace(targets=tuple(StatisticTarget('goals', side=side, family='poisson')
                                          for side in ('home', 'away')))
    values = labels(store, metadata, schema, known_at)
    if not np.isfinite(values).all():
        raise ValueError('Every supervised fixture requires both available goal labels; no imputation or row removal.')
    indexed = store._history.loc[:, [*MATCH_KEYS, 'side', 'team_id', 'opponent_id']].set_index([*MATCH_KEYS, 'side'])
    keys = list(metadata[list(MATCH_KEYS)].itertuples(index=False, name=None))
    for side, own, opponent in (('home', 'home_id', 'away_id'), ('away', 'away_id', 'home_id')):
        wanted = pd.MultiIndex.from_tuples([(*key, side) for key in keys], names=indexed.index.names)
        rows = indexed.reindex(wanted)
        if (not np.array_equal(rows.team_id.to_numpy(), metadata[own].to_numpy()) or
                not np.array_equal(rows.opponent_id.to_numpy(), metadata[opponent].to_numpy())):
            raise ValueError('Goal-label home/away identities disagree with target fixtures.')
    if outcomes is not None and not np.array_equal(np.sign(values[:, 0]-values[:, 1]), np.asarray(outcomes)):
        raise ValueError('Goal labels disagree with the declared home/draw/away outcomes.')
    return values


def poisson_goal_loss(rates, targets, *, full=False):
    """Per-fixture SUM of two marginal losses; caller owns population reduction."""
    import torch
    if rates.ndim != 2 or rates.shape != targets.shape or rates.shape[1] != 2:
        raise ValueError('Goal rates and targets must have matching (fixtures, 2) shapes.')
    loss = rates - targets*rates.log()
    if full:
        loss = loss + torch.lgamma(targets+1)
    return loss.sum(dim=1)


def initialize_goals(backend, context):
    import torch
    def targets(ctx, boundary):
        return torch.as_tensor(goal_labels(backend.store, ctx.metadata, boundary, ctx.y.iloc[:, 0]),
                               dtype=torch.float32, device=backend.device)
    backend.train_goals = targets(context, backend.fitting_boundary)
    backend.validation_goals = (targets(context.validation, backend.trained_through)
                                if context.validation is not None else None)
    if backend.config.goal_auxiliary.architecture == 'attack_defence':
        competitions = torch.as_tensor(context.metadata.competition_id.to_numpy(), device=backend.device)
        backend.goal_initialization_events = backend.model.goal_head.initialize_means(competitions, backend.train_goals)


def log1p_ratio(x):
    """Stable log(1+x)/x and its derivative, including the Poisson boundary."""
    import torch
    small = x.abs() < 1e-4
    safe = torch.where(small, torch.ones_like(x), x)
    z = torch.where(small, x, torch.zeros_like(x))
    series = 1 - z/2 + z*z/3 - z*z*z/4 + z**4/5
    return torch.where(small, series, safe.log1p()/safe)


def count_goal_loss(log_rates, targets, likelihood='poisson', alpha=None):
    """Full proper per-fixture NLL; NB sums two sides, G is one joint loss."""
    import torch
    eta, y = log_rates.double(), targets.double()
    if eta.ndim != 2 or eta.shape != y.shape or eta.shape[1] != 2:
        raise ValueError('Expected aligned (fixtures, 2) log rates and counts.')
    if not torch.isfinite(y).all() or (y < 0).any() or (y != y.floor()).any():
        raise ValueError('Count likelihoods require raw finite nonnegative integer labels.')
    rates = eta.exp()
    constant = torch.lgamma(y+1) - y*eta
    if likelihood == 'poisson':
        result = (rates + constant).sum(1)
    else:
        if likelihood not in ('independent_nb', 'shared_gamma'):
            raise ValueError('Unknown count likelihood.')
        if alpha is None or not torch.isfinite(alpha) or alpha < 0:
            raise ValueError('Dispersion alpha must be finite and nonnegative.')
        means = rates.sum(1, keepdim=True) if likelihood == 'shared_gamma' else rates
        counts = y.sum(1, keepdim=True) if likelihood == 'shared_gamma' else y
        x = alpha * means
        terms = means*log1p_ratio(x) + counts*torch.log1p(x)
        r = torch.arange(int(counts.max().item()) if counts.numel() else 0, device=eta.device, dtype=torch.float64)
        rising = (torch.log1p(alpha*r) * (r < counts.unsqueeze(-1))).sum(-1)
        result = constant.sum(1) + (terms-rising).sum(1)
    if not torch.isfinite(result).all():
        raise FloatingPointError('Nonfinite count likelihood; no score or rate clipping.')
    return result


def auxiliary_goal_loss(model, details, targets):
    if model.goal_config.architecture == 'linear':
        return poisson_goal_loss(details['rates'], targets)
    return count_goal_loss(details['log_rates'], targets, model.goal_config.likelihood, details.get('alpha'))


def goal_loss(backend, groups, targets, weights, backward):
    """Native global-weight accumulation; beta=0 excludes the unused graph."""
    import torch
    if groups is backend.train_groups:
        goals = backend.train_goals
    elif groups is backend.validation_groups:
        goals = backend.validation_goals
    else:
        raise ValueError('Goal labels missing for this loss population.')
    if goals is None:
        raise ValueError('Goal labels missing for this loss population.')
    total_weight = weights.sum()
    def one(group):
        positions, data, home, away, fixture = group
        ix = torch.as_tensor(positions, device=backend.device)
        if backend.config.goal_auxiliary.architecture == 'linear':
            logits, rates = backend.model(data, home, away, fixture, return_goals=True)
            per_goal = poisson_goal_loss(rates, goals[ix])
        else:
            logits, details = backend.model(data, home, away, fixture, return_goal_details=True)
            per_goal = auxiliary_goal_loss(backend.model, details, goals[ix])
        outcome = (torch.nn.functional.cross_entropy(logits, targets[ix], reduction='none')*weights[ix]).sum()/total_weight
        auxiliary = (per_goal*weights[ix]).sum()/total_weight
        if backward:
            # Avoid even zero goal gradients / optimizer state in the unused arm.
            objective = outcome if backend.config.goal_auxiliary.beta == 0 else outcome + backend.config.goal_auxiliary.beta*auxiliary
            objective.backward()
        return outcome.detach(), auxiliary.detach()
    scalars = []
    for i in range(len(groups)):
        with backend._group_lease(groups, i) as (group, _):
            scalars.extend(one(group))
    values = torch.stack(scalars).cpu().tolist() if scalars else []
    # Preserve the original sequential accumulation of float32 group losses.
    backend._goal_loss = sum(values[1::2])
    return sum(values[::2])
