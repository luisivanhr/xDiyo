"""Optional statistics supervision using native full-pass update cadence."""
from .statistics import labels, StatisticsScale
from .adapter import _torch


def initialize_statistics(backend, context):
    torch = _torch()
    backend._statistics_observations = {}  # Fresh fold/reinitialize owns fresh fixed labels.
    cfg = backend.config.statistics_forecast
    y = labels(backend.store, context.metadata, cfg, backend.fitting_boundary)
    backend.statistics_scale = StatisticsScale(cfg).fit(y)
    backend.train_statistics = torch.as_tensor(y, dtype=torch.float32, device=backend.device)
    backend.validation_statistics = None
    if context.validation is not None:
        v = labels(backend.store, context.validation.metadata, cfg, backend.trained_through)
        backend.validation_statistics = torch.as_tensor(v, dtype=torch.float32, device=backend.device)
    h = backend.model.statistics_head
    h.center.copy_(torch.as_tensor(backend.statistics_scale.mean, dtype=torch.float32, device=backend.device))
    h.scale.copy_(torch.as_tensor(backend.statistics_scale.scale, dtype=torch.float32, device=backend.device))
    counts = (torch.isfinite(backend.train_statistics)*backend.train_weights[:, None]).sum(0)
    if bool((counts == 0).any()):
        raise ValueError('Every requested statistics target requires positive-weight training observations.')


def statistics_loss(backend, groups, targets, weights, backward):
    torch = _torch()
    cfg = backend.config.statistics_forecast
    y = backend.train_statistics if groups is backend.train_groups else backend.validation_statistics
    if y is None:
        raise ValueError('Statistics labels missing for this loss population.')
    # Only fixed label membership/weights are cached, never learned forecasts.
    # Integer gathers avoid repeated CUDA boolean-index nonzero synchronizations.
    cache = getattr(backend, '_statistics_observations', None)
    if cache is None:
        cache = backend._statistics_observations = {}
    key = id(groups)
    if key not in cache:
        cache[key] = prepare_observations(groups, y, weights)
    denominators, observations = cache[key]
    total_weight = weights.sum()
    replay = backend._replay(groups)
    rec = backend.config.heatmaps.encoder.reconstruction if backend.config.heatmaps is not None else None
    def one(group, observed):
        positions, data, home, away, fixture = group
        ix = torch.as_tensor(positions, device=backend.device)
        logits, reconstruction, forecasts = backend.model(data, home, away, fixture,
            return_auxiliary=rec is not None, return_statistics=True)
        outcome = (torch.nn.functional.cross_entropy(logits, targets[ix], reduction='none')*weights[ix]).sum()/total_weight
        auxiliary = backend.model.statistics_head.losses(forecasts, None, None, denominators, prepared=observed)
        reconstruction = reconstruction*weights[ix].sum()/total_weight if rec is not None else outcome*0
        objective = outcome + cfg.weight*auxiliary + (rec.weight*reconstruction if rec is not None else 0)
        if backward:
            objective.backward()
        return torch.stack([outcome.detach(), auxiliary.detach(), reconstruction.detach()])
    scalars = [one(replay.group(groups, i) if replay is not None else groups[i], observations[i]) for i in range(len(groups))]
    values = torch.stack(scalars).cpu().tolist() if scalars else []
    sums = [sum(row[i] for row in values) for i in range(3)]
    backend._statistics_loss, backend._reconstruction_loss = sums[1:]
    return sums[0]


def prepare_observations(groups, labels, weights):
    """Bounded O(rows * targets) label-only cache for this initialized fit.

    Build masks on CPU once. Preserve the original device reduction for weight
    denominators and original row order; no altered summation or learned cache.
    """
    import numpy as np
    torch = _torch()
    denominator = (torch.isfinite(labels)*weights[:, None]).sum(0)
    values, positive = labels.detach().cpu().numpy(), weights.detach().cpu().numpy() > 0
    groups_out = []
    for _, positions in groups.plan.descriptors:
        observed = []
        for i in range(labels.shape[1]):
            selected = np.flatnonzero(np.isfinite(values[positions,i]) & positive[positions])
            local = torch.as_tensor(selected, device=labels.device)
            global_rows = torch.as_tensor(positions[selected], device=labels.device)
            observed.append((local, labels[global_rows,i], weights[global_rows]))
        groups_out.append(tuple(observed))
    return denominator, tuple(groups_out)
