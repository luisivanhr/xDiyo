"""Optional statistics supervision using native full-pass update cadence."""
from .statistics import labels, StatisticsScale
from .adapter import _torch


def initialize_statistics(backend, context):
    torch = _torch()
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
    denominators = (torch.isfinite(y)*weights[:, None]).sum(0)
    total_weight = weights.sum()
    replay = backend._replay(groups)
    rec = backend.config.heatmaps.encoder.reconstruction if backend.config.heatmaps is not None else None
    def one(group):
        positions, data, home, away, fixture = group
        ix = torch.as_tensor(positions, device=backend.device)
        logits, reconstruction, forecasts = backend.model(data, home, away, fixture,
            return_auxiliary=rec is not None, return_statistics=True)
        outcome = (torch.nn.functional.cross_entropy(logits, targets[ix], reduction='none')*weights[ix]).sum()/total_weight
        auxiliary = backend.model.statistics_head.losses(forecasts, y[ix], weights[ix], denominators)
        reconstruction = reconstruction*weights[ix].sum()/total_weight if rec is not None else outcome*0
        objective = outcome + cfg.weight*auxiliary + (rec.weight*reconstruction if rec is not None else 0)
        if backward:
            objective.backward()
        return torch.stack([outcome.detach(), auxiliary.detach(), reconstruction.detach()])
    scalars = [one(replay.group(groups, i) if replay is not None else groups[i]) for i in range(len(groups))]
    values = torch.stack(scalars).cpu().tolist() if scalars else []
    sums = [sum(row[i] for row in values) for i in range(3)]
    backend._statistics_loss, backend._reconstruction_loss = sums[1:]
    return sums[0]
