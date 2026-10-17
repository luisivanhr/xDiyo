"""Optional fixture execution; native full-pass objectives and update cadence.

Dependency selection is mathematically exact. Different matrix shapes and
reduction orders can change float32 gradients and AdamW trajectories.
"""
from dataclasses import asdict


def execution_metadata(config):
    return None if config.execution is None else asdict(config.execution)


def fallback_reason(backend):
    cfg = backend.config
    if cfg.execution is None or cfg.execution.mode == 'native':
        return 'native_requested'
    if cfg.heatmaps is not None:
        return 'unsupported_cnn'
    if cfg.boundary_memory is not None:
        return 'unsupported_boundary_memory'
    if cfg.statistics_forecast is not None and any(t.family != 'quantile' for t in cfg.statistics_forecast.targets):
        return 'unsupported_statistics_family'
    if getattr(getattr(backend, 'input_cache_policy', None), 'max_bytes', 0):
        return 'input_residency_enabled'
    return None


def enabled(backend):
    reason = fallback_reason(backend)
    if reason is not None and backend.config.execution is not None:
        backend._execution_trace = dict(requested_mode=backend.config.execution.mode,
            fallback_reasons={reason: 1}, whole_pass_native=True)
    return reason is None


def _observed_batch(observations, indices, lengths):
    """Reindex fixed label selections, retaining full-population denominators."""
    import torch
    if len(indices) == 1:
        return observations[indices[0]]
    result = []
    for target in range(len(observations[indices[0]])):
        selected, values, weights = [], [], []
        offset = 0
        for index, length in zip(indices, lengths):
            local, y, w = observations[index][target]
            selected.append(local + offset)
            values.append(y)
            weights.append(w)
            offset += length
        result.append((torch.cat(selected), torch.cat(values), torch.cat(weights)))
    return tuple(result)


def dependency_loss(backend, groups, targets, weights, backward):
    import torch
    from .packing import stream_batches
    from .supervision import prepare_observations, observations_have_labels
    cfg = backend.config.statistics_forecast
    total_weight = weights.sum()
    observations = None
    if cfg is not None:
        labels = backend.train_statistics if groups is backend.train_groups else backend.validation_statistics
        if labels is None:
            raise ValueError('Statistics labels missing for this loss population.')
        cache = backend._statistics_observations
        if id(groups) not in cache:
            cache[id(groups)] = prepare_observations(groups, labels, weights)
        denominators, observations = cache[id(groups)]

    def one(group, indices):
        if cfg is None:
            return torch.stack(backend._group_loss(group, targets, weights, backward, total_weight))
        positions, data, home, away, fixture = group
        ix = torch.as_tensor(positions, device=backend.device)
        logits, _, forecasts = backend.model(data, home, away, fixture, return_statistics=True)
        outcome = (torch.nn.functional.cross_entropy(logits, targets[ix], reduction='none') * weights[ix]).sum()/total_weight
        observed = _observed_batch(observations, indices, data.get('_group_lengths', (len(positions),)))
        auxiliary = backend.model.statistics_head.losses(forecasts, None, None, denominators, prepared=observed)
        objective = outcome + cfg.weight * auxiliary
        if backward:
            objective.backward()
        return torch.stack((outcome.detach(), auxiliary.detach()))

    scalars = []
    for group, indices, _ in stream_batches(backend, groups, backward=backward):
        scalars.append(one(group, indices))
        # Release inputs before requesting the next admitted pack. one() has
        # already released forecasts, losses and their autograd graph.
        del group
    values = torch.stack(scalars).cpu().tolist() if scalars else []
    if cfg is not None:
        backend._statistics_loss = sum(row[1] for row in values)
        if not backward and not observations_have_labels(observations):
            backend._statistics_loss = float('nan')
    return sum(row[0] for row in values)
