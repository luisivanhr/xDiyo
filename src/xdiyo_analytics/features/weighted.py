"""Paired scalar historical reductions, sharing native eligible-match windows."""
from dataclasses import dataclass
from numbers import Integral
from math import fsum, log

from .expressions import Expr


@dataclass(frozen=True)
class RollingWeightedMean(Expr):
    """Average source values with explicit nonnegative per-observation weights.

    min_periods counts finite pairs with strictly positive weights. Missing and
    zero-weight rows occupy window slots. Optional availability fields name UTC
    datetime columns in team history; these mask pairs AFTER window selection.
    The ordinary available_at setting still defines fixture eligibility.
    """
    source: Expr
    weights: Expr
    window: int = 5
    min_periods: int = 1
    venue: str = "all"
    source_available_at: str | None = None
    weights_available_at: str | None = None

    def __post_init__(self):
        for name in ('window', 'min_periods'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        if self.min_periods > self.window:
            raise ValueError('min_periods cannot exceed window.')
        if self.venue not in ('all', 'same'):
            raise ValueError('Historical venue must be all or same.')
        for name in ('source_available_at', 'weights_available_at'):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f'{name} must name a datetime column or be None.')


def stable_mean(values, weights):
    """Avoid overflow from a common weight scale, including signed cancellation."""
    import numpy as np
    scale = float(np.max(np.abs(values)))
    if scale == 0:
        return 0.
    def exact_mean():
        # Every finite binary64 has an exact integer ratio. Accumulate products
        # and the denominator without intermediate rounding; convert only once.
        from fractions import Fraction
        pairs = [(Fraction(float(z)), Fraction(float(w))) for z, w in zip(values, weights)]
        return float(sum((z*w for z, w in pairs), Fraction(0)) / sum((w for _, w in pairs), Fraction(0)))

    nonzero = np.abs(values[values != 0])
    if max(scale, float(weights.max())) > 1e150 or min(float(nonzero.min()), float(weights.min())) < 1e-150:
        return exact_mean()
    with np.errstate(under='ignore'):
        w = weights / weights.max()
        terms = w * (values / scale)
    # The operands can each look moderate while their scaled product is
    # subnormal (or zero). Later cancellation can make that contribution matter.
    if np.any((values != 0) & (np.abs(terms) < np.finfo(float).tiny)):
        return exact_mean()
    numerator = fsum(terms)
    if np.any(values < 0) and np.any(values > 0) and abs(numerator) <= 64*np.finfo(float).eps*fsum(np.abs(terms)):
        return exact_mean()
    normalized = numerator / fsum(w)
    if numerator != 0 and abs(normalized) < np.finfo(float).tiny:
        return exact_mean()
    return float(np.clip(normalized, -1., 1.) * scale)


def weighted_values(history, source, weights, candidates, node, *, cutoffs, available_at, group_by, scope):
    import numpy as np
    import pandas as pd
    from .history import aligned_times
    from .spatial_lineage import compose_spatial_metadata, distinct_records, lineage_id, calculation_for

    if source.shape[1] != 1 or weights.shape[1] != 1:
        raise ValueError('RollingWeightedMean inputs must each select one output column; choose explicit period, side and field.')
    if not source.index.equals(history.index) or not weights.index.equals(history.index):
        raise ValueError('RollingWeightedMean inputs must align with the same history identities.')
    keys = [c for c in ('source_league', 'source_season', 'competition_id', 'season_id', 'event_id', 'team_id') if c in history]
    if 'event_id' not in keys or 'team_id' not in keys or history[keys].isna().any().any() or history.duplicated(keys).any():
        raise ValueError('RollingWeightedMean requires unique, nonmissing source/fixture/team identities.')
    identities = history[keys].to_dict('records')
    z = source.to_numpy(dtype=float, na_value=np.nan)[:, 0]
    w = weights.to_numpy(dtype=float, na_value=np.nan)[:, 0]
    times = aligned_times(history, cutoffs, default='kickoff_at')
    releases = [None if column is None else aligned_times(history, column, default='kickoff_at')
                for column in (node.source_available_at, node.weights_available_at)]
    # Compare preconverted arrays, preserving NaT as unavailable.
    cutoff_ns = times.astype('datetime64[ns, UTC]').astype('int64').to_numpy()
    availability = [(None, None) if r is None else
                    (r.astype('datetime64[ns, UTC]').astype('int64').to_numpy(), r.notna().to_numpy()) for r in releases]
    output = np.full(len(history), np.nan)
    audit = []
    definition = dict(operator='RollingWeightedMean', formula_version=2, expression=repr(node),
        window=int(node.window), min_periods=int(node.min_periods), venue=node.venue, h2h=bool(scope),
        paired_mask='finite_source_and_weight', minimum='strictly_positive_paired_weights',
        availability='input_masks_after_fixed_eligible_window',
        source_available_at=node.source_available_at, weights_available_at=node.weights_available_at,
        source=calculation_for(source, node.source), weights=calculation_for(weights, node.weights),
        units='source expression units', group_by=list(group_by),
        fixture_availability='kickoff_proxy' if available_at is None else 'explicit')
    source_spec = next(iter(source.attrs.get('spatial_features', {}).values()), {})
    weight_spec = next(iter(weights.attrs.get('spatial_features', {}).values()), {})
    definition['units'] = source_spec.get('units', 'native source units (not declared)')
    definition['weight_units'] = weight_spec.get('units', 'native weight units (not declared)')
    from hashlib import sha256
    evidence = history[[c for c in dict.fromkeys([*keys, *group_by, 'status', 'side', 'opponent_id']) if c in history]].copy()
    evidence['value'], evidence['weight'] = z, w
    evidence['kickoff'], evidence['cutoff'] = history.kickoff_at, times
    evidence['fixture_available'] = aligned_times(history, available_at, default='kickoff_at')
    for name, release in zip(('value_available', 'weight_available'), releases):
        if release is not None:
            evidence[name] = release
    digest = sha256(pd.util.hash_pandas_object(evidence, index=False).to_numpy().tobytes())
    digest.update(repr(history.attrs).encode())
    definition['input_sha256'] = digest.hexdigest()
    history_id = lineage_id(definition)
    for row, positions in enumerate(candidates):
        selected = positions[-node.window:]
        masks = [np.ones(len(selected), dtype=bool) if r is None else valid[selected] & (r[selected] <= cutoff_ns[row])
                 for r, valid in availability]
        a, b = z[selected], w[selected]
        if np.any(masks[1] & np.isfinite(b) & (b < 0)):
            raise ValueError(f'RollingWeightedMean has a finite negative weight in the available selected window for {identities[row]}. Choose an explicitly nonnegative weighting feature.')
        paired = masks[0] & masks[1] & np.isfinite(a) & np.isfinite(b)
        positive = paired & (b > 0)
        count = int(positive.sum())
        total, log_total, ess = 0., None, None
        if count:
            active = b[positive]
            scale = float(active.max())
            scaled = active / scale
            summed = fsum(scaled)
            log_total = log(scale) + log(summed)
            with np.errstate(over='ignore'):
                total = float(np.float64(scale) * summed)
            total = total if np.isfinite(total) else None
            ess = summed * summed / fsum(scaled * scaled)
            if count >= node.min_periods:
                output[row] = stable_mean(a[positive], active)
        audit.append({**identities[row], 'history_id':history_id,
            'eligible_matches':len(positions), 'window_matches':len(selected),
            'paired_finite_count':int(paired.sum()), 'positive_weight_count':count,
            'zero_weight_count':int((paired & (b == 0)).sum()),
            'missing_value_count':int((masks[0] & ~np.isfinite(a)).sum()),
            'missing_weight_count':int((masks[1] & ~np.isfinite(b)).sum()),
            'unavailable_value_count':int((~masks[0]).sum()),
            'unavailable_weight_count':int((~masks[1]).sum()),
            'total_weight':total, 'total_weight_overflow':total is None,
            'log_total_weight':log_total, 'effective_sample_size':ess,
            'usable_output':bool(np.isfinite(output[row]))})
    result = pd.DataFrame({'value':output}, index=history.index)
    compose_spatial_metadata(result, node, source, weights, expressions=(node.source, node.weights))
    for spec in result.attrs.get('spatial_features', {}).values():
        children = spec['calculation']
        spec['calculation'] = {**definition, 'source':children['left'], 'weights':children['right']}
        spec.update(field_label='Weighted historical value', units=definition['units'])
    if 'point_map_coverage' in result.attrs:
        result.attrs['point_history_coverage'] = distinct_records(result.attrs.get('point_history_coverage', []),
            [{**record, 'expression':repr(node), 'column':'value', 'usable_values':record['positive_weight_count']} for record in audit])
    result.attrs['weighted_history_audit'] = distinct_records(
        source.attrs.get('weighted_history_audit', []), weights.attrs.get('weighted_history_audit', []),
        [dict(definition=definition, records=audit)])
    return result
