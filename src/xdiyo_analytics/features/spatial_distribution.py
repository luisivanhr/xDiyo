"""Deterministic reductions of fixed, unit-mass spatial grids."""
from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


def validate_distribution_source(history, points, spec):
    """Strict identity checks for new reducers; legacy Heatmap stays compatible."""
    from .point_geometry import validate_point_identities
    keys = validate_point_identities(history, points)
    if spec.kinds is not None and (not spec.kinds or len(set(spec.kinds)) != len(spec.kinds)
                                  or set(spec.kinds) - {'player', 'goalkeeper'}):
        raise ValueError('Spatial distribution point kinds must be distinct player and/or goalkeeper.')
    provenance = [name for name in ('raw_hash', 'source_key') if name in points]
    if provenance:
        versions = points.groupby(keys, dropna=False)[provenance].nunique()
        bad = versions.gt(1).any(axis=1)
        if bad.any():
            raise ValueError(f'Conflicting spatial source versions for maps {versions.index[bad][:5].tolist()}.')
    from hashlib import sha256
    columns = [*keys, 'kind', 'x', 'y', *[c for c in ('point_order', 'raw_hash', 'source_key', 'observed_at', 'weight') if c in points]]
    return dict(source_sha256=sha256(pd.util.hash_pandas_object(points[columns], index=False).to_numpy().tobytes()).hexdigest(),
                provenance=deepcopy(points.attrs.get('source', {})), configuration=repr(spec))


def _tolerance(value):
    if isinstance(value, bool) or not np.isfinite(value) or not 0 <= value <= 1e-6:
        raise ValueError('mass_tolerance must be finite and between 0 and 1e-6; it is a numerical tolerance, not smoothing.')


@dataclass(frozen=True)
class SpatialEntropy:
    """Discrete entropy of a unit-mass grid; wrapper order is meaningful.

    RollingMean(SpatialEntropy(Heatmap(...))) averages per-match entropy.
    SpatialEntropy(RollingMean(Heatmap(...))) describes the historical mixture.
    An observed source still requires a historical operator before prediction.
    """
    source: object
    normalized: bool = True
    base: float = math.e
    mass_tolerance: float = 1e-10

    def __post_init__(self):
        if not isinstance(self.normalized, bool):
            raise ValueError('normalized must be a boolean.')
        if isinstance(self.base, bool) or not np.isfinite(self.base) or self.base <= 1:
            raise ValueError('Entropy base must be finite and greater than one.')
        _tolerance(self.mass_tolerance)


CONCENTRATION_METRICS = ('effective_cells', 'effective_fraction', 'hhi',
                         'normalized_hhi', 'largest_cell_share', 'occupied_fraction')


@dataclass(frozen=True)
class SpatialConcentration:
    """Select one concentration summary, without adding a bundle of predictors.

    Effective cells is exp(natural entropy); effective fraction divides by K.
    HHI is sum(p**2); normalized HHI maps uniform/one-cell maps to zero/one.
    Occupied fraction counts positive cells and is sensitive to sample size.
    """
    source: object
    metric: str = 'hhi'
    mass_tolerance: float = 1e-10

    def __post_init__(self):
        if self.metric not in CONCENTRATION_METRICS:
            raise ValueError(f'Choose a spatial concentration metric from {CONCENTRATION_METRICS}.')
        _tolerance(self.mass_tolerance)


def validated_grid(frame, spec, name):
    """All-missing means unavailable; partially missing/corrupt maps fail closed."""
    n = spec['grid_size']
    if len(spec['columns']) != n*n or not set(spec['columns']) <= set(frame):
        raise ValueError(f'Spatial grid {name!r} has missing or incorrectly shaped cells.')
    values = frame[spec['columns']].to_numpy(dtype=float, na_value=np.nan)
    missing = np.isnan(values).all(axis=1)
    bad = ~missing & ((~np.isfinite(values)).any(axis=1) | (values < 0).any(axis=1))
    if bad.any():
        rows = frame.index[np.flatnonzero(bad)[:5]].tolist()
        raise ValueError(f'Invalid spatial grid {name!r} at rows {rows}: require nonnegative finite cells or a wholly missing map.')
    return values, missing


def distribution_values(frame, node):
    specs = frame.attrs.get('spatial_features', {})
    if not specs or any(s['kind'] != 'grid' for s in specs.values()):
        raise TypeError(f'{type(node).__name__} needs a spatial grid source.')
    outputs, metadata = {}, {}
    for name, spec in specs.items():
        if spec['normalization'] not in ('mass', 'density'):
            raise ValueError('Spatial distribution summaries require mass or density grids; configure Heatmap(normalization="mass").')
        values, missing = validated_grid(frame, spec, name)
        probabilities = values[~missing].copy()
        if spec['normalization'] == 'density':
            probabilities *= (100/spec['grid_size'])**2
        totals = probabilities.sum(axis=1)
        bad = (~np.isfinite(totals)) | (totals <= 0) | (np.abs(totals-1) > node.mass_tolerance)
        if bad.any():
            rows = frame.index[np.flatnonzero(~missing)[bad][:5]].tolist()
            raise ValueError(f'Spatial grid {name!r} at rows {rows} must sum to one within mass_tolerance; no corrupted-map renormalization is performed.')
        probabilities /= totals[:, None]
        logs = np.zeros_like(probabilities)
        np.log(probabilities, out=logs, where=probabilities > 0)
        entropy = -np.sum(probabilities*logs, axis=1)
        k = spec['grid_size']**2
        if isinstance(node, SpatialEntropy):
            values_out = entropy/math.log(k) if node.normalized else entropy/math.log(node.base)
            field = 'normalized_entropy' if node.normalized else 'entropy'
            label = 'Normalized spatial entropy' if node.normalized else 'Spatial entropy'
            params = dict(normalized=node.normalized, base=float(node.base), mass_tolerance=node.mass_tolerance)
            units = 'fraction of maximum entropy' if node.normalized else f'log base {node.base:g}'
        else:
            field = node.metric
            hhi = np.sum(probabilities**2, axis=1)
            values_out = {
                'effective_cells': lambda: np.exp(entropy),
                'effective_fraction': lambda: np.exp(entropy)/k,
                'hhi': lambda: hhi,
                'normalized_hhi': lambda: (hhi-1/k)/(1-1/k),
                'largest_cell_share': lambda: probabilities.max(axis=1),
                'occupied_fraction': lambda: (probabilities > 0).sum(axis=1)/k,
            }[field]()
            label = field.replace('_', ' ').capitalize()
            params = dict(metric=field, mass_tolerance=node.mass_tolerance)
            units = 'cells' if field == 'effective_cells' else 'dimensionless'
        # Inputs have already passed strict validation. Bound only output
        # roundoff at mathematical endpoints; never clip probability cells.
        lower, upper = (0., 1.)
        if isinstance(node, SpatialEntropy) and not node.normalized:
            upper = math.log(k)/math.log(node.base)
        elif field == 'effective_cells':
            lower, upper = 1., float(k)
        elif field in ('hhi', 'effective_fraction', 'largest_cell_share', 'occupied_fraction'):
            lower = 1/k
        values_out = np.clip(values_out, lower, upper)
        column = f'{name}::{field}'
        result = np.full(len(frame), np.nan)
        result[~missing] = values_out
        outputs[column] = result
        metadata[column] = {**deepcopy(spec), 'kind':'scalar', 'orientation':'team',
            'columns':[column], 'field':field, 'field_label':label, 'units':units,
            'formula_version':'spatial_distribution_v1', 'distribution':params,
            'calculation':dict(operator=type(node).__name__, **params,
                               source=deepcopy(spec.get('calculation', {})))}
    result = pd.DataFrame(outputs, index=frame.index)
    result.attrs['spatial_features'] = metadata
    return result
