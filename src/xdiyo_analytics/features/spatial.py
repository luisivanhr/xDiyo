"""Spatial point pooling, shared by historical features and heatmap plots."""
from dataclasses import dataclass
from numbers import Integral
import re

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Heatmap:
    """Observed team heatmap source; wrap in Lag/RollingMean/EMA for prediction.

    Exported coordinates use each source team's own goal on the left. Against
    observations rotate 180 degrees into the focal team's frame before temporal
    operations. orientation=home rotates final Away grids into the shared pitch;
    orientation=team retains the focal frame. heatmap_grid alone pools raw points.
    grid_size creates equal coordinate-space cells (not physical square metres).
    gaussian smooths a fine histogram before pooling to the output grid. sigma
    is in fine-grid cells. kinds=None includes every point kind. use_weights
    treats a missing weight as one; otherwise each point has unit weight.
    """
    grid_size: int = 10
    method: str = "grid"
    normalization: str = "mass"
    side: str = "for"
    sigma: float = 2.6
    resolution: int = 100
    kinds: tuple | None = None
    use_weights: bool = False
    orientation: str = "home"

    def __post_init__(self):
        if self.orientation not in ('home', 'team'):
            raise ValueError('Heatmap orientation must be home or team.')
        if isinstance(self.grid_size, bool) or not isinstance(self.grid_size, Integral) or self.grid_size < 2:
            raise ValueError("grid_size must be an integer of at least two.")
        if self.method not in ("grid", "gaussian") or self.normalization not in ("mass", "density", "count"):
            raise ValueError("Choose grid/gaussian and mass/density/count.")
        if self.side not in ("for", "against", "both"):
            raise ValueError("Heatmap side must be for, against or both.")
        if self.method == "gaussian":
            if isinstance(self.resolution, bool) or not isinstance(self.resolution, Integral) or self.resolution < self.grid_size or self.resolution % self.grid_size:
                raise ValueError("Gaussian resolution must be a positive multiple of grid_size.")
            if isinstance(self.sigma, bool) or not np.isfinite(self.sigma) or self.sigma <= 0:
                raise ValueError("Gaussian sigma must be positive and finite.")
        if self.kinds is not None:
            object.__setattr__(self, "kinds", (self.kinds,) if isinstance(self.kinds, str) else tuple(self.kinds))


def heatmap_grid(points, spec=None):
    """Return [y, x] cells; absent/empty/zero-weight observations remain NaN.

    mass sums to one; density integrates to one on the exported 100x100 plane;
    count sums to total point weight. Invalid coordinates/weights raise instead
    of being clipped into boundary cells. Endpoints 100 belong to the last cell.
    """
    spec = spec or Heatmap()
    if spec.kinds is not None:
        points = points.loc[points["kind"].isin(spec.kinds)]
    n = spec.grid_size
    if not len(points):
        return np.full((n, n), np.nan)
    xy = points[["x", "y"]].to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(xy).all() or ((xy < 0) | (xy > 100)).any():
        raise ValueError("Heatmap coordinates must be finite and within 0..100.")
    weights = (points["weight"].fillna(1).to_numpy(dtype=float) if spec.use_weights and "weight" in points
               else np.ones(len(points)))
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Heatmap weights must be finite and nonnegative.")
    total = weights.sum()
    if total <= 0:
        return np.full((n, n), np.nan)
    bins = n if spec.method == "grid" else spec.resolution
    grid = np.histogram2d(xy[:, 1], xy[:, 0], bins=bins, range=((0, 100), (0, 100)), weights=weights)[0]
    if spec.method == "gaussian":
        from scipy.ndimage import gaussian_filter
        grid = gaussian_filter(grid, sigma=spec.sigma, mode="reflect")
        block = bins // n
        grid = grid.reshape(n, block, n, block).sum(axis=(1, 3))
    if spec.normalization != "count":
        grid /= total
        if spec.normalization == "density":
            grid /= (100 / n) ** 2
    return grid


def heatmap_values(history, points, spec):
    """Align point grids to team history without floating-point identity joins."""
    if points is None:
        raise ValueError("Load heatmap_points and pass it to evaluate_features(heatmaps=...).")
    scope = [c for c in ("source_league", "source_season") if c in history and c in points]
    keys = [*scope, "event_id", "team_id"]
    grids = {key: heatmap_grid(frame, spec).ravel()
             for key, frame in points.groupby(keys, sort=False, dropna=False)}
    n = spec.grid_size
    output = {}
    metadata = {}
    for side in (("for", "against") if spec.side == "both" else (spec.side,)):
        ids = [*scope, "event_id", "team_id" if side == "for" else "opponent_id"]
        matrix = np.array([grids.get(key, np.full(n*n, np.nan))
                           for key in history[ids].itertuples(index=False, name=None)]).reshape(len(history), n*n)
        # Historical observations use the focal team's frame, independently of
        # the historical venue. Only final predictor rows adopt target venue.
        if side == 'against':
            matrix = matrix[:, ::-1]
        prefix = f"heatmap_{side}_{spec.normalization}_{n}"
        columns = []
        for y in range(n):
            for x in range(n):
                column = f"{prefix}_y{y:02}_x{x:02}"
                columns.append(column)
                output[column] = matrix[:, y*n+x]
        metadata[prefix] = dict(kind='grid', columns=columns, grid_size=n, side=side,
                                orientation=spec.orientation, normalization=spec.normalization,
                                method=spec.method, operators=[], source='Heatmap')
    result = pd.DataFrame(output, index=history.index)
    result.attrs['spatial_features'] = metadata
    return result


@dataclass(frozen=True)
class RegionMass:
    """Integrate spatial cells over the focal team's own or opponent half.

    Accepts a Heatmap source inside a historical operator, or a historical grid
    directly. Regions are team-relative even when final grids use the home frame.
    A boundary cell is apportioned by area (piecewise constant cell density).
    """
    source: object
    region: str = 'own_half'

    def __post_init__(self):
        if self.region not in ('own_half', 'opponent_half'):
            raise ValueError('Choose own_half or opponent_half.')


def region_values(frame, region):
    """Integrate each declared grid in its internal focal-team frame."""
    from copy import deepcopy
    outputs, metadata = {}, {}
    specs = frame.attrs.get('spatial_features', {})
    if not specs or any(s['kind'] != 'grid' for s in specs.values()):
        raise TypeError('RegionMass needs a spatial grid source.')
    for name, spec in specs.items():
        n = spec['grid_size']
        edges = np.linspace(0, 100, n+1)
        fraction = np.clip((50-edges[:-1])/(100/n), 0, 1)
        if region == 'opponent_half':
            fraction = 1-fraction
        weights = np.tile(fraction, n)
        if spec['normalization'] == 'density':
            weights *= (100/n)**2
        values = frame[spec['columns']].to_numpy(dtype=float, na_value=np.nan)
        # An unavailable contributing cell makes the regional quantity unknown.
        selected = weights > 0
        column = f'{name}::region_{region}'
        outputs[column] = values[:, selected] @ weights[selected]
        metadata[column] = {**deepcopy(spec), 'kind':'region', 'region':region, 'columns':[column]}
    result = pd.DataFrame(outputs, index=frame.index)
    result.attrs['spatial_features'] = metadata
    return result


def finalize_spatial(frame, history, names, feature_name):
    """Name metadata and rotate final away grid outputs into the home frame."""
    from copy import deepcopy
    specs = deepcopy(frame.attrs.get('spatial_features', {}))
    rename = dict(zip(frame.columns, names))
    result = frame.copy()
    metadata = {}
    for key, spec in specs.items():
        if spec['kind'] == 'grid' and spec['orientation'] == 'home':
            away = history.side.eq('away').to_numpy(dtype=bool)
            values = result[spec['columns']].to_numpy(copy=True)
            values[away] = values[away, ::-1]
            result[spec['columns']] = values
        spec['columns'] = [rename[c] for c in spec['columns']]
        spec['feature'] = feature_name
        output_key = feature_name if len(specs) == 1 else f'{feature_name}::{key}'
        metadata[output_key] = spec
    result.columns = names
    result.attrs = {}
    return result, metadata


def heatmap_columns(columns):
    """Discover named spatial outputs, including assembled home/away prefixes."""
    groups = {}
    for column in columns:
        match = re.fullmatch(r"(.+::heatmap_(?:for|against)_(?:mass|density|count)_(\d+))_y(\d+)_x(\d+)", str(column))
        if match:
            name, n, y, x = match.groups()
            groups.setdefault(name, {"size": int(n), "cells": {}})["cells"][int(y), int(x)] = column
    return groups
