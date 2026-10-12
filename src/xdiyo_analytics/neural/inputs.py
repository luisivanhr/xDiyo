"""Whole-map missingness shared by standalone and graph CNN consumers."""
import numpy as np


def neural_maps(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 4 or np.isinf(values).any():
        raise ValueError('Heatmap inputs must be finite or wholly missing maps shaped [rows, maps, y, x].')
    missing = np.isnan(values)
    absent = missing.all(axis=(2, 3))
    if np.any(missing.any(axis=(2, 3)) != absent):
        raise ValueError('Partially missing heatmaps are corrupt; only whole-map absence is supported.')
    with np.errstate(over='ignore'):
        maps = np.nan_to_num(values, nan=0).astype(np.float32)
    if not np.isfinite(maps).all():
        raise ValueError('Heatmap values exceed the finite neural tensor range.')
    return maps, (~absent).astype(np.float32)
