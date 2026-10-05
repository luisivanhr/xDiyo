"""Stable first-three-moment arithmetic for population skewness.

Central-moment mixtures are algebraically equivalent to weighting E[X], E[X²]
and E[X³], without subtracting large nearly equal raw powers afterwards.
"""
import math
import numpy as np


def sample_moments(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    mean = float(values.mean())
    centered = values - mean
    return mean, float(np.mean(centered**2)), float(np.mean(centered**3))


def mix_moments(a, b, weight):
    """Mix distributions with weights weight and 1-weight; preserve endpoints."""
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError('Moment weight must lie in [0, 1].')
    if weight == 1:
        return a
    if weight == 0:
        return b
    ma, va, ta = a
    mb, vb, tb = b
    delta = mb - ma
    other = 1 - weight
    mean = ma + other * delta
    variance = weight * va + other * vb + weight * other * delta**2
    third = (weight * ta + other * tb + 3 * weight * other * delta * (vb - va)
             + weight * other * (weight - other) * delta**3)
    return mean, variance, third


def skewness(moments):
    _, variance, third = moments
    if not math.isfinite(variance) or variance <= 0 or not math.isfinite(third):
        return np.nan
    # Sequential division avoids overflowing variance**1.5.
    value = (third / variance) / math.sqrt(variance)
    return value if math.isfinite(value) else np.nan


def sample_skewness(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return skewness(sample_moments(values)) if len(values) >= 3 else np.nan
