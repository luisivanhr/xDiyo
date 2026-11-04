"""Causal observed-match gaps, distinct from verified physiological rest."""
import numpy as np

LEGACY_GAP_ENCODING = 'legacy_standardized_gap_v1'
BOUNDED_GAP_ENCODING = 'bounded_observed_gap_v1'
GAP_CONFIG_FIELDS = ('fixture_gap_encoding', 'observed_gap_scale_days', 'observed_gap_recent_days')


def encode_observed_gap(days, has_observation, *, scale_days=7., recent_days=30.):
    """One bounded duration and two unscaled availability indicators per side.

    Missingness is supplied by observation provenance. A known observation with
    an invalid duration is an error, rather than an ordinary missing history.
    """
    values = np.asarray(days, dtype=float)
    known = np.asarray(has_observation)
    if values.ndim != 1 or known.shape != values.shape or known.dtype.kind != 'b':
        raise ValueError('Observed gaps require aligned durations and Boolean provenance.')
    if (scale_days, recent_days) != (7., 30.):
        raise ValueError('bounded_observed_gap_v1 uses fixed scales of 7 and 30 days.')
    if not np.isfinite(values[known]).all() or (values[known] < 0).any():
        raise ValueError('Known observed gaps must be finite and nonnegative.')
    if not np.isnan(values[~known]).all():
        raise ValueError('No eligible observation requires an unknown duration.')
    result = np.zeros((len(values), 3), dtype=float)
    result[known, 0] = values[known] / (values[known] + scale_days)
    result[:, 1] = ~known
    result[:, 2] = ~known | (values > recent_days)
    return result


def migrate_gap_config(state):
    """Absent artifact metadata means the original transform, never a new default."""
    state = dict(state)
    present = set(GAP_CONFIG_FIELDS).intersection(state)
    if present and present != set(GAP_CONFIG_FIELDS):
        raise ValueError('Saved observed-gap encoding metadata is incomplete.')
    if not present:
        state['fixture_gap_encoding'] = LEGACY_GAP_ENCODING
    state.setdefault('sender_state_mode', 'updated')
    return state
