"""Strict count-event scoring, independent of betting decisions and prices."""

from numbers import Real

import numpy as np
import pandas as pd

from .probabilities import bet_probabilities
from ..labels import BetOption, MatchTotal


DEFAULT_OU_LINES = (6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5)


def _lines(lines):
    if not isinstance(lines, (list, tuple, np.ndarray, pd.Index)):
        raise ValueError("lines must be a nonempty list of distinct positive finite half-integers.")
    try:
        values = list(lines)
    except TypeError as error:
        raise ValueError("lines must be a nonempty list of distinct positive finite half-integers.") from error
    if not values or any(not isinstance(v, Real) or isinstance(v, (bool, np.bool_))
                         or not np.isfinite(v) or v <= 0 or v % 1 != .5 for v in values):
        raise ValueError("lines must be a nonempty list of distinct positive finite half-integers.")
    if len(set(values)) != len(values):
        raise ValueError("lines must be distinct; duplicate lines change their weights.")
    return values


def _counts(values, name):
    if any(not isinstance(v, Real) or isinstance(v, (bool, np.bool_))
           or not np.isfinite(v) or v < 0 or v % 1 != 0 for v in values):
        raise ValueError(f"{name} must be finite numeric nonnegative integer counts.")


def count_ou_brier(y, p, *, lines=DEFAULT_OU_LINES):
    """Equal-weight Brier score over both sides of each configured half-line."""
    lines = _lines(lines)
    _counts(y, "Targets")
    _counts(p.columns, "Probability classes")
    if not p.index.equals(y.index):
        raise ValueError("Count O/U probabilities must align exactly with targets.")
    losses = np.zeros(len(y))
    # The source is a semantic count placeholder; no features or labels are
    # evaluated here. bet_probabilities consumes only its count-source type.
    source = MatchTotal(None)
    for line in lines:
        for side in ("under", "over"):
            probability = bet_probabilities(p, BetOption(source, side, line=line)).p_win.to_numpy()
            observed = y.lt(line) if side == "under" else y.gt(line)
            losses += (probability - observed.to_numpy(dtype=float)) ** 2
    return float(losses.mean() / (2 * len(lines))) if len(y) else np.nan


def source_count_probabilities(y, predictions, request, source_folds, pooling):
    """Restore only structural absent-class zeros, never missing source cells.

    Uses the already selected population and occurrence identities. Generic
    prediction tables and pooling behavior remain untouched.
    """
    output = request.output or "predict_proba"
    if pooling == "mean":
        raise ValueError("count_ou_brier does not support mean pooling; use occurrences, first or last.")
    frame = predictions[output]
    if not frame.index.equals(y.index):
        raise ValueError("Prediction outputs and targets must retain the same index/order.")
    if not source_folds or not len(y):
        return frame
    if not isinstance(y.index, pd.MultiIndex) or not {"fold_id", "row_position"}.issubset(y.index.names):
        raise ValueError("count_ou_brier source-fold scoring needs fold_id and row_position identities.")
    pieces = []
    fids = y.index.get_level_values("fold_id")
    for fid in fids.unique():
        if fid not in source_folds:
            raise ValueError(f"count_ou_brier cannot resolve source fold {fid!r}.")
        selected = y.index[fids == fid]
        source = source_folds[fid].predictions[output]
        if not source.index.is_unique:
            raise ValueError("Source-fold probability rows must be unique.")
        local = source.loc[selected.get_level_values("row_position")].copy()
        if not isinstance(local.columns, pd.MultiIndex) or local.columns.nlevels != 2:
            raise ValueError("Probability columns must be a (target, class) MultiIndex.")
        local = local.loc[:, local.columns.get_level_values(0) == request.target]
        local.index = selected
        # Validate before aligning supports: only absent columns become zeros.
        count_ou_brier(y.loc[selected, request.target], local.xs(request.target, level=0, axis=1),
                       **request.parameters)
        pieces.append(local)
    columns = pieces[0].columns
    for piece in pieces[1:]:
        columns = columns.union(piece.columns, sort=False)
    return pd.concat([piece.reindex(columns=columns, fill_value=0.) for piece in pieces]).reindex(y.index)
