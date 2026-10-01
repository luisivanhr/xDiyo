"""Strict count-event scoring, independent of betting decisions and prices."""

from numbers import Real

import numpy as np
import pandas as pd

from .probabilities import bet_probabilities, negative_binomial_bet_probabilities
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


def _distribution(distribution):
    if distribution not in ("categorical", "negative_binomial", "poisson"):
        raise ValueError("distribution must be categorical, negative_binomial or poisson.")
    return distribution


def count_score_output(request):
    distribution = _distribution(request.parameters.get("distribution", "categorical"))
    return request.output or ("predict_proba" if distribution == "categorical" else "count_distribution")


def _parameters(p, distribution):
    names = set(p.columns)
    allowed = ({"mean", "dispersion"},) if distribution == "negative_binomial" else ({"mean"}, {"mean", "dispersion"})
    if not p.columns.is_unique or names not in allowed:
        raise ValueError(f"{distribution} requires distinct mean/dispersion parameter columns "
                         "(Poisson may omit dispersion); no extra columns are allowed.")
    values = p.to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Count distribution parameters must be complete, finite and nonnegative.")
    dispersion = p["dispersion"] if "dispersion" in p else pd.Series(0., index=p.index)
    if distribution == "poisson" and dispersion.ne(0).any():
        raise ValueError("Poisson dispersion must be zero when supplied.")
    return p["mean"], dispersion


def count_ou_brier(y, p, *, lines=DEFAULT_OU_LINES, distribution="categorical"):
    """Equal-weight Brier score over both sides of each configured half-line.

    categorical consumes a class PMF. negative_binomial consumes mean and
    dispersion; poisson consumes mean with optional explicit zero dispersion.
    """
    lines = _lines(lines)
    _counts(y, "Targets")
    distribution = _distribution(distribution)
    if distribution == "categorical":
        _counts(p.columns, "Probability classes")
    else:
        mean, dispersion = _parameters(p, distribution)
    if not p.index.equals(y.index):
        raise ValueError("Count O/U probabilities must align exactly with targets.")
    losses = np.zeros(len(y))
    # The source is a semantic count placeholder; no features or labels are
    # evaluated here. bet_probabilities consumes only its count-source type.
    source = MatchTotal(None)
    for line in lines:
        for side in ("under", "over"):
            option = BetOption(source, side, line=line)
            events = (bet_probabilities(p, option) if distribution == "categorical" else
                      negative_binomial_bet_probabilities(mean, dispersion, option))
            probability = events.p_win.to_numpy()
            if not np.isfinite(probability).all():
                raise ValueError("Count distribution produced nonfinite event probabilities.")
            observed = y.lt(line) if side == "under" else y.gt(line)
            losses += (probability - observed.to_numpy(dtype=float)) ** 2
    return float(losses.mean() / (2 * len(lines))) if len(y) else np.nan


def source_count_probabilities(y, predictions, request, source_folds, pooling):
    """Resolve source-fold inputs; restore only categorical absent-class zeros.

    Uses the already selected population and occurrence identities. Generic
    prediction tables and pooling behavior remain untouched.
    """
    output = count_score_output(request)
    distribution = request.parameters.get("distribution", "categorical")
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
    if distribution != "categorical":
        # Parameter vectors have been validated per source. Never restore
        # missing parameters as zero, or average different source models.
        # Scoring above accepts optional zero dispersion for Poisson, so make
        # that explicit only for sources which declared a mean-only Poisson.
        if distribution == "poisson":
            for piece in pieces:
                if (request.target, "dispersion") not in piece.columns:
                    piece[(request.target, "dispersion")] = 0.
        return pd.concat(pieces).reindex(y.index)
    columns = pieces[0].columns
    for piece in pieces[1:]:
        columns = columns.union(piece.columns, sort=False)
    return pd.concat([piece.reindex(columns=columns, fill_value=0.) for piece in pieces]).reindex(y.index)
