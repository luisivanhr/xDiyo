"""Resolve discrete model distributions into BetOption settlement probabilities."""

import numpy as np
import pandas as pd

from ..labels import BetOption, MatchTotal, TeamValue, Outcome, Above


def _option(option):
    if not isinstance(option, BetOption):
        raise TypeError("Supply a BetOption, using the same source as the predicted target.")
    if option.on_equal not in {"push", "loss"} or option.draw not in {"push", "loss"}:
        raise ValueError("on_equal and draw must be push or loss.")
    if option.selection in {"over", "under"}:
        if not isinstance(option.source, (TeamValue, MatchTotal)) or option.line is None or not np.isfinite(option.line):
            raise ValueError("Over/under needs a count source and finite line.")
    elif option.line is not None:
        raise ValueError("line is only used for over/under selections.")


def bet_probabilities(probabilities, option, *, target=None):
    """Sum discrete masses, never interpolate between count classes.

    probabilities is a normalized class-probability DataFrame. For (target,class)
    columns, target chooses one output (optional when only one is present).
    The caller associates the prediction target with option.source. Status-based
    voids are settled from actual match status later, not inferred here.
    Quarter lines follow BetOption's literal threshold, not split-stake markets.
    """
    _option(option)
    if not isinstance(probabilities, pd.DataFrame):
        raise TypeError("Probabilities must be an identity-indexed DataFrame.")
    frame = probabilities
    if isinstance(frame.columns, pd.MultiIndex):
        targets = frame.columns.get_level_values(0).unique()
        if target is None:
            if len(targets) != 1:
                raise ValueError("Select the probability target explicitly.")
            target = targets[0]
        frame = frame.xs(target, axis=1, level=0)
    if not frame.columns.is_unique or not len(frame.columns):
        raise ValueError("Probability classes must be distinct and nonempty.")
    p = frame.to_numpy(dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError("Provide a complete finite class distribution summing to one per row.")
    classes = frame.columns.to_numpy()
    push = np.zeros(len(classes), dtype=bool)
    if option.selection in {"under", "over"}:
        if not pd.api.types.is_numeric_dtype(frame.columns.dtype):
            raise ValueError("Count probabilities need exact numeric count classes, not bins or an overflow label.")
        values = classes.astype(float)
        if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError("Count classes must be nonnegative integers.")
        won = values < option.line if option.selection == "under" else values > option.line
        if option.on_equal == "push":
            push = values == option.line
    elif option.selection in {"win", "draw", "loss"} and isinstance(option.source, Outcome):
        if not set(classes).issubset({-1, 0, 1}):
            raise ValueError("Outcome probabilities use loss=-1, draw=0, win=1 classes.")
        won = classes == {"win": 1, "draw": 0, "loss": -1}[option.selection]
        if option.selection != "draw" and option.draw == "push":
            push = classes == 0
    elif option.selection in {"yes", "no"} and isinstance(option.source, Above):
        if not set(classes).issubset({0, 1}):
            raise ValueError("Above probabilities use 0/1 classes.")
        won = classes == (1 if option.selection == "yes" else 0)
    else:
        raise ValueError("Selection does not match the BetOption source.")
    return pd.DataFrame({"p_win": p[:, won].sum(axis=1), "p_push": p[:, push].sum(axis=1),
                         "p_loss": p[:, ~(won | push)].sum(axis=1)}, index=frame.index)


def negative_binomial_bet_probabilities(mean, dispersion, option):
    """Exact NB2 count tails from mean/dispersion, without truncating support.

    mean is an indexed Series; dispersion is scalar or exactly aligned Series.
    Variance is mean + dispersion*mean**2. Dispersion zero uses the Poisson limit.
    A predicted mode must never be supplied in place of the conditional mean.
    """
    # Keep scipy distribution instances local: native recovery fingerprints
    # functions/configuration, not mutable scipy generator objects.
    from scipy.stats import nbinom, poisson
    _option(option)
    if option.selection not in {"under", "over"}:
        raise ValueError("Negative binomial probabilities support count over/under options.")
    if not isinstance(mean, pd.Series):
        raise TypeError("Supply predicted means as an identity-indexed Series.")
    if isinstance(dispersion, pd.Series) and not dispersion.index.equals(mean.index):
        raise ValueError("Dispersion must align exactly with mean row identities.")
    mu = mean.to_numpy(dtype=float)
    alpha = np.broadcast_to(np.asarray(dispersion, dtype=float), mu.shape)
    if not np.isfinite(mu).all() or not np.isfinite(alpha).all() or (mu < 0).any() or (alpha < 0).any():
        raise ValueError("Means and dispersions must be finite and nonnegative.")
    def probability(method, k):
        result = np.empty(len(mu))
        nb = alpha > 0
        result[nb] = getattr(nbinom, method)(k, 1/alpha[nb], 1/(1+alpha[nb]*mu[nb]))
        result[~nb] = getattr(poisson, method)(k, mu[~nb])
        return result
    line = float(option.line)
    below = probability("cdf", np.ceil(line)-1)
    above = probability("sf", np.floor(line))
    equal = probability("pmf", line) if line.is_integer() else np.zeros(len(mu))
    win, loss = (below, above) if option.selection == "under" else (above, below)
    push = equal if option.on_equal == "push" else np.zeros(len(mu))
    if option.on_equal == "loss":
        loss = loss + equal
    return pd.DataFrame({"p_win": win, "p_push": push, "p_loss": loss}, index=mean.index)
