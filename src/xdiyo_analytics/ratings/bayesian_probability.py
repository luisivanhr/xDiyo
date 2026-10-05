"""Batched calibration probabilities with the native per-row support and sum.

No approximation, renormalization, changed tail tolerance, or fast-math. The
large-kappa stability branch continues to use the original scalar function.
"""

import numpy as np

from .bayesian_core import predict_score_summary


def outcome_probabilities(home, away, kappas, *, tail_tolerance=1e-10, max_total=10000):
    from scipy.stats import binom, nbinom, poisson

    home, away, kappas = (np.asarray(v, dtype=float) for v in (home, away, kappas))
    if home.ndim != 1 or home.shape != away.shape or home.shape != kappas.shape:
        raise ValueError("Probability arrays must be aligned one-dimensional arrays.")
    result = np.empty((len(home), 3))
    means = home + away
    safe = (np.isfinite(means) & (home >= 0) & (away >= 0) & (means > 0)
            & ((np.isnan(kappas)) | ((kappas > 0) & (kappas <= 100)
                & (kappas / (kappas + means) < 1))))
    # Scalar fallback keeps validation and extreme-parameter numerics native.
    if not (0 < tail_tolerance < 1) or not isinstance(max_total, int) or isinstance(max_total, bool) or max_total < 0:
        safe[:] = False
    for index in np.flatnonzero(~safe):
        summary = predict_score_summary(home[index], away[index],
            None if np.isnan(kappas[index]) else kappas[index],
            tail_tolerance=tail_tolerance, max_total=max_total)
        result[index] = [summary[name] for name in ("p_home_win", "p_draw", "p_away_win")]
    for poisson_case in (False, True):
        indices = np.flatnonzero(safe & (np.isnan(kappas) == poisson_case))
        for offset in range(0, len(indices), 64):
            selected = indices[offset:offset + 64]
            mean, k = means[selected], kappas[selected]
            p = k / (k + mean)
            sf = (lambda values: poisson.sf(values, mean)) if poisson_case else (lambda values: nbinom.sf(values, k, p))
            upper = np.minimum(max_total, np.maximum(1, np.ceil(mean))).astype(np.int64)
            while True:
                grow = (sf(upper) > tail_tolerance) & (upper < max_total)
                if not grow.any():
                    break
                upper[grow] = np.minimum(max_total, np.maximum(upper[grow] + 1, upper[grow] * 2))
            omitted = sf(upper)
            if not np.isfinite(omitted).all() or (omitted > tail_tolerance).any():
                raise ValueError("max_total cannot achieve the requested score probability tail_tolerance")
            low, high = np.zeros(len(selected), dtype=np.int64), upper.copy()
            while (low < high).any():
                middle = (low + high) // 2
                small = sf(middle) <= tail_tolerance
                active = low < high
                high[active & small] = middle[active & small]
                low[active & ~small] = middle[active & ~small] + 1
            totals = np.arange(int(low.max()) + 1)[None, :]
            mass = (poisson.pmf(totals, mean[:, None]) if poisson_case
                    else nbinom.pmf(totals, k[:, None], p[:, None]))
            fraction = (home[selected] / mean)[:, None]
            draw = np.where(totals % 2 == 0, binom.pmf(totals // 2, totals, fraction), 0.)
            wins = binom.sf(totals // 2, totals, fraction)
            losses = binom.cdf((totals - 1) // 2, totals, fraction)
            # Keep the SAME 1D np.dot and minimal integer support as the scalar
            # implementation; an axis sum or padded dot changes rounding.
            for row, index in enumerate(selected):
                end = int(low[row]) + 1
                result[index] = [float(np.dot(mass[row, :end], v[row, :end]))
                                 for v in (wins, draw, losses)]
    return result
