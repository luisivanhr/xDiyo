"""Independent regressions for the large-dispersion score probability limit."""

from decimal import Decimal, localcontext
import math

import pytest

pytest.importorskip("scipy")
from scipy.stats import poisson, skellam

from xdiyo_analytics.ratings.bayesian_core import predict_score_summary, score_log_probability


def _decimal_reference(home, away, kappa):
    """Sum a high-precision joint PMF via its two-dimensional recurrence."""
    with localcontext() as ctx:
        ctx.prec = 100
        h, a, k = map(lambda v: Decimal(str(v)), (home, away, kappa))
        denominator = k + h + a
        p00 = (k * (k / denominator).ln()).exp()
        result = {name: Decimal(0) for name in ("p_home_win", "p_draw", "p_away_win", "p_both_score", "p_over_2_5")}
        row_start = p00
        for x in range(80):
            probability = row_start
            for y in range(80):
                result["p_home_win" if x > y else "p_away_win" if y > x else "p_draw"] += probability
                if x and y:
                    result["p_both_score"] += probability
                if x + y > 2:
                    result["p_over_2_5"] += probability
                if (x, y) == (2, 1):
                    log_score = probability.ln()
                probability *= (k + x + y) * a / (Decimal(y + 1) * denominator)
            row_start *= (k + x) * h / (Decimal(x + 1) * denominator)
        return {name: float(value) for name, value in result.items()}, float(log_score)


@pytest.mark.parametrize("kappa", [1e3, 1e6, 1e7, 1e12, 1e16, 1e20, 1e60])
def test_large_kappa_matches_independent_high_precision_joint_law(kappa):
    reference, log_score = _decimal_reference(1.7, 1.2, kappa)
    actual = predict_score_summary(1.7, 1.2, kappa, tail_tolerance=1e-13)
    for name, expected in reference.items():
        assert actual[name] == pytest.approx(expected, abs=1.1e-13)
    assert score_log_probability(2, 1, 1.7, 1.2, kappa) == pytest.approx(log_score, abs=3e-14)
    assert score_log_probability(2, 1, 1.7, 1.2, kappa) < 0
    assert sum(actual[name] for name in ("p_home_win", "p_draw", "p_away_win")) == pytest.approx(
        1 - actual["omitted_mass"], abs=3e-15)


def test_large_kappa_converges_to_poisson_without_degenerate_total():
    actual = predict_score_summary(1.7, 1.2, 1e20, tail_tolerance=1e-14)
    assert actual["p_draw"] == pytest.approx(skellam.pmf(0, 1.7, 1.2), abs=2e-14)
    assert actual["p_over_2_5"] == pytest.approx(poisson.sf(2, 2.9), abs=2e-14)
    assert actual["p_home_win"] > 0 and actual["p_away_win"] > 0
    assert actual["omitted_mass"] > 0
    with pytest.raises(ValueError, match="max_total"):
        predict_score_summary(1.7, 1.2, 1e20, max_total=2)


def test_tiny_positive_intensity_does_not_round_to_a_degenerate_total():
    result = predict_score_summary(1e-20, 0., 6.323, tail_tolerance=1e-30)
    assert result["p_home_win"] == pytest.approx(1e-20, rel=1e-13, abs=0)
    assert result["p_away_win"] == 0
    assert 0 < result["omitted_mass"] < 1e-30


@pytest.mark.parametrize("kappa", [None, .1, 6.323, 100., 1e12, 1e20])
def test_joint_score_zero_intensities_have_exact_support(kappa):
    assert score_log_probability(0, 0, 0., 0., kappa) == 0
    assert score_log_probability(1, 0, 0., 1., kappa) == -math.inf
    assert score_log_probability(0, 1, 1., 0., kappa) == -math.inf
    assert math.isfinite(score_log_probability(0, 2, 0., 1., kappa))
    actual = predict_score_summary(0., 1., kappa)
    assert actual["p_home_win"] == 0
    assert actual["p_both_score"] == 0


@pytest.mark.parametrize("kappa", [1., 6.323, 100.])
def test_stable_joint_score_preserves_ordinary_range(kappa):
    reference, log_score = _decimal_reference(1.7, 1.2, kappa)
    assert score_log_probability(2, 1, 1.7, 1.2, kappa) == pytest.approx(log_score, abs=3e-14)
    actual = predict_score_summary(1.7, 1.2, kappa, tail_tolerance=1e-12)
    for name, expected in reference.items():
        assert actual[name] == pytest.approx(expected, abs=1.1e-12)


def test_unrepresentable_special_function_result_raises_instead_of_returning_nan(monkeypatch):
    import scipy.special

    # Test the guard independently of a particular SciPy version's limits.
    monkeypatch.setattr(scipy.special, "betainc", lambda *args: float("nan"))
    with pytest.raises(ValueError, match="survival calculation"):
        predict_score_summary(1.7, 1.2, 1e308)
