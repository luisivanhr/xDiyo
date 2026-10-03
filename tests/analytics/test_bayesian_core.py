"""Independent arithmetic, quadrature and variational checks for goal ratings."""

from dataclasses import replace
import math

import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import digamma, gammaln
from scipy.stats import gamma, skellam

from xdiyo_analytics.ratings.bayesian_core import (
    BayesianParameters, GammaState, TeamState, initial_home, initial_team,
    predict_score_summary, team_summary, update_matches,
)


def test_gamma_discount_scale_and_mirrored_entry_priors():
    original = GammaState(8, 4)
    assert original.mean == 2
    assert original.var == .5
    assert original.discount(.25).mean == original.mean
    assert original.discount(.25).variance == original.variance * 4
    assert original.scale(3).mean == original.mean * 3
    assert original.scale(3).variance == original.variance * 9
    assert hash(original) == hash(GammaState(8., 4.))
    params = BayesianParameters(attack_mean=1.4, defence_mean=.8, home_mean=1.3)
    promoted, relegated = initial_team(params, "promoted"), initial_team(params, "relegated")
    assert promoted.attack.mean * relegated.attack.mean == pytest.approx(params.attack_mean ** 2)
    assert promoted.defence.mean * relegated.defence.mean == pytest.approx(params.defence_mean ** 2)
    assert promoted.attack.mean < params.attack_mean < relegated.attack.mean
    assert promoted.defence.mean > params.defence_mean > relegated.defence.mean
    assert promoted.attack.shape == relegated.attack.shape == params.entry_attack_shape
    assert initial_home(params).mean == pytest.approx(1.3)
    assert initial_team(params) == initial_team(params, "retained")


@pytest.mark.parametrize("bad", [0, -1, math.inf, -math.inf, math.nan, True])
def test_gamma_rejects_invalid_parameters(bad):
    with pytest.raises(ValueError):
        GammaState(bad, 1)
    with pytest.raises(ValueError):
        GammaState(1, bad)


@pytest.mark.parametrize("kwargs", [
    {"team_discount": 1.01}, {"season_discount": 0}, {"home_discount": math.nan},
    {"home_season_discount": -1}, {"entry_attack_shift": -1},
    {"entry_defence_shift": math.inf}, {"attack_mean": 0}, {"kappa": 0},
])
def test_fixed_parameters_validate(kwargs):
    with pytest.raises(ValueError):
        BayesianParameters(**kwargs)


def fixture():
    states = {
        "a": TeamState(GammaState(5, 3), GammaState(7, 6)),
        "b": TeamState(GammaState(8, 9), GammaState(6, 7)),
        "c": TeamState(GammaState(4, 5), GammaState(9, 8)),
        "unused": TeamState(GammaState(3, 2), GammaState(5, 2)),
    }
    return states, GammaState(12, 10), BayesianParameters(kappa=4)


@pytest.mark.parametrize("bivariate", [False, True])
def test_one_step_uses_frozen_priors_epsilon_first_and_once_per_match(bivariate):
    states, home, params = fixture()
    original = states.copy()
    result, updated_home, diagnostics = update_matches(
        states, home, [("a", "b", 3, 1)], params, bivariate=bivariate, method="one_step")
    a, b, g = states["a"], states["b"], home.mean
    effect = ((params.kappa + 4) / (params.kappa + a.attack.mean * b.defence.mean * g
                                    + b.attack.mean * a.defence.mean)) if bivariate else 1.
    assert result["a"].attack == GammaState(8, 3 + b.defence.mean * g * effect)
    assert result["a"].defence == GammaState(8, 6 + b.attack.mean * effect)
    assert result["b"].attack == GammaState(9, 9 + a.defence.mean * effect)
    assert result["b"].defence == GammaState(9, 7 + a.attack.mean * g * effect)
    assert updated_home == GammaState(15, 10 + a.attack.mean * b.defence.mean * effect)
    assert diagnostics["random_effect_means"] == pytest.approx([effect])
    assert diagnostics["iterations"] == 1
    assert result["unused"] is states["unused"]
    assert states == original and home == GammaState(12, 10)


def test_joint_batch_is_order_invariant_and_counts_shared_home_once():
    states, home, params = fixture()
    matches = [("a", "b", 2, 1), ("c", "a", 0, 0), ("b", "c", 1, 3)]
    result, updated_home, diagnostics = update_matches(states, home, matches, params)
    reverse = update_matches(dict(reversed(list(states.items()))), home, reversed(matches), params)
    assert (result, updated_home, diagnostics) == reverse
    assert diagnostics["converged"] and 1 < diagnostics["iterations"] < 200
    assert result["a"].attack.shape == states["a"].attack.shape + 2
    assert result["a"].defence.shape == states["a"].defence.shape + 1
    assert updated_home.shape == home.shape + 3
    assert states["a"].attack.shape == 5


@pytest.mark.parametrize("bivariate", [False, True])
def test_converged_vb_satisfies_full_conditional_moment_equations(bivariate):
    states, home, params = fixture()
    results, g, diagnostics = update_matches(
        states, home, [("a", "b", 0, 0)], params, bivariate=bivariate, tolerance=1e-12)
    a, b = results["a"], results["b"]
    effect = params.kappa / (params.kappa + a.attack.mean * b.defence.mean * g.mean
                             + b.attack.mean * a.defence.mean) if bivariate else 1.
    assert diagnostics["converged"]
    assert a.attack.shape == states["a"].attack.shape
    assert a.attack.rate == pytest.approx(3 + b.defence.mean * g.mean * effect, rel=2e-12)
    assert b.attack.rate == pytest.approx(9 + a.defence.mean * effect, rel=2e-12)
    assert a.defence.rate == pytest.approx(6 + b.attack.mean * effect, rel=2e-12)
    assert b.defence.rate == pytest.approx(7 + a.attack.mean * g.mean * effect, rel=2e-12)
    assert g.rate == pytest.approx(10 + a.attack.mean * b.defence.mean * effect, rel=2e-12)


def test_vb_matches_independent_elbo_optimization():
    states, home, params = fixture()
    x, y = 3, 1
    result, updated_home, diagnostics = update_matches(
        states, home, [("a", "b", x, y)], params, tolerance=1e-12)
    # Independent objective for six Gamma factors, optimizing their log rates.
    # Order: attack home, defence away, attack away, defence home, HGA, epsilon.
    priors = [states["a"].attack, states["b"].defence, states["b"].attack,
              states["a"].defence, home, GammaState(params.kappa, params.kappa)]
    prior_shapes = np.array([p.shape for p in priors])
    prior_rates = np.array([p.rate for p in priors])
    shapes = prior_shapes + np.array([x, x, y, y, x, x + y])

    def negative_elbo(log_rates):
        rates = np.exp(log_rates)
        means = shapes / rates
        expected_logs = digamma(shapes) - log_rates
        expected_likelihood = (x * expected_logs[[0, 1, 4, 5]].sum()
                               + y * expected_logs[[2, 3, 5]].sum()
                               - means[0] * means[1] * means[4] * means[5]
                               - means[2] * means[3] * means[5])
        expected_prior = (prior_shapes * np.log(prior_rates) - gammaln(prior_shapes)
                          + (prior_shapes - 1) * expected_logs - prior_rates * means).sum()
        expected_log_q = (shapes * log_rates - gammaln(shapes)
                          + (shapes - 1) * expected_logs - rates * means).sum()
        return -(expected_likelihood + expected_prior - expected_log_q)

    optimum = minimize(negative_elbo, np.log(prior_rates), method="BFGS",
                       options={"gtol": 1e-6, "maxiter": 500})
    assert optimum.success, optimum.message
    epsilon_mean = diagnostics["random_effect_means"][0]
    actual_rates = np.array([result["a"].attack.rate, result["b"].defence.rate,
                             result["b"].attack.rate, result["a"].defence.rate,
                             updated_home.rate, (params.kappa + x + y) / epsilon_mean])
    assert np.exp(optimum.x) == pytest.approx(actual_rates, rel=2e-6)
    assert optimum.fun == pytest.approx(negative_elbo(np.log(actual_rates)), abs=1e-10)


def test_nonconvergence_is_visible_and_empty_batch_does_not_discount():
    states, home, params = fixture()
    _, _, diagnostics = update_matches(states, home, [("a", "b", 5, 0)], params, max_iterations=1)
    assert not diagnostics["converged"] and diagnostics["iterations"] == 1
    result, new_home, diagnostics = update_matches(states, home, [], params)
    assert result == states and result is not states
    assert new_home is home and diagnostics["converged"] and diagnostics["iterations"] == 0


@pytest.mark.parametrize("match", [("a", "a", 1, 0), ("a", "missing", 0, 0),
                                  ("a", "b", .5, 0), ("a", "b", -1, 0),
                                  ("a", "b", True, 0), ("a", "b", math.nan, 0)])
def test_invalid_matches_fail(match):
    states, home, params = fixture()
    with pytest.raises(ValueError):
        update_matches(states, home, [match], params)


def test_log_features_are_expected_logs_and_defence_direction_is_explicit():
    state = TeamState(GammaState(1, 2), GammaState(1, 4))
    summary = team_summary(state)
    euler_gamma = .5772156649015329
    assert summary["log_attack"] == pytest.approx(-euler_gamma - math.log(2))
    assert summary["log_defence_strength"] == pytest.approx(math.log(4) + euler_gamma)
    assert summary["strength_index"] == pytest.approx(math.log(2))
    assert summary["log_attack"] != pytest.approx(math.log(state.attack.mean))
    assert summary["defence_vulnerability_mean"] == .25


def test_negative_multinomial_normalization_moments_and_outcomes():
    lh, la, k = 1.8, 1.1, 3.4
    # Independent joint score PMF; no total/binomial decomposition.
    x, y = np.meshgrid(np.arange(70), np.arange(70), indexing="ij")
    joint = np.exp(gammaln(k + x + y) - gammaln(k) - gammaln(x + 1) - gammaln(y + 1)
                   + k * math.log(k) + x * math.log(lh) + y * math.log(la)
                   - (k + x + y) * math.log(k + lh + la))
    assert joint.sum() == pytest.approx(1, abs=1e-13)
    assert (joint * x).sum() == pytest.approx(lh, abs=1e-12)
    assert (joint * y).sum() == pytest.approx(la, abs=1e-12)
    assert (joint * (x - lh) ** 2).sum() == pytest.approx(lh + lh ** 2 / k)
    assert (joint * (x - lh) * (y - la)).sum() == pytest.approx(lh * la / k)
    result = predict_score_summary(lh, la, k, tail_tolerance=1e-12)
    assert result["expected_home_goals"] == pytest.approx((joint * x).sum())
    assert result["expected_away_goals"] == pytest.approx((joint * y).sum())
    for key, mask in [("p_home_win", x > y), ("p_draw", x == y), ("p_away_win", x < y),
                      ("p_both_score", (x > 0) & (y > 0)), ("p_over_2_5", x + y > 2)]:
        assert result[key] == pytest.approx(joint[mask].sum(), abs=1.1e-12)
    assert sum(result[key] for key in ("p_home_win", "p_draw", "p_away_win")) == pytest.approx(
        1 - result["omitted_mass"], abs=1e-14)


def test_bivariate_outcomes_match_independent_random_effect_quadrature():
    lh, la, k = 1.6, .9, 4.2
    result = predict_score_summary(lh, la, k, tail_tolerance=1e-12)
    def integrand(effect, kind):
        if effect == 0:
            return 0.
        conditional = {"p_home_win": lambda: skellam.sf(0, lh * effect, la * effect),
                       "p_draw": lambda: skellam.pmf(0, lh * effect, la * effect),
                       "p_away_win": lambda: skellam.cdf(-1, lh * effect, la * effect)}[kind]()
        return conditional * gamma.pdf(effect, k, scale=1 / k)
    for key in ("p_home_win", "p_draw", "p_away_win"):
        reference, error = quad(integrand, 0, np.inf, args=(key,), epsabs=2e-11)
        assert error < 1e-8
        assert result[key] == pytest.approx(reference, abs=2e-10)


def test_poisson_control_matches_skellam_and_large_kappa_limit():
    result = predict_score_summary(1.7, 1.2)
    assert result["p_home_win"] == pytest.approx(skellam.sf(0, 1.7, 1.2), abs=1e-10)
    assert result["p_draw"] == pytest.approx(skellam.pmf(0, 1.7, 1.2), abs=1e-10)
    assert result["p_away_win"] == pytest.approx(skellam.cdf(-1, 1.7, 1.2), abs=1e-10)
    assert result["p_both_score"] == pytest.approx((1 - math.exp(-1.7)) * (1 - math.exp(-1.2)))
    limit = predict_score_summary(1.7, 1.2, 1e7)
    for key in ("p_home_win", "p_draw", "p_away_win", "p_both_score", "p_over_2_5"):
        assert limit[key] == pytest.approx(result[key], abs=1e-7)


@pytest.mark.parametrize("kappa", [None, 2.])
def test_zero_intensities_and_explicit_tail(kappa):
    zero = predict_score_summary(0, 0, kappa)
    assert zero["p_draw"] == 1 and zero["omitted_mass"] == 0
    one_sided = predict_score_summary(2., 0, kappa, tail_tolerance=1e-4)
    assert one_sided["p_away_win"] == 0 and one_sided["p_both_score"] == 0
    assert 0 < one_sided["omitted_mass"] <= 1e-4
    assert one_sided["p_home_win"] + one_sided["p_draw"] == pytest.approx(1 - one_sided["omitted_mass"])
    with pytest.raises(ValueError, match="max_total"):
        predict_score_summary(8, 6, kappa, max_total=2)


@pytest.mark.parametrize("kwargs", [{"lambda_home": -1}, {"lambda_away": math.inf},
                                    {"kappa": 0}, {"tail_tolerance": 0},
                                    {"tail_tolerance": 1}, {"max_total": -1}])
def test_prediction_validation(kwargs):
    with pytest.raises(ValueError):
        predict_score_summary(**({"lambda_home": 1., "lambda_away": 1.} | kwargs))
