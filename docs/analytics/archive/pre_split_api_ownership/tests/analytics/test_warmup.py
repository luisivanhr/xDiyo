"""Independent prior, exponential moment and handoff arithmetic."""

import math

import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.features import (ForAgainst, Hard, Lag, League, LeaveOneOut,
    LinearFade, ObservationCount, RollingMean, RollingStd, RollingZScore,
    SeededEMA, WarmStart, blend_moments, evaluate_features)
from xdiyo_analytics.features.warmup import seeded_moments
from transition_samples import A, B, OWN, STAT, history_from_games, warm_games, mover_games, movement


def warmed(history, operator, policy=None, **kwargs):
    policy = SeededEMA() if policy is None else policy
    return evaluate_features(history, {'value': WarmStart(operator, policy)}, **kwargs).iloc[:, 0]


@pytest.mark.parametrize('handoff,expected', [
    (Hard(1), [1, 0, 0, 0]), (Hard(3), [1, 1, 1, 0]),
    (LinearFade(), [1, 1, .5, 0]), (ObservationCount(5), [1, 5/6, 5/7, 5/8])])
def test_handoff_weights_have_exact_boundaries(handoff, expected):
    assert [handoff.weight(i, i) for i in range(4)] == pytest.approx(expected)


def test_blended_variance_includes_between_mean_spread_and_endpoints():
    assert blend_moments(2, 4, 10, 16, .25) == pytest.approx((8, 25))
    assert blend_moments(2, 4, math.nan, math.nan, 1) == (2, 4)
    assert blend_moments(math.nan, math.nan, 10, 16, 0) == (10, 16)
    with pytest.raises(ValueError):
        blend_moments(2, 4, 10, 16, 1.1)


def test_seeded_moments_match_explicit_weighted_distribution_and_skip_nonfinite():
    mean, variance = seeded_moments(3, 5, [10, np.nan, np.inf, 14], .5)
    # Final mass .25 on the original distribution, .25 on 10, .5 on 14.
    expected_mean = .25*3 + .25*10 + .5*14
    expected_variance = .25*(5 + (3-expected_mean)**2) + .25*(10-expected_mean)**2 + .5*(14-expected_mean)**2
    assert (mean, variance) == pytest.approx((expected_mean, expected_variance))
    assert seeded_moments(3, np.nan, [10], 1) == (10, 0)
    assert math.isnan(seeded_moments(3, np.nan, [10], .5)[1])


def test_none_and_hard_zero_preserve_ordinary_output_without_round_column():
    history = history_from_games(warm_games()).drop(columns='round')
    expressions = {'plain': RollingStd(STAT, 2), 'none': WarmStart(RollingStd(STAT, 2)),
                   'zero': WarmStart(RollingStd(STAT, 2), SeededEMA(handoff=Hard(0)))}
    result = evaluate_features(history, expressions)
    np.testing.assert_allclose(result.plain, result.none, equal_nan=True)
    np.testing.assert_allclose(result.plain, result.zero, equal_nan=True)


@pytest.mark.parametrize('rounds,expected', [(1, [3, 7, 12, 16]), (2, [3, 6.5, 12, 16]), (3, [3, 6.5, 10.25, 16])])
def test_hard_handoff_uses_completed_rounds_then_original_window(rounds, expected):
    history = history_from_games(warm_games())
    result = warmed(history, RollingMean(STAT, 2), SeededEMA(handoff=Hard(rounds)))
    np.testing.assert_allclose(result.iloc[[4, 6, 8, 10]], expected)


def test_previous_league_variance_and_population_vs_rolling_ddof():
    history = history_from_games(warm_games())
    result = warmed(history, RollingStd(STAT, 2), SeededEMA(handoff=Hard(2)))
    np.testing.assert_allclose(result.iloc[[4, 6, 8]], np.sqrt([5, 14.75, 8]))
    default = warmed(history, RollingStd(STAT, 2))
    assert default.iloc[6] == pytest.approx(math.sqrt(18))


def test_linear_fade_blends_consistent_mean_and_variance_then_restores_ddof():
    history = history_from_games(warm_games())
    policy = SeededEMA(handoff=LinearFade())
    result = evaluate_features(history, {'mean': WarmStart(RollingMean(STAT, 2), policy),
        'std': WarmStart(RollingStd(STAT, 2), policy)})
    assert result['mean'].iloc[8] == 11.125
    assert result['std'].iloc[8] == pytest.approx(math.sqrt(13.484375))
    assert result['mean'].iloc[10] == 16
    assert result['std'].iloc[10] == pytest.approx(math.sqrt(8))


def test_observation_count_uses_valid_new_observations_and_allows_partial_windows():
    history = history_from_games(warm_games())
    policy = SeededEMA(handoff=ObservationCount(5))
    mean = warmed(history, RollingMean(STAT, 2), policy)
    std = warmed(history, RollingStd(STAT, 2), policy)
    w = 5/6
    assert mean.iloc[6] == pytest.approx(w*6.5 + (1-w)*7)
    assert std.iloc[6] == pytest.approx(math.sqrt(w*14.75 + (1-w)*9 + w*(1-w)*.5**2))
    games = warm_games()
    games[2]['values'] = (None, 20)
    missing = history_from_games(games)
    assert warmed(missing, RollingMean(STAT, 2), policy).iloc[6] == 3
    assert warmed(history, RollingMean(STAT, 20), SeededEMA()).iloc[6] == pytest.approx(16/3)


def test_no_mean_prior_falls_back_to_ordinary_history():
    history = history_from_games(warm_games()[2:])
    result = evaluate_features(history, {'plain': RollingMean(STAT, 2),
        'warm': WarmStart(RollingMean(STAT, 2), SeededEMA(handoff=Hard(10)))})
    np.testing.assert_allclose(result.plain, result.warm, equal_nan=True)


@pytest.mark.parametrize('weight,expected', [(0, 3), (.5, 14), (1, 25)])
def test_mover_mean_blends_own_history_and_full_destination_without_rating_cohorts(weight, expected):
    history = history_from_games(mover_games())
    policy = SeededEMA(league_weight=weight)
    assert warmed(history, RollingMean(STAT, 2), policy, team_seasons=movement()).iloc[8] == expected
    std = warmed(history, RollingStd(STAT, 2), policy, team_seasons=movement())
    assert std.iloc[8] == pytest.approx(math.sqrt(125))
    changed = history.copy()
    changed['team_position'] = list(reversed(range(1, len(changed)+1)))
    assert warmed(changed, RollingMean(STAT, 2), policy, team_seasons=movement()).iloc[8] == expected


def test_missing_destination_variance_remains_unknown_despite_valid_team_mean():
    games = [g for g in mover_games() if not (g['competition'] == 10 and g['season'] == 1)]
    history = history_from_games(games)
    kwargs = {'team_seasons': movement()}
    assert warmed(history, RollingMean(STAT, 2), team_seasons=movement()).iloc[4] == 3
    assert pd.isna(warmed(history, RollingStd(STAT, 2), **kwargs).iloc[4])


def test_seed_remains_available_without_current_rolling_samples():
    history = history_from_games(warm_games())
    league = League(STAT, window_unit='days')
    mean = warmed(history, RollingMean(league, 3), SeededEMA(handoff=Hard(2)))
    assert mean.iloc[4] == 5  # All prior-season observations lie outside the three-day window.
    assert mean.iloc[6] == 13.75  # All current-season contributions update the seed; no huge day duration.


def test_league_loo_seed_excludes_focal_team_and_updates_remaining_contributions():
    history = history_from_games(warm_games())
    population = LeaveOneOut(League(STAT))
    policy = SeededEMA(handoff=Hard(2))
    mean = warmed(history, RollingMean(population, 2), policy)
    std = warmed(history, RollingStd(population, 2), policy)
    assert mean.iloc[4] == 7 and std.iloc[4] == 1
    assert mean.iloc[6] == 13.5 and std.iloc[6] == pytest.approx(math.sqrt(42.75))


def test_warmed_z_default_and_explicit_lag_use_same_eligible_reference():
    history = history_from_games(warm_games())
    policy = SeededEMA(handoff=Hard(2))
    result = evaluate_features(history, {'default': WarmStart(RollingZScore(STAT, 2), policy),
        'explicit': WarmStart(RollingZScore(STAT, 2, reference=Lag(STAT)), policy)})
    np.testing.assert_allclose(result.default, result.explicit, equal_nan=True)
    assert result.default.iloc[4] == pytest.approx(1/math.sqrt(5))
    assert result.default.iloc[6] == pytest.approx(3.5/math.sqrt(14.75))


def test_both_perspectives_keep_their_own_seeded_means():
    history = history_from_games(warm_games())
    result = evaluate_features(history, {'both': WarmStart(RollingMean(ForAgainst(STAT, 'both'), 2), SeededEMA())})
    np.testing.assert_allclose(result.iloc[4], [3, 7])


@pytest.mark.parametrize('factory', [lambda: Hard(-1), lambda: Hard(True), lambda: LinearFade(rounds=0),
    lambda: ObservationCount(0), lambda: SeededEMA(alpha=0), lambda: SeededEMA(alpha=1.01),
    lambda: SeededEMA(league_weight=-1), lambda: SeededEMA(handoff='automatic')])
def test_invalid_warmup_parameters_are_rejected(factory):
    with pytest.raises((ValueError, TypeError)):
        factory()
