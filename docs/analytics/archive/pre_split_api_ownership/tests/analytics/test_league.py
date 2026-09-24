"""League windows and exclusions checked against explicit observation lists."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.features import (ForAgainst, H2H, Lag, League, LeaveOneOut,
    RollingMean, RollingStd, RollingZScore, Stat, WarmStart, evaluate_features)
from transition_samples import A, B, C, D, OWN, STAT, START, history_from_games, league_games


def feature(history, expression, **kwargs):
    return evaluate_features(history, {'value': expression}, **kwargs).iloc[:, 0]


def test_round_population_pools_observations_and_freezes_every_target_fixture():
    history = history_from_games(league_games())
    population = League(STAT)
    result = evaluate_features(history, {'mean': RollingMean(population, 2),
        'std': RollingStd(population, 2), 'count_gate': RollingMean(population, 2, min_periods=8)})
    expected = [2, 8, 4, 6, 10, 20, 30, 40]
    np.testing.assert_allclose(result.iloc[8:12]['mean'], np.mean(expected))
    np.testing.assert_allclose(result.iloc[8:12]['std'], np.std(expected, ddof=1))
    np.testing.assert_allclose(result.iloc[8:12]['count_gate'], 15)


def test_unequal_round_sizes_are_observation_weighted():
    games = league_games()
    games.pop(1)  # Round one has two contributions; round two has four.
    history = history_from_games(games)
    result = evaluate_features(history, {'mean': RollingMean(League(STAT), 2), 'std': RollingStd(League(STAT), 2)})
    assert result['mean'].iloc[6] == pytest.approx(110 / 6)
    assert result['std'].iloc[6] == pytest.approx(np.std([2, 8, 10, 20, 30, 40], ddof=1))


@pytest.mark.parametrize('blocked', ['unfinished', 'late_release'])
def test_incomplete_round_retains_previous_complete_baseline(blocked):
    games = league_games()
    if blocked == 'unfinished':
        games[3]['status'] = 'postponed'
    else:
        games[3]['release'] = 7
    history = history_from_games(games)
    result = feature(history, RollingMean(League(STAT), 1), available_at='available_at')
    np.testing.assert_allclose(result.iloc[8:12], 5)


def test_kickoff_schedule_evolves_while_round_snapshot_does_not():
    history = history_from_games(league_games())
    result = feature(history, RollingMean(League(STAT, schedule='kickoff'), 1))
    assert result.iloc[8] == 25
    assert result.iloc[10] == 55  # The already played fixture in round three.


def test_target_round_uses_minimum_custom_cutoff_and_availability_boundary():
    games = league_games()
    games[3]['release'] = 6
    history = history_from_games(games)
    cutoffs = history.kickoff_at.copy()
    cutoffs.iloc[8:10] = START + pd.Timedelta(days=5)
    frozen = feature(history, RollingMean(League(STAT), 1), cutoffs=cutoffs, available_at='available_at')
    np.testing.assert_allclose(frozen.iloc[8:12], 5)
    exact = feature(history, RollingMean(League(STAT), 1), available_at='available_at')
    np.testing.assert_allclose(exact.iloc[8:12], 25)


def test_team_contribution_and_whole_fixture_exclusions_have_different_moments():
    history = history_from_games(league_games())
    team = LeaveOneOut(League(STAT))
    fixture = LeaveOneOut(League(STAT), exclude='fixtures')
    result = evaluate_features(history, {'team': RollingMean(team, 1), 'fixture': RollingMean(fixture, 1),
        'std': RollingStd(team, 1), 'gate': RollingMean(team, 1, min_periods=4)})
    assert result.team.iloc[8] == 30  # A's 10 removed; C's 20 stays.
    assert result.fixture.iloc[8] == 35
    assert result['std'].iloc[8] == 10
    assert pd.isna(result.gate.iloc[8])


def test_loo_follows_match_window_without_refill():
    history = history_from_games(league_games())
    league = League(STAT, window_unit='matches')
    result = evaluate_features(history, {'team': RollingMean(LeaveOneOut(league), 1),
        'fixture': RollingMean(LeaveOneOut(league, exclude='fixtures'), 1)})
    assert result.team.iloc[10] == 40  # B's latest fixture only; no earlier observations refill it.
    assert pd.isna(result.fixture.iloc[10])


def test_match_totals_count_each_fixture_once_and_require_fixture_exclusion():
    history = history_from_games(league_games())
    league = League(STAT, unit='match')
    result = evaluate_features(history, {'mean': RollingMean(league, 1), 'std': RollingStd(league, 1),
        'loo': RollingMean(LeaveOneOut(league, exclude='fixtures'), 1)})
    assert result['mean'].iloc[8] == 50
    assert result['std'].iloc[8] == pytest.approx(np.std([30, 70], ddof=1))
    assert result.loo.iloc[8] == 70
    with pytest.raises(ValueError, match='indivisible'):
        feature(history, RollingMean(LeaveOneOut(league), 1))


def test_days_include_lower_boundary_and_exclude_current_kickoff():
    history = history_from_games(league_games())
    assert feature(history, RollingMean(League(STAT, schedule='kickoff', window_unit='days'), 2)).iloc[8] == 35


def test_round_window_crosses_seasons_but_not_competitions():
    games = league_games() + [dict(day=9, competition=20, values=(9999, 9999)),
                             dict(day=10, season=2, round=1, values=(9999, 9999))]
    history = history_from_games(games)
    assert feature(history, RollingMean(League(STAT), 1)).iloc[-2] == 65


def test_stage_keys_keep_repeated_round_numbers_distinct():
    games = [dict(day=0, round=1, stage='group', values=(2, 8)),
             dict(day=3, round=1, stage='final', values=(100, 200))]
    history = history_from_games(games)
    league = League(STAT, round_keys=('competition_id', 'season_id', 'stage', 'round'))
    assert feature(history, RollingMean(league, 1)).iloc[2] == 5


def test_missing_observations_use_finite_count_without_replacing_window_members():
    games = league_games()
    games[2]['values'], games[3]['values'] = (10, None), (float('inf'), 40)
    history = history_from_games(games)
    result = evaluate_features(history, {'mean': RollingMean(League(STAT), 1),
        'std': RollingStd(League(STAT), 1), 'gate': RollingMean(League(STAT), 1, min_periods=3)})
    assert result['mean'].iloc[8] == 25
    assert result['std'].iloc[8] == pytest.approx(np.std([10, 40], ddof=1))
    assert pd.isna(result.gate.iloc[8])


def test_league_z_uses_explicit_historical_numerator():
    history = history_from_games(league_games())
    z = RollingZScore(League(STAT), 2, reference=Lag(STAT))
    assert feature(history, z).iloc[8] == pytest.approx((10 - 15) / np.std([2, 8, 4, 6, 10, 20, 30, 40], ddof=1))
    with pytest.raises(ValueError, match='historical reference'):
        feature(history, RollingZScore(League(STAT), 2))


@pytest.mark.parametrize('reference', [STAT, H2H(STAT), WarmStart(STAT), H2H(WarmStart(STAT))])
@pytest.mark.parametrize('source', [STAT, League(STAT)])
def test_raw_z_reference_cannot_hide_inside_wrappers(reference, source):
    with pytest.raises(ValueError, match='historical or known context'):
        feature(history_from_games(league_games()), RollingZScore(source, 2, reference=reference))


def test_period_perspective_expansion_and_input_alignment_are_preserved():
    history = history_from_games(league_games())
    history.index = pd.Index([3, 3, 1, 7, 9, 2, 2, 5, 8, 4, 4, 0], name='user_index')
    before, attrs = history.copy(deep=True), deepcopy(history.attrs)
    population = League(ForAgainst(Stat(None, 'Match overview', 'cornerKicks'), 'both'))
    result = evaluate_features(history, {'mean': RollingMean(population, 1)})
    assert result.shape == (12, 4) and result.index.equals(history.index)
    np.testing.assert_allclose(result.iloc[8], [25, 2, 25, 2])
    pd.testing.assert_frame_equal(history, before)
    assert history.attrs == attrs


@pytest.mark.parametrize('league,window', [
    (League(STAT, unit='bad'), 1), (League(STAT, schedule='bad'), 1),
    (League(STAT, window_unit='bad'), 1), (League(STAT), 0), (League(STAT), True)])
def test_invalid_population_contract_is_rejected(league, window):
    with pytest.raises(ValueError):
        feature(history_from_games(league_games()), RollingMean(league, window))
