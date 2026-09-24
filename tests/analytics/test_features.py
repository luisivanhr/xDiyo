"""Independent feature arithmetic, perspective and chronological eligibility."""

from copy import deepcopy
import math

import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.features import (
    EMA, H2H, ForAgainst, IsHome, Lag, NormalizedStanding, RollingMean,
    RollingStd, RollingZScore, Stat, eligible_history_rows,
    evaluate_features, league_season_team_counts,
)


TEAM = 2**63 + 9
STAT = Stat('ALL', 'G', 'm')
FOR = 'team::ALL::G::m::value'
AGAINST = 'opponent::ALL::G::m::value'


@pytest.fixture
def history():
    values = [2, None, 8, 4, 4, float('inf'), 10, 999]
    frame = pd.DataFrame({
        'event_id': pd.array([2**53 + i for i in range(8)], dtype='int64[pyarrow]'),
        'team_id': pd.array([TEAM] * 8, dtype='uint64[pyarrow]'),
        'opponent_id': pd.array([TEAM + 1, TEAM + 2] * 4, dtype='uint64[pyarrow]'),
        'competition_id': [10] * 8, 'season_id': [1] * 4 + [2] * 4,
        'source_league': ['League'] * 8, 'source_season': ['23_24'] * 4 + ['24_25'] * 4,
        'kickoff_at': pd.date_range('2025-01-01 15:00', periods=8, tz='UTC'),
        'status': ['finished'] * 7 + ['notstarted'],
        'side': ['home', 'away', 'away', 'home', 'away', 'home', 'home', 'away'],
        'team_position': pd.array([1, 2, 3, None, 0, 4, float('inf'), 2], dtype='float64[pyarrow]'),
        'opponent_position': pd.array([3, 2, 1, None, 1, 2, 3, 99], dtype='float64[pyarrow]'),
    })
    metadata = {}
    for role, scale in [('team', 1), ('opponent', 10)]:
        for period, observed in [('ALL', values), ('1ST', list(range(1, 9)))]:
            column = f'{role}::{period}::G::m::value'
            frame[column] = pd.array([None if x is None else x * scale for x in observed], dtype='float64[pyarrow]')
            metadata[column] = {'role': role, 'period': period, 'group_name': 'G', 'key': 'm', 'field': 'value'}
    frame['team::ALL::G::m::total'] = pd.array([None if x is None else x * 100 for x in values], dtype='float64[pyarrow]')
    metadata['team::ALL::G::m::total'] = {'role': 'team', 'period': 'ALL', 'group_name': 'G', 'key': 'm', 'field': 'total'}
    frame['team::ALL::Other::m::value'] = 5000.0
    metadata['team::ALL::Other::m::value'] = {'role': 'team', 'period': 'ALL', 'group_name': 'Other', 'key': 'm', 'field': 'value'}
    frame.index = pd.Index([2**63 + x for x in [91, 11, 91, 7, 51, 5, 88, 2]], dtype='uint64', name='stored_row')
    frame.attrs = {'stat_columns': metadata, 'source': {'version': 'example'}}
    return frame


def assert_values(actual, expected):
    np.testing.assert_allclose(np.asarray(actual, dtype=float), expected, rtol=1e-12, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize('cutoff_form', ['aligned_series', 'column_name'])
def test_documented_two_day_cutoff_forms(history, cutoff_form):
    cutoffs = history['kickoff_at'] - pd.Timedelta(days=2)
    supplied = history
    if cutoff_form == 'column_name':
        supplied = history.assign(prediction_at=cutoffs)
        cutoffs = 'prediction_at'
    actual = evaluate_features(supplied, {'lag': Lag(STAT), 'mean': RollingMean(STAT, 3)},
                               cutoffs=cutoffs)
    assert_values(actual['lag'], [np.nan, np.nan, np.nan, 2, np.nan, 8, 4, 4])
    assert_values(actual['mean'], [np.nan, np.nan, np.nan, 2, 2, 5, 6, 16 / 3])
    assert actual.index.equals(history.index)
    assert actual.attrs['availability'] == 'earlier_finished_kickoff_proxy'


def test_lags_count_matches_and_do_not_skip_missing_values(history):
    actual = evaluate_features(history, {'lag1': Lag(STAT), 'lag2': Lag(STAT, 2)})
    assert_values(actual.lag1, [np.nan, 2, np.nan, 8, 4, 4, np.nan, 10])
    assert_values(actual.lag2, [np.nan, np.nan, 2, np.nan, 8, 4, 4, np.nan])


def test_partial_window_means_and_min_periods_use_finite_values_inside_match_window(history):
    actual = evaluate_features(history, {'mean': RollingMean(STAT, 3), 'three': RollingMean(STAT, 3, min_periods=3)})
    assert_values(actual['mean'], [np.nan, 2, 2, 5, 6, 16 / 3, 4, 7])
    assert_values(actual.three, [np.nan] * 5 + [16 / 3, np.nan, np.nan])


def test_std_ddof_and_included_latest_zscore_have_independent_numeric_answers(history):
    actual = evaluate_features(history, {'std': RollingStd(STAT, 3), 'pop': RollingStd(STAT, 3, ddof=0),
                                        'z': RollingZScore(STAT, 3), 'zero_z': RollingZScore(STAT, 2)})
    assert_values(actual['std'], [np.nan, np.nan, np.nan, math.sqrt(18), math.sqrt(8), 4 / math.sqrt(3), 0, math.sqrt(18)])
    assert_values(actual['pop'], [np.nan, 0, 0, 3, 2, math.sqrt(32 / 9), 0, 3])
    assert_values(actual.z, [np.nan, np.nan, np.nan, 1 / math.sqrt(2), -1 / math.sqrt(2), -1 / math.sqrt(3), np.nan, 1 / math.sqrt(2)])
    assert pd.isna(actual.zero_z.iloc[5])  # The two observed values are both 4.


def test_ema_initializes_first_observed_and_skips_missing_without_decay(history):
    actual = evaluate_features(history, {'ema': EMA(STAT, span=3), 'one': EMA(STAT, span=1),
                                        'three': EMA(STAT, span=3, min_periods=3)})
    assert_values(actual.ema, [np.nan, 2, 2, 5, 4.5, 4.25, 4.25, 7.125])
    assert_values(actual.one, [np.nan, 2, 2, 8, 4, 4, 4, 10])
    assert_values(actual.three, [np.nan, np.nan, np.nan, np.nan, 4.5, 4.25, 4.25, 7.125])


def test_for_against_and_none_period_expand_separate_source_identities(history):
    actual = evaluate_features(history, {'both': Lag(ForAgainst(Stat(None, 'G', 'm'), 'both')),
                                        'total': Lag(Stat('ALL', 'G', 'm', 'total'))})
    assert list(actual.columns) == [
        'both::team::ALL::G::m::value', 'both::team::1ST::G::m::value',
        'both::opponent::ALL::G::m::value', 'both::opponent::1ST::G::m::value', 'total',
    ]
    assert_values(actual.iloc[3], [8, 3, 80, 30, 800])
    against = evaluate_features(history, {'against': Lag(ForAgainst(STAT, 'against'))})
    assert_values(against.against, [np.nan, 20, np.nan, 80, 40, 40, np.nan, 100])


def test_default_groups_cross_seasons_and_venues_but_custom_groups_can_restrict_them(history):
    default = evaluate_features(history, {'lag': Lag(STAT)})
    seasonal = evaluate_features(history, {'lag': Lag(STAT)}, group_by=['team_id', 'competition_id', 'season_id'])
    venue = evaluate_features(history, {'lag': Lag(STAT)}, group_by=['team_id', 'competition_id', 'side'])
    assert default.lag.iloc[4] == 4 and pd.isna(seasonal.lag.iloc[4])
    assert default.lag.iloc[3] == 8 and venue.lag.iloc[3] == 2
    other_competition = history.copy(deep=True)
    other_competition.iloc[2, other_competition.columns.get_loc('competition_id')] = 20
    assert pd.isna(evaluate_features(other_competition, {'lag': Lag(STAT)}).lag.iloc[3])
    assert evaluate_features(other_competition, {'lag': Lag(STAT)}, group_by='team_id').lag.iloc[3] == 8


def adversarial(history):
    frame = history.copy(deep=True)
    days = [3, 1, 2, 3, 4, None, 5, 6]
    frame['kickoff_at'] = pd.to_datetime([f'2025-01-{day:02d} 15:00+00:00' if day else None for day in days])
    frame['status'] = ['finished', 'finished', 'postponed', 'finished', None, 'finished', 'finished', 'notstarted']
    frame[FOR] = [300.0, 10, 999, 301, 999, 999, 50, 999]
    return frame


def test_eligibility_strict_times_finished_rows_missing_dates_and_input_tie_order(history):
    frame = adversarial(history)
    eligible = eligible_history_rows(frame)
    assert eligible[0].tolist() == [1] and eligible[3].tolist() == [1]
    assert eligible[5].tolist() == []
    assert eligible[6].tolist() == [1, 0, 3]
    assert eligible[7].tolist() == [1, 0, 3, 6]
    actual = evaluate_features(frame, {'lag': Lag(STAT), 'mean': RollingMean(STAT, 10)})
    assert actual.lag.iloc[7] == 50 and actual['mean'].iloc[7] == 661 / 4
    contaminated = frame.copy(deep=True)
    contaminated.iloc[[2, 4, 5, 7], contaminated.columns.get_loc(FOR)] = 1e12
    pd.testing.assert_frame_equal(evaluate_features(contaminated, {'lag': Lag(STAT), 'mean': RollingMean(STAT, 10)}), actual)


def test_frozen_round_cutoffs_exclude_later_matches_in_the_same_round(history):
    frame = adversarial(history)
    frozen = pd.Timestamp('2025-01-04 15:00', tz='UTC')
    frame['round_cutoff'] = frame.kickoff_at.where(frame.kickoff_at < frozen, frozen)
    frame.loc[frame.kickoff_at.isna(), 'round_cutoff'] = pd.NaT
    eligible = eligible_history_rows(frame, cutoffs='round_cutoff')
    assert eligible[6].tolist() == eligible[7].tolist() == [1, 0, 3]
    actual = evaluate_features(frame, {'lag': Lag(STAT)}, cutoffs=frame.round_cutoff)
    assert actual.lag.iloc[6] == actual.lag.iloc[7] == 301
    assert actual.attrs['availability'] == 'earlier_finished_kickoff_proxy'


def test_explicit_delayed_availability_includes_equality_and_requires_known_time(history):
    frame = adversarial(history)
    frame['available_at'] = frame.kickoff_at.astype('datetime64[us, UTC]')
    frame.iloc[0, frame.columns.get_loc('available_at')] = pd.Timestamp('2025-01-07 15:00', tz='UTC')
    frame.iloc[1, frame.columns.get_loc('available_at')] = pd.Timestamp('2025-01-06 15:00', tz='UTC')
    frame.iloc[3, frame.columns.get_loc('available_at')] = pd.Timestamp('2025-01-06 15:00', tz='UTC')
    frame.iloc[6, frame.columns.get_loc('available_at')] = pd.NaT
    eligible = eligible_history_rows(frame, available_at='available_at')
    assert eligible[7].tolist() == [1, 3]
    actual = evaluate_features(frame, {'lag': Lag(STAT)}, available_at=frame.available_at)
    assert actual.lag.iloc[7] == 301 and actual.attrs['availability'] == 'explicit'


def test_target_match_is_excluded_by_partition_identity(history):
    frame = history.copy(deep=True)
    frame.iloc[0, frame.columns.get_loc('event_id')] = frame.event_id.iloc[7]
    # Same event number in another season is a different match.
    assert 0 in eligible_history_rows(frame)[7]
    frame.iloc[0, frame.columns.get_loc('source_season')] = frame.source_season.iloc[7]
    assert 0 not in eligible_history_rows(frame)[7]


def h2h_history():
    rows = []
    for day, home, away, home_value, away_value in [(1, 1, 2, 4, 1), (2, 1, 3, 8, 2), (3, 2, 1, 6, 9), (4, 1, 2, 100, 200)]:
        for side, team, opponent, own, other in [('home', home, away, home_value, away_value), ('away', away, home, away_value, home_value)]:
            rows.append({'event_id': day, 'team_id': team, 'opponent_id': opponent, 'competition_id': 10,
                         'season_id': 1, 'side': side, 'status': 'finished',
                         'kickoff_at': pd.Timestamp(f'2025-01-0{day}', tz='UTC'), FOR: float(own), AGAINST: float(other)})
    frame = pd.DataFrame(rows)
    frame.attrs = {'stat_columns': {name: {'role': role, 'period': 'ALL', 'group_name': 'G', 'key': 'm', 'field': 'value'}
                                    for role, name in [('team', FOR), ('opponent', AGAINST)]}}
    return frame


def test_ordered_h2h_preserves_orientation_when_home_and_away_reverse():
    frame = h2h_history()
    actual = evaluate_features(frame, {'h2h': Lag(H2H(ForAgainst(STAT, 'both'))),
                                        'wrapped': H2H(Lag(STAT)), 'general': Lag(STAT)})
    assert_values(actual.iloc[4], [1, 4, 1, 1])  # B now hosts A, retaining B's perspective.
    assert_values(actual.iloc[5], [4, 1, 4, 8])  # A's last general game was against C.
    assert_values(actual.iloc[6], [9, 6, 9, 9])
    assert_values(actual.iloc[7], [6, 9, 6, 6])
    nested = evaluate_features(frame, {'nested': RollingMean(H2H(Lag(STAT)), window=2)})
    assert pd.isna(nested.nested.iloc[5]) and nested.nested.iloc[6] == 4


def test_nested_children_use_each_historical_rows_own_cutoff(history):
    frame = history.iloc[:4].copy()
    frame[FOR] = [10.0, 20, 30, 40]
    cutoffs = frame.kickoff_at.copy()
    cutoffs.iloc[1] = cutoffs.iloc[0]
    actual = evaluate_features(frame, {'nested': RollingMean(Lag(STAT), window=2)}, cutoffs=cutoffs)
    assert_values(actual.nested, [np.nan, np.nan, np.nan, 20])


def test_normalized_standings_count_full_membership_and_default_missing_to_zero(history):
    counts = league_season_team_counts(history)
    assert counts.to_dict() == {(10, 1): 3, (10, 2): 3}
    actual = evaluate_features(history, {'home': IsHome(), 'rank': NormalizedStanding(side='both')}, team_counts=counts)
    assert_values(actual.home, [1, 0, 0, 1, 0, 1, 1, 0])
    assert_values(actual['rank::team_standing'], [1, .5, 0, 0, 0, 0, 0, .5])
    assert_values(actual['rank::opponent_standing'], [0, .5, 1, 0, 1, .5, 0, 0])
    split = history.iloc[[0, 7]].copy()
    with_full_counts = evaluate_features(split, {'rank': NormalizedStanding()}, team_counts=counts)
    assert_values(with_full_counts['rank'], [1, .5])
    assert evaluate_features(split, {'rank': NormalizedStanding()})['rank'].iloc[-1] == 0


@pytest.mark.parametrize('denominator', [0, 1, np.nan, np.inf])
def test_invalid_standings_denominators_use_configured_default(history, denominator):
    counts = league_season_team_counts(history).astype(float)
    counts[:] = denominator
    actual = evaluate_features(history, {'zero': NormalizedStanding(), 'custom': NormalizedStanding(missing_value=-7)}, team_counts=counts)
    assert_values(actual.zero, [0] * 8)
    assert_values(actual.custom, [-7] * 8)


def test_context_nodes_allow_current_match_and_missing_position_column(history):
    frame = history.drop(columns=['status', 'kickoff_at', 'team_position', 'opponent_position'])
    actual = evaluate_features(frame, {'home': IsHome(), 'standing': NormalizedStanding()})
    assert_values(actual.standing, [0] * 8)
    assert actual.home.iloc[-1] == 0


def test_output_index_large_ids_input_data_and_metadata_are_preserved(history):
    before = history.copy(deep=True)
    attrs = deepcopy(history.attrs)
    expressions = {'lag': Lag(STAT), 'mean': RollingMean(STAT), 'home': IsHome()}
    actual = evaluate_features(history, expressions)
    assert actual.index.equals(history.index) and actual.index.dtype == history.index.dtype
    assert history.team_id.iloc[0] == TEAM
    pd.testing.assert_frame_equal(history, before, check_exact=True)
    assert history.attrs == attrs
    assert list(actual) == list(expressions)
    assert actual.attrs['group_by'] == ('team_id', 'competition_id')


@pytest.mark.parametrize('root', [STAT, ForAgainst(STAT, 'both'), H2H(STAT), H2H(ForAgainst(STAT))])
def test_observed_current_stat_roots_are_rejected(history, root):
    with pytest.raises(ValueError, match='Observed Stat values need'):
        evaluate_features(history, {'unsafe': root})


def test_time_alignment_cutoff_and_group_errors_are_explicit(history):
    with pytest.raises(ValueError, match='align'):
        evaluate_features(history, {'lag': Lag(STAT)}, cutoffs=history.kickoff_at.reset_index(drop=True))
    with pytest.raises(TypeError, match='numeric timestamps'):
        evaluate_features(history, {'lag': Lag(STAT)}, available_at=np.arange(8))
    with pytest.raises(ValueError, match='must not follow'):
        evaluate_features(history, {'lag': Lag(STAT)}, cutoffs=history.kickoff_at + pd.Timedelta(seconds=1))
    with pytest.raises(ValueError, match='include team_id'):
        evaluate_features(history, {'lag': Lag(STAT)}, group_by=['competition_id'])
    with pytest.raises(KeyError, match='match status'):
        evaluate_features(history.drop(columns='status'), {'lag': Lag(STAT)})


@pytest.mark.parametrize('node', [Lag(STAT, 0), RollingMean(STAT, 2, min_periods=3),
                                  RollingStd(STAT, ddof=-1), EMA(STAT, span=0)])
def test_invalid_temporal_parameters_fail(history, node):
    with pytest.raises(ValueError):
        evaluate_features(history, {'invalid': node})


def test_unavailable_statistic_invalid_orientation_and_output_collisions_fail(history):
    with pytest.raises(KeyError, match='Statistic is absent'):
        evaluate_features(history, {'missing': Lag(Stat('ALL', 'G', 'unknown'))})
    with pytest.raises(ValueError, match='side must be'):
        evaluate_features(history, {'invalid': Lag(ForAgainst(STAT, 'home'))})
    with pytest.raises(ValueError, match='duplicate output columns'):
        evaluate_features(history, {'x': Lag(ForAgainst(STAT, 'both')), f'x::{FOR}': IsHome()})


def test_empty_history_preserves_empty_index_and_named_columns(history):
    frame = history.iloc[:0].copy()
    result = evaluate_features(frame, {'lag': Lag(STAT), 'home': IsHome()})
    assert result.empty and result.index.equals(frame.index) and list(result) == ['lag', 'home']
