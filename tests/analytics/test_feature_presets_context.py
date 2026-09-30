"""Ordered bank composition and known-context temporal eligibility."""
import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.features import (FeatureBankPreset, RestDays, CalendarFeature,
    evaluate_context_features, evaluate_features, H2H, Lag, ForAgainst, Stat,
    RollingMean, Difference, Column, combine_features)
from notebooks.helpers.quantile_corners import build_feature_definitions
from transition_samples import history_from_games, warm_games, league_games


def test_preset_preserves_notebook_order_and_all_single_stat_formulas():
    identities = [('ALL', 'Match overview', 'cornerKicks'),
                  ('1ST', 'Match overview', 'cornerKicks')]
    preset = FeatureBankPreset()
    definitions = preset.build(identities)
    assert list(definitions.items()) == list(build_feature_definitions(identities).items())
    prefix = 'ALL_Match overview_cornerKicks_for'
    assert list(definitions)[:12] == [f'{prefix}_{tail}' for tail in
        ('lag1', 'lag2', 'lag3', 'mean3', 'mean5', 'mean10', 'mean20',
         'std5', 'std10', 'z5', 'ema5', 'ema10')]
    assert definitions[prefix + '_lag3'] == Lag(ForAgainst(Stat(*identities[0]), 'for'), 3)
    assert definitions[prefix + '_mean20'] == RollingMean(ForAgainst(Stat(*identities[0]), 'for'), 20)
    assert sum(name.startswith('1ST_') for name in definitions) == 6
    assert not any('ballPossession' in name for name in definitions)


def test_custom_windows_and_postassembly_arithmetic_preserve_original_formulas():
    preset = FeatureBankPreset(stats=(('Match overview', 'cornerKicks'),),
        std_windows=(7,), z_windows=(9,), half_lags=(2,), half_windows=(4,))
    definitions = preset.build([('ALL', 'Match overview', 'cornerKicks')])
    assert 'ALL_Match overview_cornerKicks_for_std7' in definitions
    assert 'ALL_Match overview_cornerKicks_for_z9' in definitions
    stem = 'ALL_Match overview_cornerKicks_for_mean'
    history_columns = [stem + '3', stem + '5', stem + '20', 'rest_days']
    columns = [f'{side}::{name}' for side in ('home', 'away') for name in history_columns]
    frame = pd.DataFrame([[3, 5, 20, 8, 4, 6, 21, 9]], columns=columns)
    derived = preset.postassembly_definitions(history_columns, columns)
    result = combine_features(frame, derived)
    assert result[f'sum::{stem}5'].iloc[0] == 11
    assert result['difference::rest_days'].iloc[0] == -1
    assert result[f'trend::home::{stem}3_vs_20'].iloc[0] == -17
    assert list(derived)[-2:] == [f'trend::{side}::{stem}3_vs_20' for side in ('home', 'away')]
    assert FeatureBankPreset(include_combinations=False).postassembly_definitions([], []) == {}
    assert FeatureBankPreset(include_calendar=False).calendar_definitions() == {}


def test_rest_days_obeys_cutoffs_availability_h2h_and_missing_history():
    history = history_from_games(league_games())
    actual = evaluate_features(history, {'rest': RestDays(), 'h2h_rest': H2H(RestDays())})
    assert np.isnan(actual.rest.iloc[0])
    assert actual.rest.iloc[4] == (history.kickoff_at.iloc[4] - history.kickoff_at.iloc[0]).total_seconds() / 86400
    assert np.isnan(actual.h2h_rest.iloc[4])  # first A-C encounter
    cutoff = pd.Series(history.kickoff_at.min(), index=history.index)
    assert evaluate_features(history, {'rest': RestDays()}, cutoffs=cutoff).rest.isna().all()
    available = history.kickoff_at + pd.Timedelta(days=100)
    assert evaluate_features(history, {'rest': RestDays()}, available_at=available).rest.isna().all()
    altered = history.copy()
    altered['result'] = 'W'
    pd.testing.assert_frame_equal(actual, evaluate_features(altered, {'rest': RestDays(), 'h2h_rest': H2H(RestDays())}))


def test_calendar_is_identical_on_history_or_metadata_and_preserves_duplicate_index():
    data = pd.DataFrame({'kickoff_at': pd.to_datetime(['2026-01-05T01:00Z', None, '2026-07-07T01:00Z']),
                         'round': ['3', 'final', 4]}, index=[8, 8, 1])
    definitions = FeatureBankPreset().calendar_definitions()
    actual = evaluate_context_features(data, definitions)
    assert actual.index.equals(data.index)
    np.testing.assert_allclose(actual['calendar::month_sin'], [0.5, np.nan, -0.5], equal_nan=True)
    np.testing.assert_allclose(actual['calendar::weekday'], [0, np.nan, 1], equal_nan=True)
    np.testing.assert_allclose(actual['calendar::round'], [3, np.nan, 4], equal_nan=True)
    pd.testing.assert_frame_equal(actual, evaluate_features(data, definitions), check_flags=False)
    with pytest.raises(ValueError, match='Calendar kind'):
        evaluate_context_features(data, {'bad': CalendarFeature('observed_goals')})


def test_hierarchical_inventory_reconciles_counts_and_retains_unknown_columns():
    columns = ['home::ALL_Match overview_cornerKicks_for_mean5',
               'away::ALL_Match overview_cornerKicks_against_mean5',
               'sum::warm::ALL_Match overview_cornerKicks_for_mean5',
               'calendar::weekday', 'league::Premier_League', 'custom_unknown']
    actual = FeatureBankPreset().inventory(columns)
    assert actual['total_columns'] == sum(group['count'] for group in actual['groups']) == 6
    assert actual['unclassified_columns'] == 1
    direct = actual['groups'][0]
    assert direct['count'] == 2 and direct['roles'] == ['against', 'for']
    assert direct['venues'] == ['away', 'home']
    assert actual['ordered_columns_sha256'] != FeatureBankPreset().inventory(columns[::-1])['ordered_columns_sha256']
    with pytest.raises(ValueError, match='unique'):
        FeatureBankPreset().inventory(columns + [columns[0]])
