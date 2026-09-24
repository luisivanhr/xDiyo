"""Notebook feature choices reuse temporal populations and transition policies."""
import numpy as np
import pandas as pd
import pytest

from notebooks.helpers.quantile_corners import build_feature_definitions
from xdiyo_analytics.features import SeededEMA, Hard, evaluate_features
from xdiyo_analytics.ratings import GlickoTransition
from transition_samples import history_from_games, warm_games, league_games, mover_games, movement, OWN, OTHER

IDENTITIES = [('ALL', 'Match overview', 'cornerKicks')]


def definitions(**overrides):
    options = dict(periods=('ALL',), windows=(2,), lags=(1,), spans=(2,),
        loo_windows=(3, 5), loo_reducers=('mean', 'std', 'z'),
        h2h_windows=(3, 5), h2h_reducers=('mean', 'std', 'z'),
        warm_policy=SeededEMA(alpha=.5, handoff=Hard(1)),
        rating_warm_policy=GlickoTransition(phi_scale=1.1))
    return build_feature_definitions(IDENTITIES, **{**options, **overrides})


def test_default_baseline_preserved_and_switches_remove_only_requested_families():
    baseline = build_feature_definitions(IDENTITIES, periods=('ALL',), windows=(2,), lags=(1,), spans=(2,))
    extended = definitions()
    for name, expression in baseline.items():
        assert extended[name] == expression
    assert len(extended) == len(set(extended))
    assert 'loo_corners_std3' in extended and 'h2h_corners_against_z5' in extended
    assert 'warm::result_glicko' in extended and 'warm::corners_glicko' in extended
    disabled = definitions(include_loo=False, include_h2h=False, warm_policy=None, rating_warm_policy=None)
    assert not any('loo_' in name or 'h2h_' in name or name.startswith('warm::') for name in disabled)


def test_loo_uses_individual_samples_and_h2h_only_previous_encounters():
    history = history_from_games(league_games())
    bank = definitions(include_ratings=False, warm_policy=None)
    result = evaluate_features(history, bank)
    # A's next prediction: round 1 contributes B=8,C=4,D=6 after A=2 is excluded.
    assert result.loo_corners_mean3.iloc[4] == 6
    assert result.loo_corners_std3.iloc[4] == 2
    assert result.loo_corners_z3.iloc[4] == -2  # A's latest 2 vs LOO mean6/std2
    # A-C have not met before their round-2 encounter.
    assert pd.isna(result.h2h_corners_for_mean3.iloc[4])


def test_warm_actual_bank_no_prior_handoff_and_future_outcome_isolation():
    history = history_from_games(warm_games())
    bank = definitions()
    result = evaluate_features(history, bank)
    mean = 'ALL_Match overview_cornerKicks_for_mean2'
    std = 'ALL_Match overview_cornerKicks_for_std5'
    assert pd.isna(result[f'warm::{mean}'].iloc[0])
    assert result[f'warm::{mean}'].iloc[2] == result[mean].iloc[2] == 2
    assert result[f'warm::{mean}'].iloc[4] == 3  # previous-season team mean
    assert result[f'warm::{std}'].iloc[4] == pytest.approx(np.sqrt(5))
    assert result[f'warm::{mean}'].iloc[6] == result[mean].iloc[6] == 7  # one completed round
    assert result['warm::h2h_corners_for_mean3'].iloc[4] == 3
    assert pd.isna(result['warm::loo_corners_mean3'].iloc[0])
    # Identical initial ratings without prior history; season-entry uncertainty inflates.
    rating_base = [c for c in result if c.startswith('result_glicko') and c.endswith('::rd')]
    rating_warm = [c for c in result if c.startswith('warm::result_glicko') and c.endswith('::rd')]
    assert len(rating_base) == len(rating_warm) == 1
    assert result[rating_warm[0]].iloc[0] == result[rating_base[0]].iloc[0]
    assert result[rating_warm[0]].iloc[4] > result[rating_base[0]].iloc[4]
    changed = history.copy()
    changed.loc[4:, [OWN, OTHER]] = 9999.
    changed.loc[4:, 'result'] = 'D'
    changed_result = evaluate_features(changed, bank)
    pd.testing.assert_frame_equal(result.iloc[:6], changed_result.iloc[:6])
    # Prediction two days earlier cannot consume the match at that exact cutoff.
    early = evaluate_features(history, bank, cutoffs=history.kickoff_at - pd.Timedelta(days=2))
    assert early[f'warm::{mean}'].iloc[6] == 3


def test_warm_stat_filter_keeps_unselected_stats_unwarmed():
    bank = build_feature_definitions(IDENTITIES + [('ALL', 'Match overview', 'ballPossession')],
        warm_policy=SeededEMA(), warm_stat_keys=('cornerKicks',), rating_warm_policy=None)
    assert any('warm::' in key and 'cornerKicks' in key for key in bank)
    assert not any('warm::' in key and 'ballPossession' in key for key in bank)
    assert any('ballPossession' in key for key in bank)


def test_explicit_movement_uses_prior_team_and_destination_league_before_entry():
    history = history_from_games(mover_games())
    bank = definitions()
    result = evaluate_features(history, bank, team_seasons=movement())
    # Previous-team mean3 and previous destination mean25 receive equal weight.
    name = 'warm::ALL_Match overview_cornerKicks_for_mean2'
    assert result[name].iloc[8] == 14
    changed = history.copy()
    changed.loc[8:, [OWN, OTHER]] = 9999.
    pd.testing.assert_frame_equal(result.iloc[:10],
        evaluate_features(changed, bank, team_seasons=movement()).iloc[:10])
