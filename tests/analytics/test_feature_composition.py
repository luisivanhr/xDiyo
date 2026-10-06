"""Reusable arithmetic, historical eligibility and fitted categorical identity."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.features import (Column, Constant, Sum, Product, Difference, Ratio,
    combine_features, IdentityIndicators, Lag, RollingMean, ForAgainst, Stat,
    evaluate_features)
from xdiyo_analytics.labels import MatchTotal, create_labels
from xdiyo_analytics.datasets import assemble_dataset
from transition_samples import STAT, OWN, OTHER, history_from_games, warm_games


def test_frame_arithmetic_matches_manual_and_preserves_row_identity_and_source():
    frame = pd.DataFrame({'home': [4., 7., np.nan], 'away': [2., 3., 6.]}, index=[8, 8, 1])
    frame.attrs = {'provenance': {'season': 'sample'}}
    before = deepcopy(frame)
    actual = combine_features(frame, {'total': Sum(Column('home'), Column('away')),
        'difference': Difference(Column('home'), Column('away')),
        'nested': Ratio(Sum(Column('home'), Constant(2)), Difference(Column('away'), 1))})
    np.testing.assert_allclose(actual.total, [6, 10, np.nan], equal_nan=True)
    np.testing.assert_allclose(actual.difference, [2, 4, np.nan], equal_nan=True)
    np.testing.assert_allclose(actual.nested, [6, 4.5, np.nan], equal_nan=True)
    assert actual.index.equals(frame.index)
    assert actual.attrs['provenance'] == frame.attrs['provenance']
    pd.testing.assert_frame_equal(frame, before)
    assert list(combine_features(frame, {'total': Sum(Column('home'), 1)}, keep_existing=False)) == ['total']
    with pytest.raises(ValueError):
        combine_features(frame, {'home': Sum(Column('home'), 1)})


def test_ratio_zero_policy_does_not_impute_missing_or_nonfinite_inputs():
    frame = pd.DataFrame({'a': [4., 0., 4., np.nan, np.inf, 6.],
                          'b': [2., 0., 0., 0., 2., np.inf]})
    actual = combine_features(frame, {'ratio': Ratio(Column('a'), Column('b')),
        'fallback': Ratio(Column('a'), Column('b'), zero_value=-1)})
    np.testing.assert_allclose(actual.ratio, [2, np.nan, np.nan, np.nan, np.nan, np.nan], equal_nan=True)
    np.testing.assert_allclose(actual.fallback, [2, -1, -1, np.nan, np.nan, np.nan], equal_nan=True)
    overflow = combine_features(pd.DataFrame({'x': [1e308]}), {'sum': Sum(Column('x'), Column('x'))})
    assert pd.isna(overflow['sum'].iloc[0])


def test_product_nested_constants_missing_overflow_and_alignment():
    from xdiyo_analytics.features.composition import arithmetic_frame
    frame = pd.DataFrame({'a': [2., -3., 0., np.nan, np.inf, 1e308],
                          'b': [4., 2., 5., 0., 0., 2.]}, index=[9, 2, 2, 7, 4, 3])
    before = frame.copy(deep=True)
    product = Product(Column('a'), Column('b'))
    actual = combine_features(frame, {'p': product, 'scaled': Product(product, Constant(-2))})
    np.testing.assert_allclose(actual.p, [8, -6, 0, np.nan, np.nan, np.nan], equal_nan=True)
    np.testing.assert_allclose(actual.scaled, [-16, 12, 0, np.nan, np.nan, np.nan], equal_nan=True)
    assert actual.index.equals(frame.index)
    pd.testing.assert_frame_equal(frame, before)
    with pytest.raises(ValueError, match='identical row indices'):
        arithmetic_frame(product, frame[['a']], frame[['b']].reset_index(drop=True))


def test_temporal_arithmetic_keeps_cutoffs_and_rejects_raw_or_multicolumn_operands():
    history = history_from_games(warm_games())
    own, against = RollingMean(STAT, 2), RollingMean(ForAgainst(STAT, 'against'), 2)
    expressions = {'total': Sum(own, against), 'difference': Difference(own, against),
        'ratio': Ratio(own, against), 'product': Product(own, against),
        'nested': RollingMean(Difference(Lag(STAT), 1), 2)}
    actual = evaluate_features(history, expressions)
    assert actual.total.iloc[4] == 10
    assert actual.difference.iloc[4] == -4
    assert actual.ratio.iloc[4] == pytest.approx(3 / 7)
    assert actual['product'].iloc[4] == 21
    assert actual.nested.iloc[4] == 1
    changed = history.copy()
    changed.loc[4:, [OWN, OTHER]] = 99999.
    pd.testing.assert_frame_equal(actual.iloc[:6], evaluate_features(changed, expressions).iloc[:6])
    early = evaluate_features(history, expressions, cutoffs=history.kickoff_at - pd.Timedelta(days=2))
    assert early.total.iloc[6] == 10
    with pytest.raises((ValueError, TypeError)):
        evaluate_features(history, {'raw': Sum(STAT, Constant(0))})
    with pytest.raises((ValueError, TypeError)):
        evaluate_features(history, {'ambiguous': Sum(RollingMean(Stat(None, 'Match overview', 'cornerKicks'), 2), 1)})
    with pytest.raises(ValueError, match='Observed Stat'):
        evaluate_features(history, {'raw': Product(STAT, Constant(0))})
    with pytest.raises(ValueError, match='one output column'):
        evaluate_features(history, {'ambiguous': Product(RollingMean(Stat(None, 'Match overview', 'cornerKicks'), 2), 1)})


def test_composed_keyed_frames_still_join_by_ids_after_shuffle():
    history = history_from_games(warm_games())
    keyed = evaluate_features(history, {'form': RollingMean(STAT, 2)}, keyed=True)
    combined = combine_features(keyed, {'plus_one': Sum(Column('form'), 1)})
    assert combined.index.equals(keyed.index)
    labels = create_labels(history, {'corners': MatchTotal(STAT)})['corners']
    direct = assemble_dataset(combined, labels, layout='match')
    shuffled = assemble_dataset(combined.iloc[::-1], labels, layout='match')
    pd.testing.assert_frame_equal(direct.X, shuffled.X)
    np.testing.assert_allclose(direct.X['home::plus_one'], direct.X['home::form'] + 1, equal_nan=True)


def test_identity_encoder_fit_only_exact_large_ids_and_unknown_missing_zero():
    first, second = 2**63 + 9, 2**63 + 10
    train = pd.DataFrame({'team_id': pd.array([second, first, first], dtype='uint64[pyarrow]')})
    evaluation = pd.DataFrame({'team_id': pd.array([first, second, first + 100, None], dtype='uint64[pyarrow]')}, index=[4, 4, 7, 2])
    encoder = IdentityIndicators(columns=('team_id',), prefixes={'team_id': 'team'})
    before = deepcopy(train)
    encoder.fit(train)
    categories = deepcopy(encoder.categories_)
    actual = encoder.transform(evaluation)
    assert set(actual.columns) == {f'team::{first}', f'team::{second}'}
    assert actual[f'team::{first}'].tolist() == [1., 0., 0., 0.]
    assert actual[f'team::{second}'].tolist() == [0., 1., 0., 0.]
    assert actual.index.equals(evaluation.index)
    assert encoder.categories_ == categories
    pd.testing.assert_frame_equal(train, before)


def test_league_indicator_parity_with_previous_notebook_columns():
    metadata = pd.DataFrame({'source_league': ['Beta', 'Alpha', 'Beta', 'Unseen', None]})
    encoder = IdentityIndicators(columns=('source_league',), prefixes={'source_league': 'league'}).fit(
        metadata.iloc[:3].sort_values('source_league', kind='stable'))
    actual = encoder.transform(metadata)
    expected = pd.DataFrame({f'league::{value}': metadata.source_league.eq(value).astype(float)
                             for value in ['Alpha', 'Beta']})
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)
