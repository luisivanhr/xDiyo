"""Fixture scope is semantic; collapsing requires paired values and timestamps."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.datasets import assemble_dataset
from xdiyo_analytics.features import (
    BayesianFixture, BayesianRating, Constant, Difference, IsHome, Lag, Ratio,
    evaluate_features,
)
from xdiyo_analytics.histories import build_team_history
from xdiyo_analytics.labels import Outcome, create_labels
from test_match_score import score_data


@pytest.fixture
def prepared():
    history = build_team_history(score_data(3))
    features = evaluate_features(history, {
        'odds': BayesianFixture(fields=('p_draw', 'expected_home_goals')),
        'strength': BayesianRating(side='for', fields=('attack_mean',)),
        'venue': IsHome(),
    }, keyed=True)
    label = create_labels(history, {'target': Outcome(perspective='home')})['target']
    return history, features, label


@pytest.mark.parametrize('explicit', [False, True])
def test_mixed_real_bayesian_outputs_collapse_only_fixture_columns(prepared, explicit):
    history, features, label = prepared
    before = deepcopy(features)
    features = features.iloc[::-1].copy()
    if explicit:
        features = features.reset_index()
    result = assemble_dataset(features, label, layout='match')
    assert list(result.X) == [
        'home::strength', 'home::venue', 'away::strength', 'away::venue',
        'fixture::odds::p_draw', 'fixture::odds::expected_home_goals',
    ]
    for side in ('home', 'away'):
        expected = before.loc[before.index.get_level_values('side') == side, 'strength']
        np.testing.assert_array_equal(result.X[f'{side}::strength'], expected)
    expected = before.loc[before.index.get_level_values('side') == 'home', 'odds::p_draw']
    np.testing.assert_array_equal(result.X['fixture::odds::p_draw'], expected)
    assert result.definitions['feature_scopes']['fixture::odds::p_draw'] == 'fixture'
    assert result.definitions['feature_scopes']['away::strength'] == 'team'
    pd.testing.assert_frame_equal(prepared[1], before)


def test_selection_order_and_single_field_name(prepared):
    history, features, label = prepared
    result = assemble_dataset(features, label, layout='match', feature_columns=[
        'odds::expected_home_goals', 'venue', 'odds::p_draw', 'strength'])
    assert list(result.X) == ['home::venue', 'home::strength', 'away::venue', 'away::strength',
                              'fixture::odds::expected_home_goals', 'fixture::odds::p_draw']
    single = evaluate_features(history, {'draw': BayesianFixture(fields=('p_draw',))}, keyed=True)
    assert list(assemble_dataset(single, label, layout='match').X) == ['fixture::draw']
    empty_label = create_labels(history.iloc[:0], {'target': Outcome(perspective='home')})['target']
    empty = assemble_dataset(single.iloc[:0], empty_label, layout='match')
    assert empty.X.empty and list(empty.X) == ['fixture::draw']


def test_fixture_metadata_roundtrip_and_label_subset_order(prepared, tmp_path):
    from dataset_samples import reorder_label
    history, features, label = prepared
    path = tmp_path / 'features.parquet'
    features.to_parquet(path)
    restored = pd.read_parquet(path).iloc[::-1]
    selected = reorder_label(label, [2, 0])
    actual = assemble_dataset(restored, selected, layout='match')
    expected = assemble_dataset(features, label, layout='match')
    pd.testing.assert_frame_equal(actual.X, expected.X.iloc[[2, 0]].reset_index(drop=True))
    # Extra records need no cutoff when they are outside the selected labels.
    ignored_event = label.metadata.event_id.iloc[1]
    restored.attrs['prediction_cutoffs']['records'] = [
        r for r in restored.attrs['prediction_cutoffs']['records'] if r['event_id'] != ignored_event]
    pd.testing.assert_frame_equal(assemble_dataset(restored, selected, layout='match').X, actual.X)


def test_explicit_scope_contract_and_nullable_fixture_dtype(prepared):
    _, features, label = prepared
    # A custom producer need not be recognizable by its expression or name.
    features.attrs.pop('features')
    features['odds::p_draw'] = pd.array([pd.NA] * len(features), dtype='Float64')
    result = assemble_dataset(features, label, layout='match')
    assert result.X['fixture::odds::p_draw'].dtype == pd.Float64Dtype()
    assert result.X['fixture::odds::p_draw'].isna().all()
    features.attrs['feature_scopes']['strength'] = 'unknown'
    with pytest.raises(ValueError, match='Feature scope must be team or fixture'):
        assemble_dataset(features, label, layout='match')


def test_equal_values_and_absent_scope_metadata_do_not_deduplicate(prepared):
    history, features, label = prepared
    equal = evaluate_features(history, {'one': Constant(1)}, keyed=True)
    assert list(assemble_dataset(equal, label, layout='match').X) == ['home::one', 'away::one']
    features.attrs.pop('feature_scopes')
    features.attrs.pop('prediction_cutoffs')
    assert len(assemble_dataset(features, label, layout='match').X.columns) == 8


def test_team_rows_keep_fixture_copies_and_allow_different_cutoffs(prepared):
    history, features, _ = prepared
    labels = create_labels(history, {'target': Outcome(perspective='team')})['target']
    features.iloc[1, 0] = 0.99
    features.attrs['prediction_cutoffs']['records'][1]['cutoff'] = None
    result = assemble_dataset(features, labels, layout='team_match')
    pd.testing.assert_frame_equal(result.X, features.reset_index(drop=True))
    assert result.X.shape == features.shape


@pytest.mark.parametrize('left,right,valid', [
    (np.nan, np.nan, True), (pd.NA, None, True), (1., 1., True),
    (np.nan, 1., False), (1., pd.NA, False), (1., 1.+1e-12, False),
])
def test_fixture_value_validation_including_missing(prepared, left, right, valid):
    _, features, label = prepared
    features['odds::p_draw'] = features['odds::p_draw'].astype(object)
    # Explicit identity selection avoids assuming how team rows are ordered.
    event = label.metadata.event_id.iloc[0]
    rows = np.flatnonzero(features.index.get_level_values('event_id') == event)
    features.iloc[rows[0], 0], features.iloc[rows[1], 0] = left, right
    if valid:
        result = assemble_dataset(features, label, layout='match')
        assert pd.isna(result.X['fixture::odds::p_draw'].iloc[0]) if pd.isna(left) else result.X['fixture::odds::p_draw'].iloc[0] == left
    else:
        with pytest.raises(ValueError, match='Inconsistent fixture-level values.*odds::p_draw'):
            assemble_dataset(features, label, layout='match', drop_missing_targets=True)


@pytest.mark.parametrize('mode', ['different', 'one_missing', 'both_missing', 'invalid', 'numeric', 'absent', 'missing_record', 'duplicate'])
def test_cutoff_provenance_is_required_and_consistent(prepared, mode):
    _, features, label = prepared
    records = features.attrs['prediction_cutoffs']['records']
    event = label.metadata.event_id.iloc[0]
    pair = [r for r in records if r['event_id'] == event]
    if mode == 'different':
        pair[1]['cutoff'] = (pd.Timestamp(pair[0]['cutoff']) - pd.Timedelta('1ns')).isoformat()
    elif mode in ('one_missing', 'both_missing'):
        pair[0]['cutoff'] = None
        if mode == 'both_missing': pair[1]['cutoff'] = None
    elif mode == 'invalid': pair[0]['cutoff'] = 'invalid'
    elif mode == 'numeric': pair[0]['cutoff'] = 123
    elif mode == 'absent': features.attrs.pop('prediction_cutoffs')
    elif mode == 'missing_record': records.remove(pair[0])
    else: records.append(deepcopy(pair[0]))
    with pytest.raises(ValueError, match='[Ff]ixture.*cutoff'):
        assemble_dataset(features, label, layout='match')


def test_equivalent_timezone_cutoffs_and_unselected_fixture_errors(prepared):
    _, features, label = prepared
    for record in features.attrs['prediction_cutoffs']['records']:
        if record['side'] == 'away':
            record['cutoff'] = pd.Timestamp(record['cutoff']).tz_convert('Asia/Tokyo').isoformat()
    assemble_dataset(features, label, layout='match')
    features.attrs.pop('prediction_cutoffs')
    result = assemble_dataset(features, label, layout='match', feature_columns=['strength'])
    assert list(result.X) == ['home::strength', 'away::strength']


def test_scope_through_arithmetic_but_not_team_history(prepared):
    history, _, label = prepared
    draw = BayesianFixture(fields=('p_draw',))
    features = evaluate_features(history, {
        'twice': Ratio(draw, .5), 'difference': Difference(draw, draw),
        'team_history': Lag(draw), 'mixed': Difference(draw, IsHome()),
    }, keyed=True)
    assert features.attrs['feature_scopes'] == {
        'twice': 'fixture', 'difference': 'fixture', 'team_history': 'team', 'mixed': 'team'}
    assert list(assemble_dataset(features, label, layout='match').X) == [
        'home::team_history', 'home::mixed', 'away::team_history', 'away::mixed',
        'fixture::twice', 'fixture::difference']


def test_real_evaluation_with_unequal_prediction_times_rejected_even_for_equal_values(prepared):
    history, _, label = prepared
    cutoffs = history.kickoff_at.copy()
    cutoffs.iloc[1] -= pd.Timedelta(hours=1)
    frame = evaluate_features(history, {'draw': BayesianFixture(fields=('p_draw',))},
                              cutoffs=cutoffs, keyed=True)
    with pytest.raises(ValueError, match='Inconsistent fixture-level prediction cutoffs'):
        assemble_dataset(frame, label, layout='match')
