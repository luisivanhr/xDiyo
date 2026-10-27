"""Observed-history provenance, bounded fixture schema and legacy compatibility."""
from dataclasses import replace
import copy
import json
import numpy as np
import pandas as pd
import pytest
import torch

from test_temporal_graph import sample, config, context, history
from xdiyo_analytics.graphs import TemporalGraphAdapter, TemporalGraphSerializer, TemporalEventStore
from xdiyo_analytics.graphs.observed_gap import encode_observed_gap, migrate_gap_config, LEGACY_GAP_ENCODING, GAP_CONFIG_FIELDS
from xdiyo_analytics.graphs.preprocessing import GraphPreprocessor, MaskedScale
from xdiyo_analytics.graphs.streaming import GroupPlan, prepare_group
from xdiyo_analytics.training import TrainingControl
from xdiyo_analytics.training.native_resume import capture_backend, restore_backend


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_gap_encoding_boundaries_monotonicity_and_unknown_provenance():
    days = np.array([0, 7, 30, np.nextafter(30., np.inf), 90, 830.2708333333334, np.finfo(float).max])
    values = encode_observed_gap(days, np.ones(len(days), bool))
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    assert (np.diff(values[:, 0]) >= 0).all()
    np.testing.assert_array_equal(values[:3, 2], [0, 0, 0])
    np.testing.assert_array_equal(values[3:, 2], [1, 1, 1, 1])
    np.testing.assert_array_equal(values[:, 1], 0)
    np.testing.assert_array_equal(values[:2, 0], [0, .5])
    np.testing.assert_allclose(values[5], [.9916395033, 0, 1], atol=5e-11, rtol=0)
    np.testing.assert_array_equal(encode_observed_gap([np.nan], [False]), [[0, 1, 1]])
    assert not np.array_equal(values[0], encode_observed_gap([np.nan], [False])[0])


@pytest.mark.parametrize('days,known', [([-1.], [True]), ([np.inf], [True]), ([-np.inf], [True]),
    ([np.nan], [True]), ([0.], [False]), ([np.inf], [False]), ([7.], [1])])
def test_invalid_observed_gap_rejected(days, known):
    with pytest.raises(ValueError): encode_observed_gap(days, known)


def prepared_pair():
    legacy = GraphPreprocessor(config(fixture_gap_encoding=LEGACY_GAP_ENCODING))
    legacy.leagues = {i: i for i in range(1, 14)}
    legacy.edge_scale = MaskedScale().fit([[0, 1], [2, 3]])
    legacy.fixture_scale = MaskedScale().fit([[0, 5, 6, 1, 2], [1, 9, 12, 3, 4], [np.nan, 7, 10, 5, 6]])
    corrected = GraphPreprocessor(replace(legacy.config, fixture_gap_encoding='bounded_observed_gap_v1'))
    corrected.leagues = legacy.leagues.copy()
    corrected.edge_scale = copy.deepcopy(legacy.edge_scale)
    corrected.fixture_scale = copy.deepcopy(legacy.fixture_scale)
    numeric = np.array([[0, 830.2708333333334, 7, 2, 4], [np.nan, np.nan, 0, 3, 5], [1, 30, 31, 4, 6]])
    meta = pd.DataFrame({'competition_id': [1, 2, 13]})
    for side, index in [('home', 1), ('away', 2)]:
        meta[f'{side}_observed_gap_days'] = numeric[:, index]
        meta[f'{side}_has_eligible_observation'] = np.isfinite(numeric[:, index])
        meta[f'{side}_latest_eligible_observed_at'] = pd.to_datetime(
            [None if np.isnan(d) else pd.Timestamp('2025-01-01', tz='UTC') - pd.Timedelta(days=d) for d in numeric[:, index]], utc=True)
    return legacy, corrected, meta, numeric


def test_actual_fixture_schema_bypasses_scaler_without_aliases_or_duplicate_masks():
    old, new, meta, numeric = prepared_pair()
    before = old.fixture(meta, numeric); after = new.fixture(meta, numeric)
    assert before.shape == (3, 24) and after.shape == (3, 26)
    assert 4*32 + new.fixture_dim == 154
    names = new.fixture_encoded_columns
    assert len(names) == len(set(names)) == new.fixture_dim
    assert not any('rest_days' in n or 'observed_gap_days' in n for n in names)
    for side, index in [('home', 1), ('away', 2)]:
        columns = [names.index(f'{side}_{suffix}') for suffix in
                   ('observed_gap_r7', 'no_eligible_history', 'no_eligible_observation_30d')]
        np.testing.assert_array_equal(after[:, columns], encode_observed_gap(numeric[:, index], np.isfinite(numeric[:, index])))
    shared = set(old.fixture_encoded_columns) & set(names)
    for name in shared:
        np.testing.assert_array_equal(before[:, old.fixture_encoded_columns.index(name)], after[:, names.index(name)])
    for field in ('edge_scale', 'fixture_scale'):
        assert getattr(old, field).state() == getattr(new, field).state()
    restored = GraphPreprocessor.from_state(new.config, json.loads(json.dumps(new.state())))
    np.testing.assert_array_equal(restored.fixture(meta, numeric), after)
    with pytest.raises(ValueError, match='provenance'): new.fixture(meta[['competition_id']], numeric)
    bad = meta.copy(); bad.loc[0, 'home_has_eligible_observation'] = False
    with pytest.raises(ValueError): new.fixture(bad, numeric)
    for state in (old.state(), dict(new.state(), fixture_schema=dict(new.fixture_schema, encoded_dim=24))):
        with pytest.raises(ValueError, match='schema'): GraphPreprocessor.from_state(new.config, state)


def test_gap_metadata_and_input_are_invariant_to_ineligible_history():
    store = sample(); cfg = config(temporal_mode='time_features')
    target = context(store, [2]); issue = store.issue_times(target.metadata, cfg)
    raw = GroupPlan(store, cfg, target, issue).raw(0)
    pre = GraphPreprocessor(cfg).fit(store.history, (), target.metadata, raw[-1])
    before = prepare_group(raw, pre, 'cpu', store)
    extra = history([(200, '2020-01-06T18:00', 1, 3, 99, 0, 10, 900, 200),
                     (201, '2020-02-01T18:00', 2, 3, 88, 0, 10, 900, 201)])
    extra['result_available_at'] = pd.Timestamp('2020-03-01', tz='UTC')
    h = store.history; h['result_available_at'] = pd.NaT
    later = TemporalEventStore(pd.concat([h, extra], ignore_index=True), store.team_seasons)
    altered = GroupPlan(later, cfg, target, issue).raw(0)
    after = prepare_group(altered, pre, 'cpu', later)
    np.testing.assert_array_equal(raw[-1], altered[-1])
    pd.testing.assert_frame_equal(raw[-2], altered[-2])
    for name in ('x', 'sender', 'receiver', 'log_weight', 'gate'):
        torch.testing.assert_close(before[1][name], after[1][name], rtol=0, atol=0)
    torch.testing.assert_close(before[-1], after[-1], rtol=0, atol=0)
    assert len(pre.temporal_columns) == 14
    store.assert_targets_unobserved(target.metadata, issue, cfg)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('mode', ['updated', 'layer_zero'])
def test_corrected_persistence_and_cross_mode_continuation(device, mode, tmp_path):
    if device.startswith('cuda') and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    store = sample(); cfg = config(temporal_mode='time_features', sender_state_mode=mode, deterministic=True)
    from xdiyo_analytics.graphs.extensions import GoalAuxiliary
    cfg = replace(cfg, goal_auxiliary=GoalAuxiliary(beta=0))
    model = TemporalGraphAdapter(cfg, store, control=TrainingControl(max_steps=2, seed=17), device=device)
    model.fit(context(store, [1, 2, 3])); target = context(store, [7, 8], fit=False)
    expected = model.predict(target)['predict_proba']; rates = model.predict_goals(target)
    serializer = TemporalGraphSerializer(store, device=device); serializer.save(model, tmp_path)
    restored = serializer.load(tmp_path)
    pd.testing.assert_frame_equal(expected, restored.predict(target)['predict_proba'], check_exact=True)
    pd.testing.assert_frame_equal(rates, restored.predict_goals(target), check_exact=True)
    assert restored.config == cfg
    with pytest.raises(ValueError, match='boundary'): restored.predict(context(store, [1], fit=False))
    state = capture_backend(model.backend_)
    changed = copy.deepcopy(state); changed['graph_semantics']['sender_state_mode'] = 'layer_zero' if mode == 'updated' else 'updated'
    with pytest.raises(ValueError, match='sender mode'): restore_backend(model.backend_, changed)
    restore_backend(model.backend_, state)
    metadata = json.loads((tmp_path/'graph.json').read_text())
    for key in ('fixture_encoded_dimension', 'fixture_gap_schema'):
        bad = copy.deepcopy(metadata); bad.pop(key); (tmp_path/'graph.json').write_text(json.dumps(bad))
        with pytest.raises(ValueError, match='incomplete'): serializer.load(tmp_path)


def test_legacy_absent_metadata_keeps_transform_fingerprint_and_predictions(tmp_path):
    store = sample(); cfg = config(fixture_gap_encoding=LEGACY_GAP_ENCODING)
    model = TemporalGraphAdapter(cfg, store, control=TrainingControl(max_steps=1, seed=17))
    model.fit(context(store, [1, 2, 3])); target = context(store, [7, 8], fit=False)
    expected = model.predict(target)['predict_proba']
    serializer = TemporalGraphSerializer(store); serializer.save(model, tmp_path)
    path = tmp_path/'graph.json'; metadata = json.loads(path.read_text())
    for name in (*GAP_CONFIG_FIELDS, 'sender_state_mode'): metadata['config'].pop(name)
    for name in ('fixture_encoded_feature_order', 'fixture_encoded_dimension', 'fixture_gap_schema'): metadata.pop(name)
    path.write_text(json.dumps(metadata)); restored = serializer.load(tmp_path)
    assert restored.config.fixture_gap_encoding == LEGACY_GAP_ENCODING and restored.config.sender_state_mode == 'updated'
    assert restored.backend_.model_version == model.backend_.model_version
    pd.testing.assert_frame_equal(expected, restored.predict(target)['predict_proba'], check_exact=True)
    for bad in ({'fixture_gap_encoding': 'bounded_observed_gap_v1'}, {'observed_gap_scale_days': 7}):
        with pytest.raises(ValueError, match='incomplete'): migrate_gap_config(bad)


@pytest.mark.parametrize('options', [dict(fixture_gap_encoding='wrong'), dict(observed_gap_scale_days=28),
    dict(observed_gap_recent_days=28), dict(sender_state_mode='frozen'),
    dict(sender_state_mode='layer_zero', temporal_mode='historical'), dict(fixture_features=('home_rest_days',))])
def test_invalid_gap_or_sender_contract_rejected(options):
    with pytest.raises(ValueError): config(**options)
