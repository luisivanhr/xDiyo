"""Historical statistic releases follow each query's information boundary."""
import os

import numpy as np
import pandas as pd
import pytest
import torch

from test_temporal_graph import config
from test_graph_temporal_attention import chain, components, slow_encode
from xdiyo_analytics.graphs import TemporalEventStore
from xdiyo_analytics.graphs.execution import deterministic_execution


STAT = 'team::shots'
OTHER = 'opponent::shots'
ENCOUNTER = pd.Timestamp('2020-01-10T18:00Z')
LATER = pd.Timestamp('2020-02-01T12:00Z')


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(params=['cpu', 'cuda:0'])
def device(request):
    value = request.param
    if value.startswith('cuda') and (
            os.environ.get('XDIYO_CUDA_CHECKS') != '1' or not torch.cuda.is_available()):
        pytest.skip('Explicit idle CUDA check required.')
    return value


def released_chain(offset=1, *, policy='per_stat_release_v1', mode='historical', layers=2):
    original = chain()
    h = original.history
    h[STAT] = 2. * h.team_id + h.event_id
    h[OTHER] = 2. * h.opponent_id + h.event_id
    cfg = config(temporal_mode=mode, statistics=(STAT, OTHER),
                 statistic_availability_policy=policy, layers=layers, deterministic=True)
    h['available_at::'+STAT] = h.kickoff_at + pd.Timedelta(days=1)
    h['available_at::'+OTHER] = h.kickoff_at + pd.Timedelta(days=1)
    release = pd.NaT if offset is None else cfg.history_cutoff(ENCOUNTER) + pd.Timedelta(offset, unit='ns')
    h.loc[h.event_id.eq(1), 'available_at::'+STAT] = release
    h.loc[h.event_id.eq(1) & h.team_id.eq(3), 'available_at::'+OTHER] = pd.NaT
    return TemporalEventStore(h, original.team_seasons), cfg


def materialize(store, cfg, pre, when, device):
    snapshot = store.snapshot(when, cfg, (1, 2, 3))
    return snapshot, pre.native_tensors(snapshot, device, store)


@pytest.mark.parametrize('offset', [-1, 0, 1, None], ids=['before', 'at', 'after', 'missing'])
def test_historical_rows_match_individual_query_cutoffs(offset, device):
    store, cfg = released_chain(offset)
    pre, _ = components(store, cfg)
    snapshot, data = materialize(store, cfg, pre, LATER, device)
    target_row = store._history.index[store._history.event_id.eq(1) & store._history.team_id.eq(2)][0]
    numeric_width = pre.edge_dim - len(pre.temporal_columns)
    mask_column = pre.edge_encoded_columns.index(STAT+'__missing')
    seen = []
    for batch, tensors in zip(snapshot._temporal_plan.batches, data['temporal_layers']):
        # Independent scalar queries preserve repeated source rows and their order.
        expected = np.concatenate([
            pre._encode_edges(store._history.iloc[[row]],
                              cutoff=cfg.history_cutoff(pd.Timestamp(int(time), tz='UTC')))
            for row, time in zip(batch['rows'], batch['times'])
        ]).astype(np.float32)
        np.testing.assert_array_equal(tensors['x'][:, :numeric_width].cpu().numpy(), expected)
        for position in np.flatnonzero(batch['rows'] == target_row):
            seen.append((int(batch['times'][position]), float(tensors['x'][position, mask_column])))
    assert {time for time, _ in seen} >= {ENCOUNTER.value, LATER.value}
    early_mask = float(offset is None or offset > 0)
    late_mask = float(offset is None)
    assert all(mask == early_mask for time, mask in seen if time == ENCOUNTER.value)
    assert all(mask == late_mask for time, mask in seen if time == LATER.value)
    assert seen.index((LATER.value, late_mask)) < seen.index((ENCOUNTER.value, early_mask))


def test_later_query_cannot_change_earlier_historical_inputs_or_output(device):
    store, cfg = released_chain()
    pre, model = components(store, cfg)
    model.to(device)
    with deterministic_execution(cfg, device), torch.no_grad():
        _, before = materialize(store, cfg, pre, ENCOUNTER, device)
        original = model.encode(before).clone()
        materialize(store, cfg, pre, LATER, device)
        _, after = materialize(store, cfg, pre, ENCOUNTER, device)
        for old, new in zip(before['temporal_layers'], after['temporal_layers']):
            for key in old:
                torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
        torch.testing.assert_close(original, model.encode(after), rtol=0, atol=0)


def test_empty_released_history_materializes_contextual_priors(device):
    store, cfg = released_chain()
    pre, model = components(store, cfg)
    model.to(device)
    _, data = materialize(store, cfg, pre, pd.Timestamp('2020-01-01T12:00Z'), device)
    assert all(layer['x'].shape == (0, pre.edge_dim) for layer in data['temporal_layers'])
    with deterministic_execution(cfg, device), torch.no_grad():
        assert torch.isfinite(model.encode(data)).all()


@pytest.mark.parametrize('layers', [1, 2, 3])
@pytest.mark.parametrize('encoding', ['rational_decay_v1', 'log1p_v1'])
def test_release_aware_batched_encoder_matches_direct_recursive_snapshots(layers, encoding):
    from dataclasses import replace
    store, cfg = released_chain(layers=layers)
    cfg = replace(cfg, time_encoding=encoding)
    pre, model = components(store, cfg)
    snapshot, data = materialize(store, cfg, pre, LATER, 'cpu')
    with torch.no_grad():
        actual = model.encode(data)
        for i, team in enumerate(snapshot.teams):
            expected = slow_encode(team, snapshot.issuance, layers, model, store, pre, cfg)
            torch.testing.assert_close(actual[i:i+1], expected, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize('mode,policy,release_columns', [
    ('snapshot', 'per_stat_release_v1', True),
    ('time_features', 'per_stat_release_v1', True),
    ('historical', 'legacy_unmasked_v1', True),
    ('historical', 'per_stat_release_v1', False),
])
def test_existing_statistic_policy_controls_materialize(mode, policy, release_columns):
    store, cfg = released_chain(mode=mode, policy=policy)
    if not release_columns:
        h = store.history.drop(columns=['available_at::'+STAT, 'available_at::'+OTHER])
        store = TemporalEventStore(h, store.team_seasons)
    pre, model = components(store, cfg)
    _, data = materialize(store, cfg, pre, LATER, 'cpu')
    with torch.no_grad():
        assert torch.isfinite(model.encode(data)).all()
