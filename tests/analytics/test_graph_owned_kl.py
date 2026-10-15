"""Native immutable graph targets validate at preparation; mutable inputs do not."""
import os
import numpy as np
import pandas as pd
import pytest
import torch
from test_heatmap_cnn import graph_setup
from test_temporal_graph import context
from test_graph_preparation_memory import assert_exact
from xdiyo_analytics.neural import Reconstruction
from xdiyo_analytics.graphs.adapter import _Backend


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_prevalidated_graph_kl_matches_mutable_updates(monkeypatch, device):
    if device.startswith('cuda') and os.environ.get('XDIYO_CUDA_CHECKS') != '1':
        pytest.skip('Explicit free-GPU gate required')
    torch.set_num_threads(1)
    spec, store = graph_setup(Reconstruction(loss='kl'), deterministic=True)
    c = context(store, [1, 2, 3, 4])
    native = _Backend(spec.config, store, 3, device)
    legacy = _Backend(spec.config, store, 3, device)
    native.initialize(c); legacy.initialize(c)
    # Public materialized groups must still validate on every reconstruction.
    legacy.train_groups = list(legacy.train_groups)
    calls = []
    from xdiyo_analytics.neural.network import HeatmapEncoder
    original = HeatmapEncoder._reconstruction_errors
    def tracked(self, *args, **kwargs):
        calls.append(kwargs['validate'])
        return original(self, *args, **kwargs)
    monkeypatch.setattr(HeatmapEncoder, '_reconstruction_errors', tracked)
    for _ in range(3):
        calls.clear(); a = native.step()
        assert calls and not any(calls)
        calls.clear(); b = legacy.step()
        assert calls and all(calls)
        assert a == b
        assert_exact(native.model.state_dict(), legacy.model.state_dict())
        assert_exact(native.optimizer.state_dict(), legacy.optimizer.state_dict())
        assert_exact([p.grad for p in native.model.parameters()], [p.grad for p in legacy.model.parameters()])
        assert native.checkpoint_metrics('train_loss', a) == legacy.checkpoint_metrics('train_loss', b)
    group = legacy.train_groups[-1]
    group[1]['maps'][0, 0, 0, 0] = -1
    with pytest.raises(ValueError, match='nonnegative'):
        legacy._group_loss(group, legacy.train_targets, legacy.train_weights, False)


def test_native_grid_validation_once_per_cached_source_and_causal_errors(monkeypatch):
    from xdiyo_analytics.graphs import TemporalEventStore
    from xdiyo_analytics.graphs.spatial import columns
    import xdiyo_analytics.neural.prepared as owned
    torch.set_num_threads(1)
    spec, store = graph_setup(Reconstruction(loss='kl'))
    backend = _Backend(spec.config, store, 3, 'cpu')
    backend.initialize(context(store, [1, 2, 3, 4]))
    prep = backend.preprocessor
    prep._map_cache = None
    calls = []
    original = owned.validate_kl_targets
    def tracked(maps, flags):
        calls.append(len(maps))
        return original(maps, flags)
    monkeypatch.setattr(owned, 'validate_kl_targets', tracked)
    snapshot = store.snapshot(pd.Timestamp('2020-01-12', tz='UTC'), spec.config, {1, 2})
    first = prep.cached_maps(snapshot, store)
    second = prep.cached_maps(snapshot, store)
    assert len(calls) == 1
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    first[0].fill(-10)
    assert (prep.cached_maps(snapshot, store)[0] >= 0).all()
    history = store.history
    history.loc[history.event_id.eq(9), columns(spec.config.heatmaps)[0]] = -1
    changed = TemporalEventStore(history, store.team_seasons, schedule=store.schedule, provenance=store.provenance)
    early = changed.snapshot(pd.Timestamp('2020-01-12', tz='UTC'), spec.config, {1, 2})
    prep.cached_maps(early, changed)  # Invalid future targets are not visited.
    late = changed.snapshot(pd.Timestamp('2020-02-01', tz='UTC'), spec.config, {1, 2})
    with pytest.raises(ValueError, match='nonnegative'):
        prep.cached_maps(late, changed)


@pytest.mark.parametrize('bad', ['negative', 'zero', 'infinite'])
def test_immutable_kl_target_validator_rejects_invalid_observed_maps(bad):
    from xdiyo_analytics.neural.prepared import validate_kl_targets
    maps = np.ones((2, 2, 10, 10), dtype=np.float32)
    flags = np.array([[1, 0], [1, 1]], dtype=np.float32)
    maps[0, 1] = np.nan  # Absent map is not a target.
    validate_kl_targets(maps, flags)
    maps[1, 0] = {'negative': -1, 'zero': 0, 'infinite': np.inf}[bad]
    with pytest.raises(ValueError, match='KL reconstruction'):
        validate_kl_targets(maps, flags)
