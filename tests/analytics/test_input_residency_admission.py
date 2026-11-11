"""Bounded partial residency without changing the default LRU contract."""
from dataclasses import replace

import pytest
import torch

from xdiyo_analytics.training.residency import InputCachePolicy, InputResident, _Pool


def owner(monkeypatch, budget=120, *, pool=None, headroom=0):
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (10**9, 10**9))
    result = InputResident('cpu', InputCachePolicy(budget, headroom, 'retain'))
    result.pool = pool or _Pool('cpu')
    result.pool.owners[result] = result.policy
    return result


def test_retain_keeps_prefix_across_larger_sequential_scans(monkeypatch):
    cached = owner(monkeypatch)
    builds = []
    def build(key):
        builds.append(key)
        return {'x': torch.arange(10.) + key, 'metadata': [key]}
    for scan in range(3):
        for key in range(3):
            with cached.lease(key, lambda: build(key), lambda: build(key)) as value:
                assert torch.equal(value['x'], torch.arange(10.) + key)
                assert value['metadata'] == [key]
                assert cached.pool.resident + cached.pool.transient <= 120
                value['x'].zero_()
                value['metadata'].clear()
    assert builds == [0, 1, 2, 2, 2]
    assert cached.pool.hits == 4 and cached.pool.misses == 5
    assert cached.pool.resident == 80 and cached.pool.transient == 0


def test_default_lru_cyclic_scan_still_uses_original_eviction_contract(monkeypatch):
    cached = owner(monkeypatch)
    cached.policy = replace(cached.policy, admission='lru')
    builds = []
    def build(key):
        builds.append(key)
        return torch.arange(10.) + key
    for _ in range(3):
        for key in range(3):
            with cached.lease(key, lambda: build(key), lambda: build(key)):
                assert cached.pool.resident + cached.pool.transient <= 120
    assert builds == [0, 1, 2]*3
    assert cached.pool.hits == 0 and cached.pool.misses == 9


def test_streaming_large_input_evicts_only_necessary_entries(monkeypatch):
    cached = owner(monkeypatch, 160)
    small = lambda: torch.ones(10)
    for key in range(3):
        with cached.lease(key, small, small):
            pass
    with cached.lease('large', lambda: torch.ones(20), small):
        assert cached.pool.resident == 80
        assert cached.pool.transient == 80
    assert len(cached.pool.entries) == 2


def test_retain_respects_shared_owner_budget_and_lifecycle(monkeypatch):
    first = owner(monkeypatch, 160)
    build = lambda: torch.ones(10)
    for key in range(3):
        with first.lease(key, build, build):
            pass
    second = owner(monkeypatch, 80, pool=first.pool)
    with second.lease('second', build, build):
        assert first.pool.resident + first.pool.transient <= 80
    assert first.pool.resident == 40
    second.close()
    first.close()
    assert not first.pool.entries and first.pool.resident == 0


def test_retain_evicts_for_physical_headroom_on_hits(monkeypatch):
    cached = owner(monkeypatch, 200)
    build = lambda: torch.ones(10)
    for key in range(3):
        with cached.lease(key, build, build):
            pass
    cached.policy = replace(cached.policy, headroom_bytes=80)
    monkeypatch.setattr(torch.cuda, 'mem_get_info',
                        lambda device: (200 - cached.pool.resident, 200))
    with cached.lease(2, build, build):
        assert cached.pool.resident == 80
        assert cached.pool.transient == 40
    assert cached.pool.hits == 1


@pytest.mark.parametrize('admission', ['lru', 'retain'])
def test_oversized_input_and_consumer_error_release_accounting(monkeypatch, admission):
    cached = owner(monkeypatch, 8)
    cached.policy = replace(cached.policy, admission=admission)
    build = lambda: torch.ones(10)
    with pytest.raises(RuntimeError, match='consumer'):
        with cached.lease(0, build, build):
            assert not cached.pool.entries
            raise RuntimeError('consumer')
    assert cached.pool.transient == cached.pool.resident == 0


def test_reusable_allocator_bytes_are_counted_only_when_requested(monkeypatch):
    pool = _Pool('cuda:0')
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (25, 200))
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda device: 100)
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda device: 60)
    assert pool._available() == 25
    assert pool._available(reusable=True) == 65


@pytest.mark.parametrize('admission', ['', None, 'fifo', True])
def test_invalid_admission_rejected(admission):
    with pytest.raises(ValueError, match='admission'):
        InputCachePolicy(admission=admission)


def test_default_native_profile_keeps_legacy_shape_and_retain_is_explicit():
    from xdiyo_analytics.training.native_resume import execution_profile
    from types import SimpleNamespace
    plain = execution_profile(SimpleNamespace(device='cpu'))
    explicit = execution_profile(SimpleNamespace(device='cpu', input_cache_policy=InputCachePolicy()))
    assert plain == explicit
    assert plain['input_cache'] == {'max_bytes': 0, 'headroom_bytes': 1073741824}
    retain = execution_profile(SimpleNamespace(device='cpu',
                               input_cache_policy=InputCachePolicy(admission='retain')))
    assert retain['input_cache']['admission'] == 'retain'


@pytest.mark.parametrize('kind', ['gat', 'cnn_classification'])
def test_retain_policy_roundtrips_native_models(tmp_path, kind):
    from native_resume_worker import fixture
    from xdiyo_analytics.graphs import TemporalGraphSerializer
    from xdiyo_analytics.neural import HeatmapCNNSerializer
    from xdiyo_analytics.training import TrainingControl
    import pandas as pd
    adapter, train, query = fixture(kind)
    adapter.configure_input_cache(2**20, 4096, admission='retain')
    adapter.fit_controlled(train, TrainingControl(max_steps=1))
    serializer = (TemporalGraphSerializer(event_store=adapter.event_store)
                  if kind == 'gat' else HeatmapCNNSerializer())
    serializer.save(adapter, tmp_path/'model')
    loaded = serializer.load(tmp_path/'model')
    assert loaded.input_cache_policy == adapter.input_cache_policy
    expected, actual = adapter.predict(query), loaded.predict(query)
    for key in expected:
        pd.testing.assert_frame_equal(expected[key], actual[key], check_exact=True)


def test_checkpoint_rejects_changed_admission_profile(tmp_path):
    from native_resume_worker import fixture
    from xdiyo_analytics.training import CheckpointPolicy, TrainingControl
    from xdiyo_analytics.training.native_resume import read_checkpoint, execution_profile
    adapter, train, _ = fixture('cnn_classification')
    adapter.configure_input_cache(2**20, 0)
    CheckpointPolicy().wrap(lambda: adapter, tmp_path/'runs', 'admission')().fit_controlled(
        train, TrainingControl(max_steps=1))
    checkpoint = adapter.training_checkpoint_
    read_checkpoint(checkpoint, profile=execution_profile(adapter))
    adapter.configure_input_cache(2**20, 0, admission='retain')
    with pytest.raises(ValueError, match='profile'):
        read_checkpoint(checkpoint, profile=execution_profile(adapter))
