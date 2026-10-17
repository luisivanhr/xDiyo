"""Admission happens on descriptors; packed receiver groups remain isolated."""
from types import SimpleNamespace
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')
from xdiyo_analytics.graphs.packing import descriptor_bound, estimate, pack_groups, stream_batches


def group(value):
    data = dict(x=torch.tensor([[value]], dtype=torch.float32),
                receiver=torch.tensor([0]), sender=torch.tensor([1]),
                log_weight=torch.zeros(1, dtype=torch.float64), gate=torch.ones(2),
                league=torch.zeros(2, dtype=torch.long), entry=torch.zeros(2, dtype=torch.long),
                _dependency_rows=[None, torch.tensor([0])])
    return np.array([int(value)]), data, torch.tensor([0]), torch.tensor([1]), torch.zeros((1, 1))


def environment(host=10**9, device='cpu', mode='dependency_packed'):
    cfg = SimpleNamespace(embedding_dim=2, heads=1, layers=2, hidden_width=3,
                          statistics_forecast=None, heatmaps=None, boundary_memory=None,
                          execution=SimpleNamespace(mode=mode, host_memory_bytes=host,
                              device_memory_bytes=10**9, max_graphs=8, headroom_bytes=0))
    store = SimpleNamespace(_history=range(1000), _source_nodes=range(100))
    plan = SimpleNamespace(store=store, config=cfg)
    backend = SimpleNamespace(config=cfg, device=device,
        preprocessor=SimpleNamespace(edge_columns=(), fixture_columns=(), leagues={}))
    class Groups:
        _resident = SimpleNamespace(pool=None)
        def __len__(self): return 3
        def __getitem__(self, i): return group(i)
    groups = Groups()
    groups.plan = plan
    return backend, groups


def selection(i, native=1000):
    return SimpleNamespace(layer_rows=(np.array([0]), np.array([0])), teams=(0, 1),
                           positions=np.array([i]), host_nbytes=256,
                           native_edge_count=native, full_node_count=100)


def test_disjoint_packing_offsets_nodes_edges_and_fixture_order():
    result = pack_groups([group(4), group(7)])
    assert result[0].tolist() == [4, 7]
    assert result[1]['receiver'].tolist() == [0, 2]
    assert result[1]['sender'].tolist() == [1, 3]
    assert result[2].tolist() == [0, 2]
    assert result[3].tolist() == [1, 3]
    assert result[1]['_dependency_rows'][0] is None
    assert result[1]['_dependency_rows'][1].tolist() == [0, 1]
    # Incoming softmax groups cannot mix issuances, even identical team IDs.
    from xdiyo_analytics.graphs.network import grouped_softmax
    weights = grouped_softmax(torch.tensor([[1.], [100.]]), result[1]['receiver'],
                              result[1]['log_weight'], 4)
    assert torch.equal(weights, torch.ones_like(weights))


def test_cpu_workspace_is_charged_to_host_and_backward_increases_estimate():
    backend, _ = environment()
    inference = estimate(backend, [selection(0)])
    training = estimate(backend, [selection(0)], backward=True)
    assert inference.device_bytes == training.device_bytes == 0
    # Two layers makes activation estimate equal here; it must never decrease.
    assert training.host_bytes >= inference.host_bytes
    backend.device = 'cuda:0'
    gpu = estimate(backend, [selection(0)], backward=True, packed=True)
    assert gpu.device_bytes > 0 and gpu.concatenation > 0 and gpu.targets > 0


def test_descriptor_refusal_never_constructs_dependency_tensors(monkeypatch):
    backend, groups = environment(host=1)
    def forbidden(*args, **kwargs):
        raise AssertionError('descriptor refused before construction')
    monkeypatch.setitem(sys.modules, 'xdiyo_analytics.graphs.dependency',
                        SimpleNamespace(select_dependencies=forbidden, prepare_dependency_group=forbidden))
    batches = list(stream_batches(backend, groups))
    assert [b[1] for b in batches] == [(0,), (1,), (2,)]
    assert all(b[2] == 'descriptor_budget' for b in batches)


def test_deterministic_descriptor_plan_and_no_unadmitted_tensor_lookahead(monkeypatch):
    backend, groups = environment()
    events = []
    def select(plan, index):
        events.append(('select', index))
        return selection(index)
    def prepare(groups, index, **kwargs):
        events.append(('prepare', index))
        return group(index)
    monkeypatch.setitem(sys.modules, 'xdiyo_analytics.graphs.dependency',
                        SimpleNamespace(select_dependencies=select, prepare_dependency_group=prepare))
    batches = list(stream_batches(backend, groups, backward=True))
    assert [b[1] for b in batches] == [(0, 1, 2)]
    assert events[:3] == [('select', 0), ('select', 1), ('select', 2)]
    first_hash = backend._execution_trace['plan_sha256']
    list(stream_batches(backend, groups, backward=True))
    assert backend._execution_trace['plan_sha256'] == first_hash
    assert backend._execution_trace['groups'] == 3


def test_device_workspace_refusal_precedes_any_dependency_tensor_allocation(monkeypatch):
    backend, groups = environment(device='cuda:0')
    backend.config.execution.device_memory_bytes = 1
    def forbidden(*args, **kwargs):
        raise AssertionError('unadmitted dependency tensors were constructed')
    monkeypatch.setitem(sys.modules, 'xdiyo_analytics.graphs.dependency', SimpleNamespace(
        select_dependencies=lambda plan, i: selection(i), prepare_dependency_group=forbidden))
    batches = list(stream_batches(backend, groups, backward=True))
    assert [b[2] for b in batches] == ['workspace_budget'] * 3
    assert backend._execution_trace['fallback_groups'] == 3


def test_dense_pack_falls_back_to_singletons_without_exceeding_native_envelope(monkeypatch):
    backend, groups = environment()
    def dense(plan, i):
        result = selection(i, native=1)
        result.full_node_count = 2
        return result
    monkeypatch.setitem(sys.modules, 'xdiyo_analytics.graphs.dependency', SimpleNamespace(
        select_dependencies=dense,
        prepare_dependency_group=lambda groups, i, **kwargs: group(i)))
    batches = list(stream_batches(backend, groups, backward=True))
    assert [b[1] for b in batches] == [(0,), (1,), (2,)]


def test_pending_descriptors_flush_before_next_descriptor_construction(monkeypatch):
    import xdiyo_analytics.graphs.packing as packing
    import weakref
    backend, groups = environment(host=1_000_000)
    monkeypatch.setattr(packing, 'descriptor_bound', lambda plan: 1_000_000)
    events = []
    references = []
    def select(plan, i):
        assert all(reference() is None for reference in references)
        events.append(('select', i))
        # SimpleNamespace has no weakref support; a small subclass does.
        class Descriptor(SimpleNamespace):
            pass
        result = Descriptor(**vars(selection(i)))
        references.append(weakref.ref(result))
        return result
    def prepare(groups, i, **kwargs):
        events.append(('prepare', i))
        return group(i)
    monkeypatch.setitem(sys.modules, 'xdiyo_analytics.graphs.dependency', SimpleNamespace(
        select_dependencies=select, prepare_dependency_group=prepare))
    stream = stream_batches(backend, groups)
    first, indices, _ = next(stream)
    assert indices == (0,)
    assert events == [('select', 0), ('prepare', 0)]
    del first
    list(stream)
    assert events[:4] == [('select', 0), ('prepare', 0), ('select', 1), ('prepare', 1)]


def test_unequal_graphs_preserve_fixture_order_and_three_layer_edge_offsets():
    first = group(8)
    first[1]['_dependency_rows'] = [None, None, torch.tensor([0])]
    second = (np.array([3, 11]), dict(
        x=torch.tensor([[2.], [3.], [4.]]), receiver=torch.tensor([0, 0, 2]),
        sender=torch.tensor([1, 2, 3]), log_weight=torch.zeros(3, dtype=torch.float64),
        gate=torch.ones(4), league=torch.zeros(4, dtype=torch.long),
        entry=torch.zeros(4, dtype=torch.long),
        _dependency_rows=[None, torch.tensor([0, 2]), torch.tensor([2])]),
        torch.tensor([0, 2]), torch.tensor([1, 3]), torch.zeros((2, 1)))
    packed = pack_groups([first, second])
    assert packed[0].tolist() == [8, 3, 11]
    assert packed[2].tolist() == [0, 2, 4]
    assert packed[3].tolist() == [1, 3, 5]
    assert packed[1]['receiver'].tolist() == [0, 2, 2, 4]
    assert packed[1]['sender'].tolist() == [1, 3, 4, 5]
    assert packed[1]['_dependency_rows'][0] is None
    assert packed[1]['_dependency_rows'][1].tolist() == [0, 1, 3]
    assert packed[1]['_dependency_rows'][2].tolist() == [0, 3]
    assert packed[1]['_group_lengths'] == (1, 2)


def test_ledger_reserves_cold_source_cache_and_full_eligible_scan():
    backend, _ = environment()
    backend.store = SimpleNamespace(_history=range(2000), _source_nodes=range(100))
    backend.preprocessor._edge_cache = None
    cold = estimate(backend, [selection(0)])
    backend.preprocessor._edge_cache = ('existing', object(), object())
    warm = estimate(backend, [selection(0)])
    assert cold == warm
    assert cold.source_cache == 2000 * (4 * 2 + 1)
    assert cold.preparation >= 1000 * 192


def test_cuda_host_metadata_and_combined_222_summary_are_explicitly_reserved():
    backend, _ = environment(device='cuda:0')
    baseline = estimate(backend, [selection(0)], backward=True)
    target = SimpleNamespace(family='quantile', quantiles=(.25, .5, .75))
    backend.config.statistics_forecast = SimpleNamespace(targets=(target,) * 74)
    expanded = estimate(backend, [selection(0)], backward=True)
    assert expanded.host_metadata > baseline.host_metadata
    assert expanded.host_metadata >= (128 * 2 + 64 * 74 + 64) * 1024
    assert expanded.activations - baseline.activations == 32 * 222
    assert expanded.host_bytes > baseline.host_bytes
    assert expanded.targets > baseline.targets


def test_empty_cone_over_nonempty_native_graph_never_constructs_dependency_tensors(monkeypatch):
    backend, groups = environment()
    def empty(plan, i):
        result = selection(i)
        result.layer_rows = (np.array([], dtype=np.int64),) * 2
        return result
    def forbidden(*args, **kwargs):
        raise AssertionError('empty cone must retain native zero-gradient semantics')
    monkeypatch.setitem(sys.modules, 'xdiyo_analytics.graphs.dependency', SimpleNamespace(
        select_dependencies=empty, prepare_dependency_group=forbidden))
    batches = list(stream_batches(backend, groups, backward=True))
    assert [b[2] for b in batches] == ['empty_dependency'] * 3


def test_descriptor_preflight_reserves_query_only_teams_without_history():
    backend, groups = environment()
    groups.plan.store._history = ()
    groups.plan.store._source_nodes = ()
    groups.plan.descriptors = ((None, np.arange(10_000)),)
    assert descriptor_bound(groups.plan) >= 10_000 * 2 * 1024


def test_quadratic_one_hot_identity_temporary_is_reserved_on_host():
    backend, _ = environment(device='cuda:0')
    baseline = estimate(backend, [selection(0)])
    backend.preprocessor.leagues = {i: i for i in range(1000)}
    expanded = estimate(backend, [selection(0)])
    assert expanded.one_hot_workspace == 8 * 1001**2
    assert expanded.one_hot_workspace > baseline.one_hot_workspace
    assert expanded.host_bytes >= expanded.one_hot_workspace


def test_persistent_scalar_receipts_use_native_pass_count_on_host_and_device():
    backend, _ = environment(device='cuda:0')
    baseline = estimate(backend, [selection(0)], backward=True)
    reserved = estimate(backend, [selection(0)], backward=True, receipt_groups=1463)
    assert reserved.scalar_receipts == 1463 * 3072
    assert reserved.host_bytes - baseline.host_bytes == 1463 * 2048
    assert reserved.device_bytes - baseline.device_bytes == 1463 * 1024
    backend.device = 'cpu'
    baseline_cpu = estimate(backend, [selection(0)], backward=True)
    reserved_cpu = estimate(backend, [selection(0)], backward=True, receipt_groups=1463)
    assert reserved_cpu.device_bytes == 0
    assert reserved_cpu.host_bytes - baseline_cpu.host_bytes == 1463 * 3072


def test_future_heavy_source_vocabulary_map_is_reserved_despite_tiny_eligible_cone():
    backend, _ = environment(device='cuda:0')
    backend.store = SimpleNamespace(_history=range(1), _source_nodes=range(2))
    tiny = selection(0, native=1)
    tiny.full_node_count = 2
    baseline = estimate(backend, [tiny])
    backend.store._source_nodes = range(1_000_000)
    future_heavy = estimate(backend, [tiny])
    assert future_heavy.source_map_workspace == 8_000_000
    assert future_heavy.preparation - baseline.preparation == 8 * (1_000_000 - 2)
    assert future_heavy.host_bytes - baseline.host_bytes == 8 * (1_000_000 - 2)
    assert future_heavy.device_bytes == baseline.device_bytes
