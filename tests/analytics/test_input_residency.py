import gc
import os
from dataclasses import replace
import pytest
import torch
from xdiyo_analytics.training.residency import InputResident, InputCachePolicy, _Pool


def fake_resident(monkeypatch, size=1024, pool=None):
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (10**9, 10**9))
    resident = InputResident('cpu', InputCachePolicy(size, 0))
    resident.pool = pool or _Pool('cpu')
    resident.pool.owners[resident] = resident.policy
    return resident


def test_resident_aliases_budget_eviction_and_owner_lifecycle(monkeypatch):
    a = fake_resident(monkeypatch, 160)
    p = a.pool
    builds = []
    def build():
        builds.append(1)
        return {'values': torch.arange(10.), 'positions': [2, 1]}
    with a.lease('x', build, build) as value:
        assert p.resident + p.transient == 80
        value['values'].fill_(88); value['positions'].clear()
    with a.lease('x', build, build) as value:
        assert value['values'].tolist() == list(range(10))
        assert value['positions'] == [2, 1]
    assert len(builds) == 1 and p.hits == 1
    b = fake_resident(monkeypatch, 80, p)
    with b.lease('y', build, build):
        assert p.resident + p.transient <= 80
    assert p.transient == 0
    a.close(); b.close()
    assert p.resident == 0


def test_low_budget_fallback_and_errors_release_accounting(monkeypatch):
    a = fake_resident(monkeypatch, 8)
    def build(): return torch.ones(10)
    with pytest.raises(RuntimeError):
        with a.lease('x', build, build):
            assert a.pool.resident == 0
            raise RuntimeError('forward failed')
    assert a.pool.transient == 0 and not a.pool.entries
    del a
    gc.collect()


@pytest.mark.parametrize('task', ['classification', 'regression'])
def test_cpu_fallback_policy_preserves_full_training(task):
    from xdiyo_analytics.neural import HeatmapCNNModel
    from xdiyo_analytics.training import TrainingControl
    from test_heatmap_cnn import standalone_context
    from test_graph_preparation_memory import assert_exact
    c = standalone_context(task)
    a, b = HeatmapCNNModel(task=task).build(), HeatmapCNNModel(task=task).build()
    b.configure_input_cache(10**7, 10**6)
    for model in (a, b): model.fit_controlled(c, TrainingControl(max_steps=2, seed=3))
    assert_exact(a.backend_.model.state_dict(), b.backend_.model.state_dict())
    assert_exact(a.backend_.optimizer.state_dict(), b.backend_.optimizer.state_dict())


@pytest.mark.skipif(os.environ.get('XDIYO_CUDA_CHECKS') != '1', reason='Explicit free-GPU gate required')
@pytest.mark.parametrize('kind', ['gat', 'gat_fixture', 'cnn_classification', 'cnn_regression'])
def test_cuda_residency_full_updates_and_inspection(kind):
    from native_resume_worker import fixture
    from test_graph_preparation_memory import assert_exact
    from xdiyo_analytics.training import TrainingControl
    a, context, query = fixture(kind)
    b, _, _ = fixture(kind)
    torch.use_deterministic_algorithms(True)
    b.configure_input_cache(256 * 1024**2, 1024**3)
    states = []
    native = torch.optim.AdamW.step
    for adapter in (a, b):
        updates = []
        def step(optimizer, *args, **kwargs):
            from xdiyo_analytics.training.native_resume import _cpu
            gradients = [_cpu(p.grad) for group in optimizer.param_groups for p in group['params']]
            result = native(optimizer, *args, **kwargs)
            updates.append((gradients, _cpu(optimizer.state_dict()), [_cpu(p) for group in optimizer.param_groups for p in group['params']]))
            return result
        torch.optim.AdamW.step = step
        try:
            adapter.configure_device('cuda:0')
            adapter.fit_controlled(context, TrainingControl(max_steps=3, seed=13))
        finally:
            torch.optim.AdamW.step = native
        states.append(updates)
    assert_exact(states[0], states[1])
    import pandas as pd
    pd.testing.assert_frame_equal(a.training_history_, b.training_history_, check_exact=True)
    for key in a.predict(query):
        pd.testing.assert_frame_equal(a.predict(query)[key], b.predict(query)[key], check_exact=True)


@pytest.mark.parametrize('spatial', [False, True])
def test_native_graph_resident_batches_match_uncached_and_do_not_alias(monkeypatch, spatial):
    from test_temporal_graph import sample, config, context
    from test_heatmap_cnn import graph_setup
    from xdiyo_analytics.graphs.adapter import _Backend
    from test_graph_preparation_memory import assert_exact
    from xdiyo_analytics.training import residency
    torch.set_num_threads(1)
    original = residency.InputResident
    def cpu_cache(device, policy=None):
        owner = original('cpu', policy)
        if policy and policy.max_bytes:
            owner.pool = _Pool('cpu')
            owner.pool.owners[owner] = policy
        return owner
    monkeypatch.setattr(residency, 'InputResident', cpu_cache)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda device: (10**9, 10**9))
    if spatial:
        spec, store = graph_setup()
        cfg = spec.config
    else:
        store, cfg = sample(), config()
    c = context(store, [1, 2, 3, 4])
    a, b = _Backend(cfg, store, 1, 'cpu'), _Backend(cfg, store, 1, 'cpu', InputCachePolicy(10**7, 0))
    a.initialize(c); b.initialize(c)
    for _ in range(3):
        assert a.step() == b.step()
        assert_exact(a.model.state_dict(), b.model.state_dict())
        assert_exact(a.optimizer.state_dict(), b.optimizer.state_dict())
    groups = b.train_groups
    with groups.lease(0) as actual:
        assert_exact(a.train_groups[0], actual)
        actual[1]['x'].fill_(98)
    with groups.lease(0) as actual:
        assert_exact(a.train_groups[0], actual)


def test_residency_ui_json_python_export_and_saved_native_model(tmp_path):
    import json
    from xdiyo_analytics.ui.recipe import catalog_for_ui, default_recipe, export_python
    from xdiyo_analytics.training import ExecutionPolicy, TrainingControl
    from xdiyo_analytics.neural import HeatmapCNNModel, HeatmapCNNSerializer
    from test_heatmap_cnn import standalone_context
    policy = ExecutionPolicy(input_cache_bytes=1024**2, input_cache_headroom_bytes=4096)
    catalog = catalog_for_ui()
    encoded = catalog.encode(policy)
    assert catalog.build(json.loads(json.dumps(encoded))) == policy
    fields = next(s for s in catalog.schema() if s['id'] == 'training.ExecutionPolicy')['fields']
    assert {f['name'] for f in fields} >= {'input_cache_bytes', 'input_cache_headroom_bytes'}
    recipe = default_recipe('data'); recipe['execution'] = encoded
    assert 'input_cache_bytes' in export_python(recipe)
    adapter = HeatmapCNNModel().build(); adapter.configure_input_cache(1024**2, 4096)
    adapter.fit_controlled(standalone_context(), TrainingControl(max_steps=1))
    serializer = HeatmapCNNSerializer(); serializer.save(adapter, tmp_path/'model')
    loaded = serializer.load(tmp_path/'model')
    assert loaded.input_cache_policy == adapter.input_cache_policy
    import pandas as pd
    for key, value in adapter.predict(standalone_context()).items():
        pd.testing.assert_frame_equal(value, loaded.predict(standalone_context())[key], check_exact=True)
