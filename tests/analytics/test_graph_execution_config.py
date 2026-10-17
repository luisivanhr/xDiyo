"""Explicit execution policies survive UI/export and remain resume identities."""
from dataclasses import FrozenInstanceError, asdict, replace
import json
from pathlib import Path

import pytest

from xdiyo_analytics.graphs import GraphExecution, TemporalGraphConfig


MODES = ('native', 'dependency', 'dependency_lean', 'dependency_packed')


def test_execution_is_opt_in_and_frozen():
    assert TemporalGraphConfig('UTC').execution is None
    policy = GraphExecution()
    assert asdict(policy) == dict(mode='dependency_packed', host_memory_bytes=128*1024**2,
        device_memory_bytes=256*1024**2, max_graphs=8, headroom_bytes=1024**3)
    assert GraphExecution(max_graphs=1).max_graphs == 1
    with pytest.raises(FrozenInstanceError):
        policy.mode = 'native'
    with pytest.raises(TypeError, match='execution must be GraphExecution'):
        TemporalGraphConfig('UTC', execution=asdict(policy))


@pytest.mark.parametrize('mode', MODES)
def test_all_ablation_modes(mode):
    policy = GraphExecution(mode=mode)
    assert TemporalGraphConfig('UTC', execution=policy).execution is policy


@pytest.mark.parametrize('mode', ['', 'automatic', None, True])
def test_reject_unknown_mode(mode):
    with pytest.raises(ValueError, match='execution mode'):
        GraphExecution(mode=mode)


@pytest.mark.parametrize('field', ['host_memory_bytes', 'device_memory_bytes', 'max_graphs', 'headroom_bytes'])
@pytest.mark.parametrize('value', [True, False, 0, -1, 1.5, '8', None])
def test_positive_integer_budgets_reject_boolean(field, value):
    with pytest.raises(ValueError, match=field):
        GraphExecution(**{field: value})


def test_recipe_roundtrip_and_disabled_inventory_default():
    from xdiyo_analytics.ui.recipe import catalog_for_ui
    import xdiyo_analytics.ui as ui
    catalog = catalog_for_ui()
    for mode in MODES:
        config = TemporalGraphConfig('UTC', execution=GraphExecution(mode=mode, max_graphs=1))
        assert catalog.build(json.loads(json.dumps(catalog.encode(config)))) == config
    legacy = {'component': 'graphs.TemporalGraphConfig', 'params': {'timezone': 'UTC'}}
    assert catalog.build(legacy).execution is None
    inventory = json.loads(Path(ui.__file__).with_name('inventory.json').read_text(encoding='utf-8'))
    graph = inventory['components']['graphs.TemporalGraphConfig']
    field = next(f for f in graph['fields'] if f['name'] == 'execution')
    assert field['default'] is None and field['kind'] == 'component'
    assert field['components'] == ['graphs.GraphExecution']
    assert 'experimental' in field['help']
    assert 'Floating-point differences can change gradients, optimization trajectories and checkpoint selection' in field['help']
    assert 'numerical interchangeability is unvalidated' in field['help']
    fields = {f['name']: f for f in inventory['components']['graphs.GraphExecution']['fields']}
    assert fields['mode']['choices'] == list(MODES)
    assert fields['mode']['default'] == 'dependency_packed'
    for name in ('host_memory_bytes', 'device_memory_bytes', 'max_graphs', 'headroom_bytes'):
        assert fields[name]['min'] == 1 and fields[name]['step'] == 1


@pytest.mark.parametrize('mode', MODES)
def test_serializer_roundtrip_and_legacy_option_omission(tmp_path, mode):
    pytest.importorskip('torch')
    import pandas as pd
    from test_temporal_graph import sample, config, context
    from xdiyo_analytics.graphs import TemporalGraphAdapter, TemporalGraphSerializer
    from xdiyo_analytics.graphs.adapter import torch_threads
    from xdiyo_analytics.training import TrainingControl
    store = sample()
    policy = GraphExecution(mode=mode, host_memory_bytes=67108864, device_memory_bytes=134217728,
                            max_graphs=1, headroom_bytes=1048576)
    model = TemporalGraphAdapter(config(execution=policy), store, control=TrainingControl(max_steps=1))
    serializer = TemporalGraphSerializer(store)
    with torch_threads(1):
        model.fit(context(store, [1, 2, 3]))
        serializer.save(model, tmp_path)
        metadata = json.loads((tmp_path/'graph.json').read_text(encoding='utf-8'))
        assert metadata['config']['execution'] == asdict(policy)
        loaded = serializer.load(tmp_path)
        assert loaded.config.execution == policy
        prediction = context(store, [7, 8], fit=False)
        pd.testing.assert_frame_equal(model.predict(prediction)['predict_proba'],
                                      loaded.predict(prediction)['predict_proba'], check_exact=True)
        metadata['config']['execution']['max_graphs'] = 2
        (tmp_path/'graph.json').write_text(json.dumps(metadata), encoding='utf-8')
        with pytest.raises(ValueError, match='Model version'):
            serializer.load(tmp_path)
        # An absent option means legacy native, never today's GraphExecution().
        legacy = TemporalGraphAdapter(config(), store, control=TrainingControl(max_steps=1))
        legacy.fit(context(store, [1, 2, 3]))
        serializer.save(legacy, tmp_path)
        path = tmp_path/'graph.json'
        metadata = json.loads(path.read_text(encoding='utf-8'))
        assert metadata['config'].pop('execution') is None
        path.write_text(json.dumps(metadata), encoding='utf-8')
        restored = serializer.load(tmp_path)
        assert restored.config.execution is None
        assert restored.backend_.model_version == legacy.backend_.model_version


def test_resume_profile_rejects_changed_execution_fields(tmp_path):
    pytest.importorskip('torch')
    from test_temporal_graph import sample, config
    from xdiyo_analytics.graphs import TemporalGraphAdapter
    from xdiyo_analytics.training.native_resume import execution_profile, read_checkpoint, write_checkpoint
    store = sample()
    policy = GraphExecution()
    adapter = TemporalGraphAdapter(config(execution=policy), store)
    profile = execution_profile(adapter)
    assert profile['graph_execution'] == asdict(policy)
    state = dict(attempt=1, restarts=0, rows=[], attempts=[], winner=None,
                 winning_value=0., winner_info=None, rng=None, active=None)
    write_checkpoint(tmp_path, state, 'fixed-source-config-identity', profile)
    assert read_checkpoint(tmp_path, 'fixed-source-config-identity', profile)[1]['profile'] == profile
    changes = dict(mode='dependency_lean', host_memory_bytes=policy.host_memory_bytes+1,
                   device_memory_bytes=policy.device_memory_bytes+1, max_graphs=1,
                   headroom_bytes=policy.headroom_bytes+1)
    for name, value in changes.items():
        changed = TemporalGraphAdapter(config(execution=replace(policy, **{name: value})), store)
        with pytest.raises(ValueError, match='execution profile'):
            read_checkpoint(tmp_path, profile=execution_profile(changed))
    native_profile = execution_profile(TemporalGraphAdapter(config(), store))
    assert native_profile['graph_execution'] is None
    with pytest.raises(ValueError, match='execution profile'):
        read_checkpoint(tmp_path, profile=native_profile)
    # A pre-option manifest is not silently upgraded to new execution defaults.
    old_profile = dict(profile)
    old_profile.pop('graph_execution')
    with pytest.raises(ValueError, match='execution profile'):
        read_checkpoint(tmp_path, profile=old_profile)


def test_resume_identity_remains_source_bound():
    from xdiyo_analytics.experiments.recovery import execution_key, training_source_files
    import xdiyo_analytics.graphs.config as module
    root = Path(module.__file__).resolve().parents[1]
    source = training_source_files(root)
    assert root/'graphs/config.py' in source
    assert root/'graphs/adapter.py' in source
    assert root/'training/native_resume.py' in source
    base = TemporalGraphConfig('UTC')
    assert execution_key(base) != execution_key(replace(base, execution=GraphExecution(mode='native')))
    assert execution_key(replace(base, execution=GraphExecution())) != execution_key(
        replace(base, execution=GraphExecution(max_graphs=1)))


def test_runtime_budget_fallback_is_persisted_in_summary_and_checkpoint_state(tmp_path):
    pytest.importorskip('torch')
    from copy import deepcopy
    from graph_execution_fixture import execution_fixture
    from xdiyo_analytics.graphs import TemporalGraphSerializer
    from xdiyo_analytics.training import TrainingControl
    from xdiyo_analytics.training.native_resume import capture_backend, restore_backend
    adapter, train, _ = execution_fixture(statistics=False)
    adapter.config = replace(adapter.config, execution=replace(adapter.config.execution, host_memory_bytes=1))
    adapter.fit_controlled(train, TrainingControl(max_steps=1))
    report = deepcopy(adapter.training_summary_['execution_report'])
    assert adapter.training_summary_['compatibility_fallback'] is None
    assert report.keys() == {'train', 'validation'}
    for trace in report.values():
        assert trace['fallback_groups'] == trace['groups'] > 0
        assert trace['fallback_reasons'] == {'descriptor_budget': trace['groups']}
    state = capture_backend(adapter.backend_)
    assert state['graph_execution_reports'] == report
    adapter.backend_._execution_reports = {}
    restore_backend(adapter.backend_, state)
    assert adapter.backend_._execution_reports == report
    # Optional trace metadata must not make older checkpoint state unreadable.
    old_state = dict(state)
    old_state.pop('graph_execution_reports')
    restore_backend(adapter.backend_, old_state)
    assert adapter.backend_._execution_reports == {}
    restore_backend(adapter.backend_, state)
    serializer = TemporalGraphSerializer(adapter.event_store)
    serializer.save(adapter, tmp_path)
    assert serializer.load(tmp_path).training_summary_['execution_report'] == report
