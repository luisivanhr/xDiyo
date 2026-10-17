"""Cold-process trajectory parity and explicit cleanup contracts."""
from pathlib import Path
import json
import os
import subprocess
import sys
import pytest
import torch
from test_graph_preparation_memory import assert_exact


@pytest.mark.parametrize('kind', ['gat', 'gat_fixture', 'gat_dependency', 'gat_dependency_statistics', 'gat_dependency_fallback', 'cnn_classification', 'cnn_regression'])
def test_fresh_process_resume_matches_uninterrupted_next_updates(tmp_path, kind):
    worker = Path(__file__).with_name('native_resume_worker.py')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1')
    for mode, code in [('reference', 0), ('interrupt', 73), ('resumed', 0)]:
        run = subprocess.run([sys.executable, str(worker), str(tmp_path), kind, mode], env=env, capture_output=True, text=True)
        assert run.returncode == code, run.stdout+run.stderr
    load = lambda name: torch.load(tmp_path/name, map_location='cpu', weights_only=True)
    reference, resumed = load('reference.pt'), load('resumed.pt')
    if kind in ('gat_dependency', 'gat_dependency_statistics', 'gat_dependency_fallback'):
        active = load('prefix-control.pt')
        assert active['stale'] > 0 and active['best_step'] >= 1
        assert active['schedule_stale'] == 0  # Patience-one scheduler already fired.
        trace = reference['updates'][0]['execution_trace']
        assert trace['fallback_groups'] == (trace['groups'] if kind == 'gat_dependency_fallback' else 0)
    assert_exact(reference['updates'], load('prefix.pt')+resumed['updates'])
    assert_exact(reference['backend'], resumed['backend'])
    assert_exact(reference['rng'], resumed['rng'])
    assert (tmp_path/'reference.json').read_bytes() == (tmp_path/'resumed.json').read_bytes()
    # A completed restart must execute no extra optimizer update.
    run = subprocess.run([sys.executable, str(worker), str(tmp_path), kind, 'completed'], env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout+run.stderr
    assert not load('completed.pt')['updates']
    assert_exact(reference['backend'], load('completed.pt')['backend'])
    assert (tmp_path/'reference.json').read_bytes() == (tmp_path/'completed.json').read_bytes()


@pytest.mark.parametrize('execution_mode', ['native', 'dependency', 'dependency_lean'])
def test_all_graph_execution_ablations_cold_resume_74_targets_and_ratings(tmp_path, execution_mode):
    """Each UI ablation must preserve its own exact resumed trajectory."""
    worker = Path(__file__).with_name('native_resume_worker.py')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1',
        XDIYO_RESUME_DEVICE='cpu', XDIYO_RESUME_CACHE='0', XDIYO_RESUME_BOUNDARY='',
        XDIYO_RESUME_EXECUTION_MODE=execution_mode)
    for mode, code in [('reference', 0), ('interrupt', 73), ('resumed', 0), ('completed', 0)]:
        run = subprocess.run([sys.executable, str(worker), str(tmp_path),
            'gat_dependency_statistics', mode], env=env, capture_output=True, text=True)
        assert run.returncode == code, run.stdout+run.stderr
    load = lambda name: torch.load(tmp_path/name, map_location='cpu', weights_only=True)
    reference, resumed, completed = load('reference.pt'), load('resumed.pt'), load('completed.pt')
    assert_exact(reference['updates'], load('prefix.pt')+resumed['updates'])
    for key in ('backend', 'rng'):
        assert_exact(reference[key], resumed[key])
    assert_exact(reference['backend'], completed['backend'])
    assert not completed['updates']
    control = load('prefix-control.pt')
    assert control['stale'] > 0 and control['best_step'] >= 1
    assert control['schedule_stale'] == 0
    assert reference['updates'][0]['execution_trace']['requested_mode'] == execution_mode
    assert (tmp_path/'reference.json').read_bytes() == (tmp_path/'resumed.json').read_bytes()
    assert (tmp_path/'reference.json').read_bytes() == (tmp_path/'completed.json').read_bytes()


@pytest.mark.parametrize('kind', ['gat', 'cnn_classification', 'cnn_regression'])
def test_prediction_finalization_requires_verified_native_checkpoint(tmp_path, kind):
    from native_resume_worker import fixture
    from xdiyo_analytics.training import TrainingControl, CheckpointPolicy
    from xdiyo_analytics.graphs import TemporalGraphSerializer
    torch.set_num_threads(1)
    adapter, train, query = fixture(kind)
    control = TrainingControl(max_steps=2)
    wrapped = CheckpointPolicy(every=100, finalization='prediction_only').wrap(lambda: adapter, tmp_path/'runs', 'finalize')()
    wrapped.fit_controlled(train, control)
    assert adapter._training_finalized and not hasattr(adapter.backend_, 'optimizer')
    assert not hasattr(adapter.backend_, 'train_groups')
    assert not hasattr(adapter.backend_, 'train')
    expected = adapter.predict(query)
    adapter.finalize_training(mode='prediction_only')
    from xdiyo_analytics.neural import HeatmapCNNSerializer
    serializer = TemporalGraphSerializer(event_store=adapter.event_store) if kind == 'gat' else HeatmapCNNSerializer()
    serializer.save(adapter, tmp_path/'model')
    actual = serializer.load(tmp_path/'model').predict(query)
    import pandas as pd
    for name in expected: pd.testing.assert_frame_equal(actual[name], expected[name], check_exact=True)
    if kind.startswith('cnn'):
        pd.testing.assert_frame_equal(adapter.encode(query), serializer.load(tmp_path/'model').encode(query), check_exact=True)
    else:
        teams = query.metadata.home_id
        queries = pd.DataFrame({'team_id': teams, 'issuance_at': adapter.event_store.issue_times(query.metadata, adapter.config)})
        loaded = serializer.load(tmp_path/'model')
        pd.testing.assert_frame_equal(adapter.encode_teams(queries), loaded.encode_teams(queries), check_exact=True)
        pd.testing.assert_frame_equal(adapter.attention_weights(queries), loaded.attention_weights(queries), check_exact=True)
    with pytest.raises(RuntimeError, match='finalized'):
        adapter.fit_controlled(train, control)
    # Same completed checkpoint restores training state before optional cleanup.
    CheckpointPolicy(every=100).wrap(lambda: adapter, tmp_path/'runs', 'finalize')().fit_controlled(train, control)
    assert hasattr(adapter.backend_, 'optimizer')


def test_corruption_and_inference_exports_are_not_training_checkpoints(tmp_path):
    from xdiyo_analytics.training.native_resume import read_checkpoint
    with pytest.raises(ValueError, match='Incomplete'):
        read_checkpoint(tmp_path)


def test_checkpoint_rejects_changed_context_profile_and_corruption(tmp_path):
    from native_resume_worker import fixture
    from xdiyo_analytics.training import TrainingControl, CheckpointPolicy
    from xdiyo_analytics.training.native_resume import read_checkpoint
    torch.set_num_threads(1)
    adapter, train, query = fixture('cnn_classification')
    policy = CheckpointPolicy(); control = TrainingControl(max_steps=1)
    policy.wrap(lambda: adapter, tmp_path/'runs', 'reject')().fit_controlled(train, control)
    path = Path(adapter.training_checkpoint_)
    state, manifest = read_checkpoint(path)
    assert state['winner']['optimizer']['state']
    with pytest.raises(ValueError, match='fingerprint'):
        read_checkpoint(path, identity='wrong')
    with pytest.raises(ValueError, match='profile'):
        read_checkpoint(path, profile={})
    train.X.iloc[0, -1] += 2
    with pytest.raises(ValueError, match='fingerprint'):
        adapter.fit_resumable(train, control=control, checkpoint=path, save_checkpoint=lambda writer: None)
    with (path/'training.pt').open('ab') as stream: stream.write(b'corruption')
    with pytest.raises(ValueError, match='corrupt'):
        read_checkpoint(path)


@pytest.mark.parametrize('change', [
    {'mode': 'dependency_lean'}, {'host_memory_bytes': 128*1024**2+1},
    {'device_memory_bytes': 256*1024**2+1}, {'max_graphs': 1}, {'headroom_bytes': 1024**3+1}])
def test_dependency_real_training_checkpoint_rejects_execution_changes(tmp_path, change):
    from dataclasses import replace
    from graph_execution_fixture import execution_fixture
    from xdiyo_analytics.graphs import TemporalGraphAdapter
    from xdiyo_analytics.training import TrainingControl, CheckpointPolicy
    from xdiyo_analytics.training.native_resume import read_checkpoint
    torch.set_num_threads(1)
    adapter, train, _ = execution_fixture(statistics=False)
    control = TrainingControl(max_steps=1)
    CheckpointPolicy().wrap(lambda: adapter, tmp_path/'runs', 'execution')().fit_controlled(train, control)
    path = Path(adapter.training_checkpoint_)
    _, manifest = read_checkpoint(path)
    assert manifest['profile']['graph_execution']['mode'] == 'dependency_packed'
    changed = replace(adapter.config, execution=replace(adapter.config.execution, **change))
    replacement = TemporalGraphAdapter(changed, adapter.event_store)
    writes = []
    with pytest.raises(ValueError, match='fingerprint|execution profile'):
        replacement.fit_resumable(train, control=control, checkpoint=path,
                                  save_checkpoint=lambda writer: writes.append(writer))
    assert not writes


def test_finalization_ui_recipe_roundtrip_and_stale_weights_rejected(tmp_path):
    from xdiyo_analytics.ui.recipe import catalog_for_ui
    from xdiyo_analytics.training import CheckpointPolicy, TrainingControl
    from native_resume_worker import fixture
    catalog = catalog_for_ui()
    policy = CheckpointPolicy(finalization='prediction_only')
    assert catalog.build(json.loads(json.dumps(catalog.encode(policy)))) == policy
    schema = next(c for c in catalog.schema() if c['id'] == 'training.CheckpointPolicy')
    field = next(f for f in schema['fields'] if f['name'] == 'finalization')
    assert field['choices'] == ['retain', 'prediction_only'] and field['default'] == 'retain'
    adapter, train, _ = fixture('gat')
    CheckpointPolicy().wrap(lambda: adapter, tmp_path, 'changed')().fit_controlled(train, TrainingControl(max_steps=1))
    with torch.no_grad(): next(adapter.backend_.model.parameters()).add_(1)
    with pytest.raises(ValueError, match='parameters'):
        adapter.finalize_training(mode='prediction_only')
    assert hasattr(adapter.backend_, 'optimizer')


def test_thinned_offers_do_not_capture_optimizer_buffers(tmp_path, monkeypatch):
    from native_resume_worker import fixture
    from xdiyo_analytics.training import TrainingControl, CheckpointPolicy
    import xdiyo_analytics.training.resume_loop as loop
    adapter, train, _ = fixture('cnn_classification')
    calls = []
    capture = loop.capture_backend
    def recorded(backend):
        calls.append(1)
        return capture(backend)
    monkeypatch.setattr(loop, 'capture_backend', recorded)
    CheckpointPolicy(every=100).wrap(lambda: adapter, tmp_path/'runs', 'thin')().fit_controlled(train, TrainingControl(max_steps=3))
    assert calls == [1]  # Only the selected backend at mandatory final publication.


@pytest.mark.skipif(os.environ.get('XDIYO_CUDA_CHECKS') != '1', reason='Explicit free-GPU gate required')
@pytest.mark.parametrize('kind', ['gat_dependency', 'gat_dependency_statistics', 'gat_dependency_fallback'])
def test_cuda_dependency_cold_process_resume(tmp_path, kind):
    """Fresh CUDA processes preserve the opted-in plan, AdamW and patience."""
    worker = Path(__file__).with_name('native_resume_worker.py')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               XDIYO_RESUME_DEVICE='cuda:0', XDIYO_RESUME_CACHE='0', PYTHONDONTWRITEBYTECODE='1')
    for mode, code in [('reference', 0), ('interrupt', 73), ('resumed', 0), ('completed', 0)]:
        run = subprocess.run([sys.executable, str(worker), str(tmp_path), kind, mode], env=env, capture_output=True, text=True)
        assert run.returncode == code, run.stdout+run.stderr
    load = lambda name: torch.load(tmp_path/name, map_location='cpu', weights_only=True)
    reference, resumed, completed = load('reference.pt'), load('resumed.pt'), load('completed.pt')
    active = load('prefix-control.pt')
    assert active['stale'] > 0 and active['best_step'] >= 1
    assert_exact(reference['updates'], load('prefix.pt')+resumed['updates'])
    for key in ('backend', 'rng'):
        assert_exact(reference[key], resumed[key])
    assert_exact(reference['backend'], completed['backend'])
    assert not completed['updates']
    assert (tmp_path/'reference.json').read_bytes() == (tmp_path/'resumed.json').read_bytes()
    assert (tmp_path/'reference.json').read_bytes() == (tmp_path/'completed.json').read_bytes()


@pytest.mark.skipif(os.environ.get('XDIYO_CUDA_CHECKS') != '1', reason='Explicit free-GPU gate required')
@pytest.mark.parametrize('kind', ['gat', 'gat_fixture', 'cnn_classification', 'cnn_regression'])
def test_cuda_cold_process_resume_with_input_residency(tmp_path, kind):
    worker = Path(__file__).with_name('native_resume_worker.py')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', CUBLAS_WORKSPACE_CONFIG=':4096:8',
               XDIYO_RESUME_DEVICE='cuda:0', XDIYO_RESUME_CACHE='1', PYTHONDONTWRITEBYTECODE='1')
    for mode, code in [('reference', 0), ('interrupt', 73), ('resumed', 0), ('completed', 0)]:
        run = subprocess.run([sys.executable, str(worker), str(tmp_path), kind, mode], env=env, capture_output=True, text=True)
        assert run.returncode == code, run.stdout+run.stderr
    load = lambda name: torch.load(tmp_path/name, map_location='cpu', weights_only=True)
    reference, resumed, completed = load('reference.pt'), load('resumed.pt'), load('completed.pt')
    assert_exact(reference['updates'], load('prefix.pt')+resumed['updates'])
    for key in ('backend', 'rng'):
        assert_exact(reference[key], resumed[key])
    assert_exact(reference['backend'], completed['backend'])
    assert not completed['updates']
    assert (tmp_path/'reference.json').read_bytes() == (tmp_path/'resumed.json').read_bytes()


@pytest.mark.parametrize('kind', ['gat', 'gat_fixture', 'cnn_classification', 'cnn_regression'])
@pytest.mark.parametrize('boundary', ['numpy', 'restart', 'early_stopping', 'nonfinite_monitor'])
def test_cold_resume_boundary_and_numpy_controls(tmp_path, kind, boundary):
    worker = Path(__file__).with_name('native_resume_worker.py')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1',
               XDIYO_RESUME_BOUNDARY=boundary, XDIYO_RESUME_DEVICE='cpu', XDIYO_RESUME_CACHE='0')
    for mode, code in [('reference', 0), ('interrupt', 73), ('resumed', 0), ('completed', 0)]:
        run = subprocess.run([sys.executable, str(worker), str(tmp_path), kind, mode], env=env, capture_output=True, text=True)
        assert run.returncode == code, run.stdout+run.stderr
    load = lambda name: torch.load(tmp_path/name, map_location='cpu', weights_only=True)
    reference, resumed, completed = load('reference.pt'), load('resumed.pt'), load('completed.pt')
    assert_exact(reference['updates'], load('prefix.pt')+resumed['updates'])
    for key in ['backend', 'rng']: assert_exact(reference[key], resumed[key])
    assert_exact(reference['backend'], completed['backend'])
    assert not completed['updates']
    assert (tmp_path/'reference.json').read_bytes() == (tmp_path/'resumed.json').read_bytes()


def test_numpy_control_and_checkpoint_metadata_are_plain_scalars(tmp_path):
    import numpy as np
    from xdiyo_analytics.training import TrainingControl, EarlyStopping, ReduceOnPlateau
    from xdiyo_analytics.training.native_resume import write_checkpoint, read_checkpoint
    c = TrainingControl(max_steps=np.int64(2), seed=np.uint64(19), restarts=np.int32(1),
        early_stopping=EarlyStopping(np.int64(2), np.float32(0)),
        scheduler=ReduceOnPlateau(np.int64(2), np.float32(.5), np.float64(1e-6)))
    assert all(type(getattr(c,k)) is int for k in ['max_steps','seed','restarts'])
    assert type(c.early_stopping.patience) is int and type(c.scheduler.patience) is int
    state = dict(attempt=np.int64(2), restarts=np.int64(1), rows=[{'fold': np.int64(0)}],
                 attempts=[], winner=None, winning_value=np.float32(1), winner_info=None,
                 rng={}, active=None, stopped=np.bool_(True))
    write_checkpoint(tmp_path, state, 'identity', {})
    actual, manifest = read_checkpoint(tmp_path)
    assert actual == state and manifest['complete'] is True
    assert type(actual['attempt']) is int and type(actual['stopped']) is bool
    assert type(actual['winning_value']) is float and type(actual['rows'][0]['fold']) is int
