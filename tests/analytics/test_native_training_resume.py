"""Cold-process trajectory parity and explicit cleanup contracts."""
from pathlib import Path
import json
import os
import subprocess
import sys
import pytest
import torch
from test_graph_preparation_memory import assert_exact


@pytest.mark.parametrize('kind', ['gat', 'gat_bayes', 'cnn_classification', 'cnn_regression'])
def test_fresh_process_resume_matches_uninterrupted_next_updates(tmp_path, kind):
    worker = Path(__file__).with_name('native_resume_worker.py')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1')
    for mode, code in [('reference', 0), ('interrupt', 73), ('resumed', 0)]:
        run = subprocess.run([sys.executable, str(worker), str(tmp_path), kind, mode], env=env, capture_output=True, text=True)
        assert run.returncode == code, run.stdout+run.stderr
    load = lambda name: torch.load(tmp_path/name, map_location='cpu', weights_only=True)
    reference, resumed = load('reference.pt'), load('resumed.pt')
    assert_exact(reference['updates'], load('prefix.pt')+resumed['updates'])
    assert_exact(reference['backend'], resumed['backend'])
    assert_exact(reference['rng'], resumed['rng'])
    assert (tmp_path/'reference.json').read_bytes() == (tmp_path/'resumed.json').read_bytes()
    # A completed restart must execute no extra optimizer update.
    run = subprocess.run([sys.executable, str(worker), str(tmp_path), kind, 'completed'], env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout+run.stderr
    assert not load('completed.pt')['updates']
    assert_exact(reference['backend'], load('completed.pt')['backend'])


def test_prediction_finalization_requires_verified_native_checkpoint(tmp_path):
    from native_resume_worker import fixture
    from xdiyo_analytics.training import TrainingControl, CheckpointPolicy
    from xdiyo_analytics.graphs import TemporalGraphSerializer
    torch.set_num_threads(1)
    adapter, train, query = fixture('gat')
    control = TrainingControl(max_steps=2)
    wrapped = CheckpointPolicy(every=100, finalization='prediction_only').wrap(lambda: adapter, tmp_path/'runs', 'finalize')()
    wrapped.fit_controlled(train, control)
    assert adapter._training_finalized and not hasattr(adapter.backend_, 'optimizer')
    assert not hasattr(adapter.backend_, 'train_groups')
    expected = adapter.predict(query)
    adapter.finalize_training(mode='prediction_only')
    serializer = TemporalGraphSerializer(event_store=adapter.event_store)
    serializer.save(adapter, tmp_path/'model')
    actual = serializer.load(tmp_path/'model').predict(query)
    import pandas as pd
    for name in expected: pd.testing.assert_frame_equal(actual[name], expected[name], check_exact=True)
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
