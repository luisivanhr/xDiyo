"""Execution policy parity, strictness, isolation, UI and artifact compatibility."""
from dataclasses import replace
import json
import os
import subprocess
import sys

import pandas as pd
import pytest
torch = pytest.importorskip('torch')

from test_temporal_graph import sample, config, context
from test_graph_preparation_memory import assert_exact
from xdiyo_analytics.graphs import TemporalGraphAdapter, TemporalGraphSerializer
from xdiyo_analytics.graphs.adapter import _Backend, torch_threads
from xdiyo_analytics.graphs.execution import deterministic_execution, prepare_determinism
from xdiyo_analytics.training import TrainingControl


def flags():
    return (torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)


@pytest.mark.parametrize('value', [0, 1, 'true', None])
def test_requires_boolean(value):
    with pytest.raises(ValueError, match='deterministic must be a boolean'):
        config(deterministic=value)


def test_nested_flags_restored_after_exception_and_disabled_inherits():
    original = flags()
    try:
        torch.use_deterministic_algorithms(False, warn_only=True)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        before = flags()
        with pytest.raises(RuntimeError, match='sentinel'):
            with deterministic_execution(config(deterministic=True), 'cpu'):
                assert flags() == (True, False, True, False)
                with deterministic_execution(config(), 'cpu'):
                    assert flags() == (True, False, True, False)
                with deterministic_execution(config(deterministic=True), 'cpu'):
                    assert flags() == (True, False, True, False)
                raise RuntimeError('sentinel')
        assert flags() == before
        with deterministic_execution(config(), 'cpu'):
            assert flags() == before
    finally:
        torch.use_deterministic_algorithms(original[0], warn_only=original[1])
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = original[2:]


def test_cuda_workspace_preflight(monkeypatch):
    cfg = config(deterministic=True)
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    prepare_determinism(cfg, 'cpu')
    assert 'CUBLAS_WORKSPACE_CONFIG' not in os.environ
    prepare_determinism(cfg, 'cuda:0')
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':16:8')
    prepare_determinism(cfg, 'cuda:0')
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':16:8'
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', 'invalid')
    with pytest.raises(RuntimeError, match='Restart the notebook'):
        prepare_determinism(cfg, 'cuda:0')
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG')
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    with pytest.raises(RuntimeError, match='before CUDA initialization'):
        prepare_determinism(cfg, 'cuda:0')
    prepare_determinism(config(), 'cuda:0')  # Disabled path has no new requirement.


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('cnn', [False, True])
def test_explicit_policy_matches_external_flags_full_updates_and_exports(tmp_path, device, cnn):
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA not installed')
    if cnn:
        from test_heatmap_cnn import graph_setup
        from xdiyo_analytics.neural import Reconstruction
        spec, store = graph_setup(reconstruction=Reconstruction(loss='kl'))
        cfg = spec.config
    else:
        cfg, store = config(), sample()
    enabled = replace(cfg, deterministic=True)
    fit = context(store, [1, 2, 3]); fit.validation = context(store, [5, 6])
    pred = context(store, [7, 8], fit=False)
    before = flags()
    with torch_threads(1):
        prepare_determinism(enabled, device)
        old, new = _Backend(cfg, store, 17, device), _Backend(enabled, store, 17, device)
        with deterministic_execution(enabled, device):
            old.initialize(fit)
        new.initialize(fit)
        assert flags() == before
        assert_exact(old.preprocessor.state(), new.preprocessor.state())
        for name in ('train_groups', 'validation_groups'):
            for a, b in zip(getattr(old, name), getattr(new, name)):
                assert_exact(a, b)
        for _ in range(3):
            with deterministic_execution(enabled, device):
                expected = old.checkpoint_metrics('train_loss', old.step())
            actual = new.checkpoint_metrics('train_loss', new.step())
            assert flags() == before
            assert_exact(expected, actual)
            assert_exact(old.model.state_dict(), new.model.state_dict())
            assert_exact([p.grad for p in old.model.parameters()], [p.grad for p in new.model.parameters()])
            assert_exact(old.optimizer.state_dict(), new.optimizer.state_dict())
        with deterministic_execution(enabled, device):
            expected = old.predict(pred)
        for name, frame in expected.items():
            pd.testing.assert_frame_equal(frame, new.predict(pred)[name], check_exact=True)
        # Execution policy is recorded as part of the artifact identity.
        assert old.model_version != new.model_version
        model = TemporalGraphAdapter(enabled, store, device=device)
        model.backend_ = new
        model.training_summary_, model.training_history_ = {}, pd.DataFrame()
        serializer = TemporalGraphSerializer(store, device=device)
        serializer.save(model, tmp_path)
        loaded = serializer.load(tmp_path)
        assert loaded.config.deterministic is True
        assert_exact(new.model.state_dict(), loaded.backend_.model.state_dict())
        queries = pd.DataFrame({'team_id':[1, 2], 'issuance_at':[store.schedule.kickoff_at.max()]*2})
        external = TemporalGraphAdapter(cfg, store, device=device)
        external.backend_ = old
        for method in ('encode_teams', 'attention_weights'):
            with deterministic_execution(enabled, device):
                expected = getattr(external, method)(queries)
            actual = getattr(model, method)(queries)
            # Only the execution-policy identity differs; coordinates, attention,
            # evidence and cutoff metadata must be exactly identical.
            pd.testing.assert_frame_equal(expected.drop(columns='model_version'),
                                          actual.drop(columns='model_version'), check_exact=True)
            pd.testing.assert_frame_equal(getattr(model, method)(queries), getattr(loaded, method)(queries), check_exact=True)
        for name, frame in new.predict(pred).items():
            pd.testing.assert_frame_equal(frame, loaded.predict(pred)[name], check_exact=True)
        assert flags() == before


def test_loading_pre_option_artifact_keeps_fingerprint(tmp_path):
    store = sample()
    model = TemporalGraphAdapter(config(), store, control=TrainingControl(max_steps=1))
    with torch_threads(1):
        model.fit(context(store, [1, 2, 3]))
        serializer = TemporalGraphSerializer(store)
        serializer.save(model, tmp_path)
        path = tmp_path/'graph.json'
        metadata = json.loads(path.read_text())
        del metadata['config']['deterministic']
        path.write_text(json.dumps(metadata))
        restored = serializer.load(tmp_path)
    assert restored.config.deterministic is False
    assert restored.backend_.model_version == model.backend_.model_version


def test_fresh_cuda_session_auto_workspace_and_strict_kernel():
    if not torch.cuda.is_available():
        pytest.skip('CUDA not installed')
    env = dict(os.environ)
    env.pop('CUBLAS_WORKSPACE_CONFIG', None)
    script = '''
import os, torch
from xdiyo_analytics.graphs import TemporalGraphConfig
from xdiyo_analytics.graphs.execution import deterministic_execution
assert not torch.cuda.is_initialized()
cfg = TemporalGraphConfig('UTC', deterministic=True)
with deterministic_execution(cfg, 'cuda:0'):
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
    x = torch.ones(32, 32, device='cuda', requires_grad=True)
    (x @ x).sum().backward()
    # Strict mode must propagate unsupported-kernel errors, never warn/fallback.
    try:
        torch.ones(10, device='cuda').put_(torch.tensor([1, 1], device='cuda'), torch.ones(2, device='cuda'))
    except RuntimeError as e:
        assert 'deterministic' in str(e)
    else:
        raise AssertionError('expected unsupported nondeterministic put_ to fail')
assert not torch.are_deterministic_algorithms_enabled()
'''
    subprocess.run([sys.executable, '-c', script], env=env, check=True, capture_output=True, text=True)
