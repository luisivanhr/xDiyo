from functools import partial
import os
import sys
from types import SimpleNamespace
import pytest
from execution_samples import TraceAdapter, configure_fake_device, fake_devices
from model_selection_samples import sample, plan
from xdiyo_analytics.training import ExecutionPolicy, DeviceAdapter, TrainingRunner, EstimatorAdapter, torch_devices
from sklearn.linear_model import Ridge


@pytest.mark.parametrize('kwargs', [
    {'n_jobs': 0}, {'n_jobs': -2}, {'n_jobs': True}, {'n_jobs': 1.5},
    {'device': 'gpu'}, {'device': 'cuda:-1'}, {'device': 'CPU'}, {'device': None},
    {'inner_threads': 0}, {'inner_threads': True}, {'inner_threads': 1.5},
    {'gpu_jobs': 0}, {'gpu_jobs': -1}, {'gpu_jobs': True}, {'gpu_jobs': 2.5},
])
def test_execution_policy_rejects_invalid_explicit_options(kwargs):
    with pytest.raises(ValueError):
        ExecutionPolicy(**kwargs)


def test_worker_budget_uses_available_cpu_and_explicit_gpu_cap(monkeypatch):
    import joblib
    monkeypatch.setattr(joblib, 'cpu_count', lambda: 4)
    assert ExecutionPolicy(-1).workers(8) == 4
    assert ExecutionPolicy(8).workers(2) == 2
    assert ExecutionPolicy(8, 'auto').workers(8) == 1
    assert ExecutionPolicy(8, 'cuda:2', gpu_jobs=2).workers(8) == 2


@pytest.mark.parametrize('requested,expected', [('cpu', 'cpu'), ('auto', 'cuda:0'), ('cuda', 'cuda:0'), ('cuda:2', 'cuda:2')])
def test_simulated_device_routing_applies_before_fit_and_records_resolution(requested, expected):
    data = sample()
    factory = lambda: DeviceAdapter(TraceAdapter(), fake_devices, configure_fake_device)
    result = TrainingRunner(factory, execution=ExecutionPolicy(4, requested)).run(data, plan(data))
    for fold in result.folds:
        assert fold.model.estimator.simulated_device == expected
        assert fold.model.device_ == expected
        assert fold.training_summary['execution']['requested_device'] == requested
        assert fold.training_summary['execution']['device'] == expected


@pytest.mark.parametrize('requested', ['cuda', 'cuda:0'])
@pytest.mark.parametrize('kind', ['plain', 'sklearn', 'declared_unavailable'])
def test_explicit_cuda_rejects_unsupported_or_unavailable_adapters(requested, kind):
    factory = {'plain': TraceAdapter, 'sklearn': lambda: EstimatorAdapter(Ridge()),
        'declared_unavailable': lambda: DeviceAdapter(TraceAdapter(), lambda: ['cpu'], configure_fake_device)}[kind]
    with pytest.raises(ValueError, match='CUDA|unavailable'):
        TrainingRunner(factory, execution=ExecutionPolicy(2, requested)).run(sample(), plan(sample()))


def test_auto_cpu_fallback_is_conservatively_serial_and_plain_sklearn_stays_cpu():
    data = sample()
    results = TrainingRunner(lambda: EstimatorAdapter(Ridge()), execution=ExecutionPolicy(8, 'auto')).run(data, plan(data))
    assert all(f.training_summary['execution']['device'] == 'cpu' for f in results.folds)
    assert all(f.training_summary['execution']['worker_pid'] == os.getpid() for f in results.folds)


@pytest.mark.parametrize('available', [[], ['gpu'], ['cpu', 1]])
def test_invalid_or_empty_device_discovery_fails_before_fit(available):
    factory = lambda: DeviceAdapter(TraceAdapter(), lambda: available, configure_fake_device)
    with pytest.raises(ValueError):
        TrainingRunner(factory, execution=ExecutionPolicy(device='auto')).run(sample(), plan(sample()))


@pytest.mark.parametrize('requested,resolved', [('cpu', 'cuda:0'), ('cuda:2', 'cuda:0'), ('cuda', 'cpu'), ('auto', 'gpu')])
def test_native_adapter_cannot_misreport_requested_device(requested, resolved):
    class Invalid(TraceAdapter):
        def configure_device(self, device):
            return resolved
    with pytest.raises(ValueError, match='honor|resolved'):
        TrainingRunner(Invalid, execution=ExecutionPolicy(device=requested)).run(sample(), plan(sample()))


@pytest.mark.parametrize('available,count,expected', [(False, 0, ['cpu']), (True, 2, ['cpu', 'cuda:0', 'cuda:1'])])
def test_optional_torch_discovery_uses_runtime_availability(monkeypatch, available, count, expected):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: available, device_count=lambda: count)))
    assert torch_devices() == expected


def test_invalid_execution_object_is_rejected():
    with pytest.raises(TypeError, match='ExecutionPolicy'):
        TrainingRunner(TraceAdapter, execution={'n_jobs': 2}).run(sample(), plan(sample()))
