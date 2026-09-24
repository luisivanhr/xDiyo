import os
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from execution_samples import ParentObserver
from execution_torch_samples import TorchLinear, move_torch_model
from model_selection_samples import sample, plan
from xdiyo_analytics.training import (TrainingRunner, IterativeAdapter, PartialFitBackend,
    TrainingControl, ExecutionPolicy, DeviceAdapter, torch_devices)


def iterative():
    return IterativeAdapter(lambda seed: PartialFitBackend(
        SGDRegressor(random_state=seed, learning_rate='constant', eta0=.001),
        preprocessor=StandardScaler()))


def test_real_iterative_loss_events_and_histories_match_serial_parallel():
    data = sample()
    control = TrainingControl(max_steps=4, seed=31)
    serial_observer, parallel_observer = ParentObserver(), ParentObserver()
    serial = TrainingRunner(iterative, control=control, observer=serial_observer).run(data, plan(data))
    parallel = TrainingRunner(iterative, control=control, observer=parallel_observer,
        execution=ExecutionPolicy(2)).run(data, plan(data))
    assert len(serial_observer.calls) == len(parallel_observer.calls) >= 8
    assert all(pid == os.getpid() for pid, _, _ in parallel_observer.calls)
    for first, second in zip(serial.folds, parallel.folds):
        pd.testing.assert_frame_equal(first.predictions['predict'], second.predictions['predict'])
        pd.testing.assert_frame_equal(first.training_history, second.training_history)
    for fold_id in [0, 1]:
        first = [event for _, _, event in serial_observer.calls if event.fold_id == fold_id]
        second = [event for _, _, event in parallel_observer.calls if event.fold_id == fold_id]
        assert first == second


def test_optional_real_cuda_parameters_and_input_tensors_are_on_observed_device():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('No observed usable CUDA device')
    data = sample()
    factory = lambda: DeviceAdapter(TorchLinear(), torch_devices, move_torch_model)
    cpu = TrainingRunner(factory, execution=ExecutionPolicy(1, 'cpu')).run(data, plan(data))
    for requested in ['auto', 'cuda:0']:
        actual = TrainingRunner(factory, execution=ExecutionPolicy(2, requested)).run(data, plan(data))
        for expected, fold in zip(cpu.folds, actual.folds):
            assert fold.model.parameter_device == fold.model.fit_tensor_device == fold.model.prediction_tensor_device == 'cuda:0'
            assert fold.training_summary['execution']['device'] == 'cuda:0'
            assert fold.training_summary['execution']['worker_pid'] == os.getpid()
            np.testing.assert_allclose(fold.predictions['predict'], expected.predictions['predict'], rtol=2e-5, atol=2e-5)
