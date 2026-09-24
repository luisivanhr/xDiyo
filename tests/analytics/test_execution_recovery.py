from dataclasses import replace
from functools import partial
import json
import os
import numpy as np
import pandas as pd
import pytest
from execution_samples import TraceAdapter, configure_fake_device, fake_devices
from football_experiment_samples import prepared, ridge, post
from football_checkpoint_samples import NativeFactory
from model_selection_samples import plan, development
from xdiyo_analytics.experiments import FootballExperiment, RefitPolicy, ExperimentStore
from xdiyo_analytics.selection import Candidate, ModelSelection
from xdiyo_analytics.training import ExecutionPolicy, CheckpointPolicy, DeviceAdapter, TrainingRunner


def test_fixed_experiment_parallel_recovery_policy_identity_and_explicit_refit(tmp_path, monkeypatch):
    preparation = prepared()
    experiment = FootballExperiment('Parallel fixed', output_dir=tmp_path)
    model = ridge()
    saved = []
    original = experiment.store.save_run
    def save(*args, **kwargs):
        saved.append(os.getpid())
        return original(*args, **kwargs)
    monkeypatch.setattr(experiment.store, 'save_run', save)
    policy = ExecutionPolicy(2)
    result = experiment.run(preparation, model=model, execution=policy, post_analysis=post(),
        refit_policy=RefitPolicy(train_positions=np.arange(len(preparation.dataset.X))))
    assert all(f.training_summary['execution']['worker_pid'] != os.getpid() for f in result.training.folds)
    assert result.refit.execution['worker_pid'] == os.getpid()
    assert saved == [os.getpid()]
    cached = experiment.run(preparation, model=model, execution=policy, post_analysis=post(),
        refit_policy=RefitPolicy(train_positions=np.arange(len(preparation.dataset.X))))
    assert cached.reused and cached.refit.model is None and cached.refit.execution == result.refit.execution
    assert all(f.model is None for f in cached.training.folds)
    for first, second in zip(result.training.folds, cached.training.folds):
        assert first.training_summary['execution'] == second.training_summary['execution']
        pd.testing.assert_frame_equal(first.predictions['predict'], second.predictions['predict'])
    fresh = experiment.run(preparation, model=model, execution=ExecutionPolicy(1), post_analysis=post(),
        refit_policy=RefitPolicy(train_positions=np.arange(len(preparation.dataset.X))))
    assert not fresh.reused and fresh.record['run_id'] != result.record['run_id']
    assert fresh.training.folds[0].training_summary['execution']['worker_pid'] == os.getpid()


def test_parallel_search_reuses_completed_trials_and_execution_changes_identity(tmp_path):
    preparation = prepared()
    store = ExperimentStore(tmp_path, 'Reusable process trials')
    selection = ModelSelection([ridge(.1), replace(ridge(1.), name='other')], metrics='mse')
    args = dict(development_positions=development(preparation.dataset), experiment=store, resume=True)
    first = selection.run(preparation.dataset, preparation.split_plan, execution=ExecutionPolicy(2), **args)
    second = selection.run(preparation.dataset, preparation.split_plan, execution=ExecutionPolicy(2), **args)
    assert [t.saved_run_id for t in first.trials] == [t.saved_run_id for t in second.trials]
    assert all(f.model is None for t in second.trials for f in t.training.folds)
    third = selection.run(preparation.dataset, preparation.split_plan, execution=ExecutionPolicy(2, inner_threads=2), **args)
    assert first.run_group != third.run_group
    assert len(store.read_runs()) == 4


def test_interrupted_parallel_native_checkpoints_resume_exact_states(tmp_path):
    preparation = prepared()
    gate = tmp_path / 'resume.flag'
    factory = NativeFactory([], gate=gate)
    candidate = Candidate('Native', factory, config=factory.cache_key())
    experiment = FootballExperiment('Parallel checkpoint', output_dir=tmp_path)
    args = dict(model=candidate, execution=ExecutionPolicy(2), checkpoint_policy=CheckpointPolicy(), post_analysis=post())
    with pytest.raises(RuntimeError, match='native checkpoint'):
        experiment.run(preparation, **args)
    pointers = list((experiment.store.path / 'checkpoints').rglob('latest.json'))
    assert pointers and not experiment.store.read_runs()
    preserved = {}
    for pointer in pointers:
        state = pointer.parent / json.loads(pointer.read_text())['directory']
        assert 1 <= json.loads((state / 'metadata.json').read_text())['cursor'] <= 3
        preserved.update({p: p.read_bytes() for p in state.iterdir()})
    gate.write_text('continue')
    resumed = experiment.run(preparation, **args)
    reference_factory = NativeFactory([])
    reference = FootballExperiment('Parallel checkpoint reference', output_dir=tmp_path).run(preparation,
        model=Candidate('Native', reference_factory, config=reference_factory.cache_key()), execution=ExecutionPolicy(2),
        checkpoint_policy=CheckpointPolicy(), post_analysis=post())
    assert preserved == {p: p.read_bytes() for p in preserved}
    assert any(any(event[0] == 'start' and event[1] > 0 for event in f.model.owner.events) for f in resumed.training.folds)
    for actual, expected in zip(resumed.training.folds, reference.training.folds):
        pd.testing.assert_frame_equal(actual.predictions['predict'], expected.predictions['predict'])
        pd.testing.assert_frame_equal(actual.training_history, expected.training_history)
        np.testing.assert_array_equal(actual.model.weights, expected.model.weights)
        assert actual.training_summary['execution']['worker_pid'] != os.getpid()
    assert experiment.run(preparation, **args).reused


def test_checkpoint_keys_separate_resolved_device_and_thread_budget(tmp_path):
    preparation = prepared()
    def factory():
        return DeviceAdapter(NativeFactory([])(), fake_devices, configure_fake_device)
    wrapped = CheckpointPolicy().wrap(factory, tmp_path / 'states', 'fixed-test-namespace')
    def run(policy):
        return TrainingRunner(wrapped, execution=policy).run(preparation.dataset, preparation.split_plan)
    first = run(ExecutionPolicy(2))
    assert len(list((tmp_path / 'states').glob('*/latest.json'))) == 2
    reused = run(ExecutionPolicy(2))
    assert len(list((tmp_path / 'states').glob('*/latest.json'))) == 2
    assert all(f.model.owner.events == [('start', 6)] for f in reused.folds)
    run(ExecutionPolicy(2, inner_threads=2))
    assert len(list((tmp_path / 'states').glob('*/latest.json'))) == 4
    simulated = run(ExecutionPolicy(2, device='cuda:0'))
    assert len(list((tmp_path / 'states').glob('*/latest.json'))) == 6
    for original, gpu_routing_only in zip(first.folds, simulated.folds):
        pd.testing.assert_frame_equal(original.predictions['predict'], gpu_routing_only.predictions['predict'])
        assert gpu_routing_only.training_summary['execution']['device'] == 'cuda:0'


def test_holdout_experiment_inherits_execution_for_outer_fit(tmp_path):
    preparation = prepared(holdout=True)
    result = FootballExperiment('Policy holdout', output_dir=tmp_path).run(preparation,
        model_selection=ModelSelection([ridge()], metrics='mse'), selection_plan=plan(preparation.dataset),
        execution=ExecutionPolicy(2, inner_threads=2), post_analysis=post())
    assert all(f.training_summary['execution']['worker_pid'] != os.getpid() for f in result.selection.winner.training.folds)
    assert result.training.folds[0].training_summary['execution']['inner_threads'] == 2
    assert result.training.folds[0].training_summary['execution']['worker_pid'] == os.getpid()
