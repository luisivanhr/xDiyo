from copy import deepcopy
from functools import partial
import os
import time
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_info
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from execution_samples import TraceAdapter, ParentObserver
from model_selection_samples import sample, plan
from split_samples import rows_for, unchanged
from xdiyo_analytics.training import ExecutionPolicy, TrainingRunner, EstimatorAdapter, refit_model, load_model
from xdiyo_analytics.training.execution import run_fold_jobs


def forest():
    return EstimatorAdapter(make_pipeline(SimpleImputer(keep_empty_features=True),
        RandomForestRegressor(n_estimators=9, max_depth=3, random_state=19, n_jobs=3)))


def test_actual_process_overlap_completion_events_and_stable_fold_order(tmp_path):
    data = sample()
    observer = ParentObserver()
    factory = partial(TraceAdapter, folder=tmp_path, barrier=True, delay=.6)
    result = TrainingRunner(factory, observer=observer, execution=ExecutionPolicy(2)).run(data, plan(data), fold_ids=[1, 0])
    assert [f.fold_id for f in result.folds] == [1, 0]
    models = [f.model for f in result.folds]
    assert len({m.pid for m in models}) == 2 and os.getpid() not in {m.pid for m in models}
    assert all(m.parent_pid == os.getpid() for m in models)
    # The file barrier proves overlap; Windows may return equal monotonic ticks.
    assert max(m.started for m in models) <= min(m.finished for m in models)
    by_id = {f.fold_id: f for f in result.folds}
    assert by_id[1].model.finished < by_id[0].model.finished
    assert [call[2].fold_id for call in observer.calls] == [1, 1, 0, 0]
    assert all(pid == os.getpid() and stamp >= by_id[event.fold_id].model.finished
        for pid, stamp, event in observer.calls)
    assert all(f.training_summary['execution']['worker_pid'] == f.model.pid for f in result.folds)
    assert all(p['num_threads'] <= 1 for m in models for p in m.native_threads)


@pytest.mark.parametrize('execution', [None, ExecutionPolicy(1)])
def test_serial_observers_stay_live_and_unpicklable_objects_remain_valid(execution):
    observer = ParentObserver()
    data = sample()
    result = TrainingRunner(partial(TraceAdapter, delay=.1), observer=observer, execution=execution).run(data, plan(data))
    by_id = {f.fold_id: f for f in result.folds}
    assert all(pid == os.getpid() for pid, _, _ in observer.calls)
    first_events = [(stamp, event) for _, stamp, event in observer.calls if event.step == 1]
    assert all(stamp <= by_id[event.fold_id].model.finished for stamp, event in first_events)
    assert any(stamp < by_id[event.fold_id].model.finished for stamp, event in first_events)
    assert all(m.model.pid == os.getpid() for m in result.folds)
    assert all(('execution' in f.training_summary) == (execution is not None) for f in result.folds)


@pytest.mark.parametrize('layout', ['match', 'team_match'])
def test_seeded_parallel_predictions_match_serial_with_nullable_schema_and_exact_ids(layout):
    data = sample(layout=layout, shuffle=True)
    data.X.loc[data.X.index[:1], 'wave'] = np.nan
    data.y.iloc[rows_for(data, [11])] = np.nan
    data.metadata['nullable'] = pd.array([None if i % 2 else 'value' for i in range(len(data.X))], dtype='string[pyarrow]')
    before = deepcopy(data)
    serial = TrainingRunner(forest).run(data, plan(data), fold_ids=[1, 0])
    parallel = TrainingRunner(forest, execution=ExecutionPolicy(2, inner_threads=2)).run(data, plan(data), fold_ids=[1, 0])
    for expected, actual in zip(serial.folds, parallel.folds):
        assert expected.fold_id == actual.fold_id
        pd.testing.assert_frame_equal(actual.predictions['predict'], expected.predictions['predict'])
        pd.testing.assert_frame_equal(actual.y_true, expected.y_true)
        pd.testing.assert_frame_equal(actual.metadata, expected.metadata)
        assert actual.metadata.event_id.tolist() == data.metadata.iloc[actual.test_positions].event_id.tolist()
        assert all(int(v) > 2**63 for v in actual.metadata.event_id)
        for attr in ['train_positions', 'test_positions', 'score_positions', 'fit_positions', 'validation_positions']:
            np.testing.assert_array_equal(getattr(actual, attr), getattr(expected, attr))
        assert expected.model.estimator[-1].n_jobs == 3
        assert actual.model.estimator[-1].n_jobs == 2
    unchanged(data, before)


@pytest.mark.parametrize('layout', ['match', 'team_match'])
def test_parallel_fit_validation_and_selected_columns_keep_original_scope(layout):
    data = sample(layout=layout, shuffle=True)
    validation = {0: rows_for(data, [4]), 1: rows_for(data, [7])}
    result = TrainingRunner(TraceAdapter, feature_columns=['wave', 'signal'], validation=validation,
        execution=ExecutionPolicy(2)).run(data, plan(data))
    for fold in result.folds:
        np.testing.assert_array_equal(fold.validation_positions, validation[fold.fold_id])
        assert not np.isin(fold.fit_positions, fold.validation_positions).any()
        assert tuple(fold.model.context.X) == ('wave', 'signal')
        assert fold.model.context.validation.metadata.event_id.tolist() == data.metadata.iloc[validation[fold.fold_id]].event_id.tolist()


def test_refit_and_saved_pipeline_retain_execution_metadata(tmp_path):
    data = sample()
    fitted = refit_model(data, forest, train_positions=rows_for(data, range(8)), execution=ExecutionPolicy(2, inner_threads=2))
    assert fitted.execution == {'requested_device': 'cpu', 'device': 'cpu', 'worker_pid': os.getpid(), 'inner_threads': 2}
    restored = load_model(fitted.save(tmp_path / 'refit'))
    assert restored.execution == fitted.execution
    pd.testing.assert_frame_equal(restored.predict(data)['predict'], fitted.predict(data)['predict'])
    result = TrainingRunner(forest, execution=ExecutionPolicy(2)).run(data, plan(data))
    from xdiyo_analytics.training import save_model
    loaded = load_model(save_model(result, tmp_path / 'fold', fold_id=1))
    assert loaded.execution == result.folds[1].training_summary['execution']


def test_failed_worker_cancels_pending_job_and_next_pool_is_usable(tmp_path):
    def job(number, observer):
        (tmp_path / f'start-{number}').write_text(str(os.getpid()))
        deadline = time.monotonic() + 10
        while not all((tmp_path / f'start-{i}').exists() for i in [0, 1]):
            if time.monotonic() > deadline:
                raise TimeoutError('Other failure probe did not start')
            time.sleep(.01)
        if number == 0:
            raise RuntimeError('bounded worker failure')
        while not (tmp_path / 'release').exists() and time.monotonic() < deadline:
            time.sleep(.01)
        (tmp_path / 'unexpected-completion').write_text('late worker side effect')
    with pytest.raises(RuntimeError, match='bounded worker failure'):
        run_fold_jobs(job, [0, 1], ExecutionPolicy(2))
    (tmp_path / 'release').write_text('release after failure')
    pids = run_fold_jobs(lambda number, observer: os.getpid(), [0, 1], ExecutionPolicy(2))
    assert all(pid != os.getpid() for pid in pids)
    assert not (tmp_path / 'unexpected-completion').exists()


def test_explicit_serial_native_thread_limit_restored_after_failure():
    before = {p['prefix']: p['num_threads'] for p in threadpool_info()}
    with pytest.raises(RuntimeError, match='execution failure') as captured:
        TrainingRunner(partial(TraceAdapter, fail=True), execution=ExecutionPolicy(1, inner_threads=1)).run(sample(), plan(sample()))
    assert any('fold 0' in note for note in captured.value.__notes__)
    after = {p['prefix']: p['num_threads'] for p in threadpool_info()}
    assert {name: after[name] for name in before} == before


def test_empty_requested_folds_return_empty_result_without_fitting():
    result = TrainingRunner(lambda: (_ for _ in ()).throw(AssertionError('fit forbidden')),
        execution=ExecutionPolicy(2)).run(sample(), plan(sample()), fold_ids=[])
    assert result.folds == []
