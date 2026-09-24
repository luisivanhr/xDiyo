from functools import partial
import os
import numpy as np
import pandas as pd
import pytest
from execution_samples import TraceAdapter, ParentObserver
from model_selection_samples import sample, plan, development, outer
from test_model_selection_nested import outer_plan, inner_plan
from split_samples import rows_for
from xdiyo_analytics.analysis import PreTrainingAnalysis
from xdiyo_analytics.reporting import FeatureSelector, FeatureSelection, StudyResult
from xdiyo_analytics.selection import Candidate, ModelSelection
from xdiyo_analytics.training import ExecutionPolicy
from xdiyo_analytics.experiments import ExperimentStore


@pytest.mark.parametrize('layout', ['match', 'team_match'])
def test_candidate_selectors_stay_parent_scoped_while_fit_and_predict_run_in_workers(layout):
    data = sample(layout=layout, shuffle=True)
    seen = []
    observer = ParentObserver()
    class Selector(FeatureSelector):
        def select(self, context):
            seen.append((os.getpid(), set(context.metadata.case)))
            return StudyResult('signal', selection=FeatureSelection(('signal',), pd.DataFrame({'feature': ['signal']})))
    candidate = Candidate('scoped', TraceAdapter, observer=observer, validation=rows_for(data, [4]),
        pre_analysis=PreTrainingAnalysis({'pick': Selector(type='per_fold', partition='train')}), features_from='pick')
    selected = ModelSelection([candidate], metrics='mse').run(data, plan(data),
        development_positions=development(data), execution=ExecutionPolicy(2))
    assert len(seen) == 2 and all(pid == os.getpid() and 4 not in cases for pid, cases in seen)
    assert len(observer.calls) == 4 and all(pid == os.getpid() for pid, _, _ in observer.calls)
    for fold in selected.winner.training.folds:
        assert fold.model.pid != os.getpid() and fold.feature_columns == ('signal',)
        assert set(fold.selection.row_positions) == set(fold.fit_positions)
        assert set(fold.model.context.validation.metadata.case) == {4}
        assert 4 not in fold.model.context.metadata.case.tolist()
        assert fold.metadata.event_id.tolist() == data.metadata.iloc[fold.test_positions].event_id.tolist()
    final = selected.evaluate(data, outer(data), validation=None)
    assert final.folds[0].training_summary['execution']['device'] == 'cpu'
    assert final.folds[0].model.pid == os.getpid()  # One outer job is serial.
    overridden = selected.evaluate(data, outer(data), validation=None, execution=None)
    assert 'execution' not in overridden.folds[0].training_summary


def test_candidates_are_sequential_and_parent_owns_trial_publication(tmp_path, monkeypatch):
    data = sample()
    store = ExperimentStore(tmp_path / 'store', 'Process search')
    saved = []
    original = store.save_run
    def save(*args, **kwargs):
        saved.append(os.getpid())
        return original(*args, **kwargs)
    monkeypatch.setattr(store, 'save_run', save)
    items = [Candidate(name, partial(TraceAdapter, folder=tmp_path / 'trace', label=name, delay=.05)) for name in ['first', 'second']]
    result = ModelSelection(items, metrics='mse').run(data, plan(data), development_positions=development(data),
        experiment=store, execution=ExecutionPolicy(2))
    first = [f.model for f in result.trials[0].training.folds]
    second = [f.model for f in result.trials[1].training.folds]
    assert min(m.started for m in second) >= max(m.finished for m in first)
    assert saved == [os.getpid(), os.getpid()]
    assert len(store.read_runs()) == 2


@pytest.mark.parametrize('layout', ['match', 'team_match'])
def test_nested_inner_jobs_parallel_outer_loops_serial_without_nested_pools(layout, monkeypatch):
    import joblib
    original = joblib.Parallel
    launches = []
    def parallel(*args, **kwargs):
        launches.append((os.getpid(), kwargs['n_jobs']))
        return original(*args, **kwargs)
    monkeypatch.setattr(joblib, 'Parallel', parallel)
    data = sample(layout=layout, shuffle=True)
    observer = ParentObserver()
    candidates = [Candidate(name, partial(TraceAdapter, label=name), observer=observer) for name in ['a', 'b']]
    result = ModelSelection(candidates, metrics='mse').run_nested(data, outer_plan(data), inner_plan, execution=ExecutionPolicy(2))
    assert launches == [(os.getpid(), 2)] * 4
    assert all(pid == os.getpid() for pid, _, _ in observer.calls)
    assert [f.fold_id for f in result.training.folds] == [0, 1]
    for i, selection in result.selections.items():
        for trial in selection.trials:
            assert all(f.model.pid != os.getpid() and f.model.parent_pid == os.getpid() for f in trial.training.folds)
        fitted = result.training.folds[i]
        assert fitted.model.pid == os.getpid()
        np.testing.assert_array_equal(fitted.test_positions, outer_plan(data).folds[i].test)
        assert fitted.training_summary['execution']['device'] == 'cpu'


def test_failed_parallel_candidate_is_recorded_and_next_candidate_completes(tmp_path, monkeypatch):
    store = ExperimentStore(tmp_path, 'Failed process trial')
    failures = []
    original = store.save_failure
    def record(**kwargs):
        failures.append(os.getpid())
        return original(**kwargs)
    monkeypatch.setattr(store, 'save_failure', record)
    data = sample()
    result = ModelSelection([Candidate('bad', partial(TraceAdapter, fail=True)), Candidate('good', TraceAdapter)],
        metrics='mse', on_error='record').run(data, plan(data), development_positions=development(data),
        experiment=store, execution=ExecutionPolicy(2))
    assert result.winner.candidate.name == 'good'
    assert failures == [os.getpid()] and [t.record['status'] for t in result.trials] == ['failed', 'complete']
    assert 'injected execution failure' in result.trials[0].error
