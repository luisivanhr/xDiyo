"""Run roles, explicit trial linkage, and retained training evidence."""
from copy import deepcopy
import io
import json
from uuid import UUID
import numpy as np
import pandas as pd
import pytest
from xdiyo_analytics.analysis import PostTrainingAnalysis
from xdiyo_analytics.experiments import ExperimentStore
from xdiyo_analytics.reporting import ExperimentLeaderboardReporter, LearningCurveReporter
from test_post_training_experiments import computed_run
from training_controls_samples import model_result


def board(store, **options):
    reporter = ExperimentLeaderboardReporter(weights={'mse': 1},
        selectors={'mse': {'target': 'count'}}, **options)
    return PostTrainingAnalysis({'runs': reporter}).run(experiment=store).studies[0].result.tables['leaderboard']


def test_manual_group_trial_final_linkage_and_default_leaderboard(tmp_path, monkeypatch):
    training, report = computed_run()
    store = ExperimentStore(tmp_path, 'manual choices')
    group = store.start_run('one declared selection', config={'seed': 7})
    UUID(group)
    manifest = json.loads((store.path / 'run-groups' / (group+'.json')).read_text())
    assert manifest['name'] == 'one declared selection' and manifest['config'] == {'seed': 7}
    trial = store.save_run(training, report, name='trial', config={'alpha': .5},
                           role='trial', run_group=group, save_predictions=False)
    assert board(store).empty
    failed = store.save_failure(name='failed trial', config={}, error='example', role='trial', run_group=group)
    # The final result is separately supplied: linking must not copy trial metrics.
    final_report = deepcopy(report)
    final_report.studies[0].result.tables['metrics'].loc[:, 'value'] = 99.
    final = store.save_run(training, final_report, name='final evaluation', config={'alpha': .5},
        run_group=group, selected_trial_id=trial['run_id'], save_predictions=False)
    assert final['metrics'][0]['value'] == 99. and final['metrics'] != trial['metrics']
    assert final['selected_trial_id'] == trial['run_id']
    def no_parquet(*args, **kwargs): raise AssertionError('Leaderboard read predictions')
    monkeypatch.setattr(pd, 'read_parquet', no_parquet)
    standard, detailed = board(store), board(store, include_trials=True, run_group=group)
    assert standard.run_id.tolist() == [final['run_id']] and standard.role.tolist() == ['final']
    assert set(detailed.run_id) == {trial['run_id'], failed['run_id'], final['run_id']}
    assert detailed.set_index('run_id').loc[failed['run_id'], 'status'] == 'failed'
    assert len(store.read_runs()) == 3 and len(store.read_runs(role='trial', run_group=group)) == 2
    assert board(store, run_group='absent').empty
    with pytest.raises(ValueError, match='already has a final'):
        store.save_failure(name='second final', config={}, error='example', run_group=group)


@pytest.mark.parametrize('problem', ['unknown_trial', 'wrong_group', 'failed_trial', 'final_as_trial', 'trial_link', 'no_group'])
def test_selected_trial_must_be_successful_trial_in_same_group(tmp_path, problem):
    training, report = computed_run(); store = ExperimentStore(tmp_path, problem)
    group, other = store.start_run('first'), store.start_run('other')
    trial = store.save_run(training, report, name='trial', config={}, role='trial', run_group=group, save_predictions=False)
    bad = store.save_failure(name='failed', config={}, error='example', role='trial', run_group=group)
    standalone = store.save_run(training, report, name='standalone', config={}, save_predictions=False)
    selected, save_group, role = trial['run_id'], group, 'final'
    if problem == 'unknown_trial': selected = 'unknown'
    elif problem == 'wrong_group': save_group = other
    elif problem == 'failed_trial': selected = bad['run_id']
    elif problem == 'final_as_trial': selected = standalone['run_id']
    elif problem == 'trial_link': role = 'trial'
    else: save_group = None
    before = store.read_runs()
    with pytest.raises(ValueError, match='selected_trial_id'):
        store.save_run(training, report, name='invalid', config={}, role=role,
            run_group=save_group, selected_trial_id=selected, save_predictions=False)
    assert store.read_runs() == before and not list((store.path/'runs').glob('.pending-*'))


@pytest.mark.parametrize('kwargs', [dict(role='search'), dict(role='trial'), dict(run_group='../outside')])
def test_invalid_role_or_group_rejected(tmp_path, kwargs):
    store = ExperimentStore(tmp_path, 'invalid')
    with pytest.raises(ValueError): store.save_failure(name='example', config={}, error='example', **kwargs)
    assert store.read_runs() == []


def test_failed_final_consumes_group_slot_and_legacy_records_are_final(tmp_path):
    store = ExperimentStore(tmp_path, 'legacy'); group = store.start_run('single result')
    failure = store.save_failure(name='final failure', config={}, error='example', run_group=group)
    with pytest.raises(ValueError, match='already has a final'):
        store.save_failure(name='retry', config={}, error='example', run_group=group)
    path = store.path / 'runs' / failure['run_id'] / 'run.json'
    record = json.loads(path.read_text()); record.pop('role'); record.pop('run_group'); record.pop('selected_trial_id')
    path.write_text(json.dumps(record), encoding='utf-8')
    assert store.read_runs(role='final', run_group=record['run_id']) == [record]
    row = board(store).iloc[0]
    assert row.role == 'final' and row.run_group == record['run_id'] and row.status == 'failed'
    assert store.read_runs(role='trial') == []
    with pytest.raises(ValueError): store.read_runs(role='unknown')


def test_training_artifact_without_predictions_preserves_histories_and_provenance(tmp_path):
    training, report = computed_run()
    fold = training.folds[0]
    fold.fit_positions = np.array([1, 3]); fold.validation_positions = np.array([5])
    fold.training_history = pd.DataFrame({'fold_id': [4, 4], 'attempt': [0, 0], 'step': [1, 2],
        'metric': ['train_loss']*2, 'value': [2., np.nan], 'learning_rate': [.1, .05]})
    fold.training_summary = {'selected_attempt': 0, 'best_step': 1,
        'attempts': [{'seed': 7, 'termination_reason': 'nonfinite_monitor'}]}
    before = deepcopy(training)
    store = ExperimentStore(tmp_path, 'evidence')
    saved = store.save_run(training, report, name='evidence', config={}, save_predictions=False)
    directory = store.path / 'runs' / saved['run_id']
    assert saved['artifacts'] == {'tables': 'tables.json', 'training': 'training.json'}
    assert not list(directory.rglob('*.parquet')) and not list(directory.rglob('*.pkl'))
    records = json.loads((directory/'training.json').read_text())
    first, fallback = records
    assert first['train_positions'] == fold.train_positions.tolist()
    assert first['fit_positions'] == [1, 3] and first['validation_positions'] == [5]
    assert first['summary'] == fold.training_summary and first['fold_metadata'] == fold.fold_metadata
    recovered = pd.read_json(io.StringIO(json.dumps(first['history'])), orient='table')
    pd.testing.assert_frame_equal(recovered, fold.training_history)
    assert fallback['fit_positions'] == training.folds[1].train_positions.tolist()
    assert fallback['validation_positions'] == []
    pd.testing.assert_frame_equal(fold.training_history, before.folds[0].training_history)
    assert fold.training_summary == before.folds[0].training_summary


def test_model_only_report_saved_without_prediction_population(tmp_path):
    class Model: pass
    result = model_result([Model()])
    report = PostTrainingAnalysis({'history': LearningCurveReporter(type='overall')}).run(result)
    store = ExperimentStore(tmp_path, 'model only')
    record = store.save_run(result, report, name='unavailable curve', config={}, save_predictions=False)
    assert record['metrics'] == [] and record['artifacts']['training'] == 'training.json'
    assert board(store).status.tolist() == ['unranked_missing_or_undefined']


@pytest.mark.parametrize('name', ['', '   ', None, 3])
def test_group_name_required(tmp_path, name):
    with pytest.raises(ValueError): ExperimentStore(tmp_path, 'names').start_run(name)
