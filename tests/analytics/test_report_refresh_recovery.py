"""Post-fit presentation changes reuse results and preserve fitted state."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from football_experiment_samples import prepared, ridge, post
from test_ui_workflow import ui_recipe
from xdiyo_analytics.analysis import PostTrainingAnalysis, PreTrainingAnalysis
from xdiyo_analytics.experiments import FootballExperiment
from xdiyo_analytics.reporting import PerformanceReporter, CorrelationAnalysis
from xdiyo_analytics.training import EstimatorAdapter, PredictionContext


def test_report_change_and_removal_reuse_predictions_and_refresh_saved_evidence(tmp_path, monkeypatch):
    data = prepared()
    experiment = FootballExperiment('Refresh reports', output_dir=tmp_path)
    result = experiment.run(data, model=ridge(), post_analysis=post())
    original_artifacts = deepcopy(result.record['artifacts'])
    original_prediction_files = {p: p.read_bytes() for p in result.path.rglob('*')
                                 if p.is_file() and ('models' in p.parts or p.suffix == '.parquet')}
    assert any(p.name.startswith('output-') and p.suffix == '.parquet'
               for p in original_prediction_files)
    def forbidden(*args, **kwargs):
        raise AssertionError('Refreshing analysis must not fit or predict')
    monkeypatch.setattr(EstimatorAdapter, 'fit', forbidden)
    monkeypatch.setattr(EstimatorAdapter, 'predict', forbidden)
    report = PostTrainingAnalysis({'Changed metric': PerformanceReporter(
        type='per_fold', partition='score', metrics=['mae'])}, title='Fresh analysis')
    refreshed = experiment.run(data, model=ridge(), post_analysis=report)
    assert refreshed.reused and refreshed.record['run_id'] == result.record['run_id']
    assert len(refreshed.post_report.studies) == 2
    assert {r['metric'] for r in refreshed.record['metrics']} == {'mae'}
    assert {r['study'] for r in refreshed.record['metrics']} == {'Changed metric'}
    reloaded = experiment.load(result.record['run_id'])
    assert [s.name for s in reloaded.post_report.studies] == ['Changed metric', 'Changed metric']
    assert reloaded.record['artifacts']['models'] == original_artifacts['models']
    assert reloaded.record['artifacts']['recovery'] != original_artifacts['recovery']
    assert (result.path / original_artifacts['recovery']).exists()
    assert original_prediction_files == {p: p.read_bytes() for p in original_prediction_files}
    removed = experiment.run(data, model=ridge(), post_analysis=PostTrainingAnalysis())
    assert removed.reused and removed.record['run_id'] == result.record['run_id']
    assert removed.post_report.studies == [] and removed.record['metrics'] == []
    assert experiment.load(result.record['run_id']).post_report.studies == []
    assert len(experiment.store.read_runs()) == 1


def test_restored_fold_models_predict_with_identical_scalers_without_refitting(tmp_path, monkeypatch):
    data = prepared()
    experiment = FootballExperiment('Restore models', output_dir=tmp_path)
    result = experiment.run(data, model=ridge(), post_analysis=post())
    def forbidden(*args, **kwargs):
        raise AssertionError('Restore must not fit')
    monkeypatch.setattr(EstimatorAdapter, 'fit', forbidden)
    loaded = experiment.load(result.record['run_id'])
    for original, restored in zip(result.training.folds, loaded.training.folds):
        rows = original.test_positions
        context = PredictionContext(data.dataset.X[list(original.feature_columns)].iloc[rows],
            data.dataset.metadata.iloc[rows], data.dataset.layout, data.dataset.match_columns,
            original.fold_id, definitions=data.dataset.definitions)
        context.X.index = context.X.index.rename('row_position')
        context.metadata.index = context.metadata.index.rename('row_position')
        pd.testing.assert_frame_equal(restored.model.predict(context)['predict'], original.predictions['predict'])
        np.testing.assert_array_equal(restored.model.estimator.named_steps['standardscaler'].mean_,
                                      original.model.estimator.named_steps['standardscaler'].mean_)
    assert all(f.model is None for f in experiment.load(result.record['run_id'], load_models=False).training.folds)


def test_pre_analysis_remains_part_of_execution_identity(tmp_path):
    data = prepared()
    experiment = FootballExperiment('Pre analysis identity', output_dir=tmp_path)
    first = experiment.run(data, model=ridge(), post_analysis=post())
    before = PreTrainingAnalysis({'Correlations': CorrelationAnalysis(type='overall', partition='train', methods=['spearman'])})
    second = experiment.run(data, model=ridge(), pre_analysis=before, post_analysis=post())
    assert not second.reused and first.record['run_id'] != second.record['run_id']


def test_presentation_source_changes_do_not_invalidate_numerical_source_does(tmp_path, monkeypatch):
    import xdiyo_analytics.experiments.recovery as recovery
    root = tmp_path / 'analytics'
    (root/'experiments').mkdir(parents=True)
    (root/'reporting').mkdir()
    (root/'training').mkdir()
    numerical = root/'training/runner.py'
    visual = root/'reporting/viewer.py'
    numerical.write_text('first numeric implementation')
    visual.write_text('first rendering implementation')
    monkeypatch.setattr(recovery, '__file__', str(root/'experiments/recovery.py'))
    first = recovery.execution_key({'parameter': 1})
    visual.write_text('different rendering implementation')
    assert recovery.execution_key({'parameter': 1}) == first
    numerical.write_text('different numeric implementation')
    assert recovery.execution_key({'parameter': 1}) != first


def test_failed_report_refresh_does_not_replace_published_manifest(tmp_path, monkeypatch):
    experiment = FootballExperiment('Atomic report', output_dir=tmp_path)
    result = experiment.run(prepared(), model=ridge(), post_analysis=post())
    manifest = result.path/'run.json'
    before = manifest.read_bytes()
    from xdiyo_analytics.reporting import AnalysisReport
    def fail(*args, **kwargs):
        raise RuntimeError('render failure')
    monkeypatch.setattr(AnalysisReport, 'to_html', fail)
    with pytest.raises(RuntimeError, match='render failure'):
        experiment.store.refresh_report(result.record['run_id'], result.post_report)
    assert manifest.read_bytes() == before
    assert experiment.load(result.record['run_id']).record['artifacts'] == result.record['artifacts']


def test_ui_recipe_post_reporters_and_options_change_without_fitting(ui_recipe, monkeypatch):
    from xdiyo_analytics.ui import run_recipe
    from xdiyo_analytics.ui.recipe import node
    first = run_recipe(ui_recipe)
    changed = deepcopy(ui_recipe)
    changed['post_reporters'] = {'New errors': node('reporting.PerformanceReporter', type='overall', partition='score', metrics=['mae'])}
    changed.setdefault('analysis_options', {})['post'] = {'title': 'Renamed output'}
    def forbidden(*args, **kwargs):
        raise AssertionError('Changing UI post analysis must not fit or predict')
    monkeypatch.setattr(EstimatorAdapter, 'fit', forbidden)
    monkeypatch.setattr(EstimatorAdapter, 'predict', forbidden)
    second = run_recipe(changed)
    assert second.reused and second.record['run_id'] == first.record['run_id']
    assert [s.name for s in second.post_report.studies] == ['New errors']
    monkeypatch.undo()
    changed['pre_reporters'] = {'Input correlations': node('reporting.CorrelationAnalysis',
        type='overall', partition='train', methods=['spearman'])}
    third = run_recipe(changed)
    assert not third.reused and third.record['run_id'] != first.record['run_id']
    assert [s.name for s in third.pre_report.studies] == ['Input correlations']


def test_model_based_report_refresh_uses_restored_coefficients(tmp_path, monkeypatch):
    from xdiyo_analytics.reporting import CoefficientReporter
    experiment = FootballExperiment('Coefficient refresh', output_dir=tmp_path)
    data = prepared()
    reporter = PostTrainingAnalysis({'Coefficients': CoefficientReporter(type='per_fold')})
    first = experiment.run(data, model=ridge(), post_analysis=reporter)
    def forbidden(*args, **kwargs):
        raise AssertionError('Coefficient refresh must use restored state, not refit/predict')
    monkeypatch.setattr(EstimatorAdapter, 'fit', forbidden)
    monkeypatch.setattr(EstimatorAdapter, 'predict', forbidden)
    second = experiment.run(data, model=ridge(), post_analysis=reporter)
    assert second.reused
    for before, after in zip(first.post_report.studies, second.post_report.studies):
        assert before.result.tables.keys() == after.result.tables.keys()
        assert before.result.tables
        for key in before.result.tables:
            pd.testing.assert_frame_equal(before.result.tables[key], after.result.tables[key])


def test_local_adapter_class_restores_and_predicts_without_fitting(tmp_path, monkeypatch):
    from test_football_experiment_search import CountingFactory
    from xdiyo_analytics.selection import Candidate
    experiment = FootballExperiment('Local class recovery', output_dir=tmp_path)
    data = prepared(holdout=True)
    events = []
    model = Candidate('Local Ridge', CountingFactory(.1, events))
    first = experiment.run(data, model=model)
    assert len(events) == 1
    def forbidden(*args, **kwargs):
        raise AssertionError('Local adapter recovery must not fit')
    monkeypatch.setattr(EstimatorAdapter, 'fit', forbidden)
    loaded = experiment.load(first.record['run_id'])
    original, restored = first.training.folds[0], loaded.training.folds[0]
    rows = original.test_positions
    context = PredictionContext(data.dataset.X[list(original.feature_columns)].iloc[rows],
        data.dataset.metadata.iloc[rows], data.dataset.layout, data.dataset.match_columns,
        original.fold_id, definitions=data.dataset.definitions)
    context.X.index = context.X.index.rename('row_position')
    context.metadata.index = context.metadata.index.rename('row_position')
    pd.testing.assert_frame_equal(restored.model.predict(context)['predict'], original.predictions['predict'])
    assert len(events) == 1


def test_explicit_model_saving_opt_out_keeps_numerical_recovery(tmp_path):
    experiment = FootballExperiment('Numerical only', output_dir=tmp_path)
    first = experiment.run(prepared(), model=ridge(), save_models=False)
    assert 'models' not in first.record['artifacts']
    loaded = experiment.load(first.record['run_id'])
    assert all(f.model is None for f in loaded.training.folds)
    for a, b in zip(first.training.folds, loaded.training.folds):
        pd.testing.assert_frame_equal(a.predictions['predict'], b.predictions['predict'])
