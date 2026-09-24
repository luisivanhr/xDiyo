"""Report fold selection never changes which folds were trained."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from football_experiment_samples import prepared, ridge
from post_training_samples import Capture, run_sample
from xdiyo_analytics.analysis import PostTrainingAnalysis
from xdiyo_analytics.experiments import FootballExperiment
from xdiyo_analytics.reporting import PerformanceReporter


@pytest.mark.parametrize('mode', ['overall', 'per_fold'])
@pytest.mark.parametrize('layout', ['match', 'team_match'])
def test_configured_filter_uses_only_selected_prediction_occurrences(mode, layout):
    training = run_sample(layout)
    original = deepcopy(training.predictions_by_fold())
    capture = Capture(type=mode, partition='test', pooling='occurrences')
    report = PostTrainingAnalysis({'captured': capture}, fold_ids=[9]).run(training)
    context = capture.contexts[0]
    assert len(report.studies) == 1
    assert set(context.y.index.get_level_values('fold_id')) == {9}
    expected = training.folds[1]
    assert context.row_positions.tolist() == expected.test_positions.tolist()
    pd.testing.assert_frame_equal(context.predictions['predict'].droplevel('fold_id'), expected.predictions['predict'])
    assert set(context.models) == {9}
    assert [fold.fold_id for fold in training.folds] == [4, 9]
    for fold_id, predictions in training.predictions_by_fold().items():
        pd.testing.assert_frame_equal(predictions, original[fold_id])


def test_run_override_and_none_default_preserve_configured_selection():
    training = run_sample()
    capture = Capture(type='overall', partition='test', pooling='occurrences')
    configured = PostTrainingAnalysis({'captured': capture}, fold_ids=[9])
    for override, expected in [(None, {9}), ([4], {4}), ([4, 9], {4, 9})]:
        configured.run(training, fold_ids=override)
        assert set(capture.contexts[-1].y.index.get_level_values('fold_id')) == expected
    assert configured.fold_ids == [9]
    PostTrainingAnalysis({'captured': capture}).run(training)
    assert set(capture.contexts[-1].y.index.get_level_values('fold_id')) == {4, 9}


@pytest.mark.parametrize('ids', [[99], [-1], [9, 9], ['9'], []])
def test_invalid_or_empty_report_fold_selection_is_rejected(ids):
    analysis = PostTrainingAnalysis({'metrics': PerformanceReporter(type='overall', partition='test',
                                                                   pooling='occurrences', metrics=['mse'])}, fold_ids=ids)
    with pytest.raises(ValueError, match='fold'):
        analysis.run(run_sample())


@pytest.mark.parametrize('mode', ['overall', 'per_fold'])
def test_experiment_propagates_report_filter_but_trains_all_folds(tmp_path, mode):
    preparation = prepared()
    analysis = PostTrainingAnalysis({'errors': PerformanceReporter(type=mode, partition='test', metrics=['mse'])},
                                    fold_ids=[1])
    experiment = FootballExperiment('Selected report folds', output_dir=tmp_path)
    result = experiment.run(preparation, model=ridge(), post_analysis=analysis)
    assert [fold.fold_id for fold in result.training.folds] == [0, 1]
    study = result.post_report.studies[0]
    selected = result.training.folds[1]
    assert len(result.post_report.studies) == 1
    assert set(study.scope.fold_id) == {1}
    assert study.row_positions.tolist() == selected.test_positions.tolist()
    expected = np.mean((selected.y_true['target'] - selected.predictions['predict']['target']) ** 2)
    metric = study.result.tables['metrics'].iloc[0]
    assert metric.value == pytest.approx(expected)
    assert metric.n == len(selected.test_positions)
    assert result.record['metrics'][0]['n'] == len(selected.test_positions)
    cached = experiment.run(preparation, model=ridge(), post_analysis=analysis)
    assert cached.reused
    assert len(cached.training.folds) == 2
    assert set(cached.post_report.studies[0].scope.fold_id) == {1}
