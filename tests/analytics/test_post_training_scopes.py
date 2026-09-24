"""Prediction occurrence scopes, explicit pooling and isolation from fitted runs."""
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from xdiyo_analytics.analysis import PostTrainingAnalysis
from xdiyo_analytics.reporting import PerformanceReporter
from post_training_samples import Capture, run_sample


@pytest.mark.parametrize('layout', ['match', 'team_match'])
@pytest.mark.parametrize('partition', ['test', 'score'])
def test_per_fold_exact_scope_and_order(layout, partition):
    training = run_sample(layout)
    capture = Capture(type='per_fold', partition=partition)
    report = PostTrainingAnalysis({'capture': capture}).run(training, fold_ids=[9, 4])
    assert [study.fold_id for study in report.studies] == [9, 4]
    for study, ctx, fold in zip(report.studies, capture.contexts, training.folds[::-1]):
        positions = fold.test_positions if partition == 'test' else fold.score_positions
        assert study.row_positions.tolist() == positions.tolist()
        assert study.scope.fold_id.tolist() == [fold.fold_id]*len(positions)
        assert study.scope.row_position.tolist() == positions.tolist()
        assert ctx.models == {fold.fold_id: fold.model}
        assert ctx.metadata.event_id.tolist() == fold.metadata.loc[positions].event_id.tolist()
        assert ctx.y.index.names == ['fold_id', 'row_position']
        assert ctx.n_matches == len(fold.metadata.loc[positions].event_id.unique())
        assert_frame_equal(ctx.y.droplevel('fold_id'), fold.y_true.loc[positions])


@pytest.mark.parametrize('policy', ['occurrences', 'first', 'last', 'mean'])
@pytest.mark.parametrize('reverse', [False, True])
def test_repeated_prediction_pooling_is_explicit_and_ordered(policy, reverse):
    training = run_sample()
    folds = training.folds[::-1] if reverse else training.folds
    capture = Capture(type='overall', partition='test', pooling=policy)
    report = PostTrainingAnalysis({'capture': capture}).run(training, fold_ids=[fold.fold_id for fold in folds])
    ctx = capture.contexts[0]
    occurrences = [(fold.fold_id, int(pos), fold.predictions['predict'].loc[pos, 'count'])
                   for fold in folds for pos in fold.test_positions]
    if policy == 'occurrences':
        expected = occurrences
    else:
        order = list(dict.fromkeys(pos for _, pos, _ in occurrences))
        if policy == 'last':
            order = [pos for i, (_, pos, _) in enumerate(occurrences) if pos not in [row[1] for row in occurrences[i+1:]]]
        expected = []
        for pos in order:
            matches = [row for row in occurrences if row[1] == pos]
            expected.append((-1, pos, sum(row[2] for row in matches)/len(matches)) if policy == 'mean'
                            else matches[-1 if policy == 'last' else 0])
    assert ctx.y.index.tolist() == [(fold, pos) for fold, pos, _ in expected]
    assert ctx.predictions['predict']['count'].tolist() == [value for _, _, value in expected]
    assert report.studies[0].scope_label == 'prediction pooling: '+policy
    assert report.studies[0].n_matches == 5


def test_repeats_require_policy_and_conflicts_are_not_silenced():
    training = run_sample()
    with pytest.raises(ValueError, match='explicit pooling'):
        PostTrainingAnalysis({'x': Capture(type='overall', partition='test')}).run(training)
    for section, column in [('y_true', 'count'), ('metadata', 'case')]:
        changed = deepcopy(training)
        getattr(changed.folds[1], section).loc[2, column] = 999
        with pytest.raises(ValueError, match='conflicting'):
            PostTrainingAnalysis({'x': Capture(type='overall', partition='test', pooling='first')}).run(changed)


def test_mean_pooling_propagates_missing_values_and_missing_class_columns():
    training = run_sample()
    for fold, classes in zip(training.folds, [[0, 1], [0, 2]]):
        frame = pd.DataFrame([[.4, .6]]*len(fold.test_positions), index=fold.y_true.index,
                             columns=pd.MultiIndex.from_product([['count'], classes]))
        fold.predictions = {'predict_proba': frame}
    training.folds[0].predictions['predict_proba'].loc[2, ('count', 0)] = np.nan
    capture = Capture(type='overall', partition='test', pooling='mean')
    PostTrainingAnalysis({'p': capture}).run(training)
    output = capture.contexts[0].predictions['predict_proba']
    assert output.loc[(-1, 2)].isna().all()
    assert output.loc[(-1, 4), ('count', 0)] == .4
    assert np.isnan(output.loc[(-1, 4), ('count', 2)])


def test_mean_rejects_class_decisions_and_different_output_or_target_sets():
    training = run_sample()
    training.folds[0].predictions['predict']['count'] = ['yes', 'no', 'yes']
    with pytest.raises(ValueError, match='numeric'):
        PostTrainingAnalysis({'p': Capture(type='overall', partition='test', pooling='mean')}).run(training)
    training = run_sample()
    training.folds[1].predictions['extra'] = training.folds[1].predictions['predict']
    with pytest.raises(ValueError, match='same named'):
        PostTrainingAnalysis({'p': Capture(type='overall', partition='test', pooling='first')}).run(training)
    training = run_sample()
    training.folds[1].y_true = training.folds[1].y_true.iloc[:, ::-1]
    with pytest.raises(ValueError, match='target columns/order'):
        PostTrainingAnalysis({'p': Capture(type='overall', partition='test', pooling='first')}).run(training)


def test_whole_population_copies_for_every_reporter():
    training = run_sample()
    before = deepcopy(training)
    mutator = Capture(type='overall', partition='test', pooling='first', mutate=True)
    observer = Capture(type='overall', partition='test', pooling='first')
    PostTrainingAnalysis({'mutator': mutator, 'observer': observer}).run(training)
    assert observer.contexts[0].y.iloc[0, 0] == 7
    assert observer.contexts[0].predictions['predict'].iloc[0, 0] == 6
    assert observer.contexts[0].metadata.case.iloc[0] == 4
    assert training.definitions == before.definitions == observer.contexts[0].definitions
    for current, old in zip(training.folds, before.folds):
        assert_frame_equal(current.y_true, old.y_true)
        assert_frame_equal(current.metadata, old.metadata)
        assert_frame_equal(current.predictions['predict'], old.predictions['predict'])


def test_score_population_and_pooled_metric_are_not_averages_of_folds():
    training = run_sample()
    training.folds[1].score_positions = np.array([2])
    report = PostTrainingAnalysis({'score': PerformanceReporter(type='overall', partition='score', metrics=['mse'])}).run(training)
    metrics = report.studies[0].result.tables['metrics']
    # Score rows: fold4 positions4,0 have errors1,1; fold9 position2 has error1.
    assert metrics.value.tolist() == [1., 1.]
    assert report.studies[0].row_positions.tolist() == [4, 0, 2]
    training.folds[1].predictions['predict'].loc[2, 'count'] = 0
    report = PostTrainingAnalysis({'score': PerformanceReporter(type='overall', partition='score', metrics=['mse'])}).run(training)
    assert report.studies[0].result.tables['metrics'].iloc[0].value == pytest.approx(11/3)


def test_empty_score_scope_retains_output_schema():
    training = run_sample()
    training.folds[0].score_positions = np.array([], dtype=int)
    report = PostTrainingAnalysis({'score': PerformanceReporter(type='per_fold', partition='score', metrics=['mse'])}).run(training, fold_ids=[4])
    assert report.studies[0].scope.columns.tolist() == ['fold_id', 'row_position']
    assert report.studies[0].scope.empty and report.studies[0].n_matches == 0
    assert report.studies[0].result.tables['metrics'].status.eq('no_valid_observations').all()


def test_timeline_stably_orders_all_prediction_frames_and_requires_times():
    training = run_sample('team_match')
    capture = Capture(type='timeline', partition='test', pooling='occurrences')
    PostTrainingAnalysis({'timeline': capture}).run(training)
    ctx = capture.contexts[0]
    assert ctx.metadata.kickoff_at.is_monotonic_increasing
    for frame in ctx.predictions.values():
        assert frame.index.equals(ctx.y.index)
    training.folds[0].metadata.loc[8, 'kickoff_at'] = pd.NaT
    with pytest.raises(ValueError, match='timestamps'):
        PostTrainingAnalysis({'timeline': capture}).run(training)


@pytest.mark.parametrize('fold_ids', [[4, 4], [0], [4, 100], []])
def test_invalid_or_empty_selected_folds(fold_ids):
    with pytest.raises(ValueError):
        PostTrainingAnalysis({'x': Capture(type='per_fold', partition='score')}).run(run_sample(), fold_ids=fold_ids)


@pytest.mark.parametrize('mode,partition,pooling', [('bad', 'test', None), ('overall', 'bad', None),
                                                 ('overall', 'experiment', None), ('overall', 'test', 'automatic')])
def test_invalid_scope_configuration(mode, partition, pooling):
    with pytest.raises(ValueError):
        PostTrainingAnalysis({'x': Capture(type=mode, partition=partition, pooling=pooling)}).run(run_sample())


def test_empty_analysis_and_reporter_errors_are_explicit():
    assert PostTrainingAnalysis().run().studies == []
    with pytest.raises(ValueError, match='nonempty'):
        PostTrainingAnalysis({'': Capture(type='per_fold', partition='test')}).run(run_sample())
    class Broken(Capture):
        def run(self, ctx):
            raise RuntimeError('expected sentinel')
    with pytest.raises(RuntimeError) as error:
        PostTrainingAnalysis({'broken': Broken(type='per_fold', partition='test')}).run(run_sample())
    assert "reporter 'broken'" in error.value.__notes__[0]
    class Wrong(Capture):
        def run(self, ctx):
            return {'table': 1}
    with pytest.raises(TypeError, match='StudyResult'):
        PostTrainingAnalysis({'wrong': Wrong(type='per_fold', partition='test')}).run(run_sample())
