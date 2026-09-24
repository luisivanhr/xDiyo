"""Explicit final fitting and disjoint evaluation without model selection."""
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge, SGDRegressor, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xdiyo_analytics.training import (EstimatorAdapter, TrainingRunner, TrainingControl,
    IterativeAdapter, PartialFitBackend, refit_model)
from xdiyo_analytics.splits import Fold, SplitPlan
from split_samples import dataset, unchanged
from training_samples import MeanAdapter


def test_final_refit_is_explicit_fresh_and_preprocessing_uses_declared_rows():
    data = dataset(12); before = deepcopy(data); created = []
    def factory():
        model = EstimatorAdapter(make_pipeline(StandardScaler(), Ridge(alpha=1.)))
        created.append(model); return model
    plan = SplitPlan([Fold(np.arange(4), np.array([5, 6]), np.array([6]))], 12, np.arange(12))
    cv = TrainingRunner(factory).run(data, plan)
    assert len(created) == 1
    np.testing.assert_allclose(created[0].estimator[0].mean_, [1.5])
    final = refit_model(data, factory, train_positions=np.arange(7))
    assert len(created) == 2 and final.model is not cv.folds[0].model
    np.testing.assert_allclose(final.model.estimator[0].mean_, [3.])
    evaluated = final.evaluate(data, test_positions=[9, 8], score_positions=[8])
    assert len(created) == 2 and evaluated.folds[0].model is final.model
    np.testing.assert_allclose(evaluated.folds[0].predictions['predict'].outcome, [9.25, 8.375])
    assert evaluated.folds[0].test_positions.tolist() == [9, 8]
    assert evaluated.prediction_frame(scored_only=True).index.tolist() == [(0, 8)]
    assert evaluated.folds[0].train_positions.tolist() == list(range(7))
    assert final.fit_positions.tolist() == list(range(7)) and not len(final.validation_positions)
    unchanged(data, before)


@pytest.mark.parametrize('overlap', [0, 3])
def test_same_dataset_evaluation_excludes_fitting_and_monitoring_rows(overlap):
    data = dataset(8)
    factory = lambda: IterativeAdapter(lambda seed: PartialFitBackend(SGDRegressor(random_state=seed)))
    final = refit_model(data, factory, train_positions=np.arange(4), validation=[3],
                        control=TrainingControl(max_steps=2))
    assert final.fit_positions.tolist() == [0, 1, 2] and final.validation_positions.tolist() == [3]
    with pytest.raises(ValueError, match='development population'):
        final.evaluate(data, test_positions=[overlap, 6])


def test_different_dataset_keeps_original_development_provenance_out_of_local_arrays():
    original = dataset(8); incoming = dataset(3)
    incoming.metadata['event_id'] = pd.array([2**63 + 999 + i for i in range(3)], dtype='uint64[pyarrow]')
    final = refit_model(original, MeanAdapter, train_positions=[0, 1, 2, 3])
    with pytest.raises(ValueError): final.evaluate(incoming, test_positions=[0, 2])
    result = final.evaluate(incoming, test_positions=[0, 2], same_dataset=False)
    fold = result.folds[0]
    assert not len(fold.train_positions) and not len(fold.fit_positions) and not len(fold.validation_positions)
    assert fold.fold_metadata == {'stage': 'final_refit', 'same_dataset': False, 'development_positions': [0, 1, 2, 3]}
    assert fold.metadata.event_id.tolist() == [2**63 + 999, 2**63 + 1001]
    assert final.train_positions.tolist() == [0, 1, 2, 3]


def test_full_development_iterative_refit_uses_explicit_budget_without_test_monitoring():
    data = dataset(9); events = []
    factory = lambda: IterativeAdapter(lambda seed: PartialFitBackend(SGDRegressor(
        random_state=seed, shuffle=False, learning_rate='constant', eta0=.001)))
    final = refit_model(data, factory, train_positions=np.arange(6), observer=events.append,
        control=TrainingControl(max_steps=4, early_stopping=None, restore_best=False, monitor=None))
    assert final.fit_positions.tolist() == list(range(6))
    assert final.model.training_summary_['monitor'] == 'train_loss'
    assert final.model.training_summary_['retained_step'] == 4
    assert final.model.training_history_.metric.tolist() == ['train_loss'] * 4
    assert len(events) == 5 and events[-1].final
    result = final.evaluate(data, test_positions=[7, 8])
    pd.testing.assert_frame_equal(result.folds[0].training_history, final.model.training_history_)


def test_final_classifier_retains_class_probabilities_and_feature_selection():
    data = dataset(10); data.X['ignored'] = 999.
    data.y['outcome'] = np.arange(10) % 2
    final = refit_model(data, lambda: EstimatorAdapter(LogisticRegression(), ('predict', 'predict_proba')),
                        train_positions=np.arange(6), feature_columns=['form'], target_columns='outcome')
    result = final.evaluate(data, test_positions=[9, 7], score_positions=[])
    probabilities = result.folds[0].predictions['predict_proba']
    assert probabilities.columns.tolist() == [('outcome', 0), ('outcome', 1)]
    assert probabilities.index.tolist() == [9, 7]
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.)
    assert final.feature_columns == ('form',) and result.prediction_frame(scored_only=True).empty


@pytest.mark.parametrize('problem', ['partial_test', 'partial_score', 'score_outside_test'])
def test_final_evaluation_preserves_whole_matches_and_score_membership(problem):
    data = dataset(6, layout='team_match')
    final = refit_model(data, MeanAdapter, train_positions=np.arange(6))
    test, score = ([8], [8]) if problem == 'partial_test' else ([8, 9], [8]) if problem == 'partial_score' else ([8, 9], [10, 11])
    with pytest.raises(ValueError): final.evaluate(data, test_positions=test, score_positions=score)


def test_final_prediction_validates_layout_and_returns_copies():
    data = dataset(6); final = refit_model(data, MeanAdapter, train_positions=[0, 1, 2])
    output = final.predict(data, positions=[5, 4]); output['predict'].iloc[0, 0] = 999
    assert final.model.returned['predict'].iloc[0, 0] == 2.
    with pytest.raises(ValueError, match='layout'):
        final.predict(dataset(6, layout='team_match'), positions=[4])


@pytest.mark.parametrize('problem', ['index', 'empty_mapping', 'unnamed', 'duplicate_columns'])
def test_final_prediction_enforces_named_exact_index_outputs(problem):
    class Bad(MeanAdapter):
        def predict(self, context):
            outputs = super().predict(context)
            if problem == 'index': outputs['predict'].index = [0, 1]
            elif problem == 'empty_mapping': return {}
            elif problem == 'unnamed': return {'': outputs['predict']}
            else:
                outputs['predict'] = pd.concat([outputs['predict'], outputs['predict']], axis=1)
            return outputs
    data = dataset(6); final = refit_model(data, Bad, train_positions=[0, 1, 2])
    with pytest.raises((ValueError, TypeError)): final.predict(data, positions=[4, 5])
