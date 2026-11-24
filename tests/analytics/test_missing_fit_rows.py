"""Fit-only missingness: identities and untouched holdouts, not just row counts."""
from copy import deepcopy
import json
import numpy as np
import pandas as pd
import pytest
from training_samples import sample, plan, MeanAdapter
from split_samples import rows_for
from xdiyo_analytics.training import MissingFitRows, TrainingRunner, fit_predict, refit_model, EstimatorAdapter


class ControlledMean(MeanAdapter):
    def fit_controlled(self, context, control):
        self.fit(context)


@pytest.mark.parametrize('layout', ['match', 'team_match'])
def test_fit_only_before_imputation_and_common_evaluation(layout):
    data = sample(layout=layout)
    fold = plan(data).folds[0]
    validation = rows_for(data, [3])
    missing = rows_for(data, [1])[0]
    data.X.loc[missing, 'signal'] = np.nan
    data.X.loc[rows_for(data, [3, 6]), 'signal'] = np.nan
    data.X['unused'] = np.nan  # unrelated missing columns must not remove rows
    before = deepcopy(data.X)
    policy = MissingFitRows(['signal'])
    filtered = fit_predict(data, fold, ControlledMean, validation=validation, missing_fit_rows=policy)
    kept = fit_predict(data, fold, ControlledMean, validation=validation)
    expected = fold.train[~np.isin(fold.train, rows_for(data, [1, 3]))]
    assert filtered.fit_positions.tolist() == expected.tolist()
    assert filtered.validation_positions.tolist() == validation.tolist()
    assert filtered.model.fit_context.validation.X.signal.isna().all()
    assert filtered.model.predict_context.X.signal.isna().any()
    for attr in ('train_positions', 'test_positions', 'score_positions'):
        np.testing.assert_array_equal(getattr(filtered, attr), getattr(kept, attr))
    pd.testing.assert_frame_equal(filtered.y_true, kept.y_true)
    pd.testing.assert_frame_equal(before, data.X)
    audit = filtered.training_summary['missing_fit_rows']
    assert set(audit['excluded_positions']) == set(rows_for(data, [1]))
    assert audit['missing_by_feature'] == {'signal': 1}
    assert len(kept.fit_positions) > len(filtered.fit_positions)
    runner = TrainingRunner(ControlledMean, validation=validation, missing_fit_rows=policy)
    assert runner.selection_plan(data, plan(data)).folds[0].train.tolist() == expected.tolist()


def test_real_imputer_learns_only_eligible_fitting_rows_and_refit():
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.dummy import DummyRegressor
    data = sample(); fold = plan(data).folds[0]
    data.X.loc[1, 'signal'] = np.nan
    data.X.loc[1, 'curve'] = 10000
    factory = lambda: EstimatorAdapter(make_pipeline(SimpleImputer(), DummyRegressor()))
    policy = MissingFitRows(['signal'])
    result = fit_predict(data, fold, factory, missing_fit_rows=policy)
    imputer = result.model.estimator.steps[0][1]
    assert imputer.statistics_[1] == pytest.approx(data.X.curve.iloc[result.fit_positions].mean())
    final = refit_model(data, factory, train_positions=fold.train, missing_fit_rows=policy)
    np.testing.assert_array_equal(final.fit_positions, result.fit_positions)
    assert len(final.predict(data, positions=fold.test)['predict']) == len(fold.test)


def test_fail_closed_required_features_and_hidden_validation():
    data = sample(); fold = plan(data).folds[0]
    with pytest.raises(ValueError, match='explicit'):
        MissingFitRows([])
    with pytest.raises(ValueError, match='selected model inputs'):
        fit_predict(data, fold, MeanAdapter, feature_columns=['curve'], missing_fit_rows=MissingFitRows(['signal']))
    data.X.signal = np.nan
    with pytest.raises(ValueError, match='no fitting rows'):
        fit_predict(data, fold, MeanAdapter, missing_fit_rows=MissingFitRows(['signal']))
    from sklearn.ensemble import HistGradientBoostingRegressor
    data = sample()
    with pytest.raises(ValueError, match='external validation'):
        fit_predict(data, fold, lambda: EstimatorAdapter(HistGradientBoostingRegressor()), target_columns=['first'], missing_fit_rows=MissingFitRows(['signal']))


def test_recipe_json_roundtrip_and_legacy_default(tmp_path):
    from xdiyo_analytics.ui import default_recipe
    from xdiyo_analytics.ui.recipe import make_candidate, read_recipe, node
    from xdiyo_analytics.ui.catalog import default_catalog
    recipe = default_recipe()
    assert make_candidate(recipe, default_catalog()).missing_fit_rows is None
    recipe['candidate']['missing_fit_rows'] = node('training.MissingFitRows', required_features=['home::corners_mean', 'away::corners_mean'])
    path = tmp_path/'recipe.json'; path.write_text(json.dumps(recipe))
    loaded = read_recipe(path)
    assert loaded == recipe
    candidate = make_candidate(loaded, default_catalog())
    assert candidate.missing_fit_rows.required_features == ('home::corners_mean', 'away::corners_mean')
    assert default_catalog().encode(candidate.missing_fit_rows) == recipe['candidate']['missing_fit_rows']


def test_calibration_reserved_before_filter_and_weights_use_actual_fit():
    from xdiyo_analytics.training import ProbabilityCalibrator, split_training_rows
    from xdiyo_analytics.weighting import ClassWeightPolicy
    data = sample(); train = np.arange(8)
    calibration = ProbabilityCalibrator(fraction=.25)
    original_fit, valid = split_training_rows(data, train, validation=[4], calibration=calibration)
    _, reserved, _ = calibration.partition(data, train)
    data.X.loc[[1, 4, *reserved], 'signal'] = np.nan
    fit, after_valid = split_training_rows(data, train, validation=[4], calibration=calibration,
                                           missing_fit_rows=MissingFitRows(['signal']))
    assert fit.tolist() == [r for r in original_fit if r != 1]
    np.testing.assert_array_equal(valid, after_valid)
    np.testing.assert_array_equal(reserved, calibration.partition(data, train)[1])
    class WeightedMean(MeanAdapter):
        supports_sample_weight = True
    data.y = pd.DataFrame({'class': np.arange(len(data.X)) % 2})
    result = fit_predict(data, plan(data).folds[0], WeightedMean,
        missing_fit_rows=MissingFitRows(['signal']), weighting=ClassWeightPolicy())
    context = result.model.fit_context
    assert context.sample_weight.index.equals(context.X.index)
    assert 1 not in context.X.index


def test_candidate_selection_and_original_position_audit():
    from xdiyo_analytics.selection import Candidate
    from xdiyo_analytics.selection.core import _fit_candidate, _FreshModels, _development, _original_positions
    data = sample()
    # Reordered, noncontiguous development subset catches local/global aliasing.
    original = np.array([7, 2, 5, 1, 8, 9])
    data.X.loc[2, 'signal'] = np.nan
    local, mapping = _development(data, original)
    from xdiyo_analytics.splits import Fold, SplitPlan
    local_plan = SplitPlan([Fold(np.array([0,1,2,3]), np.array([4,5]), np.array([4,5]))], 6, np.arange(6))
    candidate = Candidate('eligible', MeanAdapter, missing_fit_rows=MissingFitRows(['signal']))
    result = _original_positions(_fit_candidate(local, local_plan, candidate, _FreshModels()), mapping)
    fold = result.folds[0]
    assert fold.fit_positions.tolist() == [7,5,1]
    assert fold.test_positions.tolist() == [8,9]
    assert fold.training_summary['missing_fit_rows']['excluded_positions'] == [2]


def test_scalar_feature_selection_and_array_required_columns():
    data = sample(); data.X.loc[1, 'signal'] = np.nan
    policy = MissingFitRows(pd.Index(['signal']))
    runner = TrainingRunner(MeanAdapter, feature_columns='signal', missing_fit_rows=policy)
    scope = runner.selection_plan(data, plan(data))
    result = runner.run(data, plan(data))
    for expected, actual in zip(scope.folds, result.folds):
        np.testing.assert_array_equal(expected.train, actual.fit_positions)


def test_prefit_class_weight_report_uses_filtered_scope():
    from xdiyo_analytics.analysis import PreTrainingAnalysis
    from xdiyo_analytics.reporting import ClassWeightReporter
    class WeightedMean(MeanAdapter):
        supports_sample_weight = True
    data = sample(); data.X.loc[1, 'signal'] = np.nan
    data.y = pd.DataFrame({'class': np.arange(len(data.X)) % 2})
    runner = TrainingRunner(WeightedMean, feature_columns='signal', missing_fit_rows=MissingFitRows(['signal']))
    folds = plan(data)
    report = PreTrainingAnalysis({'weights': ClassWeightReporter()}).run(data, split_plan=runner.selection_plan(data, folds))
    result = runner.run(data, folds, analysis_report=report, weights_from='weights')
    for fold in result.folds:
        ctx = fold.model.fit_context
        assert ctx.sample_weight.index.equals(ctx.X.index)
        assert 1 not in ctx.X.index


def test_active_policy_parallel_folds_do_not_capture_parent_observer(tmp_path):
    from functools import partial
    from test_class_weighting import classification_data, ParentObserver, WeightedTrace
    from model_selection_samples import plan as selection_plan
    from xdiyo_analytics.training import ExecutionPolicy
    data = classification_data()
    data.y['target'] = (data.metadata.case % 3 == 0).astype(int)
    data.X.loc[1, 'signal'] = np.nan
    observer = ParentObserver()
    result = TrainingRunner(partial(WeightedTrace, folder=tmp_path, barrier=True, delay=.2),
        observer=observer, execution=ExecutionPolicy(2),
        missing_fit_rows=MissingFitRows(['signal'])).run(data, selection_plan(data))
    assert observer.calls
    for fold in result.folds:
        assert 1 not in fold.fit_positions
