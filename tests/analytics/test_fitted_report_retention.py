"""Evaluated fitted studies remain visible, scoped and recoverable."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from model_selection_samples import sample, candidate
from split_samples import rows_for
from xdiyo_analytics.analysis import PreTrainingAnalysis
from xdiyo_analytics.experiments import FootballExperiment, PreparedExperiment
from xdiyo_analytics.experiments.recovery import pack, unpack
from xdiyo_analytics.reporting import CorrelationAnalysis, TopKCorrelationSelector
from xdiyo_analytics.selection import ModelSelection
from xdiyo_analytics.splits import Fold, SplitPlan, MatchKFold, create_split_plan
from xdiyo_analytics.training import ValidationTail


def setup_case(mode, *, consume=True, study_type='per_fold'):
    data = sample(shuffle=True)
    folds = [Fold(rows_for(data, range(8)), rows_for(data, [8, 9]), rows_for(data, [8, 9])),
             Fold(rows_for(data, range(12)), rows_for(data, range(12, 16)), rows_for(data, range(12, 16)))]
    outer = SplitPlan(folds[-1:] if mode == 'holdout' else folds, len(data.X), np.arange(len(data.X)))
    reporters = PreTrainingAnalysis({
        'Correlation': CorrelationAnalysis(type=study_type, partition='train', methods=['spearman']),
        'Selector': TopKCorrelationSelector(type=study_type, partition='train', source='Correlation', k=1),
    })
    model = candidate('Signal', pre_analysis=reporters, features_from='Selector' if consume else None,
                      validation=ValidationTail(.25))
    options = {'model': model}
    if mode != 'fixed':
        options = {'model_selection': ModelSelection([model, replace(model, name='Other')], metrics='mse')}
        if mode == 'nested':
            options['inner_plan_factory'] = lambda development: create_split_plan(development, MatchKFold(2))
        else:
            from xdiyo_analytics.selection.core import _development
            development, positions = _development(data, folds[-1].train)
            inner = create_split_plan(development, MatchKFold(2))
            options['selection_plan'] = SplitPlan(
                [Fold(positions[f.train], positions[f.test], positions[f.score]) for f in inner.folds],
                len(data.X), positions[inner.row_order])
    return PreparedExperiment(data, outer), options


@pytest.mark.parametrize('mode', ['fixed', 'holdout', 'nested'])
def test_evaluated_correlation_and_selector_visible_without_inner_trial_studies(tmp_path, mode):
    prepared, options = setup_case(mode)
    experiment = FootballExperiment('Retained fitted studies', output_dir=tmp_path)
    result = experiment.run(prepared, **options)
    report = result.training.fitted_report
    assert len(report.studies) == 2 * len(result.training.folds)
    displayed = [s for s in result.report.studies if s.name.startswith('fitted/')]
    assert len(displayed) == len(report.studies)
    assert {s.name for s in displayed} == {'fitted/Correlation', 'fitted/Selector'}
    for fit in result.training.folds:
        studies = [s for s in report.studies if s.fold_id == fit.fold_id]
        assert {s.name for s in studies} == {'Correlation', 'Selector'}
        assert len(fit.validation_positions) > 0
        for study in studies:
            assert set(study.row_positions) == set(fit.fit_positions)
            assert not set(study.row_positions) & set(np.r_[fit.test_positions, fit.validation_positions])
            assert study.type == 'per_fold'
            if study.scope is not None and 'row_position' in study.scope:
                assert set(study.scope.row_position) == set(fit.fit_positions)
        assert set(fit.selection.row_positions) == set(fit.fit_positions)
        retained = next(s for s in studies if s.name == 'Selector')
        assert retained is not fit.selection
        assert not np.shares_memory(retained.row_positions, fit.selection.row_positions)
    searches = [] if mode == 'fixed' else (list(result.selection.selections.values()) if mode == 'nested' else [result.selection])
    for search in searches:
        for trial in search.trials:
            for fit in trial.training.folds:
                studies = [s for s in trial.training.fitted_report.studies if s.fold_id == fit.fold_id]
                assert len(studies) == 2
                assert set(fit.selection.row_positions) == set(fit.fit_positions)
                for study in studies:
                    assert set(study.row_positions) == set(fit.fit_positions)
                correlation = next(s for s in studies if s.name == 'Correlation')
                coefficient = correlation.result.tables['coefficients'].query("feature == 'signal'").iloc[0]
                expected = prepared.dataset.X.iloc[fit.fit_positions]['signal'].corr(
                    prepared.dataset.y.iloc[fit.fit_positions]['target'], method='spearman')
                assert coefficient.value == pytest.approx(expected)
    recovered = experiment.run(prepared, **options)
    assert recovered.reused
    assert [s.name for s in recovered.report.studies] == [s.name for s in result.report.studies]
    for actual, expected in zip(recovered.training.fitted_report.studies, report.studies):
        np.testing.assert_array_equal(actual.row_positions, expected.row_positions)
        for name, table in expected.result.tables.items():
            pd.testing.assert_frame_equal(actual.result.tables[name], table)


def test_descriptive_fitted_analysis_is_retained_without_consuming_selector(tmp_path):
    prepared, options = setup_case('fixed', consume=False)
    result = FootballExperiment('Descriptive fitted', output_dir=tmp_path).run(prepared, **options)
    assert len(result.training.fitted_report.studies) == 4
    assert all(f.selection is None and len(f.feature_columns) == 3 for f in result.training.folds)
    assert 'fitted/Correlation' in result.to_html()


def test_nested_overall_studies_become_distinct_outer_fold_views(tmp_path):
    prepared, options = setup_case('nested', consume=False, study_type='overall')
    result = FootballExperiment('Overall within outer folds', output_dir=tmp_path).run(prepared, **options)
    for study in result.training.fitted_report.studies:
        assert study.type == 'per_fold' and study.fold_id in {0, 1}
        assert study.scope_label == f'overall within outer fold {study.fold_id} fitting rows'
        assert set(study.row_positions) == set(result.training.folds[study.fold_id].fit_positions)


def test_old_training_bundle_without_fitted_report_loads_with_none(tmp_path):
    prepared, options = setup_case('fixed')
    result = FootballExperiment('Legacy field', output_dir=tmp_path).run(prepared, **options)
    encoded = pack(result.training)
    assert 'fitted_report' in encoded['fields']
    legacy = deepcopy(encoded)
    del legacy['fields']['fitted_report']
    loaded = unpack(legacy)
    assert loaded.fitted_report is None
    pd.testing.assert_frame_equal(loaded.prediction_frame(), result.training.prediction_frame())
