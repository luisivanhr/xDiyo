"""Scope selection, exact identities, custom reporters and isolation."""
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from reporting_samples import sample, plan, Custom
from split_samples import unchanged
from xdiyo_analytics.analysis import PreTrainingAnalysis
from xdiyo_analytics.reporting import StudyResult, FeatureSelection, FeatureDistributionReporter, CorrelationAnalysis, FeatureTimeline, TopKCorrelationSelector


@pytest.mark.parametrize('layout', ['match', 'team_match'])
@pytest.mark.parametrize('mode', ['per_fold', 'overall', 'timeline'])
@pytest.mark.parametrize('partition', ['train', 'test', 'score', 'all'])
def test_scope_positions_counts_and_chronology(layout, mode, partition):
    data = sample(layout=layout, shuffle=True)
    for frame in (data.X, data.y, data.metadata):
        frame.index = pd.Index(101 + 7 * np.arange(len(frame)), name='external_index')
    before = deepcopy(data)
    splits = plan(data)
    report = PreTrainingAnalysis({'inspect': Custom(mode, partition)}).run(data, split_plan=splits, fold_ids=[1, 0])
    ids = [1, 0] if mode == 'per_fold' else [None]
    assert [run.fold_id for run in report.studies] == ids
    for run in report.studies:
        if mode != 'per_fold' and partition == 'all':
            expected = list(range(len(data.X)))
        else:
            selected = [run.fold_id] if mode == 'per_fold' else [1, 0]
            positions = set()
            for i in selected:
                fold = splits.folds[i]
                positions.update(list(fold.train) + list(fold.test) if partition == 'all' else getattr(fold, partition))
            expected = sorted(positions)
        if mode == 'timeline':
            expected.sort(key=lambda i: data.metadata.iloc[i].kickoff_at)
        assert run.row_positions.tolist() == expected
        assert run.n_matches == len(set(data.metadata.iloc[expected].case))
        assert run.layout == layout
        for key, frame in [('X', data.X), ('y', data.y), ('metadata', data.metadata)]:
            reference = frame.iloc[expected].copy()
            reference.index = pd.Index(expected, dtype='int64', name='row_position')
            pd.testing.assert_frame_equal(run.result.tables[key], reference)
    unchanged(data, before)


def test_empty_report_has_no_implicit_studies_and_no_plot_dependency():
    report = PreTrainingAnalysis({}).run(sample())
    assert report.studies == [] and report.selections == {}
    html = report.to_html()
    assert 'No reporters selected' in html and 'plotly.js v' not in html


@pytest.mark.parametrize('cls,kwargs', [(FeatureDistributionReporter, {}), (CorrelationAnalysis, {}), (FeatureTimeline, {}), (TopKCorrelationSelector, {'k': 2})])
def test_type_and_partition_are_required(cls, kwargs):
    for missing in ('type', 'partition'):
        config = {'type': 'overall', 'partition': 'all', **kwargs}
        config.pop(missing)
        with pytest.raises(TypeError):
            cls(**config)


@pytest.mark.parametrize('fold_ids', [[0, 0], [True], [-1], [2], [0.5]])
def test_invalid_fold_selection(fold_ids):
    data = sample()
    with pytest.raises(ValueError, match='fold_ids'):
        PreTrainingAnalysis({}).run(data, split_plan=plan(data), fold_ids=fold_ids)


@pytest.mark.parametrize('mode,partition', [('per_fold', 'all'), ('overall', 'train'), ('timeline', 'test')])
def test_scopes_requiring_folds_reject_absent_or_empty_plan_selection(mode, partition):
    data = sample()
    for kwargs in ({}, {'split_plan': plan(data), 'fold_ids': []}):
        with pytest.raises(ValueError, match='selected fold'):
            PreTrainingAnalysis({'r': Custom(mode, partition)}).run(data, **kwargs)


def test_context_isolation_including_previous_results_and_definitions():
    data = sample()
    before, splits = deepcopy(data), plan(data)
    old_splits = deepcopy(splits)
    seen = []
    def mutate(ctx):
        seen.append(ctx.X.copy())
        ctx.X.iloc[:, :] = -500
        ctx.y.iloc[:, :] = -700
        ctx.metadata['event_id'] = 0
        ctx.definitions['features']['linear']['description'] = 'mutated'
        ctx.fold_metadata['nested'][0] = -1
        ctx.fold_rows[0][:] = 999
        return StudyResult('First', tables={'value': pd.DataFrame({'n': [3]})})
    def mutate_previous(ctx):
        ctx.previous_results['first'].tables['value'].iloc[0, 0] = 99
        return StudyResult('Second')
    result = PreTrainingAnalysis({'first': Custom('per_fold', 'train', mutate),
                                  'second': Custom('per_fold', 'train', mutate_previous),
                                  'last': Custom('per_fold', 'train')}).run(data, split_plan=splits)
    unchanged(data, before)
    assert [run.result.tables['value'].iloc[0, 0] for run in result.studies[:2]] == [3, 3]
    for old, new in zip(old_splits.folds, splits.folds):
        np.testing.assert_array_equal(old.train, new.train)
        assert old.metadata == new.metadata
    assert (result.studies[-1].result.tables['X'].linear >= 0).all()


def test_reporter_errors_retain_scope_and_bad_results_fail():
    def fail(ctx):
        raise ArithmeticError('independent reporter failure')
    with pytest.raises(ArithmeticError) as caught:
        PreTrainingAnalysis({'custom': Custom('overall', 'all', fail)}).run(sample())
    assert "Reporter 'custom', type=overall, partition=all, fold=None" in caught.value.__notes__
    with pytest.raises(TypeError, match='StudyResult'):
        PreTrainingAnalysis({'custom': Custom('overall', 'all', lambda ctx: {})}).run(sample())


@pytest.mark.parametrize('fault', ['alignment', 'plan_size', 'bad_position', 'duplicate_selection', 'unknown_selection', 'timeline_missing'])
def test_scope_and_selection_contract_errors(fault):
    data, reporter = sample(), Custom('per_fold', 'train')
    splits = plan(data)
    if fault == 'alignment': data.y.index = data.y.index + 1
    elif fault == 'plan_size': splits.n_rows += 1
    elif fault == 'bad_position': splits.folds[0].train = np.array([len(data.X)])
    elif fault in ('duplicate_selection', 'unknown_selection'):
        columns = ('linear', 'linear') if fault == 'duplicate_selection' else ('absent',)
        reporter.action = lambda ctx: StudyResult('bad', selection=FeatureSelection(columns, pd.DataFrame()))
    else:
        data.metadata.loc[0, 'kickoff_at'] = pd.NaT
        reporter = Custom('timeline', 'all')
    with pytest.raises(ValueError):
        PreTrainingAnalysis({'r': reporter}).run(data, split_plan=splits)


def test_unsupported_mode_partition_and_dependency_fail_explicitly():
    for reporter in [Custom('unknown', 'all'), Custom('overall', 'unknown'),
                     CorrelationAnalysis(type='timeline', partition='all'),
                     Custom('overall', 'all', features_from='later')]:
        with pytest.raises(ValueError):
            PreTrainingAnalysis({'r': reporter}).run(sample())
