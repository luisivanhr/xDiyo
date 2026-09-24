"""Selection composition, leakage boundaries and hand-checked fold voting."""
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from reporting_samples import sample, plan, Custom
from split_samples import unchanged
from xdiyo_analytics.analysis import PreTrainingAnalysis
from xdiyo_analytics.reporting import CorrelationAnalysis, TopKCorrelationSelector, FeatureDistributionReporter, FeatureTimeline, StudyResult


@pytest.mark.parametrize('layout', ['match', 'team_match'])
@pytest.mark.parametrize('mode,partition', [('overall', 'all'), ('overall', 'train'), ('per_fold', 'train')])
def test_reused_and_fresh_selection_agree_without_mutation(layout, mode, partition):
    data = sample(layout=layout, shuffle=True)
    original = deepcopy(data)
    reporters = {'scores': CorrelationAnalysis(type=mode, partition=partition, methods=['pearson', 'spearman']),
                 'reuse': TopKCorrelationSelector(type=mode, partition=partition, k=3, source='scores'),
                 'fresh': TopKCorrelationSelector(type=mode, partition=partition, k=3)}
    report = PreTrainingAnalysis(reporters).run(data, split_plan=plan(data))
    assert list(report.selections) == ['reuse', 'fresh']
    for fold_id, selection in report.selections['reuse'].items():
        other = report.selections['fresh'][fold_id]
        assert selection.columns == other.columns
        pd.testing.assert_frame_equal(selection.ranking, other.ranking)
        assert len(selection.columns) == 3
        assert 'constant' not in selection.columns and 'missing' not in selection.columns
    unchanged(data, original)


def test_training_selection_applies_names_on_test_without_test_correlation(monkeypatch):
    data, calls = sample(), []
    real_run = CorrelationAnalysis.run
    def spy(self, ctx):
        calls.append((ctx.partition, ctx.fold_id, tuple(ctx.row_positions)))
        return real_run(self, ctx)
    monkeypatch.setattr(CorrelationAnalysis, 'run', spy)
    report = PreTrainingAnalysis({
        'select': TopKCorrelationSelector(type='per_fold', partition='train', k=1, features=['negative', 'linear'], targets='first'),
        'inspect': Custom('per_fold', 'test', features_from='select'),
        'unselected': Custom('per_fold', 'test'),
    }).run(data, split_plan=plan(data))
    assert len(calls) == 2 and all(partition == 'train' for partition, _, _ in calls)
    for run in report.studies[2:4]:
        assert run.result.tables['X'].columns.tolist() == ['negative']
        assert 'train' in run.result.notes[-1]
    assert report.studies[-1].result.tables['X'].columns.tolist() == data.X.columns.tolist()


def test_overall_selection_reuse_requires_explicit_features_from():
    data = sample()
    report = PreTrainingAnalysis({
        'pooled': TopKCorrelationSelector(type='overall', partition='all', k=1),
        'explicit': Custom('per_fold', 'test', features_from='pooled'),
        'plain': Custom('per_fold', 'test'),
    }).run(data, split_plan=plan(data))
    assert report.studies[1].result.tables['X'].shape[1] == 1
    assert report.studies[-1].result.tables['X'].shape[1] == data.X.shape[1]
    assert 'overall/consensus' in report.studies[1].result.notes[-1]


@pytest.mark.parametrize('mismatch', ['partition', 'fold_mode', 'targets', 'features', 'method', 'order'])
def test_reuse_requires_matching_scope_and_complete_requested_scores(mismatch):
    source_config = dict(type='per_fold', partition='train', methods='spearman')
    if mismatch == 'partition': source_config['partition'] = 'test'
    if mismatch == 'fold_mode': source_config['type'] = 'overall'
    if mismatch == 'targets': source_config['targets'] = 'first'
    if mismatch == 'features': source_config['features'] = 'linear'
    if mismatch == 'method': source_config['methods'] = 'pearson'
    reporters = {'source': CorrelationAnalysis(**source_config),
                 'chosen': TopKCorrelationSelector(type='per_fold', partition='train', k=2, source='source')}
    if mismatch == 'order': reporters = dict(reversed(list(reporters.items())))
    data = sample()
    with pytest.raises(ValueError):
        PreTrainingAnalysis(reporters).run(data, split_plan=plan(data))


@pytest.mark.parametrize('cls,kwargs', [(FeatureDistributionReporter, {}), (CorrelationAnalysis, {}),
                                      (FeatureTimeline, {'team_column': 'home_id'}), (TopKCorrelationSelector, {'k': 1})])
def test_zero_eligible_selection_is_explicit_and_downstream_studies_handle_it(cls, kwargs):
    data = sample()
    data.X = data.X[['constant', 'missing']]
    report = PreTrainingAnalysis({
        'empty': TopKCorrelationSelector(type='overall', partition='all', k=5),
        'consume': cls(type='overall', partition='all', features_from='empty', **kwargs),
    }).run(data)
    assert report.selections['empty'][None].columns == ()
    assert report.studies[-1].result.notes[0] == 'No input features selected.'
    if cls is TopKCorrelationSelector:
        assert report.selections['consume'][None].columns == ()


def test_input_order_ties_and_fewer_than_k_eligible():
    data = sample()
    selection = PreTrainingAnalysis({'s': TopKCorrelationSelector(type='overall', partition='all', k=20,
                                       features=['negative', 'linear', 'constant', 'missing'], targets='first')}).run(data)
    assert selection.selections['s'][None].columns == ('negative', 'linear')
    assert 'only 2' in ' '.join(selection.studies[0].result.notes)


def known_scores(ctx):
    # c has the largest mean score, but never wins a fold's top-1 nomination.
    mapping = {0: {'a': .9, 'b': .1, 'c': .8, 'missing': np.nan},
               1: {'a': .1, 'b': .9, 'c': .8, 'missing': np.nan}}
    rows = [dict(feature=feature, target='first', metric='spearman', value=value,
                 n=len(ctx.X), status='ok' if np.isfinite(value) else 'zero_spread')
            for feature, value in mapping[ctx.fold_id].items()]
    return StudyResult('Known coefficients', tables={'coefficients': pd.DataFrame(rows)})


@pytest.mark.parametrize('fold_ids,expected,votes', [(None, ('b', 'a'), {'a': 1, 'b': 1, 'c': 0, 'missing': 0}),
                                                   ([0], ('a',), {'a': 1, 'b': 0, 'c': 0, 'missing': 0}),
                                                   ([1], ('b',), {'a': 0, 'b': 1, 'c': 0, 'missing': 0})])
def test_consensus_nomination_votes_mean_ties_subset_and_stored_rankings(fold_ids, expected, votes):
    data = sample()
    data.X = pd.DataFrame({name: np.arange(len(data.X)) for name in ['b', 'a', 'c', 'missing']})
    report = PreTrainingAnalysis({
        'source': Custom('per_fold', 'train', known_scores),
        'consensus': TopKCorrelationSelector(type='overall', partition='train', k=3, fold_k=1,
                                              across_folds=True, source='source', targets='first'),
    }).run(data, split_plan=plan(data), fold_ids=fold_ids)
    result = report.studies[-1].result
    assert result.selection.columns == expected
    ranking = result.selection.ranking.set_index('feature')
    assert ranking.votes.to_dict() == votes
    assert pd.isna(ranking.loc['missing', 'mean_magnitude'])
    assert ranking.loc['c', 'mean_magnitude'] == pytest.approx(.8)
    selected_folds = [0, 1] if fold_ids is None else fold_ids
    assert set(result.tables['fold_rankings'].fold_id) == set(selected_folds)
    assert set(result.tables['coefficients'].fold_id) == set(selected_folds)
    for fold, group in result.tables['fold_rankings'].groupby('fold_id'):
        assert group.loc[group.selected, 'feature'].tolist() == (['a'] if fold == 0 else ['b'])
    np.testing.assert_allclose(ranking.vote_fraction, ranking.votes / len(selected_folds))


def test_consensus_fresh_and_reused_equal_and_distinct_from_pooled_rows():
    data = sample()
    report = PreTrainingAnalysis({
        'corr': CorrelationAnalysis(type='per_fold', partition='train', methods='spearman'),
        'reuse': TopKCorrelationSelector(type='overall', partition='train', k=2, across_folds=True, source='corr'),
        'fresh': TopKCorrelationSelector(type='overall', partition='train', k=2, across_folds=True),
        'pooled': TopKCorrelationSelector(type='overall', partition='train', k=2),
    }).run(data, split_plan=plan(data))
    pd.testing.assert_frame_equal(report.selections['reuse'][None].ranking, report.selections['fresh'][None].ranking)
    assert 'votes' in report.selections['fresh'][None].ranking
    assert 'votes' not in report.selections['pooled'][None].ranking
    assert len(report.studies[-1].row_positions) == 7  # union once, not 4+7 observations


@pytest.mark.parametrize('kwargs', [{'k': 0}, {'k': True}, {'k': 1.2}, {'method': 'invalid'},
                                  {'fold_k': 2}, {'across_folds': True, 'fold_k': 0}])
def test_selector_invalid_configuration(kwargs):
    options = {'type': 'overall', 'partition': 'train', 'k': 2, **kwargs}
    data = sample()
    with pytest.raises(ValueError):
        PreTrainingAnalysis({'s': TopKCorrelationSelector(**options)}).run(data, split_plan=plan(data))


def test_consensus_requires_overall_and_selected_folds():
    for mode, kwargs in [('per_fold', {'split_plan': plan(sample())}), ('overall', {})]:
        with pytest.raises(ValueError, match='overall|selected folds'):
            PreTrainingAnalysis({'s': TopKCorrelationSelector(type=mode, partition='all', k=2,
                                                               across_folds=True)}).run(sample(), **kwargs)
