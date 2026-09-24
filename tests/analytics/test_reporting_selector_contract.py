"""A non-correlation selector exercises the common three-mode contract."""
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import pandas as pd
import pytest
from reporting_samples import sample, plan
from xdiyo_analytics.analysis import PreTrainingAnalysis
from xdiyo_analytics.reporting import FeatureSelector, FeatureSelection, StudyResult, vote_selections


@dataclass(kw_only=True)
class MeanSelector(FeatureSelector):
    def select(self, ctx):
        ranking = ctx.X.mean().sort_values(ascending=False, kind='stable').rename('mean').rename_axis('feature').reset_index()
        chosen = tuple(ranking.feature.head(1))
        return StudyResult('Mean selection', selection=FeatureSelection(chosen, ranking))

    def combine(self, ctx, fold_results):
        chosen = vote_selections(fold_results, feature_order=ctx.X.columns, k=None)
        return StudyResult('Mean consensus', selection=chosen)


def test_noncorrelation_selector_local_pooled_and_consensus_are_common_contracts():
    data = sample()
    data.X = pd.DataFrame({'a': [5.]*4 + [0.]*8, 'b': [0.]*4 + [10.]*8})
    original = deepcopy(data.X)
    report = PreTrainingAnalysis({
        'local': MeanSelector(type='per_fold', partition='train'),
        'pooled': MeanSelector(type='overall', partition='train'),
        'consensus': MeanSelector(type='overall', partition='train', across_folds=True),
    }).run(data, split_plan=plan(data))
    assert report.selections['local'][0].columns == ('a',)
    assert report.selections['local'][1].columns == ('b',)
    assert report.selections['pooled'][None].columns == ('b',)  # seven unique rows
    assert report.studies[2].row_positions.tolist() == list(range(7))
    assert report.selections['consensus'][None].columns == ('a', 'b')
    assert report.selections['consensus'][None].ranking.votes.tolist() == [1, 1]
    assert 'mean_score' not in report.selections['consensus'][None].ranking
    pd.testing.assert_frame_equal(data.X, original)


def selected(names, scores=None):
    frame = pd.DataFrame() if scores is None else pd.DataFrame({'feature': list(scores), 'score': list(scores.values())})
    return StudyResult('Nominations', selection=FeatureSelection(tuple(names), frame))


def test_generic_vote_counts_ties_finite_scores_and_no_score_policy():
    folds = {7: selected(['b'], {'a': 4., 'b': np.inf, 'c': 2.}),
             2: selected(['a'], {'a': 2., 'b': 8., 'c': np.nan})}
    no_score = vote_selections(folds, feature_order=['b', 'a', 'c'], k=None)
    assert no_score.columns == ('b', 'a')
    scored = vote_selections(folds, feature_order=['a', 'b', 'c'], k=1, score_column='score')
    assert scored.columns == ('b',)
    board = scored.ranking.set_index('feature')
    assert board.loc['a', 'mean_score'] == 3 and board.loc['b', 'mean_score'] == 8
    assert board.loc['b', 'scored_folds'] == 1 and board.loc['c', 'votes'] == 0
    assert board.loc['a', 'vote_fraction'] == .5
    # Missing scores never erase valid nominations under the generic policy.
    undefined = vote_selections({0: selected(['a'], {'a': np.nan})}, feature_order=['a'], k=1, score_column='score')
    assert undefined.columns == ('a',)


@pytest.mark.parametrize('bad', ['empty', 'k_zero', 'duplicate_order', 'unknown_nomination', 'duplicate_nomination', 'missing_score', 'duplicate_score'])
def test_generic_vote_contract_errors(bad):
    folds, order, k, score = {0: selected(['a'])}, ['a', 'b'], 1, None
    if bad == 'empty': folds = {}
    elif bad == 'k_zero': k = 0
    elif bad == 'duplicate_order': order = ['a', 'a']
    elif bad == 'unknown_nomination': folds = {0: selected(['z'])}
    elif bad == 'duplicate_nomination': folds = {0: selected(['a', 'a'])}
    elif bad == 'missing_score': score = 'score'
    else:
        folds = {0: selected(['a'], {'a': 1})}
        folds[0].selection.ranking = pd.DataFrame({'feature': ['a', 'a'], 'score': [1, 2]})
        score = 'score'
    with pytest.raises(ValueError):
        vote_selections(folds, feature_order=order, k=k, score_column=score)


@pytest.mark.parametrize('invalid', ['no_result', 'no_selection', 'unknown', 'duplicate'])
def test_base_validates_each_local_selection(invalid):
    @dataclass(kw_only=True)
    class Bad(FeatureSelector):
        def select(self, ctx):
            if invalid == 'no_result': return None
            if invalid == 'no_selection': return StudyResult('missing')
            return selected(['unknown'] if invalid == 'unknown' else ['linear', 'linear'])
    with pytest.raises((ValueError, TypeError)):
        PreTrainingAnalysis({'bad': Bad(type='overall', partition='all')}).run(sample())


def test_consensus_children_have_same_fold_metadata_as_per_fold_execution():
    seen = []
    @dataclass(kw_only=True)
    class MetadataSelector(MeanSelector):
        def select(self, ctx):
            seen.append((self.across_folds, ctx.fold_id, deepcopy(ctx.fold_metadata)))
            return super().select(ctx)
    data = sample()
    PreTrainingAnalysis({'local': MetadataSelector(type='per_fold', partition='train'),
                         'consensus': MetadataSelector(type='overall', partition='train', across_folds=True)}).run(data, split_plan=plan(data))
    assert [(fold, metadata) for _, fold, metadata in seen[:2]] == [(fold, metadata) for _, fold, metadata in seen[2:]]


def test_consensus_children_isolate_mutable_context_definitions():
    @dataclass(kw_only=True)
    class MutatingSelector(MeanSelector):
        def select(self, ctx):
            assert ctx.definitions['features']['linear']['description'] == 'synthetic'
            ctx.definitions['features']['linear']['description'] = 'changed'
            return super().select(ctx)
    data = sample()
    PreTrainingAnalysis({'consensus': MutatingSelector(type='overall', partition='train', across_folds=True)}).run(data, split_plan=plan(data))
    assert data.definitions['features']['linear']['description'] == 'synthetic'


def test_consensus_nested_metadata_mutation_cannot_change_outer_or_split_plan():
    seen = []
    @dataclass(kw_only=True)
    class NestedSelector(MeanSelector):
        def select(self, ctx):
            seen.append(deepcopy(ctx.fold_metadata))
            ctx.fold_metadata['nested'][0] = 999
            ctx.definitions['features']['linear']['description'] = 'local mutation'
            assert ctx.fold_metadata_by_id == {}
            return super().select(ctx)

        def combine(self, ctx, results):
            assert ctx.fold_metadata == {}
            assert ctx.fold_metadata_by_id == {0: {'nested': [0]}, 1: {'nested': [1]}}
            assert ctx.definitions['features']['linear']['description'] == 'synthetic'
            return super().combine(ctx, results)

    data = sample()
    splits = plan(data)
    PreTrainingAnalysis({'s': NestedSelector(type='overall', partition='train', across_folds=True)}).run(data, split_plan=splits)
    assert seen == [{'nested': [0]}, {'nested': [1]}]
    assert [fold.metadata for fold in splits.folds] == seen
