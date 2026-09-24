"""Count/proportion semantics on actual selector input populations."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from reporting_samples import Custom, plan, sample
from test_reporting_selection import known_scores
from test_reporting_selector_contract import selected
from xdiyo_analytics.analysis import PreTrainingAnalysis
from xdiyo_analytics.reporting import TopKCorrelationSelector, resolve_selection_count, vote_selections


@pytest.mark.parametrize('k,n,expected', [(.5, 5, 3), (.07, 100, 7), (.14, 50, 7),
                                         (.999, 5, 5), (.001, 5, 1), (.5, 0, 0),
                                         (1, 5, 1), (1., 5, 1), (2., 5, 2), (20, 5, 20)])
def test_counts_and_decimal_fraction_boundaries(k, n, expected):
    assert resolve_selection_count(k, n) == expected


@pytest.mark.parametrize('k', [0, -1, True, np.bool_(False), 1.01, np.nan, np.inf, '0.5', None])
def test_invalid_count_or_fraction(k):
    with pytest.raises(ValueError, match='positive integer|proportion'):
        resolve_selection_count(k, 5)


@pytest.mark.parametrize('mode', ['overall', 'per_fold'])
def test_fraction_denominator_includes_missing_inputs_and_respects_explicit_subset(mode):
    data = sample()
    before = deepcopy(data.X)
    report = PreTrainingAnalysis({
        'all': TopKCorrelationSelector(type=mode, partition='train', k=.5),
        'restricted': TopKCorrelationSelector(type=mode, partition='train', k=.5,
                                              features=['negative', 'linear', 'constant', 'missing'], targets='first'),
        'large': TopKCorrelationSelector(type=mode, partition='train', k=.99),
    }).run(data, split_plan=plan(data))
    # All five inputs count toward .5 -> 3, even though only three have defined scores.
    assert all(len(s.columns) == 3 for s in report.selections['all'].values())
    assert all(s.columns == ('negative', 'linear') for s in report.selections['restricted'].values())
    assert all(len(s.columns) == 3 for s in report.selections['large'].values())
    large = [s for s in report.studies if s.name == 'large']
    assert all('only 3' in ' '.join(s.result.notes) for s in large)
    pd.testing.assert_frame_equal(data.X, before)


def test_chained_fraction_uses_reduced_input_count_and_empty_selection_stays_empty():
    data = sample()
    report = PreTrainingAnalysis({
        'first': TopKCorrelationSelector(type='overall', partition='all', k=.5),
        'second': TopKCorrelationSelector(type='overall', partition='all', k=.5, features_from='first'),
        'missing': TopKCorrelationSelector(type='overall', partition='all', k=.5, features=['constant', 'missing']),
        'empty': TopKCorrelationSelector(type='overall', partition='all', k=.5, features_from='missing'),
    }).run(data)
    assert len(report.selections['first'][None].columns) == 3
    assert len(report.selections['second'][None].columns) == 2
    assert set(report.selections['second'][None].columns) <= set(report.selections['first'][None].columns)
    assert report.selections['missing'][None].columns == report.selections['empty'][None].columns == ()


def test_consensus_resolves_final_and_fold_fraction_separately():
    data = sample()
    data.X = pd.DataFrame({name: np.arange(len(data.X)) for name in ['b', 'a', 'c', 'missing']})
    report = PreTrainingAnalysis({
        'source': Custom('per_fold', 'train', known_scores),
        'consensus': TopKCorrelationSelector(type='overall', partition='train', k=.5, fold_k=.25,
                                              across_folds=True, source='source', targets='first'),
    }).run(data, split_plan=plan(data))
    result = report.studies[-1].result
    assert result.selection.columns == ('b', 'a')
    assert result.tables['fold_rankings'].groupby('fold_id').selected.sum().to_dict() == {0: 1, 1: 1}
    assert result.selection.ranking.set_index('feature').votes.to_dict() == {'b': 1, 'a': 1, 'c': 0, 'missing': 0}


def test_generic_voting_fraction_counts_input_order_but_never_adds_unnominated_features():
    votes = {0: selected(['a']), 1: selected(['b'])}
    # .5 of five inputs asks for 3; only the two nominees may survive.
    result = vote_selections(votes, feature_order=['b', 'a', 'c', 'd', 'e'], k=.5)
    assert result.columns == ('b', 'a')
    assert vote_selections(votes, feature_order=['b', 'a', 'c'], k=1.).columns == ('b',)
