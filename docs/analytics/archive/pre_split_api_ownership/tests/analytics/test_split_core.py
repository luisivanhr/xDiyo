import numpy as np
import pandas as pd
import pytest
from xdiyo_analytics.splits import Fold, GroupKFold, MatchKFold, create_split_plan
from split_samples import BASE, cases, dataset, rows_for, snapshot, unchanged, whole_matches

@pytest.mark.parametrize('layout', ['match', 'team_match'])
@pytest.mark.parametrize('shuffle', [False, True])
def test_match_kfold_original_positions_and_preservation(layout, shuffle):
    data = dataset(13, layout=layout, shuffle=shuffle)
    for frame in [data.X, data.y, data.metadata]:
        frame.index = pd.Index(np.arange(len(frame)) + 1000, name='nonpositional')
    before = snapshot(data)
    first_appearance = list(dict.fromkeys(data.metadata.case))
    expected = [set(first_appearance[:4]), set(first_appearance[4:7]), set(first_appearance[7:10]), set(first_appearance[10:])]
    folds = MatchKFold(4).folds(data)
    assert len(folds) == 4
    for fold, test in zip(folds, expected):
        whole_matches(data, fold)
        assert cases(data, fold.test) == test
        assert cases(data, fold.train) == set(range(13)) - test
        assert fold.metadata['retrospective']
        assert fold.score.tolist() == fold.test.tolist()
        assert data.metadata.iloc[fold.test].event_id.min() > 2**63
    unchanged(data, before)

def test_exact_partition_tuple_with_reused_event_id():
    data = dataset(layout='team_match', shuffle=True, specs=[{'competition': 10 + i // 3, 'event': BASE + i % 3, 'league': str(i // 3)} for i in range(12)])
    folds = MatchKFold(4).folds(data)
    assert sum(len(f.test) for f in folds) == 24
    for fold in folds:
        whole_matches(data, fold)
        assert len(set(data.groups.iloc[fold.test])) == 3

def test_match_shuffle_reproducible_and_seed_sensitive():
    data = dataset(20, layout='team_match')
    a, b, c = [MatchKFold(4, shuffle=True, random_state=seed).folds(data) for seed in [19, 19, 20]]
    assert [f.test.tolist() for f in a] == [f.test.tolist() for f in b]
    assert [f.test.tolist() for f in a] != [f.test.tolist() for f in c]
    assert sorted(row for fold in a for row in fold.test) == list(range(40))

@pytest.mark.parametrize('layout', ['match', 'team_match'])
def test_group_greedy_balance_by_matches_and_unique_sizes_ignore_shuffle(layout):
    specs = [{'season': 2000 + group} for group in range(6) for _ in range(group + 1)]
    data = dataset(layout=layout, specs=specs, shuffle=True)
    variants = [GroupKFold(3, shuffle=shuf, random_state=seed).folds(data) for shuf, seed in [(False, None), (True, 7), (True, 9)]]
    for folds in variants:
        for fold in folds:
            whole_matches(data, fold)
            train = set(data.metadata.iloc[fold.train].season_id)
            test = set(data.metadata.iloc[fold.test].season_id)
            assert not train & test
            assert len(cases(data, fold.test)) == 7
    assert [f.test.tolist() for f in variants[0]] == [f.test.tolist() for f in variants[1]] == [f.test.tolist() for f in variants[2]]

def test_group_equal_size_shuffle_and_external_tuple_groups():
    data = dataset(18, layout='team_match', shuffle=True)
    groups = pd.Series([('cohort', int(case // 3)) for case in data.metadata.case], index=data.metadata.index, dtype=object)
    before = snapshot(data)
    a, b, c = [GroupKFold(3, shuffle=True, random_state=seed).folds(data, groups=groups) for seed in [9, 9, 17]]
    assert [f.test.tolist() for f in a] == [f.test.tolist() for f in b]
    assert [f.test.tolist() for f in a] != [f.test.tolist() for f in c]
    for fold in a:
        whole_matches(data, fold)
        assert not set(groups.iloc[fold.train]) & set(groups.iloc[fold.test])
        assert len(cases(data, fold.test)) == 6
    unchanged(data, before)

@pytest.mark.parametrize('problem', ['pair_disagreement', 'missing', 'length', 'index', 'empty_columns', 'too_few_groups'])
def test_bad_group_contracts_raise(problem):
    data = dataset(8, layout='team_match')
    groups = pd.Series([int(i // 4) for i in range(16)], dtype=object)
    kwargs, splitter = {'groups': groups}, GroupKFold(2)
    if problem == 'pair_disagreement': groups.iloc[0] = 99
    elif problem == 'missing': groups.iloc[0] = None
    elif problem == 'length': kwargs['groups'] = groups.iloc[:-1].tolist()
    elif problem == 'index': groups.index = groups.index + 10
    elif problem == 'empty_columns': splitter, kwargs = GroupKFold(2, group_by=()), {}
    elif problem == 'too_few_groups': splitter, kwargs = GroupKFold(2), {}
    with pytest.raises(ValueError): splitter.folds(data, **kwargs)

def test_split_plan_stores_options_without_invoking_and_returns_array_copies():
    class MustNotRun:
        def __call__(self, *args, **kwargs): raise AssertionError('selection/refit invoked')
        def fit(self, *args, **kwargs): raise AssertionError('fit invoked')
    data = dataset(12, shuffle=True)
    splitter = MatchKFold(3)
    selector, refit = MustNotRun(), MustNotRun()
    plan = create_split_plan(data, splitter, model_selector=selector, refit_policy=refit)
    assert plan.model_selector is selector and plan.refit_policy is refit and plan.paths is None
    default = create_split_plan(data, splitter)
    assert default.model_selector is None and default.refit_policy is None
    assert plan.n_rows == 12 and plan.get_n_splits() == splitter.get_n_splits(data) == 3
    assert data.metadata.iloc[plan.row_order].case.tolist() == list(range(12))
    assert [(a.tolist(), b.tolist()) for a,b in plan.split(data.X)] == [(a.tolist(), b.tolist()) for a,b in splitter.split(data)]
    train, test = next(plan.split())
    train[0], test[0] = -99, -88
    assert min(plan.folds[0].train) >= 0 and min(plan.folds[0].test) >= 0
    with pytest.raises(ValueError): list(plan.split(data.X.iloc[:-1]))

def test_fold_default_metadata_independent_and_plan_without_times():
    first = Fold(np.array([0]), np.array([1]), np.array([1]))
    second = Fold(np.array([0]), np.array([1]), np.array([1]))
    first.metadata['a'] = 1
    assert second.metadata == {}
    data = dataset(6)
    data.metadata = data.metadata.drop(columns='kickoff_at')
    assert create_split_plan(data, MatchKFold(2)).row_order.tolist() == list(range(6))

@pytest.mark.parametrize('splitter', [MatchKFold(1), MatchKFold(True), MatchKFold(2.5), MatchKFold(99), GroupKFold(1)])
def test_invalid_fold_counts(splitter):
    with pytest.raises(ValueError): splitter.folds(dataset())

@pytest.mark.parametrize('problem', ['wrong_type', 'unaligned_length', 'unaligned_index', 'missing_identity', 'empty'])
def test_dataset_boundary_validation(problem):
    data = dataset(6)
    if problem == 'wrong_type': data = data.X
    elif problem == 'unaligned_length': data.y = data.y.iloc[:-1]
    elif problem == 'unaligned_index': data.X.index = data.X.index + 1
    elif problem == 'missing_identity': data.metadata.loc[0, 'event_id'] = pd.NA
    elif problem == 'empty': data.X, data.y, data.metadata = [f.iloc[:0] for f in [data.X, data.y, data.metadata]]
    with pytest.raises((TypeError, ValueError)): MatchKFold(2).folds(data)

@pytest.mark.parametrize('kind', ['match', 'group'])
def test_retrospective_kfold_membership_keeps_missing_targets(kind):
    data = dataset(12, specs=[{'season': 2000 + i // 3} for i in range(12)])
    data.y.iloc[:, :] = np.nan
    splitter = MatchKFold(3) if kind == 'match' else GroupKFold(3)
    assert sum(len(f.test) for f in splitter.folds(data)) == 12
