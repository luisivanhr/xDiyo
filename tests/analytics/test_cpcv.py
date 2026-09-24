from copy import deepcopy
from itertools import combinations
from math import comb
import numpy as np
import pandas as pd
import pytest
from xdiyo_analytics.splits import CPCV, MatchKFold, create_split_plan, reconstruct_paths
from split_samples import cases, dataset, rows_for, snapshot, unchanged, whole_matches

@pytest.mark.parametrize('layout', ['match', 'team_match'])
@pytest.mark.parametrize('n,k', [(4,1), (5,2), (6,2), (6,3)])
def test_all_combinations_and_each_incidence_once(layout, n, k):
    data = dataset(24, layout=layout, shuffle=True)
    before = snapshot(data)
    splitter = CPCV(n,k)
    plan = create_split_plan(data, splitter)
    assert len(plan.folds) == comb(n,k)
    assert splitter.n_paths == comb(n-1,k-1)
    assert [f.metadata['test_blocks'] for f in plan.folds] == list(combinations(range(n),k))
    counts = np.zeros(len(data.X), dtype=int)
    for fold in plan.folds:
        whole_matches(data, fold)
        counts[fold.test] += 1
        assert fold.metadata['retrospective'] and fold.metadata['availability'] == 'kickoff_proxy'
        assert len(fold.metadata['purged_train']) == len(fold.metadata['embargoed_train']) == 0
        assert set(fold.train) | set(fold.test) == set(range(len(data.X)))
    assert (counts == splitter.n_paths).all()
    assert len(plan.paths) == n*splitter.n_paths
    assert not plan.paths.duplicated(['fold_id','block_id']).any()
    for path in range(splitter.n_paths):
        rows = plan.paths.loc[plan.paths.path_id.eq(path)]
        assert sorted(rows.block_id) == list(range(n))
        assert sorted(int(i) for positions in rows.rows for i in positions) == list(range(len(data.X)))
    unchanged(data, before)

def test_batches_keep_tied_fixtures_and_have_observed_time_order():
    data = dataset(layout='team_match', shuffle=True, specs=[{'day': i//2} for i in range(18)])
    splitter = CPCV(6,2)
    plan = create_split_plan(data, splitter)
    expected_blocks = [{0,1,2,3}, {4,5,6,7}, {8,9,10,11}, {12,13}, {14,15}, {16,17}]
    for fold in plan.folds:
        whole_matches(data, fold)
        for block, rows in fold.metadata['block_rows'].items():
            assert cases(data, rows) == expected_blocks[block]
        for day in range(9):
            tied = rows_for(data,{2*day,2*day+1})
            assert set(tied) <= set(fold.test) or set(tied) <= set(fold.train)
    assert plan.row_order.tolist() == sorted(range(len(data.X)), key=lambda row: data.metadata.kickoff_at.iloc[row])

@pytest.mark.parametrize('layout', ['match', 'team_match'])
def test_interval_purging_and_run_embargo_against_pairwise_oracle(layout):
    data = dataset(24, layout=layout, shuffle=True)
    starts = data.metadata.kickoff_at - pd.to_timedelta(data.metadata.case % 3, unit='D')
    ends = data.metadata.kickoff_at + pd.to_timedelta(data.metadata.case % 2, unit='D')
    duration = pd.Timedelta('1D')
    plan = create_split_plan(data, CPCV(6,2,embargo=duration), information_start=starts, available_at=ends)
    match_info = {}
    for case in range(24):
        positions = rows_for(data,{case})
        match_info[case] = (data.metadata.kickoff_at.iloc[positions[0]], min(starts.iloc[positions]), max(ends.iloc[positions]))
    for fold in plan.folds:
        selected = cases(data, fold.test)
        candidate = set(range(24)) - selected
        overlaps = {i for i in candidate if any(match_info[i][1] <= match_info[j][2] and match_info[j][1] <= match_info[i][2] for j in selected)}
        runs = []
        for block in fold.metadata['test_blocks']:
            if runs and block == runs[-1][-1]+1: runs[-1].append(block)
            else: runs.append([block])
        boundaries = [max(match_info[i][2] for block in run for i in cases(data,fold.metadata['block_rows'][block])) for run in runs]
        embargoed = {i for i in candidate if any(end < match_info[i][0] <= end+duration for end in boundaries)}
        assert cases(data, fold.train) == candidate - overlaps - embargoed
        assert cases(data, fold.metadata['purged_train']) == overlaps
        assert cases(data, fold.metadata['embargoed_train']) == embargoed - overlaps
        whole_matches(data, fold)

def test_closed_overlap_touches_both_endpoints_and_enclosing_interval():
    data = dataset(10)
    starts, ends = data.metadata.kickoff_at.copy(), data.metadata.kickoff_at.copy()
    ends.iloc[2] = pd.Timestamp('2024-01-09',tz='UTC')
    ends.iloc[3] = starts.iloc[4]
    ends.iloc[5] = starts.iloc[6]
    folds = CPCV(5,1).folds(data, information_start=starts, available_at=ends)
    fold = next(f for f in folds if f.metadata['test_blocks'] == (2,))
    assert cases(data, fold.test) == {4,5}
    assert cases(data, fold.metadata['purged_train']) == {2,3,6}
    assert cases(data, fold.train) == {0,1,7,8,9}

def test_disjoint_test_intervals_do_not_purge_the_enclosing_gap():
    data = dataset(12)
    fold = next(f for f in CPCV(6,2).folds(data) if f.metadata['test_blocks'] == (0,4))
    assert cases(data, fold.test) == {0,1,8,9}
    assert cases(data, fold.train) == {2,3,4,5,6,7,10,11}

@pytest.mark.parametrize('selected,removed', [((1,2),{6}), ((1,4),{4,10})])
def test_embargo_applies_after_each_contiguous_run_including_upper_boundary(selected, removed):
    data = dataset(12, layout='team_match')
    fold = next(f for f in CPCV(6,2,embargo='1D').folds(data) if f.metadata['test_blocks'] == selected)
    assert cases(data, fold.metadata['embargoed_train']) == removed
    assert cases(data, fold.train) == set(range(12)) - cases(data, fold.test) - removed
    assert not len(fold.metadata['purged_train'])

def test_per_team_information_intervals_use_conservative_min_and_max():
    data = dataset(10, layout='team_match')
    starts, ends = data.metadata.kickoff_at.copy(), data.metadata.kickoff_at.copy()
    starts.iloc[rows_for(data,{4})[1]] = pd.Timestamp('2024-01-04', tz='UTC')
    ends.iloc[rows_for(data,{5})[0]] = pd.Timestamp('2024-01-07', tz='UTC')
    fold = next(f for f in CPCV(5,1).folds(data, information_start=starts, available_at=ends) if f.metadata['test_blocks'] == (2,))
    assert cases(data, fold.metadata['purged_train']) == {3,6}

@pytest.mark.parametrize('layout', ['match', 'team_match'])
@pytest.mark.parametrize('mapping', [False, True])
def test_path_reconstruction_exact_multioutput_values_nans_and_chronology(layout, mapping):
    data = dataset(18, layout=layout, shuffle=True, specs=[{'day': i//2} for i in range(18)])
    plan = create_split_plan(data, CPCV(6,2))
    predictions = [pd.DataFrame({'mean': f.test.astype(float)+fold_id/100, 'probability': [np.nan if row%3==0 else .7 for row in f.test]}, index=f.test).iloc[::-1] for fold_id,f in enumerate(plan.folds)]
    before = [frame.copy(deep=True) for frame in predictions]
    supplied = dict(enumerate(predictions)) if mapping else predictions
    result = reconstruct_paths(plan,supplied)
    assert list(result) == ['path_id','row_position','fold_id','mean','probability']
    assert result.shape == (5*len(data.X),5)
    chronological = sorted(range(len(data.X)), key=lambda row:data.metadata.kickoff_at.iloc[row])
    for path in range(5):
        rows = result.loc[result.path_id.eq(path)]
        assert rows.row_position.tolist() == chronological
        assert rows.row_position.is_unique
        assert rows.probability.isna().sum() == len(range(0,len(data.X),3))
        for item in rows.itertuples(index=False):
            assert item.mean == predictions[item.fold_id].loc[item.row_position,'mean']
            assert item.row_position in plan.folds[item.fold_id].test
    for frame, original in zip(predictions,before): pd.testing.assert_frame_equal(frame, original)

def test_nullable_numeric_predictions_are_preserved():
    data = dataset(12)
    plan = create_split_plan(data,CPCV())
    values = [pd.DataFrame({'prediction': pd.Series([pd.NA]*len(f.test),index=f.test,dtype='Float64')}) for f in plan.folds]
    result = reconstruct_paths(plan,values)
    assert str(result.prediction.dtype) == 'Float64' and result.prediction.isna().all()

@pytest.mark.parametrize('problem', ['missing_fold','extra_fold','wrong_positions','missing_position','extra_position','duplicate_position','float_index','non_frame','nonnumeric','reordered_columns','different_columns','reserved_column','empty_columns','duplicate_columns'])
def test_prediction_contract_errors(problem):
    plan = create_split_plan(dataset(12),CPCV())
    predictions = [pd.DataFrame({'a': np.ones(len(f.test)), 'b': np.zeros(len(f.test))},index=f.test) for f in plan.folds]
    if problem == 'missing_fold': predictions.pop()
    elif problem == 'extra_fold': predictions.append(predictions[-1])
    elif problem == 'wrong_positions': predictions[0].index = plan.folds[0].train[:len(predictions[0])]
    elif problem == 'missing_position': predictions[0] = predictions[0].iloc[:-1]
    elif problem == 'extra_position': predictions[0].loc[99] = [1.,0.]
    elif problem == 'duplicate_position': predictions[0].index = [0]*len(predictions[0])
    elif problem == 'float_index': predictions[0].index = predictions[0].index.astype(float)
    elif problem == 'non_frame': predictions[0] = predictions[0].to_numpy()
    elif problem == 'nonnumeric': predictions[0] = predictions[0].astype(str)
    elif problem == 'reordered_columns': predictions[0] = predictions[0][['b','a']]
    elif problem == 'different_columns': predictions[0] = predictions[0].rename(columns={'a':'z'})
    elif problem == 'reserved_column': predictions[0] = predictions[0].rename(columns={'a':'row_position'})
    elif problem == 'empty_columns': predictions[0] = predictions[0].iloc[:,:0]
    elif problem == 'duplicate_columns': predictions[0].columns = ['a','a']
    with pytest.raises(ValueError): reconstruct_paths(plan,predictions)

def test_reconstruction_requires_cpcv_and_complete_incidence_collection():
    data = dataset(12)
    with pytest.raises(ValueError): reconstruct_paths(create_split_plan(data,MatchKFold()),[])
    splitter = CPCV()
    folds = splitter.folds(data)
    with pytest.raises(ValueError): splitter.path_map(folds[:-1])
    mapping = splitter.path_map(folds)
    mapping.iloc[0].rows[0] = -1
    assert min(folds[0].metadata['block_rows'][0]) >= 0

@pytest.mark.parametrize('settings', [dict(n_blocks=1),dict(n_blocks=True),dict(n_blocks=99),dict(n_test_blocks=0),dict(n_test_blocks=6),dict(n_test_blocks=1.2),dict(embargo='-1D'),dict(embargo=pd.NaT)])
def test_invalid_cpcv_settings(settings):
    with pytest.raises(ValueError): CPCV(**settings).folds(dataset(12))

@pytest.mark.parametrize('problem', ['missing_start','missing_end','start_after_end','end_before_kickoff','numeric_start','unaligned_end','empty_after_purge','empty_after_embargo'])
def test_information_interval_errors(problem):
    data = dataset(12)
    starts, ends = data.metadata.kickoff_at.copy(), data.metadata.kickoff_at.copy()
    splitter = CPCV()
    if problem == 'missing_start': starts.iloc[0] = pd.NaT
    elif problem == 'missing_end': ends.iloc[0] = pd.NaT
    elif problem == 'start_after_end': starts.iloc[0] += pd.Timedelta('1D')
    elif problem == 'end_before_kickoff': starts -= pd.Timedelta('2D'); ends -= pd.Timedelta('1D')
    elif problem == 'numeric_start': starts = list(range(12))
    elif problem == 'unaligned_end': ends.index += 1
    elif problem == 'empty_after_purge': starts[:] = starts.min(); ends[:] = ends.max()
    elif problem == 'empty_after_embargo': splitter = CPCV(6,2,embargo='99D')
    with pytest.raises(ValueError): splitter.folds(data,information_start=starts,available_at=ends)

def test_cpcv_does_not_filter_unknown_targets():
    data = dataset(12)
    data.y.iloc[:,:] = np.nan
    plan = create_split_plan(data,CPCV())
    assert len(plan.folds) == 15
    assert all(len(f.train)==8 and len(f.test)==4 for f in plan.folds)
