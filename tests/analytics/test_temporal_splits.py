import numpy as np
import pandas as pd
import pytest
from xdiyo_analytics.splits import TemporalSplit, create_split_plan
from split_samples import cases, dataset, rows_for, snapshot, unchanged, whole_matches

@pytest.mark.parametrize('layout', ['match', 'team_match'])
@pytest.mark.parametrize('shuffle', [False, True])
@pytest.mark.parametrize('window', ['expanding', 'sliding'])
def test_multiseason_round_spans_and_step(layout, shuffle, window):
    data = dataset(layout=layout, shuffle=shuffle, specs=[{'season': 2020+i//4, 'round': i%4+1} for i in range(12)])
    before = snapshot(data)
    folds = TemporalSplit(3, test_size=3, step=2, window=window).folds(data)
    assert len(folds) == 4
    for fold, end in zip(folds, [3, 5, 7, 9]):
        whole_matches(data, fold)
        assert cases(data, fold.train) == set(range(0 if window == 'expanding' else end-3, end))
        assert cases(data, fold.test) == set(range(end, end+3))
        assert np.array_equal(fold.score, fold.test)
        assert fold.metadata['retrospective'] is False
        assert fold.metadata['availability'] == 'kickoff_proxy'
    unchanged(data, before)

@pytest.mark.parametrize('layout', ['match', 'team_match'])
def test_block_and_duration_gaps_have_different_membership(layout):
    data = dataset(12, layout=layout)
    folds = TemporalSplit(4, test_size=2, step=2, gap=1, gap_time='2D').folds(data)
    assert len(folds) == 3
    for fold, end in zip(folds, [4, 6, 8]):
        assert cases(data, fold.test) == {end+1, end+2}
        assert cases(data, fold.train) == set(range(end-1))
        assert cases(data, fold.metadata['excluded_train']) == {end-1}
        assert end not in cases(data, np.r_[fold.train, fold.test])
        assert fold.metadata['training_boundary'] == fold.metadata['fit_at'] - pd.Timedelta('2D')
        whole_matches(data, fold)

def test_postponed_match_in_an_earlier_round_is_not_training():
    data = dataset(layout='team_match', shuffle=True, specs=[{'round': r, 'day': day} for r,day in [(1,0),(1,12),(2,1),(2,2),(3,3),(3,4),(4,5),(4,6)]])
    folds = TemporalSplit(1).folds(data)
    assert cases(data, folds[0].train) == {0}
    assert cases(data, folds[0].test) == {2,3}
    assert cases(data, folds[0].metadata['excluded_train']) == {1}
    assert cases(data, folds[1].train) == {0,2,3}

def test_same_kickoff_candidate_is_excluded_and_blocks_order_by_observation():
    data = dataset(specs=[{'round': r, 'day': day} for r,day in [(50,0),(20,1),(10,1),(3,2)]])
    fold = TemporalSplit(2).folds(data)[0]
    assert cases(data, fold.train) == {0}
    assert cases(data, fold.test) == {2}
    assert cases(data, fold.metadata['excluded_train']) == {1}
    assert fold.metadata['test_blocks'] == ((2024,10),)

@pytest.mark.parametrize('form', ['series', 'column', 'list'])
@pytest.mark.parametrize('layout', ['match', 'team_match'])
def test_two_day_prediction_lead_prunes_training(form, layout):
    data = dataset(10, layout=layout, shuffle=True)
    data.metadata['prediction_time'] = data.metadata.kickoff_at - pd.Timedelta('2D')
    cutoffs = data.metadata.prediction_time
    if form == 'column': cutoffs = 'prediction_time'
    elif form == 'list': cutoffs = cutoffs.tolist()
    folds = TemporalSplit(4, test_size=2).folds(data, cutoffs=cutoffs)
    for fold in folds:
        first_test = min(cases(data, fold.test))
        assert cases(data, fold.train) == set(range(first_test-2))
        assert fold.metadata['cutoffs'] == 'explicit'
        whole_matches(data, fold)

@pytest.mark.parametrize('layout', ['match', 'team_match'])
def test_explicit_availability_and_missing_targets_remove_whole_training_match(layout):
    data = dataset(10, layout=layout, shuffle=True)
    release = data.metadata.kickoff_at.copy()
    release.iloc[rows_for(data,{2})[0]] = pd.NaT
    release.iloc[rows_for(data,{3})] = pd.Timestamp('2024-01-08', tz='UTC')
    release.iloc[rows_for(data,{4})] = pd.Timestamp('2024-01-07', tz='UTC')
    data.y.iloc[rows_for(data,{1})[0], 0] = np.nan
    data.y.iloc[rows_for(data,{6})[0], 0] = np.nan
    before = snapshot(data)
    fold = TemporalSplit(6, test_size=2).folds(data, available_at=release)[0]
    assert cases(data, fold.train) == {0,4,5}
    assert cases(data, fold.metadata['excluded_train']) == {1,2,3}
    assert cases(data, fold.test) == {6,7} and cases(data, fold.score) == {6,7}
    assert fold.metadata['availability'] == 'explicit'
    unchanged(data, before)

@pytest.mark.parametrize('representation', ['utc', 'naive', 'none'])
def test_all_missing_explicit_availability_reports_empty_training_not_timezone_error(representation):
    data = dataset(8, layout='team_match')
    values = pd.Series(pd.NaT, index=data.metadata.index, dtype='datetime64[ns, UTC]' if representation == 'utc' else 'datetime64[ns]')
    if representation == 'none': values = [None]*len(data.X)
    with pytest.raises(ValueError, match='no training'):
        TemporalSplit(3).folds(data, available_at=values)

def test_match_rows_use_earliest_prediction_and_latest_availability():
    data = dataset(8, layout='team_match')
    cutoff = data.metadata.kickoff_at.copy()
    cutoff.iloc[rows_for(data,{4})[1]] -= pd.Timedelta('1D')
    release = data.metadata.kickoff_at.copy()
    release.iloc[rows_for(data,{1})[1]] += pd.Timedelta('3D')
    fold = TemporalSplit(4, test_size=2).folds(data, cutoffs=cutoff, available_at=release)[0]
    assert fold.metadata['fit_at'] == pd.Timestamp('2024-01-04', tz='UTC')
    assert cases(data, fold.train) == {0,2}

@pytest.mark.parametrize('bounds,expected', [((2,3),{9,10}), ((None,2),{8,9}), ((3,None),{10,11})])
def test_score_only_later_season_and_round_range_remains_held_out(bounds, expected):
    data = dataset(layout='team_match', shuffle=True, specs=[{'season': 2020+i//4, 'round': i%4+1} for i in range(12)])
    fold = TemporalSplit(1, test_size=2, unit='seasons', score_start=1, score_rounds=bounds).folds(data)[0]
    assert cases(data, fold.train) == set(range(4))
    assert cases(data, fold.test) == set(range(4,12))
    assert cases(data, fold.score) == expected
    whole_matches(data, fold)

def test_score_rounds_applies_in_each_held_out_season():
    data = dataset(specs=[{'season': 2020+i//4, 'round': i%4+1} for i in range(12)])
    fold = TemporalSplit(1, test_size=2, unit='seasons', score_rounds=(2,3)).folds(data)[0]
    assert cases(data, fold.score) == {5,6,9,10}

def test_partial_final_horizon_may_have_no_scoring_rows():
    data = dataset(7)
    complete = TemporalSplit(2, test_size=3, score_start=2).folds(data)
    partial = TemporalSplit(2, test_size=3, score_start=2, allow_partial_test=True).folds(data)
    assert len(complete) == 1 and len(partial) == 2
    assert cases(data, partial[0].score) == {4}
    assert cases(data, partial[1].test) == {5,6} and partial[1].score.size == 0
    assert cases(data, partial[1].train) == set(range(5))

@pytest.mark.parametrize('layout', ['match', 'team_match'])
def test_separate_league_calendars_and_pooled_kickoff_batches(layout):
    data = dataset(layout=layout, shuffle=True, specs=[{'competition': 17+c, 'round': r+1, 'day': r} for c in range(2) for r in range(5)])
    separate = TemporalSplit(2, test_size=2).folds(data)
    pooled = TemporalSplit(2, test_size=2, unit='kickoffs').folds(data)
    explicit = TemporalSplit(2, test_size=2, unit='kickoffs', calendar_by='source_league').folds(data)
    synchronized = TemporalSplit(2, test_size=2, calendar_by=(), block_by=('source_season','round')).folds(data)
    assert len(separate) == len(explicit) == 2 and len(pooled) == len(synchronized) == 1
    for fold in separate:
        assert len(set(data.metadata.iloc[np.r_[fold.train,fold.test]].competition_id)) == 1
        assert len(cases(data, fold.train)) == len(cases(data, fold.test)) == 2
    assert cases(data, pooled[0].train) == {0,1,5,6}
    assert cases(data, pooled[0].test) == {2,3,7,8}
    assert cases(data, synchronized[0].test) == cases(data, pooled[0].test)
    for fold in [*separate,*pooled,*explicit,*synchronized]: whole_matches(data, fold)

def test_stage_keys_keep_repeated_round_numbers_separate():
    data = dataset(specs=[{'round': i%2+1, 'stage': 'group' if i<2 else 'playoff'} for i in range(4)])
    folds = TemporalSplit(1, block_by=('season_id','stage','round')).folds(data)
    assert [cases(data,f.test) for f in folds] == [{1},{2},{3}]

@pytest.mark.parametrize('settings', [dict(train_size=0), dict(train_size=True), dict(test_size=0), dict(step=0), dict(step=1.5), dict(gap=-1), dict(gap_time='-1D'), dict(gap_time=pd.NaT), dict(window='other'), dict(unit='weeks'), dict(score_start=-1), dict(score_start=1), dict(score_rounds=(4,2)), dict(score_rounds=(1,2,3)), dict(block_by=()), dict(train_size=99)])
def test_invalid_temporal_settings(settings):
    options = dict(train_size=2)
    options.update(settings)
    with pytest.raises(ValueError): TemporalSplit(**options).folds(dataset(8))

@pytest.mark.parametrize('problem', ['missing_kickoff', 'numeric_kickoff', 'inconsistent_kickoff', 'inconsistent_group', 'missing_group', 'late_cutoff', 'missing_cutoff', 'unaligned_cutoff', 'short_cutoff', 'numeric_cutoff', 'short_availability', 'all_missing_y'])
def test_timing_and_metadata_errors(problem):
    data = dataset(8, layout='team_match')
    options = {}
    if problem == 'missing_kickoff': data.metadata.loc[0,'kickoff_at'] = pd.NaT
    elif problem == 'numeric_kickoff': data.metadata['kickoff_at'] = np.arange(len(data.X))
    elif problem == 'inconsistent_kickoff': data.metadata.loc[0,'kickoff_at'] += pd.Timedelta('1h')
    elif problem == 'inconsistent_group': data.metadata.loc[0,'round'] = 999
    elif problem == 'missing_group': data.metadata['round'] = pd.Series([None]*len(data.X))
    elif problem == 'late_cutoff': options['cutoffs'] = data.metadata.kickoff_at + pd.Timedelta('1h')
    elif problem == 'missing_cutoff':
        options['cutoffs'] = data.metadata.kickoff_at.copy(); options['cutoffs'].iloc[0] = pd.NaT
    elif problem == 'unaligned_cutoff': options['cutoffs'] = data.metadata.kickoff_at.rename(index=lambda x:x+1)
    elif problem == 'short_cutoff': options['cutoffs'] = data.metadata.kickoff_at.tolist()[:-1]
    elif problem == 'numeric_cutoff': options['cutoffs'] = list(range(len(data.X)))
    elif problem == 'short_availability': options['available_at'] = [None]
    elif problem == 'all_missing_y': data.y.iloc[:,:] = np.nan
    with pytest.raises(ValueError): TemporalSplit(3).folds(data, **options)
