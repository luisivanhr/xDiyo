"""Exact team identities, explicit league aggregation and honest display gaps."""
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from reporting_samples import sample, plan, BASE
from split_samples import unchanged
from xdiyo_analytics.analysis import PreTrainingAnalysis
from xdiyo_analytics.reporting import FeatureTimeline


@pytest.mark.parametrize('layout,feature,identity', [('team_match', 'linear', 'team_id'),
                                                   ('match', 'home::linear', 'home_id'),
                                                   ('match', 'away::linear', 'away_id'),
                                                   ('match', 'linear', 'home_id')])
@pytest.mark.parametrize('mode', ['overall', 'timeline', 'per_fold'])
def test_team_timelines_preserve_ids_and_cross_season_continuity(layout, feature, identity, mode):
    data = sample(layout=layout, shuffle=True)
    data.X = data.X.rename(columns={'linear': feature})
    original = deepcopy(data)
    kwargs = {'team_column': identity} if feature == 'linear' and layout == 'match' else {}
    report = PreTrainingAnalysis({'t': FeatureTimeline(type=mode, partition='all', features=feature, **kwargs)}).run(data, split_plan=plan(data))
    for run in report.studies:
        result = run.result
        points = result.tables['points']
        assert set(points.team_id) == set(data.metadata.iloc[run.row_positions][identity])
        assert all(int(value) > 2**63 for value in points.team_id)
        assert points.kickoff_at.is_monotonic_increasing
        for row in points.itertuples():
            assert row.value == data.X.iloc[row.row_position][feature]
            assert row.team_id == data.metadata.iloc[row.row_position][identity]
        expected_entities = data.metadata.iloc[run.row_positions][['competition_id', identity]].drop_duplicates()
        assert len(result.artifacts[0].data.data) == len(expected_entities)
    unchanged(data, original)


def test_team_filter_and_missing_empty_selection():
    data = sample(layout='team_match')
    data.X.loc[4, 'linear'] = np.nan
    report = PreTrainingAnalysis({'t': FeatureTimeline(type='timeline', partition='all', features='linear', teams=[BASE+500])}).run(data)
    result = report.studies[0].result
    assert len(result.tables['points']) == 12
    assert result.tables['points'].value.isna().sum() == 1
    trace = result.artifacts[0].data.data[0]
    assert trace.connectgaps is False
    empty = PreTrainingAnalysis({'t': FeatureTimeline(type='timeline', partition='all', features='linear', teams=[-1])}).run(data)
    assert empty.studies[0].result.tables['points'].empty
    assert 'No observations' in empty.studies[0].result.artifacts[0].data.layout.annotations[0].text


@pytest.mark.parametrize('aggregation', [None, 'mean', 'median'])
def test_league_shared_values_collapse_and_explicit_row_aggregation(aggregation):
    data = sample(layout='team_match')
    # Four observation rows share each kickoff; all rows carry unit weight.
    if aggregation is None:
        data.X['league'] = (data.metadata.case // 2).astype(float)
    else:
        data.X['league'] = np.arange(len(data.X), dtype=float) ** 2
    data.X.loc[data.metadata.case >= 10, 'league'] = np.nan
    result = PreTrainingAnalysis({'t': FeatureTimeline(type='overall', partition='all', entity='league',
                                     features='league', league_aggregation=aggregation)}).run(data).studies[0].result
    points = result.tables['points']
    assert len(points) == 6 and points.value.isna().sum() == 1
    for row in points.itertuples():
        selected = data.X.loc[data.metadata.kickoff_at == row.kickoff_at, 'league'].dropna().tolist()
        if selected:
            expected = selected[0] if aggregation is None else (sum(selected)/len(selected) if aggregation == 'mean' else np.median(selected))
            assert row.value == pytest.approx(expected)
    assert 'row_position' not in points  # a displayed league point can combine rows


def test_league_conflicting_loo_values_are_not_silently_combined():
    with pytest.raises(ValueError, match='differ'):
        PreTrainingAnalysis({'t': FeatureTimeline(type='overall', partition='all', entity='league', features='linear')}).run(sample())


def test_sampled_display_retains_missing_break_and_full_points():
    data = sample(layout='team_match')
    data.X.loc[data.metadata.case.isin([3, 4, 8]), 'linear'] = np.nan
    result = PreTrainingAnalysis({'t': FeatureTimeline(type='timeline', partition='all', features='linear',
                                                      teams=[BASE+500], max_points=2)}).run(data).studies[0].result
    points, shown = result.tables['points'], result.tables['plotted_points']
    assert len(points) == 12
    assert len(shown) == 3 and shown.value.isna().sum() == 1
    assert shown.row_position.iloc[0] == points.row_position.iloc[0]
    assert shown.row_position.iloc[-1] == points.row_position.iloc[-1]
    trace = result.artifacts[0].data.data[0]
    assert trace.connectgaps is False and np.isnan(trace.y[1])


@pytest.mark.parametrize('fault', ['match_unlabelled', 'bad_entity', 'bad_aggregation', 'league_teams',
                                   'bad_max', 'bool_max', 'missing_time', 'missing_competition', 'missing_team'])
def test_timeline_configuration_and_identity_errors(fault):
    data = sample()
    kwargs = {'type': 'overall', 'partition': 'all', 'features': 'linear', 'team_column': 'home_id'}
    if fault == 'match_unlabelled': kwargs.pop('team_column')
    elif fault == 'bad_entity': kwargs['entity'] = 'season'
    elif fault == 'bad_aggregation': kwargs['league_aggregation'] = 'sum'
    elif fault == 'league_teams': kwargs.update(entity='league', teams=[1])
    elif fault == 'bad_max': kwargs['max_points'] = 1
    elif fault == 'bool_max': kwargs['max_points'] = True
    elif fault == 'missing_time': data.metadata.loc[0, 'kickoff_at'] = pd.NaT
    elif fault == 'missing_competition': data.metadata.loc[0, 'competition_id'] = np.nan
    else: data.metadata.loc[0, 'home_id'] = pd.NA
    with pytest.raises(ValueError):
        PreTrainingAnalysis({'t': FeatureTimeline(**kwargs)}).run(data)
