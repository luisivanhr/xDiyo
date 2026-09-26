"""Prediction-time boundaries survive recipe assembly and explicit overrides."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.data.loading import SeasonData
from xdiyo_analytics.splits import Fold, SplitPlan
from xdiyo_analytics.ui.recipe import catalog_for_ui, default_recipe, node, prepare_recipe, run_recipe


@pytest.fixture
def daily_recipe(monkeypatch, tmp_path):
    dates = pd.date_range('2025-01-01', periods=6, tz='UTC')
    matches = pd.DataFrame({
        'event_id': range(1, 7), 'competition_id': 17, 'season_id': 100,
        'home_id': 11, 'away_id': 12,
        'kickoff_utc': [date.timestamp() for date in dates],
        'status': 'finished', 'round': range(1, 7),
        'home_score_current': 2, 'away_score_current': 1,
    })
    statistics = pd.DataFrame([
        {'event_id': event, 'period': 'ALL', 'group_name': 'Match overview',
         'key': 'cornerKicks', 'side': side, 'team_id': team, 'value': float(event)}
        for event in range(1, 7) for side, team in [('home', 11), ('away', 12)]
    ])
    data = SeasonData({'matches': matches, 'statistics': statistics}, {})
    monkeypatch.setattr('xdiyo_analytics.data.load_seasons', lambda **kwargs: data)
    recipe = default_recipe('unused-synthetic-input')
    recipe['output_dir'] = str(tmp_path / 'experiments')
    recipe['data']['tables'] = ['matches', 'statistics']
    recipe['cutoff_hours'] = 48
    recipe['split'] = node('splits.TemporalSplit', train_size=4, test_size=1, unit='kickoffs')
    return recipe, statistics


def set_layout(recipe, layout):
    recipe['assembly']['layout'] = layout
    if layout == 'team_match':
        recipe['labels']['corners']['component'] = 'labels.TeamValue'


@pytest.mark.parametrize('layout', ['match', 'team_match'])
@pytest.mark.parametrize('drop_target', [False, True])
def test_prediction_hours_bound_training_after_assembly(daily_recipe, layout, drop_target):
    recipe, statistics = daily_recipe
    set_layout(recipe, layout)
    if drop_target:
        # A missing target on just one side removes the whole match in either layout.
        statistics.loc[(statistics.event_id == 2) & (statistics.side == 'home'), 'value'] = np.nan
    original = deepcopy(recipe)
    prepared = prepare_recipe(recipe)
    metadata = prepared.dataset.metadata
    assert len(metadata) == (5 if drop_target else 6) * (2 if layout == 'team_match' else 1)
    assert (2 not in metadata.event_id.values) == drop_target
    first = prepared.split_plan.folds[0]
    expected_day = 4 if drop_target else 3
    assert first.metadata['fit_at'] == pd.Timestamp(f'2025-01-0{expected_day}', tz='UTC')
    assert set(metadata.iloc[first.train].event_id) == ({1, 3} if drop_target else {1, 2})
    for fold in prepared.split_plan.folds:
        cutoff = metadata.iloc[fold.test].kickoff_at.min() - pd.Timedelta(hours=48)
        assert fold.metadata['fit_at'] == cutoff
        assert fold.metadata['cutoffs'] == 'explicit'
        assert (metadata.iloc[fold.train].kickoff_at < cutoff).all()
        excluded = set(metadata.iloc[fold.metadata['excluded_train']].event_id)
        assert excluded == ({4, 5} if drop_target else
                            ({3, 4} if cutoff.day == 3 else {4, 5}))
    assert recipe == original


@pytest.mark.parametrize('layout', ['match', 'team_match'])
@pytest.mark.parametrize('explicit', [None, 'kickoff_at', 'earlier'])
def test_explicit_split_cutoffs_keep_precedence(daily_recipe, layout, explicit):
    recipe, _ = daily_recipe
    set_layout(recipe, layout)
    cutoffs = explicit
    if explicit == 'earlier':
        # These are dataset-aligned timestamps, independent of history row counts.
        cutoffs = [date.isoformat() for date in pd.date_range('2024-12-31', periods=6, tz='UTC')
                   for _ in range(2 if layout == 'team_match' else 1)]
    recipe['split_options']['cutoffs'] = cutoffs
    original = deepcopy(recipe)
    prepared = prepare_recipe(recipe)
    fold = prepared.split_plan.folds[0]
    expected_day = 4 if explicit == 'earlier' else 5
    assert fold.metadata['fit_at'] == pd.Timestamp(f'2025-01-0{expected_day}', tz='UTC')
    assert set(prepared.dataset.metadata.iloc[fold.train].event_id) == set(range(1, expected_day))
    assert fold.metadata['cutoffs'] == ('kickoff' if explicit is None else 'explicit')
    assert recipe == original


@pytest.mark.parametrize('layout', ['match', 'team_match'])
def test_prediction_cutoffs_retain_explicit_result_availability(daily_recipe, layout):
    recipe, _ = daily_recipe
    set_layout(recipe, layout)
    availability = [date + pd.Timedelta(hours=36 if date.day == 2 else 12)
                    for date in pd.date_range('2025-01-01', periods=6, tz='UTC')
                    for _ in range(2 if layout == 'team_match' else 1)]
    recipe['split_options']['available_at'] = [date.isoformat() for date in availability]
    prepared = prepare_recipe(recipe)
    first = prepared.split_plan.folds[0]
    assert set(prepared.dataset.metadata.iloc[first.train].event_id) == {1}
    for fold in prepared.split_plan.folds:
        assert fold.metadata['availability'] == 'explicit'
        assert (pd.DatetimeIndex(availability)[fold.train] < fold.metadata['fit_at']).all()


@pytest.mark.parametrize('nested', [False, True])
@pytest.mark.parametrize('layout', ['match', 'team_match'])
@pytest.mark.parametrize('override', [False, True])
def test_search_inherits_prediction_hours_on_development_rows(daily_recipe, monkeypatch, nested, layout, override):
    from xdiyo_analytics.selection.core import _development
    recipe, _ = daily_recipe
    set_layout(recipe, layout)
    recipe['cutoff_hours'] = 24
    recipe['split']['params']['train_size'] = 5
    recipe['search'] = dict(nested=nested, grid={'alpha': [0.1]},
                            split=node('splits.TemporalSplit', train_size=2, test_size=1, unit='kickoffs'))
    if override:
        recipe['search']['split_options'] = {'cutoffs': None}
    # Capture the assembled plans at the experiment boundary, without fitting.
    monkeypatch.setattr('xdiyo_analytics.experiments.football.FootballExperiment.run',
                        lambda self, prepared, **options: options)
    prepared = prepare_recipe(recipe)
    options = run_recipe(recipe, prepared=prepared)
    metadata = prepared.dataset.metadata
    outer = prepared.split_plan.folds[0]
    assert outer.metadata['fit_at'] == pd.Timestamp('2025-01-05', tz='UTC')
    assert set(metadata.iloc[outer.train].event_id) == {1, 2, 3, 4}
    if nested:
        subset, _ = _development(prepared.dataset, outer.train)
        plan = options['inner_plan_factory'](subset)
        metadata = subset.metadata
    else:
        plan = options['selection_plan']
    first = plan.folds[0]
    assert first.metadata['fit_at'] == pd.Timestamp('2025-01-03' if override else '2025-01-02', tz='UTC')
    assert set(metadata.iloc[first.train].event_id) == ({1, 2} if override else {1})
    for fold in plan.folds:
        cutoff = metadata.iloc[fold.test].kickoff_at.min() - pd.Timedelta(hours=0 if override else 24)
        assert fold.metadata['fit_at'] == cutoff
        assert (metadata.iloc[fold.train].kickoff_at < cutoff).all()


def test_unset_prediction_hours_keep_default_temporal_boundary(daily_recipe):
    recipe, _ = daily_recipe
    recipe['cutoff_hours'] = None
    fold = prepare_recipe(recipe).split_plan.folds[0]
    assert fold.metadata['fit_at'] == pd.Timestamp('2025-01-05', tz='UTC')
    assert fold.metadata['cutoffs'] == 'kickoff'


def test_prediction_hours_leave_non_temporal_splits_usable(daily_recipe):
    recipe, _ = daily_recipe
    recipe['split'] = node('splits.MatchKFold', n_splits=2)
    prepared = prepare_recipe(recipe)
    assert len(prepared.split_plan.folds) == 2
    assert all(fold.metadata['retrospective'] for fold in prepared.split_plan.folds)


def test_prebuilt_split_plan_is_preserved(daily_recipe):
    recipe, _ = daily_recipe
    plan = SplitPlan([Fold(np.array([0, 1]), np.array([4]), np.array([4]))], 6, np.arange(6))
    catalog = catalog_for_ui()
    catalog.register('test.Plan', lambda: plan, category='split')
    recipe['split'] = node('test.Plan')
    assert prepare_recipe(recipe, catalog=catalog).split_plan is plan
