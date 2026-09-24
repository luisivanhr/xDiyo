"""Reporter checkboxes consume assembled column names from prepare and run jobs."""

from copy import deepcopy

import pytest

from test_ui_workflow import ui_recipe
from xdiyo_analytics.ui.recipe import ModelFactory, catalog_for_ui, node
from xdiyo_analytics.ui.server import BuilderState


def completed_job(state, action, recipe):
    handle = state.start(action, recipe)
    state.jobs[handle['id']]['future'].result(timeout=20)
    public = state.dispatch('job', handle)
    assert public['status'] == 'complete', public['message']
    return public, state.jobs[handle['id']]['object']


@pytest.mark.parametrize('layout', ['match', 'team_match'])
def test_prepare_discovers_real_columns_without_fitting_and_reporter_can_select_them(ui_recipe, tmp_path, monkeypatch, layout):
    ui_recipe['assembly']['layout'] = layout
    if layout == 'team_match':
        ui_recipe['labels']['corners']['component'] = 'labels.TeamValue'
    state = BuilderState(tmp_path, catalog_for_ui())
    try:
        with monkeypatch.context() as patch:
            patch.setattr(ModelFactory, '__call__', lambda *a, **kw: pytest.fail('Feature discovery fitted a model'))
            preview, prepared = completed_job(state, 'prepare', ui_recipe)
        columns = preview['preview']['features']['columns']
        assert columns == prepared.dataset.X.columns.tolist()
        assert preview['preview']['labels']['columns'] == prepared.dataset.y.columns.tolist()
        assert preview['preview']['folds'][0]['train'] == len(prepared.split_plan.folds[0].train)
        assert any('corners_mean' in column for column in columns)
        selected = columns[:1]  # Use returned names, with no assumptions about layout suffixes.
        configured = deepcopy(ui_recipe)
        configured['pre_reporters']['distribution'] = node('reporting.FeatureDistributionReporter',
                                                          type='overall', partition='train', features=selected)
        result_preview, result = completed_job(state, 'run', configured)
        assert result_preview['preview']['features']['columns'] == columns
        study = next(study for study in result.pre_report.studies if study.name == 'distribution')
        assert set(study.result.tables['summary'].feature) == set(selected)
        assert configured['pre_reporters']['distribution']['params']['features'] == selected
    finally:
        state.executor.shutdown(wait=True, cancel_futures=True)


def test_run_without_prior_prepare_and_recovered_run_both_publish_feature_choices(ui_recipe, tmp_path):
    state = BuilderState(tmp_path, catalog_for_ui())
    try:
        first, result = completed_job(state, 'run', ui_recipe)
        assert first['preview']['features']['columns'] == result.prepared.dataset.X.columns.tolist()
        assert first['preview']['features']['count'] == 23
        second, recovered = completed_job(state, 'run', ui_recipe)
        assert second['reused'] is True
        assert second['preview'] == first['preview']
        assert recovered.prepared.dataset.X.columns.tolist() == first['preview']['features']['columns']
    finally:
        state.executor.shutdown(wait=True, cancel_futures=True)


def test_unfiltered_preparation_lists_full_plan_even_when_holdout_search_is_configured(ui_recipe, tmp_path, monkeypatch):
    ui_recipe['split']['params']['train_size'] = 1
    ui_recipe['fold_ids'] = None  # Browser discovery deliberately clears only the submitted selection.
    ui_recipe['search'] = {'grid': {'alpha': [0.1, 1.]}, 'split': node('splits.MatchKFold', n_splits=2),
                           'options': {'metrics': ['mse']}, 'nested': False}
    original = deepcopy(ui_recipe)
    monkeypatch.setattr(ModelFactory, '__call__', lambda *a, **kw: pytest.fail('Fold discovery fitted a model'))
    state = BuilderState(tmp_path, catalog_for_ui())
    try:
        status, prepared = completed_job(state, 'prepare', ui_recipe)
        assert len(prepared.split_plan.folds) == 2
        assert [{key: value for key, value in item.items() if key != 'description'} for item in status['preview']['folds']] == [
            {'fold': i, 'train': len(f.train), 'test': len(f.test), 'score': len(f.score)}
            for i, f in enumerate(prepared.split_plan.folds)
        ]
        assert [f['fold'] for f in status['preview']['folds']] == [0, 1]
        assert status['preview']['folds'][0]['description'] == 'Alpha · train 22/23 · test 23/24'
        assert status['preview']['folds'][1]['description'] == 'Alpha · train 22/23, 23/24 · test 24/25'
        assert ui_recipe == original
    finally:
        state.executor.shutdown(wait=True, cancel_futures=True)


def test_round_gap_preview_names_retained_test_rounds(ui_recipe, tmp_path):
    ui_recipe['split']['params'].update(gap=2, gap_unit='rounds')
    state = BuilderState(tmp_path, catalog_for_ui())
    try:
        status, prepared = completed_job(state, 'prepare', ui_recipe)
        preview = status['preview']['folds'][0]
        assert preview['description'] == 'Alpha · train 22/23, 23/24 · test 24/25 · test rounds 3–7'
        assert preview['train'] == 16 and preview['test'] == 5
        assert set(prepared.dataset.metadata.iloc[prepared.split_plan.folds[0].test]['round']) == {3, 4, 5, 6, 7}
    finally:
        state.executor.shutdown(wait=True, cancel_futures=True)


def test_selected_outer_fold_is_reported_using_its_local_id(ui_recipe, tmp_path):
    ui_recipe['split']['params']['train_size'] = 1
    ui_recipe['fold_ids'] = [1]
    ui_recipe['analysis_options']['post'] = {'fold_ids': [0]}
    state = BuilderState(tmp_path, catalog_for_ui())
    try:
        discovery = deepcopy(ui_recipe)
        discovery['fold_ids'] = None
        status, _ = completed_job(state, 'prepare', discovery)
        assert [f['fold'] for f in status['preview']['folds']] == [0, 1]
        status, result = completed_job(state, 'run', ui_recipe)
        assert [f['fold'] for f in status['preview']['folds']] == [0]
        assert [f.fold_id for f in result.training.folds] == [0]
        assert set(result.training.folds[0].metadata.source_season) == {'24_25'}
        assert all(set(study.scope.fold_id) == {0} for study in result.post_report.studies)
        assert ui_recipe['fold_ids'] == [1]
        assert ui_recipe['analysis_options']['post']['fold_ids'] == [0]
    finally:
        state.executor.shutdown(wait=True, cancel_futures=True)
