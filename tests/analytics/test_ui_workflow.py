"""Synthetic end-to-end coverage of recipes; no production export is fitted."""

from copy import deepcopy
import hashlib
import inspect
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from xdiyo_analytics.ui import catalog_for_ui, default_recipe, prepare_recipe, run_recipe, predict_recipe
from xdiyo_analytics.ui.recipe import export_python, export_notebook, node, read_recipe
from xdiyo_analytics.ui.schema import stage_schema


@pytest.fixture
def ui_recipe(tmp_path):
    root = tmp_path / 'data'
    for year in (2022, 2023, 2024):
        season = year * 100 + (year + 1) % 100
        stem = f'Alpha_{year % 100:02}_{(year + 1) % 100:02}'
        version = root / f'_tables/competition=17/season={season}/versions/v1'
        version.mkdir(parents=True)
        ids = [2**63 + year * 100 + i for i in range(8)]
        home, away = 2**53 + 3, 2**53 + 5
        statuses = ['finished'] * 8
        if year == 2024:
            statuses[-1] = 'notstarted'
        matches = pd.DataFrame({
            'event_id': pd.Series(ids, dtype='uint64'), 'competition_id': 17, 'season_id': season,
            'home_id': home, 'away_id': away, 'home_name': 'Alpha Home', 'away_name': 'Alpha Away',
            'kickoff_utc': [(pd.Timestamp(f'{year}-09-01', tz='UTC') + pd.Timedelta(days=7*i)).timestamp() for i in range(8)],
            'status': statuses, 'round': np.arange(1, 9), 'is_awarded': False,
            'home_score_current': pd.Series([2]*7 + [None if year == 2024 else 2], dtype='Int64'),
            'away_score_current': pd.Series([1]*7 + [None if year == 2024 else 1], dtype='Int64'),
        })
        records = []
        for i, event in enumerate(ids):
            for side, team, value in [('home', home, 3 + i % 4), ('away', away, 2 + i % 3)]:
                records.append(dict(event_id=event, period='ALL', group_name='Match overview', key='cornerKicks',
                                    side=side, team_id=team, value=value if statuses[i] == 'finished' else None))
        stats = pd.DataFrame(records)
        stats.event_id = stats.event_id.astype('uint64')
        pregame = stats[['event_id', 'team_id', 'side']].copy()
        pregame['position'] = np.tile([1, 2], 8)
        manifest = dict(schema_version='2', parser_version='2', version='v1',
                        scope=dict(league_id=17, league_name='Alpha', season_id=season,
                                   season_start=year, season_end=year+1), tables={})
        for name, frame in dict(matches=matches, statistics=stats, pregame=pregame).items():
            path = version / (name + '.parquet')
            pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)
            raw = path.read_bytes()
            manifest['tables'][name] = dict(file=path.name, rows=len(frame), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        (version / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        (root / (stem + '.manifest.json')).write_text(json.dumps(dict(
            schema_version='2', complete=True, matches=8, season_file=stem+'.parquet',
            manifest=(version/'manifest.json').relative_to(root).as_posix())), encoding='utf-8')
    recipe = default_recipe(root)
    recipe['name'] = 'Synthetic UI verification'
    recipe['output_dir'] = str(tmp_path / 'experiments')
    recipe['data']['leagues'] = ['Alpha']
    recipe['model'] = node('sklearn.linear_model.Ridge', alpha=0.3)
    return recipe


def test_catalog_json_defaults_and_all_constructor_parameters():
    catalog = catalog_for_ui()
    schema = catalog.schema()
    json.dumps(schema, allow_nan=False)
    for entry in schema:
        parameters = inspect.signature(catalog.entries[entry['id']]['constructor']).parameters
        expected = {k for k, p in parameters.items() if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)}
        assert expected <= {f['name'] for f in entry['fields']}, entry['id']
        for field in entry['fields']:
            if 'default' in field:
                catalog.build(json.loads(json.dumps(field['default'], allow_nan=False)))


def test_numeric_rolling_windows_have_no_temporal_split_choices():
    schema = {s['id']: s for s in catalog_for_ui().schema()}
    for name in ('RollingMean', 'RollingStd', 'RollingZScore'):
        field = next(f for f in schema['features.' + name]['fields'] if f['name'] == 'window')
        assert field['kind'] == 'number'
        assert 'choices' not in field


def test_recursive_ast_tuple_defaults_remain_hashable():
    catalog = catalog_for_ui()
    stat = node('features.Stat', period='ALL', group='Match overview', key='cornerKicks')
    expression = node('features.WarmStart', source=node('features.RollingStd', source=node(
        'features.LeaveOneOut', source=node('features.League', source=stat,
                                          round_keys=['competition_id', 'season_id', 'round'])), window=3),
                      policy=node('features.SeededEMA', handoff=node('features.LinearFade', start=1, rounds=2),
                                  round_keys=['competition_id', 'season_id', 'round']))
    built = catalog.build(expression)
    assert built.source.source.source.round_keys == ('competition_id', 'season_id', 'round')
    assert built.policy.round_keys == ('competition_id', 'season_id', 'round')
    assert hash(built) == hash(catalog.build(catalog.encode(built)))


def test_recipe_exports_roundtrip_without_execution(ui_recipe, tmp_path):
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps(ui_recipe), encoding='utf-8')
    assert read_recipe(path) == ui_recipe
    compile(export_python(ui_recipe), 'recipe.py', 'exec')
    notebook = export_notebook(ui_recipe)
    assert notebook['nbformat'] == 4
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), 'notebook-cell', 'exec')
    json.dumps(stage_schema(), allow_nan=False)


def test_prepare_fixed_run_reuse_and_saved_prediction(ui_recipe, tmp_path):
    from xdiyo_analytics.training import save_model
    prepared = prepare_recipe(ui_recipe)
    assert len(prepared.dataset.X) == 23
    assert len(prepared.split_plan.folds) == 1
    assert prepared.dataset.metadata.event_id.min() > 2**63
    assert {'history', 'matches', 'features', 'labels', 'ratings', 'team_catalog'} <= set(prepared.outputs)
    result = run_recipe(ui_recipe, prepared=prepared)
    assert len(result.training.folds) == 1
    assert result.post_report.studies
    assert 'Alpha Home' in result.to_html()
    loaded = run_recipe(ui_recipe, prepared=prepared)
    assert loaded.reused
    assert loaded.record['run_id'] == result.record['run_id']
    pd.testing.assert_frame_equal(loaded.training.prediction_frame(), result.training.prediction_frame())
    model_path = tmp_path / 'saved_model'
    save_model(result.training, model_path)
    prediction_recipe = deepcopy(ui_recipe)
    prediction_recipe['prediction'].update(model_path=str(model_path), rounds={'Alpha': [8]})
    dataset, predictions = predict_recipe(prediction_recipe)
    assert len(dataset.X) == 1
    assert dataset.y.isna().all().all()
    assert dataset.metadata['round'].tolist() == [8]
    assert len(predictions['predict']) == 1


@pytest.mark.parametrize('nested', [False, True])
def test_grid_search_uses_inner_rows_and_retains_outer_predictions(ui_recipe, nested):
    if nested:
        ui_recipe['split']['params']['train_size'] = 1
    ui_recipe['search'] = dict(nested=nested, grid={'alpha': [0.1, 1.0]},
                               split=node('splits.MatchKFold', n_splits=2),
                               options={'metrics': ['mse']})
    prepared = prepare_recipe(ui_recipe)
    result = run_recipe(ui_recipe, prepared=prepared)
    assert result.selection is not None
    if not nested:
        assert result.record['name'].count('alpha=') == 1
    assert len(result.training.folds) == (2 if nested else 1)
    for actual, expected in zip(result.training.folds, prepared.split_plan.folds):
        np.testing.assert_array_equal(actual.test_positions, expected.test)
        assert set(actual.train_positions).isdisjoint(actual.test_positions)
    assert result.path.exists()
    raw = result.selection.comparison
    assert 'run_id' in raw
    display = next(study for study in result.report.studies if study.name == 'selection/comparison').result.tables['comparison']
    assert 'parameters' in display
    assert all('alpha' in value for value in display.parameters)
    assert not {'run_id', 'saved_run_id', 'config_hash', 'comparison_group'} & set(display.columns)
    assert 'run_id' in result.selection.comparison  # display must not mutate evidence


def test_selection_presentation_retains_distinct_populations_without_hashes():
    from xdiyo_analytics.selection.presentation import comparison_for_display
    source = pd.DataFrame(dict(run_id=['opaque-a', 'opaque-b'], name=['Ridge', 'Lasso'],
                               config_hash=['hash-a', 'hash-b'], comparison_group=['sample-a', 'sample-b'],
                               score=[0.3, 0.8]))
    shown = comparison_for_display(source, {'opaque-a': {'grid_parameters': {'alpha': 0.1}},
                                          'opaque-b': {'model': {'params': {'alpha': 0.5}}}})
    assert shown['comparison population'].tolist() == ['Population 1', 'Population 2']
    assert [json.loads(v) for v in shown.parameters] == [{'alpha': 0.1}, {'alpha': 0.5}]
    assert set(source.columns) == {'run_id', 'name', 'config_hash', 'comparison_group', 'score'}


def test_named_ratings_and_warmup_prepare_from_recipe(ui_recipe):
    stat = deepcopy(ui_recipe['features']['corners_mean']['params']['source'])
    ui_recipe['ratings']['corners'] = {'stat': stat, 'engine': node('ratings.Glicko2', tau=0.4)}
    ui_recipe['features']['rating'] = node('features.Rating', name='corners', fields=['rating'], side='for')
    ui_recipe['features']['warm_mean'] = node('features.WarmStart',
        source=deepcopy(ui_recipe['features']['corners_mean']),
        policy=node('features.SeededEMA', handoff=node('features.ObservationCount', strength=3)))
    ui_recipe['cutoff_hours'] = 2
    prepared = prepare_recipe(ui_recipe)
    assert set(prepared.outputs['ratings']) == {'corners'}
    assert any('rating' in column for column in prepared.dataset.X)
    assert any('warm_mean' in column for column in prepared.dataset.X)


def test_factory_creates_fresh_estimator_per_restart(ui_recipe):
    from xdiyo_analytics.ui.recipe import ModelFactory
    ui_recipe['model'] = node('sklearn.linear_model.SGDRegressor', random_state=5)
    ui_recipe['preprocessors'] = []
    ui_recipe['adapter'] = node('training.IterativeAdapter', backend_factory={
        'factory': node('training.PartialFitBackend', estimator={'ref': 'estimator'})})
    adapter = ModelFactory(ui_recipe, catalog_for_ui())()
    first, second = adapter.backend_factory(3), adapter.backend_factory(4)
    assert first is not second
    assert first.estimator is not second.estimator
    assert first.estimator.random_state == 3
    assert second.estimator.random_state == 4


def test_fitted_selector_and_parallel_fold_execution(ui_recipe):
    ui_recipe['split']['params']['train_size'] = 1
    ui_recipe['execution']['params']['n_jobs'] = 2
    ui_recipe['fitted_reporters']['select'] = node('reporting.TopKCorrelationSelector',
                                                type='per_fold', partition='train', k=1)
    ui_recipe['candidate']['features_from'] = 'select'
    result = run_recipe(ui_recipe)
    assert len(result.training.folds) == 2
    assert all(len(f.feature_columns) == 1 for f in result.training.folds)
    assert all(f.training_summary['execution']['device'] == 'cpu' for f in result.training.folds)


def test_discovered_statistics_and_async_preparation_reuse(ui_recipe, tmp_path):
    from xdiyo_analytics.ui.server import BuilderState
    state = BuilderState(tmp_path, catalog_for_ui())
    try:
        inspection = state.dispatch('inspect', {'root': ui_recipe['data']['data_root'], 'stem': 'Alpha_24_25'})
        assert inspection['stats'] == [{'period': 'ALL', 'group_name': 'Match overview', 'key': 'cornerKicks'}]
        job = state.start('prepare', ui_recipe)
        state.jobs[job['id']]['future'].result(timeout=20)
        status = state.dispatch('job', job)
        assert status['status'] == 'complete', status['message']
        assert status['preview']['features']['count'] == 23
        assert not {'object', 'recipe', 'future', 'html'} & set(status)
        prepared = state.jobs[job['id']]['object']
        run = state.start('run', ui_recipe)
        state.jobs[run['id']]['future'].result(timeout=20)
        assert state.jobs[run['id']]['status'] == 'complete', state.jobs[run['id']]['message']
        assert state.jobs[run['id']]['object'].prepared is prepared
        changed_recipe = deepcopy(ui_recipe)
        changed_recipe['config']['different_recipe'] = True
        changed = state.start('run', changed_recipe)
        state.jobs[changed['id']]['future'].result(timeout=20)
        assert state.jobs[changed['id']]['status'] == 'complete', state.jobs[changed['id']]['message']
        assert state.jobs[changed['id']]['object'].prepared is not prepared
    finally:
        state.executor.shutdown(wait=True, cancel_futures=True)


@pytest.mark.parametrize('family', ['lightgbm', 'xgboost'])
@pytest.mark.parametrize('kind', ['regression', 'classification', 'ranking'])
def test_native_boosting_cpu_with_transformed_validation(ui_recipe, family, kind):
    pytest.importorskip(family)
    prefix = 'LGBM' if family == 'lightgbm' else 'XGB'
    suffix = {'regression': 'Regressor', 'classification': 'Classifier', 'ranking': 'Ranker'}[kind]
    params = dict(n_estimators=3, max_depth=2, n_jobs=1, random_state=11)
    if family == 'lightgbm':
        params.update(verbosity=-1, min_child_samples=1)
    else:
        params.update(verbosity=0, tree_method='hist')
    ui_recipe['model'] = node(f'{family}.{prefix}{suffix}', **params)
    ui_recipe['candidate']['validation'] = node('training.ValidationTail', fraction=0.25)
    if kind == 'classification':
        ui_recipe['labels']['corners'] = node('labels.Above', source=ui_recipe['labels']['corners'], threshold=7)
        ui_recipe['prediction_methods'] = ['predict', 'predict_proba']
    elif kind == 'ranking':
        source = ui_recipe['labels']['corners']['params']['source']
        ui_recipe['labels']['corners'] = node('labels.TeamValue', source=source)
        ui_recipe['assembly']['layout'] = 'team_match'
        if family == 'xgboost':
            ui_recipe['adapter'] = node('training.BoostingAdapter', estimator={'ref': 'estimator'},
                                        fit_kwargs={'sample_weight': [1.0] * 12})
    result = run_recipe(ui_recipe)
    fold = result.training.folds[0]
    assert len(fold.validation_positions) > 0
    assert len(fold.fit_positions) < len(fold.train_positions)
    scaler = fold.model.preprocessing_.steps[-1][1]
    assert scaler.n_samples_seen_ == len(fold.fit_positions)
    assert np.isfinite(fold.predictions['predict'].to_numpy()).all()
    assert len(fold.predictions['predict']) == len(fold.test_positions)
    if kind == 'classification':
        probabilities = fold.predictions['predict_proba']
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        assert set(probabilities.columns.get_level_values('class')) == {0, 1}


def test_nested_search_explicit_refit_recipe(ui_recipe):
    ui_recipe['split']['params']['train_size'] = 1
    ui_recipe['search'] = dict(nested=True, grid={'alpha': [0.1, 1.0]},
                               split=node('splits.MatchKFold', n_splits=2), options={'metrics': ['mse']})
    ui_recipe['refit'] = {'train_positions': 'all',
                          'candidate': {'model': node('sklearn.linear_model.Ridge', alpha=0.7)}}
    result = run_recipe(ui_recipe)
    assert result.refit.model is not None
    assert result.refit.model.estimator.steps[-1][1].alpha == 0.7
    assert len(result.refit.fit_positions) == 23


@pytest.mark.parametrize('family', ['lightgbm', 'xgboost'])
def test_native_classifier_preserves_outcome_label_classes(ui_recipe, family, tmp_path):
    pytest.importorskip(family)
    prefix = 'LGBM' if family == 'lightgbm' else 'XGB'
    params = dict(n_estimators=2, max_depth=2, n_jobs=1, random_state=11)
    params.update({'verbosity': -1, 'min_child_samples': 1} if family == 'lightgbm'
                  else {'verbosity': 0, 'tree_method': 'hist'})
    ui_recipe['model'] = node(f'{family}.{prefix}Classifier', **params)
    stat = ui_recipe['labels']['corners']['params']['source']
    ui_recipe['labels']['corners'] = node('labels.Outcome', source=stat)
    ui_recipe['assembly']['layout'] = 'team_match'
    ui_recipe['prediction_methods'] = ['predict', 'predict_proba']
    ui_recipe['candidate']['validation'] = node('training.ValidationTail', fraction=0.25)
    result = run_recipe(ui_recipe)
    fold = result.training.folds[0]
    assert set(fold.predictions['predict_proba'].columns.get_level_values('class')) == {-1, 0, 1}
    assert set(fold.predictions['predict'].iloc[:, 0]) <= {-1, 0, 1}
    if family == 'xgboost':
        from dataclasses import replace
        from xdiyo_analytics.training import save_model, load_model
        save_model(result.training, tmp_path / 'xgb-model')
        saved = load_model(tmp_path / 'xgb-model')
        positions = fold.test_positions
        subset = replace(result.dataset, **{name: getattr(result.dataset, name).iloc[positions].copy()
                                            for name in ('X', 'y', 'metadata')})
        predictions = saved.predict(subset)
        for method in ('predict', 'predict_proba'):
            pd.testing.assert_frame_equal(predictions[method].reset_index(drop=True),
                                          fold.predictions[method].reset_index(drop=True))


def test_iterative_recipe_validation_and_restart_history(ui_recipe):
    ui_recipe['features'] = {'home': node('features.IsHome')}
    ui_recipe['preprocessors'] = []
    ui_recipe['model'] = node('sklearn.linear_model.SGDRegressor', learning_rate='constant', eta0=0.01)
    ui_recipe['adapter'] = node('training.IterativeAdapter', backend_factory={
        'factory': node('training.PartialFitBackend', estimator={'ref': 'estimator'})})
    ui_recipe['candidate'].update(
        control=node('training.TrainingControl', max_steps=3, restarts=1, seed=9),
        validation=node('training.ValidationTail', fraction=0.25))
    result = run_recipe(ui_recipe)
    fold = result.training.folds[0]
    assert len(fold.training_summary['attempts']) == 2
    assert [a['seed'] for a in fold.training_summary['attempts']] == [9, 10]
    assert {'train_loss', 'validation_loss'} <= set(fold.training_history.metric)
    assert fold.training_history.step.max() == 3
