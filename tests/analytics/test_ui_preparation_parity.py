"""Notebook/public-helper parity and frozen future preparation through UI recipes."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from test_ui_workflow import ui_recipe
from xdiyo_analytics.ui import prepare_recipe, run_recipe, predict_recipe, catalog_for_ui
from xdiyo_analytics.ui.recipe import node, _grid_candidates
from xdiyo_analytics.features import SeededEMA, Hard, Column, Sum, Difference, Ratio, Constant, combine_features
from xdiyo_analytics.features.contextual import CalendarFeature, evaluate_context_features
from xdiyo_analytics.training import save_model, load_model, EstimatorAdapter


def preset_recipe(recipe, *, warm=False):
    recipe = deepcopy(recipe)
    recipe['features'] = {}
    recipe['labels'] = {'total_corners': recipe['labels']['corners']}
    recipe['target'] = 'total_corners'
    recipe['split'] = node('splits.TemporalSplit', train_size=2, test_size=1, unit='seasons',
        calendar_by=[], block_by=['source_season'])
    options = dict(windows=[3, 5, 20], lags=[1], spans=[3], include_ratings=False,
        include_loo=True, loo_windows=[3], loo_reducers=['mean', 'std'],
        include_h2h=True, h2h_windows=[3], h2h_reducers=['mean', 'std'])
    if warm:
        options['warm_policy'] = node('features.SeededEMA', alpha=.4, handoff=node('features.Hard', rounds=2))
    recipe['feature_preset'] = node('preparation.FeatureBankPreset', **options)
    recipe['numeric_features'] = node('preparation.NumericFeatures', dtype='float32')
    recipe['identity_features'] = node('preparation.IdentityFeatureSpec', columns=['source_league'], prefixes={'source_league': 'league'})
    recipe['post_reporters'] = {}
    return recipe


@pytest.mark.parametrize('warm', [False, True])
def test_actual_notebook_helper_matches_ui_columns_values_order_and_dtypes(ui_recipe, warm):
    from notebooks.helpers.quantile_corners import prepare
    recipe = preset_recipe(ui_recipe, warm=warm)
    recipe['cutoff_hours'] = 2
    expected = prepare(recipe['data']['data_root'], seasons=recipe['data']['seasons'], leagues=['Alpha'],
        windows=(3, 5, 20), lags=(1,), spans=(3,), include_ratings=False, cutoff_hours=2,
        include_loo=True, loo_windows=(3,), loo_reducers=('mean', 'std'),
        include_h2h=True, h2h_windows=(3,), h2h_reducers=('mean', 'std'),
        warm_policy=SeededEMA(alpha=.4, handoff=Hard(2)) if warm else None).prepared
    actual = prepare_recipe(recipe)
    pd.testing.assert_frame_equal(actual.dataset.X, expected.dataset.X)
    pd.testing.assert_frame_equal(actual.dataset.y, expected.dataset.y)
    assert set(actual.dataset.X.dtypes.astype(str)) == {'float32'}
    for actual_fold, expected_fold in zip(actual.split_plan.folds, expected.split_plan.folds):
        np.testing.assert_array_equal(actual_fold.train, expected_fold.train)
        np.testing.assert_array_equal(actual_fold.test, expected_fold.test)
    assert any('h2h_' in name for name in actual.dataset.X)
    assert any('loo_' in name for name in actual.dataset.X)
    assert any('warm::' in name for name in actual.dataset.X) == warm


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_explicit_postassembly_arithmetic_and_calendar_match_public_helpers(ui_recipe, dtype):
    original = prepare_recipe(ui_recipe)
    recipe = deepcopy(ui_recipe)
    home, away = node('prepared.Column', name='home::corners_mean'), node('prepared.Column', name='away::corners_mean')
    recipe['derived_features'] = {
        'sum': node('prepared.Sum', left=home, right=away),
        'difference': node('prepared.Difference', left=home, right=away),
        'ratio': node('prepared.Ratio', numerator=home, denominator=away),
        'constant': node('prepared.Constant', value=3.),
        'large': node('prepared.Constant', value=1e40)}
    recipe['context_features'] = {'weekday': node('context.CalendarFeature', kind='weekday')}
    recipe['numeric_features'] = node('preparation.NumericFeatures', dtype=dtype, infinities_to_missing=True)
    actual = prepare_recipe(recipe).dataset.X
    h, a = Column('home::corners_mean'), Column('away::corners_mean')
    expected = combine_features(original.dataset.X, {'sum': Sum(h, a), 'difference': Difference(h, a),
        'ratio': Ratio(h, a), 'constant': Constant(3.), 'large': Constant(1e40)})
    expected = pd.concat([expected, evaluate_context_features(original.dataset.metadata, {'weekday': CalendarFeature('weekday')})], axis=1)
    expected = expected.astype(dtype).replace([np.inf, -np.inf], np.nan)
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.large.isna().all() if dtype == 'float32' else actual.large.notna().all()


def expanded_holdout(recipe, season='24_25'):
    from xdiyo_analytics.data import load_seasons
    data = load_seasons(**recipe['data'])
    for frame in data.tables.values():
        if 'source_league' in frame:
            frame.loc[frame.source_season.eq(season), 'source_league'] = 'Unseen league'
    extra = data.statistics.loc[data.statistics.source_season.eq(season)].copy()
    extra['group_name'], extra['key'] = 'Shots', 'shotsOnGoal'
    data.tables['statistics'] = pd.concat([data.statistics, extra], ignore_index=True)
    return data


def test_heldout_only_stat_and_league_cannot_expand_discovered_schema(ui_recipe, monkeypatch):
    recipe = preset_recipe(ui_recipe)
    expanded = expanded_holdout(recipe)
    monkeypatch.setattr('xdiyo_analytics.data.load_seasons', lambda **kwargs: deepcopy(expanded))
    prepared = prepare_recipe(recipe)
    assert not any('shotsOnGoal' in name for name in prepared.dataset.X)
    assert [name for name in prepared.dataset.X if name.startswith('league::')] == ['league::Alpha']
    test = prepared.split_plan.folds[0].test
    assert prepared.dataset.metadata.iloc[test].source_league.eq('Unseen league').all()
    assert prepared.dataset.X.iloc[test]['league::Alpha'].eq(0).all()
    recipe['preset_seasons'] = ['24_25']  # Explicit discovery population remains an opt-in override.
    explicit = prepare_recipe(recipe)
    assert any('shotsOnGoal' in name for name in explicit.dataset.X)


def test_saved_prediction_reuses_training_vocabulary_with_new_future_category(ui_recipe, monkeypatch, tmp_path):
    recipe = preset_recipe(ui_recipe)
    prepared = prepare_recipe(recipe)
    result = run_recipe(recipe, prepared=prepared)
    path = save_model(result.training, tmp_path/'prepared-model')
    expanded = expanded_holdout(recipe)
    future_id = expanded.matches.loc[expanded.matches.status.eq('notstarted'), 'event_id'].iloc[0]
    new_id = int(future_id) + 100
    for name, frame in list(expanded.tables.items()):
        if 'event_id' not in frame:
            continue
        extra = frame.loc[frame.event_id.astype(object).eq(future_id)].copy()
        extra['event_id'] = pd.Series([new_id]*len(extra), index=extra.index, dtype=frame.event_id.dtype)
        if name == 'matches':
            extra['round'] = 9
            extra['kickoff_utc'] += 7*86400
        expanded.tables[name] = pd.concat([frame, extra], ignore_index=True)
    monkeypatch.setattr('xdiyo_analytics.data.load_seasons', lambda **kwargs: deepcopy(expanded))
    recipe['prediction']['model_path'] = str(path)
    recipe['prediction']['rounds'] = {'Unseen league': [9]}
    aligned, outputs = predict_recipe(recipe)
    assert len(aligned.X) == 1 and aligned.y.isna().all().all()
    assert aligned.metadata.event_id.tolist() == [new_id]
    assert list(aligned.X) == list(prepared.dataset.X)
    assert aligned.X['league::Alpha'].eq(0).all()
    assert not any('Unseen' in column or 'shotsOnGoal' in column for column in aligned.X)
    assert np.isfinite(outputs['predict'].to_numpy()).all()
    assert aligned.definitions['identity_features'] == prepared.dataset.definitions['identity_features']
    again, predictions = predict_recipe(recipe)
    pd.testing.assert_frame_equal(again.X, aligned.X)
    pd.testing.assert_frame_equal(predictions['predict'], outputs['predict'])


def test_ordinary_recipe_optional_keys_leave_preparation_unchanged(ui_recipe):
    expected = prepare_recipe(ui_recipe)
    legacy = deepcopy(ui_recipe)
    for key in ('feature_preset', 'preset_seasons', 'derived_features', 'context_features', 'numeric_features', 'identity_features'):
        legacy.pop(key, None)
    actual = prepare_recipe(legacy)
    pd.testing.assert_frame_equal(actual.dataset.X, expected.dataset.X)
    pd.testing.assert_frame_equal(actual.dataset.y, expected.dataset.y)


@pytest.mark.parametrize('kind', ['fixture', 'team', 'mixed'])
def test_preset_fixture_alias_and_successful_passes_prepare_and_export(ui_recipe, monkeypatch, kind):
    from xdiyo_analytics.data import load_seasons
    from xdiyo_analytics.ui.recipe import export_python, export_notebook

    data = load_seasons(**ui_recipe['data'])
    passes = data.statistics.copy()
    passes['group_name'], passes['key'] = 'Passes', 'accuratePasses'
    passes['value'] = passes['value'] * 100
    data.tables['statistics'] = pd.concat([data.statistics, passes], ignore_index=True)
    monkeypatch.setattr('xdiyo_analytics.data.load_seasons', lambda **kwargs: deepcopy(data))

    recipe = deepcopy(ui_recipe)
    recipe['stat_selection'] = None
    recipe['features'] = ({'goal_mean5': node('features.BayesianFixture', fields=['expected_home_goals'])}
                          if kind in ('fixture', 'mixed') else {})
    recipe['feature_preset'] = node('preparation.FeatureBankPreset',
        stats=[['Passes', 'accuratePasses']] if kind in ('team', 'mixed') else [],
        periods=['ALL'], windows=[5], lags=[], spans=[], std_windows=[], z_windows=[],
        league_windows=[], include_ratings=False, include_loo=False, include_h2h=False,
        include_rest=False, include_calendar=False, include_combinations=True)
    before = deepcopy(recipe)
    prepared = prepare_recipe(recipe)
    frame = prepared.dataset.X
    without = deepcopy(recipe)
    without['feature_preset']['params']['include_combinations'] = False
    baseline = prepare_recipe(without).dataset.X
    pd.testing.assert_frame_equal(frame[list(baseline)], baseline)
    expected_columns = list(baseline)
    for name, scope in prepared.outputs['features'].attrs['feature_scopes'].items():
        if scope == 'team':
            expected_columns.extend([f'sum::{name}', f'difference::{name}'])
    assert list(frame) == expected_columns

    if kind in ('fixture', 'mixed'):
        assert [name for name in frame if 'goal_mean5' in name] == ['fixture::goal_mean5']
        assert prepared.dataset.definitions['feature_scopes']['fixture::goal_mean5'] == 'fixture'
        assert frame['fixture::goal_mean5'].notna().all()
    if kind in ('team', 'mixed'):
        for role in ('for', 'against'):
            name = f'ALL_Passes_accuratePasses_{role}_mean5'
            assert prepared.outputs['features'].attrs['feature_scopes'][name] == 'team'
            home, away = frame[f'home::{name}'], frame[f'away::{name}']
            assert home.dropna().ge(100).all() and home.notna().any()
            np.testing.assert_allclose(frame[f'sum::{name}'], home + away, equal_nan=True)
            np.testing.assert_allclose(frame[f'difference::{name}'], home - away, equal_nan=True)

    # Execute the actual exported preparation cells, without fitting a model.
    sources = [export_python(recipe).split('result = run_recipe', 1)[0],
               ''.join(export_notebook(recipe)['cells'][1]['source'])]
    for source in sources:
        namespace = {}
        exec(compile(source, '<exported-preparation>', 'exec'), namespace)
        assert namespace['recipe'] == recipe
        restored = namespace['prepared']
        pd.testing.assert_frame_equal(restored.dataset.X, frame)
        pd.testing.assert_frame_equal(restored.dataset.y, prepared.dataset.y)
        assert restored.dataset.definitions['feature_scopes'] == prepared.dataset.definitions['feature_scopes']
    assert recipe == before


def test_grid_dotted_paths_reach_fitted_reporter_and_preprocessor_params(ui_recipe):
    recipe = deepcopy(ui_recipe)
    recipe['model'] = node('sklearn.linear_model.LogisticRegression', C=.5)
    recipe['fitted_reporters'] = {'weights': node('reporting.ClassWeightReporter', mode='power', power=0.)}
    recipe['candidate']['weights_from'] = 'weights'
    search = {'grid': {'fitted_reporters.weights.params.power': [0., .5],
                       'preprocessors.1.params.with_mean': [True, False]}}
    candidates = list(_grid_candidates(recipe, search, catalog_for_ui(), {}))
    assert len(candidates) == 4
    assert sorted(c.pre_analysis.reporters['weights'].power for c in candidates) == [0., 0., .5, .5]
    assert {c.model_factory().estimator[1].with_mean for c in candidates} == {True, False}


def test_schema_uses_intersection_of_selected_outer_training_populations(ui_recipe, monkeypatch):
    recipe = preset_recipe(ui_recipe)
    recipe['split']['params']['train_size'] = 1
    expanded = expanded_holdout(recipe, season='23_24')
    monkeypatch.setattr('xdiyo_analytics.data.load_seasons', lambda **kwargs: deepcopy(expanded))
    common = prepare_recipe(recipe)
    assert len(common.split_plan.folds) == 2
    assert not any('shotsOnGoal' in name for name in common.dataset.X)
    assert list(common.dataset.definitions['identity_features']['categories']['source_league']) == ['Alpha']
    recipe['fold_ids'] = [1]
    selected = prepare_recipe(recipe)
    assert len(selected.split_plan.folds) == 1
    assert any('shotsOnGoal' in name for name in selected.dataset.X)
    assert list(selected.dataset.definitions['identity_features']['categories']['source_league']) == ['Alpha', 'Unseen league']


def test_direct_warm_rating_recipe_matches_public_expression(ui_recipe):
    from xdiyo_analytics.features import WarmStart, MatchResultGlicko, evaluate_features
    from xdiyo_analytics.ratings import GlickoTransition
    from xdiyo_analytics.datasets import assemble_dataset
    recipe = deepcopy(ui_recipe)
    recipe['features'] = {'warm_rating': node('features.WarmStart',
        source=node('features.MatchResultGlicko', side='for', fields=['rating', 'rd']),
        policy=node('ratings.GlickoTransition', phi_scale=1.2))}
    actual = prepare_recipe(recipe)
    values = evaluate_features(actual.outputs['history'], {'warm_rating': WarmStart(
        MatchResultGlicko(side='for', fields=('rating', 'rd')), GlickoTransition(phi_scale=1.2))}, keyed=True)
    expected = assemble_dataset(values, actual.outputs['labels']['corners'], layout='match', drop_missing_targets=True)
    pd.testing.assert_frame_equal(actual.dataset.X, expected.X)


def test_artifact_export_publishes_reports_and_reloads_evaluated_models_without_refitting(ui_recipe, monkeypatch):
    recipe = deepcopy(ui_recipe)
    prepared = prepare_recipe(recipe)
    first = run_recipe(recipe, prepared=prepared)
    def forbidden(*args, **kwargs):
        raise AssertionError('Enabling artifact export must not refit')
    monkeypatch.setattr(EstimatorAdapter, 'fit', forbidden)
    recipe['artifact_export'] = node('experiments.ArtifactExport')
    exported = run_recipe(recipe, prepared=prepare_recipe(recipe))
    assert exported.reused and exported.record['run_id'] == first.record['run_id']
    manifest = json.loads((exported.export_path/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['run_id'] == first.record['run_id']
    assert {'feature_manifest', 'selected_features', 'observed', 'match_metadata', 'predict', 'metrics'} <= {t['name'] for t in manifest['tables']}
    for item in manifest['tables']:
        assert (exported.export_path/item['file']).is_file()
        if item['name'] == 'metrics':
            assert item['partition'] == 'score' and item['type'] == 'overall'
            metrics = pd.read_csv(exported.export_path/item['file'], index_col=0)
            assert set(metrics.metric) == {'mse', 'mae'}
        if item['name'] == 'observed':
            table = pd.read_csv(exported.export_path/item['file'], index_col=0)
            pd.testing.assert_frame_equal(table, first.training.folds[item['fold_id']].y_true, check_dtype=False)
    assert len(manifest['models']) == len(first.training.folds)
    for item in manifest['models']:
        assert item['reload_verified'] is True
        restored = load_model(exported.export_path/item['path'])
        fold = first.training.folds[item['fold_id']]
        for key, frame in restored.predict(prepared.dataset, positions=fold.test_positions).items():
            pd.testing.assert_frame_equal(frame, fold.predictions[key])
    before = {p: p.read_bytes() for p in exported.export_path.rglob('*') if p.is_file()}
    recipe['artifact_export'] = node('experiments.ArtifactExport', report_tables=False,
        predictions=False, save_models=False, verify_reload=False)
    second = run_recipe(recipe, prepared=prepare_recipe(recipe))
    assert second.reused and second.record['run_id'] == first.record['run_id']
    assert second.export_path != exported.export_path
    reduced = json.loads((second.export_path/'manifest.json').read_text(encoding='utf-8'))
    assert reduced['models'] == []
    assert {t['name'] for t in reduced['tables']} == {'feature_manifest', 'selected_features'}
    assert before == {p: p.read_bytes() for p in before}


def test_export_reload_verification_requires_saving_models(ui_recipe):
    from xdiyo_analytics.experiments.exports import ArtifactExport
    result = run_recipe(ui_recipe)
    with pytest.raises(ValueError, match='save_models=True'):
        ArtifactExport(save_models=False, verify_reload=True).run(result)


def test_count_preset_reporters_export_and_reuse_end_to_end(ui_recipe, monkeypatch):
    from xdiyo_analytics.ui.adapters import BoostingAdapter
    recipe = preset_recipe(ui_recipe)
    recipe['model'] = node('xgboost.XGBClassifier', n_estimators=3, max_depth=2,
        n_jobs=1, random_state=11, tree_method='hist', device='cpu')
    recipe['adapter'] = node('training.BoostingAdapter', estimator={'ref': 'estimator'}, exact_counts=True)
    recipe['post_reporters'] = {
        'counts': node('reporting.CountClassificationReporter', type='overall', partition='score',
                       group_by=['source_league'], tolerance=2),
        'importance': node('reporting.FeatureImportanceReporter', type='overall', partition='model', top_k=3),
        'timeline': node('reporting.PredictionTimelineReporter', type='timeline', partition='score', max_points=3),
        'performance': node('reporting.PerformanceReporter', type='overall', partition='score', metrics=['mse', 'mae']),
        'leaderboard': node('reporting.ExperimentLeaderboardReporter', weights={'mae': 1.})}
    recipe['artifact_export'] = node('experiments.ArtifactExport')
    prepared = prepare_recipe(recipe)
    first = run_recipe(recipe, prepared=prepared)
    studies = {study.name.split('/')[-1]: study for study in first.report.studies}
    assert set(recipe['post_reporters']) <= set(studies)
    counts = studies['counts'].result.tables
    assert counts['metrics_by_group'].source_league.tolist() == ['Alpha']
    assert len(counts['details::total_corners']) == len(first.training.folds[0].test_positions)
    assert len(studies['timeline'].result.tables['timeline::total_corners']) > 3
    assert len(studies['timeline'].result.artifacts[0].data.data[0].x) == 3
    assert len(studies['leaderboard'].result.tables['leaderboard']) == 1
    assert set(studies['importance'].result.tables['importance'].feature) <= set(prepared.dataset.X)
    manifest = json.loads((first.export_path/'manifest.json').read_text(encoding='utf-8'))
    assert {'metrics_by_group', 'importance', 'timeline::total_corners', 'leaderboard'} <= {entry['name'] for entry in manifest['tables']}
    for entry in manifest['tables']:
        assert (first.export_path/entry['file']).is_file()
    for entry in manifest['models']:
        assert entry['reload_verified'] is True
        fold = first.training.folds[entry['fold_id']]
        restored = load_model(first.export_path/entry['path'])
        for name, values in restored.predict(prepared.dataset, positions=fold.test_positions).items():
            pd.testing.assert_frame_equal(values, fold.predictions[name])
    def forbidden(*args, **kwargs):
        raise AssertionError('Changing diagnostics must not refit')
    monkeypatch.setattr(BoostingAdapter, 'fit', forbidden)
    recipe['post_reporters']['counts']['params']['tolerance'] = 1
    second = run_recipe(recipe, prepared=prepare_recipe(recipe))
    assert second.reused and second.record['run_id'] == first.record['run_id']
    assert second.export_path != first.export_path
    repeat = {study.name.split('/')[-1]: study for study in second.report.studies}
    assert len(repeat['leaderboard'].result.tables['leaderboard']) == 1
    assert counts['metrics'].tolerance.tolist() == [2]
    assert repeat['counts'].result.tables['metrics'].tolerance.tolist() == [1]
    refreshed_manifest = json.loads((second.export_path/'manifest.json').read_text(encoding='utf-8'))
    count_export = next(entry for entry in refreshed_manifest['tables']
                        if entry.get('study', '').endswith('counts') and entry['name'] == 'metrics')
    assert pd.read_csv(second.export_path/count_export['file']).tolerance.tolist() == [1]
    pd.testing.assert_frame_equal(first.training.folds[0].predictions['predict_proba'],
                                  second.training.folds[0].predictions['predict_proba'])


def test_exact_count_boosting_retains_original_labels_and_training_class_count():
    from xgboost import XGBClassifier
    from xdiyo_analytics.ui.adapters import BoostingAdapter
    from test_class_weighting import weight_context
    context = weight_context()
    model = XGBClassifier(n_estimators=2, max_depth=1, n_jobs=1, tree_method='hist', device='cpu')
    adapter = BoostingAdapter(model, prediction_methods=('predict', 'predict_proba'), exact_counts=True)
    adapter.fit(context)
    assert model.get_params()['objective'] == 'multi:softprob'
    assert model.get_params()['num_class'] == 3
    outputs = adapter.predict(context)
    assert outputs['predict_proba'].columns.get_level_values(1).tolist() == [2, 7, 12]
    assert set(outputs['predict'].corners).issubset({2, 7, 12})
    np.testing.assert_allclose(outputs['predict_proba'].sum(axis=1), 1., atol=1e-6)


@pytest.mark.parametrize('value', [-1., 2.5, np.nan, np.inf])
def test_exact_count_boosting_rejects_non_counts_before_native_fit(value, monkeypatch):
    from xgboost import XGBClassifier
    from xdiyo_analytics.ui.adapters import BoostingAdapter
    from test_class_weighting import weight_context
    context = weight_context()
    context.y = context.y.astype(float)
    context.y.iloc[0, 0] = value
    monkeypatch.setattr(XGBClassifier, 'fit', lambda *a, **k: pytest.fail('Invalid target reached native fitting'))
    with pytest.raises(ValueError, match='finite nonnegative integer'):
        BoostingAdapter(XGBClassifier(), exact_counts=True).fit(context)


def test_target_mode_metadata_supports_adapters_with_readonly_summary():
    from test_probability_calibration import ScopedProbabilities, classification_data
    from model_selection_samples import outer
    from xdiyo_analytics.training import TrainingRunner
    from xdiyo_analytics.splits import SplitPlan
    class ReadonlySummary(ScopedProbabilities):
        @property
        def training_summary_(self):
            return {'custom_summary': True}
    data = classification_data()
    fold = outer(data)
    result = TrainingRunner(ReadonlySummary).run(data, SplitPlan([fold], len(data.X), np.arange(len(data.X))))
    summary = result.folds[0].training_summary
    assert summary['custom_summary'] is True
    assert summary['training_target_mode_rows'] == len(fold.train)
    assert summary['training_target_modes']['target'] == data.y.iloc[fold.train].target.mode().iloc[0]


def test_training_target_modes_use_fit_rows_excluding_validation_and_calibration():
    from test_probability_calibration import ScopedProbabilities, classification_data
    from model_selection_samples import outer
    from xdiyo_analytics.training import TrainingRunner, ValidationTail, ProbabilityCalibrator
    from xdiyo_analytics.splits import SplitPlan
    data = classification_data()
    data.y['target'] = ((data.metadata.case >= 6) | data.metadata.case.eq(0)).astype(int)
    fold = outer(data)
    result = TrainingRunner(ScopedProbabilities, validation=ValidationTail(.25),
        calibration=ProbabilityCalibrator(fraction=.25)).run(data,
            SplitPlan([fold], len(data.X), np.arange(len(data.X)))).folds[0]
    assert data.y.iloc[fold.train].target.mode().iloc[0] == 1
    assert result.training_summary['training_target_modes']['target'] == 0
    assert result.training_summary['training_target_mode_rows'] == len(result.fit_positions)
