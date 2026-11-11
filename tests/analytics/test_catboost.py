"""Real optional CatBoost fits through native adapters, recipes and persistence."""
from dataclasses import replace
import inspect
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pytest.importorskip('catboost')
from xdiyo_analytics.training import catboost as cb

from xdiyo_analytics.training import FitContext, JoblibSerializer, load_model, save_model
from xdiyo_analytics.ui import catalog_for_ui, run_recipe
from xdiyo_analytics.ui.adapters import BoostingAdapter
from xdiyo_analytics.ui.recipe import ModelFactory, export_python, node, read_recipe
from test_ui_workflow import ui_recipe


def context():
    # Interleaved match rows; identifiers above float's exact integer range.
    index = pd.Index([90, 20, 71, 55, 12, 88], name='row_position')
    X = pd.DataFrame({'strength': [1., 4., 3., 2., 6., 5.],
                      'venue': ['home', 'away', 'home', 'away', 'home', 'away']}, index=index)
    meta = pd.DataFrame({'competition_id': 17,
                         'match_id': pd.array([2**60+i for i in [1, 2, 3, 1, 2, 3]], dtype='uint64'),
                         'weight': [1., 2., 3., 4., 5., 6.]}, index=index)
    return FitContext(X=X, y=pd.DataFrame({'target': [0., 1., 2., 1., 2., 0.]}, index=index),
                      metadata=meta, layout='team_match', match_columns=('competition_id', 'match_id'), fold_id=0)


def params(**kwargs):
    return dict(iterations=4, depth=2, thread_count=1, random_seed=7, verbose=False,
                allow_writing_files=False, **kwargs)


@pytest.mark.parametrize('suffix', ['Regressor', 'Classifier', 'Ranker'])
def test_native_categorical_validation_and_persistence(suffix, tmp_path):
    ctx = context()
    if suffix == 'Classifier':
        ctx.y['target'] = ['loss', 'draw', 'win', 'draw', 'win', 'loss']
    validation = replace(ctx, X=ctx.X.assign(strength=ctx.X.strength + 100))
    ctx = replace(ctx, validation=validation)
    native = getattr(cb, 'CatBoost' + suffix)(**params(cat_features=['venue']))
    methods = ('predict', 'predict_proba') if suffix == 'Classifier' else ('predict',)
    adapter = BoostingAdapter(native, methods)
    adapter.fit(ctx)
    assert adapter.training_summary_['history_source'] == 'native_estimator'
    assert adapter.training_history_.metric.str.startswith('validation.').any()
    predictions = adapter.predict(ctx)
    assert predictions['predict'].index.equals(ctx.X.index)
    if suffix == 'Classifier':
        assert set(predictions['predict_proba'].columns.get_level_values('class')) == {'loss', 'draw', 'win'}
        np.testing.assert_allclose(predictions['predict_proba'].sum(axis=1), 1)
    else:
        assert np.isfinite(predictions['predict']).all().all()
    serializer = JoblibSerializer()
    serializer.save(adapter, tmp_path)
    restored = serializer.load(tmp_path)
    for method, frame in predictions.items():
        pd.testing.assert_frame_equal(frame, restored.predict(ctx)[method], check_exact=True)
    assert clone(native).get_params() == native.get_params()
    assert not clone(native).is_fitted()


def test_ranker_group_order_weights_and_native_reference(monkeypatch):
    ctx = context()
    ctx = replace(ctx, validation=replace(ctx, X=ctx.X.assign(strength=ctx.X.strength + 10)))
    original = cb.CatBoostRanker.fit
    captured = {}
    def record(self, X, y, **kwargs):
        captured.update(X=X.copy(), y=y.copy(), **kwargs)
        return original(self, X, y, **kwargs)
    monkeypatch.setattr(cb.CatBoostRanker, 'fit', record)
    adapter = BoostingAdapter(cb.CatBoostRanker(**params(loss_function='QueryRMSE', cat_features=['venue'])),
                              sample_weight_column='weight')
    adapter.fit(ctx)
    expected = [0, 3, 1, 4, 2, 5]
    pd.testing.assert_frame_equal(captured['X'], ctx.X.iloc[expected])
    np.testing.assert_array_equal(captured['group_id'], [0, 0, 1, 1, 2, 2])
    np.testing.assert_array_equal(captured['sample_weight'], ctx.metadata.weight.iloc[expected])
    pool = captured['eval_set']
    np.testing.assert_array_equal(pool.get_weight(), ctx.validation.metadata.weight.iloc[expected])
    ids = pool.get_group_id_hash()
    assert ids[0] == ids[1] and ids[2] == ids[3] and ids[4] == ids[5] and len(set(ids)) == 3
    reference = cb.CatBoostRanker(**adapter.native_model_.get_params())
    original(reference, ctx.X.iloc[expected], ctx.y.iloc[expected, 0],
             group_id=[0, 0, 1, 1, 2, 2], sample_weight=ctx.metadata.weight.iloc[expected], eval_set=pool)
    np.testing.assert_array_equal(adapter.predict(ctx)['predict'].iloc[:, 0], reference.predict(ctx.X))


def test_preprocessing_fit_only_and_numeric_nan():
    ctx = context()
    ctx.X = ctx.X[['strength']]
    ctx.X.iloc[2, 0] = np.nan
    validation = replace(ctx, X=ctx.X.fillna(100).add(1000))
    adapter = BoostingAdapter(Pipeline([('scale', StandardScaler()), ('model', cb.CatBoostRegressor(**params()))]))
    adapter.fit(replace(ctx, validation=validation))
    np.testing.assert_allclose(adapter.preprocessing_[0].mean_, ctx.X.mean())
    assert adapter.preprocessing_[0].n_samples_seen_ == 5
    assert np.isfinite(adapter.predict(validation)['predict']).all().all()


@pytest.mark.parametrize('device,task,devices', [('cpu', 'CPU', None), ('auto', 'CPU', None), ('cuda', 'GPU', '0'), ('cuda:2', 'GPU', '2')])
def test_execution_policy_mapping_without_gpu_fit(device, task, devices):
    adapter = BoostingAdapter(Pipeline([('scale', StandardScaler()), ('model', cb.CatBoostRegressor())]))
    resolved = adapter.configure_device(device)
    adapter.configure_threads(2)
    actual = adapter._native().get_params()
    assert actual['task_type'] == task and actual.get('devices') == devices
    assert actual['thread_count'] == 2
    assert resolved == ('cpu' if device == 'auto' else 'cuda:0' if device == 'cuda' else device)


@pytest.mark.parametrize('kwargs', [dict(class_weights=[1, 2, 3]), dict(auto_class_weights='Balanced')])
def test_common_weights_reject_native_double_weighting(kwargs):
    ctx = context(); ctx.X = ctx.X[['strength']]
    ctx.sample_weight = pd.Series(1., index=ctx.X.index)
    adapter = BoostingAdapter(cb.CatBoostClassifier(**params(**kwargs)))
    with pytest.raises(ValueError, match='Disable native'):
        adapter.fit(ctx)


def test_common_classifier_weights_and_no_implicit_file_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ctx = context(); ctx.X = ctx.X[['strength']]
    ctx.sample_weight = pd.Series([1., 2., 1., 2., 1., 2.], index=ctx.X.index)
    native = cb.CatBoostClassifier(iterations=2, thread_count=1, verbose=False)
    adapter = BoostingAdapter(native, ('predict', 'predict_proba'))
    adapter.fit(replace(ctx, validation=replace(ctx, sample_weight=ctx.sample_weight * 2)))
    assert not (tmp_path / 'catboost_info').exists()
    assert native.get_params()['allow_writing_files'] is False


@pytest.mark.parametrize('key', ['group_id', 'eval_set', 'pairs'])
def test_ranker_rejects_overridden_membership_and_validation(key):
    adapter = BoostingAdapter(cb.CatBoostRanker(**params()), fit_kwargs={key: []})
    ctx = context(); ctx.X = ctx.X[['strength']]
    with pytest.raises(ValueError, match='owned by the runner|custom adapter'):
        adapter.fit(ctx)


@pytest.mark.parametrize('suffix', ['Regressor', 'Classifier', 'Ranker'])
def test_recipe_training_exports_and_saved_model(ui_recipe, suffix, tmp_path):
    ui_recipe['model'] = node('catboost.CatBoost' + suffix, **params())
    ui_recipe['candidate']['validation'] = node('training.ValidationTail', fraction=0.25)
    if suffix == 'Classifier':
        source = ui_recipe['labels']['corners']['params']['source']
        ui_recipe['labels']['corners'] = node('labels.Outcome', source=source)
        ui_recipe['assembly']['layout'] = 'team_match'
    elif suffix == 'Ranker':
        source = ui_recipe['labels']['corners']['params']['source']
        ui_recipe['labels']['corners'] = node('labels.TeamValue', source=source)
        ui_recipe['assembly']['layout'] = 'team_match'
    path = tmp_path / 'recipe.json'; path.write_text(json.dumps(ui_recipe))
    assert read_recipe(path) == ui_recipe
    exported = export_python(ui_recipe)
    namespace = {'__name__': 'catboost_export_test'}
    exec(compile(exported, 'recipe.py', 'exec'), namespace)
    result = run_recipe(ui_recipe)
    fold = result.training.folds[0]
    assert isinstance(fold.model, BoostingAdapter)
    assert fold.model.preprocessing_.steps[-1][1].n_samples_seen_ == len(fold.fit_positions)
    assert fold.training_history.metric.str.startswith('validation.').any()
    save_model(result.training, tmp_path / 'saved')
    restored = load_model(tmp_path / 'saved')
    subset = replace(result.dataset, **{n: getattr(result.dataset, n).iloc[fold.test_positions].copy()
                                       for n in ('X', 'y', 'metadata')})
    for method, frame in fold.predictions.items():
        pd.testing.assert_frame_equal(restored.predict(subset)[method].reset_index(drop=True), frame.reset_index(drop=True), check_exact=True)
    if suffix == 'Classifier':
        assert set(fold.predictions['predict_proba'].columns.get_level_values('class')) == {-1, 0, 1}


def test_ui_schema_all_constructor_parameters_and_fresh_factories():
    catalog = catalog_for_ui(); schemas = {s['id']: s for s in catalog.schema()}
    for suffix in ('Regressor', 'Classifier', 'Ranker'):
        key = 'catboost.CatBoost' + suffix
        fields = {f['name']: f for f in schemas[key]['fields']}
        assert set(inspect.signature(getattr(cb, 'CatBoost' + suffix)).parameters) <= fields.keys()
        assert fields['iterations']['kind'] == 'number'
        assert fields['cat_features']['kind'] == 'list'
        assert fields['task_type']['choices'] == ['CPU', 'GPU']
        spec = node(key, **params())
        encoded = catalog.encode(catalog.build(spec))
        assert catalog.encode(catalog.build(encoded)) == encoded
        assert all(encoded['params'][k] == v for k, v in spec['params'].items())
        factory = ModelFactory(dict(model=spec), catalog)
        assert factory().estimator is not factory().estimator


@pytest.mark.parametrize('suffix', ['Regressor', 'Classifier'])
def test_clone_pipeline_grid_search_and_native_parity(suffix):
    from sklearn.model_selection import GridSearchCV
    import catboost as native
    ctx = context()
    model = getattr(cb, 'CatBoost' + suffix)(**params(cat_features=['venue']))
    reference = getattr(native, 'CatBoost' + suffix)(**model.get_params())
    model.fit(ctx.X, ctx.y.iloc[:, 0]); reference.fit(ctx.X, ctx.y.iloc[:, 0])
    np.testing.assert_array_equal(model.predict(ctx.X), reference.predict(ctx.X))
    pipeline = Pipeline([('model', clone(model))])
    search = GridSearchCV(pipeline, {'model__depth': [1, 2]}, cv=2,
                          scoring='neg_mean_squared_error' if suffix == 'Regressor' else 'accuracy',
                          error_score='raise')
    search.fit(ctx.X, ctx.y.iloc[:, 0])
    assert len(search.predict(ctx.X)) == len(ctx.X)
    assert not clone(search.best_estimator_).named_steps['model'].is_fitted()


def test_optional_catboost_absence_does_not_break_catalog():
    import subprocess, sys
    code = '''
import sys
sys.modules['catboost'] = None
from xdiyo_analytics.ui import catalog_for_ui
catalog = catalog_for_ui()
assert not any(k.startswith('catboost.') for k in catalog.entries)
assert 'sklearn.ensemble.RandomForestClassifier' in catalog.entries
'''
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)


def test_native_composition_ranker_preserves_grouped_validation():
    from xdiyo_analytics.composition import Estimator
    ctx = context()
    spec = Estimator(cb.CatBoostRanker(**params(cat_features=['venue'])))
    adapter = spec.build()
    assert isinstance(adapter, BoostingAdapter)
    adapter.fit(replace(ctx, validation=ctx))
    assert len(adapter.predict(ctx)['predict']) == len(ctx.X)
    assert not spec.estimator.is_fitted()
    assert not spec.build().estimator.is_fitted()
