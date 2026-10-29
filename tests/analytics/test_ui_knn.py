"""KNN uses the ordinary catalog, fold-local preprocessing and saved models."""
from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from test_class_weighting import weight_context
from test_ui_workflow import ui_recipe
from xdiyo_analytics.training import EstimatorAdapter, save_model, load_model
from xdiyo_analytics.ui import catalog_for_ui, prepare_recipe, run_recipe
from xdiyo_analytics.ui.recipe import export_python, node
from xdiyo_analytics.ui.inventory import inventory
from xdiyo_analytics.research.deepthroat_gg import OutcomeAdapter


@pytest.mark.parametrize('name', ['KNeighborsClassifier','KNeighborsRegressor'])
def test_catalog_recipe_search_pipeline_and_persistence(name, ui_recipe, tmp_path):
    catalog = catalog_for_ui()
    spec = node('sklearn.neighbors.'+name, n_neighbors=2, weights='distance', p=1, algorithm='brute', n_jobs=1)
    estimator = catalog.build(spec)
    restored = catalog.build(json.loads(json.dumps(catalog.encode(estimator))))
    assert restored.get_params() == clone(estimator).get_params()
    fields = {f['name']:f for f in inventory()['components']['sklearn.neighbors.'+name]['fields']}
    assert set(estimator.get_params()) <= set(fields)
    assert fields['weights']['choices'] == ['uniform','distance']
    assert fields['metric_params']['kind'] == 'map'
    assert fields['metric']['kind'] == 'text' and 'discovery' not in fields['metric']
    ui_recipe['model'] = spec
    metric = 'accuracy' if name.endswith('Classifier') else 'mse'
    ui_recipe['search'] = dict(nested=False, grid={'n_neighbors':[1,2]}, split=node('splits.MatchKFold',n_splits=2),options={'metrics':[metric]})
    scope = {}; exec(export_python(ui_recipe).split('result = run_recipe')[0], scope)
    assert scope['recipe'] == ui_recipe
    prepared = prepare_recipe(ui_recipe)
    result = run_recipe(ui_recipe, prepared=prepared)
    fold = result.training.folds[0]
    assert fold.model.estimator[-2].n_samples_seen_ == len(fold.fit_positions)
    assert set(fold.fit_positions).isdisjoint(fold.test_positions)
    if name.endswith('Classifier'):
        np.testing.assert_allclose(fold.predictions['predict_proba'].sum(axis=1), 1)
    save_model(result.training,tmp_path/'model')
    loaded = load_model(tmp_path/'model')
    data = prepared.dataset
    subset = replace(data, **{key:getattr(data,key).iloc[fold.test_positions].copy() for key in ('X','y','metadata')})
    for method, prediction in loaded.predict(subset).items():
        pd.testing.assert_frame_equal(prediction.reset_index(drop=True),fold.predictions[method].reset_index(drop=True))


@pytest.mark.parametrize('model', [KNeighborsClassifier(weights='distance',n_neighbors=2), KNeighborsRegressor(n_neighbors=2)])
def test_threads_device_and_unsupported_weights(model):
    context = weight_context()
    adapter = EstimatorAdapter(Pipeline([('scale',StandardScaler()),('model',clone(model))]))
    adapter.configure_threads(2)
    assert adapter.estimator[-1].n_jobs == 2
    assert adapter.configure_device('cpu') == 'cpu'
    with pytest.raises(ValueError, match='CPU'):
        adapter.configure_device('cuda')
    context.sample_weight = pd.Series(np.ones(len(context.X)),index=context.X.index)
    with pytest.raises(TypeError, match='sample_weight|classifier|classification'):
        adapter.fit(context)


def test_multioutput_and_fixed_missing_class_schema():
    context = weight_context(labels=(1,1,0,0,1,0))
    model = OutcomeAdapter(KNeighborsClassifier(n_neighbors=2),('predict','predict_proba'))
    model.fit(context)
    p = model.predict(context)['predict_proba']
    assert p.columns.get_level_values('class').tolist() == [1,0,-1]
    assert p.iloc[:,2].eq(0).all()
    np.testing.assert_allclose(p.sum(axis=1),1)
    context.y['away'] = np.arange(len(context.X))
    reg = EstimatorAdapter(KNeighborsRegressor(n_neighbors=2))
    reg.fit(context)
    expected = KNeighborsRegressor(n_neighbors=2).fit(context.X,context.y).predict(context.X)
    np.testing.assert_array_equal(reg.predict(context)['predict'],expected)


def test_multioutput_goal_recipe(ui_recipe):
    ui_recipe['model'] = node('sklearn.neighbors.KNeighborsRegressor',n_neighbors=2)
    ui_recipe['labels'] = {'goals':node('labels.MatchGoals')}
    ui_recipe['target'] = 'goals'
    result = run_recipe(ui_recipe)
    prediction = result.training.folds[0].predictions['predict']
    assert list(prediction) == ['goals::home_goals','goals::away_goals']
    assert prediction.shape[1] == 2
