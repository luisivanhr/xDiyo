"""Native SVM recipes use the existing estimator, search and weight paths."""

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from test_class_weighting import weight_context
from test_ui_workflow import ui_recipe
from xdiyo_analytics.training import EstimatorAdapter, load_model, save_model
from xdiyo_analytics.ui import catalog_for_ui, prepare_recipe, run_recipe
from xdiyo_analytics.ui.inventory import inventory
from xdiyo_analytics.ui.recipe import export_notebook, export_python, node
from xdiyo_analytics.weighting import ClassWeightPolicy


@pytest.mark.parametrize('name', ['SVR', 'SVC'])
@pytest.mark.parametrize('gamma', ['scale', 'auto', 0.2])
def test_svm_catalog_roundtrip_and_native_controls(name, gamma):
    catalog = catalog_for_ui()
    spec = node('sklearn.svm.' + name, C=0.7, gamma=gamma, kernel='rbf')
    estimator = catalog.build(spec)
    restored = catalog.build(json.loads(json.dumps(catalog.encode(estimator))))
    assert restored.get_params() == estimator.get_params() == clone(estimator).get_params()
    schema = inventory()['components']['sklearn.svm.' + name]
    fields = {f['name']: f for f in schema['fields']}
    assert schema['category'] == 'model'
    assert fields['gamma']['kind'] == 'rule_number'
    assert fields['gamma']['rules'] == ['scale', 'auto']
    assert 'choices' not in fields['gamma']
    assert fields['C']['primary'] and fields['kernel']['primary']
    if name == 'SVC':
        assert fields['probability']['default'] is False
        assert fields['probability']['primary']
        assert fields['class_weight']['default'] is None


@pytest.mark.parametrize('name', ['SVR', 'SVC'])
def test_svm_recipe_search_exports_probabilities_and_saved_prediction(ui_recipe, tmp_path, name):
    params = dict(C=0.7, kernel='rbf', gamma=0.2)
    if name == 'SVC':
        params.update(probability=True, random_state=17)
        ui_recipe['fitted_reporters'] = {'weights': node('reporting.ClassWeightReporter', mode='power', power=0.5)}
        ui_recipe['candidate']['weights_from'] = 'weights'
    ui_recipe['model'] = node('sklearn.svm.' + name, **params)
    metric = 'accuracy' if name == 'SVC' else 'mse'
    ui_recipe['search'] = dict(nested=False, grid={'C': [0.5, 1.0]},
                               split=node('splits.MatchKFold', n_splits=2), options={'metrics': [metric]})
    recipe = json.loads(json.dumps(ui_recipe))
    scope = {}
    exec(export_python(recipe).split('result = run_recipe')[0], scope)
    assert scope['recipe'] == recipe
    for cell in export_notebook(recipe)['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), 'svm-notebook', 'exec')
    prepared = prepare_recipe(recipe)
    result = run_recipe(recipe, prepared=prepared)
    assert result.selection is not None
    fold = result.training.folds[0]
    assert set(fold.fit_positions).isdisjoint(fold.test_positions)
    estimator = fold.model.estimator
    assert isinstance(estimator[-1], SVC if name == 'SVC' else SVR)
    assert estimator[-1].C in (0.5, 1.0)
    assert estimator[-2].n_samples_seen_ == len(fold.fit_positions)
    assert np.isfinite(fold.predictions['predict']).all().all()
    if name == 'SVC':
        assert fold.training_summary['class_balance']
        proba = fold.predictions['predict_proba']  # Automatically retained, no explicit prediction_methods.
        np.testing.assert_allclose(proba.sum(axis=1), 1., atol=1e-12)
        assert list(proba.columns.get_level_values('class')) == list(estimator.classes_)
        assert set(estimator.classes_) == set(prepared.dataset.y.iloc[fold.fit_positions, 0])
    else:
        assert 'predict_proba' not in fold.predictions
    save_model(result.training, tmp_path / 'svm-model')
    restored = load_model(tmp_path / 'svm-model')
    subset = replace(prepared.dataset, **{
        key: getattr(prepared.dataset, key).iloc[fold.test_positions].copy() for key in ('X', 'y', 'metadata')})
    for method, predictions in restored.predict(subset).items():
        pd.testing.assert_frame_equal(predictions.reset_index(drop=True), fold.predictions[method].reset_index(drop=True))


@pytest.mark.parametrize('pipeline', [False, True])
def test_svc_fold_weights_match_direct_sklearn(pipeline):
    context = weight_context()
    context.sample_weight = ClassWeightPolicy(mode='balanced').compute(context).values
    def make():
        model = SVC(C=0.7, probability=True, random_state=17)
        return Pipeline([('scale', StandardScaler()), ('classifier', model)]) if pipeline else model
    actual = EstimatorAdapter(make(), ('predict', 'predict_proba'))
    actual.fit(context)
    native = make()
    key = 'classifier__sample_weight' if pipeline else 'sample_weight'
    native.fit(context.X, context.y.corners, **{key: context.sample_weight.to_numpy()})
    np.testing.assert_allclose(actual.estimator.predict_proba(context.X), native.predict_proba(context.X), atol=1e-12)
    np.testing.assert_array_equal(actual.estimator.predict(context.X), native.predict(context.X))


def test_svc_without_probabilities_and_double_weighting_fail_explicitly():
    context = weight_context()
    with pytest.raises((TypeError, ValueError), match='predict_proba'):
        EstimatorAdapter(SVC(), ('predict_proba',)).fit(context)
    context.sample_weight = ClassWeightPolicy().compute(context).values
    with pytest.raises(ValueError, match='class_weight'):
        EstimatorAdapter(SVC(class_weight='balanced')).fit(context)


def test_svc_chronological_calibration_and_reuse(ui_recipe):
    ui_recipe['model'] = node('sklearn.svm.SVC', probability=True, random_state=17)
    ui_recipe['labels']['corners'] = node('labels.Above', source=ui_recipe['labels']['corners'], threshold=7)
    ui_recipe['candidate']['calibration'] = node('training.ProbabilityCalibrator', method='sigmoid', fraction=0.25)
    result = run_recipe(ui_recipe)
    fold = result.training.folds[0]
    populations = [set(fold.fit_positions), set(fold.calibration_positions), set(fold.test_positions)]
    assert all(populations)
    assert sum(map(len, populations)) == len(set.union(*populations))
    assert {'predict_proba_raw', 'predict_proba'} <= set(fold.predictions)
    np.testing.assert_allclose(fold.predictions['predict_proba'].sum(axis=1), 1., atol=1e-12)
    reused = run_recipe(ui_recipe)
    assert reused.reused
    pd.testing.assert_frame_equal(reused.training.folds[0].predictions['predict_proba'], fold.predictions['predict_proba'])


def test_svc_balanced_point_only_recipe(ui_recipe):
    ui_recipe['model'] = node('sklearn.svm.SVC', class_weight='balanced')
    result = run_recipe(ui_recipe)
    fold = result.training.folds[0]
    assert set(fold.predictions) == {'predict'}
    native = clone(fold.model.estimator)
    native.fit(result.dataset.X.iloc[fold.fit_positions], result.dataset.y.iloc[fold.fit_positions, 0])
    np.testing.assert_array_equal(native.predict(result.dataset.X.iloc[fold.test_positions]), fold.predictions['predict'].iloc[:, 0])


def test_svr_label_scaling_returns_original_units(ui_recipe):
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.impute import SimpleImputer
    ui_recipe['model'] = node('sklearn.svm.SVR', C=0.7, epsilon=0.2)
    ui_recipe['target_transformer'] = node('sklearn.preprocessing.StandardScaler')
    result = run_recipe(ui_recipe)
    fold = result.training.folds[0]
    native = TransformedTargetRegressor(
        regressor=Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler()),
                            ('model', SVR(C=0.7, epsilon=0.2))]), transformer=StandardScaler())
    native.fit(result.dataset.X.iloc[fold.fit_positions], result.dataset.y.iloc[fold.fit_positions, 0])
    np.testing.assert_allclose(native.predict(result.dataset.X.iloc[fold.test_positions]),
                               fold.predictions['predict'].iloc[:, 0], atol=1e-12)
