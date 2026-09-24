"""Target scaler families preserve fitting scope, units and saved predictions."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from test_target_scaling import scaling_data, recipe_for
from xdiyo_analytics.reporting import CoefficientReporter
from xdiyo_analytics.training import TrainingRunner, save_model, load_model
from xdiyo_analytics.ui.recipe import ModelFactory, catalog_for_ui, node


CASES = [
    ('StandardScaler', {}, True),
    ('MinMaxScaler', {'feature_range': [-2, 3]}, True),
    ('MaxAbsScaler', {}, True),
    ('RobustScaler', {'quantile_range': [10, 90], 'unit_variance': True}, True),
    ('RobustScaler', {'with_centering': False, 'with_scaling': False}, True),
    ('QuantileTransformer', {'n_quantiles': 6, 'output_distribution': 'uniform', 'random_state': 7}, False),
    ('QuantileTransformer', {'n_quantiles': 6, 'output_distribution': 'normal', 'random_state': 7}, False),
    ('PowerTransformer', {'method': 'yeo-johnson', 'standardize': True}, False),
    ('PowerTransformer', {'method': 'box-cox', 'standardize': False}, False),
]


def multitarget_data():
    data, plan = scaling_data()
    x = np.r_[np.arange(10.), np.arange(3., 9.)]
    data.X = pd.DataFrame({'signal': x}, index=data.X.index)
    y = np.column_stack([2 + (x + 1) ** 1.4, 80 + (x + 1) ** .7])
    y[10:] += 10000  # Fitting on evaluation labels would drastically change every transform.
    data.y = pd.DataFrame(y, index=data.X.index, columns=['corners', 'other'])
    return data, plan


@pytest.mark.parametrize('name,options,affine', CASES)
def test_family_multitarget_manual_fit_inverse_and_persistence(name, options, affine, tmp_path):
    data, plan = multitarget_data()
    catalog = catalog_for_ui()
    transformer_spec = node('targets.' + name, **options)
    recipe = recipe_for(node('sklearn.linear_model.LinearRegression'))
    recipe['target_transformer'] = transformer_spec
    fitted = TrainingRunner(ModelFactory(recipe, catalog)).run(data, plan)
    fold, adapter = fitted.folds[0], fitted.folds[0].model
    manual = TransformedTargetRegressor(
        regressor=Pipeline([('scale', StandardScaler()), ('model', LinearRegression())]),
        transformer=catalog.build(transformer_spec))
    manual.fit(data.X.iloc[:10], data.y.iloc[:10])
    np.testing.assert_allclose(fold.predictions['predict'], manual.predict(data.X.iloc[10:]), rtol=1e-9, atol=1e-8)
    np.testing.assert_allclose(adapter.transformer_.transform(data.y.iloc[:10]),
                               manual.transformer_.transform(data.y.iloc[:10].to_numpy()), atol=1e-10)
    contaminated = catalog.build(transformer_spec).fit(data.y)
    if options.get('with_centering', True) or options.get('with_scaling', True):
        assert not np.allclose(adapter.transformer_.transform(data.y.iloc[:10]),
                               contaminated.transform(data.y.iloc[:10]))
    pd.testing.assert_frame_equal(fold.y_true.reset_index(drop=True), data.y.iloc[10:].reset_index(drop=True))

    table = adapter.coefficient_table()
    assert set(table.coefficient_units) == {'original_target' if affine else 'transformed_target'}
    assert set(table.target_transformer) == {name}
    z = adapter.estimator.estimator[:-1].transform(data.X.iloc[10:])
    for column in data.y:
        terms = table.loc[table.target.eq(column)]
        linear = z @ terms.loc[terms.term.eq('feature'), 'coefficient'].to_numpy()
        linear += terms.loc[terms.term.eq('intercept'), 'coefficient'].iloc[0]
        expected = (fold.predictions['predict'][column] if affine else
                    adapter.estimator.estimator.predict(data.X.iloc[10:])[:, data.y.columns.get_loc(column)])
        np.testing.assert_allclose(linear, expected, atol=1e-8)

    if not affine:
        report = CoefficientReporter(type='overall').run(SimpleNamespace(partition='model', models={0: adapter}))
        assert any('transformed' in note.lower() for note in report.notes)
        assert len(report.tables['coefficients']) == len(table)

    path = save_model(fitted, tmp_path / name)
    loaded = load_model(path)
    heldout = replace(data, X=data.X.iloc[10:], y=data.y.iloc[10:], metadata=data.metadata.iloc[10:])
    np.testing.assert_allclose(loaded.predict(heldout)['predict'], fold.predictions['predict'])


@pytest.mark.parametrize('name', ['StandardScaler', 'MinMaxScaler', 'MaxAbsScaler', 'RobustScaler', 'QuantileTransformer'])
def test_constant_targets_roundtrip_without_division_by_zero(name, tmp_path):
    data, plan = scaling_data()
    data.y.loc[:, 'target'] = 12.
    recipe = recipe_for(node('sklearn.linear_model.LinearRegression'))
    recipe['target_transformer'] = node('targets.' + name, **({'n_quantiles': 5} if name == 'QuantileTransformer' else {}))
    result = TrainingRunner(ModelFactory(recipe, catalog_for_ui())).run(data, plan)
    np.testing.assert_allclose(result.folds[0].predictions['predict'], 12.)


@pytest.mark.parametrize('distribution', ['uniform', 'normal'])
def test_quantile_inverse_clips_to_fitted_label_range(distribution):
    catalog = catalog_for_ui()
    transformer = catalog.build(node('targets.QuantileTransformer', n_quantiles=5, output_distribution=distribution))
    transformer.fit(np.arange(10., 60., 10.).reshape(-1, 1))
    np.testing.assert_allclose(transformer.inverse_transform([[-100.], [100.]]).ravel(), [10., 50.])


def test_box_cox_rejects_zero_labels_without_silent_offset():
    data, plan = scaling_data()
    data.y.iloc[0, 0] = 0
    recipe = recipe_for()
    recipe['target_transformer'] = node('targets.PowerTransformer', method='box-cox')
    with pytest.raises(ValueError, match='strictly positive'):
        TrainingRunner(ModelFactory(recipe, catalog_for_ui())).run(data, plan)


@pytest.mark.parametrize('name', sorted({case[0] for case in CASES}))
def test_copy_false_preserves_original_labels(name):
    data, plan = multitarget_data()
    before = data.y.copy(deep=True)
    recipe = recipe_for(node('sklearn.linear_model.LinearRegression'))
    recipe['target_transformer'] = node('targets.' + name, copy=False,
        **({'n_quantiles': 5} if name == 'QuantileTransformer' else {}))
    result = TrainingRunner(ModelFactory(recipe, catalog_for_ui())).run(data, plan)
    pd.testing.assert_frame_equal(data.y, before)
    pd.testing.assert_frame_equal(result.folds[0].y_true.reset_index(drop=True), before.iloc[10:].reset_index(drop=True))


@pytest.mark.parametrize('copy', [True, False])
def test_power_inverse_domain_error_preserves_internal_prediction_frame(copy, monkeypatch):
    data, plan = multitarget_data()
    recipe = recipe_for(node('sklearn.linear_model.LinearRegression'))
    recipe['target_transformer'] = node('targets.PowerTransformer', method='box-cox', standardize=False, copy=copy)
    result = TrainingRunner(ModelFactory(recipe, catalog_for_ui())).run(data, plan)
    adapter = result.folds[0].model
    adapter.transformer_.lambdas_[:] = .3  # Valid power, inverse requires 1 + .3*u > 0.
    internal = pd.DataFrame(-100., index=data.y.iloc[10:].index, columns=data.y.columns)
    before = internal.copy(deep=True)
    monkeypatch.setattr(adapter.estimator, 'predict', lambda _: {'predict': internal})
    with np.errstate(invalid='ignore'), pytest.raises(ValueError, match='Inverse target transform produced non-finite'):
        adapter.predict(SimpleNamespace())
    pd.testing.assert_frame_equal(internal, before)


def test_catalog_and_maintained_dropdowns_expose_both_transform_families():
    from xdiyo_analytics.ui import build_inventory
    inventory = json.loads(Path(build_inventory.__file__).with_name('inventory.json').read_text(encoding='utf-8'))
    components = inventory['components']
    for name, _, _ in CASES:
        assert components['targets.' + name]['category'] == 'target_transformer'
        assert components['sklearn.preprocessing.' + name]['category'] == 'preprocessor'
    for prefix in ['targets.', 'sklearn.preprocessing.']:
        quantile = {field['name']: field for field in components[prefix + 'QuantileTransformer']['fields']}
        power = {field['name']: field for field in components[prefix + 'PowerTransformer']['fields']}
        assert set(quantile['output_distribution']['choices']) == {'uniform', 'normal'}
        assert set(power['method']['choices']) == {'yeo-johnson', 'box-cox'}
