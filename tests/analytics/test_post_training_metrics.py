"""Independent arithmetic and cohort checks for reusable metric calculations."""
import math
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from xdiyo_analytics.evaluation import Metric, evaluate_metrics, list_metrics, register_metric
from xdiyo_analytics.evaluation import metrics as module


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    monkeypatch.setattr(module, 'METRICS', dict(module.METRICS))


def frames(truth, prediction, *, classes=None):
    index = pd.MultiIndex.from_arrays([[7] * len(truth), np.arange(len(truth)) + 101],
                                      names=['fold_id', 'row_position'])
    y = pd.DataFrame({'target': truth}, index=index)
    columns = ['target'] if classes is None else pd.MultiIndex.from_product([['target'], classes], names=['target', 'class'])
    output = 'predict' if classes is None else 'predict_proba'
    return y, {output: pd.DataFrame(prediction, columns=columns, index=index)}


@pytest.mark.parametrize('name,expected', [
    ('mse', 3), ('mean_squared_error', 3), ('mae', 5/3),
    ('mean_absolute_error', 5/3), ('rmse', math.sqrt(3)), ('r2', 29/56),
])
def test_numeric_complete_case_arithmetic(name, expected):
    y, outputs = frames([1., 3., 7., np.nan, np.inf], [2., 1., 5., 4., 0.])
    actual = evaluate_metrics(y, outputs, [name]).iloc[0]
    assert actual.value == pytest.approx(expected)
    assert (actual.n, actual.n_total, actual.n_missing, actual.status) == (3, 5, 2, 'ok')
    assert actual.direction == ('maximize' if name == 'r2' else 'minimize')


@pytest.mark.parametrize('name,expected', [
    ('accuracy', 3/5), ('accuracy_score', 3/5), ('balanced_accuracy', 7/12),
    ('precision', 7/12), ('precision_score', 7/12), ('recall', 7/12),
    ('recall_score', 7/12), ('f1', 7/12), ('f1_score', 7/12), ('mcc', 1/6),
])
def test_classification_confusion_counts(name, expected):
    # TN=1, FP=1, FN=1, TP=2. Macro averages treat both classes equally.
    y, outputs = frames([0, 0, 1, 1, 1, pd.NA], [0, 1, 0, 1, 1, 0])
    actual = evaluate_metrics(y, outputs, [name]).iloc[0]
    assert actual.value == pytest.approx(expected)
    assert actual.n == 5 and actual.n_missing == 1 and actual.direction == 'maximize'


@pytest.mark.parametrize('name', ['precision', 'recall', 'f1'])
def test_binary_classification_is_explicit(name):
    y, outputs = frames(['no', 'no', 'yes', 'yes', 'yes'], ['no', 'yes', 'no', 'yes', 'yes'])
    spec = Metric(name, parameters={'average': 'binary', 'pos_label': 'yes'}, key=name+'_yes')
    actual = evaluate_metrics(y, outputs, [spec]).iloc[0]
    assert actual.value == pytest.approx(2/3)
    assert actual.metric == name+'_yes' and '"pos_label": "yes"' in actual.parameters


@pytest.mark.parametrize('name', ['log_loss', 'cross_entropy', 'binary_cross_entropy'])
@pytest.mark.parametrize('reverse', [False, True])
def test_log_losses_use_declared_class_identity(name, reverse):
    p = np.array([.9, .6, .2, 0.])
    values = np.column_stack([1-p, p])
    classes = [0, 1]
    if reverse:
        classes, values = classes[::-1], values[:, ::-1]
    y, outputs = frames([1, 0, 1, 0], values, classes=classes)
    actual = evaluate_metrics(y, outputs, [name]).iloc[0]
    assert actual.value == pytest.approx(-sum(map(math.log, [.9, .4, .2, 1.]))/4)
    assert actual.status == 'ok' and actual.direction == 'minimize'


@pytest.mark.parametrize('name,expected', [('brier_score', .2525), ('brier_score_loss', .2525), ('roc_auc', .75)])
def test_binary_probability_references(name, expected):
    y, outputs = frames([1, 0, 1, 0], [[.9, .1], [.6, .4], [.2, .8], [0., 1.]], classes=[1, 0])
    assert evaluate_metrics(y, outputs, [name]).iloc[0].value == pytest.approx(expected)


def test_entropy_uses_no_outcomes_and_has_no_default_direction():
    y, outputs = frames([pd.NA]*3, [[1., 0.], [.5, .5], [.2, .8]], classes=['yes', 'no'])
    expected = (math.log(2) - .2*math.log(.2) - .8*math.log(.8))/3
    first = evaluate_metrics(y, outputs, ['binary_entropy']).iloc[0]
    y['target'] = ['arbitrary', 'different', 'outcomes']
    second = evaluate_metrics(y, outputs, ['binary_entropy']).iloc[0]
    assert first.value == pytest.approx(expected)
    assert first.n == 3 and first.n_missing == 0 and first.direction is None
    assert first.sample_hash == second.sample_hash


def test_multiclass_loss_and_auc_are_invariant_to_output_class_order():
    classes = ['C', 'A', 'B']
    truth = ['A', 'B', 'C', 'A', 'B', 'C']
    values = np.array([[.1, .8, .1], [.2, .3, .5], [.6, .2, .2],
                       [.3, .4, .3], [.4, .5, .1], [.2, .2, .6]])
    y, outputs = frames(truth, values, classes=classes)
    loss = -np.mean([math.log(values[i, classes.index(label)]) for i, label in enumerate(truth)])
    aucs = []
    for j, label in enumerate(classes):
        positive = [values[i, j] for i, actual in enumerate(truth) if actual == label]
        negative = [values[i, j] for i, actual in enumerate(truth) if actual != label]
        aucs.append(sum((a > b) + .5*(a == b) for a in positive for b in negative)/(len(positive)*len(negative)))
    rows = evaluate_metrics(y, outputs, ['cross_entropy', 'roc_auc']).set_index('metric')
    assert rows.loc['cross_entropy', 'value'] == pytest.approx(loss)
    assert rows.loc['roc_auc', 'value'] == pytest.approx(np.mean(aucs))
    reordered = {'predict_proba': outputs['predict_proba'].iloc[:, ::-1]}
    assert_frame_equal(evaluate_metrics(y, reordered, ['cross_entropy', 'roc_auc']), rows.reset_index())


@pytest.mark.parametrize('name,truth,expected', [
    ('mse', [np.nan, np.nan], 'no_valid_observations'),
    ('r2', [1., 1.], 'undefined_constant_or_small_target'),
    ('r2', [1.], 'undefined_constant_or_small_target'),
])
def test_undefined_numeric_metrics_are_reported(name, truth, expected):
    y, outputs = frames(truth, [2.]*len(truth))
    row = evaluate_metrics(y, outputs, [name]).iloc[0]
    assert row.status == expected and math.isnan(row.value)


def test_empty_and_single_class_probability_coverage():
    y, outputs = frames([1, 1, 1], [[.2, .8], [np.nan, np.nan], [np.inf, -np.inf]], classes=[0, 1])
    row = evaluate_metrics(y, outputs, ['roc_auc']).iloc[0]
    assert (row.n, row.n_missing, row.status) == (1, 2, 'undefined_single_observed_class')
    empty = evaluate_metrics(y.iloc[:0], {key: value.iloc[:0] for key, value in outputs.items()}, ['cross_entropy'])
    assert empty.iloc[0].status == 'no_valid_observations'


@pytest.mark.parametrize('values,truth,classes,name,parameters', [
    ([[1.1, -.1]], [0], [0, 1], 'cross_entropy', {}),
    ([[.4, .4]], [0], [0, 1], 'cross_entropy', {}),
    ([[1.]], [0], [0], 'cross_entropy', {}),
    ([[.4, .6]], [2], [0, 1], 'cross_entropy', {}),
    ([[.4, .6]], [0], [0, 0], 'cross_entropy', {}),
    ([[.4, .6]], [0], [0, 1], 'cross_entropy', {'eps': 0}),
    ([[.4, .6]], [0], [0, 1], 'cross_entropy', {'eps': .5}),
    ([[.2, .3, .5]], [0], [0, 1, 2], 'binary_cross_entropy', {}),
    ([[.2, .3, .5]], [0], [0, 1, 2], 'binary_entropy', {}),
    ([[.4, .6]], [0], [0, 1], 'brier_score', {'positive_label': 'yes'}),
])
def test_malformed_probability_inputs_raise(values, truth, classes, name, parameters):
    y, outputs = frames(truth, values, classes=classes)
    with pytest.raises(ValueError):
        evaluate_metrics(y, outputs, [Metric(name, parameters=parameters)])


def test_zero_probability_clipping_and_explicit_epsilon():
    y, outputs = frames([1, 0], [[1., 0.], [0., 1.]], classes=[0, 1])
    value = evaluate_metrics(y, outputs, [Metric('cross_entropy', parameters={'eps': .01})]).iloc[0].value
    assert value == pytest.approx(-math.log(.01))


def test_sample_fingerprint_binds_actual_complete_cases_not_predictions():
    y, outputs = frames([1., 2., 3.], [1., 2., 3.])
    metadata = pd.DataFrame({'event_id': pd.array([2**63+1, 2**63+2, 2**63+3], dtype='uint64[pyarrow]')}, index=y.index)
    def digest():
        return evaluate_metrics(y, outputs, ['mse'], metadata=metadata).iloc[0].sample_hash
    original = digest()
    outputs['predict'] += 1
    assert digest() == original
    metadata.iloc[0, 0] = 2**63+4
    assert digest() != original
    metadata.iloc[0, 0] = 2**63+1
    y.iloc[0, 0] = 7.
    assert digest() != original
    y.iloc[0, 0] = 1.
    outputs['predict'].iloc[0, 0] = np.nan
    assert digest() != original


def test_custom_metrics_parameters_expansion_and_registry():
    register_metric('scaled_bias', lambda y, p, scale: scale*(p-y).mean(), kind='numeric')
    listed = list_metrics().set_index('name')
    assert listed.loc['scaled_bias', 'kind'] == 'numeric' and pd.isna(listed.loc['scaled_bias', 'direction'])
    y, outputs = frames([1., 3.], [2., 1.])
    y['second'] = y.target + 10
    outputs['predict']['second'] = y.second + 3
    result = evaluate_metrics(y, outputs, [Metric('scaled_bias', parameters={'scale': 2}, key='bias2', direction='maximize')])
    assert result.value.tolist() == [-1., 6.]
    assert result.target.tolist() == ['target', 'second'] and result.direction.eq('maximize').all()
    with pytest.raises(ValueError, match='exists'):
        register_metric('scaled_bias', lambda y, p: 0, kind='numeric')
    register_metric('scaled_bias', lambda y, p: 42, kind='numeric', replace=True)
    assert evaluate_metrics(y, outputs, ['scaled_bias']).value.tolist() == [42., 42.]


@pytest.mark.parametrize('kwargs', [{'name': '', 'function': lambda y, p: 0, 'kind': 'numeric'},
                                    {'name': 'x', 'function': 3, 'kind': 'numeric'},
                                    {'name': 'x', 'function': lambda y, p: 0, 'kind': 'bad'},
                                    {'name': 'x', 'function': lambda y, p: 0, 'kind': 'label', 'direction': 'down'}])
def test_bad_registration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        register_metric(**kwargs)


def test_metric_identity_shape_and_duplicate_request_guards():
    y, outputs = frames([0., 1.], [1., 2.])
    with pytest.raises(ValueError, match='distinct keys'):
        evaluate_metrics(y, outputs, ['mse', 'mse'])
    with pytest.raises(ValueError, match='direction'):
        evaluate_metrics(y, outputs, [Metric('mse', direction='down')])
    with pytest.raises(ValueError, match='metadata'):
        evaluate_metrics(y, outputs, ['mse'], metadata=pd.DataFrame(index=y.index[::-1]))
    with pytest.raises(ValueError, match='index/order'):
        evaluate_metrics(y, {'predict': outputs['predict'].iloc[::-1]}, ['mse'])
    register_metric('not_scalar', lambda y, p: [1, 2], kind='numeric')
    with pytest.raises(ValueError, match='scalar'):
        evaluate_metrics(y, outputs, ['not_scalar'])
    register_metric('not_finite', lambda y, p: np.inf, kind='numeric')
    assert evaluate_metrics(y, outputs, ['not_finite']).iloc[0].status == 'undefined'
