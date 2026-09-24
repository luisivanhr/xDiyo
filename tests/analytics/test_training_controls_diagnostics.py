"""Numerical coefficient oracles and retained-history model-only reporting."""
from copy import deepcopy
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, ElasticNet, Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from xdiyo_analytics.analysis import PostTrainingAnalysis
from xdiyo_analytics.reporting import LearningCurveReporter, CoefficientReporter
from xdiyo_analytics.training import EstimatorAdapter, FitContext
from training_controls_samples import model_result


def fitted_adapter(estimator, *, classification=None, multi=False):
    x = np.array([-1, -1, 1, 1] * 2, dtype=float)
    z = np.array([-1, 1, -1, 1] * 2, dtype=float)
    X = pd.DataFrame({'x1': x, 'x2': z, 'zero': 0.})
    y = pd.DataFrame({'target': 3 + 4*x - 2*z})
    if multi: y['second'] = -1 + 2*x + z
    if classification == 'binary': y['target'] = np.where(x > 0, 'top', 'bottom')
    if classification == 'multi': y['target'] = ['a', 'b', 'c', 'c'] * 2
    context = FitContext(X=X, y=y, metadata=pd.DataFrame({'event_id': range(8)}),
                         layout='match', match_columns=('event_id',), fold_id=3)
    adapter = EstimatorAdapter(estimator); adapter.fit(context)
    return adapter, context


def report(models, **kwargs):
    return PostTrainingAnalysis({'coefficients': CoefficientReporter(type='overall', **kwargs)}).run(model_result(models)).studies[0]


@pytest.mark.parametrize('model,expected', [(Lasso(alpha=.5, tol=1e-12), [3.5, -1.5, 0]),
    (ElasticNet(alpha=1., l1_ratio=.5, tol=1e-12), [7/3, -1., 0])])
def test_penalized_coefficients_match_orthogonal_closed_form_and_keep_all_terms(model, expected):
    adapter, _ = fitted_adapter(model)
    study = report([adapter])
    result = study.result
    all_terms = result.tables['coefficients']
    np.testing.assert_allclose(all_terms.loc[all_terms.term.eq('feature'), 'coefficient'], expected, atol=1e-12)
    assert result.tables['intercepts'].coefficient.tolist() == [3.]
    assert result.tables['surviving_coefficients'].feature.tolist() == ['x1', 'x2']
    figure = result.artifacts[0].data
    assert list(figure.data[0].y) == ['x2', 'x1']
    np.testing.assert_allclose(figure.data[0].x, [expected[1], expected[0]])
    assert list(figure.data[0].marker.color) == ['#ed7975', '#58c7b2']
    assert study.partition == 'model' and study.scope.empty and study.n_matches == 0
    assert study.scope_label == 'fitted model diagnostics'


def test_pipeline_transformed_names_and_order_are_retained():
    transformer = ColumnTransformer([('scaled', StandardScaler(), ['x2', 'x1'])])
    adapter, _ = fitted_adapter(make_pipeline(transformer, Lasso(alpha=.5, tol=1e-12)))
    table = report([adapter]).result.tables['coefficients']
    terms = table.loc[table.term.eq('feature')]
    assert terms.feature.tolist() == ['scaled__x2', 'scaled__x1']
    np.testing.assert_allclose(terms.coefficient, [-1.5, 3.5], atol=1e-12)


def test_coefficients_remain_in_fitted_scaled_units():
    _, context = fitted_adapter(Ridge())
    context.X['x1'] *= 10
    adapter = EstimatorAdapter(make_pipeline(StandardScaler(), Lasso(alpha=.5, tol=1e-12)))
    adapter.fit(context)
    terms = report([adapter]).result.tables['coefficients'].query("feature == 'x1'")
    np.testing.assert_allclose(terms.coefficient, [3.5], atol=1e-12)


@pytest.mark.parametrize('classes', ['binary', 'multi'])
def test_classifier_coefficient_rows_retain_their_fitted_class_meaning(classes):
    adapter, _ = fitted_adapter(LogisticRegression(C=2), classification=classes)
    table = report([adapter]).result.tables['coefficients']
    labels = ['top'] if classes == 'binary' else ['a', 'b', 'c']
    assert table.label.drop_duplicates().tolist() == labels
    for i, label in enumerate(labels):
        terms = table.loc[table.label.eq(label) & table.term.eq('feature')]
        np.testing.assert_array_equal(terms.coefficient, adapter.estimator.coef_[i])
        assert table.loc[table.label.eq(label) & table.term.eq('intercept'), 'coefficient'].tolist() == [adapter.estimator.intercept_[i]]


def test_multioutput_ridge_keeps_target_rows_and_intercepts():
    adapter, _ = fitted_adapter(Ridge(alpha=2), multi=True)
    table = report([adapter]).result.tables['coefficients']
    for target, expected, intercept in [('target', [3.2, -1.6, 0], 3), ('second', [1.6, .8, 0], -1)]:
        terms = table.loc[table.target.eq(target) & table.term.eq('feature')]
        np.testing.assert_allclose(terms.coefficient, expected, atol=1e-12)
        assert table.loc[table.target.eq(target) & table.term.eq('intercept'), 'coefficient'].tolist() == [intercept]


class TableAdapter:
    def __init__(self, table): self.table = table
    def coefficient_table(self): return self.table
    def fit(self, *args): raise AssertionError('Reporter fitted a model')
    def predict(self, *args): raise AssertionError('Reporter recomputed predictions')


def table(features):
    return pd.DataFrame([dict(target='target', label=None, feature=name, coefficient=value, term='feature')
                         for name, value in features.items()] +
                        [dict(target='target', label=None, feature='intercept', coefficient=2., term='intercept')])


def test_survival_uses_fold_denominator_and_counts_missing_features_as_zero():
    tables = [table({'a': 2, 'b': 0}), table({'a': .1, 'c': -4})]
    originals = deepcopy(tables)
    result = report([TableAdapter(t) for t in tables], tolerance=.1).result
    frequency = result.tables['survival_frequency'].set_index('feature')
    assert frequency.loc['a', ['folds_present', 'folds_survived', 'folds_inspected']].tolist() == [2, 1, 2]
    assert frequency.loc['c', ['folds_present', 'folds_survived', 'folds_inspected']].tolist() == [1, 1, 2]
    assert frequency.survival_frequency.to_dict() == {'a': .5, 'c': .5, 'b': 0.}
    for actual, original in zip(tables, originals): pd.testing.assert_frame_equal(actual, original)


def test_intercept_only_inspected_output_counts_in_survival_denominator():
    result = report([TableAdapter(table({'a': 2})), TableAdapter(table({}))]).result
    frequency = result.tables['survival_frequency'].iloc[0]
    assert frequency.folds_inspected == 2
    assert frequency.folds_present == 1 and frequency.survival_frequency == .5


@pytest.mark.parametrize('include', [False, True])
def test_all_zero_models_retain_numeric_terms_and_explicit_unavailability_note(include):
    result = report([TableAdapter(table({'a': 0, 'b': 0}))], include_zeros=include, survival_frequency=False).result
    assert len(result.tables['coefficients']) == 3 and result.tables['surviving_coefficients'].empty
    assert 'survival_frequency' not in result.tables
    assert any('No surviving' in note for note in result.notes)
    assert len([a for a in result.artifacts if a.kind == 'plotly']) == int(include)


@pytest.mark.parametrize('problem', ['duplicate', 'column', 'nonfinite', 'term'])
def test_invalid_custom_coefficient_tables_are_rejected(problem):
    data = table({'a': 1})
    if problem == 'duplicate': data = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    elif problem == 'column': data = data.drop(columns='feature')
    elif problem == 'nonfinite': data.loc[0, 'coefficient'] = np.nan
    else: data.loc[0, 'term'] = 'importance'
    with pytest.raises(ValueError): report([TableAdapter(data)])


def test_unsupported_estimator_is_not_fabricated_and_does_not_hide_valid_other_models():
    unsupported, _ = fitted_adapter(DummyRegressor())
    result = report([unsupported, TableAdapter(table({'a': 2}))]).result
    assert any('unavailable' in note for note in result.notes)
    assert result.tables['coefficients'].fold_id.unique().tolist() == [1]


def test_lasso_has_no_fabricated_loss_curve():
    adapter, _ = fitted_adapter(Lasso())
    training = model_result([adapter], histories=[adapter.training_history_], summaries=[adapter.training_summary_])
    result = PostTrainingAnalysis({'curves': LearningCurveReporter(type='overall')}).run(training).studies[0].result
    assert result.tables['history'].empty and adapter.training_summary_['history_source'] == 'unavailable'
    assert any('no iteration history' in note for note in result.notes)


def history(fold, attempt, values):
    return pd.DataFrame([dict(fold_id=fold, attempt=attempt, step=i+1, metric='train_loss', value=value, learning_rate=.1)
                         for i, value in enumerate(values)])


@pytest.mark.parametrize('mode', ['per_fold', 'overall'])
def test_learning_curves_separate_attempts_and_preserve_missing_gaps_without_predictions(mode):
    frames = [pd.concat([history(99, 0, [4, 3]), history(99, 1, [2, np.nan, 1])], ignore_index=True), history(88, 0, [6])]
    summaries = [{'selected_attempt': 1, 'attempts': [{'attempt': 0, 'steps': 2}, {'attempt': 1, 'steps': 3}]}, {'selected_attempt': 0}]
    training = model_result([object(), object()], histories=frames, summaries=summaries)
    original = deepcopy(training)
    report = PostTrainingAnalysis({'curves': LearningCurveReporter(type=mode)}).run(training, fold_ids=[1, 0])
    assert len(report.studies) == (2 if mode == 'per_fold' else 1)
    all_traces = [trace for study in report.studies for artifact in study.result.artifacts if artifact.kind == 'plotly' for trace in artifact.data.data]
    assert [len(trace.x) for trace in all_traces] == [1, 2, 3]
    assert all(trace.connectgaps is False for trace in all_traces)
    assert np.isnan(all_traces[-1].y[1])
    assert all(study.scope.empty and study.row_positions.size == 0 for study in report.studies)
    for actual, before in zip(training.folds, original.folds):
        pd.testing.assert_frame_equal(actual.training_history, before.training_history)
        assert actual.training_summary == before.training_summary


@pytest.mark.parametrize('attempts,expected', [('selected', [1]), ([0], [0]), (None, [0, 1])])
def test_learning_curve_filtering_is_explicit(attempts, expected):
    frame = pd.concat([history(0, 0, [4, 3]), history(0, 1, [3, 1])], ignore_index=True)
    training = model_result([object()], histories=[frame], summaries=[{'selected_attempt': 1}])
    result = PostTrainingAnalysis({'curves': LearningCurveReporter(type='overall', metrics='train_loss', attempts=attempts)}).run(training).studies[0].result
    assert result.tables['history'].attempt.unique().tolist() == expected


@pytest.mark.parametrize('reporter', [CoefficientReporter(type='timeline'), LearningCurveReporter(type='timeline'),
    CoefficientReporter(type='overall', partition='score'), LearningCurveReporter(type='overall', partition='score')])
def test_model_diagnostics_reject_wrong_execution_contract(reporter):
    with pytest.raises((ValueError, KeyError)):
        PostTrainingAnalysis({'invalid': reporter}).run(model_result([object()]))
