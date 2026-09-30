import numpy as np
import pandas as pd
from types import SimpleNamespace

from xdiyo_analytics.reporting import CountClassificationReporter, FeatureImportanceReporter, PredictionTimelineReporter


def test_count_classification_distinguishes_absent_class_and_missing_probability_row():
    index = pd.RangeIndex(3)
    probabilities = pd.DataFrame(
        [[.8, .2], [.2, .8], [np.nan, np.nan]],
        index=index, columns=pd.MultiIndex.from_tuples([('count', 0), ('count', 1)]))
    context = SimpleNamespace(
        y=pd.DataFrame({'count': [0, 2, 3]}, index=index),
        predictions={'predict': pd.DataFrame({'count': [0, 1, 2]}, index=index), 'predict_proba': probabilities})
    result = CountClassificationReporter(type='overall', partition='test').run(context)
    metric = result.tables['metrics'].iloc[0]
    assert metric.probability_rows == 2
    assert metric.probability_missing_rows == 1
    assert metric.unseen_count_rows == 2
    assert bool(metric.exact_log_loss_infinite) is True
    support = result.tables['class_support'].set_index('count_class')
    assert bool(support.loc[2, 'probability_column']) is False


def test_count_classification_uses_only_retained_training_mode_summary():
    index = pd.MultiIndex.from_tuples([(0, 0), (0, 1)], names=['fold_id', 'row_position'])
    probabilities = pd.DataFrame([[.8, .2], [.2, .8]], index=index,
        columns=pd.MultiIndex.from_tuples([('count', 0), ('count', 1)]))
    fold = SimpleNamespace(training_summary={'training_target_modes': {'count': 1}, 'training_target_mode_rows': 7})
    context = SimpleNamespace(y=pd.DataFrame({'count': [0, 1]}, index=index),
        predictions={'predict': pd.DataFrame({'count': [0, 1]}, index=index), 'predict_proba': probabilities},
        fold_results={0: fold}, pooling='last')
    metric = CountClassificationReporter(type='overall', partition='test').run(context).tables['metrics'].iloc[0]
    assert metric.baseline_status == 'available'
    assert metric.baseline_majority_count == 1
    assert metric.baseline_training_rows == 7


def test_count_classification_does_not_fabricate_mean_pool_baseline():
    index = pd.RangeIndex(2)
    probabilities = pd.DataFrame([[.8, .2], [.2, .8]], index=index,
        columns=pd.MultiIndex.from_tuples([('count', 0), ('count', 1)]))
    context = SimpleNamespace(y=pd.DataFrame({'count': [0, 1]}, index=index),
        predictions={'predict': pd.DataFrame({'count': [0, 1]}, index=index), 'predict_proba': probabilities},
        fold_results={}, pooling='mean')
    metric = CountClassificationReporter(type='overall', partition='test').run(context).tables['metrics'].iloc[0]
    assert metric.baseline_status == 'unavailable'


def test_prediction_timeline_max_points_only_limits_plot():
    index = pd.MultiIndex.from_tuples([(0, i) for i in range(5)], names=['fold_id', 'row_position'])
    context = SimpleNamespace(y=pd.DataFrame({'count': range(5)}, index=index),
        predictions={'predict': pd.DataFrame({'count': range(5)}, index=index)},
        metadata=pd.DataFrame({'kickoff_at': pd.date_range('2026-01-01', periods=5, freq='D')}, index=index),
        fold_id=0)
    result = PredictionTimelineReporter(type='timeline', partition='test', max_points=2).run(context)
    assert len(result.tables['timeline::count']) == 5
    assert len(result.artifacts[0].data.data[0].x) == 2


class _Booster:
    def get_score(self, importance_type='gain'):
        assert importance_type == 'gain'
        return {'f1': 3.0, 'f0': 5.0}


class _Model:
    feature_columns_ = ('wrong-fallback',)

    def _native(self):
        return self

    def get_booster(self):
        return _Booster()


def test_feature_importance_maps_native_indices_to_fold_feature_names():
    fold = SimpleNamespace(feature_columns=('home::corners', 'away::corners'))
    context = SimpleNamespace(partition='model', models={0: _Model()}, fold_results={0: fold})
    result = FeatureImportanceReporter(type='overall', top_k=1).run(context)
    assert result.tables['importance'].feature.tolist() == ['home::corners', 'away::corners']
    assert result.tables['top_k'].feature.tolist() == ['home::corners']


def test_feature_importance_uses_native_names_with_unused_trailing_feature():
    import xgboost as xgb
    model = xgb.XGBClassifier(n_estimators=2, max_depth=1, learning_rate=1.0,
                              tree_method='hist', eval_metric='logloss', random_state=1)
    x = pd.DataFrame({'first_named': [0., 0., 1., 1.], 'second_named': [0., 1., 0., 1.], 'unused_trailing': [0., 0., 0., 0.]})
    model.fit(x, [0, 1, 1, 0])
    adapter = SimpleNamespace(_native=lambda: model, feature_columns_=tuple(x.columns), preprocessing_=None)
    context = SimpleNamespace(partition='model', models={0: adapter}, fold_results={})
    result = FeatureImportanceReporter(type='overall', top_k=None).run(context)
    assert set(result.tables['importance'].feature) <= set(x.columns)
    assert 'unused_trailing' not in set(result.tables['importance'].feature)


def test_feature_importance_real_boosting_adapter_restores_human_names():
    import xgboost as xgb
    from xdiyo_analytics.training import FitContext
    from xdiyo_analytics.ui.adapters import BoostingAdapter
    x = pd.DataFrame({'human_first': [0., 0., 1., 1.], 'human_second': [0., 1., 0., 1.], 'unused_trailing': [0., 0., 0., 0.]})
    y = pd.DataFrame({'label': [0, 1, 1, 0]})
    adapter = BoostingAdapter(xgb.XGBClassifier(n_estimators=2, max_depth=1, learning_rate=1.0,
        tree_method='hist', eval_metric='logloss', random_state=1), prediction_methods=('predict',))
    adapter.fit(FitContext(X=x, metadata=pd.DataFrame(index=x.index), layout='match', match_columns=(), fold_id=0, y=y))
    context = SimpleNamespace(partition='model', models={0: adapter}, fold_results={})
    result = FeatureImportanceReporter(type='overall', top_k=None).run(context)
    assert set(result.tables['importance'].feature) <= set(x.columns)
    assert 'f0' not in set(result.tables['importance'].feature)
