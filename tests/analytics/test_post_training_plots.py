"""Independent numerical outputs behind post-training diagnostic figures."""
import math
import numpy as np
import pandas as pd
import pytest
from xdiyo_analytics.analysis import PostTrainingAnalysis
from xdiyo_analytics.reporting import (ResidualAnalysisReporter, CalibrationReporter,
    PredictionTimelineReporter, PredictionDistributionReporter, LabelPredictionDistributionReporter)
from post_training_samples import context, run_sample


def numeric_context(truth=(0., 1., 2., 4.), prediction=(1., 2., 3., 5.)):
    return context(pd.DataFrame({'count': truth}), {'predict': pd.DataFrame({'count': prediction})})


def test_residual_tables_and_histogram_match_hand_values():
    ctx = numeric_context([0., 3., 5., np.nan, 8.], [1., 1., 4., 3., np.inf])
    result = ResidualAnalysisReporter(type='overall', partition='test', bins=[-2, 0, 1, 3]).run(ctx)
    pairs = result.tables['pairs::count']
    assert pairs.residual.tolist() == [-1., 2., 1.]
    assert pairs.row_position.tolist() == [0, 1, 2]
    assert result.tables['histogram::count']['count'].tolist() == [1, 0, 2]
    assert result.tables['coverage'].iloc[0].to_dict() == {'target': 'count', 'n': 3, 'n_missing': 2}
    assert len(result.artifacts) == 3


@pytest.mark.parametrize('bandwidth', ['scott', 'silverman', .45])
def test_kde_overlay_uses_independent_normalized_gaussian_mixtures(bandwidth):
    ctx = numeric_context()
    result = PredictionDistributionReporter(type='overall', partition='test', bandwidth=bandwidth, grid_size=129).run(ctx)
    table = result.tables['distribution::count']
    grids = []
    for name, values in [('Observed', np.array([0., 1., 2., 4.])), ('Predicted', np.array([1., 2., 3., 5.]))]:
        rows = table.loc[table.distribution.eq(name)]
        grids.append(rows.x.to_numpy())
        factor = len(values)**(-1/5) if bandwidth == 'scott' else (len(values)*.75)**(-1/5) if bandwidth == 'silverman' else bandwidth
        h = factor*values.std(ddof=1)
        expected = [sum(math.exp(-.5*((x-v)/h)**2) for v in values)/(len(values)*h*math.sqrt(2*math.pi)) for x in rows.x]
        np.testing.assert_allclose(rows.value, expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(*grids)
    assert [trace.type for trace in result.artifacts[0].data.data] == ['scatter', 'scatter']
    assert LabelPredictionDistributionReporter is PredictionDistributionReporter


def test_empty_and_constant_kde_markers():
    reporter = PredictionDistributionReporter(type='overall', partition='test')
    constant = reporter.run(numeric_context([2.], [3.]))
    assert constant.tables['coverage'].status.tolist() == ['constant_marker', 'constant_marker']
    assert constant.tables['distribution::count'].x.tolist() == [2., 3.]
    assert constant.tables['distribution::count'].value.isna().all()
    assert len(constant.artifacts[0].data.layout.shapes) == 2
    empty = reporter.run(numeric_context([np.nan], [1.]))
    assert empty.tables['distribution::count'].empty
    assert empty.tables['coverage'].status.eq('empty').all()


def test_ecdf_and_categorical_frequency_use_same_paired_population():
    ecdf = PredictionDistributionReporter(type='overall', partition='test', mode='ecdf').run(numeric_context([3., 1., 1., np.nan], [2., 0., 2., 9.]))
    table = ecdf.tables['distribution::count']
    assert table.x.tolist() == [1., 1., 3., 0., 2., 2.]
    assert table.value.tolist() == [1/3, 2/3, 1., 1/3, 2/3, 1.]
    frequency = PredictionDistributionReporter(type='overall', partition='test', mode='frequency').run(numeric_context(['a', 'a', 'b', None], ['a', 'b', 'c', 'd']))
    table = frequency.tables['distribution::count'].set_index(['distribution', 'x']).value
    assert table.loc['Observed'].to_dict() == {'a': 2/3, 'b': 1/3, 'c': 0.}
    assert table.loc['Predicted'].to_dict() == {'a': 1/3, 'b': 1/3, 'c': 1/3}


@pytest.mark.parametrize('strategy', ['uniform', 'quantile'])
def test_calibration_bins_reversed_class_order(strategy):
    prediction = pd.DataFrame([[0., 1.], [.25, .75], [.5, .5], [1., 0.], [np.nan, np.nan]],
                               columns=pd.MultiIndex.from_product([['result'], [1, 0]]))
    ctx = context(pd.DataFrame({'result': [0, 0, 1, 1, 1]}), {'predict_proba': prediction})
    result = CalibrationReporter(type='overall', partition='test', strategy=strategy, n_bins=2).run(ctx)
    table = result.tables['calibration']
    for label in [1, 0]:
        values = prediction[('result', label)].iloc[:4].tolist()
        threshold = .5 if strategy == 'uniform' else (.25+.5)/2 if label == 1 else (.5+.75)/2
        expected = []
        for keep_high in [False, True]:
            selected = [i for i, value in enumerate(values) if (value >= threshold) == keep_high]
            expected.append((len(selected), sum(values[i] for i in selected)/len(selected),
                             sum(int(ctx.y.iloc[i, 0] == label) for i in selected)/len(selected)))
        rows = table.loc[table.label.eq(label)]
        np.testing.assert_allclose(rows[['n', 'mean_probability', 'observed_frequency']], expected)
    assert result.tables['coverage'].iloc[0].to_dict() == {'target': 'result', 'n': 4, 'n_missing': 1}


def test_timeline_has_exact_ids_stable_order_and_visible_gaps():
    training = run_sample('team_match')
    training.folds[0].predictions['predict'].iloc[0, 0] = np.nan
    report = PostTrainingAnalysis({'timeline': PredictionTimelineReporter(type='timeline', partition='test', pooling='occurrences',
                                                                            targets='count', group_by='team_id')}).run(training)
    table = report.studies[0].result.tables['timeline::count']
    assert table.kickoff_at.is_monotonic_increasing
    assert table.team_id.tolist() == [2**63+31, 2**63+32]*6
    missing = table.loc[(table.fold_id == 4) & (table.row_position == 8)]
    assert missing.predicted.isna().all() and missing.residual.isna().all()
    assert all(trace.connectgaps is False for trace in report.studies[0].result.artifacts[0].data.data)


@pytest.mark.parametrize('kwargs', [{'mode': 'bad'}, {'grid_size': True}, {'grid_size': 1},
                                    {'bandwidth': 0}, {'bandwidth': -1}, {'bandwidth': np.inf}, {'bandwidth': True}])
def test_invalid_distribution_configuration(kwargs):
    with pytest.raises(ValueError):
        PredictionDistributionReporter(type='overall', partition='test', **kwargs).run(numeric_context())


@pytest.mark.parametrize('kwargs', [{'n_bins': 0}, {'n_bins': True}, {'strategy': 'bad'}, {'classes': ['missing']}])
def test_invalid_calibration_configuration(kwargs):
    prediction = pd.DataFrame([[.5, .5]], columns=pd.MultiIndex.from_product([['count'], [0, 1]]))
    ctx = context(pd.DataFrame({'count': [1]}), {'predict_proba': prediction})
    with pytest.raises(ValueError):
        CalibrationReporter(type='overall', partition='test', **kwargs).run(ctx)
