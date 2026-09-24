"""Independent numerical references for distributions and associations."""
from copy import deepcopy
from itertools import combinations
import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import matthews_corrcoef
from reporting_samples import context
from xdiyo_analytics.reporting import FeatureDistributionReporter, CorrelationAnalysis


@pytest.mark.parametrize('bandwidth', ['scott', 'silverman', .4, 1.2, None])
@pytest.mark.parametrize('bins', [3, [-2, 0, 1, 3, 9]])
def test_histogram_density_and_gaussian_kernel_formula(bandwidth, bins):
    raw = [-2., -1, 0, .5, 1, 2, 4, 8, np.nan, np.inf, -np.inf, 'invalid']
    ctx = context({'x': raw})
    before = deepcopy(ctx.X)
    result = FeatureDistributionReporter(type='overall', partition='all', bins=bins,
                                         bandwidth=bandwidth, grid_size=21).run(ctx)
    values = np.array(raw[:8])
    histogram = result.tables['histogram']
    edges = np.r_[histogram.left, histogram.right.iloc[-1]]
    expected_counts = [sum(left <= v < right or (i == len(edges) - 2 and v == right) for v in values)
                       for i, (left, right) in enumerate(zip(edges[:-1], edges[1:]))]
    assert histogram['count'].tolist() == expected_counts
    np.testing.assert_allclose(histogram.density * np.diff(edges), np.array(expected_counts) / len(values))
    assert np.dot(histogram.density, np.diff(edges)) == pytest.approx(1)
    curve = result.tables['kde']
    factor = len(values) ** (-1/5) if bandwidth in ('scott', None) else (
        (3*len(values)/4) ** (-1/5) if bandwidth == 'silverman' else bandwidth)
    h = factor * np.std(values, ddof=1)
    manual = np.exp(-.5 * ((curve.x.to_numpy()[:, None] - values) / h) ** 2).mean(axis=1) / (np.sqrt(2*np.pi)*h)
    np.testing.assert_allclose(curve.density, manual, rtol=3e-14, atol=1e-15)
    np.testing.assert_allclose(curve.density, stats.gaussian_kde(values, bw_method=bandwidth)(curve.x), rtol=3e-14)
    assert result.tables['summary'].iloc[0].to_dict() == dict(feature='x', rows=12, finite=8, excluded=4, kde_status='ok')
    pd.testing.assert_frame_equal(ctx.X, before)


@pytest.mark.parametrize('values,status,finite', [([np.nan, np.inf], 'no_finite_values', 0),
                                                 ([3, np.nan], 'insufficient_spread', 1),
                                                 ([3, 3, 3], 'insufficient_spread', 3),
                                                 ([], 'no_finite_values', 0)])
def test_kde_omits_undefined_scopes(values, status, finite):
    result = FeatureDistributionReporter(type='overall', partition='all').run(context({'x': values}))
    assert result.tables['summary'].iloc[0].kde_status == status
    assert result.tables['summary'].iloc[0].finite == finite
    assert result.tables['kde'].empty


@pytest.mark.parametrize('kwargs', [{'grid_size': 1}, {'grid_size': True}, {'grid_size': 2.5},
                                  {'bins': [0, 1]}, {'bins': [0, 1, 1, 8]},
                                  {'bins': [0, np.inf]}, {'bandwidth': -.5}, {'bandwidth': 0}])
def test_distribution_rejects_invalid_configuration(kwargs):
    with pytest.raises((ValueError, TypeError)):
        FeatureDistributionReporter(type='overall', partition='all', **kwargs).run(context({'x': [0., 1, 2, 3]}))


@pytest.mark.parametrize('bandwidth', [-.5, 0, np.inf, np.nan, True, np.bool_(False), 'unknown', lambda kernel: -1])
@pytest.mark.parametrize('values', [[], [2., 2.], [0., 1., 2.]])
def test_bandwidth_guard_also_precedes_empty_and_constant_paths(bandwidth, values):
    with pytest.raises(ValueError, match='bandwidth'):
        FeatureDistributionReporter(type='overall', partition='all', bandwidth=bandwidth).run(context({'x': values}))


def kendall_pairs(x, y):
    concordant = discordant = tied_x = tied_y = 0
    for i, j in combinations(range(len(x)), 2):
        dx, dy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
        if dx*dy > 0: concordant += 1
        elif dx*dy < 0: discordant += 1
        elif dx == 0 and dy != 0: tied_x += 1
        elif dy == 0 and dx != 0: tied_y += 1
    return (concordant-discordant)/np.sqrt((concordant+discordant+tied_x)*(concordant+discordant+tied_y))


@pytest.mark.parametrize('method', ['pearson', 'spearman', 'kendall'])
def test_pairwise_finite_coefficients_against_scipy_and_tie_oracle(method):
    ctx = context({'a': [1, 1, 4, 2, 7, np.inf, 9, np.nan],
                   'b': [2, 1, 1, 4, 6, 3, np.nan, 1], 'constant': [2]*8,
                   'absent': [np.nan]*8},
                  {'first': [3, 1, 1, 2, 6, 8, np.nan, 7], 'second': [1, 0, 0, 1, 1, 0, 1, 1]})
    result = CorrelationAnalysis(type='overall', partition='all', methods=[method]).run(ctx)
    for row in result.tables['coefficients'].itertuples():
        x, y = np.asarray(ctx.X[row.feature], float), np.asarray(ctx.y[row.target], float)
        keep = np.isfinite(x) & np.isfinite(y)
        x, y = x[keep], y[keep]
        assert row.n == len(x)
        if row.feature == 'constant':
            assert row.status == 'zero_spread' and pd.isna(row.value)
        elif row.feature == 'absent':
            assert row.status == 'insufficient_pairs' and pd.isna(row.value)
        else:
            fn = {'pearson': stats.pearsonr, 'spearman': stats.spearmanr, 'kendall': stats.kendalltau}[method]
            assert row.value == pytest.approx(fn(x, y).statistic, abs=2e-14)
            if method == 'kendall': assert row.value == pytest.approx(kendall_pairs(x, y))
            assert row.status == 'ok'


def test_percentiles_are_feature_ranks_and_pooling_is_sum_with_coverage():
    ctx = context({'a': [0, 1, 2, 3], 'minus': [0, -1, -2, -3],
                   'weak': [1, -1, -1, 1], 'missing': [np.nan]*4},
                  {'linear': [0, 1, 2, 3], 'curve': [1, -1, -1, 1]})
    result = CorrelationAnalysis(type='overall', partition='all', methods=['pearson', 'spearman']).run(ctx)
    cells = result.tables['coefficients']
    linear = cells[(cells.target == 'linear') & (cells.metric == 'pearson')].set_index('feature')
    assert linear.loc['a', 'percentile'] == pytest.approx(100*2.5/3)
    assert linear.loc['minus', 'percentile'] == pytest.approx(100*2.5/3)
    assert linear.loc['weak', 'percentile'] == pytest.approx(100/3)
    assert pd.isna(linear.loc['missing', 'percentile'])
    board = result.tables['leaderboard'].set_index('feature')
    for feature in ctx.X:
        valid = [abs(r.value) for r in cells.itertuples() if r.feature == feature and np.isfinite(r.value)]
        assert board.loc[feature, 'n_coefficients'] == len(valid)
        if valid: assert board.loc[feature, 'pooled_magnitude'] == pytest.approx(sum(valid))
        else: assert pd.isna(board.loc[feature, 'pooled_magnitude'])
    no_ranks = CorrelationAnalysis(type='overall', partition='all', percentile_ranks=False).run(ctx)
    assert 'percentile' not in no_ranks.tables['coefficients']
    assert not any('percentile' in name for name in no_ranks.tables['leaderboard'])


@pytest.mark.parametrize('x,y', [([0, 1, 0, 1, 1], [0, 1, 1, 0, 1]),
                               (['W', 'D', 'L', 'W', 'D', 'L'], ['W', 'L', 'D', 'W', 'D', 'D']),
                               ([0, 1, 2, 2, 1], [2, 0, 1, 2, 1])])
def test_explicit_mcc_matches_sklearn_for_binary_and_multiclass(x, y):
    result = CorrelationAnalysis(type='overall', partition='all', methods='mcc',
                                 categorical_features=['x'], categorical_targets=['y']).run(context({'x': x}, {'y': y}))
    row = result.tables['coefficients'].iloc[0]
    assert row.value == pytest.approx(matthews_corrcoef(y, x), abs=1e-14)
    assert row.n == len(x) and row.status == 'ok'


def test_mcc_requires_both_declarations_and_uses_strict_thresholds():
    ctx = context({'x': [0, 1, 2, 2, np.inf, np.nan]}, {'y': [0, 1, 1, 2, 1, 0]})
    missing = CorrelationAnalysis(type='overall', partition='all', methods='mcc',
                                  categorical_targets=['y']).run(ctx).tables['coefficients'].iloc[0]
    assert missing.status == 'requires_categorical_declaration_or_threshold' and missing.n == 0
    result = CorrelationAnalysis(type='overall', partition='all', methods='mcc',
                                 feature_thresholds={'x': 1}, target_thresholds={'y': 1}).run(ctx)
    row = result.tables['coefficients'].iloc[0]
    assert row.n == 4 and row.value == pytest.approx(matthews_corrcoef([0, 0, 0, 1], [0, 0, 1, 1]))
    for bound in [np.nan, np.inf]:
        with pytest.raises(ValueError, match='finite'):
            CorrelationAnalysis(type='overall', partition='all', methods='mcc',
                                feature_thresholds={'x': bound}, target_thresholds={'y': 1}).run(ctx)


@pytest.mark.parametrize('x,y,status', [([1, 1, 1], [0, 1, 0], 'zero_spread'),
                                       ([None, 1], [1, None], 'insufficient_pairs'),
                                       ([np.inf, 1, 0], [0, 1, 1], 'zero_spread')])
def test_declared_mcc_missing_pairs_and_undefined_denominator(x, y, status):
    row = CorrelationAnalysis(type='overall', partition='all', methods='mcc',
                              categorical_features=['x'], categorical_targets=['y']).run(context({'x': x}, {'y': y})).tables['coefficients'].iloc[0]
    assert row.status == status and pd.isna(row.value)


@pytest.mark.parametrize('kwargs', [{'methods': []}, {'methods': ['pearson', 'pearson']}, {'methods': 'rank'},
                                  {'features': []}, {'features': ['x', 'x']}, {'targets': 'absent'}])
def test_correlation_configuration_errors(kwargs):
    with pytest.raises((ValueError, KeyError)):
        CorrelationAnalysis(type='overall', partition='all', **kwargs).run(context({'x': [1., 2, 3]}))
