"""Fit-local weights/count classes and full notebook helper on bounded inputs."""
from pathlib import Path
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from notebooks.helpers import classifier_corners as helper
from notebooks.helpers.quantile_corners import FeatureBundle
from model_selection_samples import sample
from xdiyo_analytics.experiments import PreparedExperiment
from xdiyo_analytics.splits import Fold, SplitPlan
from xdiyo_analytics.training import TrainingRunner, ValidationTail, load_model


def tiny_bundle():
    data = sample()
    data.y = pd.DataFrame({'total_corners': [3, 3, 7, 3, 7, 12, 3, 12, 12, 7, 20, 3, 7, 12, 20, 7]}, index=data.X.index)
    data.metadata['source_league'] = 'Synthetic'
    data.metadata['source_season'] = ['22_23'] * 6 + ['23_24'] * 4 + ['24_25'] * 6
    rows = np.arange(len(data.X))
    prepared = PreparedExperiment(data, SplitPlan([Fold(rows[:10], rows[10:], rows[10:])], len(rows), rows),
                                  config={'cutoff_hours': None})
    manifest = pd.DataFrame({'feature': data.X.columns, 'training_nonmissing_fraction': 1., 'training_unique_values': 10})
    return FeatureBundle(prepared, pd.DataFrame(), {'test': 'synthetic'}, manifest, pd.DataFrame(), None, 0.)


@pytest.mark.parametrize('power', [0., .5, 1.])
def test_balancing_formula_and_mean_one(power):
    labels = np.array([3] * 6 + [7] * 3 + [12])
    weights, table = helper.class_weights(labels, power)
    expected = np.array([(10 / (3 * {3: 6, 7: 3, 12: 1}[label])) ** power for label in labels])
    expected /= expected.mean()
    np.testing.assert_allclose(weights, expected)
    assert weights.mean() == pytest.approx(1.)
    assert table.count_class.tolist() == [3, 7, 12]


def test_weights_are_optimization_only_and_sparse_classes_decode(monkeypatch):
    from xgboost import XGBClassifier
    bundle = tiny_bundle()
    seen = {}
    original = XGBClassifier.fit
    def capture(self, X, y, **kwargs):
        seen['labels'] = np.asarray(y).copy()
        seen['weights'] = kwargs['sample_weight'].copy()
        return original(self, X, y, **kwargs)
    monkeypatch.setattr(XGBClassifier, 'fit', capture)
    options = dict(n_estimators=3, max_depth=2, min_child_weight=1, n_jobs=1, early_stopping_rounds=None)
    result = TrainingRunner(lambda: helper.model_factory(options, 1.), validation=ValidationTail(fraction=.2)).run(
        bundle.prepared.dataset, bundle.prepared.split_plan)
    fold = result.folds[0]
    expected, table = helper.class_weights(bundle.prepared.dataset.y.iloc[fold.fit_positions, 0], 1.)
    np.testing.assert_allclose(seen['weights'], expected)
    assert len(seen['weights']) == 8
    assert set(seen['labels']) == {0, 1, 2}
    pd.testing.assert_frame_equal(fold.model.class_balance_, table)
    assert set(fold.model.label_encoder_.classes_) == {3, 7, 12}
    assert set(fold.predictions['predict'].iloc[:, 0]) <= {3, 7, 12}
    assert set(fold.predictions['predict_proba'].columns.get_level_values(1)) == {3, 7, 12}
    assert 20 in set(fold.y_true.iloc[:, 0])
    np.testing.assert_allclose(fold.predictions['predict_proba'].sum(axis=1), 1., atol=1e-6)


def test_unseen_class_probability_metrics_keep_all_rows():
    index = pd.Index([20, 30, 40])
    truth = pd.Series([3, 20, 12], index=index)
    prediction = pd.Series([3, 7, 7], index=index)
    probabilities = pd.DataFrame([[.8, .1, .1], [.1, .8, .1], [.1, .7, .2]], index=index, columns=[3, 7, 12])
    rows, scores = helper.classification_diagnostics(truth, prediction, probabilities)
    assert scores.n.iloc[0] == 3
    assert scores.unseen_count_rows.iloc[0] == 1
    assert scores.exact_log_loss_infinite.iloc[0]
    assert scores.mae.iloc[0] == pytest.approx(6.)
    assert scores.log_loss_clipped_all.iloc[0] == pytest.approx(-np.log([.8, 1e-12, .2]).mean())
    assert rows.observed_total_corners.tolist() == [3, 20, 12]
    with pytest.raises(ValueError, match='identities'):
        helper.classification_diagnostics(truth.iloc[::-1], prediction, probabilities)


def test_grid_pipeline_save_reuse_and_numeric_tolerance_two(tmp_path):
    bundle = tiny_bundle()
    options = dict(n_estimators=3, max_depth=2, min_child_weight=1, max_bin=16)
    settings = dict(model_options=options, grid={'balance_power': [0., 1.]}, device='cpu', threads=1,
                    pre_analysis=False)
    result = helper.run_classifier(bundle, tmp_path / 'experiment', **settings)
    assert len(result.comparison) == 2
    assert result.metrics.unseen_count_rows.iloc[0] == 2
    assert len(result.predictions) == 6
    match = next(study for study in result.result.report.studies if study.name.endswith('Match predictions'))
    table = match.result.tables['matches::total_corners']
    # Report data keeps the arithmetic tolerance despite a classifier backend.
    assert table.status.eq('within tolerance').tolist() == (table.prediction - table.result).abs().le(2.).tolist()
    fold = result.result.training.folds[0]
    assert len(fold.feature_columns) == bundle.prepared.dataset.X.shape[1]
    loaded = load_model(result.result.path / 'fitted_model')
    restored = loaded.predict(result.result.dataset)
    np.testing.assert_array_equal(restored['predict'].loc[result.predictions.index].iloc[:, 0],
                                  result.predictions.predicted_total_corners)
    reused = helper.run_classifier(bundle, tmp_path / 'experiment', **settings)
    assert reused.result.reused
    pd.testing.assert_frame_equal(reused.probabilities, result.probabilities)


def test_default_features_and_grid_are_shared_with_notebook19():
    import ast
    import nbformat
    from notebooks.helpers import quantile_corners
    assert helper.prepare is quantile_corners.prepare
    assert len(helper.build_candidates()) == 24
    names = {'FEATURE_WINDOWS', 'FEATURE_LAGS', 'FEATURE_EMA_SPANS', 'FEATURE_PERIODS', 'INCLUDE_RATINGS', 'CUTOFF_HOURS'}
    def settings(path):
        notebook = nbformat.read(path, 4)
        nbformat.validate(notebook)
        result = {}
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                compile(cell.source, str(path), 'exec')
                for node in ast.parse(cell.source).body:
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names:
                        result[node.targets[0].id] = ast.literal_eval(node.value)
        return result
    assert settings(ROOT / 'notebooks/19_xgboost_quantile_total_corners.ipynb') == settings(ROOT / 'notebooks/20_xgboost_classifier_total_corners.ipynb')


def test_expanded_seasons_discovery_and_nested_boundaries():
    from notebooks.helpers import quantile_corners
    from xdiyo_analytics.splits import TemporalSplit, create_split_plan
    bundle = tiny_bundle()
    data = bundle.prepared.dataset
    seasons = ['20_21', '21_22', '22_23', '23_24', '24_25']
    data.metadata['source_season'] = [seasons[min(i // 3, 4)] for i in range(len(data.X))]
    matches = data.metadata.assign(kickoff_utc=data.metadata.kickoff_at.astype('int64'))
    development, holdout = quantile_corners.season_populations(matches.iloc[::-1])
    assert development == tuple(seasons[:-1])
    assert holdout == '24_25'
    stats = pd.DataFrame({'source_season': seasons, 'period': 'ALL', 'group_name': 'Match overview',
                          'key': ['corners', 'corners', 'third_season_only', 'fourth_season_only', 'test_only']})
    identities = quantile_corners.training_stat_identities(stats, development)
    assert set(identities.key) == {'corners', 'third_season_only', 'fourth_season_only'}
    outer = create_split_plan(data, TemporalSplit(train_size=len(development), unit='seasons',
                               calendar_by=(), block_by=('source_season',)))
    prepared = replace(bundle.prepared, split_plan=outer)
    assert len(outer.folds) == 1
    assert set(data.metadata.iloc[outer.folds[0].test].source_season) == {'24_25'}
    inner = helper.inner_plan(prepared)
    assert len(inner.folds) == 3
    assert [len(f.train) for f in inner.folds] == [3, 6, 9]
    for f in inner.folds:
        assert set(f.train) | set(f.test) <= set(outer.folds[0].train)
        assert not set(f.train) & set(f.test)
    custom = helper.inner_plan(prepared, TemporalSplit(train_size=2, unit='seasons', window='sliding',
                                calendar_by=(), block_by=('source_season',)))
    assert len(custom.folds) == 2
    assert [len(f.train) for f in custom.folds] == [6, 6]


@pytest.mark.parametrize('method,proportion', [('pearson', .4), ('spearman', .4), ('kendall', .4), ('spearman', 1.)])
def test_configured_selection_is_fit_local_with_requested_rounding(tmp_path, monkeypatch, method, proportion):
    from xdiyo_analytics.reporting import TopKCorrelationSelector
    bundle = tiny_bundle()
    data = bundle.prepared.dataset
    data.X = pd.DataFrame({'signal': data.y.iloc[:, 0], 'curve': np.arange(16) ** 2,
        'cycle': np.arange(16) % 4, 'constant': 1., 'missing': np.nan}, index=data.X.index)
    # Test-only association must never enter the selector's correlation population.
    data.X.loc[10:, 'curve'] = data.y.iloc[10:, 0] * 10000
    observed = []
    original = TopKCorrelationSelector.select
    def inspect(self, context):
        result = original(self, context)
        assert self.type == 'per_fold' and self.partition == 'train' and self.method == method
        scores = context.X.apply(lambda column: column.corr(context.y.iloc[:, 0], method=method)
                                 if column.nunique() > 1 else np.nan).abs()
        expected_count = int(np.ceil(proportion * len(context.X.columns)))
        expected = tuple(scores.dropna().sort_values(ascending=False, kind='stable').index[:expected_count])
        assert result.selection.columns == expected
        observed.append((set(context.metadata.event_id), expected))
        return result
    monkeypatch.setattr(TopKCorrelationSelector, 'select', inspect)
    result = helper.run_classifier(bundle, tmp_path, grid={'balance_power': [0.]},
        model_options={'n_estimators': 2, 'max_depth': 2, 'min_child_weight': 1},
        pre_analysis=False, threads=1, feature_selection_proportion=proportion,
        feature_selection_correlation=method)
    assert len(observed) == 2  # candidate inner fit, then independently fitted winner
    assert [rows for rows, _ in observed] == [set(data.metadata.event_id.iloc[:6]), set(data.metadata.event_id.iloc[:10])]
    assert result.result.training.folds[0].feature_columns == observed[-1][1]
    assert len(observed[-1][1]) == (3 if proportion == 1 else 2)
    import json
    summary = json.loads((result.output_path / 'summary.json').read_text())
    assert summary['feature_selection_proportion'] == proportion
    assert summary['feature_selection_correlation'] == method
    assert summary['selected_feature_count'] == len(observed[-1][1])
    assert pd.read_csv(result.output_path / 'selected_features.csv').feature.tolist() == list(observed[-1][1])


def test_selector_options_participate_in_candidate_identity():
    plain = helper.build_candidates(grid={})[0]
    assert plain.pre_analysis is None and plain.features_from is None
    pearson = helper.build_candidates(grid={}, feature_selection_proportion=.8, feature_selection_correlation='pearson')[0]
    spearman = helper.build_candidates(grid={}, feature_selection_proportion=.8)[0]
    assert pearson.config != spearman.config != plain.config
    assert spearman.pre_analysis.reporters[spearman.features_from].k == .8
    with pytest.raises(ValueError, match='proportion'):
        helper.build_candidates(feature_selection_proportion=0)
