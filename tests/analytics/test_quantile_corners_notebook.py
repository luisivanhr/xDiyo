"""Numerical and row-identity checks for the broad-feature notebook helper."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_pinball_loss

from notebooks.helpers import quantile_corners as helper
from xdiyo_analytics.datasets import ModelDataset
from xdiyo_analytics.experiments import PreparedExperiment
from xdiyo_analytics.reporting import TeamCatalog
from xdiyo_analytics.splits import Fold, SplitPlan


def test_pinball_matches_independent_sklearn_reference_and_retains_crossings():
    y = np.array([2., 7., 11.])
    predicted = np.array([[1., 2., 3.], [5., 4., 9.], [8., 12., 10.]])
    original = predicted.copy()
    quantiles = (.1, .5, .9)
    baseline = np.array([4., 6., 9.])
    scores, interval = helper.quantile_diagnostics(y, predicted, quantiles, baseline)
    for i, alpha in enumerate(quantiles):
        assert scores.iloc[i].pinball_loss == pytest.approx(
            mean_pinball_loss(y, predicted[:, i], alpha=alpha))
        assert scores.iloc[i].baseline_pinball == pytest.approx(
            mean_pinball_loss(y, np.repeat(baseline[i], len(y)), alpha=alpha))
    assert interval.iloc[0].coverage == pytest.approx(2 / 3)
    assert interval.iloc[0].mean_width == pytest.approx(8 / 3)
    assert interval.iloc[0].quantile_crossing_fraction == pytest.approx(2 / 3)
    assert interval.iloc[0].nominal_coverage == pytest.approx(.8)
    np.testing.assert_array_equal(predicted, original)


@pytest.mark.parametrize('predictions,quantiles', [
    (np.zeros((3, 2)), (.1, .5, .9)),
    (np.zeros((3, 3)), (.1, .9, .5)),
    (np.zeros((3, 3)), (.1, .5, .5)),
])
def test_diagnostics_rejects_misaligned_shapes_or_quantile_order(predictions, quantiles):
    with pytest.raises(ValueError, match='align'):
        helper.quantile_diagnostics(np.ones(3), predictions, quantiles, [1, 1, 1])


def tiny_bundle():
    rows = np.arange(24)
    x = pd.DataFrame({'signal': rows % 5, 'pace': np.sin(rows / 3)})
    y = pd.DataFrame({'total_corners': 5. + (rows % 5) + (rows % 3)})
    metadata = pd.DataFrame({
        'competition_id': 17, 'season_id': 2025, 'event_id': rows + 100,
        'home_id': 1, 'away_id': 2, 'home_name': 'Home', 'away_name': 'Away',
        'source_league': 'Synthetic', 'source_season': '24_25', 'round': rows + 1,
        'kickoff_at': pd.date_range('2025-01-01', periods=24, tz='UTC'),
    })
    keys = ('competition_id', 'season_id', 'event_id')
    dataset = ModelDataset(x, y, metadata, 'match', keys, keys, 'total')
    plan = SplitPlan([Fold(rows[:16], rows[16:], rows[16:])], len(rows), rows)
    prepared = PreparedExperiment(dataset, plan, config={'synthetic': True})
    manifest = pd.DataFrame({'feature': x.columns, 'training_nonmissing_fraction': 1.,
                             'training_unique_values': x.iloc[:16].nunique().values})
    return helper.FeatureBundle(prepared, pd.DataFrame(), {}, manifest, pd.DataFrame(),
                                TeamCatalog.from_matches(metadata), 0.)


def test_quantile_collection_rejects_differently_ordered_match_rows(monkeypatch, tmp_path):
    bundle = tiny_bundle()
    rows = bundle.prepared.split_plan.folds[0].test
    responses = []
    for i, order in enumerate((rows, rows[::-1])):
        frame = pd.DataFrame({'total_corners': 5.}, index=order)
        fit = SimpleNamespace(predictions={'predict': frame}, score_positions=order, model=None)
        responses.append(SimpleNamespace(training=SimpleNamespace(folds=[fit]),
                                         path=tmp_path / str(i), reused=False))

    class FakeExperiment:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            return responses.pop(0)

    monkeypatch.setattr(helper, 'FootballExperiment', FakeExperiment)
    with pytest.raises(ValueError, match='row identities/order'):
        helper.run_quantiles(bundle, tmp_path, quantiles=(.1, .9), device='cpu')


@pytest.mark.parametrize('early_stopping_rounds', [3, None])
def test_small_native_cpu_run_uses_fit_only_baseline_and_exact_test_rows(tmp_path, early_stopping_rounds):
    pytest.importorskip('xgboost')
    bundle = tiny_bundle()
    run = helper.run_quantiles(
        bundle, tmp_path, quantiles=(.1, .5, .9), device='cpu', threads=1,
        model_options={'n_estimators': 15, 'max_depth': 2, 'min_child_weight': 1,
                       'early_stopping_rounds': early_stopping_rounds, 'learning_rate': .1},
        validation_fraction=.25, reuse=True,
    )
    first = run.results[.1].training.folds[0]
    assert not set(first.fit_positions) & set(first.validation_positions)
    assert not set(first.train_positions) & set(first.test_positions)
    assert run.predictions.index.tolist() == list(range(16, 24))
    assert run.predictions.event_id.tolist() == list(range(116, 124))
    info = json.loads((run.output_path / 'summary.json').read_text())
    expected = np.quantile(bundle.prepared.dataset.y.iloc[first.fit_positions, 0], [.1, .5, .9])
    np.testing.assert_allclose(info['baseline_quantiles'], expected)
    for i, alpha in enumerate((.1, .5, .9)):
        assert run.metrics.iloc[i].pinball_loss == pytest.approx(mean_pinball_loss(
            run.predictions.observed_total_corners, run.predictions[f'q{alpha:g}'], alpha=alpha))
        assert (run.results[alpha].path / 'fitted_model').is_dir()
        fitted = run.results[alpha].training.folds[0]
        native = fitted.model._native()
        booster = native.get_booster()
        detail = info['fitted_models'][str(alpha)]
        if early_stopping_rounds is None:
            assert detail['best_iteration'] is None
            assert detail['best_validation_pinball'] is None
            prediction_booster = booster
        else:
            assert detail['prediction_rounds'] == native.best_iteration + 1
            prediction_booster = booster[:native.best_iteration + 1]
        expected_gain = {fitted.feature_columns[int(key[1:])]: value for key, value in
                         prediction_booster.get_score(importance_type='gain').items()}
        actual_gain = run.importance.loc[run.importance['quantile'].eq(alpha)].set_index('feature').gain.to_dict()
        assert actual_gain == pytest.approx(expected_gain)
    assert len(run.predictions) == 8
