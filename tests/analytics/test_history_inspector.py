"""Native regression checks for cutoff-safe observation regime resets."""
import json

import numpy as np
import pandas as pd
import pytest

from test_ui_workflow import ui_recipe
from xdiyo_analytics.features import (
    EMA, H2H, ForAgainst, HistoryInspector, Lag, League, RollingMean,
    RollingWeightedMean, RollingZScore, Stat, Sum, evaluate_features,
)
from xdiyo_analytics.ui import catalog_for_ui, prepare_recipe
from xdiyo_analytics.ui.recipe import node, read_recipe

STAT = Stat('ALL', 'G', 'value')
OWN = 'own'
OTHER = 'other'
WEIGHT = Stat('1ST', 'G', 'value')


def observations(times, values, *, seasons=None, against=None):
    times = pd.to_datetime(times, utc=True, format='mixed')
    count = len(times)
    frame = pd.DataFrame(dict(kickoff_at=times, event_id=range(count),
        competition_id=100, team_id=70, opponent_id=71,
        season_id=[1] * count if seasons is None else seasons, side='home',
        status=['finished'] * (count - 1) + ['notstarted'], own=values))
    frame.attrs['stat_columns'] = {
        OWN: dict(role='team', period='ALL', group_name='G', key='value', field='value'),
    }
    if against is not None:
        frame[OTHER] = against
        frame.attrs['stat_columns'][OTHER] = dict(
            role='opponent', period='ALL', group_name='G', key='value', field='value')
    return frame


def evidence(frame, name, row, column=OWN):
    return next(record for record in frame.attrs['history_inspection'][name][0]['records']
                if record['event_id'] == row and record['source_column'] == column)


def test_outage_never_changes_fixture_history_and_ema_restarts_before_reduction():
    history = observations(['2019-08-01', '2019-09-01', '2020-08-01', '2020-08-10',
                            '2021-08-01', '2022-05-01', '2022-08-01', '2022-08-10',
                            '2022-08-12', '2022-08-15'],
                           [3., 5., None, None, None, None, 9., None, 11., 999.],
                           seasons=[1, 1, 2, 2, 3, 3, 4, 4, 4, 4])
    original = history.copy(deep=True)
    mean = RollingMean(STAT, 3)
    ema = EMA(STAT, span=3)
    result = evaluate_features(history, {
        'original': ema, 'guarded': HistoryInspector(ema), 'mean': HistoryInspector(mean),
        'lag2': HistoryInspector(Lag(STAT, periods=2)),
        'three': HistoryInspector(RollingMean(STAT, 3, min_periods=3)),
    })
    pd.testing.assert_frame_equal(history, original)
    assert len(result) == len(history)
    assert result.guarded.iloc[1] == 3
    assert result.guarded.iloc[2:7].isna().all()
    np.testing.assert_allclose(result.original.iloc[6:], [4., 6.5, 6.5, 8.75])
    np.testing.assert_allclose(result.guarded.iloc[7:], [9., 9., 10.])
    np.testing.assert_allclose(result['mean'].iloc[7:], [9., 9., 10.])
    np.testing.assert_allclose(result.lag2.iloc[7:], [np.nan, 9., np.nan], equal_nan=True)
    assert result.three.isna().all()
    assert evidence(result, 'guarded', 6)['expired']
    live = evidence(result, 'mean', 7)
    assert live['restarted'] and not live['expired']
    assert live['pre_guard_eligible_fixture_count'] == 7
    assert live['eligible_fixture_count'] == live['usable_observation_count'] == 1
    assert live['latest_age_days'] == 9
    current = evidence(result, 'mean', 9)
    assert current['eligible_fixture_count'] == 3 and current['usable_observation_count'] == 2
    assert not any('inspection' in name for name in result.columns)


def test_cutoff_strict_equality_and_continuous_history_across_seasons():
    start = pd.Timestamp('2018-01-01', tz='UTC')
    exact = start + pd.Timedelta(weeks=26)
    last = exact + pd.Timedelta(weeks=26)
    history = observations([start, exact, last, last + pd.Timedelta(nanoseconds=1),
                            last + pd.Timedelta(days=1)], [5., 9., None, None, 999.],
                           seasons=[1, 1, 2, 2, 2])
    original = EMA(STAT, span=3)
    result = evaluate_features(history, {'ordinary': original, 'same': HistoryInspector(original)})
    np.testing.assert_allclose(result['same'].iloc[:3], result.ordinary.iloc[:3], equal_nan=True)
    assert result['same'].iloc[2] == 7.0  # First value is now 52 weeks old.
    assert result['same'].iloc[3:].isna().all()
    assert not evidence(result, 'same', 2)['restarted']
    # Prediction cutoff can still be fresh when the target kickoff would expire it.
    cutoffs = history.kickoff_at.copy()
    cutoffs.iloc[-1] = last
    assert evaluate_features(history, {'same': HistoryInspector(original)}, cutoffs=cutoffs)['same'].iloc[-1] == 7.


def test_irregular_availability_can_bridge_a_gap_only_once_known_per_column():
    history = observations(['2020-01-01', '2020-05-01', '2020-09-01', '2020-09-08',
                            '2020-09-15', '2020-09-22'], [10., 20., 40., None, None, 999.],
                           against=[100., None, 400., None, None, 999.])
    release = history.kickoff_at.copy()
    release.iloc[1] = history.kickoff_at.iloc[4]
    expr = HistoryInspector(EMA(ForAgainst(STAT, side='both'), span=3))
    result = evaluate_features(history, {'x': expr}, available_at=release, keyed=True)
    assert result.loc[:, 'x::own'].iloc[3:5].tolist() == [40., 27.5]
    assert result.loc[:, 'x::other'].iloc[3:5].tolist() == [400., 400.]
    assert evidence(result, 'x', 3)['restarted']
    assert not evidence(result, 'x', 4)['restarted']
    assert evidence(result, 'x', 4, OTHER)['restarted']
    shuffled = history.iloc[[5, 4, 1, 0, 2, 3]]
    shuffled_result = evaluate_features(shuffled, {'x': expr}, available_at=release.loc[shuffled.index], keyed=True)
    pd.testing.assert_frame_equal(result.sort_index(), shuffled_result.sort_index(), check_flags=False)


def test_venue_h2h_and_missing_lag_slots():
    history = observations(['2019-08-01', '2019-08-10', '2020-08-01', '2020-08-10',
                            '2020-08-15', '2020-08-20'], [1., 10., 20., None, 30., 999.])
    history.loc[2, 'opponent_id'] = 72
    history.loc[4, 'side'] = 'away'
    reducer = Lag(STAT, venue='same')
    results = evaluate_features(history, {
        'one': H2H(HistoryInspector(reducer)), 'two': HistoryInspector(H2H(reducer)),
    })
    assert results.iloc[-1].isna().all()
    for name in results.columns:
        audit = evidence(results, name, 5)
        assert audit['pre_guard_eligible_fixture_count'] == 3
        assert audit['expired']  # Match three with observed coverage is another opponent.


def test_weighted_pair_releases_and_zero_weight_do_not_bridge():
    history = observations(['2020-01-01', '2020-05-01', '2020-06-01', '2020-09-01',
                            '2020-09-08', '2020-09-15'],
                           [10., 20., 30., 40., None, 999.])
    history['weight'] = [1., 1., 0., 1., 1., 1.]
    history.attrs['stat_columns']['weight'] = dict(role='team', period='1ST', group_name='G', key='value', field='value')
    history['weight_release'] = history.kickoff_at
    history.loc[1, 'weight_release'] = history.kickoff_at.iloc[-1]
    guarded = HistoryInspector(RollingWeightedMean(STAT, WEIGHT, window=5, weights_available_at='weight_release'))
    result = evaluate_features(history, {'weighted': guarded})
    np.testing.assert_allclose(result.weighted.iloc[-2:], [40., 70. / 3.])
    assert evidence(result, 'weighted', 4, 'value')['restarted']
    assert not evidence(result, 'weighted', 5, 'value')['restarted']
    assert evidence(result, 'weighted', 5, 'value')['usable_observation_count'] == 3
    history.loc[1, 'weight'] = -1.
    with pytest.raises(ValueError, match='finite negative weight'):
        evaluate_features(history, {'weighted': guarded})


def test_ema_dense_prefix_matches_baseline_and_separate_cached_descendants():
    dates = pd.date_range('2020-01-01', periods=80, tz='UTC')
    history = observations(dates, np.sin(np.arange(80)))
    ordinary = EMA(STAT, span=12)
    guarded = HistoryInspector(ordinary)
    results = evaluate_features(history, {'plain': ordinary, 'same': guarded,
        'sum': Sum(guarded, guarded), 'other': HistoryInspector(ordinary, evidence='off')})
    np.testing.assert_allclose(results['plain'], results['same'])
    np.testing.assert_allclose(results['other'], results['same'])
    np.testing.assert_allclose(results['sum'], 2 * results['same'])
    assert len(results.attrs['history_inspection']['sum']) == 1
    assert 'other' not in results.attrs['history_inspection']


def test_predictors_are_explicit_and_nested_opt_in_rejected():
    history = observations(['2020-01-01', '2020-02-01'], [1., 99.])
    guarded = HistoryInspector(Lag(STAT), evidence='predictors')
    results = evaluate_features(history, {'value': H2H(guarded)})
    prefix = 'value::inspection::own'
    assert results[f'{prefix}::latest_age_days'].iloc[1] == 31
    assert results[f'{prefix}::eligible_fixture_count'].tolist() == [0., 1.]
    assert np.isnan(results[f'{prefix}::latest_age_days'].iloc[0])
    with pytest.raises(ValueError, match='named feature root'):
        evaluate_features(history, {'unsupported': RollingMean(guarded)})


def test_unsupported_forms_and_native_requirements():
    from xdiyo_analytics.data.catalogue.integration import feature_requirements

    with pytest.raises(TypeError, match='wraps Lag'):
        HistoryInspector(STAT)
    with pytest.raises(ValueError, match='league-population'):
        HistoryInspector(RollingMean(League(STAT)))
    with pytest.raises(ValueError, match='positive'):
        HistoryInspector(EMA(STAT), gap_weeks=True)
    with pytest.raises(ValueError, match='evidence'):
        HistoryInspector(EMA(STAT), evidence='typo')
    assert feature_requirements(HistoryInspector(RollingZScore(STAT))) == {'goals', 'stat:ALL:value'}


def test_recipe_roundtrip_assembly_preserves_labels_and_diagnostic_separation(ui_recipe, tmp_path):
    original = prepare_recipe(ui_recipe).dataset
    raw = ui_recipe['features']['corners_mean']
    ui_recipe['features']['corners_mean'] = node('features.HistoryInspector', source=raw)
    catalog = catalog_for_ui()
    expression = catalog.build(ui_recipe['features']['corners_mean'])
    assert expression == catalog.build(json.loads(json.dumps(catalog.encode(expression))))
    path = tmp_path / 'inspection.json'
    path.write_text(json.dumps(ui_recipe))
    from xdiyo_analytics.ui.server import _preparation_preview

    prepared = prepare_recipe(read_recipe(path))
    guarded = prepared.dataset
    preview = _preparation_preview(prepared)['history_inspection']
    assert 'eligible_fixture_count' in preview['columns']
    assert 'expired' in preview['columns']
    assert preview['count'] == 48 and len(preview['rows']) == 40
    event_id = preview['rows'][0][preview['columns'].index('event_id')]
    assert isinstance(event_id, str) and int(event_id) >= 2**63
    json.dumps(preview, allow_nan=False)
    pd.testing.assert_frame_equal(guarded.metadata, original.metadata)
    pd.testing.assert_frame_equal(guarded.y, original.y)
    assert list(guarded.X) == list(original.X)
    assert guarded.definitions['history_inspection']['corners_mean']
    assert guarded.X.isna().sum().sum() > original.X.isna().sum().sum()


def test_dense_ema_is_exact_against_independent_point_in_time_oracle():
    seed = np.random.default_rng(911)
    first = pd.date_range('2019-01-01', periods=81, freq='3D', tz='UTC')
    second = pd.date_range('2020-08-01', periods=81, freq='2D', tz='UTC')
    history = observations(first.append(second), seed.normal(size=162))
    # Distinct gaps in the two columns, plus missing and late observations that
    # make both selection and update unsafe to cache without a prefix check.
    history.loc[seed.choice(160, size=40, replace=False), OWN] = np.nan
    history[OTHER] = seed.normal(size=162)
    history.loc[seed.choice(160, size=47, replace=False), OTHER] = np.nan
    history.attrs['stat_columns'][OTHER] = dict(
        role='opponent', period='ALL', group_name='G', key='value', field='value')
    release = history.kickoff_at + pd.to_timedelta(seed.integers(0, 70, size=len(history)), unit='D')
    cuts = history.kickoff_at - pd.to_timedelta(seed.integers(0, 5, size=len(history)), unit='D')
    for column in (OWN, OTHER):
        history[f'available_at::{column}'] = history.kickoff_at + pd.to_timedelta(
            seed.integers(0, 120, size=len(history)), unit='D')
    shuffled = history.sample(frac=1, random_state=90)
    expr = HistoryInspector(EMA(ForAgainst(STAT, side='both'), span=11), gap_weeks=8, evidence='off')
    result = evaluate_features(shuffled, {'x': expr}, cutoffs=cuts.loc[shuffled.index],
                               available_at=release.loc[shuffled.index]).sort_index()
    threshold = pd.Timedelta(weeks=8)
    for column in (OWN, OTHER):
        values = []
        for cut in cuts:
            mask = ((history.status == 'finished') & (history.kickoff_at < cut) & (release <= cut)
                    & (history[f'available_at::{column}'] <= cut) & history[column].notna())
            known_observations = history.loc[mask, ['kickoff_at', column]]
            state, previous = np.nan, None
            for known, value in known_observations.itertuples(index=False, name=None):
                state = value if previous is None or known - previous > threshold else value / 6 + state * (5 / 6)
                previous = known
            values.append(state if previous is not None and cut - previous <= threshold else np.nan)
        np.testing.assert_allclose(result[f'x::{column}'], values, rtol=1e-13, atol=1e-15, equal_nan=True)


def test_raw_source_release_masks_reducer_and_composed_sources():
    history = observations(['2020-01-01', '2020-05-01', '2020-09-01',
                            '2020-09-08', '2020-09-15', '2020-09-22'],
                           [10., 20., 40., 9000., None, 999.],
                           against=[100., 200., 400., 9000., None, 999.])
    history['available_at::own'] = history.kickoff_at
    history['available_at::other'] = history.kickoff_at
    history.loc[1, 'available_at::own'] = history.kickoff_at.iloc[4]
    history.loc[3, 'available_at::other'] = pd.NaT
    # Weighted compounds and arithmetic must inherit the release of both
    # leaves. A later release cannot affect the earlier recomputation.
    total = Sum(STAT, ForAgainst(STAT, side='against'))
    guarded = HistoryInspector(EMA(total, span=3))
    result = evaluate_features(history, {'total': guarded,
        'lag': HistoryInspector(Lag(total)), 'plain': EMA(total, span=3)})
    assert result.total.iloc[3:5].tolist() == [440., 302.5]
    assert pd.isna(result.lag.iloc[4])
    assert result.plain.iloc[4] > 1000  # unchanged without the wrapper
    with pytest.raises(ValueError, match='unguarded nested producer'):
        evaluate_features(history, {'unsafe': HistoryInspector(EMA(EMA(total)))})
    assert len(evaluate_features(history, {'safe': HistoryInspector(EMA(guarded))})) == len(history)


@pytest.mark.parametrize(('operator', 'usable'), [(RollingMean, 1), (Lag, 0)])
def test_point_lineage_counts_the_same_cutoff_usable_source(operator, usable):
    from test_point_geometry import maps_for
    from xdiyo_analytics.features import SpatialPointSummary

    history = observations(['2020-01-01', '2020-01-02', '2020-01-03'], [10., 200., 999.])
    history['source_league'] = 'Test'
    history['source_season'] = '20'
    history['available_at::own'] = history.kickoff_at
    history.loc[1, 'available_at::own'] = pd.Timestamp('2020-01-04', tz='UTC')
    source = Sum(SpatialPointSummary(), STAT)
    result = evaluate_features(history, {'x': HistoryInspector(operator(source))}, heatmaps=maps_for(history))
    records = result.attrs['spatial_point_audit']['features']['x']['history']
    assert records[-1]['usable_values'] == usable
    assert np.isfinite(result.x.iloc[-1]) == bool(usable)
