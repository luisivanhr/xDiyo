"""Season clock uses assigned rounds and complete fixture schedules."""
import json

import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.features import SeasonProgress, Difference, Constant, evaluate_features
from xdiyo_analytics.ui.recipe import catalog_for_ui, node, prepare_recipe
from test_ui_workflow import ui_recipe
from transition_samples import history_from_games


def test_postponement_uses_original_round_and_upcoming_schedule_per_league_season():
    history = history_from_games([
        dict(round=5, day=120),  # postponed beyond round 12
        dict(round=12, day=90),
        dict(round=38, day=280, status='notstarted'),
        dict(competition=20, round=5, day=120),
        dict(competition=20, round=34, status='notstarted'),
        dict(season=2, round=5),
        dict(season=2, round=30, status='notstarted'),
    ])
    history.index = [7] * len(history)
    result = evaluate_features(history, {'season_progress': SeasonProgress()})
    expected = np.repeat([5/38, 12/38, 1, 5/34, 1, 5/30, 1], 2)
    np.testing.assert_allclose(result.season_progress, expected)
    assert result.index.equals(history.index)
    history['kickoff_at'] += pd.Timedelta(days=400)
    history['status'] = 'postponed'
    history['result'] = None
    pd.testing.assert_frame_equal(result, evaluate_features(history, {'season_progress': SeasonProgress()}))


def test_override_partial_schedule_missing_rounds_and_arithmetic():
    frame = pd.DataFrame({'round': [1, 5, None, 'playoff']})
    values = evaluate_features(frame, {'progress': SeasonProgress(38),
                                     'remaining': Difference(Constant(1), SeasonProgress(38))})
    np.testing.assert_allclose(values.progress, [1/38, 5/38, np.nan, np.nan], equal_nan=True)
    np.testing.assert_allclose(values.remaining, [37/38, 33/38, np.nan, np.nan], equal_nan=True)
    with pytest.raises(ValueError, match='competition_id'):
        evaluate_features(frame, {'progress': SeasonProgress()})
    with pytest.raises(ValueError, match='between 1'):
        evaluate_features(pd.DataFrame({'round': [39]}), {'progress': SeasonProgress(38)})


@pytest.mark.parametrize('value', [0, -1, 38.5, True, '38', float('inf')])
def test_invalid_override_rejected(value):
    with pytest.raises(ValueError, match='positive integer'):
        SeasonProgress(value)


@pytest.mark.parametrize('value', [0, -1, 1.5, float('inf')])
def test_invalid_numeric_round_rejected(value):
    frame = pd.DataFrame({'competition_id': [1], 'season_id': [1], 'round': [value]})
    with pytest.raises(ValueError, match='between 1'):
        evaluate_features(frame, {'progress': SeasonProgress()})


def test_catalog_roundtrip_and_ui_preparation_uses_future_fixture_before_label_filter(ui_recipe):
    catalog = catalog_for_ui()
    for expression in (SeasonProgress(), SeasonProgress(38), SeasonProgress(mode='kickoff'),
                       SeasonProgress(mode='kickoff', total_days=300, start_date='2026-08-01')):
        assert catalog.build(json.loads(json.dumps(catalog.encode(expression)))) == expression
    schema = next(x for x in catalog.schema() if x['id'] == 'features.SeasonProgress')
    assert schema['fields'][0]['name'] == 'mode'
    field = next(field for field in schema['fields'] if field['name'] == 'total_rounds')
    assert field['kind'] == 'number' and field['min'] == 1
    assert field['clear_when_hidden'] is True
    assert 'postponed' in field['help']
    ui_recipe['features'] = {'season_progress': node('features.SeasonProgress')}
    prepared = prepare_recipe(ui_recipe)
    dataset = prepared.dataset
    for col in ('home::season_progress', 'away::season_progress'):
        np.testing.assert_allclose(dataset.X[col], dataset.metadata['round'].astype(float)/8)
    # The final season's round 8 is unplayed and dropped from labeled data,
    # but still supplies the denominator for that season.
    final = dataset.metadata.source_season == '24_25'
    assert dataset.metadata.loc[final, 'round'].max() == 7


def test_kickoff_inclusive_days_postponements_and_separate_seasons():
    data = pd.DataFrame({
        'competition_id': [1, 1, 1, 1, 2, 2, 1, 1],
        'season_id': [1, 1, 1, 1, 1, 1, 2, 2],
        'round': [1, 5, 12, 38, 1, 34, 1, 38],
        'kickoff_at': ['2026-01-01T23:00Z', '2026-01-16T13:00Z',
                       '2026-01-11T23:00Z', '2026-01-20T15:00Z',
                       '2026-01-10T00:00Z', '2026-01-10T23:00Z',
                       '2027-01-01', '2027-01-10'],
    }, index=[9] * 8)
    result = evaluate_features(data, {'progress': SeasonProgress(mode='kickoff')})
    np.testing.assert_allclose(result.progress, [1/20, 16/20, 11/20, 1, 1, 1, 1/10, 1])
    assert result.index.equals(data.index)
    # Round 5 was postponed past round 12: kickoff progress reflects that.
    assert result.progress.iloc[1] > result.progress.iloc[2]


def test_kickoff_overrides_missing_values_and_utc_dates():
    data = pd.DataFrame({'kickoff_at': ['2026-01-01T23:30-02:00', None, 'bad']})
    feature = SeasonProgress(mode='kickoff', start_date='2026-01-01', total_days=10)
    actual = evaluate_features(data, {'progress': feature})
    np.testing.assert_allclose(actual.progress, [0.2, np.nan, np.nan], equal_nan=True)
    with pytest.raises(ValueError, match='outside'):
        evaluate_features(data, {'progress': SeasonProgress(mode='kickoff', start_date='2026-01-03', total_days=10)})
    for expression in [SeasonProgress(mode='kickoff'), SeasonProgress(mode='kickoff', total_days=10)]:
        with pytest.raises(ValueError, match='inference needs'):
            evaluate_features(data, {'progress': expression})


def test_kickoff_partial_overrides_and_unknown_schedule_identity():
    data = pd.DataFrame({'competition_id': [1, 1, None, 2], 'season_id': [1, 1, 1, 1],
                         'kickoff_at': ['2026-01-02', '2026-01-10', '2026-01-05', None]})
    for expression, expected in [
        (SeasonProgress(mode='kickoff', start_date='2026-01-01'), [0.2, 1, np.nan, np.nan]),
        (SeasonProgress(mode='kickoff', total_days=10), [0.1, 0.9, np.nan, np.nan]),
    ]:
        actual = evaluate_features(data, {'progress': expression})
        np.testing.assert_allclose(actual.progress, expected, equal_nan=True)


def test_kickoff_ui_preparation_includes_unplayed_final_date(ui_recipe):
    ui_recipe['features'] = {'season_progress': node('features.SeasonProgress', mode='kickoff')}
    dataset = prepare_recipe(ui_recipe).dataset
    # Eight weekly fixtures span 50 inclusive calendar days, even though
    # the last current-season fixture has no label and is dropped afterward.
    expected = ((dataset.metadata['round'].astype(float)-1)*7 + 1) / 50
    for column in ('home::season_progress', 'away::season_progress'):
        np.testing.assert_allclose(dataset.X[column], expected)


@pytest.mark.parametrize('params', [dict(mode='bad'), dict(total_days=2), dict(start_date='2026-01-01'),
    dict(mode='kickoff', total_rounds=38), dict(mode='kickoff', total_days=0),
    dict(mode='kickoff', total_days=True), dict(mode='kickoff', start_date='bad')])
def test_incompatible_or_invalid_mode_options(params):
    with pytest.raises(ValueError, match='SeasonProgress'):
        SeasonProgress(**params)
