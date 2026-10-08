"""Reviewed awards and multi-day completion must survive publication and use."""
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from examples.repair_historical_results_20261008 import EVIDENCE, digest, patch_table, repair
from xdiyo_analytics.data import load_season
from xdiyo_analytics.data.availability import respect_result_availability
from xdiyo_analytics.features.history import availability_times, eligible_history_rows
from xdiyo_analytics.histories import build_team_history
from xdiyo_analytics.ratings.bayesian_replay import _events
from xdiyo_analytics.ratings.bayesian_training import _training_history
from test_bayesian_training import synthetic_history


REVIEWS = json.loads(EVIDENCE.read_bytes())['records']


@pytest.mark.parametrize('review', REVIEWS, ids=lambda r: str(r['event_id']))
def test_review_preserves_provider_and_unrelated_observations(review):
    table = pa.table(dict(event_id=[review['event_id'], 123],
        home_id=[review['home_id'], 1], away_id=[review['away_id'], 2],
        home_score_current=[float(review['expected_score'][0]), 9.],
        away_score_current=[float(review['expected_score'][1]), 8.],
        is_awarded=[False, None], status=['finished', 'notstarted'],
        nested=[[{'points': [1., 2.]}], None], home_got_promoted=[True, False]))
    table = table.replace_schema_metadata({b'preserve': b'yes'})
    out = patch_table(table, review)
    for name in table.column_names:
        if name != 'is_awarded':
            assert out[name].equals(table[name])
    assert out.schema.metadata == table.schema.metadata
    assert out['is_awarded'].to_pylist() == [review['is_awarded'], None]
    assert out['provider_is_awarded'].equals(table['is_awarded'])
    assert out['home_score_played'][0].as_py() == review['played_score'][0]
    assert out['away_score_played'][0].as_py() == review['played_score'][1]
    assert patch_table(out, review).equals(out, check_metadata=True)
    with pytest.raises(ValueError, match='teams'):
        patch_table(table, {**review, 'home_id': 999})


def test_reviewed_bound_is_a_floor_without_imputing_missing_releases():
    metadata = pd.DataFrame({'result_available_at': pd.to_datetime(
        ['2011-12-14T23:00Z', '2011-12-14T23:00Z', '2011-12-14T23:00Z', None], utc=True)})
    proxy = pd.Series(pd.to_datetime(['2011-12-03T17:00Z', None, '2011-12-20T00:00Z', '2011-12-03T17:00Z'], utc=True))
    actual = respect_result_availability(metadata, proxy)
    assert actual.iloc[0] == pd.Timestamp('2011-12-14T23:00Z')
    assert pd.isna(actual.iloc[1])
    pd.testing.assert_series_equal(actual.iloc[2:], proxy.iloc[2:])
    with pytest.raises(TypeError):
        respect_result_availability(pd.DataFrame({'result_available_at': [123]}), proxy.iloc[:1])


@pytest.mark.parametrize('explicit', [False, True])
def test_history_and_bayesian_population_do_not_reveal_incomplete_match(explicit):
    history = synthetic_history(seasons=1, matches=3)
    first = history.event_id.eq(history.event_id.min())
    completion = history.kickoff_at.max() + pd.Timedelta('1h')
    history['result_available_at'] = pd.Series(pd.NaT, index=history.index, dtype='datetime64[ns, UTC]')
    history.loc[first, 'result_available_at'] = completion
    supplied = history.kickoff_at + pd.Timedelta('3h') if explicit else None
    rows = eligible_history_rows(history, available_at=supplied)
    assert 0 not in rows[2] and 0 not in rows[4]
    events, _, _ = _events(history, supplied)
    assert events[0]['available_at'] == completion
    chosen = _training_history(history, history.kickoff_at.max(), supplied)
    assert history.event_id.min() not in chosen.event_id.tolist()
    assert availability_times(history, supplied).iloc[0] == completion
    # Mutating the delayed result cannot change intervening Bayesian forecasts.
    from xdiyo_analytics.ratings import build_bayesian_ratings
    changed = history.copy()
    changed.loc[first, ['goals_for', 'goals_against']] = 7
    a = build_bayesian_ratings(history, available_at=supplied)
    b = build_bayesian_ratings(changed, available_at=supplied)
    pd.testing.assert_frame_equal(a.fixture_features(history), b.fixture_features(history))


@pytest.mark.parametrize('layout', ['match', 'team_match'])
def test_temporal_folds_respect_bound_with_default_and_three_hour_proxy(layout):
    from split_samples import dataset
    from xdiyo_analytics.splits import TemporalSplit
    data = dataset(8, layout=layout, specs=[dict(day=i, round=i+1) for i in range(8)])
    event = data.metadata.event_id.iloc[0]
    selected = data.metadata.event_id.map(lambda value: value == event)
    data.metadata['result_available_at'] = pd.Series(pd.NaT, index=data.metadata.index, dtype='datetime64[ns, UTC]')
    data.metadata.loc[selected, 'result_available_at'] = data.metadata.kickoff_at.max()+pd.Timedelta('1d')
    for proxy in (None, data.metadata.kickoff_at + pd.Timedelta('3h')):
        folds = TemporalSplit(3, unit='kickoffs').folds(data, available_at=proxy)
        assert all(event not in data.metadata.iloc[f.train].event_id.tolist() for f in folds)


def test_match_calibration_and_composition_bound_all_team_rows():
    from types import SimpleNamespace
    from split_samples import dataset
    from xdiyo_analytics.splits.core import _Matches, _respect_result_bounds
    from xdiyo_analytics.composition.training import TrainingPlan
    data = dataset(8, layout='team_match')
    metadata = data.metadata
    metadata['result_available_at'] = pd.Series(pd.NaT, index=metadata.index, dtype='datetime64[ns, UTC]')
    bound = metadata.kickoff_at.max() + pd.Timedelta('1d')
    metadata.loc[1, 'result_available_at'] = bound
    matches = _Matches(data)
    releases = _respect_result_bounds(matches, matches.times('kickoff_at', 'kickoff_at'))
    assert releases[matches.codes[1]] == bound
    plan = TrainingPlan(availability_delay='3h')
    _, _, releases = plan.times(SimpleNamespace(metadata=metadata))
    assert releases.iloc[1] == bound


def test_real_publication_roundtrip_and_exclusion_in_isolated_copy(tmp_path):
    """Small copies exercise actual table schemas, without changing source files."""
    import shutil
    root = Path('data/xDiyo_data')
    for review in REVIEWS:
        stem = review['season']
        pubpath = root / (stem + '.manifest.json')
        pub = json.loads(pubpath.read_bytes())
        manifestpath = root / pub['manifest']
        manifest = json.loads(manifestpath.read_bytes())
        for relative in [pubpath.relative_to(root), Path(pub['manifest']), Path(pub['season_file']),
                         Path(pub['manifest']).parent / manifest['tables']['matches']['file']]:
            dest = tmp_path / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / relative, dest)
    before = {str(p): digest(p) for p in tmp_path.rglob('*.parquet')}
    preview = repair(tmp_path)
    assert len(preview['seasons']) == 5
    assert before == {str(p): digest(p) for p in tmp_path.rglob('*.parquet')}
    repair(tmp_path, apply=True)
    for review in REVIEWS:
        data = load_season(tmp_path, review['season'], include_awarded=True, verify_hashes=True)
        row = data.matches.loc[data.matches.event_id.eq(review['event_id'])].iloc[0]
        assert bool(row.is_awarded) == review['is_awarded']
        normal = load_season(tmp_path, review['season'])
        assert (review['event_id'] not in normal.matches.event_id.tolist()) == review['is_awarded']
        history = build_team_history(data)
        from xdiyo_analytics.features import IsHome, evaluate_features
        from xdiyo_analytics.labels import MatchGoals, create_labels
        from xdiyo_analytics.datasets import assemble_dataset
        dataset = assemble_dataset(evaluate_features(history, {'home': IsHome()}, keyed=True),
            create_labels(history, {'goals': MatchGoals()})['goals'], layout='match')
        assembled = dataset.metadata.loc[dataset.metadata.event_id.eq(review['event_id'])].iloc[0]
        assert bool(assembled.is_awarded) == review['is_awarded']
        if review['play_status'] == 'resumed':
            assert assembled.result_available_at == pd.Timestamp(review['result_available_at'])
            times = availability_times(history, history.kickoff_at + pd.Timedelta('3h'))
            assert times[history.event_id.eq(review['event_id'])].eq(pd.Timestamp(review['result_available_at'])).all()
    before = {str(p): digest(p) for p in tmp_path.rglob('*') if p.is_file()}
    assert not any(r['changed'] for r in repair(tmp_path, apply=True)['seasons'])
    assert before == {str(p): digest(p) for p in tmp_path.rglob('*') if p.is_file()}
