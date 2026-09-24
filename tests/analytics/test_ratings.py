"""Independent Glicko arithmetic, rating replay, persistence and feature timing."""

from copy import deepcopy
from dataclasses import dataclass
import math
import random

import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.features import (
    H2H, IsHome, MatchResultGlicko, Rating, Stat, StatGlicko, evaluate_features,
)
from xdiyo_analytics.ratings import Glicko2, RatingRun, build_ratings, glicko_state


A, B, C = 2**63 + 11, 2**63 + 19, 2**63 + 27
CORNER = Stat('ALL', 'Match overview', 'cornerKicks')
STREAM = 'ALL::Match overview::cornerKicks::value'
START = pd.Timestamp('2025-01-01 15:00:00', tz='UTC')


def make_history(specs):
    records, metadata = [], {}
    for i, spec in enumerate(specs):
        day = spec.get('day', i * 2)
        kickoff = START + pd.Timedelta(days=day) if day is not None else pd.NaT
        result = spec.get('result', 'W')
        home, away = spec.get('home', A), spec.get('away', B)
        for side, team, opponent in [('home', home, away), ('away', away, home)]:
            row = {'event_id': 2**53 + i, 'competition_id': spec.get('competition', 10),
                   'season_id': spec.get('season', 1), 'side': side, 'team_id': team,
                   'opponent_id': opponent, 'kickoff_at': kickoff,
                   'status': spec.get('status', 'finished'),
                   'result': result if side == 'home' else {'W': 'L', 'L': 'W', 'D': 'D'}.get(result)}
            for period, pair in [('ALL', spec.get('corners', (2., 5.))),
                                 ('1ST', spec.get('first_half', (1., 0.)))]:
                for role in ['team', 'opponent']:
                    column = f'{role}::{period}::Match overview::cornerKicks::value'
                    pos = 0 if (side == 'home') == (role == 'team') else 1
                    row[column] = pair[pos]
                    metadata[column] = {'role': role, 'period': period, 'group_name': 'Match overview',
                                        'key': 'cornerKicks', 'field': 'value'}
            records.append(row)
    history = pd.DataFrame(records)
    for column in ['team_id', 'opponent_id']:
        history[column] = pd.array(history[column], dtype='uint64[pyarrow]')
    history['event_id'] = pd.array(history['event_id'], dtype='int64[pyarrow]')
    history['kickoff_at'] = pd.to_datetime(history['kickoff_at'], utc=True)
    for column in metadata:
        history[column] = pd.array(history[column], dtype='float64[pyarrow]')
    history.attrs = {'stat_columns': metadata, 'source': {'version': 'synthetic'}}
    return history


@dataclass(frozen=True)
class PointsEngine:
    """Expose outcomes and pre-period opponent points using simple arithmetic."""

    def initial_state(self):
        return {'points': 0., 'opponent_points': 0., 'games': 0.}

    def update_period(self, state, games):
        return {'points': state['points'] + sum(score for score, _ in games),
                'opponent_points': state['opponent_points'] + sum(other['points'] for _, other in games),
                'games': state['games'] + len(games)}


def independent_update(state, games, tau=1.):
    """Reference equations with bisection, independent of the core Illinois solver."""
    mu = (state['rating'] - 1500) / 173.7178
    phi = state['rd'] / 173.7178
    sigma = state['sigma']
    weighted, variances = [], []
    for score, opponent in games:
        g = (1 + 3 * (opponent['rd'] / 173.7178) ** 2 / math.pi ** 2) ** -.5
        probability = 1 / (1 + math.exp(-g * (state['rating'] - opponent['rating']) / 173.7178))
        weighted.append(g * (score - probability))
        variances.append(g * g * probability * (1 - probability))
    v = 1 / math.fsum(variances)
    delta = v * math.fsum(weighted)
    a = math.log(sigma * sigma)

    def equation(x):
        ex = math.exp(x)
        denominator = phi * phi + v + ex
        return ex * (delta * delta - denominator) / (2 * denominator * denominator) - (x - a) / (tau * tau)

    low, high = a - 30, a + 30
    assert equation(low) > 0 > equation(high)
    for _ in range(160):
        midpoint = (low + high) / 2
        if equation(midpoint) > 0:
            low = midpoint
        else:
            high = midpoint
    volatility = math.exp((low + high) / 4)
    deviation = (1 / (phi * phi + volatility * volatility) + 1 / v) ** -.5
    location = mu + deviation * deviation * math.fsum(weighted)
    return {'rating': 1500 + 173.7178 * location, 'rd': 173.7178 * deviation,
            'sigma': volatility, 'mu': location, 'phi': deviation}


def assert_state(actual, expected):
    for field, value in expected.items():
        assert actual[field] == pytest.approx(value, rel=1e-8, abs=1e-8), field


def ordered(run):
    return run.snapshots.sort_values([*run.scope, 'stream', 'recorded_at', 'team_id']).reset_index(drop=True)


def test_official_glickman_example_and_unchanged_inputs():
    # https://www.glicko.net/glicko/glicko2.pdf, March 2022, pp. 4-6.
    state = glicko_state(1500, 200, .06)
    opponents = [glicko_state(1400, 30), glicko_state(1550, 100), glicko_state(1700, 300)]
    games = list(zip([1., 0., 0.], opponents))
    original = deepcopy((state, games))
    actual = Glicko2(tau=.5).update_period(state, games)
    assert actual['rating'] == pytest.approx(1464.06, abs=.01)
    assert actual['rd'] == pytest.approx(151.52, abs=.01)
    assert actual['sigma'] == pytest.approx(.05999, abs=1e-5)
    assert_state(actual, independent_update(state, games, tau=.5))
    assert (state, games) == original


def test_public_and_internal_scales_and_explicit_empty_period():
    state = glicko_state(1600, 200, .06)
    assert state['mu'] == pytest.approx(100 / 173.7178)
    assert state['phi'] == pytest.approx(200 / 173.7178)
    actual = Glicko2().update_period(state, [])
    assert actual['rating'] == 1600 and actual['sigma'] == .06
    assert actual['rd'] == pytest.approx(math.hypot(200, .06 * 173.7178))
    assert actual is not state


@pytest.mark.parametrize('custom_center', [False, True])
def test_adapter_exactly_matches_legacy_for_periods_and_explicit_idle(custom_center):
    from utils.glicko_rating import Glicko2 as LegacyGlicko2, Rating as LegacyRating
    settings = dict(initial_rating=1720., initial_rd=240., initial_sigma=.045,
                    tau=.7, tolerance=1e-7) if custom_center else {}
    adapter = Glicko2(**settings)
    legacy = LegacyGlicko2(mu0=adapter.initial_rating, phi0=adapter.initial_rd,
                          sigma0=adapter.initial_sigma, tau=adapter.tau, epsilon=adapter.tolerance)
    if not custom_center:
        assert adapter.tau == LegacyGlicko2().tau == 1.
    rng = random.Random(24161)
    state = adapter.initial_state()
    for period in range(18):
        games = [(rng.choice([0., .5, 1.]), glicko_state(rng.uniform(1000, 2000),
                   rng.uniform(30, 300), rng.uniform(.03, .09))) for _ in range(period % 4)]
        public = LegacyRating(state['rating'], state['rd'], state['sigma'])
        expected = legacy.rate(public, [(score, LegacyRating(other['rating'], other['rd'], other['sigma']))
                                         for score, other in games])
        actual = adapter.update_period(state, games)
        assert (actual['rating'], actual['rd'], actual['sigma']) == (expected.mu, expected.phi, expected.sigma)
        for periods in [0., .25, 1., 8.]:
            for cap in [False, True]:
                expected_idle = legacy.advance_periods(expected, periods, cap_to_prior=cap)
                actual_idle = adapter.advance_periods(actual, periods, cap_to_prior=cap)
                assert (actual_idle['rating'], actual_idle['rd'], actual_idle['sigma']) == (
                    expected_idle.mu, expected_idle.phi, expected_idle.sigma)
                assert actual_idle['mu'] == (expected_idle.mu - 1500) / 173.7178
                assert actual_idle['phi'] == expected_idle.phi / 173.7178
        state = actual


@pytest.mark.parametrize('periods', [-1, float('nan'), float('inf')])
def test_invalid_explicit_idle_periods_rejected(periods):
    engine = Glicko2()
    with pytest.raises(ValueError):
        engine.advance_periods(engine.initial_state(), periods)


@pytest.mark.parametrize('score', [-.1, 1.1, float('nan'), float('inf')])
def test_invalid_pairwise_score_rejected(score):
    engine = Glicko2()
    with pytest.raises(ValueError):
        engine.update_period(engine.initial_state(), [(score, engine.initial_state())])


@pytest.mark.parametrize('kwargs', [{'initial_rd': 0}, {'initial_sigma': -1},
                                   {'initial_rating': float('nan')}, {'tau': 0}, {'tolerance': -1}])
def test_invalid_engine_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        Glicko2(**kwargs)


def test_both_teams_update_from_pre_match_states_once():
    history = make_history([{'result': 'W'}, {'result': 'L'}])
    run = build_ratings(history)
    initial = Glicko2().initial_state()
    won = independent_update(initial, [(1., initial)])
    lost = independent_update(initial, [(0., initial)])
    second_a = independent_update(won, [(0., lost)])
    second_b = independent_update(lost, [(1., won)])
    rows = run.snapshots.set_index(['recorded_at', 'team_id'])
    for key, expected in [((START, A), won), ((START, B), lost),
                          ((START + pd.Timedelta(days=2), A), second_a),
                          ((START + pd.Timedelta(days=2), B), second_b)]:
        assert_state(rows.loc[key], expected)
    assert len(rows) == 4 and rows.games_seen.tolist() == [1, 1, 2, 2]
    assert won['rating'] + lost['rating'] == pytest.approx(3000)


def test_simultaneous_games_use_shared_pre_batch_states_and_ignore_row_order():
    history = make_history([{'day': 0, 'home': B, 'away': C},
                            {'day': 2, 'home': A, 'away': B},
                            {'day': 2, 'home': A, 'away': C, 'result': 'L'}])
    run = build_ratings(history, engine=PointsEngine())
    last = run.snapshots.loc[run.snapshots.recorded_at.eq(START + pd.Timedelta(days=2))].set_index('team_id')
    assert last.points.to_dict() == {A: 1., B: 1., C: 1.}
    assert last.opponent_points.to_dict() == {A: 1., B: 0., C: 0.}
    assert last.games_seen.to_dict() == {A: 2, B: 2, C: 2}
    shuffled = build_ratings(history.sample(frac=1, random_state=81), engine=PointsEngine())
    pd.testing.assert_frame_equal(ordered(run), ordered(shuffled), check_exact=True)


def test_no_automatic_calendar_idle_inflation():
    short = make_history([{}, {'day': 2, 'result': 'D'}])
    long = make_history([{}, {'day': 2000, 'season': 5, 'result': 'D'}])
    actual, delayed = build_ratings(short), build_ratings(long)
    fields = list(actual.initial_state)
    pd.testing.assert_frame_equal(actual.snapshots[fields], delayed.snapshots[fields], check_exact=True)


def test_result_statistic_periods_and_lower_is_better_are_independent():
    history = make_history([{'corners': (2., 5.), 'first_half': (1., 0.)},
                            {'corners': (3., 3.), 'first_half': (0., 1.)},
                            {'corners': (None, 4.), 'first_half': (2., 2.)},
                            {'corners': (float('inf'), 1.)},
                            {'status': 'notstarted', 'corners': (7., 0.)}])
    results = build_ratings(history, engine=PointsEngine())
    corners = build_ratings(history, stat=CORNER, engine=PointsEngine())
    lower = build_ratings(history, stat=CORNER, engine=PointsEngine(), higher_is_better=False)
    assert results.snapshots.loc[results.snapshots.team_id.eq(A)].iloc[-1].points == 4
    assert corners.snapshots.loc[corners.snapshots.team_id.eq(A)].iloc[-1].points == .5
    assert lower.snapshots.loc[lower.snapshots.team_id.eq(A)].iloc[-1].points == 1.5
    assert len(corners.snapshots) == 4
    periods = build_ratings(history, stat=Stat(None, 'Match overview', 'cornerKicks'), engine=PointsEngine())
    assert set(periods.streams) == {STREAM, '1ST::Match overview::cornerKicks::value'}
    actual_full = periods.snapshots.loc[periods.snapshots.stream.eq(STREAM)].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual_full, corners.snapshots, check_exact=True)
    assert periods.snapshots.loc[periods.snapshots.stream.str.startswith('1ST')].games_seen.max() == 4


def test_scope_controls_cross_season_and_cross_competition_state():
    history = make_history([{}, {'season': 2}, {'season': 2, 'competition': 20}])
    default = build_ratings(history, engine=PointsEngine()).features(history, side='for')
    global_run = build_ratings(history, engine=PointsEngine(), scope=()).features(history, side='for')
    reset = build_ratings(history, engine=PointsEngine(), scope=('competition_id', 'season_id')).features(history, side='for')
    assert default['result::team::points'].tolist() == [0, 0, 1, 0, 0, 0]
    assert global_run['result::team::points'].tolist() == [0, 0, 1, 0, 2, 0]
    assert reset['result::team::points'].tolist() == [0] * 6


def test_target_future_and_equal_kickoff_observations_are_excluded():
    history = make_history([{'day': 0}, {'day': 0, 'away': C}, {'day': 2}, {'day': 4}])
    run = build_ratings(history, engine=PointsEngine())
    values = run.features(history, fields=('points',))
    assert values.iloc[:4].eq(0).all().all()
    assert values.loc[4, 'result::team::points'] == 2
    assert values.loc[5, 'result::opponent::points'] == 2
    changed = history.copy(deep=True)
    changed.loc[6, 'result'], changed.loc[7, 'result'] = 'L', 'W'
    future = build_ratings(changed, engine=PointsEngine()).features(changed, fields=('points',))
    pd.testing.assert_frame_equal(values, future, check_exact=True)
    frozen = history.kickoff_at.where(history.kickoff_at.le(START), START)
    assert run.features(history, cutoffs=frozen).eq(0).all().all()


def delayed_history():
    history = make_history([{'day': 0, 'result': 'W'}, {'day': 2, 'result': 'L'},
                            {'day': 4, 'status': 'notstarted'}])
    releases = pd.Series(pd.to_datetime([START + pd.Timedelta(days=3)] * 2 +
                                        [START + pd.Timedelta(days=2)] * 2 + [pd.NaT] * 2), index=history.index)
    return history, releases


def test_delayed_replay_orders_by_release_and_allows_earlier_match_at_exact_cutoff():
    history, releases = delayed_history()
    run = build_ratings(history, engine=PointsEngine(), available_at=releases)
    assert run.metadata['availability'] == 'explicit'
    assert run.snapshots.recorded_at.tolist() == [START + pd.Timedelta(days=2)] * 2 + [START + pd.Timedelta(days=3)] * 2
    query = history.iloc[-2:].copy()
    early = run.features(query, cutoffs=START + pd.Timedelta(days=2))
    exact = run.features(query, cutoffs=START + pd.Timedelta(days=3))
    assert early.eq(0).all().all()  # A match at the boundary remains excluded.
    assert exact['result::team::points'].tolist() == [1., 1.]
    assert exact['result::team::opponent_points'].tolist() == [1., 0.]
    assert exact['result::team::games'].tolist() == [2., 2.]


def test_snapshot_latest_kickoff_covers_all_contributing_events_after_delayed_release():
    history, releases = delayed_history()
    run = build_ratings(history, engine=PointsEngine(), available_at=releases)
    newest = run.snapshots.loc[run.snapshots.recorded_at.eq(START + pd.Timedelta(days=3))]
    assert newest.latest_kickoff_at.eq(START + pd.Timedelta(days=2)).all()


def test_snapshot_provenance_includes_opponent_prior_state_contributions():
    history = make_history([{'day': 4, 'home': B, 'away': C},
                            {'day': 1, 'home': A, 'away': B}])
    releases = pd.Series([START + pd.Timedelta(days=4)] * 2 +
                         [START + pd.Timedelta(days=5)] * 2, index=history.index)
    run = build_ratings(history, engine=PointsEngine(), available_at=releases)
    team_a = run.snapshots.loc[run.snapshots.team_id.eq(A)].iloc[0]
    assert team_a.games_seen == 1 and team_a.opponent_points == 1
    assert team_a.latest_kickoff_at == START + pd.Timedelta(days=4)


def test_missing_observation_times_skip_updates_and_missing_query_times_stay_missing():
    history = make_history([{'day': None}, {'day': 2}, {'day': 4, 'status': 'postponed'}])
    run = build_ratings(history, engine=PointsEngine())
    assert len(run.snapshots) == 2
    values = run.features(history)
    assert values.iloc[:2].isna().all().all()
    assert values.loc[4, 'result::team::points'] == 1
    releases = history.kickoff_at.copy()
    releases.iloc[2:4] = pd.NaT
    empty = build_ratings(history, engine=PointsEngine(), available_at=releases)
    assert empty.snapshots.empty
    assert empty.features(history).iloc[2:].eq(0).all().all()


def test_large_ids_duplicate_index_input_preservation_and_save_load(tmp_path):
    history = make_history([{}, {'day': 2, 'result': 'D'}, {'day': 4}])
    history.index = pd.Index([2**63 + i for i in [2, 2, 5, 1, 8, 8]], dtype='uint64', name='row')
    original, attrs = history.copy(deep=True), deepcopy(history.attrs)
    run = build_ratings(history)
    values = run.features(history)
    assert values.index.equals(history.index)
    assert set(run.snapshots.team_id) == {A, B}
    directory = tmp_path / 'saved'
    run.save(directory)
    saved_bytes = {path.name: path.read_bytes() for path in directory.iterdir()}
    loaded = RatingRun.load(directory)
    pd.testing.assert_frame_equal(loaded.snapshots, run.snapshots, check_exact=True)
    pd.testing.assert_frame_equal(loaded.features(history), values, check_exact=True)
    assert loaded.initial_state == run.initial_state and loaded.metadata == run.metadata
    assert loaded.scope == run.scope and loaded.streams == run.streams
    assert set(loaded.initial_state) == {'rating', 'rd', 'sigma', 'mu', 'phi'}
    assert loaded.features(history, fields=('rating',)).shape == (6, 2)
    with pytest.raises(FileExistsError):
        run.save(directory)
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == saved_bytes
    pd.testing.assert_frame_equal(history, original, check_exact=True)
    assert history.attrs == attrs


def test_external_numeric_embeddings_are_ordinary_features_and_round_trip(tmp_path):
    history = make_history([{'day': 2}, {'day': 4}])
    snapshots = pd.DataFrame({'stream': ['embedding'] * 2, 'team_id': [A, B],
                              'recorded_at': [START + pd.Timedelta(hours=2)] * 2,
                              'latest_kickoff_at': [START] * 2,
                              'embedding_0': [2.5, -3.], 'embedding_1': [7., 11.]})
    external = RatingRun(snapshots, {'embedding_0': 0., 'embedding_1': 0.}, scope=(),
                         streams=('embedding',), metadata={'producer': 'fixed test vectors; no training'})
    external.save(tmp_path / 'vectors')
    loaded = RatingRun.load(tmp_path / 'vectors')
    actual = evaluate_features(history, {'state': Rating('vectors'), 'home': IsHome()}, ratings={'vectors': loaded})
    assert actual.shape == (4, 5)
    assert actual.iloc[0].tolist() == [2.5, 7., -3., 11., 1.]
    assert actual.iloc[1].tolist() == [-3., 11., 2.5, 7., 0.]
    subset = evaluate_features(history, {'e0': Rating('vectors', side='for', fields=('embedding_0',))}, ratings={'vectors': loaded})
    assert list(subset.columns) == ['e0'] and subset.e0.tolist() == [2.5, -3., 2.5, -3.]


def test_generated_and_reused_ratings_feature_parity_and_period_expansion():
    history = make_history([{}, {'day': 2, 'result': 'D'}, {'day': 4}])
    result = build_ratings(history)
    corner = build_ratings(history, stat=Stat(None, 'Match overview', 'cornerKicks'))
    generated = evaluate_features(history, {'wdl': MatchResultGlicko(),
                                            'corners': StatGlicko(Stat(None, 'Match overview', 'cornerKicks'))})
    reused = evaluate_features(history, {'wdl': Rating('result', fields=('rating', 'rd', 'sigma')),
                                        'corners': Rating('corner', fields=('rating', 'rd', 'sigma'))},
                               ratings={'result': result, 'corner': corner})
    pd.testing.assert_frame_equal(generated, reused, check_exact=True, check_flags=False)
    assert generated.shape == (6, 18)
    assert generated.loc[2, 'wdl::result::team::rating'] > 1500
    assert generated.loc[2, f'corners::{STREAM}::team::rating'] < 1500
    assert generated.loc[3, 'wdl::result::opponent::rating'] == generated.loc[2, 'wdl::result::team::rating']


def test_generated_ratings_honor_named_cutoff_and_explicit_availability():
    history, releases = delayed_history()
    history['release_at'] = releases
    history['prediction_at'] = history.kickoff_at - pd.Timedelta(days=1)
    generated = evaluate_features(history, {'wdl': MatchResultGlicko(engine=PointsEngine(), fields=('points',))},
                                  cutoffs='prediction_at', available_at='release_at')
    expected = build_ratings(history, engine=PointsEngine(), available_at=releases).features(
        history, cutoffs=history.prediction_at, fields=('points',))
    np.testing.assert_array_equal(generated.to_numpy(), expected.to_numpy())
    assert generated.iloc[-2:].to_numpy().tolist() == [[1., 1.], [1., 1.]]


def test_rating_stream_cache_keeps_full_state_across_field_selections(monkeypatch):
    import xdiyo_analytics.ratings as ratings_module
    history = make_history([{}, {}])
    calls = []
    original = ratings_module.build_ratings

    def counted(*args, **kwargs):
        calls.append(kwargs['stat'])
        return original(*args, **kwargs)

    monkeypatch.setattr(ratings_module, 'build_ratings', counted)
    values = evaluate_features(history, {'r': MatchResultGlicko(fields=('rating',)),
                                        'full': MatchResultGlicko(fields=('rating', 'rd', 'sigma', 'mu', 'phi')),
                                        'c': StatGlicko(CORNER, fields=('rating',))})
    assert len(calls) == 2 and calls == [None, CORNER]
    assert values.shape == (4, 14)


@pytest.mark.parametrize('fault', ['duplicate_side', 'missing_away', 'availability_disagrees', 'too_early', 'bad_scope'])
def test_ambiguous_replay_inputs_rejected(fault):
    history = make_history([{}, {}])
    kwargs = {}
    if fault == 'duplicate_side':
        history.loc[1, 'side'] = 'home'
    elif fault == 'missing_away':
        history = history.iloc[:-1]
    elif fault == 'availability_disagrees':
        available = history.kickoff_at.copy()
        available.iloc[0] += pd.Timedelta(hours=2)
        kwargs['available_at'] = available
    elif fault == 'too_early':
        kwargs['available_at'] = history.kickoff_at - pd.Timedelta(seconds=1)
    else:
        kwargs['scope'] = ('team_id',)
    with pytest.raises(ValueError):
        build_ratings(history, **kwargs)


def test_lookup_and_feature_misconfiguration_errors():
    history = make_history([{}, {}])
    run = build_ratings(history)
    with pytest.raises(ValueError):
        run.features(history, fields=('unknown',))
    with pytest.raises(ValueError):
        run.features(history, cutoffs=history.kickoff_at + pd.Timedelta(seconds=1))
    with pytest.raises(ValueError):
        evaluate_features(history, {'h2h': H2H(MatchResultGlicko())})
    with pytest.raises(KeyError):
        evaluate_features(history, {'saved': Rating('absent')})
    with pytest.raises(KeyError):
        build_ratings(history, stat=Stat('ALL', 'Match overview', 'absent'))
    ambiguous = RatingRun(pd.concat([run.snapshots, run.snapshots.iloc[:1]], ignore_index=True),
                          run.initial_state, run.scope, run.streams, run.metadata)
    with pytest.raises(ValueError):
        ambiguous.features(history)
