"""Full-state transitions, independent cohorts and temporal lookup boundaries."""

from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.features import (MatchResultGlicko, Rating, Stat, StatGlicko, WarmStart, evaluate_features)
from xdiyo_analytics.ratings import Glicko2, GlickoTransition, RatingRun, build_ratings, glicko_state
from transition_samples import A, B, C, D, E, STAT, START, history_from_games, mover_games, movement, warm_games


@dataclass(frozen=True)
class VectorEngine:
    def initial_state(self):
        return {'wins': 0., 'opponent_total': 0., 'entries': 0.}

    def update_period(self, state, games):
        return {'wins': state['wins'] + sum(score for score, _ in games),
                'opponent_total': state['opponent_total'] + sum(other['wins'] for _, other in games),
                'entries': state['entries']}


@dataclass(frozen=True)
class VectorTransition:
    increment: float = 10.

    def apply(self, state, *, context, cohort):
        return {**state, 'wins': state['wins'] + self.increment, 'entries': state['entries'] + 1}


def at_entry(run, team, day, stream='result'):
    selected = run.snapshots.loc[(run.snapshots.team_id == team) &
        (run.snapshots.recorded_at == START + pd.Timedelta(days=day)) &
        (run.snapshots.stream == stream) & (run.snapshots.snapshot_kind == 'transition')]
    assert len(selected) == 1
    return selected.iloc[0]


@pytest.mark.parametrize('movement_name,rank,aggregate,expected', [
    ('promoted', 'standings', 'mean', 1500), ('relegated', 'standings', 'mean', 1400),
    ('promoted', 'rating', 'mean', 1200), ('relegated', 'rating', 'median', 1700),
    ('retained', 'rating', 'mean', 1500)])
def test_glicko_adapter_independent_rank_cohort_and_full_state_arithmetic(movement_name, rank, aggregate, expected):
    # Standings and rating order intentionally disagree.
    cohort = [dict(team_id=i, position=i, state=glicko_state(r, 50))
              for i, r in enumerate([1800, 1000, 1600, 1400], start=1)]
    old = glicko_state(1500, 100, .07)
    before = deepcopy((old, cohort))
    policy = GlickoTransition(phi_scale=1.2, movement_phi_scale=1.5, shrinkage=1,
                              top=2, bottom=2, rank_by=rank, aggregate=aggregate)
    actual = policy.apply(old, context={'movement': movement_name, 'has_prior_state': True}, cohort=cohort)
    assert actual['rating'] == expected
    assert actual['rd'] == pytest.approx(120 if movement_name == 'retained' else 180)
    assert actual['mu'] == pytest.approx((expected - 1500)/173.7178)
    assert actual['phi'] == pytest.approx(actual['rd']/173.7178)
    assert actual['sigma'] == .07 and (old, cohort) == before


def test_missing_standings_leave_location_unchanged_but_inflate_uncertainty():
    cohort = [dict(team_id=B, position=None, state=glicko_state(900, 100))]
    policy = GlickoTransition(phi_scale=2, movement_phi_scale=1.5)
    actual = policy.apply(glicko_state(1600, 50, .08), context={'movement': 'promoted', 'has_prior_state': True}, cohort=cohort)
    assert actual['rating'] == 1600 and actual['rd'] == 150 and actual['sigma'] == .08


@pytest.mark.parametrize('aggregate,expected', [('mean', 4400/3), ('median', 1600)])
def test_mean_and_median_cohort_priors_are_distinct(aggregate, expected):
    cohort = [dict(team_id=i, position=i, state=glicko_state(r))
              for i, r in enumerate([1800, 1000, 1600, 500], start=1)]
    actual = GlickoTransition(top=3, shrinkage=1, aggregate=aggregate).apply(
        glicko_state(), context={'movement': 'relegated', 'has_prior_state': True}, cohort=cohort)
    assert actual['rating'] == pytest.approx(expected)


def test_simultaneous_transitions_read_frozen_pretransition_cohort_states():
    class CohortTransition:
        def apply(self, state, *, context, cohort):
            increment = sum(r['state']['wins'] for r in cohort) if cohort else 10
            return {**state, 'wins': state['wins'] + increment, 'entries': state['entries'] + 1}
    history = history_from_games(warm_games())
    run = build_ratings(history, engine=VectorEngine(), transition=CohortTransition())
    assert at_entry(run, A, 10).wins == 22
    assert at_entry(run, B, 10).wins == 22


def test_each_stat_period_replays_and_transitions_its_own_outcomes():
    games = warm_games()
    for game in games:
        game['half'] = (5, 1)  # ALL is a loss for A; 1ST is a win.
    run = build_ratings(history_from_games(games), stat=Stat(None, 'Match overview', 'cornerKicks'),
                        transition=GlickoTransition(phi_scale=1.2))
    all_period = at_entry(run, A, 10, 'ALL::Match overview::cornerKicks::value')
    first_period = at_entry(run, A, 10, '1ST::Match overview::cornerKicks::value')
    assert all_period.rating < 1500 < first_period.rating
    assert all_period.rating + first_period.rating == pytest.approx(3000)


def test_transition_none_is_identical_to_unchanged_replay():
    history = history_from_games(warm_games())
    plain = build_ratings(history)
    explicit = build_ratings(history, transition=None)
    pd.testing.assert_frame_equal(plain.snapshots, explicit.snapshots)
    assert plain.metadata == explicit.metadata


@pytest.mark.parametrize('scope', [('competition_id',), ('competition_id', 'season_id'), ()])
def test_full_state_transitions_precede_equal_time_results_and_transfer_once(scope):
    history = history_from_games(warm_games())
    run = build_ratings(history, engine=VectorEngine(), transition=VectorTransition(), scope=scope)
    assert at_entry(run, A, 10)['wins'] == 22  # initial entry + two wins + one new entry
    assert at_entry(run, A, 10)['entries'] == 2
    first_current = run.features(history, side='for', fields=('wins', 'entries')).iloc[4]
    np.testing.assert_allclose(first_current, [22, 2])  # Excludes the result at exactly this kickoff.
    post = run.snapshots.loc[(run.snapshots.team_id == A) & (run.snapshots.recorded_at == START + pd.Timedelta(days=10))]
    assert post.snapshot_order.tolist() == [0, 1]
    assert post.wins.tolist() == [22, 23]


def test_mover_explicit_predecessor_state_transfer_uses_custom_adapter():
    history = history_from_games(mover_games())
    run = build_ratings(history, engine=VectorEngine(), transition=VectorTransition(), team_seasons=movement())
    entry = at_entry(run, A, 10)
    assert entry.wins == 22 and entry.entries == 2


def test_retained_glicko_transition_preserves_mu_sigma_and_inflates_phi_once():
    history = history_from_games(warm_games())
    baseline = build_ratings(history.iloc[:4])
    previous = baseline.snapshots.loc[baseline.snapshots.team_id == A].iloc[-1]
    run = build_ratings(history, transition=GlickoTransition(phi_scale=1.3))
    entry = at_entry(run, A, 10)
    assert entry.rating == previous.rating and entry.sigma == previous.sigma
    assert entry.rd == pytest.approx(previous.rd*1.3)
    assert entry.mu == previous.mu


def test_mover_uses_frozen_previous_destination_stream_and_departing_members():
    history = history_from_games(mover_games())
    policy = GlickoTransition(shrinkage=.5, bottom=1)
    # D was the bottom-ranked destination member and has no current-season row.
    plain = build_ratings(history)
    prior_a = plain.snapshots.loc[(plain.snapshots.team_id == A) & (plain.snapshots.competition_id == 20)].iloc[-1]
    prior_d = plain.snapshots.loc[(plain.snapshots.team_id == D) & (plain.snapshots.recorded_at < START + pd.Timedelta(days=10))].iloc[-1]
    run = build_ratings(history, transition=policy, team_seasons=movement())
    entry = at_entry(run, A, 10)
    assert entry.rating == pytest.approx((prior_a.rating + prior_d.rating)/2)
    assert entry.sigma == prior_a.sigma
    contaminated = history.copy()
    contaminated.loc[contaminated.season_id == 2, 'team_position'] = 99999
    other = build_ratings(contaminated, transition=policy, team_seasons=movement())
    assert at_entry(other, A, 10).rating == entry.rating


def test_same_time_prior_release_is_not_used_by_transition_but_is_used_afterward():
    games = warm_games()
    games[1]['release'] = 10
    history = history_from_games(games)
    run = build_ratings(history, engine=VectorEngine(), transition=VectorTransition(), available_at='available_at')
    assert at_entry(run, A, 10).wins == 21  # Delayed prior game is released after the entry event.
    assert run.features(history, side='for', fields=('wins',)).iloc[4, 0] == 21


def test_earlier_evaluator_cutoffs_and_explicit_direct_replay_boundaries():
    history = history_from_games(warm_games())
    cutoffs = history.kickoff_at - pd.Timedelta(days=2)
    policy = GlickoTransition(phi_scale=1.4)
    expression = WarmStart(MatchResultGlicko(side='for', fields=('rd',)), policy)
    generated = evaluate_features(history, {'rd': expression}, cutoffs=cutoffs)
    starts = {(10, 1): START - pd.Timedelta(days=2), (10, 2): START + pd.Timedelta(days=8)}
    run = build_ratings(history, transition=policy, season_starts=starts)
    np.testing.assert_allclose(generated.iloc[:, 0], run.features(history, cutoffs=cutoffs, side='for', fields=('rd',)).iloc[:, 0])
    assert at_entry(run, A, 8).recorded_at == cutoffs.iloc[4]


def test_warmed_unwarmed_fields_result_and_stat_periods_have_independent_cache_keys():
    history = history_from_games(warm_games())
    policy = GlickoTransition(phi_scale=1.4)
    expressions = {'plain': MatchResultGlicko(side='for', fields=('rd',)),
        'warm': WarmStart(MatchResultGlicko(side='for', fields=('rd',)), policy),
        'mu': WarmStart(MatchResultGlicko(side='for', fields=('mu',)), policy),
        'stat': WarmStart(StatGlicko(Stat(None, 'Match overview', 'cornerKicks'), side='for', fields=('rd',)), policy)}
    actual = evaluate_features(history, expressions)
    assert actual.warm.iloc[4] == pytest.approx(actual.plain.iloc[4]*1.4)
    run = build_ratings(history, transition=policy)
    assert actual.mu.iloc[4] == at_entry(run, A, 10).mu
    assert len([c for c in actual if c.startswith('stat::')]) == 2
    assert actual.warm.iloc[4] == at_entry(run, A, 10).rd


def test_save_reload_keeps_transition_order_metadata_and_lookup(tmp_path):
    history = history_from_games(warm_games())
    run = build_ratings(history, transition=GlickoTransition(phi_scale=1.2))
    run.save(tmp_path / 'run')
    loaded = RatingRun.load(tmp_path / 'run')
    pd.testing.assert_frame_equal(run.snapshots, loaded.snapshots)
    pd.testing.assert_frame_equal(run.features(history), loaded.features(history))
    assert loaded.metadata == run.metadata
    with pytest.raises(TypeError, match='saved-rating transitions'):
        evaluate_features(history, {'invalid': WarmStart(Rating('saved'), GlickoTransition())}, ratings={'saved': loaded})


def test_custom_engine_rejects_glicko_policy_and_invalid_state_fields():
    history = history_from_games(warm_games())
    with pytest.raises(TypeError, match='own transition adapter'):
        build_ratings(history, engine=VectorEngine(), transition=GlickoTransition())
    class Invalid:
        def apply(self, state, *, context, cohort):
            return {'invented': 1.}
    with pytest.raises(ValueError, match='retain finite numeric state fields'):
        build_ratings(history, engine=VectorEngine(), transition=Invalid())


def test_collapsed_simultaneous_entries_and_unsupported_scope_are_rejected():
    history = history_from_games([dict(day=0, competition=10), dict(day=0, competition=20)])
    with pytest.raises(ValueError, match='merges simultaneous season entries'):
        build_ratings(history, transition=GlickoTransition(), scope=())
    with pytest.raises(ValueError, match='competition_id/season_id'):
        build_ratings(history, transition=GlickoTransition(), scope=('source_season',))


@pytest.mark.parametrize('kwargs', [{'phi_scale': .9}, {'movement_phi_scale': math.inf}, {'shrinkage': 2},
    {'top': 0}, {'bottom': True}, {'rank_by': 'final_table'}, {'aggregate': 'max'}])
def test_invalid_glicko_transition_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        GlickoTransition(**kwargs)
