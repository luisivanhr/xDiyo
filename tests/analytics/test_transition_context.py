"""Season-entry evidence, exact identifiers and frozen availability boundaries."""

import pandas as pd
import pytest

from xdiyo_analytics.features import League, LeaguePopulation, TransitionContext, build_team_seasons
from transition_samples import A, B, STAT, START, history_from_games, league_games, mover_games, warm_games


def test_membership_adapter_requires_complete_destination_and_adjacent_origin_evidence():
    history = history_from_games(mover_games())
    info = {10: {'tier': 1, 'system': 'England'}, 20: {'tier': 2, 'system': 'England'}}
    partial = build_team_seasons(history, info).set_index(['competition_id', 'season_id', 'team_id'])
    assert partial.loc[(10, 2, A), 'movement'] == 'unknown'
    complete = build_team_seasons(history, info, complete_memberships=[(10, 1)])
    promoted = complete.set_index(['competition_id', 'season_id', 'team_id']).loc[(10, 2, A)]
    assert promoted.movement == 'promoted'
    assert promoted.previous_competition_id == 20 and promoted.previous_season_id == 11
    assert 'adjacent_tier' in promoted.evidence
    assert complete.team_id.dtype == object and A in complete.team_id.tolist()
    info[20]['system'] = 'Different system'
    unrelated = build_team_seasons(history, info, complete_memberships=[(10, 1)])
    assert unrelated.set_index(['competition_id', 'season_id', 'team_id']).loc[(10, 2, A), 'movement'] == 'unknown'


def test_explicit_years_and_evidence_override_preserve_predecessor_ids():
    history = history_from_games(mover_games()).drop(columns='source_season')
    years = {(10, 1): (2023, 2024), (20, 11): (2023, 2024), (10, 2): (2024, 2025)}
    override = dict(competition_id=10, season_id=2, team_id=A, movement='promoted',
                    evidence='explicit test membership evidence', previous_competition_id=20, previous_season_id=11)
    result = build_team_seasons(history, {10: {'tier': 1, 'system': 'x'}, 20: {'tier': 2, 'system': 'x'}},
                                season_years=years, overrides=[override])
    record = result.set_index(['competition_id', 'season_id', 'team_id']).loc[(10, 2, A)]
    assert record.previous_season_id == 11 and record.previous_competition_id == 20
    assert record.source.startswith('explicit_override')


def test_explicit_list_with_nulls_keeps_predecessor_ids_above_float_precision():
    history = history_from_games(warm_games())
    previous_comp, previous_season = 2**53 + 33, 2**53 + 49
    table = [dict(competition_id=10, season_id=2, team_id=A, movement='promoted',
                  previous_competition_id=previous_comp, previous_season_id=previous_season),
             dict(competition_id=10, season_id=2, team_id=B, movement='unknown',
                  previous_competition_id=None, previous_season_id=None)]
    context = TransitionContext(history, team_seasons=table)
    assert context.record(4)['previous_competition_id'] == previous_comp
    assert context.record(4)['previous_season_id'] == previous_season
    assert context.record(4)['team_id'] == A


def test_default_anchor_is_earliest_query_and_explicit_table_matches_mapping():
    history = history_from_games(warm_games())
    cutoffs = history.kickoff_at - pd.Timedelta(days=2)
    expected = START + pd.Timedelta(days=8)
    assert TransitionContext(history, cutoffs=cutoffs).record(4)['entry_at'] == expected
    mapping = {(10, 2): expected - pd.Timedelta(days=1)}
    table = pd.DataFrame([dict(competition_id=10, season_id=2, entry_at=mapping[(10, 2)])])
    assert TransitionContext(history, cutoffs=cutoffs, season_starts=table).record(4) == TransitionContext(
        history, cutoffs=cutoffs, season_starts=mapping).record(4)
    with pytest.raises(ValueError, match='first prediction cutoff'):
        TransitionContext(history, cutoffs=cutoffs, season_starts={(10, 2): START + pd.Timedelta(days=10)})


def test_prior_rows_respect_frozen_release_boundary_and_round_count():
    games = warm_games()
    games[1]['release'] = 11
    history = history_from_games(games)
    context = TransitionContext(history, available_at='available_at')
    assert context.prior_rows((10, 1), START + pd.Timedelta(days=10)).tolist() == [0, 1]
    assert context.completed_rounds(4, ('competition_id', 'season_id', 'round')) == 0
    assert context.completed_rounds(6, ('competition_id', 'season_id', 'round')) == 1
    assert context.record(4)['movement'] == 'retained'


def test_population_public_plan_exposes_positional_window_then_exclusion():
    history = history_from_games(league_games())
    history.index = [0] * len(history)
    population = LeaguePopulation(history)
    league = League(STAT)
    assert population.rows(league, 1)[8].tolist() == [4, 5, 6, 7]
    assert population.rows(league, 1, 'team_contributions')[8].tolist() == [5, 6, 7]
    info, row_rounds = population.round_info(league.round_keys)
    assert row_rounds[8] == (10, 1, 3)
    assert info[(10, 1, 3)][1] == START + pd.Timedelta(days=6)


@pytest.mark.parametrize('kind', ['duplicate_movements', 'duplicate_starts', 'invalid_movement'])
def test_ambiguous_context_tables_are_rejected(kind):
    history = history_from_games(warm_games())
    if kind == 'duplicate_starts':
        table = pd.DataFrame([dict(competition_id=10, season_id=2, entry_at=START)] * 2)
        kwargs = {'season_starts': table}
    else:
        record = dict(competition_id=10, season_id=2, team_id=A, movement='unknown' if kind == 'duplicate_movements' else 'new_means_promoted')
        kwargs = {'team_seasons': [record] * (2 if kind == 'duplicate_movements' else 1)}
    with pytest.raises(ValueError):
        TransitionContext(history, **kwargs)
