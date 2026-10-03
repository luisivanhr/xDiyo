"""Boundary releases, explicit incremental transitions and durable frontiers."""

import pandas as pd
import pytest

from xdiyo_analytics.ratings.bayesian import (
    BayesianConfig, BayesianModel, build_bayesian_ratings,
)


def history(specifications):
    rows = []
    for spec in specifications:
        event, league, season, kickoff, home, away, x, y = spec[:8]
        status = spec[8] if len(spec) > 8 else "finished"
        release = spec[9] if len(spec) > 9 else kickoff
        common = {"competition_id": league, "season_id": season, "event_id": event,
                  "source_season": f"{season}/{season + 1}", "status": status,
                  "kickoff_at": pd.Timestamp(kickoff, tz="UTC"),
                  "released_at": pd.Timestamp(release, tz="UTC")}
        rows.extend([
            dict(common, team_id=home, opponent_id=away, side="home", goals_for=x, goals_against=y),
            dict(common, team_id=away, opponent_id=home, side="away", goals_for=y, goals_against=x),
        ])
    return pd.DataFrame(rows)


def movements(*, destination=1, movement="retained"):
    return [dict(competition_id=destination, season_id=2021, team_id=team, movement=movement,
                 previous_competition_id=1, previous_season_id=2020) for team in (10, 11)]


FIRST = [(1, 1, 2020, "2020-08-01", 10, 11, 5, 0),
         (2, 1, 2020, "2020-08-08", 11, 10, 1, 2)]
SECOND = [(3, 1, 2021, "2021-08-01", 10, 11, 1, 1),
          (4, 1, 2021, "2021-08-08", 11, 10, 2, 0)]


def test_new_season_resume_requires_explicit_prior_context_then_matches_full_replay():
    first = build_bayesian_ratings(history(FIRST))
    with pytest.raises(ValueError, match="explicit movement and both predecessor"):
        first.update(history(SECOND))
    with pytest.raises(ValueError, match="explicit movement and both predecessor"):
        first.update(history(SECOND), team_seasons=[
            dict(competition_id=1, season_id=2021, team_id=team, movement="retained")
            for team in (10, 11)])
    resumed = first.update(history(SECOND), team_seasons=movements())
    full = build_bayesian_ratings(history(FIRST + SECOND), team_seasons=movements())
    assert resumed.checkpoint == full.checkpoint
    pd.testing.assert_frame_equal(resumed.snapshots, full.snapshots)
    pd.testing.assert_frame_equal(resumed.league_snapshots, full.league_snapshots)
    pd.testing.assert_frame_equal(resumed.predictions, full.predictions)


def test_explicit_null_predecessors_acknowledge_a_fresh_prior():
    first = build_bayesian_ratings(history(FIRST))
    fresh = [dict(record, movement="unknown", previous_competition_id=None,
                  previous_season_id=None) for record in movements()]
    resumed = first.update(history(SECOND), team_seasons=fresh)
    entries = resumed.snapshots.loc[resumed.snapshots.recorded_at.eq(pd.Timestamp("2021-08-01", tz="UTC"))
                                    & resumed.snapshots.snapshot_kind.eq("initial")]
    assert len(entries) == 2
    assert entries.games_seen.eq(0).all()
    assert entries.attack_shape.eq(first.model.parameters.attack_shape).all()


@pytest.mark.parametrize("destination,movement", [(1, "retained"), (2, "relegated")])
def test_eligible_boundary_release_precedes_frozen_entry_transfer(destination, movement):
    boundary = "2021-08-01"
    model = BayesianModel(config=BayesianConfig(transition="bridge", bridge_gaps=((1, 2, .3),)))
    source = (1, 1, 2020, "2020-08-01", 10, 11, 5, 0, "finished", boundary)
    query = (2, destination, 2021, boundary, 10, 11, None, None, "scheduled")
    at_boundary = build_bayesian_ratings(history([source, query]), model=model,
                                        available_at="released_at",
                                        team_seasons=movements(destination=destination, movement=movement))
    earlier_source = (*source[:-1], "2021-07-31 23:59:59")
    before_boundary = build_bayesian_ratings(history([earlier_source, query]), model=model,
                                            available_at="released_at",
                                            team_seasons=movements(destination=destination, movement=movement))
    query_history = history([query])
    fields = ("attack_mean", "defence_vulnerability_mean", "attack_sd", "defence_vulnerability_sd")
    pd.testing.assert_frame_equal(at_boundary.features(query_history, fields=fields),
                                  before_boundary.features(query_history, fields=fields))
    pd.testing.assert_frame_equal(at_boundary.fixture_features(query_history),
                                  before_boundary.fixture_features(query_history))
    entries = at_boundary.snapshots.loc[at_boundary.snapshots.snapshot_kind.eq(
        "bridge" if destination == 2 else "retained")]
    assert entries.games_seen.eq(1).all()
    assert at_boundary.snapshots.loc[at_boundary.snapshots.snapshot_kind.eq("result"), "snapshot_order"].eq(-1).all()


@pytest.mark.parametrize("destination,movement", [(1, "retained"), (2, "relegated")])
def test_previous_season_result_after_transition_refreshes_entry(destination, movement):
    source = (1, 1, 2020, "2020-08-01", 10, 11, 5, 0, "finished", "2021-08-02")
    query = (2, destination, 2021, "2021-08-01", 10, 11, None, None, "scheduled")
    model = BayesianModel(config=BayesianConfig(transition="bridge", bridge_gaps=((1, 2, .3),)))
    declarations = movements(destination=destination, movement=movement)
    run = build_bayesian_ratings(history([source, query]), model=model, available_at="released_at",
                                 team_seasons=declarations)
    reference = build_bayesian_ratings(history([source, query]), model=model,
                                       team_seasons=declarations)
    future = history([query]).assign(kickoff_at=pd.Timestamp("2021-08-03", tz="UTC"))
    pd.testing.assert_frame_equal(run.fixture_features(future), reference.fixture_features(future))


def test_closed_bridge_source_is_saved_and_replayed_on_late_observation():
    moved = [(3, 2, 2021, "2021-08-01", 10, 11, 1, 1)]
    model = BayesianModel(config=BayesianConfig(transition="bridge", bridge_gaps=((1, 2, .3),)))
    run = build_bayesian_ratings(history(FIRST + moved), model=model,
                                 team_seasons=movements(destination=2, movement="relegated"))
    assert run.checkpoint["closed_sources"] == [[1, 2020, 10], [1, 2020, 11]]
    delayed = [(7, 1, 2020, "2020-08-15", 10, 11, 2, 2, "finished", "2021-08-02")]
    resumed = run.update(history(delayed), available_at="released_at")
    reference = build_bayesian_ratings(history(FIRST + delayed + moved), model=model,
                                       team_seasons=movements(destination=2, movement="relegated"))
    assert resumed.checkpoint["teams"] == reference.checkpoint["teams"]
    assert resumed.checkpoint["leagues"] == reference.checkpoint["leagues"]
    assert resumed.checkpoint["closed_sources"] == run.checkpoint["closed_sources"]


def test_future_queries_do_not_advance_checkpoint_or_block_current_season_resume():
    future = [(3, 1, 2021, "2021-08-01", 10, 11, None, None, "scheduled")]
    base = build_bayesian_ratings(history(FIRST))
    with_future = build_bayesian_ratings(history(FIRST + future))
    assert with_future.checkpoint == base.checkpoint
    assert with_future.snapshots.recorded_at.max() > pd.Timestamp(with_future.checkpoint["processed_through"])
    new = [(4, 1, 2020, "2020-08-15", 10, 11, 1, 0)]
    resumed = with_future.update(history(new))
    full = build_bayesian_ratings(history(FIRST + new))
    assert resumed.checkpoint == full.checkpoint
    pd.testing.assert_frame_equal(resumed.snapshots, full.snapshots)
    assert resumed.snapshots.recorded_at.max() == pd.Timestamp("2020-08-15", tz="UTC")


def test_no_result_run_has_empty_checkpoint_and_can_assimilate_first_result():
    future = [(1, 1, 2020, "2020-08-01", 10, 11, None, None, "scheduled")]
    empty = build_bayesian_ratings(history(future))
    assert empty.checkpoint["processed_through"] is None
    assert empty.checkpoint["teams"] == empty.checkpoint["leagues"] == []
    assert not empty.snapshots.empty
    resumed = empty.update(history(FIRST[:1]))
    full = build_bayesian_ratings(history(FIRST[:1]))
    assert resumed.checkpoint == full.checkpoint
    pd.testing.assert_frame_equal(resumed.snapshots, full.snapshots)


def test_real_release_after_first_entry_and_equal_time_proxy_are_both_supported():
    first = [(1, 1, 2020, "2020-08-01", 10, 11, 5, 0, "finished", "2020-08-01 02:00")]
    second = [(2, 1, 2020, "2020-08-01 02:00", 10, 11, 1, 1)]
    run = build_bayesian_ratings(history(first + second), available_at="released_at")
    expected = build_bayesian_ratings(history(first), available_at="released_at").fixture_features(history(second))
    actual = run.fixture_features(history(second))
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.expected_home_goals.gt(1).all()
