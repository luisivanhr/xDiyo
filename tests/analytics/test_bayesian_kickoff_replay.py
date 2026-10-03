"""Dated-state refiltering must preserve the versions known at each release."""

import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.ratings.bayesian import (
    BayesianConfig, BayesianModel, BayesianRatingRun, build_bayesian_ratings,
)
from xdiyo_analytics.ratings.bayesian_core import BayesianParameters


def _history(specs):
    """One tuple is event, league, season, kickoff, home, away, x, y, release."""
    rows = []
    for event, league, season, kickoff, home, away, x, y, release in specs:
        common = dict(
            event_id=event, competition_id=league, season_id=season,
            source_season=f"{season}/{season + 1}",
            kickoff_at=pd.Timestamp(kickoff, tz="UTC"),
            released_at=pd.Timestamp(release, tz="UTC"),
            status="finished" if x is not None else "scheduled",
        )
        rows.extend([
            dict(common, team_id=home, opponent_id=away, side="home",
                 goals_for=x, goals_against=y),
            dict(common, team_id=away, opponent_id=home, side="away",
                 goals_for=y, goals_against=x),
        ])
    return pd.DataFrame(rows)


def _model(*, clock="team_match", bridge=False):
    return BayesianModel(
        parameters=BayesianParameters(team_discount=.8, home_discount=.85),
        config=BayesianConfig(
            clock=clock, transition="bridge" if bridge else "mirrored",
            bridge_gaps=((1, 2, .3),) if bridge else (),
        ),
    )


def _assert_same_state(actual, expected):
    """Knowledge timestamps/journals differ; current mathematical states must not."""
    teams = lambda run: {
        (row["competition_id"], row["season_id"], row["team_id"]): row
        for row in run.checkpoint["teams"]
    }
    left, right = teams(actual), teams(expected)
    assert left.keys() == right.keys()
    for key in left:
        np.testing.assert_allclose(left[key]["state"], right[key]["state"], rtol=2e-11, atol=2e-12)
        assert left[key]["games_seen"] == right[key]["games_seen"]
        assert left[key]["last_kickoff_at"] == right[key]["last_kickoff_at"]
    leagues = lambda run: {
        (row["competition_id"], row["season_id"]): row["state"]
        for row in run.checkpoint["leagues"]
    }
    left, right = leagues(actual), leagues(expected)
    assert left.keys() == right.keys()
    for key in left:
        np.testing.assert_allclose(left[key], right[key], rtol=2e-11, atol=2e-12)


def _chronological(history, *, model, team_seasons=None):
    # Independent reference: every score was already known when its own game
    # occurred. Comparing the eventual state isolates event-time recursions.
    return build_bayesian_ratings(
        history, model=model, available_at="kickoff_at", team_seasons=team_seasons,
    )


LATE = (1, 1, 2020, "2020-08-01", 10, 11, 0, 2, "2020-08-20")
KNOWN = [
    (2, 1, 2020, "2020-08-08", 10, 11, 6, 0, "2020-08-08 02:00"),
    (3, 1, 2020, "2020-08-15", 11, 10, 1, 3, "2020-08-15 02:00"),
]


@pytest.mark.parametrize("clock", ["team_match", "days"])
def test_late_same_season_result_refilters_team_and_home_by_kickoff(clock):
    history = _history([LATE, *KNOWN])
    model = _model(clock=clock)
    actual = build_bayesian_ratings(history, model=model, available_at="released_at")
    _assert_same_state(actual, _chronological(history, model=model))


def test_late_disjoint_teams_still_refilter_the_shared_home_state():
    history = _history([
        (1, 1, 2020, "2020-08-01", 10, 11, 0, 0, "2020-08-10"),
        (2, 1, 2020, "2020-08-08", 12, 13, 6, 0, "2020-08-08 02:00"),
    ])
    model = _model()
    actual = build_bayesian_ratings(history, model=model, available_at="released_at")
    _assert_same_state(actual, _chronological(history, model=model))


@pytest.mark.parametrize("clock", ["team_match", "days"])
def test_one_release_with_distinct_kickoffs_preserves_chronological_recency(clock):
    history = _history([
        (1, 1, 2020, "2020-08-01", 10, 11, 2, 0, "2020-08-20"),
        (2, 1, 2020, "2020-08-08", 10, 11, 4, 0, "2020-08-20"),
    ])
    model = _model(clock=clock)
    actual = build_bayesian_ratings(history, model=model, available_at="released_at")
    _assert_same_state(actual, _chronological(history, model=model))


def test_split_releases_of_equal_kickoff_games_eventually_refit_the_joint_batch():
    first = _history([(1, 1, 2020, "2020-08-01", 10, 11, 0, 0, "2020-08-01 02:00")])
    second = _history([(2, 1, 2020, "2020-08-01", 12, 13, 6, 0, "2020-08-02")])
    history = pd.concat([first, second], ignore_index=True)
    model = _model()
    before = build_bayesian_ratings(first, model=model, available_at="released_at")
    resumed = before.update(second, available_at="released_at")
    full = build_bayesian_ratings(history, model=model, available_at="released_at")
    expected = _chronological(history, model=model)
    _assert_same_state(resumed, expected)
    _assert_same_state(full, expected)


def test_refilter_does_not_rewrite_historical_features_or_issued_predictions():
    model = _model()
    known = _history(KNOWN)
    before = build_bayesian_ratings(known, model=model, available_at="released_at")
    query = _history([(99, 1, 2020, "2020-08-16", 10, 11, None, None, "2020-08-16")])
    team_features = before.features(query)
    fixture_features = before.fixture_features(query, fields=(
        "expected_home_goals", "expected_away_goals", "home_advantage_mean", "p_draw",
    ))
    original_predictions = before.predictions.copy(deep=True)
    refreshed = before.update(_history([LATE]), available_at="released_at")
    pd.testing.assert_frame_equal(refreshed.features(query), team_features)
    pd.testing.assert_frame_equal(refreshed.fixture_features(query, fields=tuple(fixture_features)), fixture_features)
    saved_predictions = refreshed.predictions.loc[refreshed.predictions.event_id.isin(known.event_id)]
    pd.testing.assert_frame_equal(saved_predictions.reset_index(drop=True), original_predictions.reset_index(drop=True))
    # A forecast issued after the late result must see the revised current state.
    future = query.assign(kickoff_at=pd.Timestamp("2020-08-21", tz="UTC"))
    assert not refreshed.fixture_features(future).equals(before.fixture_features(future))


def test_old_result_at_target_kickoff_is_usable_without_using_that_targets_result():
    late = _history([(*LATE[:-1], "2020-08-20")])
    known = _history(KNOWN)
    target = _history([(4, 1, 2020, "2020-08-20", 10, 11, 12, 0, "2020-08-20")])
    history = pd.concat([known, late, target], ignore_index=True)
    model = _model()
    actual = build_bayesian_ratings(history, model=model, available_at="released_at")
    eligible = pd.concat([known, late], ignore_index=True)
    expected = _chronological(eligible, model=model)
    fields = ("expected_home_goals", "expected_away_goals", "home_advantage_mean", "p_draw")
    pd.testing.assert_frame_equal(actual.fixture_features(target, fields=fields), expected.fixture_features(target, fields=fields))


@pytest.mark.parametrize("destination,movement", [(1, "retained"), (2, "relegated")])
def test_late_previous_season_result_refreshes_existing_transfers(destination, movement):
    initial = (1, 1, 2020, "2021-05-01", 10, 11, 2, 0, "2021-05-01 02:00")
    late = (2, 1, 2020, "2021-05-08", 10, 11, 0, 5, "2021-08-20")
    following = [
        (3, destination, 2021, "2021-08-01", 10, 11, 1, 2, "2021-08-01 02:00"),
        (4, destination, 2021, "2021-08-08", 10, 11, 3, 0, "2021-08-08 02:00"),
    ]
    declarations = [dict(
        competition_id=destination, season_id=2021, team_id=team, movement=movement,
        previous_competition_id=1, previous_season_id=2020,
    ) for team in (10, 11)]
    model = _model(bridge=destination != 1)
    before = build_bayesian_ratings(
        _history([initial, *following]), model=model,
        available_at="released_at", team_seasons=declarations,
    )
    query = _history([(99, destination, 2021, "2021-08-10", 10, 11, None, None, "2021-08-10")])
    issued = before.fixture_features(query)
    # The checkpoint must retain the accepted movement declarations; a later
    # old score must not require the caller to reconstruct those declarations.
    resumed = before.update(_history([late]), available_at="released_at")
    complete = _history([initial, late, *following])
    expected = _chronological(complete, model=model, team_seasons=declarations)
    _assert_same_state(resumed, expected)
    full = build_bayesian_ratings(
        complete, model=model, available_at="released_at", team_seasons=declarations,
    )
    _assert_same_state(full, expected)
    pd.testing.assert_frame_equal(resumed.fixture_features(query), issued)


def test_native_journal_roundtrip_preserves_late_replay_and_historical_versions(tmp_path):
    model = _model()
    before = build_bayesian_ratings(_history(KNOWN), model=model, available_at="released_at")
    restored = BayesianRatingRun.load(before.save(tmp_path / "before-late"))
    assert restored.checkpoint == before.checkpoint
    late = _history([LATE])
    resumed = restored.update(late, available_at="released_at")
    direct = before.update(late, available_at="released_at")
    _assert_same_state(resumed, direct)
    pd.testing.assert_frame_equal(resumed.snapshots, direct.snapshots)
    pd.testing.assert_frame_equal(resumed.league_snapshots, direct.league_snapshots)
    pd.testing.assert_frame_equal(resumed.predictions, direct.predictions)
    after = BayesianRatingRun.load(resumed.save(tmp_path / "after-late"))
    _assert_same_state(after, resumed)
    query = _history([(99, 1, 2020, "2020-08-16", 10, 11, None, None, "2020-08-16")])
    pd.testing.assert_frame_equal(after.fixture_features(query), before.fixture_features(query))


def test_legacy_checkpoint_can_forecast_but_requires_rebuild_to_assimilate_results(tmp_path):
    original = build_bayesian_ratings(_history(KNOWN), model=_model(), available_at="released_at")
    # Simulate the previously published native state schema: full snapshots
    # and current states, but no retained score or entry journal.
    from dataclasses import replace
    legacy = replace(original, checkpoint={key: value for key, value in original.checkpoint.items()
                                         if key not in {"chronology_schema", "event_journal", "entry_journal", "entry_usage"}})
    restored = BayesianRatingRun.load(legacy.save(tmp_path / "legacy"))
    query = _history([(99, 1, 2020, "2020-08-16", 10, 11, None, None, "2020-08-16")])
    forecast = restored.update(query)
    pd.testing.assert_frame_equal(forecast.fixture_features(query), original.fixture_features(query))
    assert forecast.checkpoint == legacy.checkpoint
    with pytest.raises(ValueError, match="lacks a kickoff replay journal; rebuild full history"):
        restored.update(_history([LATE]), available_at="released_at")


@pytest.mark.parametrize("late_kickoff", ["2020-08-06", "2020-08-01"])
def test_late_result_preserves_explicit_season_start_across_native_reload(tmp_path, late_kickoff):
    model = _model()
    starts = {(1, 2020): pd.Timestamp("2020-08-05", tz="UTC")}
    known = _history(KNOWN)
    initial = build_bayesian_ratings(
        known, model=model, available_at="released_at", season_starts=starts,
    )
    before = BayesianRatingRun.load(initial.save(tmp_path / "explicit-start"))
    late = _history([(1, 1, 2020, late_kickoff, 10, 11, 0, 2, "2020-08-20")])
    if late_kickoff < "2020-08-05":
        # Reinterpreting an inferred start is legitimate; overriding an explicit
        # declaration in order to accept incompatible past data is not.
        with pytest.raises(ValueError, match="season entry|season start|anchor"):
            before.update(late, available_at="released_at")
        return
    resumed = before.update(late, available_at="released_at")
    reference = build_bayesian_ratings(
        pd.concat([known, late], ignore_index=True), model=model,
        available_at="kickoff_at", season_starts=starts,
    )
    _assert_same_state(resumed, reference)
    for entry in resumed.checkpoint["entry_journal"]:
        assert pd.Timestamp(entry["entry_at"]) == starts[(1, 2020)]


def test_forecast_only_new_past_entry_requires_full_history_rebuild():
    before = build_bayesian_ratings(_history(KNOWN), model=_model(), available_at="released_at")
    past_entry = _history([
        (99, 1, 2020, "2020-08-10", 12, 13, None, None, "2020-08-10"),
    ])
    # There is no newly released observation timestamp at which to publish this
    # retrospective metadata change, so it must not silently change the journal.
    with pytest.raises(ValueError, match="rebuild full history|full history rebuild"):
        before.update(past_entry, available_at="released_at")
