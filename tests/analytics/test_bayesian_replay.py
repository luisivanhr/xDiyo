"""Chronological score replay and portable rating output contracts."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.ratings import (
    BayesianConfig, BayesianModel, BayesianRatingRun, build_bayesian_ratings,
)
from xdiyo_analytics.ratings.snapshots import RatingRun
from xdiyo_analytics.ratings.bayesian import TEAM_FIELDS, DEFAULT_TEAM_FIELDS
from test_bayesian_training import synthetic_history


def test_selected_fields_preserve_order_and_do_not_mutate_full_state():
    history = synthetic_history(seasons=1, matches=4)
    run = build_bayesian_ratings(history)
    before = deepcopy(run.checkpoint)
    snapshots = run.snapshots.copy(deep=True)
    all_fields = run.features(history)
    assert list(all_fields) == [f"score::{role}::{field}" for role in ("team", "opponent") for field in TEAM_FIELDS]
    defaults = run.features(history, fields=DEFAULT_TEAM_FIELDS)
    assert list(defaults) == [f"score::{role}::{field}" for role in ("team", "opponent")
                              for field in ("attack_mean", "defence_vulnerability_mean")]
    fields = ("strength_index", "attack_sd", "attack_mean")
    selected = run.features(history, side="for", fields=fields)
    assert list(selected) == [f"score::team::{field}" for field in fields]
    pd.testing.assert_series_equal(selected["score::team::attack_mean"], defaults["score::team::attack_mean"])
    fixture = run.fixture_features(history, fields=("p_draw", "expected_away_goals"))
    assert list(fixture) == ["p_draw", "expected_away_goals"]
    # Canonical home/away fixture orientation is identical on both perspectives.
    np.testing.assert_allclose(fixture.iloc[::2], fixture.iloc[1::2])
    pd.testing.assert_frame_equal(snapshots, run.snapshots)
    assert before == run.checkpoint
    for bad in ((), ("attack_mean", "attack_mean"), ("missing",), "attack_mean"):
        with pytest.raises(ValueError):
            run.features(history, fields=bad)


def test_future_scores_cannot_change_earlier_team_or_shared_home_features():
    history = synthetic_history(seasons=1, matches=4)
    changed = history.copy()
    last = changed.event_id.eq(changed.event_id.max())
    changed.loc[last, ["goals_for", "goals_against"]] = 20
    first, second = build_bayesian_ratings(history), build_bayesian_ratings(changed)
    fields = ("expected_home_goals", "home_advantage_mean", "p_draw")
    pd.testing.assert_frame_equal(first.fixture_features(history, fields=fields),
                                  second.fixture_features(history, fields=fields))
    assert first.checkpoint != second.checkpoint
    pd.testing.assert_frame_equal(first.features(history), second.features(history))


def test_delayed_release_at_boundary_usable_but_same_kickoff_proxy_excluded():
    history = synthetic_history(seasons=1, matches=2)
    history["release"] = history.kickoff_at
    first_event = history.event_id.min()
    target_time = history.kickoff_at.max()
    history.loc[history.event_id.eq(first_event), "release"] = target_time
    run = build_bayesian_ratings(history, available_at="release")
    fields = ("attack_shape", "attack_rate")
    observed = run.features(history.tail(2), fields=fields)
    earlier_only = build_bayesian_ratings(history.head(2), available_at="release")
    pd.testing.assert_frame_equal(observed, earlier_only.features(history.tail(2), fields=fields))
    just_before = pd.Series(target_time - pd.Timedelta(seconds=1), index=history.tail(2).index)
    prior = run.features(history.tail(2), cutoffs=just_before, fields=fields)
    assert not observed.equals(prior)


def test_equal_time_matches_are_permutation_invariant_and_counted_once():
    history = synthetic_history(seasons=1, matches=4)
    history["kickoff_at"] = history.kickoff_at.min()
    a = build_bayesian_ratings(history)
    b = build_bayesian_ratings(history.sample(frac=1, random_state=19))
    assert a.checkpoint == b.checkpoint
    assert [row["games_seen"] for row in a.checkpoint["teams"]] == [4, 4]
    assert len(a.predictions) == 4
    assert len(a.metadata["diagnostics"]) == 1


def test_resume_matches_full_same_season_replay_and_rejects_revisions():
    history = synthetic_history(seasons=1, matches=4)
    early = build_bayesian_ratings(history.iloc[:4])
    resumed = early.update(history.iloc[4:])
    full = build_bayesian_ratings(history)
    assert resumed.checkpoint == full.checkpoint
    pd.testing.assert_frame_equal(resumed.snapshots, full.snapshots)
    pd.testing.assert_frame_equal(resumed.predictions, full.predictions)
    with pytest.raises(ValueError, match="strictly after"):
        early.update(history.iloc[:2])


def test_checkpoint_roundtrip_keeps_cutoff_and_detects_corruption(tmp_path):
    history = synthetic_history(seasons=1, matches=3)
    model = BayesianModel(training_cutoff="2020-08-10T00:00:00Z")
    run = build_bayesian_ratings(history, model=model)
    target = run.save(tmp_path / "rating")
    restored = BayesianRatingRun.load(target)
    assert restored.model == model and restored.checkpoint == run.checkpoint
    future = history.loc[history.kickoff_at.ge(model.training_cutoff)]
    pd.testing.assert_frame_equal(restored.fixture_features(future), run.fixture_features(future))
    for rating in (run, restored):
        with pytest.raises(ValueError, match="trained after"):
            rating.features(history)
        with pytest.raises(ValueError, match="trained after"):
            rating.fixture_features(history)
    with pytest.raises(FileExistsError):
        run.save(target)
    with pytest.raises(FileNotFoundError):
        RatingRun.load(target)
    (target / "state.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        BayesianRatingRun.load(target)


def test_failed_save_removes_only_its_private_staging_directory(tmp_path, monkeypatch):
    run = build_bayesian_ratings(synthetic_history(seasons=1, matches=1))
    keep = tmp_path / "keep.txt"
    keep.write_text("existing unrelated file", encoding="utf-8")
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic write failure")
    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        run.save(tmp_path / "new_rating")
    assert list(tmp_path.iterdir()) == [keep]


def test_awards_are_excluded_and_score_basis_must_be_explicit():
    history = synthetic_history(seasons=1, matches=2)
    history["is_awarded"] = history.event_id.eq(history.event_id.max())
    run = build_bayesian_ratings(history)
    assert len(run.predictions) == 1
    assert [row["games_seen"] for row in run.checkpoint["teams"]] == [1, 1]
    model = BayesianModel(config=BayesianConfig(score_basis="regulation"))
    with pytest.raises(ValueError, match="caller-verified"):
        build_bayesian_ratings(history, model=model)
    history.attrs["score_basis"] = "regulation"
    assert len(build_bayesian_ratings(history, model=model).predictions) == 1


@pytest.mark.parametrize("mutation, message", [
    (lambda h: h.drop(index=0), "both perspectives"),
    (lambda h: h.assign(goals_for=0.5), "integers"),
    (lambda h: h.assign(goals_against=7), "paired scores"),
])
def test_invalid_score_history_fails(mutation, message):
    with pytest.raises(ValueError, match=message):
        build_bayesian_ratings(mutation(synthetic_history(seasons=1, matches=2)))
