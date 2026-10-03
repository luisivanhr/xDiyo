"""Calibration activity follows applied entry mechanisms, not movement labels."""

from copy import deepcopy
from dataclasses import replace
import json

import pandas as pd
import pytest

from xdiyo_analytics.ratings.bayesian import (
    BayesianConfig, BayesianModel,
)
from xdiyo_analytics.ratings.bayesian_core import BayesianParameters
from xdiyo_analytics.ratings.bayesian_training import train_bayesian


ENTRY_FIELDS = (
    "entry_attack_shift", "entry_defence_shift", "entry_attack_shape",
    "entry_defence_shape",
)


def exchange_history(seasons=3):
    """Every known team exchanges divisions each season, with two games per entry."""
    rows, movements = [], []
    for season in range(seasons):
        for league, initial_teams in ((10, (101, 102)), (20, (201, 202))):
            teams = initial_teams if season % 2 == 0 else (
                (201, 202) if league == 10 else (101, 102))
            if season:
                movements.extend(dict(
                    competition_id=league, season_id=2020 + season, team_id=team,
                    movement="promoted" if league == 10 else "relegated",
                    previous_competition_id=20 if league == 10 else 10,
                    previous_season_id=2019 + season,
                ) for team in teams)
            for game in range(2):
                home, away = teams if game == 0 else teams[::-1]
                x, y = (2, 1) if game == 0 else (1, 0)
                common = dict(
                    competition_id=league, season_id=2020 + season,
                    source_season=f"{2020 + season}/{2021 + season}",
                    event_id=league * 10000 + season * 100 + game,
                    kickoff_at=pd.Timestamp(f"{2020 + season}-08-01", tz="UTC")
                    + pd.Timedelta(days=game * 7), status="finished",
                )
                rows.extend([
                    dict(common, team_id=home, opponent_id=away, side="home",
                         goals_for=x, goals_against=y),
                    dict(common, team_id=away, opponent_id=home, side="away",
                         goals_for=y, goals_against=x),
                ])
    return pd.DataFrame(rows), movements


def calibrate(history, movements, config, **kwargs):
    return train_bayesian(
        history, cutoff="2023-01-01", config=config, team_seasons=movements,
        min_seasons=1, fit_fields=ENTRY_FIELDS, max_iterations=1, **kwargs,
    )


@pytest.mark.parametrize("legacy_counts", [False, True])
def test_all_bridged_entries_are_inactive_and_reuse_the_initial_replay(monkeypatch, legacy_counts):
    import xdiyo_analytics.ratings.bayesian as public

    history, movements = exchange_history()
    initial = BayesianParameters()
    config = BayesianConfig(transition="bridge", bridge_gaps=((10, 20, .3),))
    before_history, before_movements = history.copy(deep=True), deepcopy(movements)
    calls = []
    original = public.build_bayesian_ratings

    def counted(*args, **kwargs):
        calls.append(kwargs["model"])
        run = original(*args, **kwargs)
        if legacy_counts:
            run.metadata.pop("effective_entry_counts", None)
        return run

    monkeypatch.setattr(public, "build_bayesian_ratings", counted)
    trained = calibrate(history, movements, config, initial=initial)
    assert trained.report["fit_fields"] == []
    assert set(trained.report["inactive_fit_fields"]) == set(ENTRY_FIELDS)
    assert trained.report["entry_mechanism_counts"] == {"mirrored": 0, "bridge": 8}
    assert trained.report["movement_counts"]["promoted"] == 4
    assert trained.report["movement_counts"]["relegated"] == 4
    assert trained.report["fits"][0]["entry_mechanism_counts"] == {"mirrored": 0, "bridge": 8}
    assert trained.report["fits"][0]["optimizer"]["iterations"] == 0
    assert trained.report["fits"][0]["optimizer"]["evaluations"] == 1
    assert len(calls) == 1
    assert trained.model.parameters == initial
    assert "do not establish statistical identification" in trained.report["fit_fields_interpretation"]
    assert json.loads(trained.model.provenance_json)["entry_mechanism_counts"] == {"mirrored": 0, "bridge": 8}
    pd.testing.assert_frame_equal(history, before_history)
    assert movements == before_movements
    assert config == BayesianConfig(transition="bridge", bridge_gaps=((10, 20, .3),))

    # Independent mechanism check: all four coordinates really leave forecasts unchanged.
    changed = replace(initial, entry_attack_shift=1.4, entry_defence_shift=1.2,
                      entry_attack_shape=.4, entry_defence_shape=90.)
    first = original(history, model=BayesianModel(parameters=initial, config=config),
                     team_seasons=movements)
    second = original(history, model=BayesianModel(parameters=changed, config=config),
                      team_seasons=movements)
    pd.testing.assert_frame_equal(first.predictions, second.predictions)


@pytest.mark.parametrize("transition", ["mirrored", "bridge"])
def test_actual_mirrored_entries_activate_parameters_including_missing_bridge_fallback(transition):
    history, movements = exchange_history(seasons=2)
    trained = calibrate(history, movements, BayesianConfig(transition=transition))
    assert trained.report["fit_fields"] == list(ENTRY_FIELDS)
    assert trained.report["inactive_fit_fields"] == {}
    assert trained.report["entry_mechanism_counts"] == {"mirrored": 4, "bridge": 0}


def test_historical_mirrored_prior_stays_active_after_later_bridges():
    history, movements = exchange_history()
    movements.append(dict(competition_id=10, season_id=2020, team_id=101,
                          movement="promoted", previous_competition_id=None,
                          previous_season_id=None))
    config = BayesianConfig(transition="bridge", bridge_gaps=((10, 20, .3),))
    trained = calibrate(history, movements, config)
    assert trained.report["fit_fields"] == list(ENTRY_FIELDS)
    assert trained.report["entry_mechanism_counts"] == {"mirrored": 1, "bridge": 8}


def test_effective_history_counts_take_precedence_over_current_snapshot_kinds(monkeypatch):
    import xdiyo_analytics.ratings.bayesian as public

    history, movements = exchange_history()
    movements.append(dict(competition_id=10, season_id=2020, team_id=101,
                          movement="promoted"))
    original = public.build_bayesian_ratings

    def historical_evidence(*args, **kwargs):
        run = original(*args, **kwargs)
        # Simulate a replay producer whose current snapshots no longer contain
        # a prior application that affected historical fitting predictions.
        run.metadata["effective_entry_counts"] = {"mirrored": 1, "bridge": 8}
        run.snapshots = run.snapshots.loc[run.snapshots.snapshot_kind.ne("mirrored")].copy()
        return run

    monkeypatch.setattr(public, "build_bayesian_ratings", historical_evidence)
    trained = train_bayesian(
        history, cutoff="2023-01-01", team_seasons=movements,
        config=BayesianConfig(transition="bridge", bridge_gaps=((10, 20, .3),)),
        fit_fields=("entry_attack_shift",), max_iterations=1,
    )
    assert trained.report["fit_fields"] == ["entry_attack_shift"]
    assert trained.report["entry_mechanism_counts"] == {"mirrored": 1, "bridge": 8}


def test_univariate_dispersion_remains_inactive_when_entry_parameters_are_active():
    history, movements = exchange_history(seasons=2)
    trained = train_bayesian(
        history, cutoff="2023-01-01", config=BayesianConfig(bivariate=False),
        team_seasons=movements, min_seasons=1,
        fit_fields=("entry_attack_shift", "kappa"), max_iterations=1,
    )
    assert trained.report["fit_fields"] == ["entry_attack_shift"]
    assert set(trained.report["inactive_fit_fields"]) == {"kappa"}
    assert trained.model.parameters.kappa == BayesianParameters().kappa
    assert trained.report["entry_mechanism_counts"] == {"mirrored": 4, "bridge": 0}


def test_per_league_activity_and_mechanism_counts_are_aggregated_without_implying_identification():
    history, movements = exchange_history(seasons=1)
    movements.append(dict(competition_id=10, season_id=2020, team_id=101,
                          movement="promoted"))
    trained = train_bayesian(
        history, cutoff="2023-01-01", mode="per_league", team_seasons=movements,
        min_seasons=1, fit_fields=("entry_attack_shift",), max_iterations=1,
    )
    assert trained.report["fit_fields"] == ["entry_attack_shift"]
    assert trained.report["entry_mechanism_counts"] == {"mirrored": 1, "bridge": 0}
    by_league = {detail["league_counts"][0]["competition_id"]: detail
                 for detail in trained.report["fits"]}
    assert by_league[10]["fit_fields"] == ["entry_attack_shift"]
    assert by_league[20]["fit_fields"] == []
    assert by_league[20]["entry_mechanism_counts"] == {"mirrored": 0, "bridge": 0}
