"""Synthetic-only calibration, population boundaries and native persistence."""

from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.datasets import ModelDataset
from xdiyo_analytics.ratings.bayesian import BayesianConfig, BayesianModel
from xdiyo_analytics.ratings.bayesian_training import (
    BayesianScoreAdapter, BayesianScoreSerializer, train_bayesian,
)
from xdiyo_analytics.training import load_model, refit_model
from xdiyo_analytics.training.contracts import FitContext, PredictionContext


KEYS = ("competition_id", "season_id", "event_id")


def synthetic_history(*, leagues=(10,), seasons=3, matches=3):
    rows = []
    for league in leagues:
        for season in range(seasons):
            for match in range(matches):
                home, away = league * 10 + match % 2, league * 10 + 1 - match % 2
                goals = (match % 3 + (2 if league == 20 else 0), (match + season) % 2)
                kickoff = pd.Timestamp(f"{2020 + season}-08-01", tz="UTC") + pd.Timedelta(days=7 * match, hours=league)
                common = dict(competition_id=league, season_id=2020 + season,
                              source_season=f"{2020 + season}/{2021 + season}",
                              event_id=league * 10000 + season * 100 + match,
                              kickoff_at=kickoff, status="finished", round=match + 1)
                rows += [dict(common, team_id=home, opponent_id=away, side="home", goals_for=goals[0], goals_against=goals[1]),
                         dict(common, team_id=away, opponent_id=home, side="away", goals_for=goals[1], goals_against=goals[0])]
    return pd.DataFrame(rows)


def dataset(history):
    home = history.loc[history.side.eq("home")].reset_index(drop=True)
    metadata = home[[*KEYS, "source_season", "kickoff_at", "team_id", "opponent_id"]].rename(
        columns={"team_id": "home_id", "opponent_id": "away_id"})
    return ModelDataset(pd.DataFrame({"unused": np.arange(len(home), dtype=float)}),
                        home[["goals_for", "goals_against"]].rename(columns={"goals_for": "home_goals", "goals_against": "away_goals"}),
                        metadata, "match", KEYS, KEYS, "match")


def fit_context(data, rows):
    def frame(value):
        result = value.iloc[rows].copy()
        result.index = pd.Index(rows, name="row_position")
        return result
    return FitContext(X=frame(data.X), metadata=frame(data.metadata), y=frame(data.y),
                      layout="match", match_columns=KEYS, fold_id=0)


def prediction_context(data, rows):
    fit = fit_context(data, rows)
    return PredictionContext(fit.X, fit.metadata, "match", KEYS, 0)


def adapter(history=None, **kwargs):
    return BayesianScoreAdapter(history, home_goal_target="home_goals", away_goal_target="away_goals",
                                min_seasons=1, fit_fields=(), **kwargs)


def retained_entries(history):
    records = []
    for (competition, season, team), _ in history.groupby(["competition_id", "season_id", "team_id"]):
        if season > history.loc[history.competition_id.eq(competition), "season_id"].min():
            records.append(dict(competition_id=int(competition), season_id=int(season), team_id=int(team),
                                movement="retained", previous_competition_id=int(competition), previous_season_id=int(season - 1)))
    return records


def test_season_gate_and_parameter_export_are_explicit(tmp_path):
    history = synthetic_history(leagues=(10, 20), seasons=3)
    trained = train_bayesian(history, cutoff="2023-01-01", fit_fields=())
    assert trained.report["success"]
    assert trained.report["population"]["matches"] == 18
    assert trained.report["population"]["leagues"] == 2
    assert trained.model.per_league == ()
    assert trained.report["minimum_observed_seasons"] == 3
    assert all(item["observed_season_boundaries"] == 2 for item in trained.report["population"]["league_counts"])
    assert "caller-audited" in trained.report["population"]["season_completeness"]
    with pytest.raises(ValueError, match="at least 5"):
        train_bayesian(history, cutoff="2023-01-01", mode="per_league", fit_fields=())
    with pytest.raises(ValueError, match="at least 3"):
        train_bayesian(history, cutoff="2021-12-01", fit_fields=())
    independent = train_bayesian(history, cutoff="2023-01-01", mode="per_league", min_seasons=3, fit_fields=())
    assert {key for key, _ in independent.model.per_league} == {10, 20}
    assert len(independent.report["fits"]) == 2
    with pytest.raises(KeyError, match="No calibrated"):
        independent.model.parameters_for(30)
    path = trained.model.save(tmp_path / "parameters.json")
    restored = BayesianModel.load(path)
    assert restored == trained.model
    assert json.loads(restored.provenance_json)["population"]["matches"] == 18


def test_deterministic_small_parameter_optimization_reports_real_evaluations():
    history = synthetic_history(seasons=1, matches=4)
    settings = dict(cutoff="2021-01-01", min_seasons=1, fit_fields=("kappa",), max_iterations=2,
                    objective="score_log_loss")
    first, second = train_bayesian(history, **settings), train_bayesian(history, **settings)
    assert first.model == second.model
    diagnostics = first.report["fits"][0]["optimizer"]
    assert diagnostics["evaluations"] > 1
    assert diagnostics["penalized_objective"] <= diagnostics["initial_objective"]
    assert np.isfinite(diagnostics["mean_negative_log_likelihood"])
    assert 0.1 <= first.model.parameters.kappa <= 100


def test_cutoff_availability_and_future_metadata_cannot_change_training_population():
    history = synthetic_history(seasons=2)
    history["release"] = history.kickoff_at + pd.Timedelta(hours=3)
    # One result in the first season is not available until after the cutoff.
    late = history.event_id.eq(100002)
    history.loc[late, "release"] = pd.Timestamp("2022-01-01", tz="UTC")
    trained = train_bayesian(history, cutoff="2021-01-01", available_at="release", min_seasons=1, fit_fields=())
    assert trained.report["population"]["matches"] == 2
    changed = history.copy()
    changed.loc[changed.season_id.eq(2021), ["goals_for", "goals_against"]] = 999
    future_metadata = pd.DataFrame([dict(competition_id=10, season_id=2021, team_id=100,
                                         movement="invalid future metadata must never be interpreted")])
    again = train_bayesian(changed, cutoff="2021-01-01", available_at="release", min_seasons=1,
                           fit_fields=(), team_seasons=future_metadata)
    assert trained.model == again.model


def test_adapter_uses_exact_fit_identities_and_only_y_for_fit_outcomes():
    history = synthetic_history(seasons=2, matches=4)
    data = dataset(history)
    context = fit_context(data, [2, 0])
    # Source includes both excluded earlier results and future results.
    changed = history.copy()
    changed[["goals_for", "goals_against"]] = 88
    changed["status"] = "scheduled"
    first, second = adapter(history, team_seasons=retained_entries(history)), adapter(changed, team_seasons=retained_entries(history))
    first.fit(context)
    second.fit(context)
    assert first.model_ == second.model_
    assert first.training_summary_["population"]["matches"] == 2
    assert set(first.run_.predictions.event_id) == set(data.metadata.event_id.iloc[[2, 0]])
    prediction = prediction_context(data, [7, 5])
    for key, value in first.predict(prediction).items():
        pd.testing.assert_frame_equal(value, second.predict(prediction)[key])
    # A real change in y must change filtered states, even if source is unchanged.
    other_context = deepcopy(context)
    other_context.y.iloc[0, 0] = 8
    third = adapter(history)
    third.fit(other_context)
    assert not first.run_.snapshots.equals(third.run_.snapshots)


def test_prediction_is_frozen_and_native_model_roundtrip_has_full_checkpoint(tmp_path, monkeypatch):
    history = synthetic_history(seasons=3, matches=3)
    data = dataset(history)
    fitted = refit_model(data, lambda: adapter(team_seasons=retained_entries(history),
                                              season_starts={(10, 2022): pd.Timestamp("2022-07-01", tz="UTC")}),
                         train_positions=np.arange(6))
    # No source history is needed: match metadata and two y columns suffice.
    assert fitted.model.history is None
    future = deepcopy(data)
    future.y = pd.DataFrame(np.nan, index=data.y.index, columns=data.y.columns)
    rows = np.array([8, 6, 7])
    before = deepcopy(fitted.model.run_.checkpoint)
    expected = fitted.predict(future, positions=rows)
    assert fitted.model.run_.checkpoint == before
    assert expected["predict"].columns.tolist() == ["home_goals", "away_goals"]
    np.testing.assert_allclose(expected["predict_proba"].sum(axis=1), 1, atol=1e-9)
    serializer = BayesianScoreSerializer()
    path = fitted.save(tmp_path / "native", serializer=serializer)
    monkeypatch.setattr(BayesianScoreAdapter, "fit", lambda *args: pytest.fail("Loading must never fit"))
    restored = load_model(path, serializer=serializer)
    assert restored.model.run_.checkpoint == before
    pd.testing.assert_frame_equal(restored.model.run_.league_snapshots, fitted.model.run_.league_snapshots)
    for key, value in restored.predict(future, positions=rows).items():
        pd.testing.assert_frame_equal(value, expected[key])
    assert not list(path.rglob("*.joblib"))
    with pytest.raises(ValueError, match="trained after"):
        restored.predict(future, positions=[0])


@pytest.mark.parametrize("problem", ["layout", "targets", "fractional", "negative", "weights"])
def test_adapter_rejects_unsupported_training_contracts(problem):
    data = dataset(synthetic_history(seasons=1))
    context = fit_context(data, [0, 1])
    if problem == "layout":
        context.layout = "team_match"
    elif problem == "targets":
        context.y = context.y.rename(columns={"home_goals": "home_win"})
    elif problem == "fractional":
        context.y = context.y.astype(float)
        context.y.iloc[0, 0] = 0.5
    elif problem == "negative":
        context.y.iloc[0, 0] = -1
    else:
        context.sample_weight = np.ones(2)
    with pytest.raises(ValueError):
        adapter().fit(context)


def test_univariate_calibration_does_not_optimize_unused_dispersion():
    trained = train_bayesian(synthetic_history(seasons=1), cutoff="2021-01-01", min_seasons=1,
                             config=BayesianConfig(bivariate=False), fit_fields=("kappa",))
    assert trained.report["fit_fields"] == []
    assert "kappa" in trained.report["inactive_fit_fields"]
    assert trained.report["fits"][0]["optimizer"]["iterations"] == 0


def test_independent_calibration_rejects_cross_league_bridge_and_explicit_availability_is_retained():
    history = synthetic_history(seasons=1)
    with pytest.raises(ValueError, match="requires mirrored"):
        train_bayesian(history, cutoff="2021-01-01", mode="per_league", min_seasons=1,
                       config=BayesianConfig(transition="bridge"), fit_fields=())
    data = dataset(history)
    data.metadata["released_at"] = data.metadata.kickoff_at + pd.Timedelta(hours=4)
    model = adapter(available_at="released_at")
    model.fit(fit_context(data, [0, 1]))
    assert pd.Timestamp(model.model_.training_cutoff) == data.metadata.released_at.iloc[1]
    assert model.training_summary_["availability"] == "explicit"


def test_awarded_fixtures_are_excluded_by_direct_calibration_but_rejected_from_exact_adapter_fit():
    history = synthetic_history(seasons=1)
    history["is_awarded"] = history.event_id.eq(100001)
    trained = train_bayesian(history, cutoff="2021-01-01", min_seasons=1, fit_fields=())
    assert trained.report["population"]["matches"] == 2
    context = fit_context(dataset(history), [0, 1])
    with pytest.raises(ValueError, match="awarded"):
        adapter(history).fit(context)
    context.metadata["is_awarded"] = [False, True]
    with pytest.raises(ValueError, match="awarded"):
        adapter().fit(context)


def test_source_fit_does_not_parse_excluded_future_times_and_checks_selected_team_identity():
    history = synthetic_history(seasons=2)
    data = dataset(history)
    context = fit_context(data, [0, 1])
    changed = history.copy()
    changed["kickoff_at"] = changed.kickoff_at.astype(object)
    changed.loc[changed.season_id.eq(2021), "kickoff_at"] = "invalid excluded future timestamp"
    first, second = adapter(history), adapter(changed)
    first.fit(context)
    second.fit(context)
    assert first.model_ == second.model_
    wrong = deepcopy(context)
    wrong.metadata.iloc[0, wrong.metadata.columns.get_loc("home_id")] = 999
    with pytest.raises(ValueError, match="disagree"):
        adapter(history).fit(wrong)


def test_entry_fields_without_movement_evidence_are_inactive_not_learned():
    history = synthetic_history(seasons=1)
    fields = ("entry_attack_shift", "entry_defence_shift", "entry_attack_shape", "entry_defence_shape")
    trained = train_bayesian(history, cutoff="2021-01-01", min_seasons=1, fit_fields=fields)
    assert trained.report["fit_fields"] == []
    assert set(trained.report["inactive_fit_fields"]) == set(fields)
    assert trained.report["movement_counts"] == {"unknown": 2}
    assert trained.report["fits"][0]["optimizer"]["iterations"] == 0
    labels = [dict(competition_id=10, season_id=2020, team_id=100, movement="promoted")]
    informed = train_bayesian(history, cutoff="2021-01-01", min_seasons=1,
                              fit_fields=("entry_attack_shift",), team_seasons=labels, max_iterations=1)
    assert informed.report["fit_fields"] == ["entry_attack_shift"]
    assert informed.report["movement_counts"]["promoted"] == 1


def test_forecasts_apply_mirrored_relegation_without_outcomes_or_checkpoint_mutation(tmp_path):
    history = synthetic_history(leagues=(10, 20), seasons=1)
    data = dataset(history)
    movements = [
        dict(competition_id=20, season_id=2021, team_id=100, movement="relegated",
             previous_competition_id=10, previous_season_id=2020),
        dict(competition_id=20, season_id=2021, team_id=201, movement="retained",
             previous_competition_id=20, previous_season_id=2020),
        # Malformed later metadata must remain irrelevant to fit and this query.
        dict(competition_id=20, season_id=2022, team_id=100, movement="invalid future movement"),
    ]
    model = adapter(team_seasons=movements, season_starts={(20, 2021): pd.Timestamp("2021-07-01", tz="UTC")})
    model.fit(fit_context(data, list(range(6))))
    forecast = prediction_context(data, [5])
    forecast.metadata.loc[:, "season_id"] = 2021
    forecast.metadata.loc[:, "source_season"] = "2021/2022"
    forecast.metadata.loc[:, "home_id"] = 100
    forecast.metadata.loc[:, "away_id"] = 201
    forecast.metadata.loc[:, "kickoff_at"] = pd.Timestamp("2021-08-01", tz="UTC")
    checkpoint = deepcopy(model.run_.checkpoint)
    predictions = model.predict(forecast)
    assert model.run_.checkpoint == checkpoint
    # Source history is not consulted by predict, even if its accessor is unusable.
    model.history = object()
    pd.testing.assert_frame_equal(model.predict(forecast)["predict"], predictions["predict"])
    own_attack = np.exp(model.model_.parameters.entry_attack_shift)
    prior_away_defence = model.run_.snapshots.loc[
        model.run_.snapshots.competition_id.eq(20) & model.run_.snapshots.team_id.eq(201)
    ].iloc[-1].defence_vulnerability_mean
    home_advantage = model.run_.league_snapshots.loc[model.run_.league_snapshots.competition_id.eq(20)].iloc[-1]
    expected = own_attack * prior_away_defence * home_advantage.home_shape / home_advantage.home_rate
    assert predictions["predict"].iloc[0].home_goals == pytest.approx(expected)
    serializer = BayesianScoreSerializer()
    serializer.save(model, tmp_path / "forecast")
    restored = serializer.load(tmp_path / "forecast")
    assert restored.run_.checkpoint == checkpoint
    for output, values in predictions.items():
        pd.testing.assert_frame_equal(restored.predict(forecast)[output], values)
    assert restored.run_.checkpoint == checkpoint


def test_known_team_new_season_requires_explicit_movement_and_preserves_omitted_predecessors(tmp_path):
    history = synthetic_history(seasons=2)
    data = dataset(history)
    fit, future = fit_context(data, [0, 1, 2]), prediction_context(data, [3])
    model = adapter()
    model.fit(fit)
    with pytest.raises(ValueError, match="(?i)(explicit|movement|predecessor)"):
        model.predict(future)
    # Another record's predecessor columns cannot turn an omitted predecessor
    # into an explicitly declared null predecessor for this team.
    records = [dict(competition_id=10, season_id=2021, team_id=100, movement="retained"),
               dict(competition_id=10, season_id=2021, team_id=101, movement="retained",
                    previous_competition_id=10, previous_season_id=2020)]
    model = adapter(team_seasons=records)
    model.fit(fit)
    serializer = BayesianScoreSerializer()
    serializer.save(model, tmp_path / "missing-predecessor")
    restored = serializer.load(tmp_path / "missing-predecessor")
    for candidate in (model, restored):
        with pytest.raises(ValueError, match="(?i)(explicit|movement|predecessor)"):
            candidate.predict(future)


@pytest.mark.parametrize("basis", ["provider_current", "regulation"])
def test_same_season_forecast_matches_frozen_state_and_ignores_metadata_outcomes(basis):
    history = synthetic_history(seasons=1)
    history.attrs["score_basis"] = basis
    data = dataset(history)
    model = adapter(history, config=BayesianConfig(score_basis=basis))
    model.fit(fit_context(data, [0, 1]))
    context = prediction_context(data, [2])
    snapshot = model.run_.snapshots.copy(deep=True)
    before = deepcopy(model.run_.checkpoint)
    fields = ("expected_home_goals", "expected_away_goals", "p_home_win", "p_draw", "p_away_win")
    query = history.loc[history.event_id.eq(data.metadata.event_id.iloc[2]) & history.side.eq("home")]
    expected = model.run_.fixture_features(query, fields=fields).iloc[0]
    context.metadata["goals_for"], context.metadata["goals_against"] = 99, 88
    context.metadata["status"] = "finished"
    actual = model.predict(context)
    assert actual["predict"].iloc[0].home_goals == pytest.approx(expected.expected_home_goals)
    np.testing.assert_allclose(actual["predict_proba"].iloc[0], expected[["p_home_win", "p_draw", "p_away_win"]])
    assert model.run_.checkpoint == before
    pd.testing.assert_frame_equal(model.run_.snapshots, snapshot)
