"""Equivalence to the frozen b0ecb580 replay, not a second new implementation."""

from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from xdiyo_analytics.ratings.bayesian import BayesianConfig, BayesianModel, build_bayesian_ratings
from xdiyo_analytics.ratings.bayesian_core import BayesianParameters, predict_score_summary
from xdiyo_analytics.ratings.bayesian_calibration import CalibrationReplay
from xdiyo_analytics.ratings.bayesian_probability import outcome_probabilities
from xdiyo_analytics.ratings.bayesian_training import DEFAULT_FIT_FIELDS, _restrict_transitions, train_bayesian
from test_bayesian_calibration_activity import exchange_history
from test_bayesian_kickoff_replay import _history


FIXTURES = Path(__file__).with_name("fixtures")
spec = spec_from_file_location("xdiyo_analytics.ratings._b0ecb580_reference", FIXTURES / "bayesian_replay_b0ecb580.py")
REFERENCE = module_from_spec(spec)
spec.loader.exec_module(REFERENCE)
PARAMETERS = [
    BayesianParameters(),
    BayesianParameters(team_discount=.91, season_discount=.67, home_discount=.94,
                       home_season_discount=.81, kappa=2.3, entry_attack_shift=.31,
                       entry_defence_shift=.43, entry_attack_shape=3.4, entry_defence_shape=7.1),
    BayesianParameters(team_discount=.997, season_discount=.93, home_discount=.998,
                       home_season_discount=.96, kappa=65., entry_attack_shift=.12,
                       entry_defence_shift=.09, entry_attack_shape=22., entry_defence_shape=35.),
]


def original_loss(run, objective="outcome_log_loss"):
    p = run.predictions
    if objective == "score_log_loss":
        return float(-p.score_log_probability.to_numpy().mean())
    outcome = np.where(p.home_goals > p.away_goals, 0, np.where(p.home_goals == p.away_goals, 1, 2))
    probabilities = p[["p_home_win", "p_draw", "p_away_win"]].to_numpy()
    return float(-np.log(np.maximum(probabilities[np.arange(len(p)), outcome], np.finfo(float).tiny)).mean())


def assert_evidence(fast, reference):
    p = reference.predictions
    # Bit-for-bit evidence matters for all finite-difference optimizer probes.
    np.testing.assert_array_equal(fast.home, p.expected_home_goals)
    np.testing.assert_array_equal(fast.away, p.expected_away_goals)
    np.testing.assert_array_equal(fast.probabilities, p[["p_home_win", "p_draw", "p_away_win"]])
    np.testing.assert_array_equal(fast.score_log, p.score_log_probability)
    assert fast.loss == original_loss(reference)
    for item in reference.checkpoint["teams"]:
        state = fast.states.teams[(item["competition_id"], item["team_id"])][-1][2]
        assert [state.attack.shape, state.attack.rate, state.defence.shape, state.defence.rate] == item["state"]
    for item in reference.checkpoint["leagues"]:
        state = fast.states.leagues[item["competition_id"]][-1][2]
        assert [state.shape, state.rate] == item["state"]


@pytest.mark.parametrize("parameters", PARAMETERS)
@pytest.mark.parametrize("transition", ["mirrored", "bridge"])
@pytest.mark.parametrize("late", [False, True])
def test_forecasts_posteriors_checkpoints_and_late_releases(parameters, transition, late):
    history, movements = exchange_history(seasons=2)
    history["release"] = history.kickoff_at + pd.Timedelta(hours=2)
    if late:
        event = history.event_id.iloc[0]
        history.loc[history.event_id.eq(event), "release"] = pd.Timestamp("2021-08-04", tz="UTC")
    model = BayesianModel(parameters=parameters, config=BayesianConfig(
        transition=transition, bridge_gaps=((10, 20, .3),)))
    plan = CalibrationReplay(history, config=model.config, cutoff="2022-01-01T00:00:00Z",
                             available_at="release", team_seasons=movements)
    reference = REFERENCE.replay(history, model=model, available_at="release", team_seasons=movements)
    actual = build_bayesian_ratings(history, model=model, available_at="release", team_seasons=movements)
    assert actual.checkpoint == reference.checkpoint
    pd.testing.assert_frame_equal(actual.predictions, reference.predictions)
    assert_evidence(plan.evaluate(model), reference)


@pytest.mark.parametrize("parameters", PARAMETERS[1:])
@pytest.mark.parametrize("transition", ["mirrored", "bridge"])
def test_real_cross_league_season_boundaries(parameters, transition):
    fixture = json.loads((FIXTURES / "bayesian_real_boundary.json").read_text())
    history = pd.DataFrame(fixture["history"])
    history["kickoff_at"] = pd.to_datetime(history.kickoff_at, utc=True)
    history["release"] = history.kickoff_at + pd.Timedelta(hours=2)
    movements, _ = _restrict_transitions(history, fixture["team_seasons"], None)
    model = BayesianModel(parameters=parameters, config=BayesianConfig(
        transition=transition, bridge_gaps=((17, 18, .3),)))
    plan = CalibrationReplay(history, config=model.config, cutoff="2022-01-01T00:00:00Z",
                             available_at="release", team_seasons=movements)
    reference = REFERENCE.replay(history, model=model, available_at="release", team_seasons=movements)
    assert_evidence(plan.evaluate(model), reference)
    assert build_bayesian_ratings(history, model=model, available_at="release",
                                  team_seasons=movements).checkpoint == reference.checkpoint


def test_mirrored_cross_league_boundary_does_not_refilter_but_bridge_does():
    from xdiyo_analytics.ratings.bayesian_replay import _needs_refilter
    history = _history([
        (1, 1, 2020, "2021-07-25", 10, 11, 2, 1, "2021-08-03"),
        (2, 2, 2021, "2021-08-01", 10, 12, 1, 0, "2021-08-01 02:00"),
        (3, 2, 2021, "2021-08-08", 12, 10, 0, 0, "2021-08-08 02:00"),
    ])
    movements = [dict(competition_id=2, season_id=2021, team_id=10, movement="promoted",
                      previous_competition_id=1, previous_season_id=2020)]
    for transition, expected in (("mirrored", False), ("bridge", True)):
        model = BayesianModel(config=BayesianConfig(transition=transition, bridge_gaps=((1, 2, .3),)))
        plan = CalibrationReplay(history, config=model.config, cutoff="2022-01-01T00:00:00Z",
                                 available_at="released_at", team_seasons=movements)
        assert plan.refilter == expected
        assert REFERENCE._needs_refilter(plan.events, plan.schedule.entries, None)
        assert _needs_refilter(plan.events, plan.schedule.entries, None, model.config) == expected
        reference = REFERENCE.replay(history, model=model, available_at="released_at", team_seasons=movements)
        assert_evidence(plan.evaluate(model), reference)
        actual = build_bayesian_ratings(history, model=model, available_at="released_at", team_seasons=movements)
        assert actual.checkpoint == reference.checkpoint


@pytest.mark.parametrize("method,clock,bivariate", [("vb", "team_match", True), ("vb", "days", True), ("one_step", "days", False)])
@pytest.mark.parametrize("repeated", [False, True])
def test_simultaneous_and_repeated_team_batches(method, clock, bivariate, repeated):
    history = _history([
        (1, 1, 2020, "2020-08-01", 10, 11, 2, 1, "2020-08-01"),
        (2, 1, 2020, "2020-08-01", 10 if repeated else 12, 13, 0, 3, "2020-08-01"),
        (3, 1, 2020, "2020-08-08", 10, 11, 1, 0, "2020-08-08 02:00"),
        (4, 1, 2020, "2020-08-08", 12, 13, 2, 2, "2020-08-09"),
    ])
    model = BayesianModel(parameters=PARAMETERS[1], config=BayesianConfig(method=method, clock=clock, bivariate=bivariate))
    plan = CalibrationReplay(history, config=model.config, cutoff="2021-01-01T00:00:00Z", available_at="released_at")
    assert_evidence(plan.evaluate(model), REFERENCE.replay(history, model=model, available_at="released_at"))


@pytest.mark.parametrize("parameters", PARAMETERS[1:])
def test_all_nine_encoded_finite_difference_directions(parameters):
    from xdiyo_analytics.ratings.bayesian_training import _LOG_FIELDS
    history, movements = exchange_history(seasons=2)
    config = BayesianConfig()
    plan = CalibrationReplay(history, config=config, cutoff="2022-01-01T00:00:00Z", team_seasons=movements)
    for name in DEFAULT_FIT_FIELDS:
        center = np.log(getattr(parameters, name)) if name in _LOG_FIELDS else getattr(parameters, name)
        losses = []
        for offset in (-1e-8, 1e-8):
            value = np.exp(center + offset) if name in _LOG_FIELDS else center + offset
            model = BayesianModel(parameters=replace(parameters, **{name: value}), config=config)
            fast = plan.evaluate(model)
            reference = REFERENCE.replay(history, model=model, team_seasons=movements)
            assert_evidence(fast, reference)
            losses.append((fast.loss, original_loss(reference)))
        assert (losses[1][0] - losses[0][0]) / 2e-8 == (losses[1][1] - losses[0][1]) / 2e-8


def test_batched_probabilities_preserve_support_summation_and_extreme_fallbacks():
    rng = np.random.default_rng(7)
    h, a = rng.uniform(.01, 8, (2, 150))
    k = rng.uniform(.1, 100, 150)
    h[:3], a[:3], k[:3] = [0, 1, 4], [0, 3, 2], [2, 1e18, np.nan]
    actual = outcome_probabilities(h, a, k, tail_tolerance=1e-12)
    expected = [predict_score_summary(x, y, None if np.isnan(z) else z, tail_tolerance=1e-12)
                for x, y, z in zip(h, a, k)]
    np.testing.assert_array_equal(actual, [[p[n] for n in ("p_home_win", "p_draw", "p_away_win")] for p in expected])
    with pytest.raises(ValueError, match="max_total"):
        outcome_probabilities([10.], [9.], [2.], max_total=2)


@pytest.mark.parametrize("late", [False, True])
def test_trials_reuse_structure_without_frames_or_event_validation(monkeypatch, late):
    import xdiyo_analytics.ratings.bayesian_replay as replay
    history, movements = exchange_history(seasons=2)
    history["release"] = history.kickoff_at
    if late:
        history.loc[history.event_id.eq(history.event_id.iloc[0]), "release"] += pd.Timedelta(days=9)
    plan = CalibrationReplay(history, config=BayesianConfig(), cutoff="2022-01-01T00:00:00Z",
                             available_at="release", team_seasons=movements)
    def forbidden(*args, **kwargs):
        raise AssertionError("Trial rebuilt a DataFrame, validated events, or constructed a report")
    monkeypatch.setattr(replay, "_events", forbidden)
    monkeypatch.setattr(replay, "team_summary", forbidden)
    monkeypatch.setattr(pd, "DataFrame", forbidden)
    for parameters in PARAMETERS:
        assert np.isfinite(plan.evaluate(BayesianModel(parameters=parameters)).loss)


def test_plan_owns_inputs_and_rejects_different_configuration():
    history, movements = exchange_history(seasons=2)
    plan = CalibrationReplay(history, config=BayesianConfig(), cutoff="2022-01-01T00:00:00Z", team_seasons=movements)
    before = plan.evaluate(BayesianModel()).loss
    history.loc[:, ["goals_for", "goals_against"]] = 7
    movements[0]["movement"] = "unknown"
    assert plan.evaluate(BayesianModel()).loss == before
    changed = CalibrationReplay(history, config=BayesianConfig(), cutoff="2022-01-01T00:00:00Z", team_seasons=movements)
    assert changed.evaluate(BayesianModel()).loss != before
    with pytest.raises(ValueError, match="rebuild"):
        plan.evaluate(BayesianModel(config=BayesianConfig(clock="days")))
    with pytest.raises(ValueError, match="cutoff"):
        CalibrationReplay(history, config=BayesianConfig(), cutoff="2021-01-01T00:00:00Z", team_seasons=movements)


def test_calibration_keeps_optimizer_contract_and_checks_final_python_objective(monkeypatch):
    import scipy.optimize
    history, movements = exchange_history(seasons=2)
    seen = []
    original = scipy.optimize.minimize
    def minimize(*args, **kwargs):
        assert kwargs["method"] == "L-BFGS-B"
        assert kwargs["options"] == {"maxiter": 100, "ftol": 1e-8}
        assert len(args[1]) == 9
        seen.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(scipy.optimize, "minimize", minimize)
    trained = train_bayesian(history, cutoff="2022-01-01", min_seasons=1, team_seasons=movements)
    assert seen and trained.report["regularization"] == .01
    assert trained.report["fits"][0]["optimizer"]["reference_objective_verified"]
    reference = REFERENCE.replay(history, model=trained.model, team_seasons=movements)
    assert trained.report["fits"][0]["optimizer"]["reference_mean_negative_log_likelihood"] == original_loss(reference)
    assert trained.run.checkpoint == reference.checkpoint


def test_fitted_parameters_match_original_objective_with_all_nine_coordinates():
    from scipy.optimize import minimize
    from xdiyo_analytics.ratings.bayesian_training import _BOUNDS, _LOG_FIELDS
    history, movements = exchange_history(seasons=2)
    initial = PARAMETERS[1]
    trained = train_bayesian(history, cutoff="2022-01-01", initial=initial,
                             min_seasons=1, team_seasons=movements)
    encode = lambda n, x: np.log(x) if n in _LOG_FIELDS else x
    names = DEFAULT_FIT_FIELDS
    center = np.array([encode(n, getattr(initial, n)) for n in names])
    bounds = [(encode(n, a), encode(n, b)) for n in names for a, b in [_BOUNDS[n]]]
    scales = np.array([b-a for a, b in bounds])
    best = [np.inf, initial]
    def loss(vector, first=False):
        params = initial if first else replace(initial, **{
            n: float(np.exp(v) if n in _LOG_FIELDS else v) for n, v in zip(names, vector)})
        reference = REFERENCE.replay(history, model=BayesianModel(parameters=params), team_seasons=movements)
        value = original_loss(reference) + .01 * float(np.sum(((vector-center)/scales)**2))
        if value < best[0]:
            best[:] = [value, params]
        return value
    loss(center, first=True)
    minimize(loss, center, method="L-BFGS-B", bounds=bounds, options={"maxiter": 100, "ftol": 1e-8})
    assert trained.model.parameters == best[1]
    assert trained.report["fits"][0]["optimizer"]["penalized_objective"] == best[0]
    assert trained.report["fits"][0]["optimizer"]["reference_penalized_objective"] == best[0]


def test_plan_lifetime_covers_availability_metadata_anchors_and_cutoff():
    history, movements = exchange_history(seasons=2)
    history["release"] = history.kickoff_at
    settings = dict(config=BayesianConfig(), cutoff="2022-01-01T00:00:00Z",
                    available_at="release", team_seasons=movements)
    first = CalibrationReplay(history, **settings)
    before = first.evaluate(BayesianModel()).loss
    history.loc[history.event_id.eq(history.event_id.iloc[0]), "release"] += pd.Timedelta(days=9)
    changed = CalibrationReplay(history, **settings)
    assert changed.refilter and changed.evaluate(BayesianModel()).loss != before
    assert first.evaluate(BayesianModel()).loss == before
    # An explicitly earlier season entry changes the static schedule, and is
    # not accidentally reused from a previous fit of the same population.
    anchors = {(10, 2021): pd.Timestamp("2021-07-01", tz="UTC")}
    anchored = CalibrationReplay(history, **settings, season_starts=anchors)
    assert anchored.schedule.times != changed.schedule.times
    from xdiyo_analytics.ratings.bayesian_training import _training_history
    subset = _training_history(history, pd.Timestamp("2021-01-01", tz="UTC"), "release")
    early = CalibrationReplay(subset, config=BayesianConfig(), cutoff="2021-01-01T00:00:00Z",
                              available_at="_bayesian_available_at")
    assert len(early.events) < len(first.events)


def test_final_native_verification_fails_closed(monkeypatch):
    original = CalibrationReplay.evaluate
    def biased(self, *args, **kwargs):
        evidence = original(self, *args, **kwargs)
        evidence.loss += .01
        return evidence
    monkeypatch.setattr(CalibrationReplay, "evaluate", biased)
    history, movements = exchange_history(seasons=2)
    with pytest.raises(RuntimeError, match="original Python replay"):
        train_bayesian(history, cutoff="2022-01-01", min_seasons=1,
                       fit_fields=(), team_seasons=movements)


def test_late_versions_share_unchanged_batches_instead_of_copying_all_prefixes():
    from xdiyo_analytics.ratings.bayesian_calibration import _tree_leaves
    history = _history([(i + 1, 1, 2020,
                         (pd.Timestamp("2020-08-01") + pd.Timedelta(days=i)).isoformat(),
                         10, 11, i % 3, i % 2,
                         (pd.Timestamp("2020-08-01") + pd.Timedelta(days=i if i else 35)).isoformat())
                        for i in range(40)])
    plan = CalibrationReplay(history, config=BayesianConfig(), cutoff="2021-01-01T00:00:00Z",
                             available_at="released_at")
    assert plan.refilter
    leaves = [leaf for _, version in plan.versions
              for _, leaf in _tree_leaves(version.root, 0, version.size)]
    unique = {id(leaf) for leaf in leaves}
    assert len(unique) <= len(plan.events) + len(plan.schedule.entry_events)
    assert len(leaves) > 10 * len(unique)


def test_sparse_and_dense_probability_chunks_preserve_exact_values():
    # Chunk boundaries and extremely different supports must not introduce
    # padded summations (which change finite-difference gradients).
    h = np.tile([.01, .2, 10., 20.], 33)
    a = np.tile([.001, 3., 1., 14.], 33)
    for k in (.1, 100., np.nan):
        actual = outcome_probabilities(h, a, np.full(len(h), k))
        expected = [predict_score_summary(x, y, None if np.isnan(k) else k)
                    for x, y in zip(h, a)]
        np.testing.assert_array_equal(actual, [[r[n] for n in ("p_home_win", "p_draw", "p_away_win")] for r in expected])
