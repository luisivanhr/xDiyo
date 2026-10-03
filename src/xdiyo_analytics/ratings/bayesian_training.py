"""Chronological parameter calibration and the score-target training adapter.

Calibration always replays only the explicitly eligible training population.
The returned parameter bundle records its cutoff; filtered state is separately
retained by the adapter and exported by its native, non-pickle serializer.
"""

from dataclasses import dataclass, replace
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from datetime import date, datetime
from numbers import Number
import hashlib
import json
from pathlib import Path


DEFAULT_FIT_FIELDS = (
    "team_discount", "season_discount", "home_discount", "home_season_discount",
    "kappa", "entry_attack_shift", "entry_defence_shift", "entry_attack_shape",
    "entry_defence_shape",
)
_BOUNDS = {
    "team_discount": (0.8, 1.0), "season_discount": (0.2, 1.0),
    "home_discount": (0.8, 1.0), "home_season_discount": (0.2, 1.0),
    "kappa": (0.1, 100.0), "entry_attack_shift": (0.0, 1.5),
    "entry_defence_shift": (0.0, 1.5), "entry_attack_shape": (0.2, 100.0),
    "entry_defence_shape": (0.2, 100.0),
}
_LOG_FIELDS = {"kappa", "entry_attack_shape", "entry_defence_shape"}
_MATCH_FIELDS = ("source_league", "source_season", "competition_id", "season_id", "event_id")


@dataclass(frozen=True)
class BayesianTrainingResult:
    """A frozen parameter model and JSON-compatible calibration diagnostics."""

    model: object
    report: dict


def _json_value(value):
    """Normalize pandas/numpy identifiers without executable serialization."""
    import numpy as np
    import pandas as pd

    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    return value


def _cutoff(value):
    import pandas as pd

    if value is None or isinstance(value, Number):
        raise ValueError("Supply an explicit datetime training cutoff.")
    time = pd.to_datetime(value, utc=True, errors="raise")
    if not isinstance(time, pd.Timestamp) or pd.isna(time):
        raise ValueError("Training cutoff must be one valid datetime.")
    return time


def _fold_timing(metadata):
    """Read existing temporal-fold scalars without inventing row cutoffs."""
    times = {}
    for name in ("training_boundary", "fit_at"):
        if name in metadata:
            try:
                times[name] = _cutoff(metadata[name])
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"fold_metadata.{name} must be one valid datetime.") from error
    if len(times) == 2 and times["training_boundary"] > times["fit_at"]:
        raise ValueError("fold_metadata.training_boundary cannot follow fit_at.")
    return times


def _restrict_transitions(history, team_seasons, season_starts):
    """Discard metadata outside the selected population before replay sees it."""
    import pandas as pd

    seasons = pd.MultiIndex.from_frame(history[["competition_id", "season_id"]]).unique()
    if team_seasons is not None:
        keys = ["competition_id", "season_id", "team_id"]
        members = pd.MultiIndex.from_frame(history[keys]).unique()
        if isinstance(team_seasons, (pd.DataFrame, Mapping)):
            table = pd.DataFrame(team_seasons).copy()
            if not table.empty:
                table = table.loc[pd.MultiIndex.from_frame(table[keys]).isin(members)].copy()
            team_seasons = table if len(table) else None
        else:
            allowed = set(members.tolist())
            records = [dict(record) for record in team_seasons if tuple(record[name] for name in keys) in allowed]
            team_seasons = records or None
    if season_starts is not None:
        if hasattr(season_starts, "columns"):
            season_starts = season_starts.loc[pd.MultiIndex.from_frame(
                season_starts[["competition_id", "season_id"]]).isin(seasons)].copy()
        else:
            allowed = set(seasons.tolist())
            season_starts = {key: value for key, value in season_starts.items() if key in allowed}
    return team_seasons, season_starts


def _training_history(history, cutoff, available_at):
    """Select whole, valid, finished matches before constructing any context."""
    import numpy as np
    import pandas as pd
    from ..features.history import aligned_times

    required = {"competition_id", "season_id", "event_id", "team_id", "opponent_id",
                "side", "status", "kickoff_at", "goals_for", "goals_against"}
    if required - set(history):
        raise KeyError(f"Bayesian history is missing: {sorted(required - set(history))}")
    keys = [name for name in _MATCH_FIELDS if name in history]
    if history.duplicated(keys + ["side"]).any():
        raise ValueError("Bayesian training history has duplicate match sides.")
    if not history["side"].isin(["home", "away"]).all():
        raise ValueError("Bayesian training sides must be home or away.")
    if not history.groupby(keys, dropna=False).size().eq(2).all():
        raise ValueError("Bayesian training requires both perspectives of every match.")
    kickoff = aligned_times(history, None, default="kickoff_at")
    available = aligned_times(history, available_at, default="kickoff_at")
    if (available < kickoff).any():
        raise ValueError("Result availability cannot precede kickoff.")
    goals = history[["goals_for", "goals_against"]].apply(pd.to_numeric, errors="coerce")
    numeric = goals.to_numpy(dtype=float, na_value=np.nan)
    scored = np.isfinite(numeric).all(axis=1) & (numeric >= 0).all(axis=1)
    scored &= (numeric == np.floor(numeric)).all(axis=1)
    eligible = history["status"].eq("finished").fillna(False).to_numpy(dtype=bool)
    eligible = eligible & scored & kickoff.lt(cutoff).to_numpy() & available.le(cutoff).to_numpy()
    if "is_awarded" in history:
        eligible = eligible & ~history["is_awarded"].fillna(False).astype(bool).to_numpy()
    selection = history[keys].copy()
    selection["eligible"] = eligible
    # Both rows must be eligible; a single valid perspective cannot admit a match.
    good = selection.groupby(keys, sort=False, dropna=False)["eligible"].transform("all").to_numpy()
    chosen = history.loc[good].copy()
    chosen["kickoff_at"] = kickoff.loc[good].to_numpy()
    chosen["_bayesian_available_at"] = available.loc[good].to_numpy()
    if chosen.empty:
        raise ValueError("No finished valid-score matches are available before the training cutoff.")
    chosen = chosen.reset_index(drop=True)
    chosen.attrs = dict(history.attrs)
    return chosen


def _population_report(history):
    import pandas as pd

    home = history.loc[history["side"].eq("home")]
    seasons = []
    for competition, rows in home.groupby("competition_id", sort=False):
        season_count = int(rows["season_id"].nunique())
        seasons.append({"competition_id": _json_value(competition), "matches": len(rows),
                        "observed_seasons": season_count,
                        "observed_season_boundaries": max(0, season_count - 1)})
    keys = [name for name in _MATCH_FIELDS if name in home]
    evidence = home[[*keys, "team_id", "opponent_id", "kickoff_at", "_bayesian_available_at", "goals_for", "goals_against"]]
    # Row ordering does not change the population fingerprint.
    hashes = sorted(pd.util.hash_pandas_object(evidence, index=False).astype(str).tolist())
    return {"matches": len(home), "leagues": len(seasons), "league_counts": seasons,
            "first_kickoff": home["kickoff_at"].min().isoformat(),
            "last_kickoff": home["kickoff_at"].max().isoformat(),
            "last_available_at": home["_bayesian_available_at"].max().isoformat(),
            "population_sha256": hashlib.sha256("\n".join(hashes).encode()).hexdigest(),
            "season_completeness": "Observed season labels only; completeness and consecutive coverage are caller-audited."}


def train_bayesian(history, *, mode="pooled", cutoff, config=None, initial=None,
                   available_at=None, team_seasons=None, season_starts=None,
                   min_seasons=None, fit_fields=None, max_iterations=100,
                   objective="outcome_log_loss", regularization=0.01):
    """Fit bounded, regularized fixed parameters using prequential predictions.

    ``pooled`` shares parameter values but never team or home-advantage states
    between leagues. ``per_league`` calibrates independent parameter bundles.
    The default season-label gate is three per league when pooled, five when
    independent. These are pilot heuristics, not statistical guarantees or a
    verification of complete seasons. Explicit ``min_seasons`` changes the gate.

    Only kickoff < cutoff and result availability <= cutoff are eligible. Goals
    must be nonnegative integers; awarded fixtures are excluded. Availability defaults to the documented kickoff
    proxy. Partial/future season metadata is removed before transition creation.
    ``fit_fields=()`` evaluates and freezes initial parameters without optimizing.
    The optimizer is deterministic L-BFGS-B with a weak quadratic penalty around
    initial parameters, on a log scale for positive shape/dispersion parameters.
    Entry parameters without any actual mirrored-prior use, and kappa for a
    univariate model, remain at their initial values and are reported as inactive
    fields. Mechanism participation does not establish statistical identification.
    No stochastic simulation, MCMC, held-out evaluation, or research run occurs.
    """
    import numpy as np
    from scipy.optimize import minimize
    from .bayesian import BayesianConfig, BayesianModel, build_bayesian_ratings
    from .bayesian_core import BayesianParameters

    if mode not in {"pooled", "per_league"}:
        raise ValueError("mode must be 'pooled' or 'per_league'.")
    if objective not in {"outcome_log_loss", "score_log_loss"}:
        raise ValueError("objective must be outcome_log_loss or score_log_loss.")
    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer.")
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and nonnegative.")
    minimum = (3 if mode == "pooled" else 5) if min_seasons is None else min_seasons
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
        raise ValueError("min_seasons must be a positive integer.")
    config = BayesianConfig() if config is None else config
    if mode == "per_league" and config.transition == "bridge":
        raise ValueError("Independent per_league calibration requires mirrored priors; cross-league state bridges require pooled calibration.")
    initial = BayesianParameters() if initial is None else initial
    if not isinstance(initial, BayesianParameters):
        raise TypeError("initial must be BayesianParameters, not a fitted state or model.")
    requested = DEFAULT_FIT_FIELDS if fit_fields is None else tuple(fit_fields)
    if len(set(requested)) != len(requested) or set(requested) - set(_BOUNDS):
        raise ValueError(f"fit_fields must be distinct choices from {DEFAULT_FIT_FIELDS}.")
    time = _cutoff(cutoff)
    chosen = _training_history(history, time, available_at)
    population = _population_report(chosen)
    short = [item for item in population["league_counts"] if item["observed_seasons"] < minimum]
    if short:
        raise ValueError(f"Bayesian {mode} calibration needs at least {minimum} observed seasons per league; insufficient: {short}")
    team_seasons, season_starts = _restrict_transitions(chosen, team_seasons, season_starts)
    encoded = lambda name, value: float(np.log(value) if name in _LOG_FIELDS else value)

    def fit_one(rows):
        from ..features.transitions import TransitionContext

        transitions, starts = _restrict_transitions(rows, team_seasons, season_starts)
        context = TransitionContext(rows, available_at="_bayesian_available_at",
                                    team_seasons=transitions, season_starts=starts)
        movements = Counter(record["movement"] for record in context.records.values())
        initial_run = build_bayesian_ratings(
            rows, model=BayesianModel(parameters=initial, config=config),
            available_at="_bayesian_available_at", team_seasons=transitions,
            season_starts=starts)
        # Movement labels alone do not activate entry parameters: successful
        # bridges bypass the mirrored prior. Replay's effective counts preserve
        # historical uses when later knowledge changes an entry into a bridge.
        # Mechanism selection depends on history/config, not fitted coordinates,
        # so the initial replay also describes every calibration candidate.
        # Older producers expose the complete entry history in their snapshots.
        effective_entries = initial_run.metadata.get("effective_entry_counts")
        if effective_entries is None:
            effective_entries = initial_run.snapshots["snapshot_kind"].value_counts()
        entry_mechanisms = {name: int(effective_entries.get(name, 0))
                            for name in ("mirrored", "bridge")}
        inactive = {}
        for name in requested:
            if name == "kappa" and not config.bivariate:
                inactive[name] = "Univariate scoring has no shared-effect dispersion parameter."
            elif name.startswith("entry_") and not entry_mechanisms["mirrored"]:
                inactive[name] = "No mirrored entry prior was used in this fitting population; successful bridges do not use entry parameters."
        names = tuple(name for name in requested if name not in inactive)
        center = np.array([encoded(name, getattr(initial, name)) for name in names])
        bounds = [(encoded(name, low), encoded(name, high)) for name in names for low, high in [_BOUNDS[name]]]
        if any(not low <= x <= high for x, (low, high) in zip(center, bounds)):
            raise ValueError("Initial fitted parameters must lie within the calibration bounds.")
        scales = np.array([high - low for low, high in bounds])
        evaluations = 0
        best = [float("inf"), initial, float("inf")]

        def loss(vector, run=None):
            nonlocal evaluations
            evaluations += 1
            values = {name: float(np.exp(value) if name in _LOG_FIELDS else value)
                      for name, value in zip(names, vector)}
            parameters = initial if run is not None else replace(initial, **values)
            if run is None:
                model = BayesianModel(parameters=parameters, config=config)
                run = build_bayesian_ratings(rows, model=model, available_at="_bayesian_available_at",
                                             team_seasons=transitions, season_starts=starts)
            predictions = run.predictions
            if len(predictions) != len(rows) // 2:
                raise ValueError("Calibration replay did not predict every eligible match exactly once.")
            if objective == "score_log_loss":
                log_prob = predictions["score_log_probability"].to_numpy(dtype=float)
            else:
                home = predictions["home_goals"].to_numpy()
                away = predictions["away_goals"].to_numpy()
                probabilities = predictions[["p_home_win", "p_draw", "p_away_win"]].to_numpy(dtype=float)
                outcome = np.where(home > away, 0, np.where(home == away, 1, 2))
                log_prob = np.log(np.maximum(probabilities[np.arange(len(home)), outcome], np.finfo(float).tiny))
            if not np.isfinite(log_prob).all():
                raise ValueError("Nonfinite prequential likelihood during Bayesian calibration.")
            raw = float(-log_prob.mean())
            penalty = regularization * float(np.sum(((vector - center) / scales) ** 2)) if len(names) else 0.0
            value = raw + penalty
            if value < best[0]:
                best[:] = [value, parameters, raw]
            return value

        initial_objective = loss(center, initial_run)
        del initial_run
        if names:
            optimized = minimize(loss, center, method="L-BFGS-B", bounds=bounds,
                                 options={"maxiter": max_iterations, "ftol": 1e-8})
            optimizer = {"success": bool(optimized.success), "status": int(optimized.status),
                         "message": str(optimized.message), "iterations": int(optimized.nit)}
        else:
            optimizer = {"success": True, "status": 0, "message": "Initial parameters frozen; no optimized fields.", "iterations": 0}
        optimizer.update(evaluations=evaluations, initial_objective=initial_objective,
                         penalized_objective=best[0], mean_negative_log_likelihood=best[2])
        return best[1], {**_population_report(rows), "optimizer": optimizer,
                        "fit_fields": list(names), "inactive_fit_fields": inactive,
                        "movement_counts": dict(movements),
                        "entry_mechanism_counts": entry_mechanisms}

    if mode == "pooled":
        parameters, details = fit_one(chosen)
        per_league, fits = (), [details]
    else:
        bundles, fits = [], []
        for competition, rows in chosen.groupby("competition_id", sort=False):
            parameters, details = fit_one(rows.reset_index(drop=True))
            bundles.append((_json_value(competition), parameters))
            fits.append(details)
        per_league, parameters = tuple(bundles), initial
    model = BayesianModel(parameters=parameters, per_league=per_league,
                          training_cutoff=time.isoformat(), config=config)
    active = [name for name in requested if any(name in detail["fit_fields"] for detail in fits)]
    inactive = {name: fits[0]["inactive_fit_fields"][name] for name in requested if name not in active}
    movement_counts = Counter()
    entry_mechanism_counts = Counter()
    for details in fits:
        movement_counts.update(details["movement_counts"])
        entry_mechanism_counts.update(details["entry_mechanism_counts"])
    report = {"schema": 1, "mode": mode, "cutoff": time.isoformat(), "objective": objective,
              "requested_fit_fields": list(requested), "fit_fields": active,
              "inactive_fit_fields": inactive, "movement_counts": dict(movement_counts),
              "entry_mechanism_counts": dict(entry_mechanism_counts),
              "fit_fields_interpretation": "Optimization coordinates after excluding unused dispersion and entry mechanisms; participation and optimizer success do not establish statistical identification.",
              "minimum_observed_seasons": minimum,
              "regularization": float(regularization), "regularization_center": "initial_parameters",
              "optimizer": "scipy.L-BFGS-B", "fits": fits,
              "success": all(item["optimizer"]["success"] for item in fits),
              "availability": "kickoff_proxy" if available_at is None else "explicit",
              "population": population, "score_basis": config.score_basis,
              "cutoff_rule": "kickoff < cutoff and availability <= cutoff; filtered prequential forecasts"}
    model = replace(model, provenance_json=json.dumps(_json_value(report), sort_keys=True, allow_nan=False))
    return BayesianTrainingResult(model, report)


def _match_history(metadata, *, goals=None, home_target=None, away_target=None):
    """Construct only score-independent context plus explicitly supplied y."""
    import pandas as pd

    required = {"competition_id", "season_id", "event_id", "home_id", "away_id", "kickoff_at"}
    if required - set(metadata):
        raise KeyError(f"Bayesian match metadata is missing: {sorted(required - set(metadata))}")
    columns = [name for name in (*_MATCH_FIELDS, "kickoff_at", "round", "season_year", "available_at") if name in metadata]
    parts = []
    for side, other in (("home", "away"), ("away", "home")):
        part = metadata[columns].copy()
        part["team_id"], part["opponent_id"] = metadata[f"{side}_id"], metadata[f"{other}_id"]
        part["side"] = side
        if goals is not None:
            own, opponent = (home_target, away_target) if side == "home" else (away_target, home_target)
            part["goals_for"], part["goals_against"] = goals[own], goals[opponent]
            part["status"] = "finished"
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


class BayesianScoreAdapter:
    """ModelAdapter for two explicit, untransformed match-level goal targets.

    ``fit`` uses exactly FitContext identities and y; optional source history is
    contextual metadata only. All fit scores are replaced from y. Validation and
    held-out targets are never read. ``predict`` queries a frozen state checkpoint
    and applies outcome-free scheduled season entries without mutating that
    checkpoint. Known teams entering a new season require explicit movement and
    predecessor metadata. Constructor team_seasons/season_starts declarations
    remain available for future forecasts and native serialization; they are
    restricted to selected fitting identities before parameter calibration.
    X is not used.
    """

    def __init__(self, history=None, *, home_goal_target, away_goal_target,
                 mode="pooled", config=None, initial=None, min_seasons=None,
                 fit_fields=None, max_iterations=100, objective="outcome_log_loss",
                 regularization=0.01, available_at=None, team_seasons=None, season_starts=None):
        if not isinstance(home_goal_target, str) or not home_goal_target or not isinstance(away_goal_target, str) or not away_goal_target or home_goal_target == away_goal_target:
            raise ValueError("Supply two distinct explicit goal target column names.")
        self.history = history
        self.home_goal_target, self.away_goal_target = home_goal_target, away_goal_target
        self.mode, self.config, self.initial = mode, config, initial
        self.min_seasons, self.fit_fields = min_seasons, None if fit_fields is None else tuple(fit_fields)
        self.max_iterations, self.objective, self.regularization = max_iterations, objective, regularization
        self.available_at = available_at
        self.team_seasons, self.season_starts = deepcopy(team_seasons), deepcopy(season_starts)

    def fit(self, context):
        import numpy as np
        import pandas as pd
        from ..features.history import aligned_times
        from .bayesian import build_bayesian_ratings

        if context.layout != "match":
            raise ValueError("BayesianScoreAdapter requires match layout.")
        targets = (self.home_goal_target, self.away_goal_target)
        if set(context.y.columns) != set(targets) or len(context.y.columns) != 2:
            raise ValueError("Fit y must contain exactly the declared home and away goal targets.")
        if not context.X.index.equals(context.metadata.index) or not context.X.index.equals(context.y.index):
            raise ValueError("Bayesian fit metadata and targets must align with X.")
        if context.sample_weight is not None:
            raise ValueError("BayesianScoreAdapter does not support sample weights.")
        if "is_awarded" in context.metadata and context.metadata["is_awarded"].fillna(False).astype(bool).any():
            raise ValueError("Remove awarded matches from the Bayesian adapter fit population.")
        numeric = context.y[list(targets)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(numeric).all() or (numeric < 0).any() or (numeric != np.floor(numeric)).any():
            raise ValueError("Bayesian score targets must be finite nonnegative integer goals.")
        keys = list(context.match_columns)
        if not keys or "event_id" not in keys or context.metadata.duplicated(keys).any():
            raise ValueError("Bayesian fit requires unique complete match identities.")
        if self.history is None:
            history = _match_history(context.metadata, goals=context.y,
                                     home_target=targets[0], away_target=targets[1])
            release = aligned_times(context.metadata, self.available_at, default="kickoff_at")
            available = pd.concat([release, release], ignore_index=True)
        else:
            if set(keys) - set(self.history):
                raise KeyError("Source history does not contain the fit match identity columns.")
            identities = pd.MultiIndex.from_frame(context.metadata[keys])
            source_keys = pd.MultiIndex.from_frame(self.history[keys])
            mask = source_keys.isin(identities)
            history = self.history.loc[mask].copy()
            # Resolve only selected timestamps. Malformed held-out metadata must
            # not be interpreted as part of this fit.
            release = self.available_at
            if release is not None and not isinstance(release, str) and not np.isscalar(release):
                release = pd.Series(release, index=self.history.index).loc[mask]
            available = aligned_times(history, release, default="kickoff_at")
            history["_bayesian_available_at"] = available.to_numpy()
            if "is_awarded" in history and history["is_awarded"].fillna(False).astype(bool).any():
                raise ValueError("Remove awarded matches from the Bayesian adapter fit population.")
            counts = history.groupby(keys, dropna=False).size()
            if len(counts) != len(identities) or not counts.eq(2).all():
                raise ValueError("Every fit identity needs exactly two source history perspectives.")
            required_metadata = {"home_id", "away_id", "kickoff_at"}
            if required_metadata - set(context.metadata):
                raise KeyError("Bayesian fit metadata needs home_id, away_id and kickoff_at.")
            target_table = context.metadata[list(dict.fromkeys([*keys, *required_metadata]))].copy()
            target_table = target_table.rename(columns={"home_id": "_fit_home_id", "away_id": "_fit_away_id", "kickoff_at": "_fit_kickoff_at"})
            target_table["_fit_home_goals"] = context.y[targets[0]]
            target_table["_fit_away_goals"] = context.y[targets[1]]
            history = history.merge(target_table, on=keys, validate="many_to_one", sort=False)
            home = history["side"].eq("home")
            if (history["team_id"].ne(np.where(home, history["_fit_home_id"], history["_fit_away_id"])).any()
                    or history["opponent_id"].ne(np.where(home, history["_fit_away_id"], history["_fit_home_id"])).any()
                    or pd.to_datetime(history["kickoff_at"], utc=True).ne(pd.to_datetime(history["_fit_kickoff_at"], utc=True)).any()):
                raise ValueError("Source history teams/kickoff disagree with the selected fit metadata.")
            history["goals_for"] = np.where(home, history["_fit_home_goals"], history["_fit_away_goals"])
            history["goals_against"] = np.where(home, history["_fit_away_goals"], history["_fit_home_goals"])
            history["status"] = "finished"
            history = history.drop(columns=["_fit_home_goals", "_fit_away_goals", "_fit_home_id", "_fit_away_id", "_fit_kickoff_at"])
            history.attrs = dict(self.history.attrs)
            available = history["_bayesian_available_at"]
        history["result"] = np.where(history["goals_for"] > history["goals_against"], "W",
                                     np.where(history["goals_for"] == history["goals_against"], "D", "L"))
        history["_bayesian_available_at"] = available.to_numpy()
        kickoff = pd.to_datetime(history["kickoff_at"], utc=True)
        if kickoff.isna().any() or available.isna().any():
            raise ValueError("Every fitted match needs valid kickoff and availability timestamps.")
        timing = _fold_timing(context.fold_metadata)
        cutoff = timing.get("training_boundary", timing.get("fit_at"))
        if cutoff is None:
            # Standalone/refit contexts do not declare a temporal fold boundary.
            cutoff = max(kickoff.max() + pd.Timedelta(1, unit="ns"), pd.to_datetime(available, utc=True).max())
        elif kickoff.ge(cutoff).any() or pd.to_datetime(available, utc=True).gt(cutoff).any():
            raise ValueError("Bayesian fit population violates the declared fold training boundary: every selected kickoff must be earlier and every result available by that boundary; align split and adapter availability rather than dropping fit rows.")
        transitions, starts = _restrict_transitions(history, self.team_seasons, self.season_starts)
        result = train_bayesian(history, mode=self.mode, cutoff=cutoff, config=self.config,
                                initial=self.initial, available_at="_bayesian_available_at",
                                team_seasons=transitions, season_starts=starts, min_seasons=self.min_seasons,
                                fit_fields=self.fit_fields, max_iterations=self.max_iterations,
                                objective=self.objective, regularization=self.regularization)
        self.model_, self.training_summary_ = result.model, result.report
        self.training_summary_["fit_population"] = "exact FitContext identities; outcomes supplied only by context.y"
        self.training_summary_["availability"] = "kickoff_proxy" if self.available_at is None else "explicit"
        activation = timing.get("fit_at", cutoff)
        self.training_summary_["fit_timing"] = {
            "declared": {name: value.isoformat() for name, value in timing.items()},
            "data_cutoff": cutoff.isoformat(), "model_available_at": activation.isoformat(),
        }
        self.model_ = replace(self.model_, training_cutoff=activation.isoformat(), provenance_json=json.dumps(
            _json_value(self.training_summary_), sort_keys=True, allow_nan=False))
        self.training_history_ = pd.DataFrame()
        self.run_ = build_bayesian_ratings(history, model=self.model_, available_at="_bayesian_available_at",
                                           team_seasons=transitions, season_starts=starts)

    def predict(self, context):
        import pandas as pd

        if not hasattr(self, "run_"):
            raise RuntimeError("Fit BayesianScoreAdapter before predicting.")
        if context.layout != "match" or not context.X.index.equals(context.metadata.index):
            raise ValueError("Bayesian prediction requires aligned match-layout metadata.")
        timing = _fold_timing(context.fold_metadata)
        if "fit_at" in timing and self.model_.training_cutoff is not None and pd.Timestamp(self.model_.training_cutoff) > timing["fit_at"]:
            raise ValueError("Bayesian parameters were trained after the declared prediction fit_at.")
        boundary = timing.get("training_boundary", timing.get("fit_at"))
        if boundary is not None:
            population = self.training_summary_["population"]
            if pd.Timestamp(population["last_kickoff"]) >= boundary or pd.Timestamp(population["last_available_at"]) > boundary:
                raise ValueError("Bayesian fitted observations violate the declared prediction fold training boundary.")
        paired = _match_history(context.metadata)
        paired["status"] = "scheduled"
        paired["goals_for"], paired["goals_against"] = float("nan"), float("nan")
        # There are no observed scores in this frame. Its basis comes from the
        # frozen model solely to satisfy replay's score-basis contract.
        paired.attrs["score_basis"] = self.model_.config.score_basis
        times = pd.to_datetime(paired["kickoff_at"], utc=True)
        if "fit_at" in timing and times.lt(timing["fit_at"]).any():
            raise ValueError("Prediction fixture kickoff cannot precede the declared fit_at.")
        if self.model_.training_cutoff is not None and times.lt(pd.Timestamp(self.model_.training_cutoff)).any():
            raise ValueError("Bayesian parameters were trained after a requested prediction cutoff.")
        transitions, starts = _restrict_transitions(paired, self.team_seasons, self.season_starts)
        # Query-only replay owns new tables/states; the fitted checkpoint never
        # advances without explicitly observed results. Metadata can establish
        # a season entry, but no target or source-history outcome is accessed.
        forecast = self.run_.update(paired, team_seasons=transitions, season_starts=starts)
        query = paired.loc[paired["side"].eq("home")].copy()
        query.index = context.X.index
        fields = ("expected_home_goals", "expected_away_goals", "p_home_win", "p_draw", "p_away_win")
        values = forecast.fixture_features(query, fields=fields)
        predictions = pd.DataFrame({self.home_goal_target: values["expected_home_goals"],
                                    self.away_goal_target: values["expected_away_goals"]}, index=context.X.index)
        probabilities = values[["p_home_win", "p_draw", "p_away_win"]].copy()
        probabilities.columns = ["home", "draw", "away"]
        return {"predict": predictions, "predict_proba": probabilities}


@dataclass(frozen=True)
class BayesianScoreSerializer:
    """Native JSON/parquet checkpoint serializer for BayesianScoreAdapter."""

    format_id: str = "xdiyo.bayesian_score.v1"

    def save(self, adapter, directory):
        from dataclasses import asdict
        import pandas as pd

        if not isinstance(adapter, BayesianScoreAdapter) or not hasattr(adapter, "run_"):
            raise TypeError("BayesianScoreSerializer requires a fitted BayesianScoreAdapter.")
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        settings = {name: getattr(adapter, name) for name in (
            "home_goal_target", "away_goal_target", "mode", "min_seasons", "fit_fields",
            "max_iterations", "objective", "regularization")}
        settings["initial"] = None if adapter.initial is None else asdict(adapter.initial)
        # Persist only ex-ante movement declarations, never optional source
        # histories or arbitrary extra table columns that could contain scores.
        movement_fields = {"competition_id", "season_id", "team_id", "movement", "got_promoted", "got_demoted",
                           "previous_competition_id", "previous_season_id"}
        movement = None
        if adapter.team_seasons is not None:
            table = isinstance(adapter.team_seasons, (pd.DataFrame, Mapping))
            records = pd.DataFrame(adapter.team_seasons).to_dict("records") if table else list(adapter.team_seasons)
            movement = {"kind": "table" if table else "records", "records": [
                {key: value for key, value in record.items() if key in movement_fields} for record in records]}
        starts = None
        if adapter.season_starts is not None:
            if hasattr(adapter.season_starts, "columns"):
                records = adapter.season_starts[["competition_id", "season_id", "entry_at"]].to_dict("records")
            else:
                records = [dict(competition_id=key[0], season_id=key[1], entry_at=value)
                           for key, value in adapter.season_starts.items()]
            starts = records
        data = {"schema": 1, "settings": _json_value(settings), "report": _json_value(adapter.training_summary_),
                "forecast_team_seasons": _json_value(movement), "forecast_season_starts": _json_value(starts)}
        with (folder / "adapter.json").open("x", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
        adapter.run_.save(folder / "checkpoint")

    def load(self, directory):
        import pandas as pd
        from .bayesian import BayesianRatingRun
        from .bayesian_core import BayesianParameters

        folder = Path(directory)
        data = json.loads((folder / "adapter.json").read_text(encoding="utf-8"))
        if data.get("schema") != 1:
            raise ValueError("Unsupported Bayesian adapter artifact schema.")
        settings = data["settings"]
        if settings.get("initial") is not None:
            settings["initial"] = BayesianParameters(**settings["initial"])
        run = BayesianRatingRun.load(folder / "checkpoint")
        movement = data.get("forecast_team_seasons")
        team_seasons = None if movement is None else (
            pd.DataFrame(movement["records"], dtype=object) if movement["kind"] == "table" else movement["records"])
        starts = data.get("forecast_season_starts")
        season_starts = None if starts is None else {
            (record["competition_id"], record["season_id"]): record["entry_at"] for record in starts}
        adapter = BayesianScoreAdapter(config=run.model.config, team_seasons=team_seasons,
                                       season_starts=season_starts, **settings)
        adapter.model_, adapter.run_ = run.model, run
        adapter.training_summary_, adapter.training_history_ = data["report"], pd.DataFrame()
        return adapter
