"""Bayesian rating configuration, exported parameters and reusable state runs.

The model-specific producer implements the existing RatingRun contract. It does
not alter pairwise Glicko replay or experiment orchestration.
"""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

from .bayesian_core import BayesianParameters, GammaState, TeamState, initial_team, team_summary
from .snapshots import RatingRun


DEFAULT_TEAM_FIELDS = ("attack_mean", "defence_vulnerability_mean")
TEAM_FIELDS = (
    *DEFAULT_TEAM_FIELDS, "attack_sd", "defence_vulnerability_sd",
    "log_attack", "log_defence_strength", "strength_index",
    "attack_shape", "attack_rate", "defence_shape", "defence_rate",
)
DEFAULT_FIXTURE_FIELDS = ("expected_home_goals", "expected_away_goals")
FIXTURE_FIELDS = (
    *DEFAULT_FIXTURE_FIELDS, "expected_total_goals", "expected_goal_difference",
    "strength_difference", "home_advantage_mean", "home_advantage_sd",
    "p_home_win", "p_draw", "p_away_win", "p_both_score", "p_over_2_5",
)


def _select_fields(fields, defaults, allowed):
    values = defaults if fields is None else tuple(fields)
    if isinstance(fields, str) or not values or len(set(values)) != len(values):
        raise ValueError("Select a nonempty sequence of distinct Bayesian output fields.")
    if set(values) - set(allowed):
        raise ValueError(f"Unknown Bayesian output fields: {sorted(set(values) - set(allowed))}")
    return tuple(values)


def _identifier(value):
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TypeError("Bayesian competition/team/season IDs must be integers or strings.")
    return value


@dataclass(frozen=True)
class BayesianConfig:
    """Numerical and temporal conventions, saved with every parameter bundle.

    team_match discounts each team's states once per observed fixture; days
    uses elapsed kickoff days / time_unit_days. The paper uses team rounds.
    A team appearance agrees with a round in regular schedules, and deliberately
    avoids using future round completion to time postponed matches.

    provider_current explicitly uses the native goals_for/goals_against basis.
    regulation requires history.attrs['score_basis'] == 'regulation', supplied
    by a caller who has prepared and checked regulation-time goals.

    mirrored applies direction-aware entry priors; bridge retains a known
    source posterior using an explicit (source, destination, gap) log-scale map.
    Positive gap means the source competition is stronger. Reverse lookup uses
    the negative gap. Missing source/map falls back to the mirrored prior.
    """

    bivariate: bool = True
    method: str = "vb"
    tolerance: float = 1e-9
    max_iterations: int = 500
    clock: str = "team_match"
    time_unit_days: float = 7.0
    score_basis: str = "provider_current"
    transition: str = "mirrored"
    bridge_gaps: tuple = ()
    bridge_discount: float = 0.8
    tail_tolerance: float = 1e-10
    max_total_goals: int = 10000

    def __post_init__(self):
        if not isinstance(self.bivariate, bool):
            raise TypeError("bivariate must be a boolean.")
        if self.method not in ("vb", "one_step") or self.clock not in ("team_match", "days"):
            raise ValueError("Choose vb/one_step and team_match/days.")
        if self.score_basis not in ("provider_current", "regulation"):
            raise ValueError("Choose provider_current or explicitly prepared regulation scores.")
        if self.transition not in ("mirrored", "bridge"):
            raise ValueError("Choose mirrored priors or an explicit bridge.")
        for name in ("tolerance", "time_unit_days", "tail_tolerance", "bridge_discount"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.bridge_discount > 1 or self.tail_tolerance >= 1 or self.tolerance >= 1:
            raise ValueError("Discount must be <= 1; numerical tolerances must be < 1.")
        for name in ("max_iterations", "max_total_goals"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        gaps, seen = [], set()
        for source, destination, gap in self.bridge_gaps:
            source, destination = _identifier(source), _identifier(destination)
            if source == destination or frozenset((source, destination)) in seen:
                raise ValueError("Bridge gaps must identify distinct, unique league pairs.")
            if not math.isfinite(gap) or abs(gap) > 10:
                raise ValueError("Bridge gaps must be finite log multipliers within [-10, 10].")
            gaps.append((source, destination, float(gap)))
            seen.add(frozenset((source, destination)))
        object.__setattr__(self, "bridge_gaps", tuple(gaps))


@dataclass(frozen=True)
class BayesianModel:
    """Exportable frozen coefficients, distinct from accumulated team states.

    An empty per_league tuple denotes shared parameters. Otherwise each item
    is (competition_id, BayesianParameters), and unknown competitions fail.
    training_cutoff records the earliest permitted prediction time, respecting
    the calibration information boundary and any declared fold fit_at;
    historical feature queries before this timestamp are forbidden.
    """

    parameters: BayesianParameters = BayesianParameters()
    per_league: tuple = ()
    training_cutoff: str | None = None
    config: BayesianConfig = BayesianConfig()
    provenance_json: str = "{}"

    def __post_init__(self):
        import pandas as pd
        if not isinstance(self.parameters, BayesianParameters) or not isinstance(self.config, BayesianConfig):
            raise TypeError("Supply BayesianParameters and BayesianConfig objects.")
        values, seen = [], set()
        for key, parameters in self.per_league:
            key = _identifier(key)
            if key in seen or not isinstance(parameters, BayesianParameters):
                raise ValueError("per_league must contain unique IDs and BayesianParameters.")
            seen.add(key)
            values.append((key, parameters))
        object.__setattr__(self, "per_league", tuple(values))
        if self.training_cutoff is not None:
            time = pd.to_datetime(self.training_cutoff, utc=True)
            if pd.isna(time):
                raise ValueError("training_cutoff must be a valid timestamp.")
            object.__setattr__(self, "training_cutoff", time.isoformat())
        if not isinstance(json.loads(self.provenance_json), dict):
            raise ValueError("provenance_json must encode an object.")

    def parameters_for(self, competition_id):
        if not self.per_league:
            return self.parameters
        for key, parameters in self.per_league:
            if key == competition_id:
                return parameters
        raise KeyError(f"No calibrated Bayesian parameters for competition {competition_id!r}.")

    def to_dict(self):
        return {"schema": 1, "parameters": asdict(self.parameters),
                "per_league": [[key, asdict(value)] for key, value in self.per_league],
                "training_cutoff": self.training_cutoff, "config": asdict(self.config),
                "provenance": json.loads(self.provenance_json)}

    @classmethod
    def from_dict(cls, value):
        if value.get("schema") != 1:
            raise ValueError("Unsupported Bayesian parameter schema.")
        return cls(BayesianParameters(**value["parameters"]),
                   tuple((key, BayesianParameters(**parameters)) for key, parameters in value["per_league"]),
                   value["training_cutoff"], BayesianConfig(**value["config"]),
                   json.dumps(value.get("provenance", {}), sort_keys=True, allow_nan=False))

    def save(self, path):
        """Write a new JSON parameter file; never overwrite existing artifacts."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2, allow_nan=False)
            stream.write("\n")
        return target

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class BayesianRatingRun(RatingRun):
    """RatingRun with league-state history and a complete Bayesian checkpoint.

    Team features obey the inherited snapshot contract. Fixture summaries look
    up both teams and the shared home advantage at the same prediction cutoff.
    The stored shape/rate state is independent of requested output columns.
    """

    model: BayesianModel = BayesianModel()
    league_snapshots: object = None
    predictions: object = None
    checkpoint: dict = field(default_factory=dict)

    def _times(self, history, cutoffs):
        import pandas as pd
        from ..features.history import aligned_times
        times = aligned_times(history, cutoffs, default="kickoff_at")
        kickoff = aligned_times(history, None, default="kickoff_at")
        if (times > kickoff).any():
            raise ValueError("Prediction cutoffs must not follow target kickoff.")
        if self.model.training_cutoff is not None and (times < pd.Timestamp(self.model.training_cutoff)).any():
            raise ValueError("Bayesian parameters were trained after a requested prediction cutoff.")
        for competition in history.competition_id.drop_duplicates():
            self.model.parameters_for(competition)
        return times

    def features(self, history, *, cutoffs=None, side="both", fields=None):
        # Preserve RatingRun's generic fields=None contract. The BayesianRating
        # feature expression explicitly selects the two default means.
        fields = _select_fields(fields, TEAM_FIELDS, TEAM_FIELDS)
        times = self._times(history, cutoffs)
        # Each per-league run supplies its own neutral fallback, even if a
        # query team was not present in the saved history at all.
        import pandas as pd
        result = None
        for competition, positions in history.groupby("competition_id", sort=False).indices.items():
            frame = history.iloc[positions]
            initial = team_summary(initial_team(self.model.parameters_for(competition)))
            base = RatingRun(self.snapshots, initial, self.scope, self.streams, self.metadata)
            values = base.features(frame, cutoffs=times.iloc[positions], side=side, fields=fields)
            if result is None:
                result = pd.DataFrame(float("nan"), index=history.index, columns=values.columns)
            result.iloc[positions] = values.to_numpy()
        if result is None:
            roles = {"for": ("team",), "against": ("opponent",), "both": ("team", "opponent")}
            if side not in roles:
                raise ValueError("side must be for, against or both.")
            result = pd.DataFrame(index=history.index, columns=[f"score::{role}::{name}" for role in roles[side] for name in fields], dtype=float)
        return result

    def fixture_features(self, history, *, cutoffs=None, fields=None):
        from .bayesian_replay import fixture_features
        fields = _select_fields(fields, DEFAULT_FIXTURE_FIELDS, FIXTURE_FIELDS)
        times = self._times(history, cutoffs)
        return fixture_features(self, history, times, fields)

    def save(self, directory):
        """Save complete state and histories in a new checksummed directory.

        The native format deliberately cannot be loaded as a bare RatingRun:
        doing so would lose its calibrated-parameter time boundary.
        """
        import hashlib
        import shutil
        from uuid import uuid4
        target = Path(directory)
        if target.exists():
            raise FileExistsError(f"Bayesian artifact already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        stage = target.parent / f".pending-bayesian-{uuid4()}"
        stage.mkdir()
        try:
            for name, frame in (("snapshots", self.snapshots), ("league_snapshots", self.league_snapshots), ("predictions", self.predictions)):
                frame.to_parquet(stage / f"{name}.parquet", index=False)
            description = {"schema": 1, "model": self.model.to_dict(), "initial_state": self.initial_state,
                           "metadata": self.metadata, "checkpoint": self.checkpoint}
            (stage / "state.json").write_text(json.dumps(description, indent=2, allow_nan=False), encoding="utf-8")
            hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in stage.iterdir()}
            (stage / "bayesian.json").write_text(json.dumps({"schema": 1, "files": hashes}, indent=2), encoding="utf-8")
            stage.rename(target)
        except Exception:
            # Only this newly-created private staging directory is removed.
            if stage.exists() and not stage.is_symlink() and stage.resolve().parent == target.parent.resolve():
                shutil.rmtree(stage)
            raise
        return target

    @classmethod
    def load(cls, path):
        import hashlib
        import pandas as pd
        root = Path(path).resolve()
        manifest = json.loads((root / "bayesian.json").read_text(encoding="utf-8"))
        required = {"snapshots.parquet", "league_snapshots.parquet", "predictions.parquet", "state.json"}
        if manifest.get("schema") != 1 or set(manifest.get("files", {})) != required:
            raise ValueError("Invalid Bayesian checkpoint manifest.")
        for name, digest in manifest["files"].items():
            path = root / name
            if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Bayesian checkpoint file is missing or changed: {name}")
        state = json.loads((root / "state.json").read_text(encoding="utf-8"))
        if state.get("schema") != 1:
            raise ValueError("Unsupported Bayesian checkpoint schema.")
        return cls(pd.read_parquet(root / "snapshots.parquet"), state["initial_state"],
                   ("competition_id",), ("score",), state["metadata"],
                   BayesianModel.from_dict(state["model"]), pd.read_parquet(root / "league_snapshots.parquet"),
                   pd.read_parquet(root / "predictions.parquet"), state["checkpoint"])

    def update(self, history, *, available_at=None, team_seasons=None, season_starts=None, cutoffs=None):
        """Append new releases, replaying late kickoffs from the saved journal.

        Releases must be strictly later than the durable frontier. Corrections
        to existing events and equal-release appends require full-history replay.
        """
        return build_bayesian_ratings(history, model=self.model, available_at=available_at,
                                      team_seasons=team_seasons, season_starts=season_starts,
                                      cutoffs=cutoffs, checkpoint=self)


def build_bayesian_ratings(history, *, model=None, available_at=None, team_seasons=None,
                           season_starts=None, cutoffs=None, checkpoint=None):
    """Produce cutoff-safe RatingRun snapshots from each paired score once."""
    from .bayesian_replay import replay
    return replay(history, model=BayesianModel() if model is None else model,
                  available_at=available_at, team_seasons=team_seasons,
                  season_starts=season_starts, cutoffs=cutoffs, checkpoint=checkpoint)
