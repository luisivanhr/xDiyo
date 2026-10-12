"""Configuration and explicit calendar-day forecast policy (no torch import)."""
from dataclasses import dataclass
from zoneinfo import ZoneInfo
import math
import pandas as pd
from .spatial import GraphHeatmaps

CLASS_ORDER = (1, 0, -1)  # Native Outcome(perspective="home"): win, draw, loss.
ENTRY_ORDER = ("retained", "promoted", "relegated", "unknown")
MATCH_KEYS = ("competition_id", "season_id", "event_id")


@dataclass(frozen=True)
class TemporalGraphConfig:
    timezone: str
    forecast_policy: str = "by_round"
    round_keys: tuple = ("competition_id", "season_id", "round")
    embedding_dim: int = 32
    layers: int = 2
    heads: int = 2
    half_life_days: float = 180.
    seasonal_retention: float = .5
    support_kappa: float = 5.
    hidden_width: int = 64
    statistics: tuple = ()
    fixture_features: tuple = ()
    learning_rate: float = .001
    weight_decay: float = .001
    prior_penalty: float = .001
    heatmaps: GraphHeatmaps | None = None
    include_goal_features: bool = True

    def __post_init__(self):
        if self.heatmaps is not None and not isinstance(self.heatmaps, GraphHeatmaps):
            raise TypeError('heatmaps must be GraphHeatmaps or None.')
        ZoneInfo(self.timezone)
        if self.forecast_policy not in ("by_round", "one_day_ahead"):
            raise ValueError("forecast_policy must be by_round (By Round) or one_day_ahead (One Day Ahead).")
        for name in ("round_keys", "statistics", "fixture_features"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if type(self.include_goal_features) is not bool:
            raise ValueError('include_goal_features must be a boolean.')
        if not self.include_goal_features and not self.statistics:
            raise ValueError('include_goal_features=False requires at least one explicit statistic.')
        if not {"competition_id", "season_id", "round"} <= set(self.round_keys):
            raise ValueError("round_keys must contain competition_id, season_id and round; add stage when needed.")
        for name in ("embedding_dim", "layers", "heads", "hidden_width"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("half_life_days", "support_kappa", "learning_rate"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if not 0 < self.seasonal_retention <= 1:
            raise ValueError("seasonal_retention must lie in (0, 1].")
        for name in ("weight_decay", "prior_penalty"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
        for fields in (self.statistics, self.fixture_features):
            if len(set(fields)) != len(fields):
                raise ValueError("Feature selections must be distinct.")
            if any("xg" in f.lower() or "expectedgoal" in f.lower().replace("_", "") for f in fields):
                raise ValueError("xG is excluded from the initial graph model.")

    def history_cutoff(self, issuance):
        """Exclusive local midnight: all observations through the previous day."""
        t = pd.Timestamp(issuance)
        if t.tzinfo is None or pd.isna(t):
            raise ValueError("Issuance must be a nonmissing timezone-aware timestamp.")
        return t.tz_convert(self.timezone).normalize().tz_convert("UTC")

    def labels_known_at(self, kickoffs):
        times = pd.to_datetime(kickoffs, utc=True).dt.tz_convert(self.timezone)
        # Calendar DateOffset preserves midnight across daylight-saving changes.
        return (times.dt.normalize() + pd.DateOffset(days=1)).dt.tz_convert("UTC").astype('datetime64[ns, UTC]')
