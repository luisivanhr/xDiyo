"""Configuration and explicit calendar-day forecast policy (no torch import)."""
from dataclasses import dataclass
from zoneinfo import ZoneInfo
import math
import pandas as pd
from .spatial import GraphHeatmaps
from .extensions import BoundaryMemory, StatisticsForecast, GoalAuxiliary

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
    deterministic: bool = False
    boundary_memory: BoundaryMemory | None = None
    statistics_forecast: StatisticsForecast | None = None
    temporal_mode: str = "snapshot"
    time_scales_days: tuple = (7., 30., 180.)
    time_encoding: str = "rational_decay_v1"
    age_scale_days: float = 180.
    gap_scale_days: float = 7.
    goal_auxiliary: GoalAuxiliary | None = None
    fixture_gap_encoding: str = "bounded_observed_gap_v1"
    observed_gap_scale_days: float = 7.
    observed_gap_recent_days: float = 30.
    sender_state_mode: str = "updated"

    def __post_init__(self):
        from .observed_gap import LEGACY_GAP_ENCODING, BOUNDED_GAP_ENCODING
        if self.fixture_gap_encoding not in (LEGACY_GAP_ENCODING, BOUNDED_GAP_ENCODING):
            raise ValueError('Unsupported fixture_gap_encoding.')
        for name, expected in (('observed_gap_scale_days', 7.), ('observed_gap_recent_days', 30.)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value != expected:
                raise ValueError(f'{name} must be {expected} for the versioned observed-gap encoding.')
            object.__setattr__(self, name, float(value))
        if self.sender_state_mode not in ('updated', 'layer_zero'):
            raise ValueError('sender_state_mode must be updated or layer_zero.')
        if self.sender_state_mode == 'layer_zero' and (self.temporal_mode == 'historical' or self.boundary_memory is not None):
            raise ValueError('layer_zero sender context does not support historical recursion or boundary memory.')
        if self.fixture_gap_encoding == BOUNDED_GAP_ENCODING:
            reserved = {f'{side}_{name}' for side in ('home', 'away') for name in
                        ('rest_days', 'observed_gap_days', 'latest_eligible_observed_at', 'has_eligible_observation',
                         'observed_gap_r7', 'no_eligible_history', 'no_eligible_observation_30d')}
            if reserved.intersection(self.fixture_features):
                raise ValueError('Observed-gap aliases cannot also be selected as ordinary fixture features.')
        if self.goal_auxiliary is not None:
            if not isinstance(self.goal_auxiliary, GoalAuxiliary):
                raise TypeError('goal_auxiliary must be GoalAuxiliary or None.')
            if any(x is not None for x in (self.boundary_memory, self.statistics_forecast, self.heatmaps)):
                raise ValueError('Goal auxiliary supervision requires memory, statistics_forecast and heatmaps disabled.')
        if self.temporal_mode not in ('snapshot', 'time_features', 'historical'):
            raise ValueError('temporal_mode must be snapshot, time_features or historical.')
        if self.time_encoding not in ('log1p_v1', 'rational_decay_v1'):
            raise ValueError('time_encoding must be log1p_v1 or rational_decay_v1.')
        for name in ('age_scale_days', 'gap_scale_days'):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f'{name} must be finite and positive.')
            object.__setattr__(self, name, float(value))
        scales = tuple(self.time_scales_days)
        if not scales or any(isinstance(x, bool) or not isinstance(x, (int, float))
                             or not math.isfinite(x) or x <= 0 for x in scales):
            raise ValueError('time_scales_days must contain finite positive numbers.')
        if len(set(scales)) != len(scales):
            raise ValueError('time_scales_days must be distinct.')
        object.__setattr__(self, 'time_scales_days', tuple(float(x) for x in scales))
        if self.temporal_mode != 'snapshot' and any(x is not None for x in
                (self.boundary_memory, self.statistics_forecast, self.heatmaps)):
            raise ValueError('Temporal attention experiments require boundary_memory, statistics_forecast and heatmaps disabled.')
        for name, cls in (("boundary_memory", BoundaryMemory), ("statistics_forecast", StatisticsForecast)):
            if getattr(self, name) is not None and not isinstance(getattr(self, name), cls):
                raise TypeError(f"{name} must be {cls.__name__} or None.")
        if type(self.deterministic) is not bool:
            raise ValueError('deterministic must be a boolean.')
        if self.heatmaps is not None and not isinstance(self.heatmaps, GraphHeatmaps):
            raise TypeError('heatmaps must be GraphHeatmaps or None.')
        ZoneInfo(self.timezone)
        if self.boundary_memory is not None and self.boundary_memory.policy == 'fixed_days':
            anchor = pd.Timestamp(self.boundary_memory.anchor).tz_convert(self.timezone)
            if anchor != anchor.normalize():
                raise ValueError('Fixed-segment anchor must be local midnight in the model timezone.')
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
