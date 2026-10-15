from __future__ import annotations

from dataclasses import dataclass
import math

API_ORIGIN = "https://www.sofascore.com/api/v1"
PARSER_VERSION = "2"


@dataclass(frozen=True)
class Profile:
    include_pregame: bool = True
    include_statistics: bool = True
    include_shotmap: bool = True
    include_heatmaps: bool = True
    include_lineups: bool = True
    include_injuries: bool = True
    include_comments: bool = False
    include_goal_actions: bool = False
    duration_mode: str = "metadata"
    injury_scope: str = "availability"
    include_upcoming: bool = False
    lineup_window_hours: float = 24
    upcoming_refresh_seconds: float = 1800
    settling_window_hours: float = 48
    settling_delay_seconds: float = 21600
    negative_cache_seconds: float = 604800
    locale: str = "en"

    def __post_init__(self):
        for name,value in vars(self).items():
            if name.startswith("include_") and not isinstance(value,bool):
                raise ValueError(f"{name} must be boolean")
        if self.duration_mode not in {"off", "metadata", "incidents"}:
            raise ValueError("duration_mode must be off, metadata or incidents")
        if self.injury_scope not in {"availability", "availability_and_match_incidents"}:
            raise ValueError("Unknown injury_scope")
        for name in ("lineup_window_hours", "upcoming_refresh_seconds", "settling_window_hours",
                     "settling_delay_seconds", "negative_cache_seconds"):
            if not math.isfinite(getattr(self,name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not self.locale or any(c in self.locale for c in "\r\n|"):
            raise ValueError("Invalid locale")


@dataclass(frozen=True)
class Scope:
    league_id: int
    season_id: int
    rounds: tuple[int, ...]
    league_name: str = ""
    season_start: int | None = None
    season_end: int | None = None

    def __post_init__(self):
        if self.league_id <= 0 or self.season_id <= 0:
            raise ValueError("Provider league and season IDs must be positive")
        if not self.rounds or any(isinstance(r, bool) or r < 1 for r in self.rounds):
            raise ValueError("Supply actual round numbers, starting at 1")
        object.__setattr__(self, "rounds", tuple(sorted(set(self.rounds))))


@dataclass(frozen=True)
class RatePolicy:
    # Operational defaults, not a claim about Sofascore's allowed quota.
    min_interval_seconds: float | None = 10
    per_minute: int | None = 6
    per_hour: int | None = 120
    max_attempts_per_run: int | None = 40
    max_attempts_per_job: int = 3
    backoff_seconds: float = 15
    max_backoff_seconds: float = 300
    default_cooldown_seconds: float = 3600
    page_load_timeout_seconds: float = 30
    random_delay_max_seconds: float | None = None

    def __post_init__(self):
        optional = {"min_interval_seconds", "per_minute", "per_hour", "max_attempts_per_run"}
        for name, value in vars(self).items():
            if name == "random_delay_max_seconds":
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                          or not math.isfinite(value) or value < 0):
                    raise ValueError("random_delay_max_seconds must be None or a finite nonnegative number")
                continue
            if name in optional and value is None:
                continue
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("per_minute", "per_hour", "max_attempts_per_run", "max_attempts_per_job"):
            if getattr(self, name) is not None and not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an integer")
