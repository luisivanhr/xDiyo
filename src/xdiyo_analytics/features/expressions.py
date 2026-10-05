"""Small, immutable feature expressions (the feature AST)."""

from dataclasses import dataclass


class Expr:
    """Base for expressions evaluated against a team history."""


@dataclass(frozen=True)
class Stat(Expr):
    """An observed statistic. None selects each available period separately."""

    period: str | None
    group: str
    key: str
    field: str = "value"


@dataclass(frozen=True)
class MatchScore(Expr):
    """Observed native-current goals; requires a historical operator.

    for/against select the focal team's scored/conceded goals; both emits two
    columns. Current is the provider's score, not necessarily regulation time.
    No half, extra-time or penalty reconstruction or fallback is performed.
    """

    score_field: str = "current"
    side: str = "for"

    def __post_init__(self):
        if self.score_field != "current":
            raise ValueError("MatchScore score_field must be 'current'; other score bases are not supported.")
        if self.side not in ("for", "against", "both"):
            raise ValueError("MatchScore side must be 'for', 'against' or 'both'.")


@dataclass(frozen=True)
class ForAgainst(Expr):
    """Select a Stat, MatchScore or Heatmap perspective, without another join."""

    source: Expr
    side: str = "for"


@dataclass(frozen=True)
class H2H(Expr):
    """Restrict historical operators to the ordered team/opponent pair."""

    source: Expr


@dataclass(frozen=True)
class IsHome(Expr):
    """Known home/away context for the current prediction row."""


@dataclass(frozen=True)
class NormalizedStanding(Expr):
    """First place=1, last=0; missing position defaults to zero, without a prior."""

    side: str = "for"
    missing_value: float = 0.0


@dataclass(frozen=True)
class Lag(Expr):
    source: Expr
    periods: int = 1
    venue: str = "all"


@dataclass(frozen=True)
class RollingMean(Expr):
    source: Expr
    window: int = 5
    min_periods: int = 1
    venue: str = "all"


@dataclass(frozen=True)
class RollingStd(Expr):
    source: Expr
    window: int = 5
    min_periods: int = 1
    ddof: int = 1
    venue: str = "all"


@dataclass(frozen=True)
class RollingSkewness(Expr):
    """Population skewness m3 / m2**1.5 over earlier eligible observations.

    At least three finite observations and positive variance are required.
    No small-sample correction is applied, including during seeded warm-up.
    """

    source: Expr
    window: int = 5
    min_periods: int = 3
    venue: str = "all"


@dataclass(frozen=True)
class RollingZScore(Expr):
    """Latest eligible value versus its trailing window, including that value."""

    source: Expr
    window: int = 5
    min_periods: int = 1
    ddof: int = 1
    reference: Expr | None = None
    venue: str = "all"


@dataclass(frozen=True)
class EMA(Expr):
    """Recursive EMA: alpha=2/(span+1), initialized by the first observed value.

    Missing observations are skipped without decay (ignore_na=True). Uses all
    eligible history in its group, with no seasonal blend or prior.
    """

    source: Expr
    span: float = 5
    min_periods: int = 1
    venue: str = "all"
