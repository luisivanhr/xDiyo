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
class ForAgainst(Expr):
    """Select a Stat's team/opponent columns, without another join."""

    source: Stat
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


@dataclass(frozen=True)
class RollingMean(Expr):
    source: Expr
    window: int = 5
    min_periods: int = 1


@dataclass(frozen=True)
class RollingStd(Expr):
    source: Expr
    window: int = 5
    min_periods: int = 1
    ddof: int = 1


@dataclass(frozen=True)
class RollingZScore(Expr):
    """Latest eligible value versus its trailing window, including that value."""

    source: Expr
    window: int = 5
    min_periods: int = 1
    ddof: int = 1
    reference: Expr | None = None


@dataclass(frozen=True)
class EMA(Expr):
    """Recursive EMA: alpha=2/(span+1), initialized by the first observed value.

    Missing observations are skipped without decay (ignore_na=True). Uses all
    eligible history in its group, with no seasonal blend or prior.
    """

    source: Expr
    span: float = 5
    min_periods: int = 1
