"""Feature expressions for generated or explicitly supplied rating histories."""

from dataclasses import dataclass

from .expressions import Expr, Stat
from ..ratings.glicko import Glicko2


@dataclass(frozen=True)
class Rating(Expr):
    """Read a named RatingRun passed in evaluate_features(..., ratings={...}).

    fields=None exports all numeric state fields, including custom embeddings.
    The state producer must respect training and information-availability limits.
    """

    name: str
    side: str = "both"
    fields: tuple | None = None


@dataclass(frozen=True)
class MatchResultGlicko(Expr):
    """Generate pre-prediction Glicko-2 features from match W/D/L."""

    side: str = "both"
    fields: tuple = ("rating", "rd", "sigma")
    engine: Glicko2 = Glicko2()
    scope: tuple = ("competition_id",)


@dataclass(frozen=True)
class StatGlicko(Expr):
    """Glicko-2 from a statistic comparison, not its count or winning margin.

    Stat(None, ...) creates independent period streams. Missing pairs skip the
    update; lower-is-better quantities may set higher_is_better=False.
    """

    source: Stat
    side: str = "both"
    fields: tuple = ("rating", "rd", "sigma")
    engine: Glicko2 = Glicko2()
    scope: tuple = ("competition_id",)
    higher_is_better: bool = True
