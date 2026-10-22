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


@dataclass(frozen=True)
class BayesianRating(Expr):
    """Filter goal scores with fixed Bayesian parameters and select team fields.

    model=None uses the documented priors. A pretrained BayesianModel must have
    been fitted before the prediction cutoff. No parameter fitting occurs while
    evaluating features. Output selection never changes the full internal state.
    """

    model: object = None
    side: str = "both"
    fields: tuple = ("attack_mean", "defence_vulnerability_mean")


@dataclass(frozen=True)
class BayesianFixture(Expr):
    """Opponent/venue-dependent predictions, in the fixture's home/away frame.

    Shares the BayesianRating producer for the same fixed model. Alternatively,
    name reads a BayesianRatingRun supplied through evaluate_features(ratings=).
    A named run and an explicit model are mutually exclusive.
    """

    model: object = None
    fields: tuple = ("expected_home_goals", "expected_away_goals")
    name: str | None = None

    def __post_init__(self):
        if self.name is not None and self.model is not None:
            raise ValueError("Choose a named Bayesian run or a fixed model, not both.")

@dataclass(frozen=True)
class GAPFixture(Expr):
    """Read a named GAP run; default exports signed home-minus-away differences.

    No fitting or replay is triggered. Home/away predictions can be explicitly
    selected for inspection. Diagnostics are available only on the run itself.
    """
    name: str
    statistics: tuple | None = None
    fields: tuple = ('difference',)


def gap_feature_set(name, statistics):
    """Explicit opt-in mapping using the existing feature naming convention."""
    from ..ratings.gap_core import STATISTICS
    statistics = tuple(statistics)
    if not statistics or len(set(statistics)) != len(statistics) or set(statistics) - set(STATISTICS):
        raise ValueError('Choose distinct paper GAP statistics.')
    return {f'gap_{stat}_diff': GAPFixture(name, (stat,)) for stat in statistics}
