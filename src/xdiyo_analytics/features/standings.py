"""Season-scoped, score-derived league and form-table feature selection."""
from dataclasses import dataclass
from numbers import Integral
from .expressions import Expr


@dataclass(frozen=True)
class DerivedStandings(Expr):
    """Rank every team using its own last-w matches; never average past ranks.

    Form ranks are (N-R)/(N-1). Optional cumulative position is R or R/N.
    Complete verified membership and schedule are required. The first round
    must be completed and available before any noninitial ranks are published.
    No previous-season state, penalties or provider standings are inherited.
    """
    normalized_ranks: tuple = (5,)
    include_raw: bool = False
    normalize: bool = False
    side: str = "for"

    def __post_init__(self):
        windows = () if self.normalized_ranks is None else self.normalized_ranks
        if not isinstance(windows, (tuple, list)):
            raise ValueError("normalized_ranks must be an ordered list of positive integer windows.")
        if any(isinstance(w, bool) or not isinstance(w, Integral) or w < 1 for w in windows):
            raise ValueError("normalized_ranks must contain positive integers, not booleans.")
        if len(set(windows)) != len(windows):
            raise ValueError("normalized_ranks must not contain duplicates.")
        object.__setattr__(self, "normalized_ranks", tuple(int(w) for w in windows))
        if not isinstance(self.include_raw, bool) or not isinstance(self.normalize, bool):
            raise TypeError("include_raw and normalize must be booleans.")
        if not windows and not self.include_raw:
            raise ValueError("Select at least one form window or include_raw=True.")
        if self.side not in ("for", "against", "both"):
            raise ValueError("side must be for, against or both.")
