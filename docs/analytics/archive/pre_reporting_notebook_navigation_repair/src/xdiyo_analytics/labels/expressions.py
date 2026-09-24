"""Observed target definitions, separate from prediction-time feature expressions."""

from dataclasses import dataclass

from ..features.expressions import Stat


class LabelExpr:
    """Base for target expressions consumed by create_labels, not feature evaluation."""


@dataclass(frozen=True)
class TeamValue(LabelExpr):
    """Observed Stat from each focal team's perspective; side=for/against/both."""
    source: Stat
    side: str = "for"


@dataclass(frozen=True)
class MatchTotal(LabelExpr):
    """Sum of both teams' Stat, one row per match; either missing side gives missing."""
    source: Stat


@dataclass(frozen=True)
class Outcome(LabelExpr):
    """Loss=-1, draw=0, win=1 for team, home or away perspective.

    source=None uses the history builder's score-based W/D/L result, without
    resolving shootouts or inventing a regulation-time score. A Stat compares
    own/opponent values; higher_is_better=False reverses this ordering. The team
    perspective returns team-match rows; home/away returns one row per match.
    """
    source: Stat | None = None
    perspective: str = "team"
    higher_is_better: bool = True


@dataclass(frozen=True)
class Above(LabelExpr):
    """Strict source > threshold, encoded 1/0; missing stays missing."""
    source: LabelExpr
    threshold: float


@dataclass(frozen=True)
class BetOption(LabelExpr):
    """Generic settlement of a selected label; no odds, profit or bookmaker rules.

    selection='yes'/'no' takes Above; 'win'/'draw'/'loss' takes Outcome;
    'over'/'under' takes TeamValue or MatchTotal and requires a finite line.
    Over/under equality follows on_equal='push' (default) or 'loss'. A drawn
    Outcome on a win/loss selection follows draw='loss' (default) or 'push'.
    Wins/losses map to 1/0. Push and void numeric values default to missing and
    may be assigned explicit finite values; settlement strings always remain
    distinct. Only explicitly listed void_statuses produce voids, even if the
    outcome itself is missing. Other unfinished/missing results remain missing.
    Single-threshold comparisons only: quarter-line splitting and parlays are
    deferred. For example, line=9.25 means the literal threshold 9.25, not a
    bookmaker's split-stake market. void_statuses accepts a tuple or list.
    """
    source: LabelExpr
    selection: str = "yes"
    line: float | None = None
    on_equal: str = "push"
    draw: str = "loss"
    push_value: float | None = None
    void_value: float | None = None
    void_statuses: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.void_statuses, (tuple, list)) or any(
            not isinstance(value, str) or not value for value in self.void_statuses
        ):
            raise TypeError("void_statuses must be a tuple/list of explicit status strings.")
        object.__setattr__(self, "void_statuses", tuple(self.void_statuses))
