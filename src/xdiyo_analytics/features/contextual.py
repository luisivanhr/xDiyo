"""Known fixture context and eligible-history rest intervals."""
from copy import deepcopy
from dataclasses import dataclass

from .expressions import Expr


@dataclass(frozen=True)
class RestDays(Expr):
    """Days since the latest eligible finished kickoff in the current scope.

    Uses the evaluator's grouping, cutoffs and result availability. No eligible
    history yields missing; current and future fixtures never supply the prior.
    The interval ends at this fixture's kickoff, even with an earlier cutoff.
    """


@dataclass(frozen=True)
class CalendarFeature(Expr):
    """UTC fixture calendar: month_sin, month_cos, weekday, or numeric round.

    Month uses sin/cos(2*pi*month/12); weekday is Monday=0. Missing timestamps or
    nonnumeric rounds stay missing. Scheduled fixture context is known input.
    """
    kind: str = "month_sin"


def evaluate_context_features(metadata, definitions):
    """Evaluate named CalendarFeature expressions without changing row identity.

    Works on team history or assembled match metadata. Calling after assembly
    avoids duplicated home/away columns for shared fixture calendar context.
    """
    import numpy as np
    import pandas as pd
    columns = {}
    for name, expression in definitions.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Context feature names must be nonempty strings.")
        if not isinstance(expression, CalendarFeature):
            raise TypeError("Context features require CalendarFeature expressions.")
        if expression.kind == "round":
            value = pd.to_numeric(metadata["round"], errors="coerce").astype(float)
        elif expression.kind in ("month_sin", "month_cos", "weekday"):
            kickoff = pd.to_datetime(metadata["kickoff_at"], utc=True)
            if expression.kind == "weekday":
                value = kickoff.dt.dayofweek.astype(float)
            else:
                angle = 2 * np.pi * kickoff.dt.month / 12
                value = np.sin(angle) if expression.kind == "month_sin" else np.cos(angle)
        else:
            raise ValueError("Calendar kind must be month_sin, month_cos, weekday or round.")
        columns[name] = value
    result = pd.DataFrame(columns, index=metadata.index)
    result.attrs = deepcopy(metadata.attrs)
    return result
