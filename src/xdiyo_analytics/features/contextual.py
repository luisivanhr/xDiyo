"""Known fixture context and eligible-history rest intervals."""
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral

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


@dataclass(frozen=True)
class SeasonProgress(Expr):
    """Normalized season position by assigned round or UTC kickoff calendar day.

    Postponed fixtures keep their original round: a round-5 match played during
    round 12 still has progress 5 / total_rounds. Kickoff order, completed games
    and results do not enter this calculation. By default the denominator is
    the maximum assigned round per competition/season in the full input schedule,
    including future fixtures. Supply total_rounds for a partial schedule.
    Missing/nonnumeric rounds stay
    missing; invalid numeric rounds raise instead of being silently clipped.

    mode='kickoff' uses inclusive UTC calendar days from the first through the
    last kickoff per league-season. Postponements use their actual recorded
    kickoff date. total_days and start_date optionally override these boundaries.
    Inference assumes a full schedule; revised schedules can change the scale.
    """

    total_rounds: int | None = None
    mode: str = 'rounds'
    total_days: int | None = None
    start_date: str | None = None

    def __post_init__(self):
        if self.mode not in ('rounds', 'kickoff'):
            raise ValueError("SeasonProgress mode must be rounds or kickoff.")
        if self.total_rounds is not None and (isinstance(self.total_rounds, bool) or not isinstance(self.total_rounds, Integral) or self.total_rounds < 1):
            raise ValueError("SeasonProgress total_rounds must be a positive integer.")
        if self.total_days is not None and (isinstance(self.total_days, bool) or not isinstance(self.total_days, Integral) or self.total_days < 1):
            raise ValueError("SeasonProgress total_days must be a positive integer.")
        if self.mode == 'rounds' and (self.total_days is not None or self.start_date is not None):
            raise ValueError('SeasonProgress total_days/start_date require kickoff mode.')
        if self.mode == 'kickoff' and self.total_rounds is not None:
            raise ValueError('SeasonProgress total_rounds requires rounds mode.')
        if self.start_date is not None:
            import pandas as pd
            if not isinstance(self.start_date, str) or pd.isna(pd.to_datetime(self.start_date, utc=True, errors='coerce')):
                raise ValueError('SeasonProgress start_date must be a valid date string.')


def season_progress(metadata, expression):
    import numpy as np
    import pandas as pd

    if expression.mode == 'kickoff':
        days = pd.to_datetime(metadata['kickoff_at'], utc=True, errors='coerce', format='mixed').dt.normalize().reset_index(drop=True)
        keys = ['competition_id', 'season_id']
        if expression.start_date is None or expression.total_days is None:
            if any(key not in metadata for key in keys):
                raise ValueError('SeasonProgress kickoff inference needs competition_id and season_id, or both start_date and total_days.')
            schedule = metadata[keys].reset_index(drop=True).copy()
            schedule['_day'] = days
            grouped = schedule.groupby(keys, dropna=False)['_day']
            start = grouped.transform('min')
            end = grouped.transform('max')
            unknown = schedule[keys].isna().any(axis=1)
        else:
            unknown = pd.Series(False, index=days.index)
        if expression.start_date is not None:
            start = pd.to_datetime(expression.start_date, utc=True).normalize()
        position = (days - start).dt.days + 1
        total = expression.total_days if expression.total_days is not None else (end - start).dt.days + 1
        if ((position < 1) | (position > total)).any():
            raise ValueError('SeasonProgress kickoff is outside the configured start_date/total_days.')
        result = (position / total).astype(float).mask(unknown)
        result.index = metadata.index
        return result

    rounds = pd.to_numeric(metadata['round'], errors='coerce').astype(float)
    invalid = rounds.notna() & (~np.isfinite(rounds) | (rounds < 1) | (rounds % 1 != 0))
    if expression.total_rounds is None:
        keys = ['competition_id', 'season_id']
        if any(key not in metadata for key in keys):
            raise ValueError('SeasonProgress inference needs competition_id and season_id, or an explicit total_rounds.')
        # Positional grouping preserves duplicated/custom input indexes.
        schedule = metadata[keys].reset_index(drop=True).copy()
        schedule['_round'] = rounds.to_numpy()
        totals = schedule.groupby(keys, dropna=False)['_round'].transform('max').to_numpy()
        totals[schedule[keys].isna().any(axis=1).to_numpy()] = np.nan
    else:
        totals = expression.total_rounds
        invalid |= rounds > totals
    if invalid.any():
        raise ValueError('SeasonProgress rounds must be integers between 1 and total_rounds; check the season or stage configuration.')
    return rounds / totals


def evaluate_context_features(metadata, definitions):
    """Evaluate named CalendarFeature/SeasonProgress expressions with stable rows.

    Works on team history or assembled match metadata. Calling after assembly
    avoids duplicated home/away columns for shared fixture calendar context.
    SeasonProgress inference requires the full schedule; evaluate before row
    filtering or supply explicit boundaries when using an assembled subset.
    """
    import numpy as np
    import pandas as pd
    columns = {}
    for name, expression in definitions.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Context feature names must be nonempty strings.")
        if isinstance(expression, SeasonProgress):
            columns[name] = season_progress(metadata, expression)
            continue
        if not isinstance(expression, CalendarFeature):
            raise TypeError("Context features require CalendarFeature or SeasonProgress expressions.")
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
