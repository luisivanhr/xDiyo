"""Opt-in observation-regime boundaries for the native history reducers."""

from dataclasses import dataclass
from math import isfinite
from numbers import Real

from .expressions import Expr, H2H, Lag, RollingMean, RollingStd, RollingSkewness, RollingZScore, EMA


@dataclass(frozen=True)
class HistoryInspector(Expr):
    """Reset a historical reducer after a gap in usable observations.

    The cutoff and fixture eligibility rules of the wrapped reducer still apply.
    A strict gap in elapsed kickoff times starts a new regime. Age is measured
    at the prediction cutoff, not the target fixture's kickoff.
    """
    source: Expr
    gap_weeks: float = 26
    evidence: str = "diagnostics"

    def __post_init__(self):
        from .league import League, LeaveOneOut
        from .weighted import RollingWeightedMean

        if (isinstance(self.gap_weeks, bool) or not isinstance(self.gap_weeks, Real)
                or not isfinite(self.gap_weeks) or self.gap_weeks <= 0):
            raise ValueError("HistoryInspector gap_weeks must be a finite positive number.")
        if self.evidence not in ('off', 'diagnostics', 'predictors'):
            raise ValueError("HistoryInspector evidence must be off, diagnostics or predictors.")
        operator = self.source
        while isinstance(operator, H2H):
            operator = operator.source
        if not isinstance(operator, (Lag, RollingMean, RollingStd, RollingSkewness,
                                     RollingZScore, EMA, RollingWeightedMean)):
            raise TypeError("HistoryInspector wraps Lag, a team Rolling reducer, or EMA; "
                            "rating, warm-start, league and standalone spatial operators are not supported.")
        if isinstance(operator.source, (League, LeaveOneOut)):
            raise ValueError("HistoryInspector does not support league-population reductions.")


EVIDENCE_FIELDS = ('eligible_fixture_count', 'usable_observation_count', 'latest_age_days',
                   'expired', 'restarted')
_NS_DAY = 86_400_000_000_000


class Inspection:
    """Evaluate regime membership on one aligned candidate history at a time."""

    def __init__(self, inspector, history, candidates, columns):
        import numpy as np
        from .history import aligned_times

        self.inspector = inspector
        self.candidates = candidates
        numerator, denominator = float(inspector.gap_weeks).as_integer_ratio()
        # The comparison is strict, and elapsed kickoff times are integer ns.
        self.threshold = numerator * (7 * _NS_DAY) // denominator
        self.kicks = aligned_times(history, None, default='kickoff_at').astype('datetime64[ns, UTC]').astype('int64').to_numpy()
        self.identities = (history[[col for col in ('source_league', 'source_season', 'competition_id',
                                                    'season_id', 'event_id', 'team_id', 'side') if col in history]]
                           .to_dict('records') if inspector.evidence != 'off' else None)
        self.records = []
        self.evidence_columns = {
            column: np.full((len(candidates), len(EVIDENCE_FIELDS)), np.nan)
            for column in columns
        } if inspector.evidence == 'predictors' else {}

    def usable(self, row, positions, values, bounds):
        return release_mask(self.candidates.cutoffs[row], positions, bounds, values=values)

    def select(self, row, positions, usable, column, size, *, record=True):
        import numpy as np

        # Positions are already sorted by kickoff and filtered at this row's
        # cutoff, excluding the target identity. Never infer coverage from
        # observations not available at the cutoff.
        observed = np.flatnonzero(usable)
        age_ns = None
        expired = False
        restarted = False
        begin = 0
        if len(observed):
            times = self.kicks[positions[observed]]
            age_ns = int(self.candidates.cutoffs[row]) - int(times[-1])
            expired = age_ns > self.threshold
            # Unsigned subtraction gives the full nonnegative delta for
            # sorted int64 timestamps, even across the Unix epoch.
            if self.threshold < np.iinfo(np.uint64).max:
                breaks = np.flatnonzero(np.diff(times.view(np.uint64)) > np.uint64(self.threshold))
                if len(breaks):
                    begin = observed[breaks[-1] + 1]
                    restarted = True
        if expired or not len(observed):
            selected = positions[:0]
            contributing = 0
        else:
            begin = max(begin, len(positions) - size) if size is not None else begin
            selected = positions[begin:]
            contributing = int(usable[begin:].sum())
        if record and self.identities is not None:
            values = (len(selected), contributing, None if age_ns is None else age_ns / _NS_DAY,
                      expired, restarted)
            evidence = dict(zip(EVIDENCE_FIELDS, values))
            self.records.append({**self.identities[row], 'source_column': column,
                                 'pre_guard_eligible_fixture_count': len(positions), **evidence})
            if self.inspector.evidence == 'predictors':
                self.evidence_columns[column][row] = values
        return selected

    def attach(self, frame):
        if self.identities is not None:
            frame.attrs['history_inspection'] = dict(expression=repr(self.inspector), records=self.records)
        if self.evidence_columns:
            frame.attrs['history_inspection_predictors'] = {
                f'{column}::{field}': values[:, i].tolist()
                for column, values in self.evidence_columns.items() for i, field in enumerate(EVIDENCE_FIELDS)
            }


def release_mask(cutoff, positions, bounds, *, values=None):
    """A supplied NaT is unknown, not permission to fall back to kickoff."""
    import numpy as np

    valid = np.ones(len(positions), dtype=bool) if values is None else np.isfinite(values)
    for release, known in bounds:
        valid &= known[positions] & (release[positions] <= cutoff)
    return valid
