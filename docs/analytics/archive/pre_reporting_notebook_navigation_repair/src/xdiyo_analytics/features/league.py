"""League populations: choose a window, exclude contributions, then aggregate."""

from dataclasses import dataclass
from numbers import Integral

from .expressions import Expr, Stat, ForAgainst
from .history import aligned_times


@dataclass(frozen=True)
class League(Expr):
    """A league population, not yet a reduced feature.

    unit='team' counts one team contribution; 'match' sums both teams once.
    schedule='completed_rounds' freezes at the minimum prediction cutoff of the
    target round; 'kickoff' updates at each row's cutoff. window_unit chooses
    rounds, matches or days. Round completeness refers to the supplied fixture
    population; pass round_keys including a stage if round numbers repeat.
    Windows can cross seasons within a competition.
    """

    source: Expr
    unit: str = "team"
    schedule: str = "completed_rounds"
    window_unit: str = "rounds"
    round_keys: tuple = ("competition_id", "season_id", "round")


@dataclass(frozen=True)
class LeaveOneOut(Expr):
    """Exclude focal team contributions (default) or its entire fixtures.

    Exclusion happens AFTER window selection, without refilling the window.
    Match-total populations require exclude='fixtures'. This is unrelated to
    leave-one-covariate-out model importance.
    """

    source: League
    exclude: str = "team_contributions"


class LeaguePopulation:
    """Reusable chronological plans shared by league mean/std/Z-score operators."""

    def __init__(self, history, *, cutoffs=None, available_at=None):
        import numpy as np
        self.history = history
        self.cutoff = aligned_times(history, cutoffs, default="kickoff_at").reset_index(drop=True)
        self.kickoff = aligned_times(history, None, default="kickoff_at").reset_index(drop=True)
        self.available = aligned_times(history, available_at, default="kickoff_at").reset_index(drop=True)
        if (self.cutoff > self.kickoff).any():
            raise ValueError("Prediction cutoffs must not follow target kickoff.")
        self.finished = history["status"].eq("finished").fillna(False).to_numpy(dtype=bool)
        self.valid = self.finished & self.kickoff.notna().to_numpy() & self.available.notna().to_numpy()
        self.teams = history["team_id"].to_numpy()
        self.opponents = history["opponent_id"].to_numpy()
        self.competitions = history["competition_id"].to_numpy()
        match_keys = [x for x in ("competition_id", "season_id", "event_id") if x in history]
        self.matches = list(zip(*(history[x].tolist() for x in match_keys)))
        self._plans = {}
        self._rounds = {}

    def round_info(self, keys):
        """Group fixture rows and calculate frozen anchors and completion times."""
        import pandas as pd
        if keys not in self._rounds:
            if self.history[list(keys)].isna().any().any():
                raise ValueError("Round keys must be present and nonmissing.")
            groups = self.history.groupby(list(keys), sort=False, dropna=False).indices
            info, row_keys = {}, [None] * len(self.history)
            for key, positions in groups.items():
                key = key if isinstance(key, tuple) else (key,)
                anchor = self.cutoff.iloc[positions].min()
                close = self.available.iloc[positions].max() if self.valid[positions].all() else pd.NaT
                info[key] = (positions, anchor, close, self.kickoff.iloc[positions].max())
                for i in positions:
                    row_keys[i] = key
            self._rounds[keys] = info, row_keys
        return self._rounds[keys]

    def rows(self, league, window, exclude=None):
        import numpy as np
        import pandas as pd
        if league.unit not in {"team", "match"} or league.schedule not in {"completed_rounds", "kickoff"}:
            raise ValueError("Choose unit=team/match and schedule=completed_rounds/kickoff.")
        if league.window_unit not in {"rounds", "matches", "days"}:
            raise ValueError("window_unit must be rounds, matches or days.")
        if isinstance(window, bool) or not isinstance(window, Integral) or window < 1:
            raise ValueError("League window must be a positive integer.")
        if exclude not in {None, "team_contributions", "fixtures"}:
            raise ValueError("LOO excludes team_contributions or fixtures.")
        if league.unit == "match" and exclude == "team_contributions":
            raise ValueError("A match-total observation is indivisible; use exclude='fixtures'.")
        key = (league.unit, league.schedule, league.window_unit, league.round_keys, window, exclude)
        if key in self._plans:
            return self._plans[key]
        use_rounds = league.schedule == "completed_rounds" or league.window_unit == "rounds"
        info, row_round = self.round_info(league.round_keys) if use_rounds else ({}, [None] * len(self.history))
        by_comp = self.history.groupby("competition_id", sort=False).indices
        base_cache, output = {}, []
        for row in range(len(self.history)):
            time = info[row_round[row]][1] if league.schedule == "completed_rounds" else self.cutoff.iloc[row]
            if pd.isna(time) or pd.isna(self.kickoff.iloc[row]) or pd.isna(self.cutoff.iloc[row]):
                output.append(np.array([], dtype=int))
                continue
            signature = (self.competitions[row], time)
            if signature not in base_cache:
                positions = by_comp[self.competitions[row]]
                mask = self.valid[positions] & (self.kickoff.iloc[positions] < time).to_numpy()
                mask &= (self.available.iloc[positions] <= time).to_numpy()
                positions = positions[mask]
                if league.schedule == "completed_rounds":
                    complete = {k for k, (_, _, close, last_kick) in info.items()
                                if pd.notna(close) and close <= time and last_kick < time}
                    positions = np.array([i for i in positions if row_round[i] in complete], dtype=int)
                positions = positions[np.argsort(self.kickoff.iloc[positions].to_numpy(), kind="stable")]
                if league.window_unit == "rounds":
                    # Completion order in frozen mode; latest eligible observation
                    # order in kickoff mode (which may include a partial round).
                    seen = list(dict.fromkeys(row_round[i] for i in positions))
                    if league.schedule == "completed_rounds":
                        seen.sort(key=lambda k: (info[k][2], repr(k)))
                    else:
                        latest = {row_round[i]: self.kickoff.iloc[i] for i in positions}
                        seen.sort(key=lambda k: (latest[k], repr(k)))
                    chosen = set(seen[-window:])
                    positions = np.array([i for i in positions if row_round[i] in chosen], dtype=int)
                elif league.window_unit == "matches":
                    chosen = set(list(dict.fromkeys(self.matches[i] for i in positions))[-window:])
                    positions = np.array([i for i in positions if self.matches[i] in chosen], dtype=int)
                else:
                    positions = positions[(self.kickoff.iloc[positions] >= time - pd.Timedelta(days=window)).to_numpy()]
                if league.unit == "match":
                    positions = positions[self.history.iloc[positions]["side"].eq("home").to_numpy(dtype=bool)]
                base_cache[signature] = positions
            positions = base_cache[signature]
            if exclude is not None:
                keep = self.teams[positions] != self.teams[row]
                if exclude == "fixtures":
                    keep &= self.opponents[positions] != self.teams[row]
                positions = positions[keep]
            output.append(positions)
        self._plans[key] = output
        return output


def league_values(history, source, frame, unit):
    """Convert selected Stat inputs to team contributions or match totals."""
    import pandas as pd
    if not isinstance(source, (Stat, ForAgainst)):
        raise TypeError("League currently takes Stat or ForAgainst(Stat).")
    if unit == "team":
        return frame
    if not isinstance(source, Stat):
        raise TypeError("Match totals take a Stat; ForAgainst would duplicate the total.")
    metadata = history.attrs.get("stat_columns", {})
    values = {}
    for name in frame:
        identity = metadata[name]
        other = [column for column, info in metadata.items() if info == {**identity, "role": "opponent"}]
        if len(other) != 1:
            raise ValueError("Match totals require the matching opponent statistic.")
        values[name] = frame[name] + history[other[0]]
    return pd.DataFrame(values, index=history.index)


def reduce_league(node, population, evaluate):
    """Reduce pooled observations, never a series of round averages."""
    import numpy as np
    import pandas as pd
    from .expressions import RollingMean, RollingStd, RollingZScore
    if not isinstance(node, (RollingMean, RollingStd, RollingZScore)):
        raise TypeError("League populations support RollingMean, RollingStd and RollingZScore.")
    selection = node.source
    league = selection.source if isinstance(selection, LeaveOneOut) else selection
    if not isinstance(league, League):
        raise TypeError("LeaveOneOut takes a League population.")
    if isinstance(node.min_periods, bool) or not isinstance(node.min_periods, Integral) or node.min_periods < 1:
        raise ValueError("min_periods must be a positive observation count.")
    if hasattr(node, "ddof") and (isinstance(node.ddof, bool) or not isinstance(node.ddof, Integral) or node.ddof < 0):
        raise ValueError("ddof must be a nonnegative integer.")
    frame, scope = evaluate(league.source)
    if scope:
        raise ValueError("H2H cannot change a league population.")
    frame = league_values(population.history, league.source, frame, league.unit)
    values = frame.to_numpy(dtype=float, na_value=np.nan)
    output = np.full(values.shape, np.nan)
    reference = None
    if isinstance(node, RollingZScore):
        if node.reference is None:
            raise ValueError("League Z-score needs a historical reference, e.g. reference=Lag(stat).")
        reference, _ = evaluate(node.reference)
        if isinstance(node.reference, (Stat, ForAgainst, League, LeaveOneOut)):
            raise ValueError("A league Z-score reference must be historical or known context.")
        if reference.shape[1] != frame.shape[1]:
            raise ValueError("Z-score reference and population must have matching column counts.")
        reference = reference.to_numpy(dtype=float, na_value=np.nan)
    rows = population.rows(league, node.window, selection.exclude if isinstance(selection, LeaveOneOut) else None)
    for row, positions in enumerate(rows):
        for col in range(values.shape[1]):
            sample = values[positions, col]
            sample = sample[np.isfinite(sample)]
            if len(sample) < node.min_periods:
                continue
            if isinstance(node, RollingMean):
                output[row, col] = sample.mean()
            elif len(sample) > node.ddof:
                std = sample.std(ddof=node.ddof)
                if isinstance(node, RollingStd):
                    output[row, col] = std
                elif std > 0 and np.isfinite(reference[row, col]):
                    output[row, col] = (reference[row, col] - sample.mean()) / std
    return pd.DataFrame(output, index=frame.index, columns=frame.columns)
