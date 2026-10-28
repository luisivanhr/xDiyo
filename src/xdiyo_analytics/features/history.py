"""Shared grouping, eligibility and league-season membership helpers."""


def frozen_group_anchors(history, keys, cutoffs):
    """Minimum cutoff per complete identity, preserving the input row order.

    Call with the complete fixture schedule, not a prediction subset. Postponed
    fixtures retain their assigned round; stage belongs in keys when required.
    """
    import pandas as pd
    if history[list(keys)].isna().any().any():
        raise ValueError("Round keys must be present and nonmissing.")
    times = aligned_times(history, cutoffs, default="kickoff_at")
    work = history[list(keys)].copy()
    work["_anchor_time"] = times
    return pd.Series(work.groupby(list(keys), sort=False, dropna=False)["_anchor_time"].transform("min"), index=history.index)


def aligned_times(history, values, *, default):
    """Resolve a datetime column, scalar or position-aligned sequence into UTC.

    Series must have the history's index. Numeric Unix times should be converted
    by the caller with their explicit unit before passing them here.
    """
    import pandas as pd
    from pandas.api.types import is_numeric_dtype

    if values is None:
        values = history[default]
    elif isinstance(values, str):
        values = history[values]
    if isinstance(values, pd.Series) and not values.index.equals(history.index):
        raise ValueError("Datetime Series must align with the history index.")
    values = pd.Series(values, index=history.index)
    if is_numeric_dtype(values.dtype):
        raise TypeError("Pass datetimes, not numeric timestamps without a unit.")
    return pd.to_datetime(values, utc=True, errors="raise")


def availability_times(history, values=None):
    """Resolve the selected release policy and honor reviewed completion bounds."""
    from ..data.availability import respect_result_availability
    return respect_result_availability(history, aligned_times(history, values, default="kickoff_at"))


def league_season_team_counts(history):
    """Count distinct participating team IDs from the complete loaded history.

    Call before filtering the experiment population into folds. This counts
    observed membership; callers loading only part of a league should supply
    team_counts to evaluate_features from the full league-season population.
    """
    keys = ["competition_id", "season_id"]
    membership = history[keys + ["team_id"]]
    if "opponent_id" in history:
        import pandas as pd
        opponents = history[keys + ["opponent_id"]].rename(columns={"opponent_id": "team_id"})
        membership = pd.concat([membership, opponents], ignore_index=True)
    return membership.groupby(keys, dropna=False)["team_id"].nunique().rename("team_count")


def eligible_history_rows(history, *, group_by=("team_id", "competition_id"),
                          head_to_head=False, cutoffs=None, available_at=None):
    """Return earlier eligible row positions for every prediction row.

    Group by team and competition by default, across seasons and venues. Add
    season_id or side to group_by to restrict either; H2H adds opponent_id.
    Histories are ordered by kickoff, with input order breaking historical ties.
    Equal-time matches cannot contribute to each other. Candidates must have
    status='finished', kickoff strictly before the cutoff and availability no
    later than it. The target match itself is always excluded.

    cutoffs defaults to kickoff_at, or accepts a datetime column/scalar/sequence
    (e.g. frozen round cutoffs). available_at has the same API. If omitted, earlier
    finished kickoffs are the retrospective availability assumption: completion
    and publication times cannot be inferred from the current export.
    """
    return list(_eligible_history_index(history, group_by=group_by,
                 head_to_head=head_to_head, cutoffs=cutoffs, available_at=available_at))


def _eligible_history_index(history, *, group_by=("team_id", "competition_id"),
                            head_to_head=False, cutoffs=None, available_at=None, venue="all"):
    import numpy as np

    if venue not in ('all', 'same', 'home', 'away'):
        raise ValueError('Historical venue must be all, same, home or away.')
    if venue != 'all' and ('side' not in history or not history.side.isin(['home', 'away']).all()):
        raise ValueError('Venue history requires valid home/away side metadata.')
    keys = list(dict.fromkeys((*group_by, 'side') if venue == 'same' else group_by))
    if "team_id" not in keys:
        raise ValueError("Historical groups must include team_id.")
    if head_to_head and "opponent_id" not in keys:
        keys.append("opponent_id")
    cutoff = aligned_times(history, cutoffs, default="kickoff_at")
    kickoff = aligned_times(history, None, default="kickoff_at")
    available = availability_times(history, available_at)
    if (cutoff > kickoff).any():
        raise ValueError("Prediction cutoffs must not follow the target kickoff.")
    if history[keys].isna().any().any():
        raise ValueError("History grouping identifiers must be nonmissing.")
    if "status" not in history:
        raise KeyError("Historical operators require match status.")
    finished = history["status"].eq("finished").fillna(False).to_numpy(dtype=bool)
    valid = finished & kickoff.notna().to_numpy() & available.notna().to_numpy()
    if venue in ("home", "away"):
        valid &= history.side.eq(venue).to_numpy(dtype=bool)
    # Force one unit before comparing timestamps from different pandas backends.
    def nanos(series):
        return series.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    kick_ns, available_ns, cutoff_ns = map(nanos, (kickoff, available, cutoff))
    missing_target_time = cutoff.isna().to_numpy() | kickoff.isna().to_numpy()
    identities = [name for name in ("source_league", "source_season", "event_id") if name in history]
    if "event_id" not in identities:
        raise KeyError("History must include event_id.")
    match_ids = list(zip(*(history[name].tolist() for name in identities)))
    groups = []
    row_group = np.empty(len(history), dtype=int)
    for positions in history.groupby(keys, sort=False, dropna=False).indices.values():
        candidates = positions[valid[positions]]
        candidates = candidates[np.argsort(kick_ns[candidates], kind="stable")]
        row_group[positions] = len(groups)
        groups.append((candidates, kick_ns[candidates]))
    return _HistoryRows(groups, row_group, available_ns, cutoff_ns, missing_target_time, match_ids)


class _HistoryRows:
    """Invocation-local sorted indices; never retain every preceding prefix.

    Public eligible_history_rows still returns independent arrays. Internally a
    bounded query scans backwards in chunks, stopping after the same last k
    eligible match slots (including observations with missing feature values).
    """
    def __init__(self, groups, row_group, available, cutoffs, missing, identities):
        self.groups, self.row_group = groups, row_group
        self.available, self.cutoffs = available, cutoffs
        self.missing, self.identities = missing, identities
        self._short_rows = {}

    def __len__(self):
        return len(self.row_group)

    def __iter__(self):
        for row in range(len(self)):
            yield self[row]

    def __getitem__(self, row):
        return self.window(row)

    def window(self, row, size=None):
        import numpy as np
        if row >= len(self) or row < -len(self):
            raise IndexError(row)
        if self.missing[row]:
            return np.empty(0, dtype=int)
        candidates, kicks = self.groups[self.row_group[row]]
        # Bound memoization by group size, never by an unbounded history prefix.
        if len(candidates) <= 64:
            if row not in self._short_rows:
                cutoff = self.cutoffs[row]
                end = np.searchsorted(kicks, cutoff, side='left')
                eligible = candidates[:end]
                eligible = eligible[self.available[eligible] <= cutoff]
                self._short_rows[row] = np.array([i for i in eligible if self.identities[i] != self.identities[row]], dtype=int)
            values = self._short_rows[row]
            return values.copy() if size is None else values[-size:].copy()
        cutoff = self.cutoffs[row]
        end = np.searchsorted(kicks, cutoff, side='left')
        if size is None:
            eligible = candidates[:end]
            eligible = eligible[self.available[eligible] <= cutoff]
            return np.array([i for i in eligible if self.identities[i] != self.identities[row]], dtype=int)
        selected = []
        while end and len(selected) < size:
            begin = max(0, end-max(64, size-len(selected)))
            chunk = candidates[begin:end]
            chunk = chunk[self.available[chunk] <= cutoff]
            for i in chunk[::-1]:
                if self.identities[i] != self.identities[row]:
                    selected.append(i)
                    if len(selected) == size:
                        break
            end = begin
        return np.asarray(selected[::-1], dtype=int)
