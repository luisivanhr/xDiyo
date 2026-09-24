"""Shared grouping, eligibility and league-season membership helpers."""


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
    import numpy as np

    keys = list(dict.fromkeys(group_by))
    if "team_id" not in keys:
        raise ValueError("Historical groups must include team_id.")
    if head_to_head and "opponent_id" not in keys:
        keys.append("opponent_id")
    cutoff = aligned_times(history, cutoffs, default="kickoff_at")
    kickoff = aligned_times(history, None, default="kickoff_at")
    available = aligned_times(history, available_at, default="kickoff_at")
    if (cutoff > kickoff).any():
        raise ValueError("Prediction cutoffs must not follow the target kickoff.")
    if history[keys].isna().any().any():
        raise ValueError("History grouping identifiers must be nonmissing.")
    if "status" not in history:
        raise KeyError("Historical operators require match status.")
    finished = history["status"].eq("finished").fillna(False).to_numpy(dtype=bool)
    valid = finished & kickoff.notna().to_numpy() & available.notna().to_numpy()
    # Force one unit before comparing timestamps from different pandas backends.
    def nanos(series):
        return series.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    kick_ns, available_ns, cutoff_ns = map(nanos, (kickoff, available, cutoff))
    missing_target_time = cutoff.isna().to_numpy() | kickoff.isna().to_numpy()
    identities = [name for name in ("source_league", "source_season", "event_id") if name in history]
    if "event_id" not in identities:
        raise KeyError("History must include event_id.")
    match_ids = list(zip(*(history[name].tolist() for name in identities)))
    result = [np.array([], dtype=int) for _ in range(len(history))]
    for positions in history.groupby(keys, sort=False, dropna=False).indices.values():
        candidates = positions[valid[positions]]
        candidates = candidates[np.argsort(kick_ns[candidates], kind="stable")]
        for row in positions:
            if missing_target_time[row]:
                continue
            end = np.searchsorted(kick_ns[candidates], cutoff_ns[row], side="left")
            eligible = candidates[:end]
            eligible = eligible[available_ns[eligible] <= cutoff_ns[row]]
            result[row] = np.array(
                [i for i in eligible if match_ids[i] != match_ids[row]], dtype=int,
            )
    return result
