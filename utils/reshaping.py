from __future__ import annotations
from typing import Iterable, Optional, Tuple, Set, Dict, List
import pandas as pd

def to_team_match_long(
    df: pd.DataFrame,
    *,
    home_prefix: str = "home_",
    away_prefix: str = "away_",
    home_team_col: str = "home_team",
    away_team_col: str = "away_team",
    match_id_col: Optional[str] = "match_id",   # if None, we’ll create one from the index
    shared_passthrough: Optional[Iterable[str]] = None,   # None => auto: all non-prefixed columns
    include_only: Optional[Iterable[str]] = None,  # optional whitelist of TEAM base names (w/o prefix)
    exclude: Optional[Iterable[str]] = None,       # optional blacklist of TEAM base names (w/o prefix)
    opponent_include_only: Optional[Iterable[str]] = None, # whitelist of OPPONENT base names (w/o prefix)
    keep_original_prefixed: bool = False,          # keep or drop home_*/away_* after stacking
) -> pd.DataFrame:
    """
    Convert match-wide (home_/away_ columns) into team-match long format (two rows per match).

    Rules
    -----
    - Columns without the `home_`/`away_` prefix are treated as shared context and copied to both rows.
    - For columns with home_/away_:
        - On the home row:  home_{x} -> team_{x},  away_{x} -> opponent_{x}
        - On the away row:  away_{x} -> team_{x},  home_{x} -> opponent_{x}
    - Adds:
        - team_id: value from home_team_col/away_team_col
        - opponent_id: the opposing team
    (Note: This function does NOT add or modify `is_home`. If present in the input, it will be passed through as a shared column.)
    - Preserves everything else (date/league/season/round/etc.) as shared columns.

    Parameters
    ----------
    shared_passthrough : list of column names to always pass through unchanged (non-prefixed).
                        If None, auto-detect as all columns that don't start with home_/away_.
    include_only : base names (without prefix) to include for TEAM features; if provided, only those stats
                are transformed into `team_*`.
    exclude : base names to exclude from TEAM transformation.
    opponent_include_only : base names (without prefix) to include for OPPONENT features. If None, no opponent
                            features are created by default (prevents feature explosion).
    keep_original_prefixed : keep home_*/away_* columns in the output (default False -> drop).

    Returns
    -------
    stacked : pd.DataFrame
        Long table with columns like:
        [match_id, team_id, opponent_id, league, season, round, ..., team_*, opponent_* (whitelisted), shared...]
    """
    if match_id_col is None:
        # Create a deterministic id from the original order
        df = df.copy()
        df["_tmp_match_row_id"] = range(len(df))
        match_id_col = "_tmp_match_row_id"
    else:
        if match_id_col not in df.columns:
            # fall back to index if requested but missing
            df = df.copy()
            df["_tmp_match_row_id"] = range(len(df))
            match_id_col = "_tmp_match_row_id"

    # Identify prefixed and shared columns
    cols = list(df.columns)
    home_cols = [c for c in cols if c.startswith(home_prefix)]
    away_cols = [c for c in cols if c.startswith(away_prefix)]
    home_bases = {c[len(home_prefix):] for c in home_cols}
    away_bases = {c[len(away_prefix):] for c in away_cols}
    # base names that appear under either prefix
    all_bases = sorted(home_bases | away_bases)

    # Filters for TEAM features
    include_set: Optional[Set[str]] = set(include_only) if include_only is not None else None
    exclude_set: Set[str] = set(exclude) if exclude is not None else set()

    def keep_team_base(b: str) -> bool:
        if include_set is not None and b not in include_set:
            return False
        if b in exclude_set:
            return False
        return True

    team_bases_to_use = [b for b in all_bases if keep_team_base(b)]

    # Filter for OPPONENT features (whitelist only; default None => no opponent columns)
    opponent_set: Optional[Set[str]] = set(opponent_include_only) if opponent_include_only is not None else None
    def keep_opp_base(b: str) -> bool:
        return opponent_set is not None and b in opponent_set

    opp_bases_to_use = [b for b in all_bases if keep_opp_base(b)]

    # Shared passthrough columns: non-prefixed by default (plus match id and team names)
    if shared_passthrough is None:
        shared_cols = [
            c for c in cols
            if not c.startswith(home_prefix) and not c.startswith(away_prefix)
        ]
    else:
        shared_cols = list(shared_passthrough)

    # Ensure keys needed are present in passthrough
    for key in (match_id_col, home_team_col, away_team_col):
        if key not in shared_cols:
            shared_cols.append(key)

    # Build rename maps for the home and away projections
    home_ren: Dict[str, str] = {}
    away_ren: Dict[str, str] = {}

    # TEAM mappings (always included for selected team bases)
    for b in team_bases_to_use:
        h = home_prefix + b
        a = away_prefix + b
        if h in df.columns:
            home_ren[h] = f"team_{b}"
        if a in df.columns:
            away_ren[a] = f"team_{b}"

    # OPPONENT mappings (only for whitelisted opponent bases)
    for b in opp_bases_to_use:
        h = home_prefix + b
        a = away_prefix + b
        if a in df.columns:
            home_ren[a] = f"opponent_{b}"
        if h in df.columns:
            away_ren[h] = f"opponent_{b}"

    # Create a set for efficient lookup
    shared_set = set(shared_cols)
    
    # Create the two projections
    # Home side
    extra_home_keys = [k for k in home_ren.keys() if k not in shared_set] # Guard vs duplicates
    home_view_cols = shared_cols + extra_home_keys
    home_df = df[home_view_cols].rename(columns=home_ren).copy()
    home_df["team_id"] = df[home_team_col].values
    home_df["opponent_id"] = df[away_team_col].values
    # mark side for interleaving
    home_df["_tmp_side_order"] = 0

    # Away side
    extra_away_keys = [k for k in away_ren.keys() if k not in shared_set] # Guard vs duplicates
    away_view_cols = shared_cols + extra_away_keys
    away_df = df[away_view_cols].rename(columns=away_ren).copy()
    away_df["team_id"] = df[away_team_col].values
    away_df["opponent_id"] = df[home_team_col].values
    # mark side for interleaving
    away_df["_tmp_side_order"] = 1

    # --- CONCATENATION LOGIC (interleaved by match) -------------------
    stacked = pd.concat([home_df, away_df], axis=0, ignore_index=True, sort=False)

    # Stable sort so we get: match1 home, match1 away, match2 home, match2 away, ...
    sort_keys = [match_id_col, "_tmp_side_order"]
    # If a temporary per-match creation order exists, use it to preserve original order
    if "_tmp_match_row_id" in stacked.columns:
        sort_keys.append("_tmp_match_row_id")

    stacked = stacked.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
    # -------------------------------------------------------------------------

    # Optionally drop original prefixed columns if any slipped into passthrough
    if not keep_original_prefixed:
        drop_cols = [c for c in stacked.columns if c.startswith(home_prefix) or c.startswith(away_prefix)]
        if drop_cols:
            stacked = stacked.drop(columns=drop_cols)

    # Put useful keys up front for readability
    front = [c for c in [match_id_col, "team_id", "opponent_id"] if c in stacked.columns]
    for c in ("league", "league_index", "season", "season_start", "round", "normalized_round", "date_utc"):
        if c in stacked.columns and c not in front:
            front.append(c)

    team_like = [c for c in stacked.columns if c.startswith("team_")]
    opp_like = [c for c in stacked.columns if c.startswith("opponent_")]
    others = [c for c in stacked.columns if c not in front + team_like + opp_like + ["_tmp_side_order"]]
    stacked = stacked[front + team_like + opp_like + others]

    # Clean up temp side marker
    if "_tmp_side_order" in stacked.columns:
        stacked = stacked.drop(columns=["_tmp_side_order"])

    # If we created a temp match id earlier, we leave it (caller can drop/rename later)
    return stacked