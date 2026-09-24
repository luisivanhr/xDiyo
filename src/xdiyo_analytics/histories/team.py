"""Reshape selected match observations into chronological team histories."""

from urllib.parse import quote

from ..data.loading import SeasonData


def build_team_history(data: SeasonData, *, stat_fields=("value",)):
    """Return a DataFrame with two rows per match, ordered by UTC kickoff.

    Pass the output of select_stats (or loaded matches with optional statistics
    and pregame). Matches remain present even when their statistics are missing.
    kickoff_utc is Unix seconds; kickoff_at is a UTC-aware datetime. Missing or
    unrepresentable timestamps become NaT and sort last. No calendar-day
    truncation, timezone localization, historical lag or eligibility filter occurs.

    Context columns include event/team/opponent IDs, side, competition/season,
    names when supplied, goals_for/against and result. Goals are the provider's
    score_current values. W/D/L is assigned only for status='finished' with both
    scores present; this is a score comparison, not a penalty-shootout winner.
    Unfinished matches remain in the output and have no result.

    Statistics use team::period::group::key::field and opponent::... columns.
    Components are percent-escaped (spaces/underscores remain readable), keeping
    arbitrary group/key names distinct. stat_fields defaults to numeric value;
    total and display may be explicitly included without parsing or scaling them.
    history.attrs['stat_columns'] maps each output column to its original labels.
    Pregame position becomes team_position/opponent_position when supplied.

    Source league/season identify partitions when present. No dependency on
    source_key, coverage or raw payloads. Source frames and files are unchanged.
    This table contains observed outcomes, not prediction-ready features; kickoff
    alone does not establish when results or statistics became available.
    """
    import pandas as pd

    if data.matches is None:
        raise KeyError("Load matches before building team history.")
    if isinstance(stat_fields, str):
        stat_fields = (stat_fields,)
    stat_fields = tuple(dict.fromkeys(stat_fields))
    if not stat_fields or set(stat_fields) - {"value", "total", "display"}:
        raise ValueError("stat_fields must select value, total and/or display.")

    matches = data.matches.reset_index(drop=True)
    keys = [name for name in ("source_league", "source_season") if name in matches]
    keys += ["event_id"]
    required = [*keys, "home_id", "away_id", "kickoff_utc"]
    missing = set(required) - set(matches.columns)
    if missing:
        raise KeyError(f"Matches is missing columns: {sorted(missing)}")
    if matches[keys].isna().any().any() or matches.duplicated(keys).any():
        raise ValueError("Each match must have a unique, nonmissing partition/event identity.")

    context = keys + [name for name in (
        "competition_id", "tournament_id", "season_id", "season_year", "round",
        "status", "status_code", "is_awarded",
    ) if name in matches]
    seconds = matches["kickoff_utc"]
    # Arrow floats can overflow before pandas applies errors='coerce'. Mask
    # unrepresentable seconds and use the numeric conversion path explicitly.
    seconds = seconds.where(seconds.between(
        pd.Timestamp.min.value / 1e9, pd.Timestamp.max.value / 1e9,
    )).to_numpy(dtype="float64", na_value=float("nan"))
    kickoff = pd.to_datetime(seconds, unit="s", utc=True, errors="coerce")
    rows = []
    for side, opponent in (("home", "away"), ("away", "home")):
        part = matches[context].copy()
        part["kickoff_at"] = kickoff
        part["side"] = pd.Series(side, index=part.index, dtype="string")
        part["team_id"] = matches[f"{side}_id"]
        part["opponent_id"] = matches[f"{opponent}_id"]
        for prefix, role in ((side, "team"), (opponent, "opponent")):
            if f"{prefix}_name" in matches:
                part[f"{role}_name"] = matches[f"{prefix}_name"]
        for prefix, label in ((side, "goals_for"), (opponent, "goals_against")):
            name = f"{prefix}_score_current"
            part[label] = matches[name] if name in matches else pd.Series(
                pd.NA, index=part.index, dtype="Float64",
            )
        rows.append(part)
    history = pd.concat(rows, ignore_index=True)
    history.attrs = {}
    history["result"] = pd.Series(pd.NA, index=history.index, dtype="string")
    if "status" in history:
        finished = history["status"].eq("finished").fillna(False)
        scored = finished & history[["goals_for", "goals_against"]].notna().all(axis=1)
        for label, comparison in (
            ("W", history["goals_for"] > history["goals_against"]),
            ("D", history["goals_for"] == history["goals_against"]),
            ("L", history["goals_for"] < history["goals_against"]),
        ):
            history.loc[scored & comparison.fillna(False), "result"] = label

    side_keys = [*keys, "side"]
    reference = pd.MultiIndex.from_frame(history[side_keys])

    def check_sides(frame, label):
        missing = set(side_keys) - set(frame.columns)
        if missing:
            raise KeyError(f"{label} is missing match identifiers: {sorted(missing)}")
        if not pd.MultiIndex.from_frame(frame[side_keys]).isin(reference).all():
            raise ValueError(f"{label} contains a match/side absent from matches.")
        if "team_id" in frame:
            expected = frame[side_keys].merge(
                history[side_keys + ["team_id"]], on=side_keys, validate="many_to_one",
            )["team_id"]
            supplied = frame["team_id"].reset_index(drop=True)
            if (supplied.notna() & supplied.ne(expected).fillna(True)).any():
                raise ValueError(f"{label} team_id disagrees with its match side.")

    def attach_both(wide, names):
        nonlocal history
        history = history.merge(
            wide.rename(columns=names["team"]), on=side_keys,
            how="left", sort=False, validate="one_to_one",
        )
        other = wide.copy(deep=False)
        other["side"] = other["side"].map({"home": "away", "away": "home"})
        history = history.merge(
            other.rename(columns=names["opponent"]), on=side_keys,
            how="left", sort=False, validate="one_to_one",
        )

    column_info = {}
    if data.statistics is not None:
        statistics = data.statistics
        check_sides(statistics, "Statistics")
        fields = ["period", "group_name", "key"]
        missing = set([*fields, *stat_fields]) - set(statistics.columns)
        if missing:
            raise KeyError(f"Statistics is missing columns: {sorted(missing)}")
        if statistics[fields].isna().any().any():
            raise ValueError("Statistic period, group and key must be nonmissing.")
        if not statistics.empty:
            # pivot, not pivot_table: duplicate observations must not be averaged.
            wide = statistics.pivot(index=side_keys, columns=fields, values=list(stat_fields))
            names = {"team": {}, "opponent": {}}
            labels = list(wide.columns)
            # Temporary integer labels keep identifier columns flat during merges.
            wide.columns = range(len(labels))
            for i, (field, period, group, key) in enumerate(labels):
                for role in names:
                    components = (role, period, group, key, field)
                    name = "::".join(quote(str(value), safe=" _") for value in components)
                    names[role][i] = name
                    column_info[name] = {
                        "role": role, "period": period, "group_name": group,
                        "key": key, "field": field,
                    }
            attach_both(wide.reset_index(), names)

    if data.pregame is not None:
        pregame = data.pregame
        check_sides(pregame, "Pregame")
        if "position" not in pregame:
            raise KeyError("Pregame is missing position.")
        attach_both(pregame[side_keys + ["position"]], {
            "team": {"position": "team_position"},
            "opponent": {"position": "opponent_position"},
        })

    history = history.sort_values(
        ["kickoff_at", *keys, "side"], kind="stable", na_position="last",
    ).reset_index(drop=True)
    history.attrs = {"source": data.provenance, "stat_columns": column_info}
    return history
