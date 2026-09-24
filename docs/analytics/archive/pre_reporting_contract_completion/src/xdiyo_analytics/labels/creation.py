"""Evaluate observed outcomes with explicit units and identifier alignment."""

from dataclasses import dataclass
from numbers import Real
import math

from ..features.expressions import Stat
from .expressions import LabelExpr, TeamValue, MatchTotal, Outcome, Above, BetOption


@dataclass
class LabelData:
    """One named target, ready for later dataset assembly.

    y is a numeric DataFrame (Stat(None) may expand multiple columns). metadata
    has the same index and row order, with match identifiers, time/status and
    team IDs. unit is 'match' or 'team_match'; perspective is 'team', 'home',
    'away' or 'total'. identity_columns, rather than a possibly repeated pandas
    index, identify rows. definition records the expression. BetOption also
    supplies a same-shape settlement DataFrame: win/loss/push/void/missing.

    These are realized targets, not predictors. No rows are dropped or imputed;
    dataset assembly owns selecting a target, aligning X and choosing final layout.
    """
    y: object
    metadata: object
    unit: str
    perspective: str
    identity_columns: tuple[str, ...]
    definition: LabelExpr
    settlement: object = None


def create_labels(history, labels):
    """Return {name: LabelData} from a complete paired team-history population.

    Each supplied match must have exactly one home and one away row with reversed
    team/opponent IDs and agreeing status. Input order/index/dtypes are preserved
    for team targets; match targets follow the input's home-row order, with home
    and away IDs in metadata. Outcome(perspective='away') uses away observations
    but retains that same match order. Identifiers remain exact (no float casts).

    Only status='finished' produces ordinary labels. Missing/nonfinite numeric
    observations stay NaN, and MatchTotal/Outcome need both sides. No lag, cutoff,
    rolling window, feature warm-up, automatic settlement or population filtering
    is applied. Predictions for unfinished fixtures can therefore retain missing y.
    Stat(None) selects each available period independently, as in feature APIs.
    One-column y uses the requested name; multi-column outputs append the source
    identity. Sources and definitions are not mutated; no file I/O occurs.

    BetOption settlement is a generic rule selected by the caller, not an assertion
    about any bookmaker. An explicitly configured void status overrides missing
    outcomes; missing observations never imply a void on their own.
    """
    import numpy as np
    import pandas as pd

    if not labels:
        raise ValueError("Provide at least one named label expression.")
    required = {"event_id", "team_id", "opponent_id", "side", "status"}
    missing = required - set(history.columns)
    if missing:
        raise KeyError(f"Label history is missing columns: {sorted(missing)}")
    match_keys = [name for name in ("source_league", "source_season", "competition_id", "season_id", "event_id") if name in history]
    if history[match_keys + ["team_id", "opponent_id"]].isna().any().any():
        raise ValueError("Label match and team identifiers must be nonmissing.")
    if not history["side"].isin(["home", "away"]).all():
        raise ValueError("Label history sides must be home or away.")
    if history.duplicated(match_keys + ["side"]).any():
        raise ValueError("Label history contains duplicate match sides.")
    identities = list(zip(*(history[name].tolist() for name in match_keys)))
    home = np.flatnonzero(history["side"].eq("home").to_numpy(dtype=bool))
    away_map = {identities[i]: i for i in np.flatnonzero(history["side"].eq("away").to_numpy(dtype=bool))}
    if set(away_map) != {identities[i] for i in home}:
        raise ValueError("Labels require both team rows for every supplied match.")
    away = np.array([away_map[identities[i]] for i in home], dtype=int)
    teams, opponents = history["team_id"].tolist(), history["opponent_id"].tolist()
    statuses = history["status"].fillna("__missing_status__").tolist()
    for i, j in zip(home, away):
        if teams[i] == opponents[i] or teams[i] != opponents[j] or opponents[i] != teams[j]:
            raise ValueError("Paired label rows must identify opposite teams.")
        if statuses[i] != statuses[j]:
            raise ValueError("Paired label rows must agree on match status.")
    all_rows = np.arange(len(history))
    finished = history["status"].eq("finished").fillna(False).to_numpy(dtype=bool)
    stat_metadata = history.attrs.get("stat_columns", {})
    cache = {}

    def finite_parameter(value, name):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number.")

    def numeric_values(columns):
        frame = history[columns]
        # Object blocks can try float(pd.NA) before to_numpy applies na_value.
        # Replace missing sentinels before conversion; leave source dtypes/IDs intact.
        return frame.where(frame.notna(), np.nan).to_numpy(
            dtype=float, na_value=np.nan, copy=True,
        )

    def pair_values(stat):
        if not isinstance(stat, Stat):
            raise TypeError("Label statistic sources must be Stat references.")
        names = [name for name, info in stat_metadata.items()
                 if info["role"] == "team" and info["group_name"] == stat.group
                 and info["key"] == stat.key and info["field"] == stat.field
                 and (stat.period is None or info["period"] == stat.period)]
        if not names:
            raise KeyError(f"Statistic is absent from selected history: {stat!r}")
        others = []
        for name in names:
            matches = [column for column, info in stat_metadata.items()
                       if info == {**stat_metadata[name], "role": "opponent"}]
            if len(matches) != 1:
                raise ValueError("Target statistics need one matching opponent column.")
            others.append(matches[0])
        own = numeric_values(names)
        other = numeric_values(others)
        own[~np.isfinite(own)] = np.nan
        other[~np.isfinite(other)] = np.nan
        return own, other, names, others

    def evaluate(node):
        if not isinstance(node, LabelExpr):
            raise TypeError("Use label expressions with create_labels; Stat needs TeamValue, MatchTotal or Outcome.")
        if node in cache:
            return cache[node]
        settlement = None
        if isinstance(node, (TeamValue, MatchTotal)):
            own, other, names, other_names = pair_values(node.source)
            if isinstance(node, MatchTotal):
                with np.errstate(over="ignore", invalid="ignore"):
                    values = own[home] + other[home]
                rows, unit, perspective = home, "match", "total"
            else:
                if node.side not in {"for", "against", "both"}:
                    raise ValueError("TeamValue side must be for, against or both.")
                if node.side == "against":
                    values, names = other, other_names
                elif node.side == "both":
                    values, names = np.column_stack((own, other)), names + other_names
                else:
                    values = own
                rows, unit, perspective = all_rows, "team_match", "team"
            values[~np.isfinite(values)] = np.nan
            values[~finished[rows]] = np.nan
        elif isinstance(node, Outcome):
            if node.perspective not in {"team", "home", "away"}:
                raise ValueError("Outcome perspective must be team, home or away.")
            if not isinstance(node.higher_is_better, bool):
                raise TypeError("higher_is_better must be a boolean.")
            if node.source is None:
                if not node.higher_is_better:
                    raise ValueError("higher_is_better=False requires a Stat; match results keep W/D/L meanings.")
                values = history["result"].map({"W": 1., "D": 0., "L": -1.}).to_numpy(dtype=float, na_value=np.nan)[:, None]
                names = ["result"]
            else:
                own, other, names, _ = pair_values(node.source)
                values = np.where(own > other, 1., np.where(own < other, -1., 0.))
                values[~(np.isfinite(own) & np.isfinite(other))] = np.nan
                if not node.higher_is_better:
                    values = -values
            selected = all_rows if node.perspective == "team" else home if node.perspective == "home" else away
            values = values[selected].copy()
            values[~finished[selected]] = np.nan
            rows = all_rows if node.perspective == "team" else home
            unit = "team_match" if node.perspective == "team" else "match"
            perspective = node.perspective
        elif isinstance(node, (Above, BetOption)):
            source, rows, names, unit, perspective, _ = evaluate(node.source)
            if isinstance(node, Above):
                if isinstance(node.source, BetOption):
                    raise TypeError("Above takes a quantity/outcome label, not a settled BetOption.")
                finite_parameter(node.threshold, "threshold")
                values = (source > node.threshold).astype(float)
                values[~np.isfinite(source)] = np.nan
            else:
                if node.on_equal not in {"push", "loss"} or node.draw not in {"push", "loss"}:
                    raise ValueError("on_equal and draw must be push or loss.")
                for name in ("push_value", "void_value"):
                    if getattr(node, name) is not None:
                        finite_parameter(getattr(node, name), name)
                if not isinstance(node.void_statuses, tuple) or any(not isinstance(s, str) or not s for s in node.void_statuses):
                    raise TypeError("void_statuses must be a tuple of explicit status strings.")
                valid = np.isfinite(source)
                won, pushed = np.zeros(source.shape, dtype=bool), np.zeros(source.shape, dtype=bool)
                if node.selection in {"over", "under"}:
                    if not isinstance(node.source, (TeamValue, MatchTotal)):
                        raise TypeError("Over/under takes TeamValue or MatchTotal.")
                    finite_parameter(node.line, "line")
                    won = source > node.line if node.selection == "over" else source < node.line
                    if node.on_equal == "push":
                        pushed = source == node.line
                else:
                    if node.line is not None:
                        raise ValueError("line is only used for over/under selections.")
                    if node.selection in {"win", "draw", "loss"}:
                        if not isinstance(node.source, Outcome):
                            raise TypeError("Win/draw/loss selections take Outcome.")
                        won = source == {"win": 1., "draw": 0., "loss": -1.}[node.selection]
                        if node.selection != "draw" and node.draw == "push":
                            pushed = source == 0
                    elif node.selection in {"yes", "no"}:
                        if not isinstance(node.source, Above):
                            raise TypeError("Yes/no selections take Above.")
                        won = source == (1 if node.selection == "yes" else 0)
                    else:
                        raise ValueError("selection must be over, under, win, draw, loss, yes or no.")
                settlement = np.full(source.shape, "missing", dtype=object)
                settlement[valid] = "loss"
                settlement[valid & won] = "win"
                settlement[valid & pushed] = "push"
                void = history["status"].iloc[rows].isin(node.void_statuses).to_numpy(dtype=bool)
                settlement[void] = "void"
                values = np.full(source.shape, np.nan)
                values[settlement == "win"] = 1.
                values[settlement == "loss"] = 0.
                if node.push_value is not None:
                    values[settlement == "push"] = node.push_value
                if node.void_value is not None:
                    values[settlement == "void"] = node.void_value
        else:
            raise TypeError(f"Unsupported label expression: {type(node).__name__}")
        result = values, rows, names, unit, perspective, settlement
        cache[node] = result
        return result

    results = {}
    context_columns = list(dict.fromkeys([*match_keys, *[c for c in (
        "kickoff_at", "round", "stage", "status", "team_id", "opponent_id", "side",
    ) if c in history]]))
    for name, node in labels.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Label names must be nonempty strings.")
        values, rows, columns, unit, perspective, settlement = evaluate(node)
        columns = [name] if len(columns) == 1 else [f"{name}::{column}" for column in columns]
        index = history.index.take(rows)
        y = pd.DataFrame(values.copy(), index=index, columns=columns)
        metadata = history.iloc[rows][context_columns].copy()
        identity_columns = tuple(match_keys)
        if unit == "match":
            metadata = metadata.rename(columns={"team_id": "home_id", "opponent_id": "away_id"}).drop(columns="side")
        else:
            identity_columns += ("team_id",)
        results[name] = LabelData(
            y=y, metadata=metadata, unit=unit, perspective=perspective,
            identity_columns=identity_columns, definition=node,
            settlement=None if settlement is None else pd.DataFrame(settlement.copy(), index=index, columns=columns),
        )
    return results
