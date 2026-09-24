"""Replay match results or statistic comparisons into independent rating streams."""

from collections import defaultdict
from itertools import groupby
from urllib.parse import quote

from .glicko import Glicko2
from .snapshots import RatingRun


def build_ratings(history, *, stat=None, engine=None, higher_is_better=True,
                  scope=("competition_id",), available_at=None, transition=None,
                  team_seasons=None, season_starts=None, transition_context=None):
    """Build saved-state data from the complete two-rows-per-match team history.

    stat=None uses finished-match W/D/L. A Stat reference compares that team's
    and its opponent's selected measure (greater/equal/less gives 1/.5/0).
    None periods expand into separate streams; no margins or counts are fed
    directly into the Glicko outcome formula. Missing pairs skip the update.
    higher_is_better=False reverses comparisons, e.g. for errors or fouls.

    The default Glicko2 engine uses one event-driven period per match. Events
    released at the same timestamp are a simultaneous batch: every update reads
    pre-batch opponent states. No automatic calendar idle periods or season
    resets. Scope defaults to competition; scope=() follows teams across leagues.
    Add season_id explicitly if a season reset is desired.

    Updates are ordered by availability (kickoff proxy when omitted), then kickoff
    and match identity. Both perspectives must agree on result availability.
    Source frames are unchanged; returned snapshots can be saved and reused.
    New/revised histories are replayed; incremental appending is not implemented.

    transition=None retains the original replay unchanged. An explicit adapter
    applies once to full team state at each season entry, before equal-time
    result updates. GlickoTransition controls Glicko-specific inflation/shrinkage;
    custom engines supply their own adapter. team_seasons carries explicit or
    evidence-derived movement; season_starts optionally sets earlier boundaries.
    Direct replay defaults to first kickoff; the feature evaluator instead uses
    the first supplied prediction cutoff. Transition snapshots remain saved.
    """
    import numpy as np
    import pandas as pd
    from ..features.history import aligned_times

    engine = Glicko2() if engine is None else engine
    scope = (scope,) if isinstance(scope, str) else tuple(scope)
    if len(set(scope)) != len(scope) or {"team_id", "opponent_id", "side"} & set(scope):
        raise ValueError("Rating scope must contain distinct shared match context columns.")
    initial = dict(engine.initial_state())
    transition_replay = None
    if transition is not None:
        from .transitions import GlickoTransition, TransitionReplay
        from ..features.transitions import TransitionContext
        if isinstance(transition, GlickoTransition) and not isinstance(engine, Glicko2):
            raise TypeError("Custom engines need their own transition adapter; GlickoTransition is for Glicko2.")
        if not callable(getattr(transition, "apply", None)):
            raise TypeError("Rating transition adapters must implement apply(state, context=..., cohort=...).")
        if transition_context is None:
            transition_context = TransitionContext(history, available_at=available_at,
                                                   team_seasons=team_seasons, season_starts=season_starts)
        transition_replay = TransitionReplay(transition_context, scope, transition, initial)
    if not initial or not all(np.isfinite(value) for value in initial.values()):
        raise ValueError("Rating engines must provide a finite numeric initial state.")
    reserved = {*scope, "team_id", "stream", "recorded_at", "latest_kickoff_at", "games_seen", "snapshot_order", "snapshot_kind"}
    if reserved & set(initial):
        raise ValueError("Rating state fields conflict with snapshot identifiers.")
    match_keys = [name for name in (
        "source_league", "source_season", "competition_id", "season_id", "event_id",
    ) if name in history]
    if "event_id" not in match_keys:
        raise KeyError("History must include event_id.")
    if not history["side"].isin(["home", "away"]).all():
        raise ValueError("History sides must be home or away.")
    if history.duplicated(match_keys + ["side"]).any():
        raise ValueError("History has duplicate match sides.")
    if not history.groupby(match_keys, dropna=False).size().eq(2).all():
        raise ValueError("Rating replay requires both team rows for each match.")
    if history[list(dict.fromkeys([*scope, *match_keys, "team_id", "opponent_id"]))].isna().any().any():
        raise ValueError("Rating match, team and scope identifiers must be nonmissing.")
    kickoff = aligned_times(history, None, default="kickoff_at")
    available = aligned_times(history, available_at, default="kickoff_at")
    if (available < kickoff).any():
        raise ValueError("Results cannot be available before their match kickoff.")
    timing = history[match_keys].copy()
    timing["available"] = available.to_numpy()
    if timing.groupby(match_keys, dropna=False)["available"].nunique(dropna=False).gt(1).any():
        raise ValueError("Both match perspectives must share the result-availability time.")
    home_mask = history["side"].eq("home").to_numpy(dtype=bool)
    home = history.loc[home_mask].reset_index(drop=True)
    kicks = kickoff.loc[home_mask].reset_index(drop=True)
    release = available.loc[home_mask].reset_index(drop=True)
    finished = home["status"].eq("finished").fillna(False).to_numpy(dtype=bool)
    valid_time = kicks.notna().to_numpy() & release.notna().to_numpy()
    comparisons = {}
    if stat is None:
        comparisons["result"] = home["result"].map({"W": 1., "D": .5, "L": 0.}).to_numpy(dtype=float, na_value=np.nan)
    else:
        metadata = history.attrs.get("stat_columns", {})
        for name, info in metadata.items():
            if not (info["role"] == "team" and info["group_name"] == stat.group and
                    info["key"] == stat.key and info["field"] == stat.field and
                    (stat.period is None or stat.period == info["period"])):
                continue
            opponent = [column for column, other in metadata.items() if
                        other == {**info, "role": "opponent"}]
            if len(opponent) != 1:
                raise ValueError("Statistic ratings require one matching opponent column.")
            own = home[name].to_numpy(dtype=float, na_value=np.nan)
            other = home[opponent[0]].to_numpy(dtype=float, na_value=np.nan)
            outcome = np.where(own > other, 1., np.where(own < other, 0., .5))
            outcome[~(np.isfinite(own) & np.isfinite(other))] = np.nan
            if not higher_is_better:
                outcome = 1 - outcome
            label = "::".join(quote(str(info[key]), safe=" _") for key in ("period", "group_name", "key", "field"))
            comparisons[label] = outcome
        if not comparisons:
            raise KeyError(f"Statistic is absent from selected history: {stat!r}")

    scopes = list(zip(*(home[name].tolist() for name in scope))) if scope else [()] * len(home)
    match_ids = list(zip(*(home[name].tolist() for name in match_keys)))
    teams, opponents = home["team_id"].tolist(), home["opponent_id"].tolist()
    records = []
    for stream, outcomes in comparisons.items():
        states, games_seen, latest_contributing = {}, defaultdict(int), {}
        valid = finished & valid_time & np.isfinite(outcomes)
        events = sorted(np.flatnonzero(valid), key=lambda i: (release.iloc[i], kicks.iloc[i], repr(match_ids[i])))
        batches = {time: list(batch) for time, batch in groupby(events, key=lambda i: release.iloc[i])}
        times = sorted(set(batches) | (set(transition_replay.events) if transition_replay is not None else set()))
        for time in times:
            if transition_replay is not None:
                records.extend(transition_replay.apply(time, states, games_seen, latest_contributing, stream))
            batch = batches.get(time, ())
            games, latest = defaultdict(list), {}
            for i in batch:
                home_key, away_key = (*scopes[i], teams[i]), (*scopes[i], opponents[i])
                if home_key == away_key:
                    raise ValueError("A team cannot play itself.")
                for key, other, score in ((home_key, away_key, outcomes[i]), (away_key, home_key, 1 - outcomes[i])):
                    games[key].append((float(score), dict(states.get(other, initial))))
                    latest[key] = max(
                        latest.get(key, latest_contributing.get(key, kicks.iloc[i])),
                        latest_contributing.get(other, kicks.iloc[i]),
                        kicks.iloc[i],
                    )
            # Commit only after every team has read the same pre-batch states.
            updated = {key: engine.update_period(dict(states.get(key, initial)), observations)
                       for key, observations in games.items()}
            for key, state in updated.items():
                if set(state) != set(initial) or not all(np.isfinite(x) for x in state.values()):
                    raise ValueError("Engine updates must retain finite numeric state fields.")
                games_seen[key] += len(games[key])
                records.append({**dict(zip(scope, key[:-1])), "team_id": key[-1],
                                "stream": stream, "recorded_at": time,
                                "latest_kickoff_at": latest[key], "games_seen": games_seen[key],
                                **({"snapshot_order": 1, "snapshot_kind": "result"} if transition_replay is not None else {}), **state})
            states.update(updated)
            latest_contributing.update(latest)
    columns = [*scope, "team_id", "stream", "recorded_at", "latest_kickoff_at", "games_seen",
               *(["snapshot_order", "snapshot_kind"] if transition_replay is not None else []), *initial]
    snapshots = pd.DataFrame.from_records(records, columns=columns)
    for name in ("recorded_at", "latest_kickoff_at"):
        snapshots[name] = pd.to_datetime(snapshots[name], utc=True)
    metadata = {"engine": repr(engine), "stat": repr(stat), "higher_is_better": higher_is_better,
                "periods": "event_batches; no calendar inactivity inflation",
                "availability": "kickoff_proxy" if available_at is None else "explicit"}
    if transition is not None:
        metadata["transition"] = repr(transition)
        metadata["transition_boundary"] = "explicit_or_first_prediction; frozen prior state; transition before equal-time results"
    return RatingRun(snapshots, initial, scope, tuple(comparisons), metadata)
