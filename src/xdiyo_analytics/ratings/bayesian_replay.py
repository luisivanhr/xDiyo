"""Score-aware chronological state producer behind the RatingRun adapter."""

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import replace
import math

from .bayesian_core import (
    GammaState, TeamState, initial_home, initial_team, predict_score_summary,
    team_summary, update_matches, score_log_probability,
)


def _team_payload(state):
    return [state.attack.shape, state.attack.rate, state.defence.shape, state.defence.rate]


def _team_restore(values):
    return TeamState(GammaState(*values[:2]), GammaState(*values[2:]))


def _maximum(*values):
    import pandas as pd
    valid = [value for value in values if value is not None and pd.notna(value)]
    return max(valid) if valid else pd.NaT


def _events(history, available_at):
    import numpy as np
    import pandas as pd
    from ..features.history import aligned_times
    from .bayesian import _identifier
    required = {"competition_id", "season_id", "event_id", "team_id", "opponent_id",
                "side", "status", "kickoff_at", "goals_for", "goals_against"}
    if required - set(history):
        raise KeyError(f"Bayesian history is missing columns: {sorted(required - set(history))}")
    keys = [name for name in ("source_league", "source_season", "competition_id", "season_id", "event_id") if name in history]
    identities = list(dict.fromkeys([*keys, "team_id", "opponent_id"]))
    if history[identities].isna().any().any() or history.duplicated(keys + ["side"]).any():
        raise ValueError("Bayesian fixtures need unique nonmissing match/side identities.")
    if not history.side.isin(["home", "away"]).all() or not history.groupby(keys, dropna=False).size().eq(2).all():
        raise ValueError("Bayesian replay requires both perspectives of each fixture.")
    for name in ("competition_id", "season_id", "team_id", "opponent_id", "event_id"):
        for value in history[name].unique():
            _identifier(value)
    kickoff = aligned_times(history, None, default="kickoff_at")
    available = aligned_times(history, available_at, default="kickoff_at")
    if (available < kickoff).any():
        raise ValueError("Results cannot be available before kickoff.")
    numbered = history.reset_index(drop=True)
    events = []
    for _, positions in numbered.groupby(keys, sort=False, dropna=False).indices.items():
        rows = numbered.iloc[positions]
        hi = int(positions[np.flatnonzero(rows.side.eq("home").to_numpy())[0]])
        ai = int(positions[np.flatnonzero(rows.side.eq("away").to_numpy())[0]])
        h, a = numbered.iloc[hi], numbered.iloc[ai]
        if h.team_id == h.opponent_id or h.team_id != a.opponent_id or h.opponent_id != a.team_id:
            raise ValueError("Bayesian match perspectives disagree on the teams.")
        for series, label in ((kickoff, "kickoff"), (available, "result availability")):
            left, right = series.iloc[hi], series.iloc[ai]
            if (pd.isna(left) != pd.isna(right)) or (pd.notna(left) and left != right):
                raise ValueError(f"Both match perspectives must agree on {label}.")
        if h.status != a.status:
            raise ValueError("Both match perspectives must agree on status.")
        if h.status != "finished" or pd.isna(kickoff.iloc[hi]) or pd.isna(available.iloc[hi]):
            continue
        if "is_awarded" in rows and rows.is_awarded.fillna(False).astype(bool).any():
            continue
        scores = [h.goals_for, h.goals_against, a.goals_against, a.goals_for]
        if any(pd.isna(value) for value in scores):
            continue
        scores = np.asarray(scores, dtype=float)
        if not np.isfinite(scores).all() or (scores < 0).any() or (scores != np.floor(scores)).any():
            raise ValueError("Finished Bayesian scores must be finite nonnegative integers.")
        if not np.array_equal(scores[:2], scores[2:]):
            raise ValueError("Both match perspectives must agree on the paired scores.")
        events.append({**{name: h[name] for name in keys}, "home_id": _identifier(h.team_id),
                       "away_id": _identifier(h.opponent_id), "home_goals": int(scores[0]),
                       "away_goals": int(scores[1]), "kickoff_at": kickoff.iloc[hi],
                       "available_at": available.iloc[hi], "home_position": hi})
    return events, keys, available


def _bridge(state, source, destination, old_parameters, parameters, config):
    for origin, target, value in config.bridge_gaps:
        if (origin, target) == (source, destination):
            gap = value
        elif (origin, target) == (destination, source):
            gap = -value
        else:
            continue
        # Baseline ratios explicitly align units before applying the gap.
        attack_scale = parameters.attack_mean / old_parameters.attack_mean * math.exp(gap)
        defence_scale = parameters.defence_mean / old_parameters.defence_mean * math.exp(-gap)
        return TeamState(state.attack.scale(attack_scale), state.defence.scale(defence_scale)).discount(config.bridge_discount)
    return None


def _entry_key(record):
    from .bayesian import _identifier
    return tuple(_identifier(record[name]) for name in ("competition_id", "season_id", "team_id"))


def _event_key(record):
    return tuple(record[name] for name in ("competition_id", "season_id", "event_id"))


def _journal_entry(record):
    import pandas as pd
    from .bayesian import _identifier
    result = {name: _identifier(record[name]) for name in ("competition_id", "season_id", "team_id")}
    result.update(movement=record["movement"], entry_at=record["entry_at"].isoformat(),
                  anchor_explicit=record.get("anchor_explicit", False))
    for name in ("previous_competition_id", "previous_season_id"):
        value = record.get(name)
        result[name] = None if value is None or pd.isna(value) else _identifier(value)
    return result


def _journal_event(record):
    from .bayesian import _identifier
    result = {name: _identifier(record[name]) for name in
              ("competition_id", "season_id", "event_id", "home_id", "away_id")}
    result.update(home_goals=int(record["home_goals"]), away_goals=int(record["away_goals"]),
                  kickoff_at=record["kickoff_at"].isoformat(), available_at=record["available_at"].isoformat())
    return result


def _event_history(events, score_basis):
    """Reconstruct only validated observations, with no source-history targets."""
    import pandas as pd
    rows = []
    for event in events:
        common = {name: event[name] for name in ("competition_id", "season_id", "event_id", "kickoff_at")}
        for side, team, other, own, against in (
                ("home", event["home_id"], event["away_id"], event["home_goals"], event["away_goals"]),
                ("away", event["away_id"], event["home_id"], event["away_goals"], event["home_goals"])):
            rows.append(dict(common, side=side, team_id=team, opponent_id=other,
                             goals_for=own, goals_against=against, status="finished"))
    columns = ("competition_id", "season_id", "event_id", "kickoff_at", "side", "team_id",
               "opponent_id", "goals_for", "goals_against", "status")
    frame = pd.DataFrame(rows, columns=columns)
    frame["kickoff_at"] = pd.to_datetime(frame.kickoff_at, utc=True)
    frame.attrs["score_basis"] = score_basis
    return frame


def _entry_usage(run, entries, through):
    """Actual prior mechanisms, identified once per entry and mechanism."""
    import pandas as pd
    if through is None or pd.isna(through):
        return set()
    lookup = {(r["competition_id"], r["team_id"], r["entry_at"]): _entry_key(r) for r in entries}
    usage = set()
    for row in run.snapshots.itertuples(index=False):
        if row.snapshot_kind in ("initial", "retained", "bridge", "mirrored") and row.recorded_at <= through:
            identity = lookup.get((row.competition_id, row.team_id, row.recorded_at))
            if identity is not None:
                usage.add((*identity, row.snapshot_kind))
    return usage


def _needs_refilter(events, entries, checkpoint):
    """Check shared league clocks and consumed transfer boundaries, not just teams."""
    import pandas as pd
    latest = {}
    if checkpoint is not None:
        for event in checkpoint.checkpoint["event_journal"]:
            key, time = event["competition_id"], pd.Timestamp(event["kickoff_at"])
            latest[key] = max(latest.get(key, time), time)
        frontier = checkpoint.checkpoint["processed_through"]
        if frontier is not None:
            known_entries = {tuple(v) for v in checkpoint.checkpoint["entries"]}
            if any(_entry_key(r) not in known_entries and r["entry_at"] <= pd.Timestamp(frontier) for r in entries):
                return True
            old_anchors = {_entry_key(r): pd.Timestamp(r["entry_at"]) for r in checkpoint.checkpoint["entry_journal"]}
            if any(_entry_key(r) in old_anchors and r["entry_at"] != old_anchors[_entry_key(r)] for r in entries):
                return True
    batches = defaultdict(list)
    for event in events:
        batches[(event["available_at"], event["competition_id"])].append(event)
        for entry in entries:
            if (entry["competition_id"] == event["competition_id"]
                    or entry.get("previous_competition_id") == event["competition_id"]):
                # A score that became known at an entry boundary is handled by
                # the normal kernel before that entry. Later releases need replay.
                if event["kickoff_at"] < entry["entry_at"] < event["available_at"]:
                    return True
    for (release, competition), batch in sorted(batches.items(), key=lambda item: (item[0][0], repr(item[0][1]))):
        kicks = {event["kickoff_at"] for event in batch}
        if len(kicks) != 1:
            return True
        kickoff = next(iter(kicks))
        if competition in latest and kickoff <= latest[competition]:
            return True
        latest[competition] = kickoff
    return False


def replay(history, *, model, available_at=None, team_seasons=None, season_starts=None,
           cutoffs=None, checkpoint=None):
    """Kickoff-ordered filtering, published only when observations are available.

    Normal chronological releases use the incremental kernel. A late observation
    re-filters the retained observation/entry journal in kickoff order at each
    affected knowledge boundary; previously published snapshots stay unchanged.
    """
    import pandas as pd
    from ..features.transitions import TransitionContext
    from .bayesian import BayesianRatingRun
    if model.config.score_basis == "regulation" and history.attrs.get("score_basis") != "regulation":
        raise ValueError("Regulation mode requires caller-verified history.attrs['score_basis']='regulation'.")
    events, keys, available = _events(history, available_at)
    if team_seasons is not None and not isinstance(team_seasons, (pd.DataFrame, Mapping)):
        team_seasons = list(team_seasons)
    context = TransitionContext(history, cutoffs=cutoffs, available_at=available,
                                team_seasons=team_seasons, season_starts=season_starts)
    incoming_entries = [record for record in context.records.values() if pd.notna(record["entry_at"])]
    explicit_starts = (set(season_starts[["competition_id", "season_id"]].itertuples(index=False, name=None))
                       if isinstance(season_starts, pd.DataFrame) else set(season_starts or ()))
    for record in incoming_entries:
        record["anchor_explicit"] = (record["competition_id"], record["season_id"]) in explicit_starts
    retained_events, retained_entries, frontier = [], {}, None
    usage = set()
    if checkpoint is not None:
        if not isinstance(checkpoint, BayesianRatingRun) or checkpoint.model != model:
            raise ValueError("A checkpoint must belong to the same frozen Bayesian model.")
        payload = checkpoint.checkpoint
        if payload.get("chronology_schema") != 1:
            if not events:
                # Legacy artifacts can still serve the frozen adapter's
                # outcome-free season projection. They cannot assimilate any
                # new observations without a complete historical journal.
                return _replay_in_order(history, model=model, available_at=available_at,
                                        team_seasons=team_seasons, season_starts=season_starts,
                                        cutoffs=cutoffs, checkpoint=checkpoint)
            raise ValueError("This checkpoint lacks a kickoff replay journal; rebuild full history before updating it.")
        frontier = pd.Timestamp(payload["processed_through"]) if payload["processed_through"] else None
        for item in payload["event_journal"]:
            retained_events.append(dict(item, kickoff_at=pd.Timestamp(item["kickoff_at"]),
                                        available_at=pd.Timestamp(item["available_at"])))
        for item in payload["entry_journal"]:
            record = dict(item, entry_at=pd.Timestamp(item["entry_at"]))
            retained_entries[_entry_key(record)] = record
        usage = {tuple(item) for item in payload.get("entry_usage", ())}
        seen = {_event_key(record) for record in retained_events}
        for event in events:
            if _event_key(event) in seen or (frontier is not None and event["available_at"] <= frontier):
                raise ValueError("Incremental replay requires new events released strictly after the checkpoint; replay full history for corrections/equal-time batches.")
        explicit = {}
        if team_seasons is not None:
            rows = (pd.DataFrame(team_seasons, dtype=object).to_dict("records")
                    if isinstance(team_seasons, (pd.DataFrame, Mapping)) else team_seasons)
            explicit = {_entry_key(row): row for row in rows}
        known_teams = {record["team_id"] for record in retained_entries.values()}
        for record in incoming_entries:
            identity = _entry_key(record)
            if identity not in retained_entries and record["team_id"] in known_teams:
                required = {"movement", "previous_competition_id", "previous_season_id"}
                if required - set(explicit.get(identity, {})):
                    raise ValueError("Resuming a known team in a new season/league requires explicit movement and both predecessor IDs in team_seasons; use explicit null predecessors to request a fresh prior.")
    # Preserve movement declarations from complete historical context. Inferred
    # season anchors can move earlier when an older fixture is first supplied;
    # caller-declared anchors remain authoritative across incremental slices.
    merged = {_entry_key(record): record for record in incoming_entries}
    merged.update(retained_entries)
    anchors = defaultdict(list)
    for record in [*incoming_entries, *retained_entries.values()]:
        anchors[(record["competition_id"], record["season_id"])].append(record)
    for season, candidates in anchors.items():
        fixed = {r["entry_at"] for r in candidates if r.get("anchor_explicit", False)}
        if len(fixed) > 1:
            raise ValueError("A declared season anchor changed; rebuild full history for corrections.")
        anchor = next(iter(fixed)) if fixed else min(r["entry_at"] for r in candidates)
        for key, record in merged.items():
            if key[:2] == season:
                merged[key] = dict(record, entry_at=anchor, anchor_explicit=bool(fixed))
    entries = sorted(merged.values(), key=lambda r: (r["entry_at"], repr(_entry_key(r))))
    if checkpoint is not None and frontier is not None and not events:
        if any(r["entry_at"] <= frontier and (_entry_key(r) not in retained_entries
               or r["entry_at"] != retained_entries[_entry_key(r)]["entry_at"]) for r in entries):
            raise ValueError("A newly supplied historical entry needs a new result release; otherwise rebuild full history. Forecast-only updates cannot silently revise past membership.")
    all_events = [*retained_events, *events]
    data_frontier = _maximum(frontier, *(event["available_at"] for event in events))
    if _needs_refilter(events, entries, checkpoint):
        run, used = _refilter_versions(history, model, events, all_events, entries, frontier,
                                      data_frontier, checkpoint)
        usage.update(used)
    else:
        run = _replay_in_order(history, model=model, available_at=available_at,
                               team_seasons=team_seasons, season_starts=season_starts,
                               cutoffs=cutoffs, checkpoint=checkpoint, _entries=entries,
                               _predict=False)
        usage.update(_entry_usage(run, entries, data_frontier))
    durable_entries = [r for r in entries if pd.notna(data_frontier) and r["entry_at"] <= data_frontier]
    run.checkpoint.update(
        chronology_schema=1,
        event_journal=[_journal_event(r) for r in sorted(all_events, key=lambda r: (r["kickoff_at"], repr(_event_key(r))))],
        entry_journal=[_journal_entry(r) for r in durable_entries],
        entry_usage=[list(item) for item in sorted(usage, key=repr)],
    )
    run.metadata.update(chronology="kickoff_order_with_availability_versions",
                        availability="kickoff_proxy" if available_at is None else "explicit",
                        effective_entry_counts=dict(Counter(item[-1] for item in usage)))
    # Historical diagnostics are evaluated against knowledge-time versions, never
    # the counterfactual historical forecasts made by a chronological rebuild.
    _populate_predictions(run, history, events, keys, cutoffs,
                          checkpoint.predictions if checkpoint is not None else None)
    return run


def _refilter_versions(history, model, events, all_events, entries, frontier, data_frontier, checkpoint):
    """Publish corrected present states without rewriting the past information set."""
    import pandas as pd
    from .bayesian import BayesianRatingRun, TEAM_FIELDS
    snapshots = (checkpoint.snapshots.loc[checkpoint.snapshots.recorded_at.le(frontier)].to_dict("records")
                 if checkpoint is not None and frontier is not None else [])
    leagues = (checkpoint.league_snapshots.loc[checkpoint.league_snapshots.recorded_at.le(frontier)].to_dict("records")
               if checkpoint is not None and frontier is not None else [])
    diagnostics = list(checkpoint.metadata.get("diagnostics", ())) if checkpoint is not None else []
    usage = set()
    previous_teams, previous_leagues = {}, {}
    for row in snapshots:
        previous_teams[(row["competition_id"], row["team_id"])] = row
    for row in leagues:
        previous_leagues[row["competition_id"]] = row
    # Start from the same empty durable payload as the ordinary kernel when
    # there are no observations; forecast-only entries remain nondurable.
    empty = _replay_in_order(_event_history([], model.config.score_basis), model=model,
                             _entries=[], _predict=False)
    payload = dict(checkpoint.checkpoint) if checkpoint is not None else empty.checkpoint
    times = sorted({*(event["available_at"] for event in events),
                    *(r["entry_at"] for r in entries if frontier is None or r["entry_at"] > frontier)})
    for time in times:
        current_entries = [r for r in entries if r["entry_at"] <= time]
        # Scores kicking off exactly now remain unavailable to any forecast at
        # this cutoff, even with the documented kickoff-availability proxy.
        for order, include_equal in ((0, False), (2, True)):
            if include_equal and not any(r["available_at"] <= time and r["kickoff_at"] == time for r in all_events):
                continue
            eligible = [r for r in all_events if r["available_at"] <= time
                        and (r["kickoff_at"] <= time if include_equal else r["kickoff_at"] < time)]
            canonical = _replay_in_order(_event_history(eligible, model.config.score_basis),
                                         model=model, _entries=current_entries,
                                         _capture_at=time, _predict=False)
            if pd.notna(data_frontier) and time <= data_frontier:
                usage.update(_entry_usage(canonical, current_entries, time))
            diagnostics.append({"recorded_at": time.isoformat(), "method": "kickoff_refilter",
                                "match_count": len(eligible), "converged": True,
                                "iterations": sum(int(d["iterations"]) for d in canonical.metadata["diagnostics"])})
            # Every dependent state is republished if it changed, including
            # teams in other competitions reached through a later bridge.
            for _, group in canonical.snapshots.groupby(["competition_id", "team_id"], sort=False):
                record = group.iloc[-1].to_dict()
                key = (record["competition_id"], record["team_id"])
                old = previous_teams.get(key)
                compare = (*TEAM_FIELDS, "games_seen", "latest_kickoff_at")
                same = old is not None and all((pd.isna(record[n]) and pd.isna(old[n])) or record[n] == old[n] for n in compare)
                if same:
                    continue
                record.update(recorded_at=time, snapshot_order=order)
                snapshots.append(record)
                previous_teams[key] = record
            for competition, group in canonical.league_snapshots.groupby("competition_id", sort=False):
                record = group.iloc[-1].to_dict()
                old = previous_leagues.get(competition)
                compare = ("home_shape", "home_rate", "latest_kickoff_at")
                same = old is not None and all((pd.isna(record[n]) and pd.isna(old[n])) or record[n] == old[n] for n in compare)
                if same:
                    continue
                record.update(recorded_at=time, snapshot_order=order)
                leagues.append(record)
                previous_leagues[competition] = record
            if pd.notna(data_frontier) and time == data_frontier:
                payload = canonical.checkpoint
    columns = ["competition_id", "team_id", "stream", "recorded_at", "latest_kickoff_at", "games_seen",
               "snapshot_order", "snapshot_kind", *TEAM_FIELDS]
    table = pd.DataFrame(snapshots, columns=columns)
    league_table = pd.DataFrame(leagues, columns=["competition_id", "recorded_at", "latest_kickoff_at",
                                                 "snapshot_order", "home_shape", "home_rate"])
    for frame in (table, league_table):
        for name in ("recorded_at", "latest_kickoff_at"):
            frame[name] = pd.to_datetime(frame[name], utc=True)
    metadata = {"engine": "Ridall Gamma-mixed Poisson", "score_basis": model.config.score_basis,
                "clock": model.config.clock, "method": model.config.method,
                "prediction_uncertainty": "state_means_with_integrated_match_effect", "diagnostics": diagnostics}
    return BayesianRatingRun(table, team_summary(initial_team(model.parameters)), ("competition_id",), ("score",),
                             metadata, model, league_table, pd.DataFrame(), payload), usage


def _populate_predictions(run, history, events, keys, cutoffs, previous_predictions):
    import pandas as pd
    unchecked = replace(run, model=replace(run.model, training_cutoff=None))
    records = []
    if events:
        queries = history.iloc[[event["home_position"] for event in events]].copy()
        prediction_times = None
        if cutoffs is not None:
            from ..features.history import aligned_times
            aligned = aligned_times(history, cutoffs, default="kickoff_at")
            prediction_times = aligned.iloc[[event["home_position"] for event in events]]
        predictions = unchecked.fixture_features(queries, cutoffs=prediction_times,
                                                 fields=("expected_home_goals", "expected_away_goals", "p_home_win", "p_draw", "p_away_win"))
        for index, event in enumerate(events):
            values = predictions.iloc[index].to_dict()
            parameters = run.model.parameters_for(event["competition_id"])
            kappa = parameters.kappa if run.model.config.bivariate else None
            records.append({**{name: event[name] for name in (*keys, "home_id", "away_id", "home_goals", "away_goals", "kickoff_at", "available_at")},
                            **values, "kappa": kappa,
                            "score_log_probability": score_log_probability(event["home_goals"], event["away_goals"],
                                                                             values["expected_home_goals"], values["expected_away_goals"], kappa)})
    run.predictions = pd.DataFrame(records)
    if previous_predictions is not None:
        run.predictions = pd.concat([previous_predictions, run.predictions], ignore_index=True)


def _replay_in_order(history, *, model, available_at=None, team_seasons=None, season_starts=None,
                     cutoffs=None, checkpoint=None, _entries=None, _capture_at=None,
                     _predict=True):
    import numpy as np
    import pandas as pd
    from ..features.transitions import TransitionContext
    from .bayesian import BayesianRatingRun, TEAM_FIELDS, _identifier
    config = model.config
    if config.score_basis == "regulation" and history.attrs.get("score_basis") != "regulation":
        raise ValueError("Regulation mode requires caller-verified history.attrs['score_basis']='regulation'.")
    events, keys, available = _events(history, available_at)
    if team_seasons is not None and not isinstance(team_seasons, (pd.DataFrame, Mapping)):
        team_seasons = list(team_seasons)
    context = (TransitionContext(history, cutoffs=cutoffs, available_at=available,
                                 team_seasons=team_seasons, season_starts=season_starts)
               if _entries is None else None)
    states, home_states, team_season, home_season = {}, {}, {}, {}
    latest, home_latest, last_kick = {}, {}, {}
    counts, processed_entries = Counter(), set()
    snapshots, leagues, diagnostics = [], [], []
    frontier = None
    seen_events, closed_sources = set(), set()
    previous_predictions = None
    if checkpoint is not None:
        if not isinstance(checkpoint, BayesianRatingRun) or checkpoint.model != model:
            raise ValueError("A checkpoint must belong to the same frozen Bayesian model.")
        payload = checkpoint.checkpoint
        frontier = pd.to_datetime(payload["processed_through"], utc=True) if payload["processed_through"] else None
        for row in payload["teams"]:
            key = (row["competition_id"], row["team_id"])
            states[key], team_season[key] = _team_restore(row["state"]), row["season_id"]
            latest[key] = pd.to_datetime(row["latest_kickoff_at"], utc=True)
            last_kick[key] = pd.to_datetime(row["last_kickoff_at"], utc=True)
            counts[key] = row["games_seen"]
        for row in payload["leagues"]:
            competition = row["competition_id"]
            home_states[competition] = GammaState(*row["state"])
            home_season[competition] = row["season_id"]
            home_latest[competition] = pd.to_datetime(row["latest_kickoff_at"], utc=True)
        processed_entries = {tuple(entry) for entry in payload["entries"]}
        seen_events = {tuple(entry) for entry in payload["event_ids"]}
        closed_sources = {tuple(entry) for entry in payload.get("closed_sources", ())}
        # Forecast-only season entries never advance a resumable checkpoint.
        # Rebuild those provisional entries from the newly supplied history.
        snapshots = (checkpoint.snapshots.loc[checkpoint.snapshots.recorded_at.le(frontier)].to_dict("records")
                     if frontier is not None else [])
        leagues = (checkpoint.league_snapshots.loc[checkpoint.league_snapshots.recorded_at.le(frontier)].to_dict("records")
                   if frontier is not None else [])
        previous_predictions = checkpoint.predictions
        for event in events:
            identity = (event["competition_id"], event["season_id"], event["event_id"])
            if identity in seen_events or (frontier is not None and event["available_at"] <= frontier):
                raise ValueError("Incremental replay requires new events released strictly after the checkpoint; replay full history for corrections/equal-time batches.")

    explicit_entries = {}
    if team_seasons is not None:
        supplied = (pd.DataFrame(team_seasons, dtype=object).to_dict("records")
                    if isinstance(team_seasons, (pd.DataFrame, Mapping)) else list(team_seasons))
        explicit_entries = {tuple(record[name] for name in ("competition_id", "season_id", "team_id")): record
                            for record in supplied}
    entry_events = defaultdict(list)
    for record in (context.records.values() if _entries is None else _entries):
        identity = tuple(_identifier(record[name]) for name in ("competition_id", "season_id", "team_id"))
        if identity in processed_entries or pd.isna(record["entry_at"]):
            continue
        time = record["entry_at"]
        key = (identity[0], identity[2])
        if (checkpoint is not None and any(team == identity[2] for _, team in states)
                and (key not in states or team_season[key] != identity[1])):
            explicit = explicit_entries.get(identity, {})
            required = {"movement", "previous_competition_id", "previous_season_id"}
            if required - set(explicit):
                raise ValueError("Resuming a known team in a new season/league requires explicit movement and both predecessor IDs in team_seasons; use explicit null predecessors to request a fresh prior.")
        # New teams in a resumed current season enter when first supplied. A
        # known season must not be discounted twice because a subset starts later.
        if frontier is not None and time <= frontier:
            if key in states and team_season[key] == identity[1]:
                continue
            raise ValueError("A new season/team entry precedes the checkpoint; rebuild full history.")
        entry_events[time].append(record)
    releases = defaultdict(list)
    for event in events:
        releases[event["available_at"]].append(event)
    times = sorted(set(entry_events) | set(releases))
    data_frontier = _maximum(frontier, *(event["available_at"] for event in events))
    if _capture_at is not None:
        data_frontier = _capture_at
        times = sorted({*times, _capture_at})

    def date(value):
        return None if value is None or pd.isna(value) else value.isoformat()

    def capture(boundary):
        return {
            "processed_through": date(boundary),
            "teams": [{"competition_id": _identifier(key[0]), "team_id": _identifier(key[1]),
                       "season_id": _identifier(team_season[key]), "state": _team_payload(state),
                       "latest_kickoff_at": date(latest.get(key)), "last_kickoff_at": date(last_kick.get(key)),
                       "games_seen": counts[key]} for key, state in sorted(states.items(), key=lambda item: repr(item[0]))],
            "leagues": [{"competition_id": _identifier(key), "season_id": _identifier(home_season[key]),
                         "state": [state.shape, state.rate], "latest_kickoff_at": date(home_latest.get(key))}
                        for key, state in sorted(home_states.items(), key=lambda item: repr(item[0]))],
            "entries": [list(map(_identifier, entry)) for entry in sorted(processed_entries, key=repr)],
            "event_ids": [list(map(_identifier, entry)) for entry in sorted(seen_events, key=repr)],
            "closed_sources": [list(map(_identifier, entry)) for entry in sorted(closed_sources, key=repr)],
        }

    payload = capture(frontier)

    def record_team(key, time, order, kind):
        snapshots.append({"competition_id": key[0], "team_id": key[1], "stream": "score",
                          "recorded_at": time, "latest_kickoff_at": latest.get(key, pd.NaT),
                          "games_seen": counts[key], "snapshot_order": order, "snapshot_kind": kind,
                          **team_summary(states[key])})

    def record_home(competition, time, order):
        state = home_states[competition]
        leagues.append({"competition_id": competition, "recorded_at": time,
                        "latest_kickoff_at": home_latest.get(competition, pd.NaT),
                         "snapshot_order": order, "home_shape": state.shape, "home_rate": state.rate})

    def process_batch(batch, time, order):
        by_league = defaultdict(list)
        for event in batch:
            competition, season = event["competition_id"], event["season_id"]
            for team in (event["home_id"], event["away_id"]):
                key = (competition, team)
                if ((competition, season, team) in closed_sources
                        or (key in team_season and team_season[key] != season)
                        or (competition in home_season and home_season[competition] != season)):
                    raise ValueError("A previous-season result arrived after a season transition or state bridge; out-of-sequence seasonal observations are unsupported.")
                if key not in states or competition not in home_states:
                    raise ValueError("Every result must follow its initialized season entry.")
            by_league[competition].append(event)
        for competition, games in sorted(by_league.items(), key=lambda item: repr(item[0])):
            parameters = model.parameters_for(competition)
            games.sort(key=lambda event: repr((event["season_id"], event["event_id"])))
            appearances = Counter(team for event in games for team in (event["home_id"], event["away_id"]))
            involved = {}
            contributing = [home_latest.get(competition, pd.NaT)]
            for team, number in appearances.items():
                key = (competition, team)
                kicks = [event["kickoff_at"] for event in games if team in (event["home_id"], event["away_id"])]
                previous = last_kick.get(key, pd.NaT)
                exponent = number if config.clock == "team_match" else (
                    max(0.0, (max(kicks) - previous).total_seconds() / 86400 / config.time_unit_days)
                    if pd.notna(previous) else 0.0)
                involved[team] = states[key].discount(parameters.team_discount ** exponent)
                contributing.extend([latest.get(key, pd.NaT), *kicks])
            home = home_states[competition].discount(parameters.home_discount ** len(games))
            observations = [(event["home_id"], event["away_id"], event["home_goals"], event["away_goals"]) for event in games]
            updated, hga, diagnostic = update_matches(involved, home, observations, parameters,
                                                     bivariate=config.bivariate, method=config.method,
                                                     tolerance=config.tolerance, max_iterations=config.max_iterations)
            if not diagnostic["converged"]:
                raise RuntimeError(f"Bayesian {config.method} did not converge at {time}: {diagnostic}")
            diagnostic = {**diagnostic, "competition_id": _identifier(competition), "recorded_at": time.isoformat()}
            diagnostics.append(diagnostic)
            newest = _maximum(*contributing)
            home_states[competition], home_latest[competition] = hga, newest
            for team in appearances:
                key = (competition, team)
                states[key], latest[key] = updated[team], newest
                counts[key] += appearances[team]
                last_kick[key] = _maximum(last_kick.get(key), *(event["kickoff_at"] for event in games if team in (event["home_id"], event["away_id"])))
                record_team(key, time, order, "result")
            record_home(competition, time, order)
            seen_events.update((event["competition_id"], event["season_id"], event["event_id"]) for event in games)

    for time in times:
        real_releases = [event for event in releases.get(time, ()) if event["kickoff_at"] < time]
        before_entries, after_entries = [], []
        # Earlier games available exactly at the entry boundary belong in the
        # source posterior. Keep each league's real-release batch simultaneous.
        # A league with any uninitialized participant waits until entries exist.
        real_by_league = defaultdict(list)
        for event in real_releases:
            real_by_league[event["competition_id"]].append(event)
        for competition, games in real_by_league.items():
            initialized = (competition in home_states and all(
                (competition, team) in states and team_season[(competition, team)] == event["season_id"]
                and home_season[competition] == event["season_id"]
                for event in games for team in (event["home_id"], event["away_id"])))
            (before_entries if initialized else after_entries).extend(games)
        process_batch(before_entries, time, -1)
        # Season-entry reads use a frozen source population, including when
        # several leagues exchange teams at the same instant.
        pending = []
        touched_home = set()
        for record in sorted(entry_events.get(time, ()), key=lambda r: repr((r["competition_id"], r["team_id"]))):
            competition, season, team = (_identifier(record[name]) for name in ("competition_id", "season_id", "team_id"))
            parameters = model.parameters_for(competition)
            key = (competition, team)
            movement = record["movement"]
            old_competition, old_season = record.get("previous_competition_id"), record.get("previous_season_id")
            old_key = (old_competition, team)
            state, contributing, reason = None, pd.NaT, "initial"
            if key in states and team_season[key] == season:
                state, contributing, reason = states[key], latest.get(key), "continued"
            elif movement == "retained" and old_key in states and team_season[old_key] == old_season:
                state = states[old_key].discount(parameters.season_discount)
                contributing, reason = latest.get(old_key), "retained"
            elif config.transition == "bridge" and movement in ("promoted", "relegated") and old_key in states and team_season[old_key] == old_season:
                state = _bridge(states[old_key], old_competition, competition,
                                model.parameters_for(old_competition), parameters, config)
                if state is not None:
                    contributing, reason = latest.get(old_key), "bridge"
            if state is None:
                state = initial_team(parameters, movement if movement in ("promoted", "relegated") else "unknown")
                reason = "mirrored" if movement in ("promoted", "relegated") else "initial"
            pending.append((key, season, state, contributing, reason, old_key,
                            counts[old_key], last_kick.get(old_key, pd.NaT)))
            if reason in ("retained", "bridge"):
                source_key = next(existing for existing in states if existing == old_key)
                closed_sources.add((source_key[0], team_season[source_key], team))
            processed_entries.add((competition, season, team))
            if competition not in touched_home:
                if competition not in home_states:
                    home_states[competition] = initial_home(parameters)
                elif home_season[competition] != season:
                    home_states[competition] = home_states[competition].discount(parameters.home_season_discount)
                home_season[competition] = season
                touched_home.add(competition)
        for key, season, state, contributing, reason, old_key, source_count, source_last_kick in pending:
            states[key], team_season[key], latest[key] = state, season, contributing
            if reason == "bridge":
                counts[key] = source_count
                last_kick[key] = source_last_kick
            elif reason in ("mirrored", "initial"):
                counts[key] = 0
                last_kick[key] = pd.NaT
            record_team(key, time, 0, reason)
        for competition in sorted(touched_home, key=repr):
            record_home(competition, time, 0)

        process_batch(after_entries, time, 1)
        process_batch([event for event in releases.get(time, ()) if event["kickoff_at"] == time], time, 2)
        if pd.notna(data_frontier) and time == data_frontier:
            payload = capture(time)

    columns = ["competition_id", "team_id", "stream", "recorded_at", "latest_kickoff_at", "games_seen",
               "snapshot_order", "snapshot_kind", *TEAM_FIELDS]
    table = pd.DataFrame(snapshots, columns=columns)
    league_table = pd.DataFrame(leagues, columns=["competition_id", "recorded_at", "latest_kickoff_at", "snapshot_order", "home_shape", "home_rate"])
    for frame in (table, league_table):
        for name in ("recorded_at", "latest_kickoff_at"):
            frame[name] = pd.to_datetime(frame[name], utc=True)

    metadata = {"engine": "Ridall Gamma-mixed Poisson", "availability": "kickoff_proxy" if available_at is None else "explicit",
                "score_basis": config.score_basis, "clock": config.clock, "method": config.method,
                "prediction_uncertainty": "state_means_with_integrated_match_effect",
                "diagnostics": [*(checkpoint.metadata.get("diagnostics", []) if checkpoint else []), *diagnostics]}
    run = BayesianRatingRun(table, team_summary(initial_team(model.parameters)), ("competition_id",), ("score",),
                            metadata, model, league_table, pd.DataFrame(), payload)
    if _predict:
        _populate_predictions(run, history, events, keys, cutoffs, previous_predictions)
    return run


def fixture_features(run, history, times, fields):
    import numpy as np
    import pandas as pd
    result = pd.DataFrame(np.nan, index=history.index, columns=fields)
    if history.empty:
        return result
    required = {"competition_id", "team_id", "opponent_id", "side", "kickoff_at"}
    if required - set(history) or not history.side.isin(["home", "away"]).all():
        raise ValueError("Fixture queries need competition, teams, home/away side and kickoff.")
    team_fields = ("attack_mean", "defence_vulnerability_mean", "strength_index")
    team_values = run.features(history, cutoffs=times, side="both", fields=team_fields)
    groups = {key: group.sort_values(["recorded_at", "snapshot_order"], kind="stable")
              for key, group in run.league_snapshots.groupby("competition_id", sort=False)}
    want_probability = any(name.startswith("p_") for name in fields)
    for position, (_, row) in enumerate(history.iterrows()):
        time = times.iloc[position]
        if pd.isna(time) or pd.isna(row.kickoff_at):
            continue
        parameters = run.model.parameters_for(row.competition_id)
        home_state = initial_home(parameters)
        league = groups.get(row.competition_id)
        if league is not None:
            eligible = league.recorded_at.le(time) & (league.latest_kickoff_at.isna() | league.latest_kickoff_at.lt(time))
            if eligible.any():
                last = league.loc[eligible].iloc[-1]
                home_state = GammaState(last.home_shape, last.home_rate)
        values = team_values.iloc[position]
        hrole, arole = ("team", "opponent") if row.side == "home" else ("opponent", "team")
        get = lambda role, name: float(values[f"score::{role}::{name}"])
        mean_home = get(hrole, "attack_mean") * get(arole, "defence_vulnerability_mean") * home_state.mean
        mean_away = get(arole, "attack_mean") * get(hrole, "defence_vulnerability_mean")
        summaries = {"expected_home_goals": mean_home, "expected_away_goals": mean_away,
                     "expected_total_goals": mean_home + mean_away, "expected_goal_difference": mean_home - mean_away,
                     "strength_difference": get(hrole, "strength_index") - get(arole, "strength_index"),
                     "home_advantage_mean": home_state.mean, "home_advantage_sd": home_state.sd}
        if want_probability:
            summaries.update(predict_score_summary(mean_home, mean_away, parameters.kappa if run.model.config.bivariate else None,
                                                   tail_tolerance=run.model.config.tail_tolerance,
                                                   max_total=run.model.config.max_total_goals))
        result.iloc[position] = [summaries[name] for name in fields]
    return result
