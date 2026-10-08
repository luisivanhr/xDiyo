"""Frozen global-prefix publication oracle from 6f9c2fa; test use only."""
from xdiyo_analytics.ratings.bayesian_replay import _replay_in_order, _event_history, _entry_usage
from xdiyo_analytics.ratings.bayesian_core import team_summary, initial_team
def _refilter_versions(history, model, events, all_events, entries, frontier, data_frontier, checkpoint):
    """Publish corrected present states without rewriting the past information set."""
    import pandas as pd
    from xdiyo_analytics.ratings.bayesian import BayesianRatingRun, TEAM_FIELDS
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
