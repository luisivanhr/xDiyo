"""Feature coverage is separate from successful endpoint retrieval."""
from collections import Counter


def assess_features(tables, profile):
    features, warnings = {}, []

    def feature(name, requested, status, **details):
        features[name] = {"requested": requested, "status": status if requested else "not_requested", **details}

    shots = tables.get("shots", [])
    with_xg = sum(row.get("xg") is not None for row in shots)
    feature("shot_xg", profile.include_shotmap, "present" if with_xg == len(shots) and shots else
            "partial" if with_xg else "not_supplied", shots=len(shots), shots_with_xg=with_xg)
    heatmaps = tables.get("heatmap_points", [])
    teams = Counter(str(row["team_id"]) for row in heatmaps)
    feature("team_heatmaps", profile.include_heatmaps, "present" if len(teams) == 2 else
            "partial" if teams else "not_supplied", points_by_team=dict(teams))
    lineups = tables.get("lineup_players", [])
    feature("lineups", profile.include_lineups, "present" if len({r["side"] for r in lineups}) == 2 else "partial" if lineups else "not_supplied",
            distinct_players=len({r["player_id"] for r in lineups}), observations=len({r["observed_at"] for r in lineups}))
    absences = tables.get("availability", [])
    feature("reported_absences", profile.include_injuries, "present" if absences else "none_reported_or_not_supplied",
            distinct_players=len({r["player_id"] for r in absences}),
            classifications=dict(Counter(r.get("classification") for r in absences)))
    if any(r.get("classification") == "unclassified_reported_absence" for r in absences):
        warnings.append("Absence reason codes are preserved but not interpreted as specific injuries or suspensions.")
    sequences = tables.get("goal_sequences", [])
    actions = tables.get("goal_actions", [])
    feature("goal_sequences", profile.include_goal_actions, "present" if sequences else "none_reported_or_not_supplied",
            sequences=len(sequences), actions=len(actions), action_types=dict(Counter(r.get("action_type") for r in actions)))
    feature("explicit_pass_recipients", profile.include_goal_actions,
            "present" if any(r.get("recipient_id") is not None for r in actions) else "not_supplied",
            explicit_edges=len(tables.get("passing_edges", [])))
    timing = tables.get("timing", [])
    feature("exact_elapsed_duration", profile.duration_mode != "off",
            "present" if any(r.get("elapsed_seconds") is not None for r in timing) else "not_supplied")
    feature("added_time_announcements", profile.duration_mode != "off",
            "present" if any(r.get("announced_added_minutes") is not None for r in timing) else "not_supplied",
            periods={r["period"]:r["announced_added_minutes"] for r in timing if r.get("announced_added_minutes") is not None})
    positions = tables.get("pregame", [])
    feature("pregame_positions", profile.include_pregame,
            "present" if len({r["side"] for r in positions if r.get("position") is not None}) == 2 else "not_supplied")
    feature("normalized_standings", profile.include_pregame, "not_verified",
            note="This collector preserves raw pregame positions. The legacy loader divides each side's positions by that side's maximum observed position in the loaded season; this single-match import cannot validate its season-wide denominator. Current standings cannot replace historical pregame positions.")
    feature("promotion_relegation_flags", True, "separate_season_processing",
            note="Derived from verified season memberships after collection; not supplied by the match response.")
    totals = {r["side"]:r["value"] for r in tables.get("statistics", [])
              if r.get("key") == "expectedGoals" and r.get("period") == "ALL" and r.get("group_name") == "Match overview" and r.get("value") is not None}
    sums = {side:sum(r["xg"] for r in shots if r.get("side") == side and r.get("xg") is not None) for side in totals}
    if totals and shots and with_xg == len(shots) and any(abs(totals[s]-sums[s]) > .02 for s in totals):
        warnings.append("Provider match-level xG differs from the sum of supplied shot-level xG; both are retained separately.")
    if tables.get("matches") and all(row.get("status") != "finished" for row in tables["matches"]):
        for name in ("shot_xg", "team_heatmaps", "goal_sequences", "explicit_pass_recipients",
                     "exact_elapsed_duration", "added_time_announcements"):
            if features[name]["requested"]:
                features[name]["status"] = "not_applicable_before_completion"
    return {"features":features, "warnings":warnings,
            "xg_comparison":{"match_statistics":totals, "shot_sums":sums},
            "note":"Present means observed in these responses. Endpoint success does not establish every requested feature or season-wide coverage."}
