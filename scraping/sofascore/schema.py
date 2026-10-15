"""Versioned, nullable table schemas. Field order is deterministic."""
from __future__ import annotations

SCHEMA_VERSION = "2"

MATCH_AWARDED_CONTRACT = {
    "is_awarded": {
        "source_field": "event.isAwarded",
        "meaning": "Provider awarded-result flag, independent of status and status_code",
        "missing_field": "False when omitted from a valid provider event object",
        "unknown": "Null when event metadata is unavailable or the flag is null or non-boolean",
    },
}

MATCH_TIME_CONTRACT = {
    "kickoff_utc": {
        "source_field": "event.startTimestamp",
        "meaning": "Provider scheduled or revised event start; actual kickoff unverified",
        "unit": "seconds_since_unix_epoch", "timezone": "UTC",
    },
    "event_metadata_observed_at": {
        "source_field": "collector event registry observation timestamp",
        "meaning": "Fetch time of the event metadata used in this match row; not match start or finish",
        "unit": "seconds_since_unix_epoch", "timezone": "UTC",
    },
}

CONTEXT = {"event_id":"int64","source_key":"string","raw_hash":"string","observed_at":"float64"}
TEAM = {"side":"string","team_id":"int64","phase":"string"}


def fields(names, kind="string"):
    return {name:kind for name in names.split()}


TABLES = {
    "matches":{**fields("event_id competition_id tournament_id season_id status_code round home_id away_id","int64"),
        **fields("custom_id season_year status stage_json time_json home_name home_slug away_name away_slug"),
        "kickoff_utc":"float64","event_metadata_observed_at":"float64","is_awarded":"bool",
        **{side+"_score_"+part:"float64" for side in ("home","away") for part in
           ("current","display","period1","period2","extra1","extra2","overtime","penalties")}},
    "coverage":{**CONTEXT,**fields("component side status latest_attempt_status")},
    "pregame":{**CONTEXT,"side":"string","team_id":"int64","position":"float64","value_json":"string"},
    "statistics":{**CONTEXT,**fields("period group_name key name side display value_type statistics_type source_item_json"),
        **fields("team_id group_order stat_order","int64"),**fields("value total","float64")},
    "shots":{**CONTEXT,**fields("source_order shot_id team_id player_id","int64"),
        **fields("side player_name shot_type goal_type situation body_part period source_item_json xg_quality xgot_quality"),
        **fields("xg xgot legacy_xg_stored legacy_xgot_stored time added_time time_seconds player_x player_y player_z goal_mouth_x goal_mouth_y goal_mouth_z draw_x draw_y draw_z","float64")},
    "heatmap_points":{**CONTEXT,**fields("team_id point_order","int64"),**fields("kind source_item_json"),
        **fields("x y weight","float64")},
    "lineup_teams":{**CONTEXT,**TEAM,**fields("formation roster_status"),"confirmed":"bool"},
    "lineup_players":{**CONTEXT,**TEAM,**fields("source_order player_id","int64"),
        **fields("player_name player_slug position shirt_number statistics_status source_item_json"),"substitute":"bool"},
    "player_statistics":{**CONTEXT,**TEAM,**fields("source_order player_id","int64"),
        **fields("key value_json"),"value":"float64"},
    "availability":{**CONTEXT,**TEAM,**fields("source_order player_id","int64"),
        **fields("player_name player_slug reason_json status_json classification source_item_json")},
    "incidents":{**CONTEXT,**fields("source_order incident_id player_id","int64"),
        **fields("incident_type incident_class source_item_json"),**fields("time added_time time_seconds","float64"),"is_home":"bool"},
    "comments":{**CONTEXT,"source_order":"int64","time":"float64","text":"string","is_home":"bool","source_item_json":"string"},
    "goal_sequences":{**CONTEXT,**fields("sequence_id goal_class status source_item_json shot_link_method"),
        **fields("goal_id source_order team_id scorer_id action_count shot_id","int64"),**fields("time added_time","float64")},
    "goal_actions":{**CONTEXT,**fields("sequence_id action_type recipient_evidence time_base start_coordinates_json end_coordinates_json coordinate_system source_item_json"),
        **fields("goal_id action_order actor_id recipient_id next_action_actor_id","int64"),
        **fields("completed is_home is_assist","bool"),"time":"float64"},
    "passing_edges":{**CONTEXT,**fields("sequence_id evidence"),**fields("goal_id action_order from_player_id to_player_id","int64")},
    "timing":{**CONTEXT,**fields("period source_field source_value_json quality"),
        **fields("announced_added_minutes candidate_added_minutes elapsed_seconds","float64")},
    "team_seasons":{**fields("competition_id season_id team_id previous_competition_id previous_season_id","int64"),
        **fields("movement evidence source membership_status derivation_version"),
        **fields("got_promoted got_demoted","bool")},
    "legacy_fields":{**CONTEXT,**fields("key value_json")},
}


def arrow_schema(table):
    import pyarrow as pa
    fields = []
    for name, kind in TABLES[table].items():
        description = (MATCH_TIME_CONTRACT.get(name) or MATCH_AWARDED_CONTRACT.get(name)) if table == "matches" else None
        metadata = {k.encode():v.encode() for k,v in description.items()} if description else None
        fields.append(pa.field(name,pa.type_for_alias(kind),nullable=True,metadata=metadata))
    return pa.schema(fields,
                     metadata={b"xdiyo_schema_version":SCHEMA_VERSION.encode()})


def to_arrow(table, rows):
    import pyarrow as pa
    schema = arrow_schema(table)
    for row in rows:
        unknown = row.keys() - TABLES[table].keys()
        if unknown:
            raise ValueError(f"Unmapped {table} fields: {sorted(unknown)}")
    return pa.Table.from_pylist(rows,schema=schema)
