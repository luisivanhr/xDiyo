"""Pure transforms. No network, files, imputation or copied prior lineups."""
from __future__ import annotations

from collections import defaultdict
import json
import math

from .discovery import event_status
from .event_round import match_round, match_season


def encoded(value):
    return None if value is None else json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)


def number(value):
    if value is None or isinstance(value,bool):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError,TypeError):
        return None


def records(payload, key):
    result = payload.get(key)
    if not isinstance(result,list) or any(not isinstance(item,dict) for item in result):
        raise ValueError(f"{key} must be a list of objects")
    return result


def validate_payload(component, payload):
    if not isinstance(payload,dict) or "error" in payload:
        raise ValueError("Expected a successful JSON object")
    list_keys = {"discovery":"events", "season_registry":"seasons", "statistics":"statistics", "shotmap":"shotmap",
                 "incidents":"incidents", "comments":"comments"}
    if component in list_keys:
        values = records(payload,list_keys[component])
        if component == "discovery" and any(not isinstance(e.get("id"),int) for e in values):
            raise ValueError("Event ID absent")
        if component == "discovery" and "hasNextPage" in payload and not isinstance(payload["hasNextPage"], bool):
            raise ValueError("Discovery hasNextPage must be boolean")
        if component == "season_registry" and any(
                not isinstance(season.get("id"), int) or not isinstance(season.get("year"), str)
                for season in values):
            raise ValueError("Season registry ID or year absent")
        if component == "statistics":
            for period in values:
                if not isinstance(period.get("period"),str):
                    raise ValueError("Statistics period label absent")
                for group in records(period,"groups"):
                    records(group,"statisticsItems")
        return not values
    if component == "heatmap":
        return not (records(payload,"playerPoints") + records(payload,"goalkeeperPoints"))
    if component == "lineups":
        sides = [payload[s] for s in ("home","away") if isinstance(payload.get(s),dict)]
        if not sides:
            raise ValueError("Neither lineup side exists")
        for side in sides:
            for key in ("players","missingPlayers"):
                if key in side:
                    records(side,key)
        return False  # roster and availability emptiness are independent
    if component == "pregame":
        if not any(isinstance(payload.get(s),dict) for s in ("homeTeam","awayTeam")):
            raise ValueError("Pregame team objects absent")
        return False
    if component == "event":
        if not isinstance(payload.get("event"),dict) or not isinstance(payload["event"].get("id"),int):
            raise ValueError("Event object absent")
        return False
    raise ValueError(f"Unknown endpoint component: {component}")


def awarded_flag(event):
    """Project the optional provider flag; never infer it from the match status.

    Sofascore omits isAwarded on ordinary event objects. Missing *metadata*,
    explicit nulls and malformed values remain unknown rather than false.
    """
    if not isinstance(event, dict) or type(event.get("id")) is not int:
        return None
    value = event.get("isAwarded", False)
    return value if type(value) is bool else None


def parse_match(event, *, observed_at=None):
    """Retain the provider's event start, without asserting an actual kickoff.

    startTimestamp is UTC Unix seconds. It can be a scheduled/revised start;
    completion status alone does not verify the first whistle's timestamp.
    observed_at describes when this metadata was fetched, never when play began.
    """
    tournament = event.get("tournament") or {}
    season = match_season(event)
    row = {"event_id":event["id"], "custom_id":event.get("customId"),
           "competition_id":(tournament.get("uniqueTournament") or {}).get("id"),
           "tournament_id":tournament.get("id"), "season_id":season.get("id"),
           "season_year":str(season["year"]) if "year" in season else None,
           "kickoff_utc":number(event.get("startTimestamp")),
           "event_metadata_observed_at":number(observed_at),"status":event_status(event),
           "is_awarded":awarded_flag(event),
           "status_code":(event.get("status") or {}).get("code"),
           "round":match_round(event),
           "stage_json":encoded(event.get("roundInfo")),"time_json":encoded(event.get("time"))}
    for side in ("home","away"):
        team = event.get(side+"Team") or {}
        row.update({side+"_id":team.get("id"),side+"_name":team.get("name"),side+"_slug":team.get("slug")})
        scores = event.get(side+"Score") or {}
        for field in ("current","display","period1","period2","extra1","extra2","overtime","penalties"):
            row[side+"_score_"+field] = number(scores.get(field))
    return row


def parse_component(job, payload, event, profile, *, observed_at, raw_hash):
    validate_payload(job.component,payload)
    tables = defaultdict(list)
    context = {"event_id":event["id"],"source_key":job.key,"raw_hash":raw_hash,"observed_at":observed_at}

    def put(table, row):
        tables[table].append({**context, **row})

    def coverage(component, side, status):
        put("coverage",{"component":component,"side":side,"status":status})

    if job.component == "pregame":
        for side in ("home","away"):
            obj = payload.get(side+"Team") or {}
            put("pregame",{"side":side,"team_id":(event.get(side+"Team") or {}).get("id"),
                           "position":number(obj.get("position")),"value_json":encoded(obj)})
    elif job.component == "statistics":
        for period in payload["statistics"]:
            for group_index,group in enumerate(period["groups"]):
                for item_index,item in enumerate(group["statisticsItems"]):
                    for side in ("home","away"):
                        put("statistics",{"period":period["period"],"group_name":group.get("groupName"),
                            "group_order":group_index,"stat_order":item_index,
                            "key":str(item.get("key",item.get("name",f"unnamed_{item_index}"))),
                            "name":item.get("name"),"side":side,
                            "team_id":(event.get(side+"Team") or {}).get("id"),
                            "value":number(item.get(side+"Value")),"total":number(item.get(side+"Total")),
                            "display":str(item[side]) if side in item else None,
                            "value_type":item.get("valueType"),"statistics_type":item.get("statisticsType"),
                            "source_item_json":encoded(item)})
    elif job.component == "shotmap":
        for index,shot in enumerate(payload["shotmap"]):
            player = shot.get("player") or {}
            is_home = shot.get("isHome")
            side = "home" if is_home is True else "away" if is_home is False else None
            row = {"source_order":index,"shot_id":shot.get("id"),"side":side,
                   "team_id":(event.get((side or "")+"Team") or {}).get("id"),
                   "player_id":player.get("id"),"player_name":player.get("name"),
                   "xg":number(shot.get("xg")),"xgot":number(shot.get("xgot")),
                   "xg_quality":"source_value" if number(shot.get("xg")) is not None else "unavailable",
                   "xgot_quality":"source_value" if number(shot.get("xgot")) is not None else "unavailable",
                   "shot_type":shot.get("shotType"),"goal_type":shot.get("goalType"),
                   "situation":shot.get("situation"),"body_part":shot.get("bodyPart"),
                   "period":str(shot["period"]) if "period" in shot else None,
                   "time":number(shot.get("time")),"added_time":number(shot.get("addedTime")),
                   "time_seconds":number(shot.get("timeSeconds")),"source_item_json":encoded(shot)}
            for key,prefix in (("playerCoordinates","player"),("goalMouthCoordinates","goal_mouth"),
                               ("draw","draw")):
                coordinates = shot.get(key) or {}
                if isinstance(coordinates,dict):
                    for axis in ("x","y","z"):
                        row[prefix+"_"+axis] = number(coordinates.get(axis))
            put("shots",row)
    elif job.component == "heatmap":
        for key,kind in (("playerPoints","player"),("goalkeeperPoints","goalkeeper")):
            for index,point in enumerate(payload[key]):
                put("heatmap_points",{"team_id":job.team_id,"kind":kind,"point_order":index,
                    "x":number(point.get("x")),"y":number(point.get("y")),"weight":number(point.get("weight")),
                    "source_item_json":encoded(point)})
        coverage("heatmap",job.reasons[0] if job.reasons else str(job.team_id),
                 "present" if payload["playerPoints"] or payload["goalkeeperPoints"] else "empty")
    elif job.component == "lineups":
        status = event_status(event)
        phase = "postgame" if status == "finished" else "inprogress" if status in {"inprogress", "interrupted"} else "pregame"
        for side in ("home","away"):
            team_id = (event.get(side+"Team") or {}).get("id")
            data = payload.get(side)
            base = {"side":side,"team_id":team_id,"phase":phase}
            if not isinstance(data,dict):
                for component,enabled in (("lineups",profile.include_lineups),("availability",profile.include_injuries)):
                    if enabled:
                        coverage(component,side,"unavailable")
                continue
            if profile.include_lineups:
                confirmation = payload.get("confirmed",data.get("confirmed"))
                put("lineup_teams",{**base,"formation":data.get("formation"),
                    "confirmed":confirmation if isinstance(confirmation,bool) else None,
                    "roster_status":"unavailable" if "players" not in data else "present" if data["players"] else "empty"})
                coverage("lineups",side,"unavailable" if "players" not in data else "present" if data["players"] else "empty")
                for index,item in enumerate(data.get("players",[])):
                    player = item.get("player") or {}
                    roster = {**base,"source_order":index,"player_id":player.get("id"),
                        "player_name":player.get("name"),"player_slug":player.get("slug"),
                        "position":item.get("position",player.get("position")),
                        "substitute":item.get("substitute"),"shirt_number":str(item["shirtNumber"]) if "shirtNumber" in item else None,
                        "statistics_status":"present" if item.get("statistics") else "empty" if "statistics" in item else "unavailable",
                        "source_item_json":encoded(item)}
                    put("lineup_players",roster)
                    for key,value in (item.get("statistics") or {}).items():
                        put("player_statistics",{**base,"source_order":index,"player_id":player.get("id"),
                            "key":key,"value":number(value),"value_json":encoded(value)})
            if profile.include_injuries:
                coverage("availability",side,"unavailable" if "missingPlayers" not in data else "present" if data["missingPlayers"] else "empty")
                for index,item in enumerate(data.get("missingPlayers",[])):
                    player = item.get("player") or {}
                    # Codes retain their raw value until their provider meaning is validated.
                    put("availability",{**base,"source_order":index,"player_id":player.get("id"),
                        "player_name":player.get("name"),"player_slug":player.get("slug"),
                        "reason_json":encoded(item.get("reason")),"status_json":encoded(item.get("type",item.get("status"))),
                        "classification":"unclassified_reported_absence","source_item_json":encoded(item)})
    elif job.component == "incidents":
        for index,item in enumerate(payload["incidents"]):
            put("incidents",{"source_order":index,"incident_id":item.get("id"),
                "incident_type":item.get("incidentType"),"incident_class":item.get("incidentClass"),
                "time":number(item.get("time")),"added_time":number(item.get("addedTime")),
                "time_seconds":number(item.get("timeSeconds")),"is_home":item.get("isHome"),
                "player_id":(item.get("player") or {}).get("id"),"source_item_json":encoded(item)})
    elif job.component == "comments":
        for index,item in enumerate(payload["comments"]):
            put("comments",{"source_order":index,"time":number(item.get("time")),
                "text":item.get("text"),"is_home":item.get("isHome"),"source_item_json":encoded(item)})
    return dict(tables)
