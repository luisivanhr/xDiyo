"""Goal-action capability boundary; no guessed provider routes or graph edges."""
from __future__ import annotations

CAPABILITY_STATUS = "bundled_in_incidents_observed_2026_09_12"


def derive_pass_edges(actions):
    """Derive edges from explicitly identified passes in canonical actions.

    Input is the normalized action contract, not an assumed Sofascore payload.
    The provider adapter must supply corroborated actor/recipient identity.
    Consecutive actors alone never substantiate a completed pass.
    """
    edges = []
    for action in actions:
        if action.get("action_type") != "pass" or action.get("completed") is not True:
            continue
        if action.get("actor_id") is None or action.get("recipient_id") is None:
            continue
        edges.append({"event_id":action["event_id"],"goal_id":action["goal_id"],
            "action_order":action["action_order"],"from_player_id":action["actor_id"],
            "to_player_id":action["recipient_id"]})
    return edges


def parse_goal_sequences(payload, event, context, shots=()):
    """Preserve the incident's ordered buildup, including non-pass movements.

    The observed payload does not explicitly name each pass recipient. Keep the
    next actor as a sequence attribute, not a claim that every action is a pass.
    Confirmed edges require independent identity evidence (currently the supplied
    assist annotation plus the incident's assist/scorer IDs).
    """
    from .parsers import encoded, number
    sequences, actions, edges = [], [], []
    for order, incident in enumerate(payload.get("incidents", [])):
        if incident.get("incidentType") != "goal":
            continue
        goal_id = incident.get("id")
        sequence_id = f'{event["id"]}:{goal_id if goal_id is not None else "ordinal-"+str(order)}'
        source = incident.get("footballPassingNetworkAction")
        scorer_id = (incident.get("player") or {}).get("id")
        own = incident.get("incidentClass") in {"ownGoal", "own"}
        is_home = incident.get("isHome")
        team_id = (event.get("homeTeam" if is_home is True else "awayTeam" if is_home is False else "") or {}).get("id")
        valid = isinstance(source, list) and all(isinstance(a, dict) for a in source)
        candidates = [shot for shot in shots if shot.get("event_id") == event["id"] and
                      shot.get("shot_type") == "goal" and scorer_id is not None and
                      shot.get("player_id") == scorer_id and team_id is not None and shot.get("team_id") == team_id and
                      number(incident.get("time")) is not None and shot.get("time") == number(incident.get("time")) and
                      (shot.get("added_time") or 0) == (number(incident.get("addedTime")) or 0)] if not own else []
        shot_id = candidates[0].get("shot_id") if len(candidates) == 1 else None
        status = "present" if valid and source else "empty_sequence" if valid else "not_supplied" if source is None else "invalid_sequence"
        base = {**context, "sequence_id": sequence_id, "goal_id": goal_id}
        sequences.append({**base, "source_order": order, "team_id": team_id, "scorer_id": scorer_id,
                          "time": number(incident.get("time")), "added_time": number(incident.get("addedTime")),
                          "goal_class": incident.get("incidentClass"), "status": status,
                          "shot_id":shot_id,"shot_link_method":"unique_event_team_scorer_clock" if shot_id is not None else "ambiguous" if len(candidates)>1 else "unlinked",
                          "action_count": len(source) if valid else None, "source_item_json": encoded(incident)})
        if not valid:
            continue
        for index, item in enumerate(source):
            actor_id = (item.get("player") or {}).get("id")
            next_actor = (source[index+1].get("player") or {}).get("id") if index+1 < len(source) else None
            assist = item.get("eventType") == "pass" and item.get("isAssist") is True and not own and (
                actor_id is not None and actor_id == (incident.get("assist1") or {}).get("id") and scorer_id is not None)
            action = {**base, "action_order": index, "actor_id": actor_id,
                      "action_type": item.get("eventType"), "recipient_id": scorer_id if assist else None,
                      "recipient_evidence": "incident_assist_and_scorer" if assist else "not_explicitly_supplied",
                      "next_action_actor_id": next_actor, "completed": True if assist else None,
                      "is_home": item.get("isHome"), "is_assist": item.get("isAssist"),
                      "time": number(item.get("time")), "time_base": "provider_minute_label",
                      "start_coordinates_json": encoded(item.get("playerCoordinates")),
                      "end_coordinates_json": encoded(item.get("passEndCoordinates", item.get("goalShotCoordinates"))),
                      "coordinate_system": "provider_goal_animation_unverified_orientation",
                      "source_item_json": encoded(item)}
            actions.append(action)
            for edge in derive_pass_edges([action]):
                edges.append({**context, **edge, "sequence_id": sequence_id,
                              "evidence": action["recipient_evidence"]})
    return {"goal_sequences": sequences, "goal_actions": actions, "passing_edges": edges}
