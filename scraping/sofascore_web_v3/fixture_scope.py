"""Preserve absent MLS round labels using an exact saved-fixture scope.

MLS's ordinary tournament 18 has no roundInfo in the original season pages or
full event metadata. The schema already permits null rounds. Scope is supplied
by validated fixture IDs; neither dates nor fixture order become fake rounds.
Other competitions retain their existing positive-round requirement.
"""
from scraping.sofascore.config import Scope

MLS_REVIEWED_SEASONS = {12825, 15813, 20108, 26780, 35964, 40071, 47955, 57317, 70158, 86668}


def saved_fixture_scope(competition_id, season_id, matches, **metadata):
    """Accept normalized identity/round rows from a verified Chrome selection."""
    if not matches or len({m["event_id"] for m in matches}) != len(matches):
        raise ValueError("Saved fixture scope needs distinct selected events")
    if any((m["competition_id"], m["season_id"]) != (competition_id, season_id) for m in matches):
        raise ValueError("Saved fixture scope crosses a provider season")
    rounds = tuple(m["round"] for m in matches)
    if any(n is None for n in rounds):
        if (competition_id != 242 or season_id not in MLS_REVIEWED_SEASONS
                or any(m["tournament_id"] != 18 or m["round"] is not None for m in matches)):
            raise ValueError("Missing rounds require an explicit reviewed ordinary-season contract")
        return Scope(competition_id, season_id, (), event_ids=tuple(m["event_id"] for m in matches), **metadata)
    return Scope(competition_id, season_id, rounds, **metadata)


def event_scope_row(event, number, season):
    return {"event_id": event["id"], "competition_id": event["tournament"]["uniqueTournament"]["id"],
            "tournament_id": event["tournament"].get("id"), "season_id": season["id"], "round": number}
