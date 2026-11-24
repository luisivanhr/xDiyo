"""Retained Egyptian source records require all reviewed identity guards."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scraping.sofascore.event_round import match_round
from scraping.sofascore.parsers import parse_match
from scraping.sofascore_web_v3.discovery import select_regular_records
from scraping.sofascore_web_v3.historical_formats import verified_replacement_id, reviewed_home_allocation
from scraping.sofascore_web_v3.archive_formats import validate_schedule

FIXTURE = json.loads((Path(__file__).parents[1] /
    "fixtures/sofascore/coverage_egypt4_reviewed_fixtures.json").read_text(encoding="utf-8"))
GUARDS = ("id", "tournament.uniqueTournament.id", "season.id", "season.year",
          "roundInfo.round", "homeTeam.id", "awayTeam.id", "startTimestamp",
          "status.type", "status.code", "homeScore.current", "awayScore.current", "isAwarded")


def changed(event, path):
    result = deepcopy(event)
    obj = result
    keys = path.split(".")
    for key in keys[:-1]:
        obj = obj.setdefault(key, {})
    value = obj.get(keys[-1])
    obj[keys[-1]] = value + 1 if type(value) is int else "unreviewed"
    return result


@pytest.mark.parametrize("row", FIXTURE["round_corrections"] + FIXTURE["missing_round_metadata"],
                         ids=lambda r: str(r["event"]["id"]))
def test_round_correction_requires_exact_identity_and_preserves_source(row):
    event = row["event"]
    original = deepcopy(event)
    assert match_round(event) == row["round"]
    assert parse_match(event)["round"] == row["round"]
    for path in GUARDS:
        other = changed(event, path)
        assert match_round(other) == (other.get("roundInfo") or {}).get("round"), path
    assert event == original


@pytest.mark.parametrize("row", FIXTURE["missing_round_metadata"], ids=lambda r: str(r["event"]["id"]))
def test_missing_round_recovery_rejects_different_stage_and_existing_round(row):
    event = row["event"]
    for path in ("tournament.id", "tournament.isGroup"):
        assert match_round(changed(event, path)) is None
    for round_info in ({}, {"round": 0}, {"round": 35}, {"round": 1, "name": "Final"}):
        other = deepcopy(event)
        other["roundInfo"] = round_info
        assert match_round(other) == round_info.get("round")


@pytest.mark.parametrize("row", FIXTURE["replacements"], ids=lambda r: str(r["original"]["id"]))
def test_obsolete_identity_needs_exact_played_replacement(row):
    event, played = row["original"], row["replacement"]
    original = deepcopy(row)
    assert verified_replacement_id(event, {played["id"]: played}) == played["id"]
    assert verified_replacement_id(event, {}) is None
    for path in GUARDS:
        assert verified_replacement_id(changed(event, path), {played["id"]: played}) is None, path
        assert verified_replacement_id(event, {played["id"]: changed(played, path)}) is None, path
    records = [{"id": e["id"], "competition": 808, "season": e["season"]["id"],
                "round": e["roundInfo"]["round"], "regular": 1, "payload": json.dumps(e)}
               for e in (event, played)]
    selected, superseded = select_regular_records(records)
    assert [r["id"] for r in selected] == [played["id"]]
    assert superseded[0]["replacement_event_id"] == played["id"]
    assert row == original


def test_official_home_allocation_preserves_both_actual_matches_and_full_schedule():
    review = deepcopy(FIXTURE["home_allocation"])
    before = deepcopy(review)
    events, spec = review["events"], review["spec"]
    standings = review["standings"]
    assert reviewed_home_allocation(events, spec, standings) == {(134186, 35187): 2, (35187, 134186): 0}
    assert validate_schedule(events, spec, standings) == []
    for event in events:
        if event["id"] in (7675984, 7779213):
            parsed = parse_match(event)
            assert (parsed["home_id"], parsed["away_id"]) == (134186, 35187)
    assert review == before


@pytest.mark.parametrize("event_id", (7675984, 7779213))
@pytest.mark.parametrize("path", GUARDS + ("tournament.id", "tournament.isGroup"))
def test_official_home_allocation_requires_both_exact_source_identities(event_id, path):
    review = deepcopy(FIXTURE["home_allocation"])
    events, spec = review["events"], review["spec"]
    standings = review["standings"]
    index = next(i for i, e in enumerate(events) if e["id"] == event_id)
    events[index] = changed(events[index], path)
    assert reviewed_home_allocation(events, spec, standings) == {}


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "round", "other_sides", "standings",
    "competition", "season", "snapshot", "teams", "round_limit", "meetings", "groups"))
def test_official_home_allocation_does_not_hide_other_schedule_errors(mutation):
    review = deepcopy(FIXTURE["home_allocation"])
    events, spec = review["events"], review["spec"]
    standings = review["standings"]
    other = next(e for e in events if e["id"] not in (7675984, 7779213, 7589258, 7618577))
    if mutation == "missing": events.remove(other)
    elif mutation == "duplicate": events.append(deepcopy(other))
    elif mutation == "round": other["roundInfo"]["round"] = 17 if other["roundInfo"]["round"] != 17 else 18
    elif mutation == "other_sides": other["homeTeam"], other["awayTeam"] = other["awayTeam"], other["homeTeam"]
    elif mutation == "standings": standings = {e[s]["id"]: 33 for e in events for s in ("homeTeam", "awayTeam")}
    else:
        key, value = {"competition": ("competition_id", 809), "season": ("season", "2018/2019"),
            "snapshot": ("snapshot", True), "teams": ("expected_teams", 19), "round_limit": ("round_limit", 35),
            "meetings": ("meetings_per_pair", 3), "groups": ("groups", 2)}[mutation]
        spec[key] = value
    assert validate_schedule(events, spec, standings)
