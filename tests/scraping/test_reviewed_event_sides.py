"""Exact Saudi corrections retain source bytes and team-attributed measurements."""
from copy import deepcopy
import json
from pathlib import Path
import socket
import sqlite3

import pytest

from scraping.sofascore.config import Profile
from scraping.sofascore.event_sides import match_team_ids, reviewed_reversed_sides
from scraping.sofascore.goal_actions import parse_goal_sequences
from scraping.sofascore.models import Job
from scraping.sofascore.parsers import parse_component, parse_match
from scraping.sofascore.export import read_tables
from scraping.sofascore_web_v3.awarded import SavedAwardedFlags
from scraping.sofascore_web_v3.campaign import Campaign, prepare_campaign
from scraping.sofascore_web_v3.publish import hash_file, publish_captured_season
from scraping.sofascore_web_v3.reconcile_publication import reconcile_publication

FIXTURES = Path(__file__).parents[1] / "fixtures/sofascore"
EVENTS = json.loads((FIXTURES / "coverage_reviewed_fixtures.json").read_text(encoding="utf-8"))["side_reversals"]
UKRAINE_EVENTS = json.loads((FIXTURES / "coverage_europe3_reviewed_fixtures.json").read_text(encoding="utf-8"))["side_reversals"]
SAUDI_EVENTS = EVENTS
EGYPT_EVENTS = json.loads((FIXTURES / "coverage_egypt4_reviewed_fixtures.json").read_text(encoding="utf-8"))["side_reversals"]
CHINA_EVENTS = json.loads((FIXTURES / "remaining_side_corrections.json").read_text(encoding="utf-8"))
EVENTS = SAUDI_EVENTS + UKRAINE_EVENTS + EGYPT_EVENTS + CHINA_EVENTS


@pytest.mark.parametrize("event", EVENTS, ids=lambda e: str(e["id"]))
def test_reviewed_sides_require_complete_exact_identity_and_do_not_rewrite_source(event):
    before = deepcopy(event)
    assert reviewed_reversed_sides(event)
    match = parse_match(event)
    assert (match["home_id"], match["away_id"]) == (event["awayTeam"]["id"], event["homeTeam"]["id"])
    assert (match["home_score_current"], match["away_score_current"]) == (
        event["awayScore"]["current"], event["homeScore"]["current"])
    assert match_team_ids(event) == (match["home_id"], match["away_id"])
    for path in ("id", "tournament.uniqueTournament.id", "season.id", "season.year",
                 "roundInfo.round", "homeTeam.id", "awayTeam.id", "startTimestamp",
                 "status.type", "status.code", "homeScore.current", "awayScore.current", "isAwarded"):
        changed = deepcopy(event)
        obj = changed
        parts = path.split(".")
        for key in parts[:-1]:
            obj = obj[key]
        value = obj.get(parts[-1])
        obj[parts[-1]] = True if value is None else value + 1 if type(value) is int else "unreviewed"
        assert not reviewed_reversed_sides(changed), path
    corrected = deepcopy(event)
    for suffix in ("Team", "Score"):
        corrected["home" + suffix], corrected["away" + suffix] = corrected["away" + suffix], corrected["home" + suffix]
    assert not reviewed_reversed_sides(corrected)  # Never reverse an already-correct provider response.
    assert event == before


@pytest.mark.parametrize("event", EVENTS, ids=lambda e: str(e["id"]))
def test_all_normalized_side_fields_keep_raw_team_player_and_value_attribution(event):
    # Deliberately distinct synthetic component measurements expose accidental
    # relabelling of teams, swapped values, or double reversal.
    payloads = {
        "pregame": {"homeTeam": {"position": 11}, "awayTeam": {"position": 3}},
        "statistics": {"statistics": [{"period": "ALL", "groups": [{"groupName": "Shots",
            "statisticsItems": [{"key": "totalShotsOnGoal", "homeValue": 12, "awayValue": 7,
                                 "home": "12", "away": "7"}]}]}]},
        "lineups": {"confirmed": True,
            "home": {"formation": "4-4-2", "players": [{"player": {"id": 101, "name": "Raw home player"},
                      "statistics": {"rating": 6.2}}], "missingPlayers": [{"player": {"id": 102}}]},
            "away": {"formation": "4-3-3", "players": [{"player": {"id": 201, "name": "Raw away player"},
                      "statistics": {"rating": 7.1}}], "missingPlayers": [{"player": {"id": 202}}]}},
        "shotmap": {"shotmap": [{"id": 1, "player": {"id": 101}, "isHome": True,
            "shotType": "goal", "time": 74, "xg": 0.2, "playerCoordinates": {"x": 90, "y": 40}},
            {"id": 2, "player": {"id": 201}, "isHome": False, "shotType": "save", "time": 82, "xg": 0.7}]},
        "incidents": {"incidents": [{"id": 3, "incidentType": "goal", "isHome": True,
            "player": {"id": 101}, "time": 74, "footballPassingNetworkAction": [
                {"player": {"id": 101}, "isHome": True, "eventType": "shot"}]}]},
        "heatmap": {"playerPoints": [{"x": 30, "y": 40}], "goalkeeperPoints": []},
    }
    before = deepcopy(payloads)
    tables = {}
    for component, payload in payloads.items():
        job = Job(f"/event/{event['id']}/{component}", component, event["id"],
            team_id=event["homeTeam"]["id"] if component == "heatmap" else None,
            reasons=("home",) if component == "heatmap" else ())
        parsed = parse_component(job, payload, event, Profile(), observed_at=1, raw_hash="raw-fixture")
        tables.update(parsed)
    old_home, old_away = event["homeTeam"]["id"], event["awayTeam"]["id"]
    assert [(r["team_id"], r["side"], r["value"]) for r in tables["statistics"]] == [
        (old_home, "away", 12), (old_away, "home", 7)]
    assert [(r["team_id"], r["side"], r["position"]) for r in tables["pregame"]] == [
        (old_home, "away", 11), (old_away, "home", 3)]
    assert [(r["player_id"], r["team_id"], r["side"]) for r in tables["lineup_players"]] == [
        (101, old_home, "away"), (201, old_away, "home")]
    assert [(r["team_id"], r["side"], r["xg"]) for r in tables["shots"]] == [
        (old_home, "away", 0.2), (old_away, "home", 0.7)]
    assert tables["shots"][0]["player_x"] == 90  # Never invent a coordinate rotation.
    assert json.loads(tables["shots"][0]["source_item_json"])["isHome"] is True
    assert tables["incidents"][0]["is_home"] is False
    assert tables["heatmap_points"][0]["team_id"] == old_home
    goals = parse_goal_sequences(payloads["incidents"], event, {"event_id": event["id"]}, tables["shots"])
    assert goals["goal_sequences"][0]["team_id"] == old_home
    assert goals["goal_sequences"][0]["shot_id"] == 1
    assert goals["goal_actions"][0]["is_home"] is False
    assert json.loads(goals["goal_actions"][0]["source_item_json"])["isHome"] is True
    assert payloads == before


@pytest.mark.parametrize("stem,events", [("Saudi_Pro_League_18_19", SAUDI_EVENTS),
                                        ("Chinese_Super_League_22_22", CHINA_EVENTS),
                                        ("Ukrainian_Premier_League_20_21", UKRAINE_EVENTS),
                                        *[(f"Egyptian_Premier_League_{year.replace('/', '_')}",
                                           [e for e in EGYPT_EVENTS if e["season"]["year"] == year])
                                          for year in sorted({e["season"]["year"] for e in EGYPT_EVENTS})]])
def test_reviewed_sides_survive_capture_publication_and_reconciliation(tmp_path, monkeypatch, stem, events):
    def no_network(*args, **kwargs):
        raise AssertionError("Fixture regression must stay offline")
    monkeypatch.setattr(socket.socket, "connect", no_network)
    references = {event["id"]: {"event": event, "observed_at": 1700000000,
                               "season_stem": stem} for event in events}
    info = prepare_campaign(list(references), base=tmp_path / "campaigns", references=references)
    directory = Path(info["directory"])
    with Campaign(directory) as campaign:
        for offset, event in enumerate(events):
            job = campaign.next_match()
            eid = event["id"]
            assert job["event_id"] == eid
            # Actual saved event metadata, deliberately synthetic components.
            # No synthetic measurement is written to a production capture store.
            responses = [
                ("", {"event": event}),
                ("/pregame-form", {"homeTeam": {"position": 11}, "awayTeam": {"position": 3}}),
                ("/statistics", {"statistics": [{"period": "ALL", "groups": [{"groupName": "Shots",
                    "statisticsItems": [{"key": "totalShotsOnGoal", "homeValue": 12, "awayValue": 7,
                                         "home": "12", "away": "7"}]}]}]}),
                ("/shotmap", {"shotmap": []}),
                ("/lineups", {"home": {"players": []}, "away": {"players": []}}),
                ("/incidents", {"incidents": []}),
                *[(f"/heatmap/{event[side]['id']}", {"playerPoints": [{"x": 1, "y": 2}],
                                                     "goalkeeperPoints": []})
                  for side in ("homeTeam", "awayTeam")],
            ]
            for index, (suffix, payload) in enumerate(responses):
                campaign.observe(eid, job["lease"], {
                    "startedDateTime": f"2026-11-09T02:{offset:02d}:{index:02d}Z", "time": 10,
                    "request": {"method": "GET", "url": f"https://www.sofascore.com/api/v1/event/{eid}{suffix}"},
                    "response": {"status": 200, "content": {"text": json.dumps(payload)}},
                })
            assert campaign.finish(eid, job["lease"], page_event_id=eid)["state"] == "complete"
    original = {name: hash_file(directory / name) for name in ("config.json", "queue.sqlite")}
    output = tmp_path / "published"
    result = publish_captured_season(directory, stem, output=output)
    assert result["publication_state"] == "published"
    assert result["matches"] == len(events)
    tables = read_tables(output / result["manifest"], verify_hashes=True)
    with SavedAwardedFlags([directory / "queue.sqlite"]) as awarded:
        for event, match in zip(events, tables["matches"]):
            assert match["event_id"] == event["id"]
            assert (match["home_id"], match["away_id"]) == (event["awayTeam"]["id"], event["homeTeam"]["id"])
            assert (match["home_score_current"], match["away_score_current"]) == (
                event["awayScore"]["current"], event["homeScore"]["current"])
            assert awarded.get(match).value is False
            stats = [r for r in tables["statistics"] if r["event_id"] == event["id"]]
            assert {(r["team_id"], r["side"], r["value"]) for r in stats} == {
                (event["homeTeam"]["id"], "away", 12), (event["awayTeam"]["id"], "home", 7)}
    preview = reconcile_publication(directory, stem, output=output)
    assert preview["ready"] is True
    assert publish_captured_season(directory, stem, output=output)["publication_state"] == "already_published"
    assert original == {name: hash_file(directory / name) for name in original}
    with sqlite3.connect((directory / "queue.sqlite").as_uri() + "?mode=ro", uri=True) as db:
        for eid, encoded in db.execute("SELECT event_id,entry_json FROM observations WHERE path='/event/' || event_id"):
            assert json.loads(json.loads(encoded)["response"]["content"]["text"])["event"] == references[eid]["event"]
