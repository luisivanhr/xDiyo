"""Frozen observed payloads. Tests replay locally and make no API requests."""
from dataclasses import replace
import json
from pathlib import Path

from scraping.sofascore.config import Profile
from scraping.sofascore.goal_actions import parse_goal_sequences
from scraping.sofascore.models import Job
from scraping.sofascore.parsers import parse_component
from scraping.sofascore.planner import component_jobs
from scraping.sofascore.schema import to_arrow
from scraping.sofascore.timing import timing_evidence

FIXTURES = Path(__file__).parents[1]/"fixtures"/"sofascore"


def fixture(eid, component):
    return json.loads((FIXTURES/f"{eid}_{component}.json").read_text(encoding="utf-8"))


def test_observed_goal_bundle_shares_incidents_and_preserves_non_passes():
    event = fixture(12436870, "event")["event"]
    profile = Profile(include_goal_actions=True, duration_mode="incidents", include_lineups=False,
                      include_injuries=False, include_comments=False, include_shotmap=False)
    jobs = component_jobs(event, profile, 2000000000)
    assert len([j for j in jobs if j.component == "incidents"]) == 1
    assert not {"comments", "lineups", "shotmap"} & {j.component for j in jobs}
    payload = fixture(12436870, "incidents")
    parsed = parse_goal_sequences(payload, event, {"event_id": event["id"]})
    assert len(parsed["goal_sequences"]) == 1
    assert parsed["goal_sequences"][0]["goal_id"] == 185982125
    assert [a["action_type"] for a in parsed["goal_actions"]] == ["pass", "ball-movement", "pass", "pass", "pass", "pass", "goal"]
    assert parsed["goal_actions"][0]["recipient_id"] is None
    assert parsed["goal_actions"][0]["next_action_actor_id"] == 288205
    assert len(parsed["passing_edges"]) == 1
    shot_job = Job("/event/12436870/shotmap","shotmap",12436870)
    shots = parse_component(shot_job,fixture(12436870,"shotmap"),event,profile,
                            observed_at=None,raw_hash="fixture")["shots"]
    linked = parse_goal_sequences(payload,event,{"event_id":event["id"]},shots)
    assert linked["goal_sequences"][0]["shot_id"] == 3601064
    ambiguous = parse_goal_sequences(payload,event,{"event_id":event["id"]},shots+shots)
    assert ambiguous["goal_sequences"][0]["shot_id"] is None
    assert ambiguous["goal_sequences"][0]["shot_link_method"] == "ambiguous"
    edge = parsed["passing_edges"][0]
    assert (edge["from_player_id"], edge["to_player_id"]) == (1135873, 917007)
    for name, rows in parsed.items():
        assert to_arrow(name, rows).num_rows == len(rows)


def test_observed_2015_xg_is_missing_not_one_and_timing_stays_unmeasured():
    event = fixture(6767929, "event")["event"]
    job = Job("/event/6767929/shotmap", "shotmap", 6767929)
    shots = parse_component(job, fixture(6767929, "shotmap"), event, Profile(), observed_at=None, raw_hash="frozen-fixture")["shots"]
    assert shots and all(shot["xg"] is None for shot in shots)
    recent = fixture(12436870, "event")["event"]
    timing = timing_evidence(recent, fixture(12436870, "incidents"))
    assert all(row["elapsed_seconds"] is None for row in timing)
    assert sorted(row["announced_added_minutes"] for row in timing if row["announced_added_minutes"] is not None) == [1, 4]
    assert any(row["quality"] == "invalid_or_sentinel" for row in timing)


def test_missing_or_empty_goal_sequence_does_not_invent_a_graph():
    event = {"id": 1}
    payload = {"incidents": [{"incidentType": "goal", "id": 2},
                             {"incidentType": "goal", "id": 3, "footballPassingNetworkAction": []},
                             {"incidentType": "penaltyShootout", "id": 4}]}
    parsed = parse_goal_sequences(payload, event, {"event_id": 1})
    assert [s["status"] for s in parsed["goal_sequences"]] == ["not_supplied", "empty_sequence"]
    assert parsed["goal_actions"] == parsed["passing_edges"] == []
