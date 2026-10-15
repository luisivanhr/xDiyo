"""Recovery requests missing data only and preserves the published baseline."""
from copy import deepcopy
import json
from pathlib import Path
import socket

import pytest

from scraping.sofascore.export import read_tables
from scraping.sofascore_web_v3.campaign import Campaign, prepare_campaign
from scraping.sofascore_web_v3.publish import hash_file, publish_captured_season
from scraping.sofascore_web_v3.recovery import (merge_match, prepare_recovery,
    process_season, queue_hash, queue_rows, retry_targets, watch_recovery)
import scraping.sofascore_web_v3.recovery as module

EID = 12436870
STEM = "Premier_League_24_25"
FIXTURES = Path(__file__).parents[1] / "fixtures/sofascore"


def payload(name):
    return json.loads((FIXTURES / f"{EID}_{name}.json").read_text(encoding="utf-8"))


def observe(campaign, job, suffix, body, *, day=12):
    return campaign.observe(EID, job["lease"], {
        "startedDateTime": f"2026-09-{day:02d}T02:00:00Z", "time": 10,
        "_observationSource": "chrome_session_request",
        "request": {"method": "GET", "url": f"https://www.sofascore.com/api/v1/event/{EID}{suffix}"},
        "response": {"status": 200 if body is not None else 404,
            "content": {"text": json.dumps(body)} if body is not None else {}}})


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    def offline(*args, **kwargs):
        raise AssertionError("Recovery preparation and publication must remain offline")
    monkeypatch.setattr(socket, "create_connection", offline)
    event = payload("event")["event"]
    info = prepare_campaign([EID], base=tmp_path / "campaigns", policy={"collection_mode": "session"},
        references={EID: {"event": event, "observed_at": 1700000000, "season_stem": STEM}})
    source = Path(info["directory"])
    with Campaign(source) as campaign:
        job = campaign.next_match()
        for suffix, body in [("", payload("event")), ("/pregame-form", None),
                ("/statistics", {"statistics": []}), ("/shotmap", None),
                ("/lineups", payload("lineups")), ("/incidents", payload("incidents")),
                *[(f"/heatmap/{event[s]['id']}", {"playerPoints": [{"x": 1, "y": 2}], "goalkeeperPoints": []}) for s in ("homeTeam", "awayTeam")]]:
            observe(campaign, job, suffix, body)
        campaign.finish(EID, job["lease"], page_event_id=None,
            page_evidence={"collection_mode": "session", "session_request_count": 8})
    output = tmp_path / "output"
    old = publish_captured_season(source, STEM, output=output)
    inventory = tmp_path / "inventory"
    inventory.mkdir()
    (inventory / "summary.json").write_text(json.dumps({"source_queue_sha256": queue_hash(queue_rows(source))}))
    (inventory / "core_gap_matches.csv").write_text(f"event_id\n{EID}\n")
    before = queue_rows(source)
    result = prepare_recovery(source, inventory, base=tmp_path / "campaigns", output=output)
    return {"source": source, "inventory": inventory, "before": before,
        "directory": Path(result["directory"]), "output": output, "old": old, "result": result}


def collect(data, body):
    with Campaign(data["directory"]) as campaign:
        job = campaign.next_match(resume=True)
        observe(campaign, job, "/shotmap", body, day=17)
        return campaign.finish(EID, job["lease"], page_event_id=None,
            page_evidence={"collection_mode": "session", "session_request_count": 1})


def test_preparation_reuses_success_and_pregame_404_without_changing_original(prepared):
    d = prepared
    assert d["result"]["components"] == 1
    assert d["result"]["seeded_observations"] == 7
    assert queue_rows(d["source"]) == d["before"]
    with Campaign(d["directory"]) as campaign:
        coverage = campaign.coverage(EID)["components"]
        assert [row["component"] for row in coverage if row["status"] == "not_observed"] == ["shotmap"]
        assert next(row for row in coverage if row["component"] == "pregame")["status"] == "unavailable"
    assert hash_file(d["output"] / f"{STEM}.parquet") == d["old"]["sha256"]


def test_changed_queue_refuses_preparation(prepared, tmp_path):
    d = prepared
    (d["inventory"] / "summary.json").write_text(json.dumps({"source_queue_sha256": "changed"}))
    with pytest.raises(ValueError, match="Queue changed"):
        prepare_recovery(d["source"], d["inventory"], base=tmp_path / "new", output=d["output"])
    assert not (tmp_path / "new").exists()


def test_still_404_is_recorded_without_replacing_season(prepared):
    d = prepared
    collect(d, None)
    result = process_season(d["directory"], STEM)
    assert result["state"] == "no_new_data"
    assert result["outcome_counts"] == {"still_unavailable_or_empty": 1}
    assert hash_file(d["output"] / f"{STEM}.parquet") == d["old"]["sha256"]
    assert watch_recovery(d["directory"], watch=False)["state"] == "finished_with_gaps"


def test_normal_publisher_refuses_partial_recovery_cohort(prepared, tmp_path):
    collect(prepared, payload("shotmap"))
    with pytest.raises(ValueError, match="not a full season"):
        publish_captured_season(prepared["directory"], STEM, output=tmp_path / "wrong_output")
    assert not (tmp_path / "wrong_output").exists()


def test_successful_fill_is_versioned_and_idempotent(prepared):
    d = prepared
    baseline = read_tables(d["output"] / d["old"]["manifest"])
    collect(d, payload("shotmap"))
    result = process_season(d["directory"], STEM)
    assert result["state"] == "published"
    assert result["outcome_counts"] == {"recovered": 1}
    current = read_tables(d["output"] / f"{STEM}.manifest.json")
    assert current["shots"]
    for name in ("matches", "pregame", "lineup_players", "heatmap_points", "incidents"):
        assert current.get(name) == baseline.get(name)
    assert hash_file(Path(result["backup"]) / f"{STEM}.parquet") == d["old"]["sha256"]
    again = process_season(d["directory"], STEM)
    assert again["replacement"] == result["replacement"]
    assert queue_rows(d["source"]) == d["before"]


def test_partial_entry_switch_rolls_back_and_can_resume(prepared, monkeypatch):
    d = prepared
    collect(d, payload("shotmap"))
    real = module.atomic_write_status
    failed = False
    def fail_second_entry(path, body):
        nonlocal failed
        if Path(path) == d["output"] / f"{STEM}.manifest.json" and not failed:
            failed = True
            raise OSError("simulated manifest switch failure")
        return real(path, body)
    monkeypatch.setattr(module, "atomic_write_status", fail_second_entry)
    with pytest.raises(OSError, match="simulated"):
        process_season(d["directory"], STEM)
    assert hash_file(d["output"] / f"{STEM}.parquet") == d["old"]["sha256"]
    assert read_tables(d["output"] / f"{STEM}.manifest.json")
    monkeypatch.setattr(module, "atomic_write_status", real)
    assert process_season(d["directory"], STEM)["state"] == "published"


def test_targets_exclude_pregame_and_xg_only_absence():
    report = {"coverage": [
        {"component": "pregame", "status": "unavailable", "path": "p"},
        {"component": "shotmap", "status": "success", "path": "s"},
        {"component": "lineups", "status": "success", "path": "l"}]}
    features = {"shot_xg": {"status": "not_supplied"}, "lineups": {"status": "not_supplied"}}
    assert [row["path"] for row in retry_targets(report, features)] == ["l"]


def test_published_manifest_change_is_not_silently_overwritten(prepared):
    d = prepared
    collect(d, payload("shotmap"))
    path = d["output"] / f"{STEM}.manifest.json"
    value = json.loads(path.read_text())
    value["unexpected_edit"] = True
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="baseline changed"):
        process_season(d["directory"], STEM)
    assert hash_file(d["output"] / f"{STEM}.parquet") == d["old"]["sha256"]


def test_heatmap_recovery_fills_only_selected_team():
    identity = {"event_id": 2, "competition_id": 17, "season_id": 1, "home_id": 3, "away_id": 4}
    existing = {"event_id": 2, "team_id": 3, "point_order": 0, "x": 55}
    tables = {"matches": [identity], "heatmap_points": [existing], "coverage": []}
    captured = {"matches": [identity], "heatmap_points": [{**existing, "x": 99},
        {"event_id": 2, "team_id": 4, "point_order": 0, "x": 44}],
        "coverage": [{"event_id": 2, "component": "heatmap", "side": "4", "status": "success"}]}
    target = {"component": "heatmap", "team_id": 4, "path": "/event/2/heatmap/4", "status": "unavailable"}
    assert merge_match(tables, captured, [target])[0]["outcome"] == "recovered"
    assert tables["heatmap_points"] == [existing, captured["heatmap_points"][1]]


def test_empty_lineup_recovery_keeps_old_absences_and_match_order():
    identity = {"event_id": 2, "competition_id": 17, "season_id": 1, "home_id": 3, "away_id": 4}
    old_absence = {"event_id": 2, "player_id": 20, "observed_at": 1.0}
    old_header = {"event_id": 2, "side": "home", "observed_at": 1.0}
    tables = {"matches": [{**identity, "event_id": 9}, identity, {**identity, "event_id": 1}],
        "lineup_teams": [old_header], "availability": [old_absence],
        "lineup_players": [{"event_id": 9, "player_id": 99}, {"event_id": 1, "player_id": 11}],
        "coverage": [{"event_id": 2, "component": "lineups", "status": "unavailable"}]}
    recovered = {"matches": [identity], "lineup_teams": [{"event_id": 2, "side": "home", "observed_at": 2.0}],
        "lineup_players": [{"event_id": 2, "side": "home", "player_id": 30}, {"event_id": 2, "side": "away", "player_id": 40}],
        "coverage": [{"event_id": 2, "component": "lineups", "status": "success"}]}
    targets = [{"component": "lineups", "path": "/event/2/lineups", "status": "success"}]
    assert merge_match(tables, recovered, targets)[0]["outcome"] == "recovered"
    assert tables["availability"] == [old_absence]
    assert old_header in tables["lineup_teams"]
    assert [row["event_id"] for row in tables["lineup_players"]] == [9, 2, 2, 1]
    saved = deepcopy(tables)
    assert merge_match(tables, recovered, targets)[0]["outcome"] == "already_present_in_published_version"
    assert tables == saved
