"""Use frozen JSON responses inside HAR fixtures; importing makes no network calls."""
import base64
from dataclasses import replace
import json
from pathlib import Path
import socket

import pytest

from scraping.sofascore.export import read_tables
from scraping.sofascore_web_v3.capture import PROFILE
from scraping.sofascore_web_v3.import_har import import_har

FIXTURES = Path(__file__).parents[1]/"fixtures/sofascore"
EVENT_ID = 12436870


def fixture(component):
    return json.loads((FIXTURES/f"{EVENT_ID}_{component}.json").read_text(encoding="utf-8"))


def entry(suffix, payload=None, *, status=200, seconds=0, encoding=None, query=""):
    content = {"mimeType":"application/json"}
    if payload is not None:
        text = json.dumps(payload)
        content["text"] = base64.b64encode(text.encode()).decode() if encoding == "base64" else text
    if encoding:
        content["encoding"] = encoding
    return {"startedDateTime":f"2026-09-12T02:00:{seconds:02d}Z", "time":10,
            "request":{"method":"GET", "url":f"https://www.sofascore.com/api/v1/event/{EVENT_ID}"+suffix+query,
                       "headers":[{"name":"Authorization", "value":"MUST_NOT_PERSIST"}],
                       "cookies":[{"name":"session", "value":"MUST_NOT_PERSIST"}]},
            "response":{"status":status, "content":content}}


def archive(tmp_path, entries, name="recording.har"):
    path = tmp_path/name
    path.write_text(json.dumps({"log":{"version":"1.2", "entries":entries}}), encoding="utf-8")
    return path


def full_entries():
    event = fixture("event")["event"]
    return [entry("", fixture("event")), entry("/pregame-form", status=404),
            entry("/statistics", {"statistics":[]}), entry("/shotmap", fixture("shotmap"), encoding="base64"),
            entry("/lineups", fixture("lineups")), entry("/incidents", fixture("incidents")),
            *[entry(f"/heatmap/{event[side]['id']}", {"playerPoints":[{"x":1,"y":2}], "goalkeeperPoints":[]})
              for side in ("homeTeam", "awayTeam")]]


def test_network_archive_to_parquet_is_offline_complete_and_sanitized(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("HAR import must not open a network connection")
    monkeypatch.setattr(socket, "create_connection", no_network)
    entries = full_entries()+[entry("/comments", {"comments":[{"text":"COMMENTARY_MUST_NOT_PERSIST"}]})]
    result = import_har(archive(tmp_path, entries), event_id=EVENT_ID, output_root=tmp_path/"out")
    assert result["complete"] and result["requests_sent_by_importer"] == 0
    tables = read_tables(result["manifest"])
    assert len(tables["shots"]) == len(fixture("shotmap")["shotmap"])
    assert any(row["xg"] is not None for row in tables["shots"])
    assert len(tables["goal_actions"]) == 7
    assert len(tables["heatmap_points"]) == 2
    assert len(tables["lineup_players"]) > 22
    assert tables["matches"][0]["event_metadata_observed_at"] == pytest.approx(1789178400.01)
    assert "comments" not in tables
    for path in Path(result["directory"]).rglob("*"):
        if path.is_file():
            assert b"MUST_NOT_PERSIST" not in path.read_bytes()
    assert not (tmp_path/"collection_state").exists()


def test_missing_body_is_failure_not_empty_success(tmp_path):
    entries = [entry("", fixture("event")), entry("/shotmap")]
    result = import_har(archive(tmp_path, entries), event_id=EVENT_ID, output_root=tmp_path/"out")
    assert not result["complete"]
    assert next(row["status"] for row in result["coverage"] if row["component"] == "shotmap") == "failed"
    records = json.loads((Path(result["directory"])/"requests.json").read_text())
    assert records[-1]["capture"] == "response_body_missing"


def test_verified_round_zero_metadata_imports_as_round_four_without_rewriting_archive(tmp_path):
    event = fixture("event")["event"]
    event.update(id=6881878, season={"id": 10753, "year": "15/16"},
                 roundInfo={"round": 0}, startTimestamp=1442946600)
    event["tournament"]["uniqueTournament"]["id"] = 53
    event["homeTeam"]["id"], event["awayTeam"]["id"] = 2742, 2706
    recording = entry("", {"event": event})
    recording["request"]["url"] = "https://www.sofascore.com/api/v1/event/6881878"
    path = archive(tmp_path, [recording])
    original = path.read_bytes()
    profile = replace(PROFILE, include_pregame=False, include_statistics=False,
        include_shotmap=False, include_heatmaps=False, include_lineups=False,
        include_injuries=False, include_goal_actions=False, duration_mode="off")
    result = import_har(path, event_id=6881878, output_root=tmp_path/"out", profile=profile)
    assert result["complete"] and result["requests_sent_by_importer"] == 0
    row = read_tables(result["manifest"])["matches"][0]
    assert row["round"] == 4 and json.loads(row["stage_json"])["round"] == 0
    assert path.read_bytes() == original


def test_query_variants_and_other_match_bodies_are_ignored(tmp_path):
    other = entry("/shotmap", {"shotmap":[{"id":999}]})
    other["request"]["url"] = other["request"]["url"].replace(str(EVENT_ID), "999")
    query = entry("/shotmap", fixture("shotmap"))
    query["_queryPresent"] = True  # Companion export stripped the original query values.
    result = import_har(archive(tmp_path, [entry("", fixture("event")), other, query]),
                        event_id=EVENT_ID, output_root=tmp_path/"out")
    assert "shots" not in result["table_rows"] and not result["complete"]


def test_duplicate_archives_do_not_duplicate_lineup_snapshots(tmp_path):
    first = archive(tmp_path, full_entries())
    second = archive(tmp_path, full_entries(), "same-export.har")
    result = import_har([first, second], event_id=EVENT_ID, output_root=tmp_path/"out")
    tables = read_tables(result["manifest"])
    assert result["duplicate_observations_skipped"] == 8
    assert len(tables["lineup_players"]) == sum(len(fixture("lineups")[side]["players"]) for side in ("home", "away"))


def test_later_block_preserves_last_good_but_is_incomplete(tmp_path):
    entries = full_entries()+[entry("/shotmap", status=403, seconds=3)]
    result = import_har(archive(tmp_path, entries[::-1]), event_id=EVENT_ID, output_root=tmp_path/"out")
    assert not result["complete"]
    assert result["table_rows"]["shots"] > 0
    assert next(row["status"] for row in result["coverage"] if row["component"] == "shotmap") == "blocked"


def test_reference_remains_explicit_when_metadata_missing(tmp_path):
    reference = {"event":fixture("event")["event"], "observed_at":1700000000}
    result = import_har(archive(tmp_path, [entry("/shotmap", fixture("shotmap"))]),
                        event_id=EVENT_ID, reference=reference, output_root=tmp_path/"out")
    assert result["event_metadata_source"] == "reference_only" and not result["complete"]
    with pytest.raises(ValueError, match="no usable"):
        import_har(archive(tmp_path, []), event_id=EVENT_ID, output_root=tmp_path/"out")


def test_archived_metadata_overrides_newer_identity_only_reference(tmp_path):
    reference_event = fixture("event")["event"]
    reference_event["homeScore"]["current"] = 123
    reference = {"event":reference_event, "observed_at":2000000000}
    result = import_har(archive(tmp_path, full_entries()), event_id=EVENT_ID,
                        reference=reference, output_root=tmp_path/"out")
    match = read_tables(result["manifest"])["matches"][0]
    assert match["home_score_current"] == fixture("event")["event"]["homeScore"]["current"]
    assert match["event_metadata_observed_at"] < reference["observed_at"]


def test_mismatched_metadata_and_bad_timestamps_fail_closed(tmp_path):
    metadata = fixture("event")
    metadata["event"]["homeTeam"]["id"] = 999
    bad_time = entry("/shotmap", fixture("shotmap"))
    bad_time["startedDateTime"] = "2026-09-12T02:00:00"  # No timezone.
    result = import_har(archive(tmp_path, [entry("", metadata), bad_time]), event_id=EVENT_ID,
                        reference={"event":fixture("event")["event"], "observed_at":1700000000}, output_root=tmp_path/"out")
    assert not result["complete"]
    assert result["rejected_entries"][0]["capture"] == "invalid_timestamp"
    assert next(row["status"] for row in result["coverage"] if row["component"] == "event") == "failed"


def test_optional_roster_and_availability_exports(tmp_path):
    result = import_har(archive(tmp_path, full_entries()), event_id=EVENT_ID, output_root=tmp_path/"out",
                        profile=replace(PROFILE, include_lineups=False, include_injuries=False))
    assert result["complete"]
    assert "lineup_players" not in result["table_rows"] and "availability" not in result["table_rows"]


def test_wrong_match_metadata_only_archive_explains_empty_capture(tmp_path):
    other = entry("/lineups", status=304)
    other["request"]["url"] = other["request"]["url"].replace(str(EVENT_ID), "16434120")
    path = archive(tmp_path, [other])
    with pytest.raises(ValueError, match="observed match IDs: 16434120"):
        import_har(path, event_id=EVENT_ID, output_root=tmp_path/"out")
    result = import_har(path, event_id=EVENT_ID, output_root=tmp_path/"out",
                        reference={"event":fixture("event")["event"], "observed_at":1700000000})
    assert not result["complete"]
    assert result["capture_diagnostics"]["observed_match_ids"] == {"16434120":1}
    assert result["capture_diagnostics"]["selected_response_bodies_present"] == 0
    assert result["event_metadata_source"] == "reference_only"


def test_cached_archive_bodies_keep_http_status_and_validate_normally(tmp_path):
    entries = full_entries()
    for item in entries:
        if item["response"]["status"] == 200:
            item["response"]["status"] = 304
    result = import_har(archive(tmp_path, entries), event_id=EVENT_ID, output_root=tmp_path/"out")
    assert result["complete"]
    assert result["capture_diagnostics"]["selected_response_bodies_present"] == 7
    assert result["capture_diagnostics"]["warnings"] == []
    assert result["table_rows"]["shots"] == len(fixture("shotmap")["shotmap"])
    records = json.loads((Path(result["directory"])/"requests.json").read_text())
    saved = [row for row in records if row["capture"] == "saved"]
    assert all(row["status"] == 304 and row["body_source"] == "archived_cached_body" for row in saved)


def test_cached_status_without_body_is_not_success(tmp_path):
    entries = [entry("", fixture("event")), entry("/shotmap", status=304)]
    result = import_har(archive(tmp_path, entries), event_id=EVENT_ID, output_root=tmp_path/"out")
    assert not result["complete"]
    assert "shots" not in result["table_rows"]
    assert next(row["status"] for row in result["coverage"] if row["component"] == "shotmap") == "failed"
