"""Deferred retries preserve source evidence and cannot interrupt collection."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
import sqlite3

import pytest

from scraping.sofascore_web_v3.campaign import Campaign, collector_lock, prepare_campaign, update_policy
from scraping.sofascore_web_v3.retry_deferred import retry_deferred
import scraping.sofascore_web_v3.retry_deferred as retry_module

EID = 12436870
FIXTURES = Path(__file__).parents[1] / "fixtures/sofascore"


def payload(name):
    return json.loads((FIXTURES / f"{EID}_{name}.json").read_text(encoding="utf-8"))


def observation(suffix, body=None, code=200, second=0):
    return {"startedDateTime": f"2026-09-12T02:00:{second:02d}Z", "time": 10,
            "request": {"method": "GET", "url": f"https://www.sofascore.com/api/v1/event/{EID}{suffix}"},
            "response": {"status": code, "content": {"text": json.dumps(body)} if body is not None else {}}}


@pytest.fixture
def deferred(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("Deferred retry must not make network connections")
    monkeypatch.setattr(socket, "create_connection", no_network)
    base, lock = tmp_path / "campaigns", tmp_path / "collector.lock"
    event = deepcopy(payload("event"))
    event["event"]["isAwarded"] = True
    config = prepare_campaign([EID], base=base, references={EID: {"event": event["event"], "observed_at": 1700000000}})
    directory = Path(config["directory"])
    with Campaign(directory) as campaign:
        job = campaign.next_match()
        for entry in [observation("", event), observation("/pregame-form", code=404),
                      observation("/statistics", code=404), observation("/lineups", code=404),
                      observation("/incidents", payload("incidents"))]:
            campaign.observe(EID, job["lease"], entry)
        result = campaign.finish(EID, job["lease"], page_event_id=EID,
            page_evidence={"available_views": ["standings", "matches"], "recovery_refreshed": True})
        assert result["state"] == "deferred"
    update_policy(config["campaign_id"], {"collection_mode": "session"}, base=base, lock_path=lock)
    return config["campaign_id"], base, lock, directory


def queue_snapshot(directory):
    with sqlite3.connect(directory / "queue.sqlite") as connection:
        return (connection.execute("SELECT * FROM queue ORDER BY event_id").fetchall(),
                connection.execute("SELECT * FROM observations ORDER BY digest").fetchall())


def test_preview_is_read_only_while_writer_lock_is_held(deferred):
    cid, base, lock, directory = deferred
    before = queue_snapshot(directory)
    with collector_lock(lock):
        result = retry_deferred(cid, [EID], base=base, lock_path=lock)
    assert result["state"] == "preview"
    assert len(result["selected"][0]["remaining_paths_in_saved_report"]) == 3
    assert len(result["selected"][0]["accounted_paths_in_saved_report"]) == 5
    assert queue_snapshot(directory) == before
    assert not (directory / "retries").exists()


def test_apply_refuses_connected_writer(deferred):
    cid, base, lock, directory = deferred
    before = queue_snapshot(directory)
    with collector_lock(lock), pytest.raises(RuntimeError, match="Disconnect"):
        retry_deferred(cid, [EID], apply=True, base=base, lock_path=lock)
    assert queue_snapshot(directory) == before


def test_apply_refuses_ordinary_work_and_keeps_all_rows(deferred):
    cid, base, lock, directory = deferred
    path = directory / "config.json"
    config = json.loads(path.read_text())
    extra = deepcopy(config["items"][0]); extra["event_id"] = EID + 1
    config["items"].append(extra); path.write_text(json.dumps(config))
    with sqlite3.connect(directory / "queue.sqlite") as connection:
        connection.execute("INSERT INTO queue VALUES (?,1,'pending',NULL,NULL,NULL,NULL)", (EID + 1,))
    before = queue_snapshot(directory)
    assert retry_deferred(cid, [EID], base=base)["ordinary_work_remaining"] == 1
    with pytest.raises(ValueError, match="ordinary queue work"):
        retry_deferred(cid, [EID], apply=True, base=base, lock_path=lock)
    assert queue_snapshot(directory) == before
    assert not (directory / "retries").exists()


def test_requeue_reuses_five_components_and_requires_three_new_outcomes(deferred):
    cid, base, lock, directory = deferred
    before = queue_snapshot(directory)
    report_path = Path(before[0][0][4]); original = report_path.read_bytes()
    result = retry_deferred(cid, [EID], apply=True, base=base, lock_path=lock)
    after = queue_snapshot(directory)
    assert after[1] == before[1]
    assert after[0][0][2] == "pending" and after[0][0][3] is None and after[0][0][6] is None
    assert after[0][0][4] == str(report_path) and report_path.read_bytes() == original
    receipt = json.loads(Path(result["receipt"]).read_text())
    assert receipt["state"] == "requeued"
    assert receipt["selected"][0]["report_sha256"] == hashlib.sha256(original).hexdigest()
    with pytest.raises(ValueError, match="not an awarded fixture deferred"):
        retry_deferred(cid, [EID], apply=True, base=base, lock_path=lock)
    with Campaign(directory) as campaign:
        job = campaign.next_match(resume=True)
        pending = [c for c in job["coverage"]["components"] if c["status"] == "not_observed"]
        assert len(pending) == 3 and not job["coverage"]["complete"]
        for c in pending:
            entry = observation(c["path"].removeprefix(f"/event/{EID}"), code=404, second=5)
            entry["_observationSource"] = "chrome_session_request"
            campaign.observe(EID, job["lease"], entry)
        finished = campaign.finish(EID, job["lease"], page_event_id=None,
            page_evidence={"collection_mode": "session", "session_request_count": 3})
        assert finished["state"] == "complete"
        report = json.loads(Path(finished["report"]).read_text())
        assert report["complete"] and all(c["status"] in {"success", "unavailable"} for c in report["coverage"])
    assert report_path.read_bytes() == original


def test_corrupt_capture_refuses_requeue(deferred):
    cid, base, lock, directory = deferred
    before = queue_snapshot(directory)
    report = json.loads(Path(before[0][0][4]).read_text())
    manifest_path = Path(report["manifest"])
    manifest = json.loads(manifest_path.read_text())
    table = manifest_path.parent / next(iter(manifest["tables"].values()))["file"]
    table.write_bytes(b"damaged")
    with pytest.raises(ValueError):
        retry_deferred(cid, [EID], apply=True, base=base, lock_path=lock)
    assert queue_snapshot(directory) == before
    assert not (directory / "retries").exists()


@pytest.mark.parametrize("ids", [[], [True], [-1], [EID, EID], [EID + 1], [EID, EID + 1]])
def test_invalid_selection_never_changes_queue(deferred, ids):
    cid, base, lock, directory = deferred
    before = queue_snapshot(directory)
    with pytest.raises(ValueError):
        retry_deferred(cid, ids, apply=True, base=base, lock_path=lock)
    assert queue_snapshot(directory) == before


def test_intent_receipt_failure_prevents_queue_mutation(deferred, monkeypatch):
    cid, base, lock, directory = deferred
    before = queue_snapshot(directory)
    def fail(*args, **kwargs):
        raise OSError("receipt unavailable")
    monkeypatch.setattr(retry_module, "atomic_write_status", fail)
    with pytest.raises(OSError, match="receipt unavailable"):
        retry_deferred(cid, [EID], apply=True, base=base, lock_path=lock)
    assert queue_snapshot(directory) == before


def test_database_failure_rolls_back_and_preserves_prepared_receipt(deferred):
    cid, base, lock, directory = deferred
    with sqlite3.connect(directory / "queue.sqlite") as connection:
        connection.execute("CREATE TRIGGER refuse_retry BEFORE UPDATE ON queue BEGIN SELECT RAISE(ABORT,'retry refused'); END")
    before = queue_snapshot(directory)
    with pytest.raises(sqlite3.IntegrityError, match="retry refused"):
        retry_deferred(cid, [EID], apply=True, base=base, lock_path=lock)
    assert queue_snapshot(directory) == before
    receipts = list((directory / "retries").glob("*.json"))
    assert len(receipts) == 1 and json.loads(receipts[0].read_text())["state"] == "prepared"


def test_receipt_failure_after_commit_reports_that_queue_was_requeued(deferred, monkeypatch):
    cid, base, lock, directory = deferred
    original = retry_module.atomic_write_status
    calls = []
    def fail_second(path, data):
        calls.append(path)
        if len(calls) == 2:
            raise OSError("receipt update unavailable")
        original(path, data)
    monkeypatch.setattr(retry_module, "atomic_write_status", fail_second)
    with pytest.raises(RuntimeError, match="Queue was requeued"):
        retry_deferred(cid, [EID], apply=True, base=base, lock_path=lock)
    assert queue_snapshot(directory)[0][0][2] == "pending"
    assert json.loads(calls[0].read_text())["state"] == "prepared"
