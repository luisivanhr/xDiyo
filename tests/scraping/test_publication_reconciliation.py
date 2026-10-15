"""Replacing a completed season must preserve both bundles and survive failure."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import socket
import sqlite3

import pytest

from scraping.sofascore.campaign import publish_season
from scraping.sofascore.config import Scope
from scraping.sofascore.export import read_tables
from scraping.sofascore_web_v3.campaign import Campaign, collector_lock, prepare_campaign
from scraping.sofascore_web_v3.capture import PROFILE
from scraping.sofascore_web_v3.publish import hash_file
from scraping.sofascore_web_v3.reconcile_publication import reconcile_publication, recover_reconciliation
import scraping.sofascore_web_v3.reconcile_publication as module

EID = 12436870
STEM = "Premier_League_24_25"
FIXTURES = Path(__file__).parents[1] / "fixtures/sofascore"


def payload(name):
    return json.loads((FIXTURES / f"{EID}_{name}.json").read_text(encoding="utf-8"))


@pytest.fixture
def publication(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("Publication reconciliation must stay offline")
    monkeypatch.setattr(socket, "create_connection", no_network)
    event = payload("event")["event"]
    reference = {"event": event, "observed_at": 1700000000, "season_stem": STEM}
    info = prepare_campaign([EID], base=tmp_path / "campaigns", policy={"collection_mode": "session"},
                            references={EID: reference})
    directory = Path(info["directory"])
    components = [("", payload("event")), ("/pregame-form", None),
                  ("/statistics", {"statistics": []}), ("/shotmap", payload("shotmap")),
                  ("/lineups", payload("lineups")), ("/incidents", payload("incidents"))]
    components += [(f"/heatmap/{event[side]['id']}", {"playerPoints": [{"x": 1, "y": 2}],
                    "goalkeeperPoints": []}) for side in ("homeTeam", "awayTeam")]
    with Campaign(directory) as campaign:
        job = campaign.next_match()
        for second, (suffix, body) in enumerate(components):
            entry = {"startedDateTime": f"2026-09-12T02:00:{second:02d}Z", "time": 10,
                     "_observationSource": "chrome_session_request",
                     "request": {"method": "GET", "url": f"https://www.sofascore.com/api/v1/event/{EID}{suffix}"},
                     "response": {"status": 200 if body is not None else 404,
                                  "content": {"text": json.dumps(body)} if body is not None else {}}}
            campaign.observe(EID, job["lease"], entry)
        result = campaign.finish(EID, job["lease"], page_event_id=None,
            page_evidence={"collection_mode": "session", "session_request_count": 8})
        assert result["state"] == "complete"
    source = json.loads(Path(result["report"]).read_text())
    tables = read_tables(source["manifest"], verify_hashes=True)
    old = deepcopy(tables)
    old["timing"] = []
    output = tmp_path / "output"
    scope = Scope(event["tournament"]["uniqueTournament"]["id"], event["season"]["id"],
                  tuple({m["round"] for m in tables["matches"]}), "Premier_League", 2024, 2025)
    report = publish_season(old, output, scope, replace(PROFILE, duration_mode="metadata"), STEM,
                            {"collection": "direct_capture_before_chrome"})
    return {"directory": directory, "output": output, "lock_path": tmp_path / "collector.lock",
            "old_sha": report["sha256"], "tables": tables, "old_tables": old}


def invoke(data, **kwargs):
    return reconcile_publication(data["directory"], STEM, output=data["output"],
        lock_path=data["lock_path"], expected_sha256=data["old_sha"], **kwargs)


def file_hashes(root):
    # Windows intentionally denies reading the byte held by an active OS lock.
    return {p.relative_to(root).as_posix(): hash_file(p) for p in root.rglob("*")
            if p.is_file() and p.suffix != ".lock"}


def queue_rows(directory):
    with sqlite3.connect(directory / "queue.sqlite") as db:
        return (db.execute("SELECT * FROM queue ORDER BY event_id").fetchall(),
                db.execute("SELECT * FROM observations ORDER BY digest").fetchall())


def old_preserved(data, before):
    for relative, digest in before.items():
        assert hash_file(data["output"] / relative) == digest, relative


def receipt_path(data):
    receipts = list((data["output"] / "_collection/chrome_reconciliations").rglob("receipt.json"))
    assert len(receipts) == 1
    return receipts[0]


def test_preview_is_read_only_with_both_live_locks(publication):
    d = publication
    queue_before = queue_rows(d["directory"])
    with collector_lock(d["lock_path"]), collector_lock(d["output"] / "_collection/chrome_publication.lock"):
        before = file_hashes(d["output"])
        result = invoke(d)
        assert result["state"] == "preview" and result["ready"]
        assert result["prior"]["profile"]["duration_mode"] == "metadata"
        assert result["requested_profile"]["duration_mode"] == "incidents"
        assert file_hashes(d["output"]) == before
    assert queue_rows(d["directory"]) == queue_before


@pytest.mark.parametrize("which", ["collector", "publisher"])
def test_apply_refuses_live_locks(publication, which):
    d = publication
    path = d["lock_path"] if which == "collector" else d["output"] / "_collection/chrome_publication.lock"
    with collector_lock(path):
        before = file_hashes(d["output"])
        with pytest.raises(RuntimeError, match="Disconnect"):
            invoke(d, apply=True)
        old_preserved(d, before)
    assert not (d["output"] / "_collection/chrome_reconciliations").exists()


@pytest.mark.parametrize("state", ["pending", "running", "deferred"])
def test_unfinished_target_refuses_replacement(publication, state):
    d = publication
    with sqlite3.connect(d["directory"] / "queue.sqlite") as db:
        db.execute("UPDATE queue SET state=?", (state,))
    before = file_hashes(d["output"])
    assert not invoke(d)["ready"]
    with pytest.raises(ValueError, match="entire queue"):
        invoke(d, apply=True)
    old_preserved(d, before)
    assert not (d["output"] / "_collection/chrome_reconciliations").exists()


def test_other_unfinished_season_also_blocks(publication):
    d = publication
    path = d["directory"] / "config.json"
    config = json.loads(path.read_text())
    item = deepcopy(config["items"][0]); item["event_id"] += 1
    item["reference"]["season_stem"] = "Other_24_25"
    config["items"].append(item)
    path.write_text(json.dumps(config))
    with sqlite3.connect(d["directory"] / "queue.sqlite") as db:
        db.execute("INSERT INTO queue(event_id,ordinal,state) VALUES (?,2,'pending')", (EID + 1,))
    result = invoke(d)
    assert result["unfinished_target_matches"] == {} and result["unfinished_queue_matches"] == 1
    with pytest.raises(ValueError, match="entire queue"):
        invoke(d, apply=True)


def test_reviewed_hash_is_required(publication):
    d = publication
    before = file_hashes(d["output"])
    with pytest.raises(ValueError, match="reviewed existing season SHA-256"):
        reconcile_publication(d["directory"], STEM, output=d["output"], apply=True,
                              expected_sha256="0" * 64, lock_path=d["lock_path"])
    old_preserved(d, before)


def test_publish_preserves_old_bundle_sources_and_is_idempotent(publication):
    d = publication
    queue_before = queue_rows(d["directory"])
    old_files = file_hashes(d["output"])
    source_files = file_hashes(d["directory"] / "imports")
    result = invoke(d, apply=True)
    assert result["state"] == "published"
    run = Path(result["receipt"]).parent
    for relative, digest in old_files.items():
        assert hash_file(run / "prior" / relative) == digest
    old_manifest = json.loads((run / "prior" / f"{STEM}.manifest.json").read_text())
    new_manifest = json.loads((d["output"] / f"{STEM}.manifest.json").read_text())
    assert read_tables(run / "prior" / old_manifest["manifest"], verify_hashes=True) == d["old_tables"]
    assert read_tables(d["output"] / new_manifest["manifest"], verify_hashes=True) == d["tables"]
    assert hash_file(d["output"] / old_manifest["manifest"]) == old_files[old_manifest["manifest"]]
    assert queue_rows(d["directory"]) == queue_before
    assert file_hashes(d["directory"] / "imports") == source_files
    installed = file_hashes(d["output"])
    assert invoke(d, apply=True)["state"] == "already_published"
    assert file_hashes(d["output"]) == installed


@pytest.mark.parametrize("damage", ["parquet", "table", "current", "escape"])
def test_corrupt_or_unexplained_prior_bundle_is_not_replaced(publication, damage):
    d = publication
    outer_path = d["output"] / f"{STEM}.manifest.json"
    outer = json.loads(outer_path.read_text())
    canonical = d["output"] / outer["manifest"]
    if damage == "parquet":
        (d["output"] / f"{STEM}.parquet").write_bytes(b"damaged")
    elif damage == "table":
        table = next(iter(json.loads(canonical.read_text())["tables"].values()))
        (canonical.parent / table["file"]).write_bytes(b"damaged")
    elif damage == "current":
        (canonical.parents[2] / "CURRENT.json").write_text('{"version":"different"}')
    else:
        outer["manifest"] = "../outside/manifest.json"
        outer_path.write_text(json.dumps(outer))
    before = file_hashes(d["output"])
    with pytest.raises(ValueError):
        invoke(d, apply=True)
    old_preserved(d, before)


def test_bad_capture_fails_before_replacing_prior(publication):
    d = publication
    with sqlite3.connect(d["directory"] / "queue.sqlite") as db:
        report = json.loads(Path(db.execute("SELECT report FROM queue").fetchone()[0]).read_text())
    source_manifest = Path(report["manifest"])
    table = next(iter(json.loads(source_manifest.read_text())["tables"].values()))
    (source_manifest.parent / table["file"]).write_bytes(b"damaged capture")
    before = file_hashes(d["output"])
    with pytest.raises(ValueError, match="hash mismatch"):
        invoke(d, apply=True)
    old_preserved(d, before)


def test_entry_failure_rolls_back_all_pointers_and_allows_retry(publication, monkeypatch):
    d = publication
    before = file_hashes(d["output"])
    original = module.atomic_write_status
    failed = []
    def fail_once(path, data):
        if Path(path) == d["output"] / f"{STEM}.manifest.json" and not failed:
            failed.append(True)
            raise OSError("simulated sharing conflict")
        return original(path, data)
    monkeypatch.setattr(module, "atomic_write_status", fail_once)
    with pytest.raises(RuntimeError, match="prior season was restored"):
        invoke(d, apply=True)
    old_preserved(d, before)
    assert json.loads(receipt_path(d).read_text())["state"] == "rolled_back"
    assert invoke(d, apply=True)["state"] == "published"


def test_receipt_failure_after_commit_recovers_as_published(publication, monkeypatch):
    d = publication
    original = module.atomic_write_status
    def fail_published_receipt(path, data):
        if Path(path).name == "receipt.json" and json.loads(data).get("state") == "published":
            raise OSError("receipt is locked")
        return original(path, data)
    monkeypatch.setattr(module, "atomic_write_status", fail_published_receipt)
    with pytest.raises(RuntimeError, match="Replacement is published"):
        invoke(d, apply=True)
    receipt = receipt_path(d)
    assert json.loads(receipt.read_text())["state"] == "prepared"
    new_hash = hash_file(d["output"] / f"{STEM}.parquet")
    assert new_hash != d["old_sha"]
    monkeypatch.setattr(module, "atomic_write_status", original)
    result = recover_reconciliation(receipt, output=d["output"], lock_path=d["lock_path"])
    assert result["state"] == "published"
    assert hash_file(d["output"] / f"{STEM}.parquet") == new_hash


@pytest.mark.parametrize("tamper", [False, True])
def test_interrupted_commit_recovers_or_refuses_unexplained_change(publication, monkeypatch, tamper):
    d = publication
    before = file_hashes(d["output"])
    original = module.atomic_write_status
    writes = []
    def fail_commit_and_rollback(path, data):
        path = Path(path)
        if path.parent == d["output"]:
            writes.append(path)
            if len(writes) in (2, 3):
                raise OSError("simulated interrupted replacement")
        return original(path, data)
    monkeypatch.setattr(module, "atomic_write_status", fail_commit_and_rollback)
    with pytest.raises(RuntimeError, match="needs recovery"):
        invoke(d, apply=True)
    receipt = receipt_path(d)
    assert hash_file(d["output"] / f"{STEM}.parquet") != d["old_sha"]
    monkeypatch.setattr(module, "atomic_write_status", original)
    if tamper:
        (d["output"] / f"{STEM}.manifest.json").write_text('{"unexplained":"edit"}')
        mixed = file_hashes(d["output"])
        with pytest.raises(ValueError, match="Unexpected entry-file change"):
            recover_reconciliation(receipt, output=d["output"], lock_path=d["lock_path"])
        assert file_hashes(d["output"]) == mixed
    else:
        assert recover_reconciliation(receipt, output=d["output"], lock_path=d["lock_path"])["state"] == "rolled_back"
        old_preserved(d, before)
