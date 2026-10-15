"""Review, or requeue after collection, awarded fixtures with unobserved views.

Preview is read-only. Applying only returns selected deferred rows to pending;
the existing Chrome session collector must still obtain the missing responses.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, closing
from datetime import datetime, timezone
import json
import sqlite3
import time
from uuid import uuid4

from scraping.sofascore.config import Profile
from scraping.sofascore.export import read_tables
from scraping.sofascore.planner import component_jobs
from scraping.sofascore.store import json_bytes
from .campaign import BASE, ROOT, TERMINAL, campaign_path, collector_lock
from .publish import contained_file, hash_file, read_json
from .status_write import atomic_write_status

REASON = "awarded_fixture_without_requested_website_views"


def _review(directory, event_ids, connection):
    config_path = directory / "config.json"
    config_hash = hash_file(config_path)
    config = read_json(config_path)
    if config["policy"]["collection_mode"] != "session" or config["profile"]["include_comments"]:
        raise ValueError("Use the existing session collector with commentary disabled")
    items = {item["event_id"]: item for item in config["items"]}
    rows = {row["event_id"]: dict(row) for row in connection.execute("SELECT * FROM queue")}
    if set(items) != set(rows):
        raise ValueError("Saved queue and configured fixture inventory differ")
    selected = []
    for eid in event_ids:
        row, item = rows.get(eid), items.get(eid)
        if row is None or row["state"] != "deferred" or row["reason"] != REASON:
            raise ValueError(f"Fixture {eid} is not an awarded fixture deferred for missing views")
        report_path = contained_file(row["report"], directory / "imports")
        report_hash = hash_file(report_path)
        report = read_json(report_path)
        manifest_path = contained_file(report["manifest"], report_path.parent)
        manifest = read_json(manifest_path)
        for table in manifest["tables"].values():
            contained_file(manifest_path.parent / table["file"], report_path.parent)
        if report.get("complete") is not False or manifest.get("profile") != config["profile"]:
            raise ValueError("Deferred report completion or collection profile changed")
        reference = item["reference"]["event"]
        expected = {job.path for job in component_jobs(
            {**reference, "status": {"type": "finished"}}, Profile(**config["profile"]), time.time())}
        expected.add(f"/event/{eid}")
        coverage = report["coverage"]
        if len(coverage) != len(expected) or {c["path"] for c in coverage} != expected:
            raise ValueError("Saved report does not account for the configured component paths")
        missing = [c for c in coverage if c["status"] not in TERMINAL]
        if not missing or any(c["status"] != "not_observed" for c in missing):
            raise ValueError("Only unobserved deferred components may be requeued")
        tables = read_tables(manifest_path, verify_hashes=True)
        matches = tables.get("matches", [])
        identity = (eid, item["competition_id"], item["season_id"], reference["homeTeam"]["id"], reference["awayTeam"]["id"])
        fields = ("event_id", "competition_id", "season_id", "home_id", "away_id")
        if len(matches) != 1 or tuple(matches[0].get(k) for k in fields) != identity:
            raise ValueError("Deferred capture fixture identity mismatch")
        if any(record.get("event_id", eid) != eid for records in tables.values() for record in records):
            raise ValueError("Deferred capture contains another fixture's rows")
        selected.append({"event_id": eid, "prior_queue_row": row, "report_sha256": report_hash,
                         "manifest_sha256": hash_file(manifest_path),
                         "remaining_paths_in_saved_report": [c["path"] for c in missing],
                         "accounted_paths_in_saved_report": [c["path"] for c in coverage if c["status"] in TERMINAL]})
    ordinary = sum(row["state"] not in {"complete", "deferred"} for row in rows.values())
    return {"recorded_at": datetime.now(timezone.utc).isoformat(), "state": "preview",
            "campaign_id": config["campaign_id"], "config_sha256": config_hash,
            "ordinary_work_remaining": ordinary, "selected": selected,
            "scope": "No provider requests. Saved components and reports are preserved. Requeueing is not collection completion; the session collector evaluates current observations on resume."}


def retry_deferred(campaign_id, event_ids, *, apply=False, base=BASE, lock_path=None):
    ids = list(event_ids)
    if not ids or any(type(eid) is not int or eid <= 0 for eid in ids) or len(ids) != len(set(ids)):
        raise ValueError("Select distinct positive integer event IDs")
    directory = campaign_path(campaign_id, base)
    uri = (directory / "queue.sqlite").as_uri()
    with ExitStack() as stack:
        if apply:
            try:
                stack.enter_context(collector_lock(lock_path or ROOT / "data/collection_state/collector.lock"))
            except OSError as error:
                raise RuntimeError("Disconnect the collector before applying deferred retries") from error
        connection = stack.enter_context(closing(sqlite3.connect(uri + ("?mode=rw" if apply else "?mode=ro"), uri=True)))
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN IMMEDIATE" if apply else "BEGIN")
        try:
            result = _review(directory, ids, connection)
            if not apply:
                return result
            if result["ordinary_work_remaining"]:
                raise ValueError("Finish all ordinary queue work before applying deferred retries")
            receipt = directory / "retries" / f"{uuid4().hex}.json"
            result.update(state="prepared", receipt=str(receipt))
            atomic_write_status(receipt, json_bytes(result))
            if hash_file(directory / "config.json") != result["config_sha256"]:
                raise ValueError("Campaign configuration changed during retry preparation")
            for selected in result["selected"]:
                old = selected["prior_queue_row"]
                changed = connection.execute(
                    "UPDATE queue SET state='pending',lease=NULL,finished=NULL,reason='session_retry_requested' "
                    "WHERE event_id=? AND state='deferred' AND reason=? AND report=?",
                    (selected["event_id"], REASON, old["report"])).rowcount
                if changed != 1:
                    raise ValueError("Deferred queue row changed during retry preparation")
            connection.commit()
        finally:
            if connection.in_transaction:
                connection.rollback()
        result.update(state="requeued", completed_at=datetime.now(timezone.utc).isoformat())
        try:
            atomic_write_status(receipt, json_bytes(result))
        except OSError as error:
            raise RuntimeError(f"Queue was requeued but its receipt update failed; inspect {receipt} before retrying") from error
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--event-id", type=int, action="append", required=True)
    parser.add_argument("--apply", action="store_true", help="Requeue only after ordinary collection finishes and the writer disconnects")
    args = parser.parse_args()
    print(json.dumps(retry_deferred(args.campaign_id, args.event_id, apply=args.apply), indent=2))


if __name__ == "__main__":
    main()
