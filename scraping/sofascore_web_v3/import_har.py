"""Import already-received Chrome Network responses. Never launches a browser or sends HTTP."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
from urllib.parse import urlsplit
import uuid

from scraping.sofascore.config import Scope
from scraping.sofascore.discovery import event_status, is_completed_event
from scraping.sofascore.event_round import match_round, match_season
from scraping.sofascore.export import normalize_store, publish_tables
from scraping.sofascore.models import Job
from scraping.sofascore.parsers import validate_payload
from scraping.sofascore.planner import component_jobs
from scraping.sofascore.store import Store, atomic_write, json_bytes, season_year_pair
from .capture import ROOT, PROFILE, api_path


def response_time(entry):
    """Approximate response completion from HAR request start plus elapsed milliseconds."""
    value = datetime.fromisoformat(entry["startedDateTime"].replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("HAR timestamps must include a timezone")
    elapsed = entry.get("time", 0)
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed):
        raise ValueError("HAR elapsed time must be finite")
    return value.timestamp() + max(0, elapsed)/1000


def response_json(entry):
    content = entry.get("response", {}).get("content", {})
    body = content.get("text")
    if body is None or body == "":
        raise ValueError("response_body_missing")
    encoding = content.get("encoding")
    if encoding == "base64":
        body = base64.b64decode(body, validate=True).decode("utf-8-sig")
    elif encoding not in (None, ""):
        raise ValueError("unsupported_body_encoding")
    return json.loads(body)


def _check_event(event, reference, *, allow_unfinished=False):
    validate_payload("event", {"event":event})
    accepted = (event_status(event) in {"finished", "notstarted", "postponed", "inprogress", "interrupted",
                                       "cancelled", "abandoned", "awarded"}) if allow_unfinished else (
                                           is_completed_event(event))
    if event["id"] != reference["id"] or not accepted:
        raise ValueError("event_identity_or_status_mismatch")
    for key in ("homeTeam", "awayTeam"): 
        if event.get(key, {}).get("id") != reference[key]["id"]:
            raise ValueError("event_context_mismatch")
    if not match_season(event).get("id") or match_season(event).get("id") != match_season(reference).get("id"):
        raise ValueError("event_context_mismatch")
    if event.get("tournament", {}).get("uniqueTournament", {}).get("id") != reference["tournament"]["uniqueTournament"]["id"]:
        raise ValueError("competition_mismatch")


def archive_diagnostics(observations, event_id):
    """Report mismatched/empty recordings without treating fixture identity as capture."""
    observed_ids = Counter()
    selected = []
    pattern = re.compile(rf"/event/{event_id}(?:/(?:pregame-form|statistics|shotmap|lineups|incidents)|/heatmap/\d+)?")
    for record, entry in observations:
        match = re.match(r"/event/(\d+)(?:/|$)", record["path"])
        if match:
            observed_ids[match[1]] += 1
        if pattern.fullmatch(record["path"]) and record["method"] == "GET" and not record["query_present"]:
            selected.append((record, entry))
    bodies = sum(record["status"] in (200, 304) and
                 isinstance(entry.get("response", {}).get("content", {}).get("text"), str) and
                 bool(entry["response"]["content"]["text"]) for record, entry in selected)
    warnings = []
    if observed_ids and str(event_id) not in observed_ids:
        warnings.append(f"No requests for selected match {event_id}; observed match IDs: {', '.join(sorted(observed_ids))}.")
    if not bodies:
        warnings.append("No selected component response bodies present. Reference-derived rows are not captured match data.")
    return {"observed_match_ids": dict(observed_ids), "selected_component_entries": len(selected),
            "selected_response_bodies_present": bodies, "warnings": warnings}


def import_har(paths, *, event_id, reference=None, output_root=None, profile=PROFILE):
    """Accept Chrome sanitized HAR or the companion extension's filtered HAR.

    The original archives may contain unrelated data. Only selected endpoint
    bodies and sanitized request metadata are persisted in this import's folder.
    Multiple archives are merged chronologically; identical observations are
    deduplicated. Existing scraper stores and network ledgers are never opened.
    """
    if isinstance(event_id, bool) or not isinstance(event_id, int) or event_id <= 0:
        raise ValueError("Select one positive integer event ID")
    if profile.include_comments:
        raise ValueError("This importer does not retain commentary")
    if isinstance(paths, (str, Path)):
        paths = [paths]
    if not paths:
        raise ValueError("Supply at least one HAR file")
    observations, sources, rejected = [], [], []
    seen = set()
    duplicates = 0
    for path in map(Path, paths):
        if path.stat().st_size > 250*1024*1024:
            raise ValueError("HAR file exceeds 250 MiB; export a narrower Network selection")
        body = path.read_bytes()
        document = json.loads(body.decode("utf-8-sig"))
        entries = document.get("log", {}).get("entries")
        if not isinstance(entries, list):
            raise ValueError(f"Not a HAR archive: {path.name}")
        sources.append({"file":path.name, "sha256":hashlib.sha256(body).hexdigest(), "entries":len(entries)})
        for index, entry in enumerate(entries):
            request = entry.get("request", {})
            url = request.get("url", "")
            endpoint = api_path(url)
            if endpoint is None:
                continue
            query = bool(urlsplit(url).query or request.get("queryString") or entry.get("_queryPresent"))
            record = {"path":endpoint, "method":request.get("method"), "query_present":query,
                      "status":entry.get("response", {}).get("status"), "source_file":path.name,
                      "entry_index":index}
            try:
                record["observed_at"] = response_time(entry)
            except (KeyError, TypeError, ValueError, OverflowError):
                record["capture"] = "invalid_timestamp"
                rejected.append(record)
                continue
            # Do not retain cookies, headers, request bodies, query values, or full URLs.
            fingerprint = hashlib.sha256(json_bytes({
                "path":endpoint, "method":record["method"], "query":query,
                "observed_at":record["observed_at"], "status":record["status"],
                "content":entry.get("response", {}).get("content", {})})).hexdigest()
            if fingerprint in seen:
                duplicates += 1
                continue
            seen.add(fingerprint)
            observations.append((record, entry))
    observations.sort(key=lambda pair:pair[0]["observed_at"])
    diagnostics = archive_diagnostics(observations, event_id)
    metadata_path = f"/event/{event_id}"
    if reference is None:
        for record, entry in observations:
            if record["path"] == metadata_path and record["method"] == "GET" and not record["query_present"] and record["status"] in (200, 304):
                try:
                    candidate = response_json(entry)["event"]
                    if candidate.get("id") == event_id:
                        reference = {"event":candidate, "observed_at":record["observed_at"],
                                     "source":"selected event response in supplied HAR"}
                        break
                except (KeyError, TypeError, ValueError, UnicodeError):
                    continue
    if reference is None:
        raise ValueError("HAR has no usable selected event metadata; supply reference_from_cache(event_id). " + " ".join(diagnostics["warnings"]))
    event = reference["event"]
    if event.get("id") != event_id or (not profile.include_upcoming and not is_completed_event(event)):
        raise ValueError("Reference must identify the selected completed match")
    _check_event(event, event, allow_unfinished=profile.include_upcoming)
    years = season_year_pair(match_season(event)["year"])
    if years is None:
        raise ValueError("Reference needs an unambiguous season year pair")
    scope = Scope(event["tournament"]["uniqueTournament"]["id"], match_season(event)["id"],
                  (match_round(event),), season_start=years[0], season_end=years[1])
    metadata_job = Job(metadata_path, "event", event_id)
    planning_time = reference["observed_at"] if profile.include_upcoming else time.time()
    jobs = {job.path:job for job in component_jobs(event, profile, planning_time)}
    jobs[metadata_path] = metadata_job
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = Path(output_root or ROOT/"data/website_capture_v3") / f"{stamp}-har-{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True)
    atomic_write(directory/"reference.json", json_bytes(reference))
    imported_metadata = False
    records = []
    with Store(directory/"captured") as store:
        store.add_events([event], scope, metadata_job, reference["observed_at"])
        for record, entry in observations:
            records.append(record)
            job = jobs.get(record["path"])
            if job is None or record["method"] != "GET" or record["query_present"]:
                record["capture"] = "not_selected"
                continue
            status = record["status"]
            observed = record["observed_at"]
            if status not in (200, 304):
                state = "unavailable" if status == 404 else "blocked" if status in (401, 403) else "rate_limited" if status == 429 else "failed"
                store.record(job, state, observed, http_status=status)
                record["capture"] = state
                continue
            try:
                payload = response_json(entry)
                empty = validate_payload(job.component, payload)
                if job.component == "event":
                    _check_event(payload["event"], reference["event"], allow_unfinished=profile.include_upcoming)
                    store.add_events([payload["event"]], scope, job, observed)
                    # The selected archive is authoritative for this import even
                    # when the identity-only cache reference was observed later.
                    with store.db:
                        store.db.execute("UPDATE events SET payload=?,observed=? WHERE event_id=?",
                                         (json_bytes(payload["event"]).decode(), observed, event_id))
                    event = payload["event"]
                    imported_metadata = True
                store.record(job, "empty" if empty else "success", observed, payload=payload,
                             http_status=status, event_context=event)
                record["capture"] = "saved"
                record["body_source"] = "archived_cached_body" if status == 304 else "archived_response_body"
                record["raw_hash"] = store.get(job)["raw_hash"]
            except (KeyError, TypeError, ValueError, UnicodeError) as exc:
                missing = entry.get("response", {}).get("content", {}).get("text") in (None, "")
                record["capture"] = "response_body_missing" if missing else "body_or_schema_error"
                record["error_type"] = type(exc).__name__
                store.record(job, "failed", observed, http_status=status, error=record["capture"])
        coverage = []
        for job in jobs.values():
            state = store.get(job)
            coverage.append({"component":job.component, "team_id":job.team_id, "path":job.path,
                             "status":state["status"] if state else "not_observed",
                             "raw_hash":state["raw_hash"] if state else None})
        metadata_state = store.get(metadata_job)
        complete = (imported_metadata and metadata_state["status"] == "success" and
                    all(row["status"] in {"success", "empty", "unavailable"} for row in coverage) and
                    not any(row["path"] in jobs for row in rejected))
        tables = normalize_store(store, scope, profile, now=planning_time)
        tables.setdefault("coverage", []).append({"event_id":event_id, "component":"event",
            "status":metadata_state["status"] if metadata_state else "pending",
            "latest_attempt_status":metadata_state["status"] if metadata_state else "pending"})
        metadata_source = "captured_network_export" if imported_metadata else "reference_only"
        manifest = publish_tables(tables, directory/"prepared", scope=scope, profile=profile,
            provenance={"artifact_kind":"single_match_network_archive_import", "selected_event_id":event_id,
                        "event_metadata_source":metadata_source, "network_archive_coverage_complete":complete,
                        "source_archives":sources, "imported_at":time.time(),
                        "observed_at_semantics":"HAR request start plus nonnegative elapsed milliseconds"})
    report = {"experiment":"sofascore_web_v3_har", "event_id":event_id, "directory":str(directory.resolve()),
              "closed":True, "complete":complete, "manifest":str(manifest), "event_metadata_source":metadata_source,
              "coverage":coverage, "table_rows":{name:len(rows) for name,rows in tables.items()},
              "source_archives":sources, "observed_api_entries":len(records), "duplicate_observations_skipped":duplicates,
              "capture_diagnostics":diagnostics,
              "rejected_entries":rejected, "http_status_counts":dict(Counter(str(r["status"]) for r in records)),
              "requests_sent_by_importer":0,
              "accounting_note":"Archived observations only; live request ledger untouched. A filtered or incomplete HAR is not a full browser request count."}
    atomic_write(directory/"requests.json", json_bytes(records))
    atomic_write(directory/"report.json", json_bytes(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--har", type=Path, nargs="+", required=True)
    parser.add_argument("--event-id", type=int, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8")) if args.reference else None
    print(json.dumps(import_har(args.har, event_id=args.event_id, reference=reference, output_root=args.output), indent=2))


if __name__ == "__main__":
    main()
