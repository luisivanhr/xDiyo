from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import uuid

from .config import PARSER_VERSION
from .parsers import parse_component, parse_match
from .planner import component_jobs
from .schema import MATCH_AWARDED_CONTRACT, MATCH_TIME_CONTRACT, SCHEMA_VERSION, to_arrow
from .store import atomic_write, json_bytes
from .timing import timing_evidence
from .goal_actions import parse_goal_sequences


def normalize_store(store, scope, profile, *, now=None):
    now = time.time() if now is None else now
    tables = defaultdict(list)
    events = store.events(scope.league_id,scope.season_id)
    for event in events:
        match = parse_match(event,observed_at=store.event_observed_at(event["id"]))
        if match["competition_id"] is None:
            match["competition_id"] = scope.league_id
        if match["season_id"] is None:
            match["season_id"] = scope.season_id
        tables["matches"].append(match)
        if profile.include_upcoming and match["status"] != "finished":
            for component, enabled in (("statistics", profile.include_statistics),
                    ("shotmap", profile.include_shotmap), ("heatmap", profile.include_heatmaps),
                    ("goal_actions", profile.include_goal_actions)):
                if enabled:
                    tables["coverage"].append({"event_id": event["id"], "component": component,
                                               "status": "not_applicable_before_completion"})
        incidents = None
        for job in component_jobs(event,profile,now):
            state = store.get(job)
            payload = store.payload(job)
            if job.component == "incidents":
                incidents = payload
                if profile.include_goal_actions and payload is not None:
                    parsed_goals = parse_goal_sequences(payload,event,{"event_id":event["id"],"source_key":job.key,
                        "raw_hash":state["raw_hash"],"observed_at":state["last_success"]},tables.get("shots",[]))
                    for name, rows in parsed_goals.items():
                        tables[name].extend(rows)
                    tables["coverage"].append({"event_id":event["id"],"component":"goal_actions",
                        "status":"per_goal_coverage_recorded" if parsed_goals["goal_sequences"] else "no_goal_incidents"})
            if payload is not None:
                snapshots=store.snapshots(job) if job.component=="lineups" else []
                if not snapshots:
                    snapshots=[{"payload":payload,"event":event,"fetched":state["last_success"],"raw_hash":state["raw_hash"]}]
                for snapshot in snapshots:
                    parsed = parse_component(job,snapshot["payload"],snapshot["event"] or event,profile,
                                             observed_at=snapshot["fetched"],raw_hash=snapshot["raw_hash"])
                    for name,rows in parsed.items():
                        tables[name].extend(rows)
            tables["coverage"].append({"event_id":event["id"],"source_key":job.key,
                "raw_hash":state["raw_hash"] if state else None,"observed_at":state["last_success"] if state else None,
                "component":job.component,"side":str(job.team_id) if job.team_id else None,
                "status":"cached_last_good" if payload is not None else state["status"] if state else "pending",
                "latest_attempt_status":state["status"] if state else "pending"})
        for component,enabled in (("shotmap",profile.include_shotmap),("heatmap",profile.include_heatmaps),
                                  ("lineups",profile.include_lineups),("availability",profile.include_injuries),
                                  ("comments",profile.include_comments),("goal_actions",profile.include_goal_actions)):
            if not enabled:
                tables["coverage"].append({"event_id":event["id"],"component":component,
                    "status":"not_requested"})
        if profile.duration_mode != "off":
            tables["timing"].extend(timing_evidence(event,incidents))
    return dict(tables)


def publish_tables(tables, output_root, *, scope, profile=None, provenance=None, compression="zstd"):
    """Publish a complete version first, then atomically replace CURRENT.json."""
    import pyarrow.parquet as pq
    root = Path(output_root).resolve()
    partition = root / f"competition={scope.league_id}" / f"season={scope.season_id}"
    version = uuid.uuid4().hex
    destination = partition / "versions" / version
    destination.mkdir(parents=True)
    manifest = {"schema_version":SCHEMA_VERSION,"parser_version":PARSER_VERSION,"version":version,
        "created_at":time.time(),"scope":asdict(scope),"profile":asdict(profile) if profile else None,
        "provenance":provenance or "sofascore_raw_cache","tables":{}}
    manifest["match_time_contract"] = MATCH_TIME_CONTRACT
    manifest["match_awarded_contract"] = MATCH_AWARDED_CONTRACT
    coverage = tables.get("coverage",[])
    incomplete = {"pending","failed","blocked","rate_limited"}
    manifest["coverage_summary"] = dict(Counter(r["status"] for r in coverage))
    manifest["requested_components_terminal"] = bool(tables.get("matches")) and not any(
        r.get("status") in incomplete or r.get("latest_attempt_status") in incomplete for r in coverage)
    event_ids = [r["event_id"] for r in tables.get("matches",[])]
    event_id_set = set(event_ids)
    if len(event_ids) != len(event_id_set):
        raise ValueError("Duplicate event IDs in match export")
    for name,rows in sorted(tables.items()):
        if name != "matches" and any(r.get("event_id") is not None and r["event_id"] not in event_id_set for r in rows):
            raise ValueError(f"Orphan event reference in {name}")
        arrow = to_arrow(name,rows)
        path = destination / f"{name}.parquet"
        pq.write_table(arrow,path,compression=compression)
        # pq.read_table may append Hive directory partition columns. Read file for parity.
        loaded = pq.ParquetFile(path).read()
        if not loaded.equals(arrow,check_metadata=True):
            raise ValueError(f"Parquet round-trip failed: {name}")
        manifest["tables"][name] = {"file":path.name,"rows":len(rows),"bytes":path.stat().st_size,
            "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    atomic_write(destination/"manifest.json",json_bytes(manifest))
    atomic_write(partition/"CURRENT.json",json_bytes({"version":version,"manifest":f"versions/{version}/manifest.json"}))
    return destination / "manifest.json"


def read_tables(manifest_path, *, names=None, verify_hashes=True):
    import pyarrow.parquet as pq
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if "tables" not in manifest and "manifest" in manifest:
        path = path.parent / manifest["manifest"]
        manifest = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for name,info in manifest["tables"].items():
        if names is not None and name not in names:
            continue
        table_path = path.parent / info["file"]
        if verify_hashes and hashlib.sha256(table_path.read_bytes()).hexdigest() != info["sha256"]:
            raise ValueError(f"Prepared table hash mismatch: {name}")
        result[name] = pq.ParquetFile(table_path).read().to_pylist()
    return result
