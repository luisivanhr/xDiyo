"""Prepare a component-only Chrome retry and merge recovered data offline.

The original campaign is read-only. Existing successful components seed the new
queue; pregame 404s and xG-only absence never become retry targets.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from uuid import uuid4

from scraping.sofascore.campaign import publish_season
from scraping.sofascore.config import Profile, Scope
from scraping.sofascore.export import read_tables
from scraping.sofascore.store import json_bytes
from .campaign import BASE, ROOT, Campaign, collector_lock, prepare_campaign
from .publish import hash_file, read_json
from .reconcile_publication import (_bundle, _copy_files, _path, _restore, _save,
                                    _verify_files, IDENTITY)
from .status_write import atomic_write_status

COMPONENT_TABLES = {
    "statistics": ("statistics",), "shotmap": ("shots",),
    "heatmap": ("heatmap_points",),
    "lineups": ("lineup_teams", "lineup_players", "player_statistics", "availability"),
}


def queue_rows(directory):
    with closing(sqlite3.connect((Path(directory).resolve() / "queue.sqlite").as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute("SELECT * FROM queue ORDER BY ordinal")]


def queue_hash(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def retry_targets(report, features):
    targets = []
    for row in report["coverage"]:
        component, status = row["component"], row["status"]
        if component not in COMPONENT_TABLES:
            continue
        reason = "http_404" if status == "unavailable" else "not_observed" if status == "not_observed" else None
        feature = "team_heatmaps" if component == "heatmap" else component
        if status == "success" and component in {"lineups", "heatmap"} and features[feature]["status"] == "not_supplied":
            reason = "successful_response_without_requested_feature"
        if reason:
            targets.append({**row, "reason": reason})
    return targets


def prepare_recovery(source, inventory, *, base=BASE, output=ROOT / "data/xDiyo_data"):
    import csv

    source, inventory, output = Path(source).resolve(), Path(inventory).resolve(), Path(output).resolve()
    frozen = read_json(inventory / "summary.json")
    original = queue_rows(source)
    if queue_hash(original) != frozen["source_queue_sha256"]:
        raise ValueError("Queue changed since the gap inventory; review a new inventory first")
    if set(row["state"] for row in original) - {"complete", "deferred"}:
        raise ValueError("Finish ordinary collection before preparing recovery")
    with (inventory / "core_gap_matches.csv").open(encoding="utf-8-sig", newline="") as stream:
        selected = {int(row["event_id"]) for row in csv.DictReader(stream)}
    config = read_json(source / "config.json")
    items = {item["event_id"]: item for item in config["items"]}
    if not selected or selected - items.keys():
        raise ValueError("Recovery inventory contains unknown or no fixtures")
    rows = [row for row in original if row["event_id"] in selected]
    frozen_matches, seeds, season_outputs = {}, [], {}
    with closing(sqlite3.connect((source / "queue.sqlite").as_uri() + "?mode=ro", uri=True)) as db:
        for number, row in enumerate(rows, 1):
            eid = row["event_id"]
            report_path = Path(row["report"])
            if not report_path.resolve().is_relative_to(source / "imports"):
                raise ValueError("Original report escapes its campaign")
            report = read_json(report_path)
            feature_path = report_path.parent / "feature_coverage.json"
            targets = retry_targets(report, read_json(feature_path)["features"])
            if not targets or report["event_id"] != eid:
                raise ValueError(f"Inventory no longer has retryable gaps for {eid}")
            tables = read_tables(report["manifest"], verify_hashes=True)
            expected = items[eid]
            event = expected["reference"]["event"]
            identity = (eid, expected["competition_id"], expected["season_id"], event["homeTeam"]["id"], event["awayTeam"]["id"])
            if len(tables["matches"]) != 1 or tuple(tables["matches"][0][key] for key in IDENTITY) != identity:
                raise ValueError(f"Capture identity mismatch for {eid}")
            paths = {target["path"] for target in targets}
            for observation in db.execute("SELECT * FROM observations WHERE event_id=? ORDER BY observed,rowid", (eid,)):
                entry = json.loads(observation[4])
                if entry.get("_xdiyoSelected") and observation[2] not in paths:
                    if hashlib.sha256(observation[4].encode()).hexdigest() != observation[0]:
                        raise ValueError("Original observation digest mismatch")
                    seeds.append(tuple(observation))
            stem = expected["reference"]["season_stem"]
            if stem not in season_outputs:
                manifest_path = output / f"{stem}.manifest.json"
                outer = read_json(manifest_path)
                season_outputs[stem] = {"manifest_sha256": hash_file(manifest_path), "parquet_sha256": outer["sha256"]}
            frozen_matches[str(eid)] = {"event_id": eid, "season_stem": stem, "label": expected["label"],
                "original_queue_row": row, "original_report_sha256": hash_file(report_path),
                "original_manifest_sha256": hash_file(report["manifest"]),
                "original_feature_sha256": hash_file(feature_path), "targets": targets}
            if number % 100 == 0:
                print(f"Verified and prepared {number}/{len(rows)} recovery fixtures", flush=True)
    if queue_rows(source) != original:
        raise ValueError("Original queue changed during recovery preparation")
    source_info = {"collection": "component_recovery", "source_campaign": str(source),
        "source_queue_sha256": queue_hash(original), "inventory": str(inventory),
        "targets": sum(len(row["targets"]) for row in frozen_matches.values()),
        "pregame_retries": False, "xg_only_retries": False, "source_values": "Previously captured Chrome responses; no legacy feature values"}
    prepared = prepare_campaign([row["event_id"] for row in rows], name="chrome-recovery",
        profile=Profile(**config["profile"]), policy=config["policy"], base=base,
        references={eid: items[eid]["reference"] for eid in selected}, source_metadata=source_info)
    directory = Path(prepared["directory"])
    with Campaign(directory) as campaign:
        with campaign.db:
            campaign.db.executemany("INSERT INTO observations VALUES (?,?,?,?,?)", seeds)
        for eid in selected:
            missing = {entry["path"] for entry in campaign.coverage(eid)["components"] if entry["status"] == "not_observed"}
            expected = {entry["path"] for entry in frozen_matches[str(eid)]["targets"]}
            if missing != expected:
                raise ValueError(f"Seeded queue would request unexpected components for {eid}: {missing ^ expected}")
        campaign.write_status("ready", "Targeted recovery prepared; existing captured components reused")
    plan = {**prepared, **source_info, "created_at": datetime.now(timezone.utc).isoformat(),
        "output": str(output), "seeded_observations": len(seeds), "matches": frozen_matches,
        "season_outputs": season_outputs,
        "note": "One retry pass. Old reports, bodies and public outputs are preserved. A repeated 404 remains unresolved data, although collection of that attempt is complete."}
    atomic_write_status(directory / "recovery.json", json_bytes(plan))
    return {"campaign_id": prepared["campaign_id"], "directory": str(directory), "matches": len(rows),
        "components": source_info["targets"], "seasons": len(season_outputs), "seeded_observations": len(seeds),
        "recovery_plan": str(directory / "recovery.json")}


def _selected(rows, eid, component, team_id=None):
    return [row for row in rows if row.get("event_id") == eid and
            (component != "heatmap" or row.get("team_id") == team_id)]


def _insert_match_rows(rows, additions, order):
    """Retain within-match order; new rows stay with their original match."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["event_id"]].append(row)
    for row in additions:
        if row not in grouped[row["event_id"]]:
            grouped[row["event_id"]].append(row)
    return [row for eid in order for row in grouped[eid]]


def merge_match(tables, captured, targets):
    """Fill absent primary data only; preserve earlier lineup snapshots and absences."""
    eid = captured["matches"][0]["event_id"]
    existing_match = next(row for row in tables["matches"] if row["event_id"] == eid)
    if tuple(existing_match[k] for k in IDENTITY) != tuple(captured["matches"][0][k] for k in IDENTITY):
        raise ValueError("Recovered match identity differs from published data")
    order = [row["event_id"] for row in tables["matches"]]
    outcomes = []
    for target in targets:
        component, team_id = target["component"], target.get("team_id")
        names = COMPONENT_TABLES[component]
        primary = "lineup_players" if component == "lineups" else names[0]
        old = _selected(tables.get(primary, []), eid, component, team_id)
        new = _selected(captured.get(primary, []), eid, component, team_id)
        valid = bool(new)
        if component == "lineups":
            valid = {row["side"] for row in new} == {"home", "away"}
        if old:
            status = "already_present_in_published_version"
        elif not valid:
            status = "still_unavailable_or_empty"
        else:
            status = "recovered"
            for name in names:
                additions = _selected(captured.get(name, []), eid, component, team_id)
                if additions:
                    tables[name] = _insert_match_rows(tables.get(name, []), additions, order)
            def selected_coverage(row):
                return row.get("event_id") == eid and row.get("component") == component and (
                    component != "heatmap" or row.get("side") == str(team_id))
            replacements = [row for row in captured["coverage"] if selected_coverage(row)]
            if not replacements:
                raise ValueError("Recovered data lacks component provenance")
            tables["coverage"] = _insert_match_rows([row for row in tables["coverage"] if not selected_coverage(row)], replacements, order)
            if component == "shotmap":
                # Only fill newly supportable links; preserve goal actions and old links.
                links = {row["sequence_id"]: row for row in captured.get("goal_sequences", []) if row.get("shot_id") is not None}
                for row in tables.get("goal_sequences", []):
                    link = links.get(row.get("sequence_id"))
                    if row.get("event_id") == eid and row.get("shot_id") is None and link:
                        row.update(shot_id=link["shot_id"], shot_link_method=link["shot_link_method"])
        outcomes.append({"event_id": eid, "path": target["path"], "component": component,
            "prior_status": target["status"], "outcome": status, "previous_rows": len(old), "recovery_rows": len(new)})
    return outcomes


def _install_bundle(output, stage, backup, receipt_path, receipt, expected):
    """Guarded three-entry switch with preserved bundles and durable rollback."""
    prior, replacement = receipt["prior"], receipt["replacement"]
    _copy_files(output, backup, prior["files"])
    _bundle(backup, receipt["stem"], expected)
    _save(receipt_path, receipt, "prepared")
    dependencies = {key: value for key, value in replacement["files"].items() if key not in replacement["mutable"]}
    _copy_files(stage, output, dependencies, reuse_identical=True)
    _verify_files(output, prior["files"])
    try:
        for relative in replacement["mutable"]:
            atomic_write_status(_path(output, relative), _path(stage, relative).read_bytes())
        if _bundle(output, receipt["stem"], expected)["files"] != replacement["files"]:
            raise ValueError("Installed recovery bundle differs from its staged version")
    except BaseException as error:
        _restore(output, backup, receipt)
        _save(receipt_path, receipt, "rolled_back", error=str(error))
        raise
    _save(receipt_path, receipt, "published")


def process_season(directory, stem, *, install=True):
    directory = Path(directory).resolve()
    plan = read_json(directory / "recovery.json")
    output = Path(plan["output"])
    targets = [row for row in plan["matches"].values() if row["season_stem"] == stem]
    queue = {row["event_id"]: row for row in queue_rows(directory)}
    if any(queue[row["event_id"]]["state"] != "complete" for row in targets):
        raise ValueError("Finish this season's recovery matches before merging")
    receipt_path = directory / "recovery_results" / f"{stem}.json"
    reports = {str(row["event_id"]): {"path": queue[row["event_id"]]["report"],
        "sha256": hash_file(queue[row["event_id"]]["report"])} for row in targets}
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt["reports"] != reports:
            raise ValueError("Recovery reports changed after this season's merge")
        if receipt["state"] in {"published", "no_new_data"}:
            _verify_files(output, receipt.get("replacement", receipt["prior"])["files"])
            expected_sha = receipt.get("replacement", {}).get("season_sha256", plan["season_outputs"][stem]["parquet_sha256"])
            if hash_file(output / f"{stem}.parquet") != expected_sha:
                raise ValueError("Published season changed after recovery")
            return receipt
        if receipt["state"] not in {"prepared", "rolled_back"}:
            raise ValueError("Inspect unfinished recovery receipt before proceeding")
        # Interrupted switches are restored before a new staged attempt.
        with collector_lock(output / "_collection/chrome_publication.lock"):
            _restore(output, Path(receipt["backup"]), receipt)
    config = read_json(Path(plan["source_campaign"]) / "config.json")
    items = [item for item in config["items"] if item["reference"].get("season_stem") == stem]
    expected = {item["event_id"]: (item["event_id"], item["competition_id"], item["season_id"],
        item["reference"]["event"]["homeTeam"]["id"], item["reference"]["event"]["awayTeam"]["id"]) for item in items}
    with collector_lock(output / "_collection/chrome_publication.lock"):
        prior = _bundle(output, stem, expected)
        if prior["season_sha256"] != plan["season_outputs"][stem]["parquet_sha256"] or hash_file(output / f"{stem}.manifest.json") != plan["season_outputs"][stem]["manifest_sha256"]:
            raise ValueError("Published baseline changed since recovery preparation")
        outer = read_json(output / f"{stem}.manifest.json")
        tables = read_tables(output / outer["manifest"], verify_hashes=True)
        original = deepcopy(tables)
        outcomes = []
        for target in targets:
            report = read_json(reports[str(target["event_id"])]["path"])
            if report.get("complete") is not True:
                raise ValueError("Incomplete recovery import")
            captured = read_tables(report["manifest"], verify_hashes=True)
            outcomes.extend(merge_match(tables, captured, target["targets"]))
        # Collection does not alter scores, order, standings, timing or other fixtures.
        for name in ("matches", "pregame", "incidents", "timing", "goal_actions", "passing_edges", "team_seasons"):
            if tables.get(name) != original.get(name):
                raise ValueError(f"Recovery altered preserved {name}")
        receipt = {"campaign_id": plan["campaign_id"], "campaign_directory": str(directory),
            "output": str(output), "stem": stem, "reports": reports, "outcomes": outcomes,
            "outcome_counts": dict(Counter(row["outcome"] for row in outcomes)), "prior": prior,
            "verification": {"match_order_and_count": "unchanged", "scores_and_pregame": "unchanged", "original_tables_hashes": "passed"}}
        if not any(row["outcome"] == "recovered" for row in outcomes):
            _save(receipt_path, receipt, "no_new_data")
            return receipt
        run = output / "_collection" / "chrome_recoveries" / plan["campaign_id"] / stem / uuid4().hex
        stage, backup = run / "replacement", run / "prior"
        manifest = read_json(output / outer["manifest"])
        scope_data = dict(manifest["scope"])
        scope_data["rounds"] = tuple(scope_data["rounds"])
        profile = Profile(**outer["profile"])
        lineage = f"_collection/chrome_recovery_exports/{stem}.{run.name}.json"
        atomic_write_status(stage / lineage, json_bytes({"campaign": str(directory), "reports": reports,
            "outcomes": outcomes, "prior_manifest": str(output / outer["manifest"]),
            "prior_parquet_sha256": prior["season_sha256"]}))
        reconciliation = {"collection": "chrome_component_recovery", "campaign_id": plan["campaign_id"],
            "source_lineage": lineage, "previous_reconciliation": outer.get("reconciliation"),
            "recovered_components": receipt["outcome_counts"]["recovered"],
            "scope": "Only absent requested components filled; original match order and pregame positions retained."}
        result = publish_season(tables, stage, Scope(**scope_data), profile, stem, reconciliation)
        if read_tables(stage / result["manifest"], verify_hashes=True) != tables:
            raise ValueError("Staged relational recovery differs from merged tables")
        replacement = _bundle(stage, stem, expected)
        if replacement["mutable"] != prior["mutable"]:
            raise ValueError("Recovery entry paths differ from the baseline")
        receipt.update(replacement=replacement, stage=str(stage), backup=str(backup))
        if install:
            _install_bundle(output, stage, backup, receipt_path, receipt, expected)
        else:
            _save(receipt_path, receipt, "staged")
        return receipt


def watch_recovery(directory, *, interval_seconds=30, watch=True):
    directory = Path(directory).resolve()
    plan = read_json(directory / "recovery.json")
    grouped = defaultdict(list)
    for row in plan["matches"].values():
        grouped[row["season_stem"]].append(row["event_id"])
    results = {}
    state_path = directory / "recovery_publication_watch.json"
    def record(state, error=None):
        counts = Counter(row["state"] for row in queue_rows(directory))
        outcomes = Counter()
        for result in results.values():
            outcomes.update(result["outcome_counts"])
        data = {"recorded_at": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
            "campaign_id": plan["campaign_id"], "state": state, "queue_counts": dict(counts),
            "processed_seasons": len(results), "total_seasons": len(grouped),
            "outcome_counts": dict(outcomes), "error": error}
        atomic_write_status(state_path, json_bytes(data))
        return data
    with collector_lock(directory / "recovery_publication.lock"):
        record("starting")
        try:
            while True:
                queue = {row["event_id"]: row for row in queue_rows(directory)}
                for stem, ids in grouped.items():
                    if stem in results or not all(queue[eid]["state"] == "complete" for eid in ids):
                        continue
                    record("merging")
                    results[stem] = process_season(directory, stem)
                    print(json.dumps({"season": stem, "state": results[stem]["state"], "outcomes": results[stem]["outcome_counts"]}), flush=True)
                if len(results) == len(grouped):
                    return record("finished_with_gaps" if any(r["outcome_counts"].get("still_unavailable_or_empty") for r in results.values()) else "finished")
                record("waiting")
                if not watch:
                    return record("pass_complete")
                time.sleep(interval_seconds)
        except BaseException as error:
            record("stopped" if isinstance(error, KeyboardInterrupt) else "failed", str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--inventory", type=Path, required=True)
    prepare.add_argument("--receipt", type=Path, required=True)
    watch = commands.add_parser("watch")
    watch.add_argument("--campaign", type=Path, required=True)
    watch.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        if args.receipt.exists():
            raise ValueError("Preparation receipt exists; resume its campaign instead of duplicating it")
        result = prepare_recovery(args.source, args.inventory)
        atomic_write_status(args.receipt, json_bytes(result))
    else:
        result = watch_recovery(args.campaign, watch=not args.once)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
