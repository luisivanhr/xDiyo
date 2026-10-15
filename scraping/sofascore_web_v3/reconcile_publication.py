"""Preserve an older season bundle before publishing its completed Chrome cohort.

The default preview is read-only. Apply requires a finished queue, disconnected
collector, and the reviewed old Parquet hash. No provider requests are made.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, closing
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
from uuid import uuid4

from scraping.sofascore.export import read_tables
from scraping.sofascore.store import json_bytes
from .campaign import ROOT, collector_lock
from .publish import hash_file, publish_captured_season, read_json, snapshot_season
from .status_write import atomic_write_status

IDENTITY = ("event_id", "competition_id", "season_id", "home_id", "away_id")


def _path(root, relative, *, exists=True):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Artifact paths must be relative and contained")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root or (exists and not path.is_file()):
        raise ValueError(f"Artifact must stay inside {root}: {relative}")
    return path


def _queue(directory, stem):
    if not re.fullmatch(r"[A-Za-z0-9_]+_\d{2}_\d{2}", stem):
        raise ValueError("Use the original season filename stem")
    config_path = directory / "config.json"
    digest = hash_file(config_path)
    config = read_json(config_path)
    items = config["items"]
    if len({i["event_id"] for i in items}) != len(items):
        raise ValueError("Duplicate configured fixture IDs")
    with closing(sqlite3.connect((directory / "queue.sqlite").as_uri() + "?mode=ro", uri=True)) as db:
        rows = dict(db.execute("SELECT event_id,state FROM queue"))
    if set(rows) != {i["event_id"] for i in items}:
        raise ValueError("Queue and configured inventory differ")
    selected = [i for i in items if i["reference"].get("season_stem") == stem]
    if not selected:
        raise ValueError("Season is absent from the queue")
    identities = {(i["competition_id"], i["season_id"]) for i in selected}
    if len(identities) != 1:
        raise ValueError("Season stem maps to multiple provider identities")
    expected = {i["event_id"]: (i["event_id"], i["competition_id"], i["season_id"],
                i["reference"]["event"]["homeTeam"]["id"],
                i["reference"]["event"]["awayTeam"]["id"]) for i in selected}
    unfinished = {eid: state for eid, state in rows.items() if state != "complete"}
    target_unfinished = {eid: rows[eid] for eid in expected if rows[eid] != "complete"}
    if hash_file(config_path) != digest:
        raise ValueError("Configuration changed during review")
    return config, expected, {"config_sha256": digest, "queue_matches": len(rows),
        "unfinished_queue_matches": len(unfinished), "unfinished_target_matches": target_unfinished}


def _bundle(root, stem, expected):
    """Validate a complete bundle's hashes, identities, and all three entry files."""
    import pyarrow.parquet as pq

    root = Path(root).resolve()
    outer_path = _path(root, f"{stem}.manifest.json")
    season_path = _path(root, f"{stem}.parquet")
    outer = read_json(outer_path)
    if outer.get("complete") is not True or outer.get("season_file") != season_path.name:
        raise ValueError("Existing season is not a complete named output pair")
    if hash_file(season_path) != outer.get("sha256"):
        raise ValueError("Season Parquet hash mismatch")
    canonical = _path(root, outer["manifest"])
    manifest = read_json(canonical)
    first = next(iter(expected.values()))
    prefix = Path("_tables")
    cohort = outer.get("reconciliation", {}).get("archive_cohort", {})
    if cohort.get("shared_provider_identity"):
        if (cohort.get("stem") != stem or cohort.get("competition_id") != first[1]
                or cohort.get("season_id") != first[2] or not cohort.get("schedule_verified")
                or manifest.get("provenance", {}).get("archive_cohort") != cohort):
            raise ValueError("Shared-provider cohort provenance disagrees with the selected season")
        prefix = prefix / "cohorts" / stem
    prefix = prefix / f"competition={first[1]}" / f"season={first[2]}"
    version = manifest.get("version", "")
    if not re.fullmatch(r"[0-9a-f]{32}", version):
        raise ValueError("Unexpected relational version identifier")
    if canonical != _path(root, prefix / "versions" / version / "manifest.json"):
        raise ValueError("Canonical manifest does not belong to this season partition")
    current = _path(root, prefix / "CURRENT.json")
    pointer = read_json(current)
    if pointer != {"version": version, "manifest": f"versions/{version}/manifest.json"}:
        raise ValueError("Season entry and CURRENT pointer disagree")
    if (manifest.get("profile") != outer.get("profile") or
            manifest.get("schema_version") != outer.get("schema_version") or
            manifest.get("requested_components_terminal") is not True):
        raise ValueError("Season and relational manifest metadata disagree")
    files = [season_path, outer_path, current, canonical]
    for info in manifest["tables"].values():
        table = _path(canonical.parent, info["file"])
        if hash_file(table) != info["sha256"] or pq.ParquetFile(table).metadata.num_rows != info["rows"]:
            raise ValueError("Relational table hash or row count mismatch")
        files.append(table)
    matches = read_tables(canonical, names=["matches"], verify_hashes=True)["matches"]
    actual = [tuple(r[k] for k in IDENTITY) for r in matches]
    if len(actual) != len(expected) or set(actual) != set(expected.values()):
        raise ValueError("Season fixture identities differ from the queued cohort")
    parquet = pq.ParquetFile(season_path)
    nested = [tuple(r[k] for k in IDENTITY) for r in parquet.read(columns=list(IDENTITY)).to_pylist()]
    metadata = parquet.schema_arrow.metadata or {}
    if (nested != actual or outer.get("matches") != len(expected) or
            metadata.get(b"xdiyo_canonical_manifest") != outer["manifest"].encode() or
            metadata.get(b"xdiyo_season_stem") != stem.encode()):
        raise ValueError("Nested season and relational fixture identities disagree")
    lineage = outer.get("reconciliation", {}).get("source_lineage")
    if lineage:
        files.append(_path(root, lineage))
    inventory = {p.relative_to(root).as_posix(): {"sha256": hash_file(p), "bytes": p.stat().st_size}
                 for p in files}
    return {"files": inventory, "mutable": [season_path.name, outer_path.name,
                current.relative_to(root).as_posix()], "season_sha256": outer["sha256"],
            "profile": outer["profile"], "matches": len(expected),
            "chrome_source_fingerprint": outer.get("reconciliation", {}).get("chrome_source_fingerprint")}


def _verify_files(root, inventory):
    for relative, info in inventory.items():
        path = _path(root, relative)
        if hash_file(path) != info["sha256"] or path.stat().st_size != info["bytes"]:
            raise ValueError(f"Artifact changed: {relative}")


def _copy_files(source, destination, inventory, *, reuse_identical=False):
    """Copy immutable artifacts without deleting or overwriting an existing file."""
    _verify_files(source, inventory)
    for relative, info in inventory.items():
        src = _path(source, relative)
        dst = _path(destination, relative, exists=False)
        if dst.exists() and reuse_identical:
            _verify_files(destination, {relative: info})
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open("rb") as inp, dst.open("xb") as out:
            shutil.copyfileobj(inp, out)
            out.flush()
            os.fsync(out.fileno())
    _verify_files(destination, inventory)


def _save(receipt_path, receipt, state, **extra):
    receipt.update(state=state, updated_at=datetime.now(timezone.utc).isoformat(), **extra)
    atomic_write_status(receipt_path, json_bytes(receipt))


def _restore(output, prior_root, receipt):
    prior, new = receipt["prior"], receipt["replacement"]
    _verify_files(prior_root, prior["files"])
    # Refuse unexplained concurrent edits before restoring any pointer.
    for relative in prior["mutable"]:
        path = _path(output, relative, exists=False)
        if not path.exists():
            continue
        digest = hash_file(path)
        if digest not in {prior["files"][relative]["sha256"], new["files"][relative]["sha256"]}:
            raise ValueError(f"Unexpected entry-file change; inspect {relative}")
    for relative in prior["mutable"]:
        atomic_write_status(_path(output, relative, exists=False), _path(prior_root, relative).read_bytes())
    _verify_files(output, prior["files"])


def _locks(stack, output, lock_path):
    try:
        stack.enter_context(collector_lock(lock_path or ROOT / "data/collection_state/collector.lock"))
        stack.enter_context(collector_lock(output / "_collection/chrome_publication.lock"))
    except OSError as error:
        raise RuntimeError("Disconnect the collector and wait for the publisher before applying reconciliation") from error


def reconcile_publication(campaign_directory, stem, *, output=ROOT / "data/xDiyo_data",
                          apply=False, expected_sha256=None, lock_path=None):
    directory, output = Path(campaign_directory).resolve(), Path(output).resolve()
    with ExitStack() as stack:
        if apply:
            _locks(stack, output, lock_path)
        config, expected, queue = _queue(directory, stem)
        prior = _bundle(output, stem, expected)
        result = {"state": "preview", "campaign_directory": str(directory), "output": str(output),
                  "stem": stem, **queue, "prior": prior, "requested_profile": config["profile"],
                  "ready": queue["unfinished_queue_matches"] == 0,
                  "scope": "No network requests or queue writes. Older season files and relational versions are preserved; source capture stores are untouched."}
        if not apply:
            return result
        if not result["ready"]:
            raise ValueError("Finish the entire queue, including deferred fixtures, before reconciliation")
        if prior["chrome_source_fingerprint"]:
            # This command only replaces older direct exports, never a different Chrome publication.
            _, _, _, items, sources, fingerprint = snapshot_season(directory, stem)
            outer = read_json(output / f"{stem}.manifest.json")
            lineage = read_json(_path(output, outer["reconciliation"]["source_lineage"]))
            matches = read_tables(_path(output, outer["manifest"]), names=["matches"], verify_hashes=True)["matches"]
            if (fingerprint != prior["chrome_source_fingerprint"] or lineage.get("fingerprint") != fingerprint or
                    lineage.get("sources") != sources or
                    [r["event_id"] for r in matches] != [i["event_id"] for i in items]):
                raise ValueError("Existing Chrome publication has different source provenance")
            return {**result, "state": "already_published"}
        if expected_sha256 != prior["season_sha256"]:
            raise ValueError("Supply the reviewed existing season SHA-256 before applying")
        relative_run = Path("_collection/chrome_reconciliations") / stem / uuid4().hex
        run = _path(output, relative_run, exists=False)
        stage, backup = run / "replacement", run / "prior"
        publish_captured_season(directory, stem, output=stage)
        replacement = _bundle(stage, stem, expected)
        if replacement["mutable"] != prior["mutable"]:
            raise ValueError("Replacement entry paths differ from the prior season")
        if replacement["profile"] != config["profile"]:
            raise ValueError("Replacement profile differs from the requested collection")
        _copy_files(output, backup, prior["files"])
        _bundle(backup, stem, expected)
        if hash_file(directory / "config.json") != queue["config_sha256"]:
            raise ValueError("Configuration changed while staging reconciliation")
        receipt_path = run / "receipt.json"
        receipt = {**result, "replacement": replacement, "receipt": str(receipt_path)}
        _save(receipt_path, receipt, "prepared")
        dependencies = {k:v for k,v in replacement["files"].items() if k not in replacement["mutable"]}
        try:
            _copy_files(stage, output, dependencies, reuse_identical=True)
            _verify_files(output, prior["files"])
        except Exception as error:
            _save(receipt_path, receipt, "preparation_failed", failure=str(error))
            raise RuntimeError(f"Preparation failed; the prior entry was not replaced. Receipt: {receipt_path}") from error
        try:
            for relative in replacement["mutable"]:
                atomic_write_status(_path(output, relative), _path(stage, relative).read_bytes())
            installed = _bundle(output, stem, expected)
            if installed["files"] != replacement["files"]:
                raise ValueError("Installed bundle differs from the staged replacement")
        except BaseException as error:
            try:
                _restore(output, backup, receipt)
                _save(receipt_path, receipt, "rolled_back", failure=str(error))
            except BaseException as recovery_error:
                raise RuntimeError(f"Reconciliation needs recovery; both bundles remain at {run}. Use --recover-receipt {receipt_path}") from recovery_error
            raise RuntimeError(f"Replacement failed; the prior season was restored. Receipt: {receipt_path}") from error
        try:
            _save(receipt_path, receipt, "published")
        except OSError as error:
            raise RuntimeError(f"Replacement is published but its receipt update failed. Use --recover-receipt {receipt_path}") from error
        return receipt


def recover_reconciliation(receipt_path, *, output=ROOT / "data/xDiyo_data", lock_path=None):
    """Finalize a fully installed replacement or restore the prior complete bundle."""
    output = Path(output).resolve()
    receipt_path = Path(receipt_path).resolve()
    receipt = read_json(receipt_path)
    stem = receipt["stem"]
    # Keep every restored path within the explicitly selected output and run.
    if not re.fullmatch(r"[A-Za-z0-9_]+_\d{2}_\d{2}", stem):
        raise ValueError("Invalid receipt season stem")
    run = receipt_path.parent
    expected_parent = _path(output, Path("_collection/chrome_reconciliations") / stem, exists=False)
    if (run.parent != expected_parent or not re.fullmatch(r"[0-9a-f]{32}", run.name) or
            receipt_path.name != "receipt.json" or receipt["output"] != str(output)):
        raise ValueError("Receipt does not belong to the selected output")
    with ExitStack() as stack:
        _locks(stack, output, lock_path)
        directory = Path(receipt["campaign_directory"]).resolve()
        _, expected, queue = _queue(directory, stem)
        if queue["unfinished_queue_matches"] or queue["config_sha256"] != receipt["config_sha256"]:
            raise ValueError("Queue or configuration changed since reconciliation")
        prior = _bundle(run / "prior", stem, expected)
        replacement = _bundle(run / "replacement", stem, expected)
        if prior != receipt["prior"] or replacement != receipt["replacement"]:
            raise ValueError("Preserved bundle differs from its receipt")
        if all(_path(output, k, exists=False).is_file() and
               hash_file(_path(output, k)) == replacement["files"][k]["sha256"] for k in replacement["mutable"]):
            _verify_files(output, replacement["files"])
            state = "published"
        else:
            _restore(output, run / "prior", receipt)
            state = "rolled_back"
        _save(receipt_path, receipt, state)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--stem")
    parser.add_argument("--output", type=Path, default=ROOT / "data/xDiyo_data")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--expected-existing-sha256")
    parser.add_argument("--recover-receipt", type=Path)
    args = parser.parse_args()
    if args.recover_receipt:
        if args.apply or args.campaign or args.stem or args.expected_existing_sha256:
            parser.error("Recovery takes only --recover-receipt and --output")
        result = recover_reconciliation(args.recover_receipt, output=args.output)
    else:
        if not args.campaign or not args.stem:
            parser.error("Supply --campaign and --stem")
        result = reconcile_publication(args.campaign, args.stem, output=args.output, apply=args.apply,
                                       expected_sha256=args.expected_existing_sha256)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
