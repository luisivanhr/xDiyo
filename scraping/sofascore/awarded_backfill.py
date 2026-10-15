"""Backfill current season publications from saved Chrome metadata, offline.

Only is_awarded changes. Other relational files are copied byte-for-byte, all
old immutable versions remain, and the normal manifests carry the revision.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time
from uuid import uuid4

from .schema import MATCH_AWARDED_CONTRACT, arrow_schema
from .store import json_bytes
from ..sofascore_web_v3.awarded import SavedAwardedFlags
from ..sofascore_web_v3.campaign import collector_lock
from ..sofascore_web_v3.publish import hash_file, read_json
from ..sofascore_web_v3.reconcile_publication import IDENTITY, _bundle, _copy_files, _path, _verify_files
from ..sofascore_web_v3.status_write import atomic_write_status

ROOT = Path(__file__).resolve().parents[2]


def _with_flag(table, values):
    import pyarrow as pa

    field = arrow_schema("matches").field("is_awarded")
    column = pa.array(values, type=field.type)
    if "is_awarded" in table.column_names:
        return table.set_column(table.schema.get_field_index("is_awarded"), field, column)
    return table.append_column(field, column)


def _write_table(table, path, *, nested=False):
    import pyarrow.parquet as pq

    pq.write_table(table, path, compression="zstd", use_compliant_nested_type=False,
                   **({"row_group_size": 64} if nested else {}))
    if not pq.ParquetFile(path).read().equals(table, check_metadata=True):
        raise ValueError("Awarded backfill Parquet round-trip mismatch")


def backfill_season(output, stem, flags, *, apply=False):
    """Validate, stage and install one revision, restoring public files on failure."""
    import pyarrow.parquet as pq

    output = Path(output).resolve()
    if not stem or Path(stem).name != stem:
        raise ValueError("Expected a season stem, not a path")
    lock = collector_lock(output / "_collection/chrome_publication.lock") if apply else nullcontext()
    with lock:
        outer = read_json(_path(output, f"{stem}.manifest.json"))
        canonical = _path(output, outer["manifest"])
        original_manifest = read_json(canonical)
        matches_path = _path(canonical.parent, original_manifest["tables"]["matches"]["file"])
        matches = pq.ParquetFile(matches_path).read()
        rows = matches.to_pylist()
        expected = {row["event_id"]: tuple(row[key] for key in IDENTITY) for row in rows}
        if len(expected) != len(rows):
            raise ValueError("Duplicate event IDs in season")
        prior = _bundle(output, stem, expected)
        evidence = [flags.get(row) for row in rows]
        values = [item.value for item in evidence]
        result = {"season": stem, "matches": len(rows), "awarded": values.count(True),
                  "ordinary": values.count(False), "unknown": values.count(None)}
        flat_path = _path(output, f"{stem}.parquet")
        with pq.ParquetFile(flat_path) as flat:
            has_column = "is_awarded" in matches.column_names and "is_awarded" in flat.schema_arrow.names
            unchanged = (has_column and matches["is_awarded"].to_pylist() == values and
                         flat.read(columns=["is_awarded"])["is_awarded"].to_pylist() == values and
                         original_manifest.get("match_awarded_contract") == MATCH_AWARDED_CONTRACT)
        if unchanged:
            return {**result, "state": "unchanged", "manifest": str(canonical)}
        if not apply:
            return {**result, "state": "preview"}
        if result["unknown"]:
            unknown = [row["event_id"] for row, value in zip(rows, values) if value is None]
            raise ValueError(f"Cannot backfill without usable saved awarded metadata: {unknown}")

        version = uuid4().hex
        partition = canonical.parent.parent.parent.relative_to(output)
        new_relative = partition / "versions" / version / "manifest.json"
        with TemporaryDirectory(prefix="xdiyo_awarded_backfill_") as temporary:
            stage, backup = Path(temporary) / "replacement", Path(temporary) / "prior"
            destination = stage / new_relative.parent
            destination.mkdir(parents=True)
            for relative in prior["mutable"]:
                path = backup / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(output / relative, path)
            manifest = deepcopy(original_manifest)
            manifest.update(version=version, created_at=time.time(), match_awarded_contract=MATCH_AWARDED_CONTRACT)
            manifest["is_awarded_backfill"] = {
                "previous_manifest": outer["manifest"], "source": "saved_chrome_event_metadata",
                "source_fingerprint": hashlib.sha256(json_bytes([
                    {"event_id": row["event_id"], **asdict(item)} for row, item in zip(rows, evidence)
                ])).hexdigest(),
                "counts": {key: result[key] for key in ("matches", "awarded", "ordinary", "unknown")},
            }
            for name, info in manifest["tables"].items():
                target = _path(destination, info["file"], exists=False)
                target.parent.mkdir(parents=True, exist_ok=True)
                if name == "matches":
                    _write_table(_with_flag(matches, values), target)
                    info.update(bytes=target.stat().st_size, sha256=hash_file(target))
                else:
                    shutil.copy2(_path(canonical.parent, info["file"]), target)
                    if hash_file(target) != info["sha256"]:
                        raise ValueError(f"Unchanged relational table differs: {name}")
            (stage / new_relative).write_bytes(json_bytes(manifest))
            current = stage / partition / "CURRENT.json"
            current.write_bytes(json_bytes({"version": version, "manifest": f"versions/{version}/manifest.json"}))
            # Release Windows file handles before replacing the public Parquet.
            with pq.ParquetFile(flat_path) as flat:
                nested = _with_flag(flat.read(), values)
            nested = nested.replace_schema_metadata({**nested.schema.metadata,
                                                      b"xdiyo_canonical_manifest": new_relative.as_posix().encode()})
            season_path = stage / flat_path.name
            _write_table(nested, season_path, nested=True)
            updated = {**outer, "manifest": new_relative.as_posix(), "completed_at": time.time(),
                       "sha256": hash_file(season_path), "bytes": season_path.stat().st_size}
            (stage / f"{stem}.manifest.json").write_bytes(json_bytes(updated))
            lineage = outer.get("reconciliation", {}).get("source_lineage")
            if lineage:
                target = _path(stage, lineage, exists=False)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(_path(output, lineage), target)
            replacement = _bundle(stage, stem, expected)
            immutable = {key: value for key, value in replacement["files"].items()
                         if key not in replacement["mutable"]}
            _copy_files(stage, output, immutable, reuse_identical=True)
            _verify_files(output, prior["files"])
            try:
                for relative in replacement["mutable"]:
                    atomic_write_status(output / relative, (stage / relative).read_bytes())
                if _bundle(output, stem, expected)["files"] != replacement["files"]:
                    raise ValueError("Installed awarded backfill differs from staged bundle")
                _verify_files(output, {key: value for key, value in prior["files"].items()
                                       if key not in prior["mutable"]})
            except BaseException:
                for relative in prior["mutable"]:
                    atomic_write_status(output / relative, (backup / relative).read_bytes())
                _verify_files(output, prior["files"])
                raise
        return {**result, "state": "published", "manifest": str(output / new_relative)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/xDiyo_data")
    parser.add_argument("--campaigns", type=Path, default=ROOT / "data/website_capture_v3/campaigns")
    parser.add_argument("--stem", action="append", help="Limit to named seasons; default is every current season")
    parser.add_argument("--apply", action="store_true", help="Publish checked revisions; otherwise preview only")
    args = parser.parse_args()
    databases = sorted(args.campaigns.glob("*/queue.sqlite"))
    stems = args.stem or [path.name.removesuffix(".manifest.json")
                          for path in sorted(args.output.glob("*.manifest.json"))]
    if not databases or not stems:
        parser.error("Saved Chrome campaigns and published seasons are required")
    with SavedAwardedFlags(databases) as flags:
        for stem in stems:
            print(json.dumps(backfill_season(args.output, stem, flags, apply=args.apply)), flush=True)


if __name__ == "__main__":
    main()
