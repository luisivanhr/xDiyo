"""Verify published odds bytes, original manifest pins and Parquet row counts.

Run from any directory after hydrating the OddsPapi Git LFS object. This checks
publication integrity; it does not certify historical quote availability.
"""

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1] / "data" / "odds"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(root=ROOT):
    publication = json.loads((root / "publication.json").read_text(encoding="utf-8"))
    files = publication["files"]
    records = {item["path"]: item for item in files}
    assert len(records) == len(files), "Duplicate publication paths"
    rows = 0
    for name, item in records.items():
        path = root / name
        assert path.resolve().is_relative_to(root.resolve()), name
        assert path.stat().st_size == item["bytes"], f"Size mismatch: {name}; hydrate Git LFS if needed"
        assert digest(path) == item["sha256"], f"Hash mismatch: {name}"
        if path.suffix == ".parquet":
            count = pq.ParquetFile(path).metadata.num_rows
            assert count == item["rows"], f"Row count mismatch: {name}"
            rows += count

    snapshot = "footiqo/29a84259c0b8a53578d8b6bc"
    manifest = json.loads((root / snapshot / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        saved = records[f"{snapshot}/{entry['path']}"]
        assert saved["sha256"] == entry["sha256"], entry["path"]
        assert saved["rows"] == entry["rows"], entry["path"]

    for directory in ("mappings", "footiqo_crosswalks"):
        for name in records:
            if not name.startswith(f"{directory}/") or not name.endswith("/manifest.json"):
                continue
            mapping = json.loads((root / name).read_text(encoding="utf-8"))
            assert mapping["snapshot"] == manifest["snapshot"], name
            assert mapping["snapshot_manifest_sha256"] == records[f"{snapshot}/manifest.json"]["sha256"], name
            for entry in mapping["files"]:
                key = (Path(name).parent / entry["path"]).as_posix()
                assert records[key]["sha256"] == entry["sha256"], key

    archive = "oddspapi/20261028"
    consolidation = json.loads((root / archive / "consolidation.json").read_text(encoding="utf-8"))
    saved = records[f"{archive}/historical_odds.parquet"]
    assert saved["rows"] == consolidation["quote_rows"]
    assert saved["bytes"] == consolidation["bytes"]
    assert not consolidation["complete"], "Update publication documentation for a completed archive"
    assert pq.ParquetFile(root / archive / "historical_odds.parquet").metadata.num_columns == consolidation["schema_columns"]
    result = {"files_verified": len(files), "bytes_verified": sum(item["bytes"] for item in files),
              "parquet_rows_verified": rows, "oddspapi_complete": consolidation["complete"],
              "original_snapshot_and_crosswalk_pins": "unchanged"}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    verify()
