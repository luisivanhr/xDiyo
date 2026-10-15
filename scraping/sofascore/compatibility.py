"""Explicit CSV exports and non-destructive updates for existing notebooks."""
from __future__ import annotations

import json
from pathlib import Path
import uuid

import pandas as pd

from .audit import hash_file
from .legacy import NESTED, decode_nested, read_compatibility_views
from .store import atomic_write, json_bytes


def export_csv(manifest_path, output_root, *, name="matches", include_lineups=True):
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("CSV export name must be a filename stem")
    matches, lineups = read_compatibility_views(manifest_path, include_lineups=include_lineups,
                                               include_rosters=include_lineups)
    root = Path(output_root).resolve()
    version = root/"versions"/uuid.uuid4().hex
    files = []
    for filename, frame, nested in ((name+".csv", matches, NESTED),
                                    (name+"_lineups.csv", lineups, ("home", "away"))):
        if frame is None:
            continue
        frame = frame.copy()
        for column in nested:
            if column in frame:
                frame[column] = frame[column].map(lambda v: json.dumps(v, ensure_ascii=False, allow_nan=False)
                    if isinstance(v, (dict, list, tuple)) else None)
        path = version/filename
        atomic_write(path, frame.to_csv(index=False).encode("utf-8"))
        reread = pd.read_csv(path, converters={c: decode_nested for c in nested})
        if reread.match_id.tolist() != frame.match_id.tolist():
            raise ValueError("Compatibility export identity check failed")
        files.append({"file": filename, "rows": len(frame), "sha256": hash_file(path)})
    manifest = version/"manifest.json"
    atomic_write(manifest, json_bytes({"canonical_manifest": str(Path(manifest_path).resolve()), "files": files}))
    atomic_write(root/"CURRENT.json", json_bytes({"manifest": manifest.relative_to(root).as_posix()}))
    return manifest


def merge_recent_leagues_columns(leagues, columns=None, *, output_dir="data/compatibility_updates"):
    """Return a versioned ID upsert. Input files are never replaced or removed.

    The column argument keeps its ordering role; later columns are retained.
    Missing cells in a partial update cannot erase earlier observed facts.
    """
    if len(leagues) != 2:
        raise ValueError("Supply exactly the original and update season files")
    paths = [Path(p) if Path(p).suffix == ".csv" else Path(str(p)+".csv") for p in leagues]
    frames = [pd.read_csv(p) for p in paths]
    for frame in frames:
        if "match_id" not in frame or frame.match_id.isna().any() or frame.match_id.duplicated().any():
            raise ValueError("A season update needs unique non-null match IDs")
    old, new = (frame.set_index("match_id") for frame in frames)
    common = old.index.intersection(new.index)
    for key in ("season_id", "home_id", "away_id"):
        if key in old and key in new and not old.loc[common, key].eq(new.loc[common, key]).all():
            raise ValueError(f"Conflicting {key} for existing match IDs")
    merged = new.combine_first(old)
    ids = [*old.index, *(eid for eid in new.index if eid not in old.index)]
    merged = merged.loc[ids].reset_index()
    if "kickoff_utc" in merged and merged.kickoff_utc.notna().all():
        merged = merged.sort_values(["kickoff_utc", "match_id"], kind="stable")
    ordered = list(dict.fromkeys([*(columns or []), *merged.columns]))
    merged = merged.reindex(columns=ordered).drop(columns=[c for c in ordered if c == "index" or c.startswith("Unnamed:")], errors="ignore")
    destination = Path(output_dir).resolve()/uuid.uuid4().hex/paths[0].name
    atomic_write(destination, merged.to_csv(index=False).encode("utf-8"))
    atomic_write(destination.with_suffix(".manifest.json"), json_bytes({"file": destination.name,
        "sha256": hash_file(destination), "rows": len(merged), "sources": [{"file": str(p.resolve()), "sha256": hash_file(p)} for p in paths]}))
    return destination
