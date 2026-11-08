"""Publish exact original-13 native annotations with reproducible input snapshots."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from xdiyo_analytics.data.fixture_rounds import _identity, SELECTED, file_sha256
from xdiyo_analytics.data.catalogue import load_reviewed_team_directory
from xdiyo_analytics.data.native_fixture_rounds import (
    RELEASE, LEAGUES, BY_ID, NATIVE_COLUMNS, SEMANTICS,
    build_native_fixture_rounds, load_native_fixture_rounds,
)


def native_snapshot(native_root, targets):
    root = Path(native_root).resolve()
    first = int(targets.season.str[:4].min())
    last = int(targets.season.str[:4].max())
    records, receipts = [], []
    for lid, country, label, stem, comp, tournament in LEAGUES:
        for year in range(first, last + 1):
            p = root/f"{stem}_{year%100:02}_{(year+1)%100:02}.manifest.json"
            if not p.exists():
                receipts.append(dict(publication=p.name, status="missing_publication"))
                continue
            publication_bytes = p.read_bytes()
            pub = json.loads(publication_bytes)
            mp = (root/pub["manifest"]).resolve()
            if not mp.is_relative_to(root):
                raise ValueError("Native publication manifest escapes data root")
            manifest_bytes = mp.read_bytes()
            manifest = json.loads(manifest_bytes)
            scope = manifest["scope"]
            if (scope["league_id"], scope["season_start"], scope["season_end"]) != (comp, year, year+1):
                raise ValueError(f"Native publication scope disagrees: {p.name}")
            descriptor = manifest["tables"]["matches"]
            path = (mp.parent/descriptor["file"]).resolve()
            if not path.is_relative_to(mp.parent) or file_sha256(path) != descriptor["sha256"]:
                raise ValueError(f"Native matches checksum mismatch: {p.name}")
            frame = pd.read_parquet(path)
            if len(frame) != descriptor["rows"] or not frame.event_id.is_unique:
                raise ValueError(f"Native match population mismatch: {p.name}")
            if not (frame.competition_id.eq(comp).all() and frame.season_id.eq(scope["season_id"]).all()):
                raise ValueError(f"Native row identity disagrees with publication: {p.name}")
            frame["season"] = f"{year}/{year+1}"
            civil = pd.to_datetime(frame.kickoff_utc, unit="s", utc=True).dt.tz_convert("Europe/Berlin").dt.tz_localize(None)
            frame["native_date"] = civil.dt.strftime("%Y-%m-%d")
            frame["native_civil_datetime"] = civil.map(lambda t: t.isoformat())
            frame["publication"] = p.name
            frame["matches_sha256"] = descriptor["sha256"]
            records.append(frame[list(NATIVE_COLUMNS)])
            if p.read_bytes() != publication_bytes or mp.read_bytes() != manifest_bytes or file_sha256(path) != descriptor["sha256"]:
                raise ValueError(f"Native publication changed during snapshot: {p.name}")
            receipts.append(dict(publication=p.name, status="hash_verified", scope=scope,
                publication_sha256=file_sha256(p), manifest_path=mp.relative_to(root).as_posix(),
                manifest_sha256=file_sha256(mp), matches_path=path.relative_to(root).as_posix(),
                matches_sha256=descriptor["sha256"], rows=len(frame)))
    if not records:
        raise ValueError("No original-13 native publications")
    return pd.concat(records, ignore_index=True), receipts


def publish(matches, native_root, output):
    matches, output = Path(matches), Path(output)
    if output.exists():
        raise FileExistsError("Immutable native round release already exists")
    checksum = file_sha256(matches)
    targets = _identity(pd.read_parquet(matches))
    targets = targets.loc[targets.league_id.isin(BY_ID)].reset_index(drop=True)
    directory = load_reviewed_team_directory()
    native, sources = native_snapshot(native_root, targets)
    built = build_native_fixture_rounds(targets, native, directory)
    if file_sha256(matches) != checksum:
        raise ValueError("Workbook input changed during snapshot")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".pending-original13-", dir=output.parent))
    try:
        provenance = stage/"provenance"
        provenance.mkdir()
        tables = {"fixtures.parquet": built.selections, "provenance/workbook_targets.parquet": targets,
                  "provenance/native_annotations.parquet": native, "provenance/lineage.parquet": built.lineage,
                  "provenance/candidates.parquet": built.candidates}
        for name, frame in tables.items():
            frame.to_parquet(stage/name, index=False, compression="zstd")
        (provenance/"reviewed_team_directory.json").write_text(json.dumps(directory, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        (provenance/"native_publications.json").write_text(json.dumps(sources, indent=2)+"\n", encoding="utf-8")
        unresolved = built.lineage.loc[~built.lineage.selection_status.eq("matched_exact_league_season_teams_date")]
        unresolved.merge(targets, on="match_id", validate="one_to_one").to_csv(provenance/"unresolved.csv", index=False)
        summary = targets[["match_id", "country", "league", "season", "kickoff_date"]].merge(
            built.lineage[["match_id", "selection_status"]], on="match_id", validate="one_to_one")
        summary.groupby(["country", "league", "season", "selection_status"]).size().rename("fixtures").reset_index().to_csv(provenance/"coverage.csv", index=False)
        manifest = dict(schema=1, release=RELEASE, identity_namespace="workbook_match_id",
            input_workbook_sha256=checksum, input_workbook_name=matches.name,
            team_directory_digest=directory["sha256"], selected_columns=list(SELECTED), counts=built.counts,
            selection_semantics=SEMANTICS, native_date_comparison="Europe/Berlin civil date; does not assert workbook timezone",
            acceptance="Unique exact league/season/ordered canonical teams/date only; scores never match identities",
            cross_date_policy="Review-only candidates; no proximity, unique-pair or score fallback",
            round_scope="Retained main-stage season ordinal; Scottish split keeps 34..38; Belgian reset substages use phase_round",
            unresolved_policy="All target rows retained with selected typed nulls; no coalescing",
            feature_asof_approved=False, model_input_approved=False, forecast_timing_changed=False,
            experiment_prepared=False, experiment_run=False,
            files={p.relative_to(stage).as_posix():dict(sha256=file_sha256(p), bytes=p.stat().st_size)
                   for p in sorted(stage.rglob("*")) if p.is_file()})
        (stage/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
        load_native_fixture_rounds(stage)
        os.rename(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(json.dumps(dict(output=str(output), **built.counts), indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", required=True, type=Path)
    parser.add_argument("--native-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    publish(args.matches, args.native_root, args.output)
