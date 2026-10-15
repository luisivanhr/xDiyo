"""Offline, versioned season-entry flags and comparison with legacy outputs.

Match appearances establish observed participation. They do not certify that a
competition's membership is complete. Completeness and overrides require an
explicit evidence file; reference CSV flags are audited, never silently trusted.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import re
import time
import uuid

import pandas as pd
import pyarrow.parquet as pq

from utils.team_seasons import Membership, derive_team_seasons, join_team_season_flags
from .audit import hash_file
from .schema import to_arrow
from .store import atomic_write, json_bytes

DEFAULT_REGISTRY = Path(__file__).with_name("competition_registry.json")
IDENTITY = {"match_id", "season_id", "home_id", "away_id", "home_team", "away_team"}
FLAGS = {s+"_got_"+m for s in ("home", "away") for m in ("promoted", "demoted")}


def full_year(value):
    year = int(value)
    if year < 100:
        year += 2000  # This project contains 21st-century seasons only.
    return year


def inventory_memberships(data_dir, registry, confirmations=()):
    confirmations = list(confirmations)
    confirmed = {(c["competition_id"], c["season_id"]): c for c in confirmations}
    if len(confirmed) != len(confirmations):
        raise ValueError("Duplicate membership confirmation")
    memberships, sources, aliases, issues = [], [], defaultdict(set), []
    for path in sorted(Path(data_dir).glob("*.csv")):
        found = re.fullmatch(r"(.+)_(\d{2,4})_(\d{2,4})", path.stem)
        if not found:
            continue
        league, first, last = found.groups()
        if league not in registry:
            raise ValueError(f"Competition missing from registry: {league}")
        config = registry[league]
        frame = pd.read_csv(path, usecols=lambda c: c in IDENTITY | {"season_year"})
        if not IDENTITY.issubset(frame.columns) or frame.empty or frame.match_id.duplicated().any():
            raise ValueError(f"Invalid match identities: {path}")
        seasons = frame.season_id.dropna().unique()
        if len(seasons) != 1 or frame[list(IDENTITY)].isna().any().any():
            raise ValueError(f"Missing or mixed season/team identity: {path}")
        season_id, competition_id = int(seasons[0]), config["competition_id"]
        if "season_year" in frame:
            labels = frame.season_year.dropna().unique()
            expected = (full_year(first), full_year(last))
            actual = []
            for label in labels:
                parts = re.fullmatch(r"(\d{2,4})/(\d{2,4})", str(label))
                actual.append(tuple(full_year(p) for p in parts.groups()) if parts else None)
            if actual != [expected] or frame.season_year.isna().any():
                issues.append({"file": str(path.resolve()), "sha256": hash_file(path),
                               "status": "quarantined_season_identity_conflict", "rows": len(frame),
                               "filename_years": expected, "stored_years": actual, "season_id": season_id})
                continue
        team_ids = set()
        for side in ("home", "away"):
            for tid, slug in zip(frame[side+"_id"], frame[side+"_team"]):
                if int(tid) != tid:
                    raise ValueError("Team IDs must be integers")
                team_ids.add(int(tid))
                aliases[(competition_id, season_id, slug)].add(int(tid))
        digest = hash_file(path)
        confirmation = confirmed.pop((competition_id, season_id), None)
        if confirmation:
            if not confirmation.get("evidence") or set(confirmation["team_ids"]) != team_ids:
                raise ValueError("Complete membership evidence must enumerate exactly the observed team IDs")
        evidence = "legacy_match_appearances:sha256:"+digest
        if confirmation:
            evidence += ";complete_membership:"+confirmation["evidence"]
        membership = Membership(competition_id, season_id, full_year(first), full_year(last),
                                config["tier"], frozenset(team_ids), bool(confirmation), evidence, config["system"])
        memberships.append(membership)
        sources.append({"file": str(path.resolve()), "sha256": digest, "league": league,
                        "competition_id": competition_id, "season_id": season_id,
                        "start_year": membership.start_year, "end_year": membership.end_year,
                        "rows": len(frame), "team_count": len(team_ids)})
    if confirmed:
        raise ValueError(f"Membership confirmations refer to absent seasons: {sorted(confirmed)}")
    if not memberships:
        raise ValueError("No season match files found")
    return memberships, sources, aliases, issues


def resolve_slug_seeds(seeds, sources, aliases):
    """Pin notebook-style aliases to exact years and IDs; fail on ambiguity."""
    resolved = []
    for seed in seeds:
        seasons = [s for s in sources if s["league"] == seed["league"] and
                   s["start_year"] == seed["start_year"] and s["end_year"] == seed["end_year"]]
        if len(seasons) != 1:
            raise ValueError(f"Seed requires one exact season: {seed}")
        season = seasons[0]
        ids = aliases[(season["competition_id"], season["season_id"], seed["slug"])]
        if len(ids) != 1:
            raise ValueError(f"Seed alias is absent or ambiguous: {seed}")
        resolved.append({"competition_id": season["competition_id"], "season_id": season["season_id"],
                         "team_id": next(iter(ids)), "movement": seed["movement"], "evidence": seed["evidence"]})
    return resolved


def compare_legacy(rows, sources, legacy_dir):
    lookup = {(r["competition_id"], r["season_id"], r["team_id"]): r for r in rows}
    reports, totals = [], Counter()
    for source in sources:
        path = Path(legacy_dir)/Path(source["file"]).name
        if not path.exists():
            reports.append({"file": str(path), "status": "missing_reference"})
            continue
        old = pd.read_csv(path, usecols=lambda c: c in IDENTITY | FLAGS)
        if not (IDENTITY | FLAGS).issubset(old.columns) or old.match_id.duplicated().any():
            raise ValueError(f"Invalid legacy flag reference: {path}")
        base = pd.read_csv(source["file"], usecols=lambda c: c in IDENTITY)
        identity_check = base.merge(old[list(IDENTITY)], on="match_id", how="outer",
                                    validate="one_to_one", indicator=True, suffixes=("_source", "_reference"))
        if not identity_check._merge.eq("both").all() or any(
            not identity_check[c+"_source"].eq(identity_check[c+"_reference"]).all() for c in IDENTITY-{"match_id"}):
            raise ValueError(f"Legacy reference has different match identities: {path}")
        observations = defaultdict(set)
        for side in ("home", "away"):
            for tid, p, d in old[[side+"_id", side+"_got_promoted", side+"_got_demoted"]].itertuples(index=False, name=None):
                if p not in (0, 1) or d not in (0, 1):
                    raise ValueError("Legacy flag values must be zero or one")
                observations[int(tid)].add((bool(p), bool(d)))
        counts, differences = Counter(), []
        for tid, flags in sorted(observations.items()):
            current = lookup[(source["competition_id"], source["season_id"], tid)]
            new = (current["got_promoted"], current["got_demoted"])
            status = "inconsistent_legacy" if len(flags) != 1 else "unresolved" if current["movement"] == "unknown" else "agreement" if new in flags else "changed"
            counts[status] += 1
            if status != "agreement":
                differences.append({"team_id": tid, "status": status, "legacy_flags": sorted(flags),
                                    "new_flags": new, "movement": current["movement"], "reason": current["evidence"]})
        totals.update(counts)
        reports.append({"file": str(path.resolve()), "sha256": hash_file(path),
                        "competition_id": source["competition_id"], "season_id": source["season_id"],
                        "counts": dict(counts), "differences": differences})
    return {"unit": "unique team per competition-season", "counts": dict(totals), "files": reports}


def build_movements(data_dir, output_dir, *, registry_path=DEFAULT_REGISTRY, evidence_path=None,
                    legacy_dir=None, leagues=None):
    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    evidence = json.loads(Path(evidence_path).read_text(encoding="utf-8")) if evidence_path else {}
    if set(evidence)-{"complete_memberships", "overrides", "slug_seeds"}:
        raise ValueError("Unknown movement evidence keys")
    memberships, sources, aliases, issues = inventory_memberships(data_dir, registry, evidence.get("complete_memberships", []))
    overrides = [*evidence.get("overrides", []), *resolve_slug_seeds(evidence.get("slug_seeds", []), sources, aliases)]
    rows = derive_team_seasons(memberships, overrides)
    if leagues:
        selected = {registry[name]["competition_id"] for name in leagues}
        rows = [r for r in rows if r["competition_id"] in selected]
        sources = [s for s in sources if s["competition_id"] in selected]
        memberships = [m for m in memberships if m.competition_id in selected]
    table = to_arrow("team_seasons", rows)
    comparison = compare_legacy(rows, sources, legacy_dir) if legacy_dir else None
    root = Path(output_dir).resolve()
    version = root/"versions"/uuid.uuid4().hex
    version.mkdir(parents=True, exist_ok=False)
    table_path = version/"team_seasons.parquet"
    pq.write_table(table, table_path, compression="zstd")
    if not pq.read_table(table_path).equals(table):
        raise ValueError("Team-season Parquet validation failed")
    report = {"api_requests": 0, "team_season_rows": len(rows), "seasons": len(sources), "identity_issues": issues,
              "movement_counts": dict(Counter(r["movement"] for r in rows)), "legacy_comparison": comparison,
              "limitations": ["Match appearance evidence is retrospective, not a verified pregame observation.",
                              "No unobserved team is treated as absent without explicit complete membership evidence."]}
    atomic_write(version/"audit.json", json_bytes(report))
    membership_records = [{**asdict(m), "teams": sorted(m.teams)} for m in memberships]
    atomic_write(version/"membership_registry.json", json_bytes(membership_records))
    manifest = {"schema_version": "1", "created_at_utc": time.time(), "sources": sources,
                "registry_sha256": hash_file(registry_path), "evidence_sha256": hash_file(evidence_path) if evidence_path else None,
                "resolved_overrides": overrides, "rows": len(rows), "files": {
                    name: {"file": name, "sha256": hash_file(version/name)}
                    for name in ("team_seasons.parquet", "audit.json", "membership_registry.json")}}
    atomic_write(version/"manifest.json", json_bytes(manifest))
    atomic_write(root/"CURRENT.json", json_bytes({"manifest": (version/"manifest.json").relative_to(root).as_posix()}))
    return version/"manifest.json", report


def read_flags(path):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if "manifest" in manifest:
        path = path.parent/manifest["manifest"]
        manifest = json.loads(path.read_text(encoding="utf-8"))
    info = manifest["files"]["team_seasons.parquet"]
    table_path = path.parent/info["file"]
    if hash_file(table_path) != info["sha256"]:
        raise ValueError("Team-season file hash mismatch")
    return pq.read_table(table_path).to_pylist()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-dir", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args(argv)
    path, report = build_movements(args.input, args.output, registry_path=args.registry,
                                   evidence_path=args.evidence, legacy_dir=args.legacy_dir)
    print(json.dumps({"manifest": str(path), **{k: v for k, v in report.items() if k != "legacy_comparison"},
                      "legacy_counts": (report["legacy_comparison"] or {}).get("counts")}, indent=2))


if __name__ == "__main__":
    main()
