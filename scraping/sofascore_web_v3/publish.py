"""Assemble complete Chrome capture cohorts offline into same-stem season files.

The active queue is opened read-only. No browser, API request, or source-store
mutation occurs. Existing season outputs are verified or refused, never replaced.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time

from scraping.sofascore.campaign import publish_season
from scraping.sofascore.config import PARSER_VERSION, Profile, Scope
from scraping.sofascore.export import read_tables
from scraping.sofascore.planner import component_jobs
from scraping.sofascore.schema import SCHEMA_VERSION
from scraping.sofascore.store import atomic_write, json_bytes, season_year_pair
from .campaign import collector_lock
from .awarded import SavedAwardedFlags

ROOT = Path(__file__).resolve().parents[2]
TERMINAL = {"success", "empty", "unavailable"}


def hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def contained_file(path, root):
    path = Path(path).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Source file must exist inside its capture directory: {path}")
    return path


def completed_season_stems(campaign_directory):
    """List complete queue cohorts without changing the live campaign."""
    directory = Path(campaign_directory).resolve()
    config = read_json(directory / "config.json")
    if (config.get("source_metadata") or {}).get("collection") == "daily_update":
        return []  # The daily runner merges reused history and unfinished fixtures.
    with closing(sqlite3.connect((directory / "queue.sqlite").as_uri() + "?mode=ro", uri=True)) as connection:
        states = dict(connection.execute("SELECT event_id,state FROM queue"))
    cohorts = defaultdict(list)
    for item in config["items"]:
        stem = item["reference"].get("season_stem")
        if stem:
            cohorts[stem].append(states.get(item["event_id"]))
    return [stem for stem, values in cohorts.items() if values and all(state == "complete" for state in values)]


def snapshot_season(campaign_directory, stem, *, published_output=None):
    directory = Path(campaign_directory).resolve()
    if not re.fullmatch(r"[A-Za-z0-9_]+_\d{2}_\d{2}", stem):
        raise ValueError("Use the original league and season filename stem")
    config = read_json(directory / "config.json")
    if (config.get("source_metadata") or {}).get("collection") == "daily_update":
        raise ValueError("Daily updates must use publish_daily_campaign, including their reused history")
    if (config.get("source_metadata") or {}).get("collection") == "component_recovery":
        raise ValueError("Component-only recovery is not a full season; use the recovery merger instead")
    profile = Profile(**config["profile"])
    if profile.include_comments:
        raise ValueError("Chrome publication does not retain commentary")
    items = [item for item in config["items"] if item["reference"].get("season_stem") == stem]
    if not items or len({item["event_id"] for item in items}) != len(items):
        raise ValueError("Select one nonempty queued season with distinct event IDs")
    identities = {(item["competition_id"], item["season_id"], season_year_pair(item["season_year"])) for item in items}
    if len(identities) != 1:
        raise ValueError("Season filename maps to multiple provider identities")
    competition_id, season_id, years = identities.pop()
    if years is None or tuple(2000 + int(value) for value in stem.rsplit("_", 2)[1:]) != years:
        raise ValueError("Season filename and provider year disagree")
    connection = sqlite3.connect((directory / "queue.sqlite").as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        queue = {row["event_id"]: dict(row) for row in connection.execute(
            "SELECT event_id,state,report FROM queue")}
    finally:
        connection.close()
    states = Counter(queue.get(item["event_id"], {}).get("state", "absent") for item in items)
    if states != {"complete": len(items)}:
        raise ValueError(f"Refusing an unfinished season cohort: {dict(states)}")
    sources = []
    imports_available = (directory / "imports").exists()
    if not imports_available and published_output is not None:
        # Raw imports may be explicitly deleted after conversion. Verify the
        # retained bundle and its original lineage before reusing that snapshot.
        # This path cannot create a new publication from missing captures.
        from .reconcile_publication import _bundle

        output = Path(published_output).resolve()
        prior = read_json(contained_file(output / f"{stem}.manifest.json", output))
        if prior.get("profile") != asdict(profile):
            raise ValueError("Saved publication profile differs from the queue profile")
        expected = {item["event_id"]: (item["event_id"], item["competition_id"], item["season_id"],
                    item["reference"]["event"]["homeTeam"]["id"],
                    item["reference"]["event"]["awayTeam"]["id"]) for item in items}
        _bundle(output, stem, expected)
        lineage = read_json(contained_file(output / prior["reconciliation"]["source_lineage"], output))
        sources = lineage.get("sources", [])
        if (Path(lineage.get("campaign_directory", "")).resolve() != directory or lineage.get("stem") != stem or
                [source.get("event_id") for source in sources] != [item["event_id"] for item in items]):
            raise ValueError("Existing source lineage mismatch")
        for source in sources:
            report = Path(source["report"]).resolve()
            if not report.is_relative_to(directory / "imports") or report != Path(queue[source["event_id"]]["report"]).resolve():
                raise ValueError("Saved source lineage differs from the completed queue")
    items_to_read = items if imports_available or published_output is None else []
    for item in items_to_read:
        report_path = contained_file(queue[item["event_id"]]["report"], directory / "imports")
        report = read_json(report_path)
        manifest_path = contained_file(report["manifest"], report_path.parent)
        manifest = read_json(manifest_path)
        if report.get("complete") is not True or manifest.get("requested_components_terminal") is not True:
            raise ValueError(f"Incomplete captured export for {item['event_id']}")
        if manifest.get("profile") != asdict(profile):
            raise ValueError("Captured profile differs from the queue profile")
        if (manifest.get("schema_version"), manifest.get("parser_version")) != (SCHEMA_VERSION, PARSER_VERSION):
            raise ValueError("Captured schema/parser version needs an explicit migration")
        expected_event = {**item["reference"]["event"], "status": {"type": "finished"}}
        expected_paths = {job.path for job in component_jobs(expected_event, profile, time.time())}
        expected_paths.add(f"/event/{item['event_id']}")
        coverage = report["coverage"]
        if len(coverage) != len(expected_paths) or {entry["path"] for entry in coverage} != expected_paths or any(
            entry["status"] not in TERMINAL for entry in coverage
        ):
            raise ValueError(f"Incomplete requested-component accounting for {item['event_id']}")
        for info in manifest["tables"].values():
            table_path = contained_file(manifest_path.parent / info["file"], manifest_path.parent)
            if hash_file(table_path) != info["sha256"]:
                raise ValueError("Captured table hash mismatch")
        feature_path = contained_file(report_path.parent / "feature_coverage.json", report_path.parent)
        sources.append({"event_id": item["event_id"], "report": str(report_path),
                        "report_sha256": hash_file(report_path), "manifest": str(manifest_path),
                        "manifest_sha256": hash_file(manifest_path), "feature_coverage": str(feature_path),
                        "feature_coverage_sha256": hash_file(feature_path), "tables": manifest["tables"]})
    fingerprint = hashlib.sha256(json_bytes({"campaign_id": config["campaign_id"], "stem": stem,
        "profile": asdict(profile), "identities": [competition_id, season_id, years], "sources": sources})).hexdigest()
    return directory, config, profile, items, sources, fingerprint


def publish_captured_season(campaign_directory, stem, *, output=ROOT / "data/xDiyo_data"):
    directory, config, profile, items, sources, fingerprint = snapshot_season(campaign_directory, stem, published_output=output)
    output = Path(output).resolve()
    season_path, report_path = output / f"{stem}.parquet", output / f"{stem}.manifest.json"
    # Separate from the live collection lock; only offline publishers share it.
    with collector_lock(output / "_collection/chrome_publication.lock"), SavedAwardedFlags(
            [directory / "queue.sqlite"]) as awarded:
        if season_path.exists() or report_path.exists():
            if not season_path.is_file() or not report_path.is_file():
                raise ValueError("Existing incomplete output pair; inspect it before publication")
            prior = read_json(report_path)
            if prior.get("reconciliation", {}).get("chrome_source_fingerprint") != fingerprint:
                raise ValueError("Existing season output has different provenance; it will not be replaced")
            if hash_file(season_path) != prior.get("sha256"):
                raise ValueError("Existing season Parquet hash mismatch")
            lineage_path = contained_file(output / prior["reconciliation"]["source_lineage"], output)
            lineage = read_json(lineage_path)
            if lineage.get("fingerprint") != fingerprint or lineage.get("sources") != sources:
                raise ValueError("Existing source lineage mismatch")
            canonical = contained_file(output / prior["manifest"], output)
            saved = read_tables(canonical, verify_hashes=True)
            if [row["event_id"] for row in saved.get("matches", [])] != [item["event_id"] for item in items]:
                raise ValueError("Existing season export changed the captured cohort")
            return {**prior, "publication_state": "already_published", "output": str(output)}

        tables, feature_counts = defaultdict(list), defaultdict(Counter)
        for item, source in zip(items, sources):
            eid = item["event_id"]
            for name in ("report", "manifest", "feature_coverage"):
                if hash_file(source[name]) != source[name + "_sha256"]:
                    raise ValueError("Capture source changed after the publication snapshot")
            captured = read_tables(source["manifest"], verify_hashes=True)
            if len(captured.get("matches", [])) != 1:
                raise ValueError(f"Expected one captured match for {eid}")
            match = captured["matches"][0]
            reference = item["reference"]["event"]
            if (match["event_id"], match["competition_id"], match["season_id"], match["home_id"], match["away_id"], match["status"]) != (
                eid, item["competition_id"], item["season_id"], reference["homeTeam"]["id"], reference["awayTeam"]["id"], "finished"):
                raise ValueError(f"Captured fixture identity mismatch for {eid}")
            # Older schema-2 imports predate this additive field. Upgrade the
            # assembled row from saved metadata without rewriting its source.
            if "is_awarded" not in match:
                match["is_awarded"] = awarded.get(match).value
            for name, rows in captured.items():
                if len(rows) != source["tables"][name]["rows"]:
                    raise ValueError(f"Captured table row count mismatch for {name}")
                if name == "comments" and rows:
                    raise ValueError("Unexpected retained commentary")
                if any(row.get("event_id") != eid for row in rows):
                    raise ValueError(f"Orphan or cross-match rows in {name}")
                tables[name].extend(rows)
            for name, value in read_json(source["feature_coverage"])["features"].items():
                feature_counts[name][value["status"]] += 1
        first = items[0]
        years = season_year_pair(first["season_year"])
        scope = Scope(first["competition_id"], first["season_id"], tuple(row["round"] for row in tables["matches"]),
                      league_name=stem.rsplit("_", 2)[0], season_start=years[0], season_end=years[1])
        reconciliation = {"collection": "chrome_campaign", "campaign_id": config["campaign_id"],
            "chrome_source_fingerprint": fingerprint, "expected_event_ids": [item["event_id"] for item in items],
            "missing_expected_ids": [], "extra_event_ids": [], "verified_capture_imports": len(items),
            "scope": "Complete saved fixture cohort; provider-wide fixture discovery was not repeated.",
            "source_lineage": f"_collection/chrome_exports/{stem}.{fingerprint}.json",
            "feature_status_counts": dict(feature_counts),
            "standings": "Raw pregame positions retained; normalize over the assembled season when loading.",
            "promotion_relegation_flags": "Separate season-membership derivation; not inferred here."}
        if (config.get("source_metadata") or {}).get("collection") == "season_discovery":
            curtailed = any(s.get("competition_id") == first["competition_id"]
                            and s.get("schedule_validation") == "curtailed_final_standings"
                            for s in config["source_metadata"].get("seasons", []))
            reconciliation["scope"] = (
                "Browser-discovered regular season: every advertised numbered round was enumerated; "
                + ("played fixtures reconciled with the final standings and verified curtailed-season totals."
                   if curtailed else "stage exclusions and team-pair completeness were checked before queue creation."))
        if (config.get("source_metadata") or {}).get("collection") == "archive_discovery":
            cohorts = [s for s in config["source_metadata"]["seasons"]
                       if s.get("competition_id") == first["competition_id"]
                       and s.get("season_id") == first["season_id"] and s.get("stem") == stem]
            if len(cohorts) != 1 or not cohorts[0].get("schedule_verified"):
                raise ValueError("Missing verified archive cohort in campaign metadata")
            cohort = cohorts[0]
            if cohort["selected_matches"] != len(items):
                raise ValueError("Archive publication differs from the discovered fixture selection")
            reconciliation["scope"] = (
                "Completed regular-season snapshot at discovery time; reconciled with current standings."
                if cohort["snapshot"] else "Completed fixtures from the verified historical regular-season selection.")
            # Use the normal publication manifest; no extra columns or reports.
            reconciliation["archive_cohort"] = cohort
        # Validate everything before creating the season artifact. Refuse even
        # an unexpected new output that appeared while inputs were checked.
        if season_path.exists() or report_path.exists():
            raise ValueError("A season output appeared during validation; it will not be replaced")
        atomic_write(output / reconciliation["source_lineage"], json_bytes({
            "campaign_directory": str(directory), "stem": stem, "fingerprint": fingerprint, "sources": sources}))
        result = publish_season(dict(tables), output, scope, profile, stem, reconciliation)
        canonical = output / result["manifest"]
        published = read_tables(canonical, verify_hashes=True)
        if published != dict(tables):
            raise ValueError("Assembled season differs from the verified capture tables")
        return {**result, "publication_state": "published", "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data/xDiyo_data")
    args = parser.parse_args()
    report = publish_captured_season(args.campaign, args.stem, output=args.output)
    print(json.dumps({key: report[key] for key in ("publication_state", "season_file", "matches", "sha256", "output")}))


if __name__ == "__main__":
    main()
