"""Verified reuse and versioned daily publications, joined only by event ID."""
from __future__ import annotations

from collections import defaultdict
from contextlib import closing
from copy import deepcopy
import gc
import json
from pathlib import Path
import shutil
import sqlite3
from tempfile import TemporaryDirectory
import time

from scraping.sofascore.campaign import publish_season
from scraping.sofascore.config import Profile, Scope
from scraping.sofascore.export import read_tables
from scraping.sofascore.parsers import parse_match
from scraping.sofascore.planner import component_jobs
from scraping.sofascore.schema import to_arrow
from scraping.sofascore.store import json_bytes
from .campaign import collector_lock
from .publish import contained_file, hash_file, read_json
from .import_har import _check_event
from .reconcile_publication import IDENTITY, _bundle, _copy_files, _verify_files
from .status_write import atomic_write_status

SNAPSHOTS = {"lineup_teams", "lineup_players", "player_statistics", "availability"}
COMPONENT = {"statistics": "statistics", "shots": "shotmap", "heatmap_points": "heatmap",
             "pregame": "pregame", "incidents": "incidents", "goal_sequences": "incidents",
             "goal_actions": "incidents", "passing_edges": "incidents"}


def source_snapshot(root, stem):
    root = Path(root).resolve()
    entry = root / f"{stem}.manifest.json"
    if not entry.exists() and not (root / f"{stem}.parquet").exists():
        return None, {}
    outer = read_json(entry)
    path = contained_file(root / outer["manifest"], root)
    manifest = read_json(path)
    rows = read_tables(path, names=["matches"])["matches"]
    expected = {r["event_id"]: tuple(r[k] for k in IDENTITY) for r in rows}
    _bundle(root, stem, expected)
    return {"manifest": str(path), "sha256": hash_file(path), "root": str(root),
            "stem": stem, "profile": manifest["profile"]}, {r["event_id"]: r for r in rows}


def compatible_profile(saved, requested):
    prior, wanted = Profile(**saved), Profile(**requested)
    for name in ("include_pregame", "include_statistics", "include_shotmap", "include_heatmaps",
                 "include_lineups", "include_injuries", "include_goal_actions"):
        if getattr(wanted, name) and not getattr(prior, name):
            return False
    return prior.locale == wanted.locale and (wanted.duration_mode != "incidents" or prior.duration_mode == "incidents")


def same_finished_match(row, event):
    current = parse_match(event)
    if tuple(row[k] for k in IDENTITY) != tuple(current[k] for k in IDENTITY):
        raise ValueError(f"Saved fixture identity mismatch: {event['id']}")
    if row.get("status") not in {"finished", "awarded"} or event.get("status", {}).get("type") != "finished":
        return False
    # Listing responses can omit detail-only fields. Compare observable facts,
    # not absence of a field in the more compact round listing.
    keys = ["status", "round", "kickoff_utc"] + [key for key in current if "_score_" in key]
    if "isAwarded" in event:
        keys.append("is_awarded")
    return row.get("is_awarded") is not None and all(current.get(k) is None or row.get(k) == current[k] for k in keys)


def plan_reuse(stem, events, config):
    sources, saved, coverages = [], {}, []
    for root in dict.fromkeys([config["output_root"], config["history_root"]]):
        source, rows = source_snapshot(root, stem)
        if source is None:
            continue
        index = len(sources)
        sources.append(source)
        coverage = defaultdict(list)
        for row in read_tables(source["manifest"], names=["coverage"]).get("coverage", []):
            coverage[row["event_id"]].append(row)
        coverages.append(coverage)
        for eid, row in rows.items():
            # Prefer a verified final result over an older unfinished daily row.
            if eid not in saved or (saved[eid][1].get("status") not in {"finished", "awarded"}
                                    and row.get("status") in {"finished", "awarded"}):
                saved[eid] = (index, row)
    reused = []
    profile = Profile(**config["profile"])
    for event in events:
        entry = saved.get(event["id"])
        if not entry or not same_finished_match(entry[1], event) or not compatible_profile(sources[entry[0]]["profile"], config["profile"]):
            continue
        row = entry[1]
        # A recent result may acquire corrected statistics after its first final
        # observation. Follow the shared settling policy, without duplicating it.
        now = time.time()
        if (0 <= now - row["kickoff_utc"] <= profile.settling_window_hours * 3600 and
                now - (row.get("event_metadata_observed_at") or 0) >= profile.settling_delay_seconds):
            continue
        coverage = coverages[entry[0]][event["id"]]
        if all(any(c["component"] == job.component and
                   c.get("side") == (str(job.team_id) if job.team_id else None) and
                   c["status"] in {"success", "cached_last_good", "empty", "unavailable"}
                   for c in coverage) for job in component_jobs(event, profile, now)):
            reused.append(event["id"])
    return {"sources": sources, "reused_ids": sorted(reused)}


def _read_source(source):
    path = Path(source["manifest"])
    if hash_file(path) != source["sha256"]:
        raise ValueError("A saved source manifest changed after the daily plan")
    return read_tables(path, verify_hashes=True)


def merge_capture(tables, captured):
    """Replace one match row; keep distinct lineup snapshots and last-good data."""
    if len(captured.get("matches", [])) != 1:
        raise ValueError("A capture must contain exactly one match")
    match = captured["matches"][0]
    eid = match["event_id"]
    prior = next((r for r in tables["matches"] if r["event_id"] == eid), None)
    if prior and tuple(prior[k] for k in IDENTITY) != tuple(match[k] for k in IDENTITY):
        raise ValueError("An update changed fixture identity")
    newest = deepcopy(captured)
    unavailable = [r for r in newest.get("coverage", []) if r.get("latest_attempt_status") == "unavailable"]
    preserved = set()
    for name in set(tables) | set(newest):
        old = tables.get(name, [])
        incoming = newest.get(name, [])
        if name in SNAPSHOTS:
            seen = {json_bytes(r) for r in old}
            tables[name] = old.copy()
            for row in incoming:
                key = json_bytes(row)
                if key not in seen:
                    tables[name].append(row)
                    seen.add(key)
        elif name in COMPONENT:
            retained = []
            for row in old:
                if row.get("event_id") != eid:
                    retained.append(row)
                    continue
                for missing in unavailable:
                    if missing["component"] == COMPONENT[name] and (
                            name != "heatmap_points" or str(row.get("team_id")) == missing.get("side")):
                        retained.append(row)
                        preserved.add((missing["component"], missing.get("side")))
                        break
            tables[name] = retained + incoming
        else:
            tables[name] = [r for r in old if r.get("event_id") != eid] + incoming
    for row in tables.get("coverage", []):
        if row.get("event_id") == eid and (row.get("component"), row.get("side")) in preserved:
            row["status"] = "cached_last_good"


def _split_matches(tables):
    """One pass over large point/player tables instead of scanning per match."""
    grouped = defaultdict(lambda: defaultdict(list))
    for name, rows in tables.items():
        for row in rows:
            if row.get("event_id") is not None:
                grouped[row["event_id"]][name].append(row)
            elif name != "team_seasons":
                raise ValueError(f"Missing event ID in {name}")
    return grouped


def assemble_season(cohort, reports, profile, items):
    chosen, season_rows, table_names = {}, [], {"matches"}
    for source in cohort["sources"]:
        current = _read_source(source)
        table_names.update(current)
        season_rows.extend(current.get("team_seasons", []))
        grouped = _split_matches(current)
        for eid, match_tables in grouped.items():
            if len(match_tables.get("matches", [])) != 1:
                raise ValueError("Duplicate or orphan saved match IDs")
            row = match_tables["matches"][0]
            if (row["competition_id"], row["season_id"]) != (cohort["competition_id"], cohort["season_id"]):
                raise ValueError("Saved season has a different provider identity")
            if eid not in chosen:
                chosen[eid] = match_tables
            elif chosen[eid]["matches"][0].get("status") not in {"finished", "awarded"} and row.get("status") in {"finished", "awarded"}:
                merge_capture(chosen[eid], match_tables)
        del current, grouped
    for eid in cohort["event_ids"]:
        if eid in reports:
            report_path = Path(reports[eid])
            report = read_json(report_path)
            if report.get("complete") is not True:
                raise ValueError(f"Daily capture {eid} is incomplete")
            manifest = contained_file(report["manifest"], report_path.parent)
            specification = read_json(manifest)
            if specification.get("profile") != profile or specification.get("requested_components_terminal") is not True:
                raise ValueError("Daily capture has an incomplete or different profile")
            reference = read_json(contained_file(report_path.parent / "reference.json", report_path.parent))
            _check_event(reference["event"], items[eid]["reference"]["event"], allow_unfinished=True)
            expected = {j.path for j in component_jobs(reference["event"], Profile(**profile), reference["observed_at"])} | {f"/event/{eid}"}
            coverage = report.get("coverage", [])
            if len(coverage) != len(expected) or {c["path"] for c in coverage} != expected or any(
                    c["status"] not in {"success", "empty", "unavailable"} for c in coverage):
                raise ValueError("Incomplete daily requested-component accounting")
            captured = read_tables(manifest, verify_hashes=True)
            if [r["event_id"] for r in captured.get("matches", [])] != [eid]:
                raise ValueError("Daily capture has a different match ID")
            if tuple(captured["matches"][0][k] for k in IDENTITY) != tuple(parse_match(reference["event"])[k] for k in IDENTITY):
                raise ValueError("Daily capture has a different fixture identity")
            for name, rows in captured.items():
                if len(rows) != specification["tables"][name]["rows"] or any(r.get("event_id") != eid for r in rows):
                    raise ValueError("Daily capture has orphan or mismatched rows")
                if name == "comments" and rows:
                    raise ValueError("Unexpected retained commentary")
            table_names.update(captured)
            merge_capture(chosen.setdefault(eid, {"matches": []}), captured)
        elif eid not in cohort["reused_ids"]:
            raise ValueError(f"Daily capture is missing for {eid}")
    # Drop stale unplayed fixtures outside this run's selected rounds, preserving
    # their prior version. All previously completed history remains available.
    keep = set(cohort["event_ids"]) | {eid for eid, t in chosen.items() if t["matches"][0].get("status") in {"finished", "awarded"}}
    order = sorted((t["matches"][0] for eid, t in chosen.items() if eid in keep),
                   key=lambda r: (r.get("kickoff_utc") or float("inf"), r["event_id"]))
    ids = [r["event_id"] for r in order]
    if len(ids) != len(set(ids)) or not set(cohort["event_ids"]) <= set(ids):
        raise ValueError("Duplicate or missing match IDs after daily merge")
    tables = {name: [row for eid in ids for row in chosen[eid].get(name, [])] for name in sorted(table_names)}
    if "team_seasons" in table_names:
        tables["team_seasons"] = list({json_bytes(row): row for row in season_rows}.values())
    # Additive match columns must also be present in reused older captures.
    tables["matches"] = to_arrow("matches", tables["matches"]).to_pylist()
    return dict(tables)


def install_season(tables, output, stem, scope, profile, provenance, *, expected_prior=None):
    """Use the normal versioned format, with guarded public-pointer replacement."""
    output = Path(output).resolve()
    expected = {r["event_id"]: tuple(r[k] for k in IDENTITY) for r in tables["matches"]}
    with collector_lock(output / "_collection/chrome_publication.lock"):
        prior, _ = source_snapshot(output, stem)
        current_hash = prior["sha256"] if prior else None
        current_provenance = read_json(prior["manifest"]).get("provenance", {}) if prior else {}
        own_partial = isinstance(current_provenance, dict) and current_provenance.get("cohort_sha256") == provenance["cohort_sha256"]
        if current_hash != expected_prior and not own_partial:
            raise ValueError("Daily output changed since this run was planned; prepare a fresh discovery before merging")
        if prior:
            old = _read_source(prior)
            if old == tables:
                return {"manifest": prior["manifest"], "season_file": str(output / f"{stem}.parquet"), "state": "unchanged"}
            previous_expected = {r["event_id"]: tuple(r[k] for k in IDENTITY) for r in old["matches"]}
            before = _bundle(output, stem, previous_expected)
            provenance["previous_manifest"] = str(Path(prior["manifest"]).relative_to(output))
            del old
        else:
            before = None
        with TemporaryDirectory(prefix="xdiyo_daily_publish_") as temporary:
            stage, backup = Path(temporary) / "replacement", Path(temporary) / "prior"
            if before:
                for relative in before["mutable"]:
                    target = backup / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(output / relative, target)
            report = publish_season(tables, stage, scope, profile, stem, provenance)
            if read_tables(stage / report["manifest"]) != tables:
                raise ValueError("Daily publication changed assembled values")
            after = _bundle(stage, stem, expected)
            _copy_files(stage, output, {k: v for k, v in after["files"].items() if k not in after["mutable"]}, reuse_identical=True)
            if before:
                _verify_files(output, before["files"])
            else:
                if any((output / relative).exists() for relative in after["mutable"]):
                    raise ValueError("A daily publication appeared concurrently")
            try:
                for relative in after["mutable"]:
                    atomic_write_status(output / relative, (stage / relative).read_bytes())
                if _bundle(output, stem, expected)["files"] != after["files"]:
                    raise ValueError("Installed daily publication differs from staging")
            except BaseException:
                for relative in after["mutable"]:
                    if before:
                        atomic_write_status(output / relative, (backup / relative).read_bytes())
                    else:
                        (output / relative).unlink(missing_ok=True)
                raise
        return {"manifest": str(output / report["manifest"]), "season_file": str(output / report["season_file"]), "state": "published"}


def publish_daily_campaign(campaign_directory):
    directory = Path(campaign_directory).resolve()
    metadata = read_json(directory / "config.json").get("source_metadata") or {}
    if metadata.get("collection") != "daily_update":
        raise ValueError("Select a daily-update campaign")
    output = Path(metadata["output_root"]).resolve()
    with collector_lock(output / "_collection/daily_update.lock"):
        return _publish_daily_campaign(directory)


def _set_latest(output, publication_path, result):
    latest = output / "LATEST.json"
    if latest.exists():
        prior = read_json(latest)
        if prior.get("planned_at", 0) > result["planned_at"]:
            return  # Resuming an older completed run must not reactivate old predictions.
    atomic_write_status(latest, json_bytes({"publication": str(publication_path.resolve()),
        "sha256": hash_file(publication_path), "discovery_id": result["discovery_id"], "planned_at": result["planned_at"]}))


def _publish_daily_campaign(directory):
    directory = Path(directory).resolve()
    config = read_json(directory / "config.json")
    metadata = config.get("source_metadata") or {}
    if metadata.get("collection") != "daily_update":
        raise ValueError("Select a daily-update campaign")
    discovery = Path(metadata["discovery_directory"])
    cohort_path = discovery / "cohort.json"
    cohort = read_json(cohort_path)
    output = Path(metadata["output_root"]).resolve()
    publication_path = discovery / "publication.json"
    if cohort["discovery_id"] != metadata["discovery_id"] or cohort["profile"] != config["profile"]:
        raise ValueError("Daily campaign and cohort configuration disagree")
    # The frozen plan keeps its original location after a completed publication
    # is moved. Its hashes and relocated files are verified below; an unfinished
    # run must still publish to the directory against which reuse was planned.
    if not publication_path.exists() and Path(cohort["output_root"]).resolve() != output:
        raise ValueError("Daily campaign and cohort configuration disagree")
    selected_ids = [eid for season in cohort["seasons"] for eid in season["event_ids"]]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Duplicate match IDs across daily season cohorts")
    expected_queue = {eid for season in cohort["seasons"] for eid in season["event_ids"] if eid not in season["reused_ids"]}
    items = {item["event_id"]: item for item in config["items"]}
    if set(items) != expected_queue or set(cohort["queued_ids"]) != expected_queue:
        raise ValueError("Daily cohort does not account for every queued match")
    with closing(sqlite3.connect((directory / "queue.sqlite").as_uri() + "?mode=ro", uri=True)) as db:
        queue = list(db.execute("SELECT event_id,state,report FROM queue"))
    if {r[0] for r in queue} != set(cohort["queued_ids"]):
        raise ValueError("Daily cohort and queue IDs disagree")
    if any(state != "complete" for _, state, _ in queue):
        raise ValueError("Daily collection is not complete yet")
    if publication_path.exists():
        saved = read_json(publication_path)
        if saved["cohort_sha256"] != hash_file(cohort_path) or saved["campaign_id"] != config["campaign_id"]:
            raise ValueError("Saved publication belongs to a different daily cohort")
        for season in saved["seasons"]:
            if season.get("manifest"):
                manifest = contained_file(season["manifest"], output)
                if hash_file(manifest) != season["manifest_sha256"]:
                    raise ValueError("Saved daily publication manifest changed")
                read_tables(manifest, verify_hashes=True)
        _set_latest(output, publication_path, saved)
        return saved
    # A verified publication remains usable after its raw capture imports have
    # been cleaned up. Only an unpublished campaign needs those imports.
    reports = {eid: str(contained_file(report, directory / "imports")) for eid, _, report in queue}
    result = {"discovery_id": cohort["discovery_id"], "campaign_id": config["campaign_id"], "created_at": time.time(),
              "planned_at": cohort["created_at"], "cohort_sha256": hash_file(cohort_path), "seasons": [], "api_requests_sent_by_python": 0}
    for season in cohort["seasons"]:
        if not season["event_ids"]:
            result["seasons"].append({"stem": season["stem"], "state": season["state"], "history_ids": [], "prediction_ids": [], "unfinished_ids": []})
            continue
        tables = assemble_season(season, reports, config["profile"], items)
        years = cohort["years"]
        scope = Scope(season["competition_id"], season["season_id"], tuple(r["round"] for r in tables["matches"]),
                      league_name=season["league"], season_start=years[0], season_end=years[1])
        baseline = next((source["sha256"] for source in season["sources"] if Path(source["root"]).resolve() == output.resolve()), None)
        published = install_season(tables, output, season["stem"], scope, Profile(**config["profile"]),
            {"collection": "daily_update", "campaign_id": config["campaign_id"], "discovery_id": cohort["discovery_id"],
             "cohort_sha256": result["cohort_sha256"], "scope": "Observed completed history plus selected current/next regular round; not a claim that the season is finished.",
             "reused_matches": len(season["reused_ids"]), "last_completed_round": season["last_completed_round"], "target_round": season["target_round"]},
             expected_prior=baseline)
        rows = tables["matches"]
        result["seasons"].append({**published, "stem": season["stem"], "league_state": season["state"],
            "manifest_sha256": hash_file(published["manifest"]), "matches": len(rows),
            "history_ids": [r["event_id"] for r in rows if r["status"] in {"finished", "awarded"}],
            "prediction_ids": [r["event_id"] for r in rows if r["status"] == "notstarted" and (r.get("kickoff_utc") or 0) > time.time()],
            "unfinished_ids": [r["event_id"] for r in rows if r["status"] not in {"finished", "awarded"}]})
        del tables
        gc.collect()
    atomic_write_status(publication_path, json_bytes(result))
    # An all-finished/all-unavailable run also replaces this pointer. Consumers
    # therefore cannot accidentally keep predicting an older round.
    _set_latest(output, publication_path, result)
    return result


def read_daily_tables(output, *, view="prediction", now=None):
    """Load verified current selections; expired kickoff times are never predictions."""
    if view not in {"history", "prediction", "unfinished", "all"}:
        raise ValueError("Choose history, prediction, unfinished or all")
    now = time.time() if now is None else now
    output = Path(output).resolve()
    pointer = read_json(output / "LATEST.json")
    path = Path(pointer["publication"])
    if hash_file(path) != pointer["sha256"]:
        raise ValueError("Latest daily publication checksum mismatch")
    publication = read_json(path)
    if publication["discovery_id"] != pointer["discovery_id"]:
        raise ValueError("Latest daily discovery identity mismatch")
    result = {}
    for season in publication["seasons"]:
        if not season.get("manifest"):
            result[season["stem"]] = {"matches": []}
            continue
        manifest = contained_file(season["manifest"], output)
        if hash_file(manifest) != season["manifest_sha256"]:
            raise ValueError("Daily season manifest checksum mismatch")
        tables = read_tables(manifest, verify_hashes=True)
        selected = set(season[{"history": "history_ids", "prediction": "prediction_ids", "unfinished": "unfinished_ids"}[view]]) if view != "all" else {r["event_id"] for r in tables["matches"]}
        if view == "prediction":
            selected &= {r["event_id"] for r in tables["matches"] if r["status"] == "notstarted" and (r.get("kickoff_utc") or 0) > now}
        result[season["stem"]] = {name: [row for row in rows if row.get("event_id") in selected]
                                 for name, rows in tables.items() if name != "team_seasons"}
    return result
