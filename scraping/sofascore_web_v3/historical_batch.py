"""Prepare several historical discovery runs and use the existing converter.

The extension handles discovery and capture, one season at a time. This helper
only prepares local state and watches it; it never sends provider requests or
starts a browser. Restarting reuses the same discovery and collection IDs.
"""
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path
import time

from scraping.sofascore.store import json_bytes, season_year_pair
from .campaign import ROOT, campaign_path, collector_lock, read_json
from .capture import PROFILE
from .discovery import DISCOVERY_BASE, normalize_leagues, open_discovery, prepare_discovery
from .publish_watch import watch_publications
from .status_write import atomic_write_status

BATCH_BASE = ROOT / "data/website_capture_v3/batches"


def group_requests(requests):
    """Normalize a league/season selection before any files are created."""
    leagues = normalize_leagues(requests)
    groups = defaultdict(list)
    for league, request in zip(leagues, requests):
        seasons = request.get("seasons")
        if not isinstance(seasons, list) or not seasons:
            raise ValueError("Each league needs a nonempty seasons list")
        seen = set()
        for season in seasons:
            years = season_year_pair(season)
            if years is None or years[1] != years[0] + 1:
                raise ValueError("Use explicit historical seasons such as 15/16")
            if years in seen:
                raise ValueError("Duplicate league/season selection")
            seen.add(years)
            groups[years].append(league)
    return [{"season": f"{a % 100:02d}/{b % 100:02d}", "years": [a, b], "leagues": leagues,
             "stems": [f"{league['name']}_{a % 100:02d}_{b % 100:02d}" for league in leagues]}
            for (a, b), leagues in sorted(groups.items())]


def prepare_historical_batch(requests, *, name, include_lineups=True, random_delay_max_seconds=None,
                             output=ROOT / "data/xDiyo_data", base=BATCH_BASE,
                             discovery_base=DISCOVERY_BASE):
    groups = group_requests(requests)
    if type(include_lineups) is not bool:
        raise ValueError("include_lineups must be boolean")
    from .campaign import validate_policy
    validate_policy({"random_action_delay_max_seconds": random_delay_max_seconds})
    directory = campaign_path(name, base)
    path = directory / "batch.json"
    settings = {"groups": groups, "include_lineups": include_lineups,
        "random_delay_max_seconds": random_delay_max_seconds,
        "output": str(Path(output).resolve()), "discovery_base": str(Path(discovery_base).resolve())}
    with collector_lock(directory / "prepare.lock"):
        saved = read_json(path) if path.exists() else {"settings": settings, "runs": []}
        if saved["settings"] != settings:
            raise ValueError("Saved batch settings differ; resume the original settings or choose a new batch name")
        known = {run["season"]: run for run in saved["runs"]}
        if len(known) != len(saved["runs"]) or not set(known) <= {g["season"] for g in groups}:
            raise ValueError("Saved batch contains duplicate or unexpected seasons")
        # Fail before creating any new discovery if a requested output already
        # belongs to a different run. Published outputs of this batch are valid.
        for group in groups:
            if group["season"] not in known:
                for stem in group["stems"]:
                    if any((Path(output) / f"{stem}{suffix}").exists() for suffix in (".parquet", ".manifest.json")):
                        raise ValueError(f"Existing output for {stem}; use its saved run or the daily update workflow")
        for group in groups:
            if group["season"] in known:
                run = known[group["season"]]
                config = read_json(campaign_path(run["discovery_id"], discovery_base) / "config.json")
                if (run["stems"] != group["stems"] or config["years"] != group["years"]
                        or config["leagues"] != group["leagues"] or config.get("mode") != "season"
                        or config["output_root"] != settings["output"]
                        or config["profile"] != asdict(replace(PROFILE, include_lineups=include_lineups))
                        or config["policy"]["random_action_delay_max_seconds"] != random_delay_max_seconds):
                    raise ValueError("Saved discovery no longer matches this batch")
                continue
            prepared = prepare_discovery(season=group["season"], leagues=group["leagues"],
                include_lineups=include_lineups, random_delay_max_seconds=random_delay_max_seconds,
                output_root=output, base=discovery_base)
            saved["runs"].append({"season": group["season"], "discovery_id": prepared["discovery_id"],
                                  "stems": group["stems"]})
            atomic_write_status(path, json_bytes(saved))
    return {**saved, "path": str(path)}


def print_batch(batch, *, emit=print):
    emit(f"Output: {batch['settings']['output']}")
    emit(f"Saved batch: {batch['path']}")
    for run in batch["runs"]:
        emit(f"{run['season']} | {run['discovery_id']} | {len(run['stems'])} leagues")


def watch_historical_batch(batch, *, season=None, once=False, poll_seconds=5, emit=print, sleep=time.sleep):
    """Convert each run in year order; --season can resume one independently."""
    if not 1 <= poll_seconds <= 60:
        raise ValueError("Use a polling interval between 1 and 60 seconds")
    runs = [r for r in batch["runs"] if season is None or r["season"] == season]
    if not runs:
        raise ValueError("Season is not in this saved batch")
    results = []
    for run in runs:
        directory = campaign_path(run["discovery_id"], batch["settings"]["discovery_base"])
        emit(f"\n{run['season']}: paste Discovery ID {run['discovery_id']} into xDiyo.")
        last = None
        while True:
            with open_discovery(directory, read_only=True) as discovery:
                summary = discovery.status()
                pause = discovery.db.execute("SELECT value FROM state WHERE key='pause'").fetchone()
            signature = json_bytes({"jobs": summary["jobs"], "ready": summary["ready_to_collect"],
                                    "pause": pause[0] if pause else None})
            if signature != last:
                emit(f"Discovery: {summary['jobs']}; ready to collect: {summary['ready_to_collect']}")
                if pause:
                    emit(f"Paused: {pause[0]}")
                if summary["discovery_complete"] and not summary["ready_to_collect"]:
                    emit(f"Discovery needs review: {summary['seasons']}")
                if summary["ready_to_collect"] and not summary.get("campaign"):
                    emit("In Chrome, select Create collection campaign, then Start / resume.")
                last = signature
            campaign = summary.get("campaign")
            if campaign:
                emit(f"Campaign ID: {campaign['campaign_id']}. Use Start / resume in the extension.")
                result = watch_publications(Path(campaign["directory"]), output=Path(batch["settings"]["output"]),
                    watch=not once, interval_seconds=poll_seconds, sleep=sleep)
                results.append(result)
                if result["state"] in {"stopped", "finished_with_gaps"}:
                    emit("This run stopped or has gaps. Saved captures remain; review it before continuing.")
                    return results
                if result["state"] == "finished":
                    if set(result["verified_in_this_run"]) != set(run["stems"]):
                        raise ValueError("Published seasons do not match the requested batch")
                    emit(f"{run['season']}: all {len(run['stems'])} season files verified in xDiyo_data.")
                break
            if once:
                break
            sleep(poll_seconds)
    return results
