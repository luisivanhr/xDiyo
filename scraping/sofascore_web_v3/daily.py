"""Prepare, resume and publish a daily Chrome run. No provider HTTP in Python."""
from __future__ import annotations

from contextlib import closing
from dataclasses import replace
from pathlib import Path
import sqlite3
import time

from .campaign import ROOT, campaign_path, collector_lock
from .capture import PROFILE
from .daily_storage import publish_daily_campaign
from .discovery import DISCOVERY_BASE, open_discovery, prepare_discovery, league_catalog
from .publish import read_json
from .status_write import atomic_write_status
from scraping.sofascore.store import json_bytes

OUTPUT = ROOT / "data/xDiyo_data"


def prepare_daily_run(*, season, leagues=None, include_lineups=True, random_delay_max_seconds=None,
                      lineup_window_hours=24, history_root=None, output_root=None,
                      discovery_base=DISCOVERY_BASE):
    """Reuse an unfinished run with exactly these settings; otherwise make a new run."""
    history_root = OUTPUT if history_root is None else Path(history_root)
    output_root = OUTPUT if output_root is None else Path(output_root)
    # These settings are a stable local key, not provider data or a queue mutation.
    leagues = league_catalog() if leagues is None else leagues
    settings = {"season": season, "leagues": leagues, "include_lineups": include_lineups,
        "random_delay_max_seconds": random_delay_max_seconds, "lineup_window_hours": lineup_window_hours,
        "history_root": str(Path(history_root).resolve()), "output_root": str(Path(output_root).resolve())}
    output_root = Path(output_root).resolve()
    state = output_root / "_collection/last_run.json"
    with collector_lock(output_root / "_collection/daily_prepare.lock"):
        if state.exists():
            saved = read_json(state)
            directory = campaign_path(saved["discovery_id"], discovery_base)
            if saved["settings"] == settings and not (directory / "publication.json").exists():
                with open_discovery(directory, read_only=True) as discovery:
                    return discovery.status()
        prepared = prepare_discovery(season=season, leagues=leagues, mode="daily",
            collection_profile=replace(PROFILE, include_upcoming=True, include_lineups=include_lineups,
                                       lineup_window_hours=lineup_window_hours),
            random_delay_max_seconds=random_delay_max_seconds, history_root=history_root,
            output_root=output_root, base=discovery_base)
        atomic_write_status(state, json_bytes({"discovery_id": prepared["discovery_id"], "settings": settings}))
        return prepared


def daily_status(discovery_id, *, discovery_base=DISCOVERY_BASE):
    """Read an active run without opening another writer or changing queue state."""
    directory = campaign_path(discovery_id, discovery_base)
    with open_discovery(directory, read_only=True) as discovery:
        if discovery.config.get("mode") != "daily":
            raise ValueError("Select a daily discovery ID")
        snapshot = discovery.status()
    pause = None
    with closing(sqlite3.connect((directory / "discovery.sqlite").as_uri() + "?mode=ro", uri=True)) as db:
        row = db.execute("SELECT value FROM state WHERE key='pause'").fetchone()
        if row:
            pause = row[0]
    result = {"discovery": snapshot, "phase": "discovery", "pause": pause}
    campaign = snapshot.get("campaign")
    if campaign:
        capture = Path(campaign["directory"])
        with closing(sqlite3.connect((capture / "queue.sqlite").as_uri() + "?mode=ro", uri=True)) as db:
            counts = dict(db.execute("SELECT state,COUNT(*) FROM queue GROUP BY state"))
        paused = capture / "PAUSE"
        result.update(phase="collection", counts=counts, pause=paused.read_text(encoding="utf-8") if paused.exists() else None,
                      capture_complete=not any(state != "complete" for state in counts))
    if (directory / "publication.json").exists():
        result["phase"] = "published"
    return result


def watch_daily_run(discovery_id, *, discovery_base=DISCOVERY_BASE, poll_seconds=5, once=False,
                    emit=print, sleep=time.sleep):
    """Wait locally; the extension owns discovery/capture. Publish when complete."""
    if not 1 <= poll_seconds <= 60:
        raise ValueError("Use a local check interval between 1 and 60 seconds")
    emit(f"Discovery ID: {discovery_id}")
    emit("In xDiyo Network Capture: paste this Discovery ID, Create / reconnect, then Discover / resume.")
    last = None
    while True:
        snapshot = daily_status(discovery_id, discovery_base=discovery_base)
        discovery = snapshot["discovery"]
        campaign = discovery.get("campaign")
        signature = (snapshot["phase"], json_bytes(discovery["jobs"]), discovery["ready_to_collect"],
                     campaign["campaign_id"] if campaign else None, json_bytes(snapshot.get("counts", {})), snapshot["pause"])
        if signature != last:
            if campaign:
                if last is None or last[3] != campaign["campaign_id"]:
                    emit(f"Campaign ID: {campaign['campaign_id']}")
                    emit(f"Reusing {campaign['reused_matches']} completed matches; collecting {campaign['matches']} new or unfinished matches.")
                    if campaign["matches"]:
                        emit("In Automatic collection: Start / resume. If disconnected, paste the Campaign ID and Connect to Python first.")
                emit(f"Collection: {snapshot.get('counts', {})}")
            else:
                emit(f"Discovery: {discovery['jobs']}")
                if discovery["discovery_complete"]:
                    for season in discovery["seasons"]:
                        emit(f"{season['league']}: {season['state']}; last completed round {season['last_completed_round']}; target round {season['target_round']}.")
                    if discovery["ready_to_collect"]:
                        emit("In the extension: Create collection campaign. The campaign ID will appear here automatically.")
                    else:
                        emit("Discovery needs review: " + str([s for s in discovery["seasons"] if s["issues"]]))
            if snapshot["pause"]:
                emit(f"Paused: {snapshot['pause']}. Resolve the displayed issue and use the extension's resume button.")
            last = signature
        if campaign and snapshot.get("capture_complete"):
            result = publish_daily_campaign(campaign["directory"])
            emit(f"Published: {Path(campaign['output_root']) / 'LATEST.json'}")
            emit(f"History: {sum(len(s['history_ids']) for s in result['seasons'])}; upcoming prediction candidates at publication: {sum(len(s['prediction_ids']) for s in result['seasons'])}.")
            return result
        if once:
            return snapshot
        sleep(poll_seconds)
