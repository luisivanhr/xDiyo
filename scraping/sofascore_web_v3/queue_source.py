"""Read the original scraper's fixture queue; never fetch or restore match values."""
from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from scraping.sofascore.store import season_year_pair
from .campaign import ROOT, prepare_campaign, read_json
from .capture import match_url

DEFAULT_INVENTORY = ROOT / "data/xDiyo_data/_collection/inventory.json"
DEFAULT_CACHE = ROOT / "data/collection_state"
IDENTITY_COLUMNS = ("match_id", "custom_id", "season_id", "season_year", "round",
                    "home_id", "home_team", "away_id", "away_team")


def positive_int(value):
    try:
        number = Decimal(str(value))
        if not number.is_finite() or number != int(number) or number <= 0:
            raise ValueError("Expected a positive integer identity")
        return int(number)
    except (InvalidOperation, TypeError) as error:
        raise ValueError("Expected a positive integer identity") from error


def inventory_selection(path=DEFAULT_INVENTORY, *, season_stems=None, max_matches=None):
    path = Path(path).resolve()
    raw = path.read_bytes()
    document = json.loads(raw)
    seasons = document["seasons"]
    stems = [s["stem"] for s in seasons]
    if len(stems) != len(set(stems)): raise ValueError("Duplicate season entries in source queue")
    selected = set(season_stems) if season_stems is not None else set(stems)
    if not selected or selected-set(stems): raise ValueError("Choose season stems present in the source queue")
    if max_matches is not None and (type(max_matches) is not int or max_matches < 1):
        raise ValueError("max_matches must be None or a positive integer")
    eligible, blocked, seen = [], [], set()
    for item in seasons:
        if item["stem"] not in selected: continue
        if item.get("legacy_identity_valid") is not True or item.get("season_id") is None:
            blocked.append({"stem":item["stem"], "reason":"Source queue marks the season identity invalid or unresolved",
                            "legacy_rows":item.get("legacy_rows")})
            continue
        match = re.fullmatch(r"(.+)_(\d{2})_(\d{2})",item["stem"])
        years = (positive_int(item["season_start"]),positive_int(item["season_end"]))
        if not match or years != (2000+int(match[2]),2000+int(match[3])):
            raise ValueError(f"Filename/season-year conflict: {item['stem']}")
        positive_int(item["season_id"]); positive_int(item["league_id"])
        ids = [positive_int(i) for i in item["legacy_event_ids"]]
        if len(ids) != item["legacy_rows"] or len(ids) != len(set(ids)) or not ids:
            raise ValueError(f"Invalid event ID inventory: {item['stem']}")
        if seen.intersection(ids): raise ValueError("A match ID occurs in multiple source seasons")
        seen.update(ids)
        eligible.append((item,ids))
    count = sum(len(ids) for _,ids in eligible)
    remaining = max_matches if max_matches is not None else count
    chosen = []
    for item, ids in eligible:
        take = ids[:remaining]
        if take: chosen.append((item,take))
        remaining -= len(take)
    summary = {"source_inventory":str(path), "source_inventory_sha256":hashlib.sha256(raw).hexdigest(),
               "source_seasons":len(seasons), "selected_eligible_seasons":len(eligible),
               "eligible_matches":count, "queued_matches":sum(len(ids) for _,ids in chosen),
               "queued_seasons":len(chosen), "blocked_seasons":blocked,
               "order":"Source season order, then stored legacy_event_ids order",
               "source_values_imported":False,"discovery_requests":0}
    return chosen,summary


def queue_preview(path=DEFAULT_INVENTORY, *, season_stems=None, max_matches=None):
    """Cheap local preview: just the inventory JSON, not the large CSV bodies."""
    return inventory_selection(path,season_stems=season_stems,max_matches=max_matches)[1]


def _source_rows(item):
    path = Path(item["source"]).resolve()
    before = path.stat()
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b""): digest.update(chunk)
    if digest.hexdigest() != item["source_sha256"]:
        raise ValueError(f"Source CSV changed since the original queue was prepared: {item['stem']}")
    rows = {}
    with path.open(encoding="utf-8-sig",newline="") as stream:
        reader = csv.DictReader(stream)
        if set(IDENTITY_COLUMNS)-set(reader.fieldnames or []):
            raise ValueError(f"Missing fixture identity columns: {path.name}")
        for row in reader:
            # Discard all scores, standings, lineups, shots, heatmaps and flags.
            metadata = {key:row[key] for key in IDENTITY_COLUMNS}
            eid = positive_int(metadata["match_id"])
            if eid in rows: raise ValueError(f"Duplicate event {eid} in {path.name}")
            rows[eid] = metadata
    after=path.stat()
    if (before.st_mtime_ns,before.st_size) != (after.st_mtime_ns,after.st_size):
        raise ValueError(f"Source CSV changed during the identity read: {path.name}")
    if set(rows) != set(item["legacy_event_ids"]):
        raise ValueError(f"CSV event IDs disagree with the source queue: {path.name}")
    return rows


def _identity(row,item):
    years = (item["season_start"],item["season_end"])
    if positive_int(row["season_id"]) != item["season_id"] or season_year_pair(row["season_year"]) != years:
        raise ValueError(f"CSV season identity conflict: {item['stem']}, event {row['match_id']}")
    round_number=positive_int(row["round"])
    if round_number not in item["rounds"]: raise ValueError("CSV round outside the source queue's scope")
    teams={}
    for side in ("home","away"):
        slug=row[side+"_team"]
        if not re.fullmatch(r"[a-z0-9-]+",slug or ""):
            raise ValueError(f"Invalid legacy team slug: event {row['match_id']}")
        teams[side+"Team"]={"id":positive_int(row[side+"_id"]),"slug":slug,"name":slug}
    if teams["homeTeam"]["id"] == teams["awayTeam"]["id"]: raise ValueError("Identical opposing team IDs")
    return {"id":positive_int(row["match_id"]),"customId":row["custom_id"],
            "slug":row["home_team"]+"-"+row["away_team"], **teams,
            "season":{"id":item["season_id"],"year":row["season_year"]},
            "roundInfo":{"round":round_number},
            "tournament":{"uniqueTournament":{"id":item["league_id"]}}}


def prepare_from_scraper_queue(path=DEFAULT_INVENTORY, *, cache_root=DEFAULT_CACHE,
                               season_stems=None, max_matches=None, on_progress=None, **options):
    """Reuse queued IDs and routing metadata. New responses must prove identity/status."""
    selected,summary=inventory_selection(path,season_stems=season_stems,max_matches=max_matches)
    if not selected: raise ValueError("No valid source fixtures selected; inspect queue_preview")
    cache_path=Path(cache_root).resolve()/"manifest.sqlite"
    cached={}
    if cache_path.exists():
        with sqlite3.connect(cache_path.as_uri()+"?mode=ro",uri=True) as db:
            cached={row[0]:(json.loads(row[1]),row[2]) for row in db.execute("SELECT event_id,payload,observed FROM events")}
    references, ids, reused = {}, [], 0
    for item,chosen in selected:
        rows=_source_rows(item)
        for eid in chosen:
            identity=_identity(rows[eid],item)
            metadata_source="legacy_identity_columns"
            observed=None
            if eid in cached:
                event,observed=cached[eid]
                if event.get("id") != eid or any(event.get(k,{}).get("id") != identity[k]["id"] for k in ("homeTeam","awayTeam","season")) or \
                   event.get("tournament",{}).get("uniqueTournament",{}).get("id") != item["league_id"] or \
                   season_year_pair(event.get("season",{}).get("year")) != (item["season_start"],item["season_end"]) or \
                   event.get("customId") != identity["customId"]:
                    raise ValueError(f"Cached event identity conflicts with source queue: {eid}")
                identity["slug"]=event.get("slug")
                for side in ("homeTeam","awayTeam"):
                    identity[side]["name"]=event[side].get("name",identity[side]["name"])
                metadata_source="cached_provider_route_and_legacy_identity"
                reused+=1
            match_url(identity)  # Validate routing tokens; live page + response checks still apply.
            references[eid]={"kind":"queue_identity", "event":identity,"observed_at":observed,
                "season_stem":item["stem"],"source":str(Path(item["source"]).resolve()),
                "source_sha256":item["source_sha256"],"route_source":metadata_source,
                "note":"Routing and identity constraints only. Status, scores, timing and components must come from the new browser capture."}
            ids.append(eid)
        if on_progress: on_progress({"season":item["stem"],"prepared_matches":len(ids)})
    summary.update(cached_provider_routes=reused,legacy_pair_routes=len(ids)-reused,
                   fixture_cache=str(cache_path),legacy_route_note="Team-slug pair plus stored custom_id and match_id. Collection requires a matching displayed fixture and new event response; these constructed routes are not yet all live-validated.")
    return prepare_campaign(ids,references=references,source_metadata=summary,
                            name=options.pop("name","chrome-from-queue"),**options)
