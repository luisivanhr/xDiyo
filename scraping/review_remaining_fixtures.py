"""Audit saved remaining-league pages read-only; this never approves collection.

Run with --output to save the current review. Original provider responses and
the active worker are unchanged. Missing/partial page chains remain explicit.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scraping.prepare_remaining_coverage import phase_decision
from scraping.sofascore.discovery import event_status, is_completed_event
from scraping.sofascore.event_round import match_round
from scraping.sofascore.store import atomic_write, json_bytes
from scraping.sofascore_web_v3.event_pages import validate_event_pages

PREPARATION = ROOT / "data/website_capture_v3/batches/coverage_remaining_preparation_2026-11-14"
INVENTORY = ROOT / "data/website_capture_v3/discovery/discover-inventory-937d9ffd589c/discovery.sqlite"
FIELDS = ("matches", "wins", "draws", "losses", "scoresFor", "scoresAgainst")


def body(db, path):
    row = db.execute("SELECT body,sha256 FROM responses WHERE path=?", (path,)).fetchone()
    if not row:
        return None
    if hashlib.sha256(row[0].encode()).hexdigest() != row[1]:
        raise ValueError(f"Changed original evidence: {path}")
    return json.loads(row[0])


def fixture_pages(db, cid, sid):
    saved = db.execute("SELECT value FROM state WHERE key=?", (f"page_inventory:{cid}:{sid}:last",)).fetchone()
    if not saved:
        return None
    receipt = json.loads(saved[0])
    pages = [{"page": page, "payload": body(db, f"/unique-tournament/{cid}/season/{sid}/events/last/{page}")}
             for page in range(receipt["pages"])]
    return validate_event_pages(pages, competition_id=cid, season_id=sid,
                                reviewed_overlaps=receipt.get("reviewed_overlaps", []))


def table_selection(db, cid, sid):
    payload = body(db, f"/unique-tournament/{cid}/season/{sid}/standings/total")
    if not payload:
        return []
    selected = []
    for table in payload["standings"]:
        if phase_decision(cid, sid, table) != "ordinary_candidate":
            continue
        # Exactly reviewed duplicate table; identical unrelated phases stay separate.
        if (cid, sid, table["id"]) == (11539, 35235, 70125):
            original = next(t for t in payload["standings"] if t["id"] == 70122)
            fingerprint = lambda t: sorted((r["team"]["id"], *(r.get(k) for k in FIELDS), r.get("points")) for r in t["rows"])
            assert fingerprint(table) == fingerprint(original)
            continue
        selected.append(table)
    return selected


def compact(event):
    return {"id": event["id"], "home": event["homeTeam"]["id"], "away": event["awayTeam"]["id"],
            "home_name": event["homeTeam"]["name"], "away_name": event["awayTeam"]["name"],
            "round": match_round(event), "roundInfo": event.get("roundInfo"),
            "kickoff": event["startTimestamp"], "status": event.get("status"),
            "score": [event.get(s, {}).get("current") for s in ("homeScore", "awayScore")],
            "isAwarded": event.get("isAwarded")}


def compare_phase(events, tables):
    expected = {}
    for table in tables:
        for row in table["rows"]:
            team = row["team"]["id"]
            if team in expected:
                raise ValueError("Overlapping ordinary tables in one phase")
            expected[team] = row
    actual = defaultdict(Counter)
    for event in events:
        if event_status(event) not in {"finished", "awarded"}:
            continue
        home, away = event["homeTeam"]["id"], event["awayTeam"]["id"]
        hg, ag = [event.get(s, {}).get("current") for s in ("homeScore", "awayScore")]
        if type(hg) is not int or type(ag) is not int:
            continue
        for team, gf, ga in [(home, hg, ag), (away, ag, hg)]:
            actual[team].update(matches=1, wins=int(gf > ga), draws=int(gf == ga), losses=int(gf < ga), scoresFor=gf, scoresAgainst=ga)
    differences = [{"team": team, "name": expected[team]["team"]["name"], "field": field,
                    "table": expected[team].get(field), "events": actual[team][field]}
                   for team in expected for field in FIELDS if expected[team].get(field) != actual[team][field]]
    pairs = Counter(tuple(sorted((e["homeTeam"]["id"], e["awayTeam"]["id"]))) for e in events)
    by_round = defaultdict(list)
    for e in events:
        for side in ("homeTeam", "awayTeam"):
            if match_round(e) is not None:
                by_round[match_round(e), e[side]["id"]].append(e["id"])
    return {"tournament_id": tables[0]["tournament"]["id"], "tables": [t["id"] for t in tables],
            "clubs": len(expected), "events": len(events), "finished": sum(is_completed_event(e) for e in events),
            "table_matches": sum(r["matches"] for r in expected.values()) / 2,
            "statuses": dict(Counter(event_status(e) for e in events)),
            "rounds": sorted({match_round(e) for e in events if match_round(e) is not None}),
            "missing_rounds": [compact(e) for e in events if match_round(e) is None],
            "nonfinished_or_awarded": [compact(e) for e in events if event_status(e) != "finished" or e.get("isAwarded")],
            "pair_multiplicities": dict(Counter(pairs.values())),
            "round_collisions": [{"round": r, "team": t, "events": ids} for (r, t), ids in by_round.items() if len(ids) > 1],
            "foreign_clubs": sorted(set(actual) - set(expected)), "table_differences": differences}


def review(db, record):
    cid, sid = record["competition_id"], record["season_id"]
    result = {k: record[k] for k in ("country", "competition_id", "season_id", "provider_year")}
    events = fixture_pages(db, cid, sid)
    if events is None:
        return dict(result, phase="page_inspection_incomplete")
    tables = table_selection(db, cid, sid)
    by_stage = defaultdict(list)
    for table in tables:
        tid = table["tournament"]["id"]
        if cid == 649 and sid == 36257 and table["id"] in {81313, 81314} and tid in {97576, 97577}:
            tid = 652  # Provider events share a stage; the two tables have disjoint clubs.
        by_stage[tid].append(table)
    phases = [compare_phase([e for e in events if e["tournament"]["id"] == tid], values)
              for tid, values in by_stage.items()]
    excluded = Counter((e["tournament"]["id"], e["tournament"]["name"]) for e in events if e["tournament"]["id"] not in by_stage)
    return dict(result, phase="complete_pages_pending_fixture_validation", page_events=len(events),
                ordinary_events=sum(p["events"] for p in phases), phases=phases,
                excluded_stages=[{"id": tid, "name": name, "events": n} for (tid, name), n in excluded.items()],
                collection_ready=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    draft = json.loads((PREPARATION / "next_wave_draft.json").read_text(encoding="utf-8"))
    with sqlite3.connect(INVENTORY.as_uri() + "?mode=ro", uri=True) as db:
        with db:
            db.execute("BEGIN")
            seasons = [review(db, s) for s in draft["seasons"]]
            jobs = dict(db.execute("SELECT state,COUNT(*) FROM jobs GROUP BY state"))
    result = {"checked_at": datetime.now(timezone.utc).isoformat(), "inspection_only": True,
              "jobs": jobs, "seasons": seasons, "collection_ready": False}
    atomic_write(args.output, json_bytes(result))
    ready = [s for s in seasons if "phases" in s]
    print(json.dumps({"complete_page_inventories": len(ready), "total": len(seasons), "jobs": jobs,
        "review": [{"country": s["country"], "year": s["provider_year"], "events": s["ordinary_events"],
                    "table_differences": sum(len(p["table_differences"]) for p in s["phases"]),
                    "missing_rounds": sum(len(p["missing_rounds"]) for p in s["phases"]),
                    "special_statuses": sum(len(p["nonfinished_or_awarded"]) for p in s["phases"])} for s in ready]}))


if __name__ == "__main__":
    main()
