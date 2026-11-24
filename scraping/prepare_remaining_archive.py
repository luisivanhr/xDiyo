"""Audit the remaining 88 captured season inventories and prepare one archive.

Default writes a reproducible review only. --prepare requires every phase to
pass and imports original Chrome evidence into a new immutable archive. It
never creates a collection queue, contacts the provider, or starts a worker.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scraping.prepare_remaining_coverage import RULES
from scraping.review_remaining_fixtures import PREPARATION, INVENTORY, table_selection
from scraping.sofascore.event_round import match_round
from scraping.sofascore.store import atomic_write, json_bytes
from scraping.sofascore_web_v3.archive_discovery import prepare_archive
from scraping.sofascore_web_v3.campaign import collector_lock
from scraping.sofascore_web_v3.discovery import open_discovery
from scraping.sofascore_web_v3.event_pages import validate_event_pages
from scraping.sofascore_web_v3.import_har import response_json, response_time
from scraping.sofascore_web_v3.paged_archive import import_reviewed_evidence
from scraping.sofascore_web_v3.phase_schedule import reconcile_phase, validate_phase_format


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def saved_response(db, path):
    row = db.execute("SELECT observed,body,sha256 FROM responses WHERE path=?", (path,)).fetchone()
    if not row or hashlib.sha256(row[1].encode()).hexdigest() != row[2]:
        raise ValueError(f"Missing or changed original response: {path}")
    latest = db.execute("SELECT http_status,observed,error FROM attempts WHERE path=? ORDER BY id DESC LIMIT 1", (path,)).fetchone()
    job = db.execute("SELECT state FROM jobs WHERE path=?", (path,)).fetchone()
    if not job or job[0] != "complete" or not latest or latest[0] != 200 or latest[1] != row[0]:
        raise ValueError(f"Source observation is not a completed original HTTP 200 capture: {path}")
    return dict(path=path, observed=row[0], body=row[1], sha256=row[2], origin=str(INVENTORY))


def har_response(filename, path):
    source = PREPARATION / "sources" / filename
    entries = load(source)["log"]["entries"]
    if len(entries) != 1:
        raise ValueError("Use the retained one-response original HAR selection")
    entry = entries[0]
    if (entry["request"]["method"] != "GET" or entry["request"]["url"] != "https://www.sofascore.com/api/v1"+path
            or entry["response"]["status"] != 200):
        raise ValueError("HAR route/status mismatch")
    body = json.dumps(response_json(entry), ensure_ascii=False, allow_nan=False)
    return dict(path=path,observed=response_time(entry),body=body,sha256=hashlib.sha256(body.encode()).hexdigest(),
                origin=dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))


def phase_rule(record, tid, tables, events, reconciled, full_standings, australian):
    cid,sid = record["competition_id"],record["season_id"]
    teams = reconciled["expected_rows"]
    nominal = [e for e in events if e["id"] in set(reconciled["standings_ids"])]
    pairs = Counter(tuple(sorted((e["homeTeam"]["id"], e["awayTeam"]["id"]))) for e in nominal)
    snapshots = {(136,98511,144),(11653,88493,64475),(11539,88503,189591),(406,88529,34468)}
    result = dict(competition_id=cid,season_id=sid,tournament_id=tid,teams=len(teams),snapshot=(cid,sid,tid) in snapshots,
        table_selectors=[dict(standings_id=t["id"],tournament_id=t["tournament"]["id"],name=t["name"]) for t in tables],
        club_groups=[[r["team"]["id"] for r in t["rows"]] for t in tables],pair_rule="round_robin",meetings=1,
        round_mode="strict",rounds=sorted({match_round(e) for e in nominal if match_round(e) is not None}))
    if (cid,sid,tid) in {(136,34942,144),(242,12825,18),(242,15813,18)}:
        result["round_mode"]="provider_week_buckets"
    if cid==242 and sid not in {12825,15813}:
        result["round_mode"]="absent"
    if cid==136:
        result.update(pair_rule="a_league",played_per_club=27 if sid in {14404,17802,54447} else 26)
        matrix=next((s for s in australian["seasons"] if s["provider_reference"]["season_id"]==sid),None)
        if matrix:
            mapping={k:v["team_id"] for k,v in matrix["provider_reference"]["team_identities"].items()}
            result["independent_results"]=[[mapping[e["home"]],mapping[e["away"]],*e["score"]] for e in matrix["results_for_reconciliation"]]
            result["independent_reference"]={k:australian[k] for k in ("source_url","source_sha256")}
            result["independent_reference"]["source_page"]=matrix["source_page"]
            result["independent_reference"]["reviewed_reference_corrections"]=matrix["reviewed_reference_corrections"]
    elif cid==242:
        conferences=[t for t in full_standings["standings"] if t.get("name","").lower().endswith(("eastern conference","western conference"))]
        result.update(pair_rule="mls",max_meetings=4 if sid==26780 else 3,
                      club_groups=[[r["team"]["id"] for r in t["rows"]] for t in conferences])
    elif cid==11653:
        result["meetings"]=1 if sid==13475 else 2
        if sid==22328:
            # Official ANFP curtailed final table; every recorded match/goal is
            # reconciled but the unplayed remainder is not requested.
            result["snapshot"]=True
            result["curtailed"]=True
    elif cid==649:
        result["meetings"]=2
    elif cid==240:
        result["meetings"]=2 if sid in {12847,16023,20069,71184,89674} else 1
    elif cid==406 and tid in {57598,57645}:
        result["meetings"]=2
    if (cid==155 and sid in {70268,77826,87913}) or (cid==406 and sid==35813 and tid==51):
        groups=result["club_groups"];group={t:i for i,g in enumerate(groups) for t in g}
        extra=[list(p) for p in pairs for _ in range(pairs[p]) if group[p[0]]!=group[p[1]]]
        result.update(pair_rule="round_robin_extra",extra_pairs=sorted(extra),extra_per_club=2 if cid==155 else 1)
    elif cid in {11539,11536} and len(nominal)==200:
        result.update(pair_rule="round_robin_extra",extra_pairs=sorted([list(p) for p,n in pairs.items() if n==2]),extra_per_club=1)
    return result


def audit():
    draft=load(PREPARATION/"next_wave_draft.json")
    preview=load(PREPARATION/"combined_identity_preview.json")
    decisions=load(PREPARATION/"fixture_decisions_2026-11-14.json")
    if decisions["unresolved_source_reviews"]:
        raise ValueError("Unresolved fixture source reviews remain")
    australian=load(PREPARATION/"australia_reviewed_opponent_matrices.json")
    specs,responses,reviews=[],{},[]
    with sqlite3.connect(INVENTORY.as_uri()+"?mode=ro",uri=True) as db:
        db.execute("BEGIN")
        for record in draft["seasons"]:
            cid,sid=record["competition_id"],record["season_id"]
            sources={}
            def take(path):
                r=saved_response(db,path);sources[path]=r;return json.loads(r["body"])
            take(f"/unique-tournament/{cid}/seasons")
            table_path=f"/unique-tournament/{cid}/season/{sid}/standings/total"
            standings=take(table_path)
            selected=table_selection(db,cid,sid)
            receipt=json.loads(db.execute("SELECT value FROM state WHERE key=?",(f"page_inventory:{cid}:{sid}:last",)).fetchone()[0])
            paths=[f"/unique-tournament/{cid}/season/{sid}/events/last/{n}" for n in range(receipt["pages"])]
            pages=[dict(page=n,payload=take(path)) for n,path in enumerate(paths)]
            events=validate_event_pages(pages,competition_id=cid,season_id=sid,reviewed_overlaps=receipt.get("reviewed_overlaps",[]))
            recovered=[]
            if (cid,sid)==(11539,88503):
                fresh=har_response("colombia-2026-standings-88503-fresh-selected.har",table_path)
                sources[table_path]=fresh;standings=json.loads(fresh["body"])
                selected=[t for t in standings["standings"] if t["id"] in {t["id"] for t in selected}]
                r=har_response("colombia-2026-fortaleza-cali-16390806-selected.har","/event/16390806")
                sources[r["path"]]=r;recovered.append(r["path"]);events.append(json.loads(r["body"])["event"])
            by_stage=defaultdict(list)
            for t in selected:
                tid=t["tournament"]["id"]
                if (cid,sid)==(649,36257) and t["id"] in {81313,81314}:tid=652
                by_stage[tid].append(t)
            event_reviews=[r for r in decisions["event_reviews"] if (r["guard"]["competition_id"],r["guard"]["season_id"])==(cid,sid)]
            table_reviews=[r for r in decisions["table_reviews"] if r["competition_id"]==cid and r["table_id"] in {t["id"] for t in selected}]
            phases,ids,phase_reviews=[],[],[]
            for tid,tables in by_stage.items():
                values=[e for e in events if e["tournament"]["id"]==tid]
                r=reconcile_phase(values,tables,event_reviews=event_reviews,table_reviews=table_reviews)
                if r["table_differences"]:raise ValueError((cid,sid,tid,r["table_differences"][:3]))
                contract=phase_rule(record,tid,tables,values,r,standings,australian)
                result=validate_phase_format(values,tables,contract,r)
                phases.append(contract);ids.extend(r["selected_ids"])
                phase_reviews.append(dict(tournament_id=tid,**result,selected=len(r["selected_ids"]),annulled=r["annulled_played_ids"]))
            name=next(r["name"] for r in preview["records"] if (r["competition_id"],r["provider_season_id"])==(cid,sid))
            contract=dict(schema=1,sources=[{k:r[k] for k in ("path","observed","sha256")} for r in sources.values()],
                page_paths=paths,reviewed_overlaps=receipt.get("reviewed_overlaps",[]),recovered_paths=recovered,standings_path=table_path,
                phases=phases,event_reviews=event_reviews,table_reviews=table_reviews,
                selected_matches=len(ids),selected_ids_sha256=hashlib.sha256(json.dumps(sorted(ids),separators=(",",":")).encode()).hexdigest(),
                review_sources=RULES[record["country"]]["reference_sources"],
                review_receipt=str(PREPARATION/"fixture_decisions_2026-11-14.json"))
            event_ids = {e["id"] for e in events}
            contract["normalizations"] = {
                key: [r for r in decisions[key] if r["event_id"] in event_ids]
                for key in ("round_corrections", "side_corrections")}
            score_review = decisions["score_correction"]
            contract["normalizations"]["score_corrections"] = (
                [score_review] if score_review["event_id"] in event_ids else [])
            specs.append(dict(name=name,competition_id=cid,provider_season_id=sid,season=record["provider_year"],aliases=[record["provider_year"]],
                meetings_per_pair=2,expected_teams=None,round_limit=None,groups=1,snapshot=any(p["snapshot"] for p in phases),fixture_contract=contract))
            responses.update(sources)
            reviews.append(dict(league=name,competition_id=cid,season_id=sid,year=record["provider_year"],phases=phase_reviews,selected_matches=len(ids)))
    return specs,list(responses.values()),reviews


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--prepare",action="store_true")
    args=parser.parse_args()
    specs,responses,reviews=audit()
    result=dict(checked_at=datetime.now(timezone.utc).isoformat(),cohorts=len(specs),phases=sum(len(s["phases"]) for s in reviews),
        selected_matches=sum(s["selected_matches"] for s in reviews),seasons=reviews,fixture_validation_passed=True,collection_started=False)
    if args.prepare:
        batch=prepare_archive("coverage-remaining-2017-2027",specs)
        with collector_lock(Path(batch["directory"])/"collector.lock"):
            with open_discovery(batch["directory"]) as discovery:
                status=import_reviewed_evidence(discovery,responses,origin=str(INVENTORY))
                if not status["ready_to_collect"]:raise ValueError("Imported archive failed revalidation")
        result.update(discovery_id=batch["discovery_id"],batch_path=batch["path"],ready_to_collect=True)
    atomic_write(args.output,json_bytes(result))
    print(json.dumps({k:v for k,v in result.items() if k!='seasons'}))


if __name__=="__main__":main()
