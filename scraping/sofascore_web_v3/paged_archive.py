"""Reviewed ordinary-phase archives from complete, original Chrome page captures.

This imports evidence, not invented observations. Frozen source hashes, phase
selectors, standings reconciliation and format checks are rerun before queue
creation. The metadata inventory remains unchanged and cannot itself collect.
"""
from collections import Counter
import hashlib
import json

from scraping.sofascore.discovery import event_status
from scraping.sofascore.event_round import match_round
from scraping.sofascore.store import season_year_pair
from .archive_selection import resolve_captured_season, select_captured_tables
from .capture import match_url
from .event_pages import validate_event_pages
from .phase_schedule import checked_review, reconcile_phase, validate_phase_format


def evidence_body(db, source):
    row = db.execute("SELECT observed,body,sha256 FROM responses WHERE path=?", (source["path"],)).fetchone()
    if (not row or row[0] != source["observed"] or row[2] != source["sha256"]
            or hashlib.sha256(row[1].encode()).hexdigest() != row[2]):
        raise ValueError(f"Missing or changed captured evidence: {source['path']}")
    return json.loads(row[1])


def validate_paged_cohort(db, spec):
    contract = spec["fixture_contract"]
    if contract.get("schema") != 1 or not contract.get("review_sources"):
        raise ValueError("Missing reviewed phase contract or provenance")
    cid, sid = spec["competition_id"], spec["provider_season_id"]
    sources = contract["sources"]
    by_path = {r["path"]: r for r in sources}
    if len(sources) != len(by_path):
        raise ValueError("Duplicate evidence path")
    payloads = {path: evidence_body(db, source) for path, source in by_path.items()}
    catalog = payloads[f"/unique-tournament/{cid}/seasons"]
    identity = resolve_captured_season(catalog["seasons"], spec)
    if identity is None:
        raise ValueError("Pinned season absent from original catalog")
    years = season_year_pair(identity["year"])
    pages = []
    observations = {}
    for n, path in enumerate(contract["page_paths"]):
        if path != f"/unique-tournament/{cid}/season/{sid}/events/last/{n}":
            raise ValueError("Page route differs from exact selected season")
        pages.append(dict(page=n, payload=payloads[path]))
        for e in payloads[path]["events"]:
            observations.setdefault(e["id"], by_path[path]["observed"])
    events = validate_event_pages(pages, competition_id=cid, season_id=sid,
                                 reviewed_overlaps=contract.get("reviewed_overlaps", []))
    ids = {e["id"] for e in events}
    for path in contract.get("recovered_paths", []):
        event = payloads[path].get("event")
        if not event or path != f"/event/{event['id']}" or event["id"] in ids:
            raise ValueError("Recovered event is duplicated or has a conflicting route")
        if (event["tournament"]["uniqueTournament"]["id"], event["season"]["id"]) != (cid,sid):
            raise ValueError("Recovered event belongs to another season")
        events.append(event)
        ids.add(event["id"])
        observations[event["id"]] = by_path[path]["observed"]
    standings = payloads[contract["standings_path"]]
    phases, selected, regular, used_tables, used_stages, reviews_used = [], [], [], set(), set(), set()
    for phase in contract["phases"]:
        if (phase["competition_id"], phase["season_id"]) != (cid,sid):
            raise ValueError("Foreign phase contract")
        tid = phase["tournament_id"]
        table_ids = {s["standings_id"] for s in phase["table_selectors"]}
        if tid in used_stages or table_ids & used_tables:
            raise ValueError("Overlapping ordinary phases/tables")
        used_stages.add(tid)
        used_tables.update(table_ids)
        tables = select_captured_tables(standings, competition_id=cid, selectors=phase["table_selectors"])
        values = [e for e in events if e["tournament"]["id"] == tid]
        if not values:
            raise ValueError("Selected ordinary stage has no captured fixtures")
        for e in values:
            if season_year_pair(e["season"].get("year")) != years:
                raise ValueError("Fixture year differs from the pinned season")
            if type(e.get("startTimestamp")) is not int or e["homeTeam"]["id"] == e["awayTeam"]["id"]:
                raise ValueError("Invalid kickoff or fixture clubs")
            match_url(e)
            review = checked_review(e, contract["event_reviews"])
            if review:
                reviews_used.add(e["id"])
        reconciliation = reconcile_phase(values, tables, event_reviews=contract["event_reviews"],
                                          table_reviews=contract["table_reviews"])
        if reconciliation["table_differences"]:
            raise ValueError(f"Stage {tid}: fixture W/D/L/goals differ from the reviewed table: {reconciliation['table_differences'][:3]}")
        checked = validate_phase_format(values, tables, phase, reconciliation)
        selected.extend(reconciliation["selected_ids"])
        regular.extend(values)
        phases.append(dict(tournament_id=tid, table_ids=sorted(table_ids), **checked,
            selected_matches=len(reconciliation["selected_ids"]),
            annulled_played_ids=reconciliation["annulled_played_ids"],
            standings_reconciled_fields=["matches","wins","draws","losses","scoresFor","scoresAgainst"]))
    if reviews_used != {r["event_id"] for r in contract["event_reviews"]}:
        raise ValueError("Unused reviewed event disposition")
    table_review_ids = [r["table_id"] for r in contract["table_reviews"]]
    if len(table_review_ids) != len(set(table_review_ids)) or set(table_review_ids) - used_tables:
        raise ValueError("Duplicate or unused reviewed table correction")
    if len(selected) != len(set(selected)) or not selected:
        raise ValueError("Empty or overlapping selected fixture set")
    selected = sorted(selected)
    target_hash = hashlib.sha256(json.dumps(selected, separators=(",", ":")).encode()).hexdigest()
    if target_hash != contract["selected_ids_sha256"] or len(selected) != contract["selected_matches"]:
        raise ValueError("Selected fixture set differs from the reviewed target set")
    return dict(identity=identity, events=events, regular=regular, observations=observations,
                selected_ids=selected, phases=phases, selected_ids_sha256=target_hash)


def paged_summary(discovery, index, spec):
    """Convert validated evidence into the existing archive/publication contract."""
    issues, result = [], None
    try:
        result = validate_paged_cohort(discovery.db, spec)
        # The indexed event view must still be exactly the original evidence.
        stored = {r["id"]: r for r in discovery.db.execute(
            "SELECT * FROM events WHERE competition=? AND season=?", (spec["competition_id"], spec["provider_season_id"]))}
        regular = {e["id"] for e in result["regular"]}
        if set(stored) != {e["id"] for e in result["events"]}:
            raise ValueError("Indexed fixture set differs from original page evidence")
        for event in result["events"]:
            row = stored[event["id"]]
            if (json.loads(row["payload"]) != event or row["observed"] != result["observations"][event["id"]]
                    or row["round"] != match_round(event) or bool(row["regular"]) != (event["id"] in regular)):
                raise ValueError("Indexed fixture metadata changed from original evidence")
    except (ValueError, KeyError, TypeError) as error:
        issues.append(str(error))
    resolved = discovery._resolved(index)
    if result and (not resolved or resolved["season_id"] != result["identity"]["id"]
                   or resolved["year"] != result["identity"]["year"]):
        issues.append("Missing or conflicting indexed season identity")
    events = result["regular"] if result else []
    return dict(league=spec["name"], season=spec["season"], competition_id=spec["competition_id"],
        season_id=spec["provider_season_id"], stem=resolved["stem"] if resolved else None,
        regular_matches=len(events), finished=len(result["selected_ids"]) if result else 0,
        selected_matches=len(result["selected_ids"]) if result and not issues else 0,
        statuses=dict(Counter(event_status(e) for e in events)), schedule_verified=not issues,
        validation_issues=issues, superseded_fixtures=[], snapshot=spec["snapshot"],
        selection="reviewed_ordinary_phases", schedule_validation="complete_pages_phase_tables_and_pair_contract",
        format_source=spec["fixture_contract"]["review_sources"],
        reviewed_fixture_dispositions=spec["fixture_contract"]["event_reviews"],
        reviewed_table_corrections=spec["fixture_contract"]["table_reviews"],
        reviewed_normalizations=spec["fixture_contract"].get("normalizations", {}),
        captured_evidence=spec["fixture_contract"]["sources"],
        reviewed_phases=result["phases"] if result else [],
        selected_ids_sha256=result["selected_ids_sha256"] if result else None)


def import_reviewed_evidence(discovery, responses, *, origin):
    """Seed a new disconnected archive from checksum-verified original responses.

    Each response includes the real timestamp and body from a successful Chrome
    observation. No lease or HTTP attempt is invented. Import receipts name the
    existing inventory/HAR. Existing collection campaigns cannot be changed.
    """
    if discovery.current or discovery._value("campaign") or not origin:
        raise ValueError("Evidence import requires a disconnected uncollected archive")
    if any("fixture_contract" not in s for s in discovery.specs):
        raise ValueError("All imported cohorts need explicit reviewed phase contracts")
    paths = {r["path"] for r in responses}
    if len(paths) != len(responses):
        raise ValueError("Duplicate import response")
    expected = {s["path"] for spec in discovery.specs for s in spec["fixture_contract"]["sources"]}
    if paths != expected:
        raise ValueError("Evidence imports differ from the frozen source manifest")
    with discovery.db:
        for r in responses:
            if hashlib.sha256(r["body"].encode()).hexdigest() != r["sha256"] or not r.get("origin"):
                raise ValueError("Unverified source body or missing original receipt")
            prior = discovery.db.execute("SELECT observed,body,sha256 FROM responses WHERE path=?", (r["path"],)).fetchone()
            if prior and tuple(prior) != (r["observed"],r["body"],r["sha256"]):
                raise ValueError("Never replace a previously imported response")
            discovery.db.execute("INSERT OR IGNORE INTO responses VALUES(?,?,?,?)", (r["path"],r["observed"],r["body"],r["sha256"]))
        for i, spec in enumerate(discovery.specs):
            checked = validate_paged_cohort(discovery.db, spec)
            identity = checked["identity"]
            sid, cid, year = identity["id"], spec["competition_id"], identity["year"]
            years = season_year_pair(year)
            discovery._save(f"resolved:{i}", dict(season_id=sid, year=year, years=years,
                stem=f"{spec['name']}_{years[0]%100:02d}_{years[1]%100:02d}"))
            regular = {e["id"] for e in checked["regular"]}
            for event in checked["events"]:
                row = (event["id"],cid,sid,match_round(event),int(event["id"] in regular),
                       json.dumps(event,ensure_ascii=False,allow_nan=False),checked["observations"][event["id"]])
                old = discovery.db.execute("SELECT * FROM events WHERE id=?", (event["id"],)).fetchone()
                if old and tuple(old) != row:
                    raise ValueError("Existing imported fixture differs")
                discovery.db.execute("INSERT OR IGNORE INTO events VALUES(?,?,?,?,?,?,?)", row)
            for source in spec["fixture_contract"]["sources"]:
                discovery.add_job(source["path"],cid,"imported_chrome_evidence",sid)
                discovery.db.execute("UPDATE jobs SET state='complete',lease=NULL,error=NULL WHERE path=?", (source["path"],))
        discovery._save("evidence_import", dict(origin=origin, sources=[{k:r[k] for k in ("path","observed","sha256","origin")} for r in responses]))
    return discovery.status(refresh=True)
