"""Reproduce the remaining-league review in a new directory from saved evidence.

This report is not an executable archive selection and cannot start collection.
The canonical campaign plan and original captures are never changed here.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "data/website_capture_v3/batches/coverage_remaining_preparation_2026-11-14"
SOURCE_BASE = BASE
INV = REPO / "data/website_capture_v3/discovery/discover-inventory-937d9ffd589c"
FIELDS = ("matches", "wins", "draws", "losses", "scoresFor", "scoresAgainst", "points")
SELECTIONS = [
    ("Argentina", "Liga_Profesional", 155, [13950,18113,24239,37231,41884,47647,57478,70268,77826,87913]),
    ("Australia", "A_League", 136, [14404,17802,24802,34942,38979,44709,54447,64864,82603,98511]),
    ("Chile", "Chile_Primera_Division", 11653, [13475,15995,22328,26951,36048,40515,48017,57883,71131,88493]),
    ("China", "Chinese_Super_League", 649, [12938,16186,22556,28232,36257,41892,49950,58388,71364,90049]),
    ("Colombia", "Colombia_Primera_A", 11539, [12882,16044,20160,26956,35235,40320,48283,57374,70681,88503]),
    ("Colombia", "Colombia_Primera_A", 11536, [12882,19511,24014,37473,42387,52847,63819,77825]),
    ("Ecuador", "LigaPro", 240, [12847,16023,20069,26952,35552,40503,48720,58043,71184,89674]),
    ("Peru", "Peru_Liga_1", 406, [12872,16007,22349,27096,35813,40118,48078,57741,70962,88529]),
    ("USA", "MLS", 242, [12825,15813,20108,26780,35964,40071,47955,57317,70158,86668]),
]

RULES = {
    "Argentina": {
        "resolved": ["Exact split-year and calendar season IDs mapped. No separate 2020 league is invented.",
            "2025 Apertura and Clausura have separate season IDs with the same year label; 2026 combines both in one season ID.",
            "2025/2026 ordinary groups retain cross-zone matches; annual and averages tables are not additional stages."],
        "blockers": ["Exact season-ID selection must support both 2025 records without ambiguous year aliases or output-name collisions.",
            "Capture ordinary event identities and explicit Apertura/Clausura round keys; a named round of 16 can have round number 5 and must remain excluded.",
            "Verify the 16-round per-stage cross-zone opponent matrix and historical statuses before collection."],
        "reference_sources": ["https://www.afa.com.ar/upload/builder/NuevaCarpeta%201/NewFolder/Bolet%C3%ADn%205768%20%2827-4-2020%20%29.pdf",
            "https://www.afa.com.ar/site/posts/formato-de-los-torneos-de-lpf-para-2025",
            "https://assets1.afa.com.ar/2026/BOLETINES/Boletin-Resoluciones-6818---Complementario-01---Reglamento-Torneos-LPF-Primera-2026.pdf"],
        "validation": "Era-specific full league or two ordinary zones with exact cross-zone pairs; exclude all knockout stages and separate cups."},
    "Australia": {
        "resolved": ["Eight official historical matchup tables extracted for 2017/18-2024/25; seven reconcile all club played, W/D/L and goal totals with saved provider standings.",
            "2022/23 source matrix includes the 1-6 Grand Final; excluded only from the derived reference with exact page 321 evidence, retaining the original.",
            "2023/24 has 27 games per club and 162 ordinary matches; other seasons have distinct opponent multiplicities."],
        "blockers": ["2024/25 official matrix scores disagree with its own final ladder and provider totals: GF/GA and W/L are reversed for every club. Reference is quarantined, not silently corrected.",
            "2020/21 provider round inventory stops at 24 despite 26 played matches per club; inspect event rounds before configuring bounds.",
            "Verify actual events, rescheduled 2019/20 dates, awards and finals; obtain 2025/26 and current-snapshot exact opponent references."],
        "reference_sources": ["https://aleagues.com.au/wp-content/uploads/sites/17/2025/10/Isuzu_UTE_A-League_2025-26_Season_Guide.pdf"],
        "validation": "Exact per-season directed opponent matrix, per-club totals and result reconciliation; explicit regular-season/finals boundary."},
    "Chile": {
        "resolved": ["ANFP's original 2019 report recovered. Page 22 confirms 194 played matches and each of the 16 club rows matches saved provider standings on played/W/D/L/GF/GA/points.",
            "Provider inventory identifies 2017 Transicion and 18-team 2020/17-team 2021 seasons; 2021 includes byes."],
        "blockers": ["Resolve the 2016/17 Clausura identity against the calendar-2017 scope boundary; do not silently omit or add it.",
            "Reconcile the exact curtailed 2019 fixture set and all other event/status identities before collection; 194 alone is insufficient.",
            "Exclude relegation qualification ties; preserve delayed 2020 season identity."],
        "reference_sources": ["https://www.anfp.cl/?jet_download=9709a3d50c40172afee6952ce6416cf49c60464a",
            "https://cruzados.cl/wp-content/uploads/2022/02/Memoria-Cruzados-SADP-2017.pdf"],
        "validation": "Per-era full league or independently supported curtailed fixture set; byes and administrative events reviewed individually."},
    "China": {
        "resolved": ["2020 and 2021 first phases are 112 matches, two groups of eight with 14 games per club.",
            "Ordinary initial group identities mapped; 2021 aggregate table duplicates them, while later championship/relegation tables carry totals forward."],
        "blockers": ["Reconcile 2022 administrative awards and withdrawals against primary event-specific decisions; standings' nominal 306 is not an actual-played guarantee.",
            "Capture and validate the exact initial group events; do not interpret every table with two groups as ordinary.",
            "Reconcile other seasons and the 2026 snapshot at the actual discovery cutoff."],
        "reference_sources": ["https://www.thecfa.cn/zyls1/20200713/28732.html",
            "https://www.sport.gov.cn/n20001280/n20067662/n20067613/c23432765/content.html",
            "https://www.thecfa.cn/lstz/20220425/30604.html"],
        "validation": "Year-specific ordinary group allowlist, exact event-status exclusions and no carryover-table double counting."},
    "Colombia": {
        "resolved": ["Legacy competition 241 has no requested recent seasons and is not selected; 11539/11536 cover the observed tournaments.",
            "2020 has one ordinary league; 2026 is combined under 11539/88503, and 2017 Apertura/Clausura share a season number but have different competition IDs.",
            "2018 and 2024 have 19 ordinary games per club; 2021 Apertura has 19 clubs and 18 games. Duplicate 2021 Apertura tables are deduplicated by exact rows."],
        "blockers": ["Verify designated extra derby pairs independently for 20-round tournaments.",
            "Capture event lists for both ordinary tournaments and enforce unique competition/season/stage keys.",
            "Exclude championship quadrangular groups, 2020 Liguilla and finals despite grouped labels."],
        "reference_sources": ["https://dimayor.com.co/2017/12/12/balance-de-la-asamblea-extraordinaria-de-la-dimayor/",
            "https://dimayor.com.co/2020/09/09/sintesis-de-la-asamblea-extraordinaria-de-la-dimayor-9/",
            "https://dimayor.com.co/wp-content/uploads/2024/09/REGLAMENTO-LIGA_2DO-SEMESTRE_3.pdf",
            "https://dimayor.com.co/wp-content/uploads/2025/04/REGLAMENTO-LIGA-V1-2025-04-03.pdf",
            "https://dimayor.com.co/wp-content/uploads/2026/01/REGLAMENTO-LIGA-2026-V12.pdf"],
        "validation": "Tournament-specific single round robin, byes or exact additional derby matrix; ordinary-stage table reconciliation."},
    "Ecuador": {
        "resolved": ["Saved tables distinguish both full-field ordinary stages in 2017/18 and 2020-2024 from later rank-based splits in 2025/26.",
            "2017/18 each ordinary stage has 132 matches; 2020-2024 stages each report 120. Aggregate tables cannot be used as additional fixtures.",
            "2020 and 2022 aggregate tables disagree with the sums of ordinary-stage tables; these contradictions are retained for event-level reconciliation."],
        "blockers": ["Confirm earlier historical rules and reconcile 2020 aggregate 228 versus stage 240, and 2022 aggregate 237 versus stage 240.",
            "Capture explicit stage identities for repeated round numbers and validate both ordinary stages without accepting championship/relegation splits.",
            "Check every award, abandonment, duplicate and delayed fixture against original captured evidence."],
        "reference_sources": ["https://ligapro.ec/wp-content/uploads/2023/12/Reglamento-Competiciones-ed.-2024-1.pdf",
            "https://ligapro.ec/wp-content/uploads/2024/09/Sistema-de-Campeonato-2025-POST.pdf",
            "https://ligapro.ec/se-termino-la-espera-volvio-nuestra-liga-ecuabet/"],
        "validation": "Explicit full-field ordinary stage selection; per-stage pair counts and independent reconciliation of conflicting aggregate records."},
    "Peru": {
        "resolved": ["Mapped Verano/Apertura/Clausura, pandemic-era groups, 2021 crossover groups and changing 2025 memberships.",
            "2021 first-stage groups each sum to 81 club appearances: combined 81 matches, not 40+40 after rounding. Both ordinary crossover fixtures and later 153-match stage require explicit support.",
            "2017/18 Verano ordinary groups are retained as candidates; finals and aggregate tables are separate."],
        "blockers": ["Finish primary historical rule and exact award/withdrawal review; provider table totals are not proof that every match was played.",
            "Reconcile 2022 aggregate 337 against stage sum 342 and the 2025 change from 19 to 18 clubs.",
            "Capture exact 2020/2021 event-stage and crossover identities and verify that later finals stay excluded."],
        "reference_sources": ["https://img.fifa.com/image/upload/jlr5corccbsef4n4brde.pdf",
            "https://liga1.pe/wp-content/uploads/2026/03/Reglamento-Liga1-Te-Apuesto-2026.pdf"],
        "validation": "Per-stage team membership and opponent matrix, source-backed crossover rounds, exact event-status exceptions, no floor of partial-group appearance totals."},
    "USA": {
        "resolved": ["Mapped ten calendar-season provider IDs; full-league and conference tables overlap and must not be added together.",
            "Saved rounds for available MLS seasons contain playoff labels only. A round-1-to-N discovery cannot safely recover the ordinary schedule."],
        "blockers": ["Review and implement a paginated ordinary event-list discovery route after Egypt, with complete-page coverage and no invented rounds.",
            "Resolve the retained MLS 2022 rounds HTTP 404 and eleven pending metadata jobs through a supported route; do not repeatedly retry the active connection.",
            "Build historical conference/cross-conference matrices and reconcile 2020 MLS is Back group games, replacement games and unequal final club totals."],
        "reference_sources": ["https://www.mlssoccer.com/news/2020-mls-competition-guidelines",
            "https://www.mlssoccer.com/news/mls-use-points-game-determine-final-2020-regular-season-standings",
            "https://www.mlssoccer.com/news/mls-conference-alignment-and-schedule-construction-for-2021-season",
            "https://www.mlssoccer.com/news/2022-mls-regular-season-schedule-announced"],
        "validation": "Full paginated event coverage, unequal conference and cross-conference schedule matrices, recorded completed-match cutoff and exact 2020 exceptions."},
}


def dump(name, data):
    (BASE / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def phase_decision(cid, sid, table):
    tid, name = table["tournament"]["id"], table.get("name", "").lower()
    if cid == 155:
        return "ordinary_candidate" if not any(x in name for x in ["anual", "promedios"]) else "aggregate_not_additional_stage"
    if cid in {136, 11653}:
        return "ordinary_candidate"
    if cid == 649:
        if sid == 36257:
            return "ordinary_candidate" if tid in {97576,97577} else ("aggregate_not_additional_stage" if tid == 652 else "excluded_rank_based_split")
        return "ordinary_candidate"
    if cid in {11539,11536}:
        return "ordinary_candidate" if tid in {19232,19236,189591} else "excluded_later_stage"
    if cid == 240:
        if sid in {20069,71184,89674}:
            return "ordinary_candidate" if tid in {16887,70775} else "excluded_rank_based_split"
        return "ordinary_candidate" if (tid == 1387 or "first stage" in name or "primera etapa" in name) else "aggregate_not_additional_stage"
    if cid == 406:
        return "aggregate_not_additional_stage" if ("overall" in name or "acumulada" in name or name in {"primera division", "liga 1"}) else "ordinary_candidate"
    if cid == 242:
        return "conference_audit_only_not_additional_events" if "conference" in name else "ordinary_candidate"
    raise AssertionError(cid)


def chile_reference(responses):
    # Independently transcribed from the visually inspected ANFP 2019 report, page 22.
    official = [
        (3151,"Universidad Catolica",24,16,5,3,44,14,53),
        (3155,"Colo-Colo",24,11,7,6,37,30,40),(3157,"Palestino",24,10,8,6,42,31,38),
        (48242,"Union La Calera",25,9,10,6,29,23,37),(3165,"Coquimbo Unido",24,8,10,6,29,27,34),
        (3164,"Huachipato",24,9,7,8,31,30,34),(3162,"Audax Italiano",24,10,4,10,35,35,34),
        (3163,"O'Higgins",24,10,4,10,34,35,34),(3153,"Union Espanola",25,10,4,11,32,35,34),
        (3167,"Cobresal",25,10,4,11,31,39,34),(5032,"Everton",24,7,8,9,21,24,29),
        (5601,"Deportes Antofagasta",24,7,6,11,34,35,27),(33595,"Curico Unido",24,6,8,10,36,43,26),
        (33596,"Deportes Iquique",25,6,7,12,24,40,25),(3161,"Universidad de Chile",24,4,12,8,32,38,24),
        (5034,"Universidad de Concepcion",24,5,8,11,23,35,23)]
    path = "/unique-tournament/11653/season/22328/standings/total"
    saved = {r["team"]["id"]: r for r in responses[path]["data"]["standings"][0]["rows"]}
    checked = []
    for team_id, name, *values in official:
        ref = dict(zip(FIELDS,values))
        assert all(saved[team_id][k] == v for k,v in ref.items()), (team_id,ref)
        checked.append({"team_id":team_id,"official_club_label":name,"provider_name":saved[team_id]["team"]["name"],**ref})
    assert sum(r["matches"] for r in checked) == 388
    dump("chile_2019_official_table.json", {
        "status":"official_table_reconciled_not_fixture_validation","source_page":22,
        "source_url":RULES["Chile"]["reference_sources"][0],
        "source_sha256":hashlib.sha256((SOURCE_BASE/"sources/anfp-memoria-2019.pdf").read_bytes()).hexdigest(),
        "provider_response_path":path,"provider_response_sha256":responses[path]["sha256"],
        "clubs":checked,"ordinary_matches":194,"all_16_provider_rows_match":True,
        "pending":"Exact event IDs, pairings, round occupancy and status reconciliation remain required."})


def main():
    global BASE, INV
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New review directory; never overwrites an earlier review")
    parser.add_argument("--inventory", type=Path, default=INV)
    args = parser.parse_args()
    BASE, INV = args.output.resolve(), args.inventory.resolve()
    BASE.mkdir(parents=True, exist_ok=False)
    now = datetime.now(timezone.utc).isoformat()
    con = sqlite3.connect((INV / "discovery.sqlite").resolve().as_uri()+"?mode=ro",uri=True)
    con.row_factory = sqlite3.Row
    responses = {}
    for row in con.execute("select path,body,sha256,observed from responses"):
        assert hashlib.sha256(row["body"].encode()).hexdigest() == row["sha256"], row["path"]
        responses[row["path"]] = {"data":json.loads(row["body"]),"sha256":row["sha256"],"observed":row["observed"]}
    jobs = dict(con.execute("select state,count(*) from jobs group by state"))
    unfinished = [dict(r) for r in con.execute("select path,competition,kind,season,round,state,error from jobs where state!='complete'")]
    event_count = con.execute("select count(*) from events").fetchone()[0]
    con.close()
    candidates = []
    for country, output, cid, ids in SELECTIONS:
        catalog_path = f"/unique-tournament/{cid}/seasons"
        catalog = {s["id"]:s for s in responses[catalog_path]["data"]["seasons"]}
        for sid in ids:
            source = catalog[sid]
            entry = {"country":country,"output_name_family":output,"competition_id":cid,"season_id":sid,
                "provider_name":source["name"],"provider_year":source["year"],
                "catalog_evidence":{"path":catalog_path,"sha256":responses[catalog_path]["sha256"]},
                "status":"metadata_mapped_fixture_validation_pending","tables":[]}
            seen = set()
            for suffix in ["rounds","standings/total"]:
                path = f"/unique-tournament/{cid}/season/{sid}/{suffix}"
                record = responses.get(path)
                if record is None:
                    entry[suffix.replace('/','_')] = {"status":"not_captured","path":path}
                    continue
                entry[suffix.replace('/','_')] = {"status":"captured","path":path,"sha256":record["sha256"],"observed":record["observed"]}
                if suffix == "rounds":
                    entry["round_inventory"] = record["data"]
                    continue
                for table in record["data"].get("standings",[]):
                    rows = [{"team_id":r["team"]["id"],"team_name":r["team"]["name"],
                             **{k:r.get(k) for k in FIELDS}} for r in table["rows"]]
                    decision = phase_decision(cid,sid,table)
                    fingerprint = json.dumps(sorted(rows,key=lambda r:r["team_id"]),sort_keys=True)
                    if decision == "ordinary_candidate":
                        if fingerprint in seen: decision = "duplicate_ordinary_table_not_additional_events"
                        seen.add(fingerprint)
                    entry["tables"].append({"standings_id":table["id"],"tournament_id":table["tournament"]["id"],
                        "name":table.get("name"),"isGroup":table["tournament"].get("isGroup"),
                        "decision":decision,"clubs":len(rows),
                        "club_appearances":sum(r["matches"] for r in rows),"rows":rows})
            appearances = sum(t["club_appearances"] for t in entry["tables"] if t["decision"] == "ordinary_candidate")
            entry["ordinary_table_matches_observed"] = appearances / 2 if entry["tables"] else None
            entry["count_warning"] = "Metadata checkpoint only; current snapshots can advance and administrative awards may be included. Not an approved capture target."
            candidates.append(entry)
    assert len(candidates) == 88
    assert len({(e["competition_id"],e["season_id"]) for e in candidates}) == 88
    chile_reference(responses)
    evidence = {"recorded_at":now,"inventory_id":INV.name,"read_only":True,
        "response_count":len(responses),"all_original_response_body_hashes_match":True,
        "response_hashes":{k:v["sha256"] for k,v in sorted(responses.items())},
        "config_sha256":hashlib.sha256((INV/"config.json").read_bytes()).hexdigest(),
        "jobs":jobs,"unfinished_jobs":unfinished,"event_count":event_count}
    dump("inventory_evidence.json",evidence)
    draft = {"kind":"offline_preparation_not_executable_batch","created_at":now,
        "status":"prepared_metadata_and_reference_checks_fixture_discovery_pending",
        "collection_ready":False,
        "scope":"Remaining eight requested leagues, split seasons from 2017/18 or calendar 2017 through completed 2026/27 or calendar 2026 fixtures. Existing ordinary-stage exclusions remain in force.",
        "grouping":"Combined preparation for all eight leagues. Do not create one collection campaign per league by default. Final batch/output boundaries require stage and filename checks.",
        "provider_season_candidates":88,"intended_output_count":None,
        "count_note":"Provider season records are not output seasons or collection waves; separate and combined tournaments must be reconciled first.",
        "source_inventory":"inventory_evidence.json","league_reviews":RULES,"seasons":candidates,
        "startup_gates":[
            "Finish and validate Egypt with its existing worker and settings.",
            "Resolve outstanding historical source/format contradictions; preserve evidence and missing statistics.",
            "Implement exact season and stage selection plus necessary schedule validators, with focused regression checks, while disconnected from the completed campaign.",
            "Capture original fixture inventories through the existing browser; audit all intended seasons, statuses, counts, opponent pairs, round occupancy and snapshot boundaries.",
            "Require every selected season to pass and an exact deduplicated set before creating the next combined collection campaign. Reuse saved progress; never reset Egypt."]}
    dump("next_wave_draft.json",draft)
    print(json.dumps({"candidates":len(candidates),"countries":dict(Counter(e['country'] for e in candidates)),
                      "inventory_hashes_verified":len(responses),"chile2019_rows_checked":16,"collection_ready":False}))


if __name__ == "__main__":
    main()
