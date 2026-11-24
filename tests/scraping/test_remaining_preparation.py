"""Reviewed archive selectors and schedule checks; no live data or network."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket

import pytest

from scraping.sofascore_web_v3.archive_discovery import prepare_archive, validate_specs
from scraping.sofascore_web_v3.archive_formats import _spec
from scraping.sofascore_web_v3.archive_selection import (
    resolve_captured_season, select_captured_tables, validate_selector_identities)
from scraping.sofascore_web_v3.discovery import open_discovery
from scraping.sofascore_web_v3.event_pages import validate_event_pages
from scraping.sofascore_web_v3.schedule_reference import validate_fixture_reference
from test_archive_discovery import events as synthetic_schedule, drain, observe

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/sofascore/coverage_remaining_metadata.json"
EVIDENCE = json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: pytest.fail("Offline candidate sent a network request"))
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("Offline candidate sent a network request"))


def captured(path):
    record = EVIDENCE["responses"][path]
    body, digest = record["body"], record["sha256"]
    assert hashlib.sha256(body.encode()).hexdigest() == digest
    return json.loads(body)


def pinned(name, sid, year="2025", cid=155):
    return _spec(name, cid, year, teams=4, rounds=3, meetings=1, provider_season_id=sid)


def test_real_argentina_same_year_records_select_exact_ids_without_mutation():
    values = captured("/unique-tournament/155/seasons")["seasons"]
    before = deepcopy(values)
    specs = [pinned("Liga_Profesional_Apertura", 70268), pinned("Liga_Profesional_Clausura", 77826)]
    validate_specs(specs)
    assert [resolve_captured_season(values,s)["id"] for s in specs] == [70268,77826]
    assert values == before
    unpinned = deepcopy(specs[0]); unpinned.pop("provider_season_id")
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_captured_season(values, unpinned)


@pytest.mark.parametrize("mutation", ["missing_pin", "foreign_year", "duplicate_pin_record"])
def test_exact_pin_never_falls_back(mutation):
    values = [{"id":70268,"year":"2025"},{"id":77826,"year":"2025"}]
    spec = pinned("League",70268)
    if mutation == "missing_pin":
        assert resolve_captured_season(values[1:],spec) is None
    else:
        if mutation == "foreign_year": values[0]["year"]="2024"
        else: values.append(deepcopy(values[0]))
        with pytest.raises(ValueError): resolve_captured_season(values,spec)


@pytest.mark.parametrize("mutation", ["same_output", "same_pin", "mixed_pinned_alias", "bad_pin", "same_pin_other_year"])
def test_conflicting_selectors_stay_blocked(mutation):
    a,b=pinned("Apertura",70268),pinned("Clausura",77826)
    if mutation=="same_output": b["name"]=a["name"]
    elif mutation=="same_pin": b["provider_season_id"]=70268
    elif mutation=="mixed_pinned_alias": b.pop("provider_season_id")
    elif mutation=="bad_pin": b["provider_season_id"]=True
    else: b.update(provider_season_id=70268,season="2024",aliases=["2024"])
    with pytest.raises(ValueError): validate_specs([a,b])


def test_two_pinned_same_year_seasons_make_one_test_campaign_and_resume(tmp_path):
    specs=[pinned("Example_Apertura",70268),pinned("Example_Clausura",77826)]
    saved=prepare_archive("offline-test",specs,base=tmp_path/"batches",discovery_base=tmp_path/"discoveries",output=tmp_path/"outputs")
    values=synthetic_schedule(range(1,5),year="2025",cid=155,sid=70268,meetings=1,id_base=1000)
    values+=synthetic_schedule(range(1,5),year="2025",cid=155,sid=77826,meetings=1,id_base=2000)
    before=deepcopy(values)
    with open_discovery(saved["directory"]) as discovery:
        drain(discovery,values)
        state=discovery.status(refresh=True)
        assert state["ready_to_collect"] and len(state["seasons"])==2
        assert {s["stem"] for s in state["seasons"]}=={"Example_Apertura_25_25","Example_Clausura_25_25"}
        first=discovery.prepare_collection(campaign_base=tmp_path/"campaigns")
    with open_discovery(saved["directory"]) as discovery:
        assert discovery.prepare_collection(campaign_base=tmp_path/"campaigns")==first
    config=json.loads((Path(first["directory"])/"config.json").read_text(encoding="utf-8"))
    assert len(config["items"])==12
    assert len({i["event_id"] for i in config["items"]})==12
    assert len(config["source_metadata"]["seasons"])==2
    assert values==before


def test_missing_pinned_tournament_blocks_collection_even_with_other_completed(tmp_path):
    specs=[pinned("Example_Apertura",70268),pinned("Example_Clausura",77826)]
    saved=prepare_archive("offline-missing",specs,base=tmp_path/"batches",discovery_base=tmp_path/"discoveries",output=tmp_path/"outputs")
    values=synthetic_schedule(range(1,5),year="2025",cid=155,sid=70268,meetings=1)
    with open_discovery(saved["directory"]) as discovery:
        drain(discovery,values)
        assert not discovery.status(refresh=True)["ready_to_collect"]
        with pytest.raises(ValueError,match="every requested season"):
            discovery.prepare_collection(campaign_base=tmp_path/"campaigns")


def test_all_88_saved_provider_records_resolve_exactly():
    draft=EVIDENCE
    specs=[]
    for record in draft["seasons"]:
        cid,sid=record["competition_id"],record["season_id"]
        spec=pinned(f"Identity_{cid}_{sid}",sid,year=record["provider_year"],cid=cid)
        catalog=captured(f"/unique-tournament/{cid}/seasons")["seasons"]
        assert resolve_captured_season(catalog,spec)["id"]==sid
        specs.append(spec)
    assert len(specs)==88
    validate_specs(specs)


def test_all_captured_candidate_phase_tables_keep_exact_identities():
    draft=EVIDENCE
    checked=0
    for record in draft["seasons"]:
        if not record["tables"]: continue
        selectors=[{k:t[k] for k in ("standings_id","tournament_id","name")}
                   for t in record["tables"] if t["decision"]=="ordinary_candidate"]
        payload=captured(record["standings_total"]["path"])
        before=deepcopy(payload)
        result=select_captured_tables(payload,competition_id=record["competition_id"],selectors=selectors)
        assert sum(r["matches"] for t in result for r in t["rows"])/2==record["ordinary_table_matches_observed"]
        assert payload==before
        checked+=1
    assert checked==82  # Six earlier MLS standings responses are still absent.


def test_real_colombia_duplicate_table_is_not_an_extra_stage():
    payload=captured("/unique-tournament/11539/season/35235/standings/total")
    selectors=[{"standings_id":t["id"],"tournament_id":t["tournament"]["id"],"name":t["name"]} for t in payload["standings"]]
    selectors[1]["duplicate_of"]=selectors[0]["standings_id"]
    result=select_captured_tables(payload,competition_id=11539,selectors=selectors)
    assert len(result)==1 and sum(r["matches"] for r in result[0]["rows"])==342


def test_matching_stage_rows_are_not_silently_merged():
    payload=captured("/unique-tournament/11539/season/35235/standings/total")
    selectors=[{"standings_id":t["id"],"tournament_id":t["tournament"]["id"],"name":t["name"]} for t in payload["standings"]]
    assert len(select_captured_tables(payload,competition_id=11539,selectors=selectors))==2
    selectors[1]["duplicate_of"]=selectors[0]["standings_id"]
    payload["standings"][1]["rows"][0]["matches"]-=1
    with pytest.raises(ValueError,match="duplicate"):
        select_captured_tables(payload,competition_id=11539,selectors=selectors)


@pytest.mark.parametrize("mutation",["id","tournament","name","competition","type","duplicate","missing","team_duplicate"])
def test_changed_stage_evidence_is_rejected(mutation):
    payload=captured("/unique-tournament/649/season/36257/standings/total")
    t=next(t for t in payload["standings"] if t["tournament"]["id"]==97576)
    selector={"standings_id":t["id"],"tournament_id":97576,"name":t["name"]}
    if mutation=="id": t["id"]+=1
    elif mutation=="tournament": t["tournament"]["id"]+=1
    elif mutation=="name": t["name"]="Championship Group A"
    elif mutation=="competition": t["tournament"]["uniqueTournament"]["id"]+=1
    elif mutation=="type": t["type"]="home"
    elif mutation=="duplicate": payload["standings"].append(deepcopy(t))
    elif mutation=="missing": payload["standings"].remove(t)
    else: t["rows"].append(deepcopy(t["rows"][0]))
    with pytest.raises(ValueError): select_captured_tables(payload,competition_id=649,selectors=[selector])


def reference_case(*,two_stages=False):
    # Entirely synthetic round robin, plus an explicitly scheduled extra pair.
    values=synthetic_schedule(range(1,5),year="2025",cid=155,sid=70268,meetings=1)
    if two_stages:
        extra=deepcopy(values)
        for e in extra: e["id"]+=1000; e["tournament"]["id"]=156
        values+=extra
    else:
        extra=deepcopy(values[0]);extra["id"]+=1000;extra["roundInfo"]["round"]=4
        values.append(extra)
    pairs=Counter((e["tournament"]["id"],e["homeTeam"]["id"],e["awayTeam"]["id"]) for e in values)
    rounds={}
    for e in values: rounds.setdefault(str(e["tournament"]["id"]),set()).add(e["roundInfo"]["round"])
    ref={"scope":"full_ordinary_schedule","source_sha256":"0"*64,"source_url":"https://example.test/synthetic-test-only",
         "competition_id":155,"season_id":70268,"season_year":"2025",
         "rounds_by_tournament":{k:sorted(v) for k,v in rounds.items()},
         "directed_opponents":[{"tournament_id":tid,"home":h,"away":a,"count":n} for (tid,h,a),n in pairs.items()],
         "played_per_club":dict(Counter(e[side]["id"] for e in values for side in ("homeTeam","awayTeam"))),
         "results":[{"tournament_id":e["tournament"]["id"],"home":e["homeTeam"]["id"],"away":e["awayTeam"]["id"],"score":[e["homeScore"]["current"],e["awayScore"]["current"]]} for e in values]}
    return values,ref


@pytest.mark.parametrize("two_stages",[False,True])
def test_uneven_pairs_and_repeated_rounds_require_exact_reference(two_stages):
    values,ref=reference_case(two_stages=two_stages)
    before=deepcopy((values,ref))
    assert validate_fixture_reference(values,ref)==[]
    assert (values,ref)==before


@pytest.mark.parametrize("mutation",["missing","duplicate_id","extra_pair","foreign_team","foreign_season","foreign_competition",
    "wrong_year","score","awarded","unfinished","round_collision","round_missing","other_stage","final","round_of_16"])
def test_reference_rejects_incomplete_or_changed_fixtures(mutation):
    values,ref=reference_case(); e=values[0]
    if mutation=="missing": values.pop()
    elif mutation=="duplicate_id": values[-1]["id"]=e["id"]
    elif mutation=="extra_pair": values.append(deepcopy(e));values[-1]["id"]+=10000
    elif mutation=="foreign_team": e["homeTeam"]["id"]=99999
    elif mutation=="foreign_season": e["season"]["id"]+=1
    elif mutation=="foreign_competition": e["tournament"]["uniqueTournament"]["id"]+=1
    elif mutation=="wrong_year": e["season"]["year"]="2024"
    elif mutation=="score": e["homeScore"]["current"]+=1
    elif mutation=="awarded": e["isAwarded"]=True
    elif mutation=="unfinished": e["status"]={"type":"notstarted","code":0}
    elif mutation=="round_collision": values[-1]["roundInfo"]["round"]=1
    elif mutation=="round_missing": values[-1]["roundInfo"]["round"]=5
    elif mutation=="other_stage": e["tournament"]["id"]=999
    elif mutation=="final": e["roundInfo"]["name"]="Final"
    else: e["roundInfo"]["name"]="Round of 16"
    assert validate_fixture_reference(values,ref)


def test_changed_pairs_rejected_even_when_club_totals_still_match():
    values,ref=reference_case()
    a,b=values[2:4]
    a["awayTeam"],b["awayTeam"]=b["awayTeam"],a["awayTeam"]
    assert dict(Counter(e[s]["id"] for e in values for s in ("homeTeam","awayTeam")))==ref["played_per_club"]
    assert validate_fixture_reference(values,ref)


def test_wrong_stage_assignments_rejected_with_unchanged_whole_season_totals():
    values,ref=reference_case(two_stages=True)
    # Swap complete rounds containing different pairs. Both stages still have
    # every round and every club, and the season-wide pair/result totals match.
    first=[e for e in values if e["tournament"]["id"]==155 and e["roundInfo"]["round"]==1]
    second=[e for e in values if e["tournament"]["id"]==156 and e["roundInfo"]["round"]==2]
    for e in first: e["tournament"]["id"]=156;e["roundInfo"]["round"]=2
    for e in second: e["tournament"]["id"]=155;e["roundInfo"]["round"]=1
    assert validate_fixture_reference(values,ref)


@pytest.mark.parametrize("mutation",["round_contract","matrix_count","played","result","provenance","snapshot"])
def test_incomplete_reference_cannot_approve_a_schedule(mutation):
    values,ref=reference_case()
    if mutation=="round_contract": ref.pop("rounds_by_tournament")
    elif mutation=="matrix_count": ref["directed_opponents"][0]["count"]+=1
    elif mutation=="played": ref["played_per_club"][1]+=1
    elif mutation=="result": ref["results"].pop()
    elif mutation=="provenance": ref.pop("source_sha256")
    else: ref["scope"]="snapshot"
    with pytest.raises(ValueError): validate_fixture_reference(values,ref)


def page_case():
    values=synthetic_schedule(range(1,5),cid=242,sid=40071,year="2022",meetings=1)
    return [{"page":0,"payload":{"hasNextPage":True,"events":values[:3]}},
            {"page":1,"payload":{"hasNextPage":False,"events":values[3:]}}]


def test_complete_page_chain_preserves_original_events():
    pages=page_case(); before=deepcopy(pages)
    assert len(validate_event_pages(pages,competition_id=242,season_id=40071))==6
    assert pages==before


@pytest.mark.parametrize("mutation",["missing_first","missing_last","gap","repeat_page","missing_flag","string_flag","false_terminal",
    "empty_intermediate","repeated_event","foreign_season","foreign_competition"])
def test_incomplete_or_inconsistent_pages_block_discovery(mutation):
    pages=page_case()
    if mutation=="missing_first": pages.pop(0)
    elif mutation=="missing_last": pages.pop()
    elif mutation=="gap": pages[1]["page"]=2
    elif mutation=="repeat_page": pages[1]["page"]=0
    elif mutation=="missing_flag": pages[-1]["payload"].pop("hasNextPage")
    elif mutation=="string_flag": pages[-1]["payload"]["hasNextPage"]="false"
    elif mutation=="false_terminal": pages[0]["payload"]["hasNextPage"]=False
    elif mutation=="empty_intermediate": pages[0]["payload"]["events"]=[]
    elif mutation=="repeated_event": pages[1]["payload"]["events"].append(deepcopy(pages[0]["payload"]["events"][0]))
    elif mutation=="foreign_season": pages[0]["payload"]["events"][0]["season"]["id"]+=1
    else: pages[0]["payload"]["events"][0]["tournament"]["uniqueTournament"]["id"]+=1
    with pytest.raises(ValueError): validate_event_pages(pages,competition_id=242,season_id=40071)


@pytest.mark.parametrize("value",[None, [], "invalid"])
def test_malformed_page_payloads_are_rejected(value):
    pages=page_case();pages[0]["payload"]=value
    with pytest.raises(ValueError): validate_event_pages(pages,competition_id=242,season_id=40071)
