"""Offline regressions for frozen multi-stage Chrome evidence imports."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket

import pytest

from scraping.sofascore_web_v3.archive_discovery import prepare_archive
from scraping.sofascore_web_v3.discovery import open_discovery
from scraping.sofascore_web_v3.paged_archive import import_reviewed_evidence
from scraping.sofascore_web_v3.phase_schedule import fixture_identity, reconcile_phase, validate_phase_format


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **kw: pytest.fail("Offline evidence test"))


def sample():
    cid, sid = 155, 999999
    base = json.loads((Path(__file__).parents[1] / "fixtures/sofascore/12436870_event.json").read_text(encoding="utf8"))["event"]
    events, tables, phases = [], [], []
    for tid in (10, 20):
        for n, (home, away) in enumerate(((1, 2), (2, 1)), 1):
            event = deepcopy(base)
            event.update(id=tid+n, season={"id": sid, "year": "2024"},
                tournament={"id": tid, "uniqueTournament": {"id": cid}},
                homeTeam={"id": home, "name": f"Team {home}"}, awayTeam={"id": away, "name": f"Team {away}"},
                roundInfo={"round": n}, startTimestamp=1705000000+tid*10000+n*100,
                homeScore={"current": 2}, awayScore={"current": 0}, status={"type": "finished", "code": 100})
            event.pop("isAwarded", None)
            events.append(event)
        table = dict(id=tid, name=f"Ordinary {tid}", type="total", tournament={"id":tid,"uniqueTournament":{"id":cid}},
            rows=[dict(team={"id":t,"name":f"Team {t}"},matches=2,wins=1,draws=0,losses=1,scoresFor=2,scoresAgainst=2) for t in (1,2)])
        tables.append(table)
        phases.append(dict(competition_id=cid,season_id=sid,tournament_id=tid,teams=2,snapshot=False,
            table_selectors=[dict(standings_id=tid,tournament_id=tid,name=table["name"])],
            club_groups=[[1,2]],pair_rule="round_robin",meetings=2,round_mode="strict",rounds=[1,2]))
    knockout = deepcopy(events[0]); knockout["id"] = 90; knockout["tournament"]["id"] = 90
    knockout["roundInfo"] = {"round":1,"name":"Final"}
    paths = [f"/unique-tournament/{cid}/seasons", f"/unique-tournament/{cid}/season/{sid}/standings/total",
             f"/unique-tournament/{cid}/season/{sid}/events/last/0"]
    payloads = [{"seasons":[{"id":sid,"year":"2024"}]}, {"standings":tables},
                {"hasNextPage":False,"events":events+[knockout]}]
    responses=[]
    for path, payload in zip(paths,payloads):
        body=json.dumps(payload)
        responses.append(dict(path=path,observed=1750000000,body=body,sha256=hashlib.sha256(body.encode()).hexdigest(),origin="offline regression fixture"))
    contract=dict(schema=1,sources=[{k:r[k] for k in ("path","observed","sha256")} for r in responses],
        page_paths=paths[2:],standings_path=paths[1],phases=phases,event_reviews=[],table_reviews=[],
        selected_matches=4,selected_ids_sha256=hashlib.sha256(b"[11,12,21,22]").hexdigest(),review_sources=["offline format fixture"])
    spec=dict(name="Test_League",competition_id=cid,provider_season_id=sid,season="2024",aliases=["2024"],
              meetings_per_pair=2,expected_teams=None,round_limit=None,groups=1,snapshot=False,fixture_contract=contract)
    return spec,responses,events,tables


def prepare(tmp_path, spec):
    return prepare_archive("paged-test",[spec],base=tmp_path/"batches",discovery_base=tmp_path/"discovery",output=tmp_path/"output")


def test_import_then_queue_preserves_distinct_stages_round_resets_and_original_evidence(tmp_path):
    spec,responses,events,_=sample()
    original=deepcopy(responses)
    saved=prepare(tmp_path,spec)
    with open_discovery(saved["directory"]) as discovery:
        assert not discovery.status()["ready_to_collect"]
        status=import_reviewed_evidence(discovery,responses,origin="original inventory")
        assert status["ready_to_collect"]
        assert status["requests"]==0
        assert status["seasons"][0]["selected_matches"]==4
        assert len(status["seasons"][0]["reviewed_phases"])==2
        assert import_reviewed_evidence(discovery,responses,origin="original inventory")==status
        result=discovery.prepare_collection(campaign_base=tmp_path/"campaigns")
        assert discovery.db.execute("SELECT COUNT(*) FROM events").fetchone()[0]==5
        assert {r[0] for r in discovery.db.execute("SELECT sha256 FROM responses")}=={r["sha256"] for r in responses}
        with pytest.raises(ValueError,match="uncollected"):
            import_reviewed_evidence(discovery,responses,origin="original inventory")
    config=json.loads((Path(result["directory"])/"config.json").read_text())
    assert [r["event_id"] for r in config["items"]]==[e["id"] for e in events]
    assert [r["reference"]["event"] for r in config["items"]]==events
    assert responses==original


@pytest.mark.parametrize("mutation",["body","timestamp","target","nonterminal","foreign","table_goals","missing","duplicate","unreviewed_round"])
def test_import_rejects_corrupt_or_incomplete_evidence_and_rolls_back(tmp_path,mutation):
    spec,responses,_,_=sample()
    contract=spec["fixture_contract"]
    if mutation=="body":responses[2]["body"]+=" "
    elif mutation=="timestamp":responses[2]["observed"]+=1
    elif mutation=="target":contract["selected_matches"]+=1
    else:
        idx=1 if mutation=="table_goals" else 2
        payload=json.loads(responses[idx]["body"])
        if mutation=="nonterminal":payload["hasNextPage"]=True
        elif mutation=="foreign":payload["events"][0]["season"]["id"]+=1
        elif mutation=="table_goals":payload["standings"][0]["rows"][0]["scoresFor"]+=1
        elif mutation=="missing":payload["events"].pop(0)
        elif mutation=="duplicate":
            extra=deepcopy(payload["events"][0]);extra["id"]=50;payload["events"].append(extra)
        elif mutation=="unreviewed_round":payload["events"][0]["roundInfo"]={"round":77}
        responses[idx]["body"]=json.dumps(payload)
        responses[idx]["sha256"]=hashlib.sha256(responses[idx]["body"].encode()).hexdigest()
        contract["sources"][idx]["sha256"]=responses[idx]["sha256"]
    saved=prepare(tmp_path,spec)
    with open_discovery(saved["directory"]) as discovery:
        with pytest.raises(ValueError):import_reviewed_evidence(discovery,responses,origin="test inventory")
        assert discovery.db.execute("SELECT COUNT(*) FROM responses").fetchone()[0]==0
        assert not discovery.status(refresh=True)["ready_to_collect"]


@pytest.mark.parametrize("mutation",["payload","round","resolved","regular","observation"])
def test_index_corruption_blocks_later_campaign_creation(tmp_path,mutation):
    spec,responses,_,_=sample();saved=prepare(tmp_path,spec)
    with open_discovery(saved["directory"]) as discovery:
        assert import_reviewed_evidence(discovery,responses,origin="test")["ready_to_collect"]
        if mutation=="resolved":discovery.db.execute("DELETE FROM state WHERE key='resolved:0'")
        elif mutation=="payload":discovery.db.execute("UPDATE events SET payload='{}' WHERE id=11")
        elif mutation=="observation":discovery.db.execute("UPDATE events SET observed=1 WHERE id=11")
        elif mutation=="round":discovery.db.execute("UPDATE events SET round=8 WHERE id=11")
        else:discovery.db.execute("UPDATE events SET regular=0 WHERE id=11")
        discovery.db.commit()
        assert not discovery.status(refresh=True)["ready_to_collect"]
        with pytest.raises(ValueError):discovery.prepare_collection(campaign_base=tmp_path/"campaigns")


def test_administrative_score_reconciles_table_but_unplayed_match_is_not_captured():
    spec,_,events,tables=sample();event=events[0]
    review=dict(event_id=event["id"],guard=fixture_identity(event),disposition="exclude_unplayed",
                reason="Fixture-specific official award",standings_score=[2,0],source="offline ruling")
    result=reconcile_phase(events[:2],tables[:1],event_reviews=[review])
    assert result["selected_ids"]==[12]
    assert result["standings_ids"]==[11,12] and result["table_differences"]==[]
    assert validate_phase_format(events[:2],tables[:1],spec["fixture_contract"]["phases"][0],result)["matches"]==2
    changed=deepcopy(event);changed["homeScore"]["current"]=3
    with pytest.raises(ValueError,match="identity changed"):
        reconcile_phase([changed,events[1]],tables[:1],event_reviews=[review])


def test_pair_contract_rejects_equal_table_totals_with_wrong_home_allocation():
    spec,_,events,tables=sample()
    # Same clubs, W/D/L and goals, but both matches incorrectly assigned to one home club.
    wrong=deepcopy(events[:2]);wrong[1]["homeTeam"],wrong[1]["awayTeam"]=wrong[1]["awayTeam"],wrong[1]["homeTeam"]
    wrong[1]["homeScore"],wrong[1]["awayScore"]=wrong[1]["awayScore"],wrong[1]["homeScore"]
    result=reconcile_phase(wrong,tables[:1]);assert not result["table_differences"]
    with pytest.raises(ValueError,match="home allocation"):
        validate_phase_format(wrong,tables[:1],spec["fixture_contract"]["phases"][0],result)
