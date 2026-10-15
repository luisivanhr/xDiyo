from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import socket
import sqlite3

import pytest

from scraping.sofascore_web_v3.queue_source import prepare_from_scraper_queue,queue_preview,IDENTITY_COLUMNS
from scraping.sofascore_web_v3.campaign import Campaign,prepare_campaign
from scraping.sofascore.export import read_tables

FIXTURES=Path(__file__).parents[1]/"fixtures/sofascore"
EID=12436870

def source(tmp_path, *, cache=True):
    event=json.loads((FIXTURES/f"{EID}_event.json").read_text())["event"]
    csv_path=tmp_path/"Premier_League_24_25.csv"
    with csv_path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=[*IDENTITY_COLUMNS,"home_score","shots","home_got_promoted"])
        writer.writeheader()
        for eid in (EID,EID+1):
            writer.writerow({"match_id":eid,"custom_id":event["customId"],"season_id":event["season"]["id"],
                "season_year":event["season"]["year"],"round":event["roundInfo"]["round"],
                **{side+suffix: event[side+"Team"][key] for side in ("home","away") for suffix,key in (("_id","id"),("_team","slug"))},
                "home_score":999,"shots":"OLD_VALUES_MUST_NOT_APPEAR","home_got_promoted":1})
    item={"stem":csv_path.stem,"source":str(csv_path),"source_sha256":hashlib.sha256(csv_path.read_bytes()).hexdigest(),
          "legacy_identity_valid":True,"legacy_event_ids":[EID,EID+1],"legacy_rows":2,
          "season_id":event["season"]["id"],"season_start":2024,"season_end":2025,"rounds":[event["roundInfo"]["round"]],
          "league_id":event["tournament"]["uniqueTournament"]["id"]}
    inventory=tmp_path/"inventory.json"
    inventory.write_text(json.dumps({"seasons":[item]}))
    cache_root=tmp_path/"cache"
    if cache:
        cache_root.mkdir()
        with sqlite3.connect(cache_root/"manifest.sqlite") as db:
            db.execute("CREATE TABLE events(event_id INTEGER PRIMARY KEY,payload TEXT,observed REAL)")
            db.execute("INSERT INTO events VALUES(?,?,?)",(EID,json.dumps(event),1700000000))
    return inventory,cache_root,event,item

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def reject(*args,**kwargs): raise AssertionError("Queue reuse must not make network connections")
    monkeypatch.setattr(socket,"create_connection",reject)

def test_imports_original_ids_in_order_with_minimal_identity_and_no_old_values(tmp_path):
    inventory,cache,event,item=source(tmp_path)
    before={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in (inventory,Path(item["source"]),cache/"manifest.sqlite")}
    result=prepare_from_scraper_queue(inventory,cache_root=cache,base=tmp_path/"new")
    assert result["event_ids"]==[EID,EID+1]
    assert result["source_metadata"]["cached_provider_routes"]==1
    assert result["source_metadata"]["legacy_pair_routes"]==1
    config=json.loads((Path(result["directory"])/"config.json").read_text())
    for match in config["items"]:
        ref=match["reference"]
        assert ref["kind"]=="queue_identity"
        assert not set(ref["event"]).intersection({"status","homeScore","awayScore","startTimestamp","time"})
        assert f"#id:{match['event_id']}" in match["url"]
    assert "OLD_VALUES_MUST_NOT_APPEAR" not in json.dumps(config)
    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==h for p,h in before.items())

def test_invalid_source_season_is_visible_and_not_imported(tmp_path):
    inventory,cache,event,item=source(tmp_path)
    invalid={**item,"stem":"Eredivisie_16_17","season_id":None,"legacy_identity_valid":False,"legacy_event_ids":[]}
    inventory.write_text(json.dumps({"seasons":[item,invalid]}))
    preview=queue_preview(inventory)
    assert preview["source_seasons"]==2 and preview["eligible_matches"]==2
    assert preview["blocked_seasons"][0]["stem"]=="Eredivisie_16_17"
    result=prepare_from_scraper_queue(inventory,cache_root=cache,max_matches=1,base=tmp_path/"new")
    assert result["event_ids"]==[EID]
    assert queue_preview(inventory,season_stems=[item["stem"]])["blocked_seasons"]==[]

def test_csv_hash_and_ids_are_checked_against_original_queue(tmp_path):
    inventory,cache,event,item=source(tmp_path)
    p=Path(item["source"])
    p.write_text(p.read_text().replace(str(EID+1),str(EID+2)))
    with pytest.raises(ValueError,match="changed"): prepare_from_scraper_queue(inventory,cache_root=cache,base=tmp_path/"new")
    item["source_sha256"]=hashlib.sha256(p.read_bytes()).hexdigest()
    inventory.write_text(json.dumps({"seasons":[item]}))
    with pytest.raises(ValueError,match="disagree"): prepare_from_scraper_queue(inventory,cache_root=cache,base=tmp_path/"new")

def test_conflicting_cached_metadata_is_not_silently_used(tmp_path):
    inventory,cache,event,item=source(tmp_path)
    event["homeTeam"]["id"]=999
    with sqlite3.connect(cache/"manifest.sqlite") as db:
        db.execute("UPDATE events SET payload=?",(json.dumps(event),))
    with pytest.raises(ValueError,match="conflicts"): prepare_from_scraper_queue(inventory,cache_root=cache,base=tmp_path/"new")

def test_legacy_route_requires_new_metadata_before_any_parquet_export(tmp_path):
    inventory,cache,event,item=source(tmp_path,cache=False)
    result=prepare_from_scraper_queue(inventory,cache_root=cache,max_matches=1,base=tmp_path/"new")
    with Campaign(result["directory"]) as campaign:
        job=campaign.next_match()
        assert len(job["paths"])==8
        with pytest.raises(ValueError,match="Fresh event metadata"):
            campaign.finish(EID,job["lease"],page_event_id=EID)
        assert not list(Path(result["directory"]).rglob("*.parquet"))
        entry={"startedDateTime":"2026-09-12T02:00:00Z","time":1,
               "request":{"method":"GET","url":f"https://www.sofascore.com/api/v1/event/{EID}"},
               "response":{"status":304,"content":{"text":json.dumps({"event":event})}}}
        campaign.observe(EID,job["lease"],entry)
        finished=campaign.finish(EID,job["lease"],page_event_id=EID)
        report=json.loads(Path(finished["report"]).read_text())
        row=read_tables(report["manifest"])["matches"][0]
        assert row["home_score_current"]==event["homeScore"]["current"] != 999
        assert report["event_metadata_source"]=="captured_network_export"

def test_large_queue_status_is_bounded_without_losing_total_counts(tmp_path):
    inventory,cache,event,item=source(tmp_path)
    refs={eid:{"event":{**deepcopy(event),"id":eid},"observed_at":1700000000} for eid in range(1,202)}
    result=prepare_campaign(list(refs),references=refs,base=tmp_path/"new")
    with Campaign(result["directory"]) as campaign:
        snapshot=campaign.write_status()
        assert snapshot["total_matches"]==201 and snapshot["counts"]=={"pending":201}
        assert len(snapshot["matches"])==100 and snapshot["matches_truncated"]
        assert len(json.dumps(snapshot).encode())<1024*1024
        with campaign.db: campaign.db.execute("UPDATE queue SET state='complete'")
        assert campaign.next_match()["status"]["state"]=="complete"

def test_unknown_season_selection_rejected(tmp_path):
    inventory,cache,event,item=source(tmp_path)
    with pytest.raises(ValueError,match="stems"): queue_preview(inventory,season_stems=["typo"])
