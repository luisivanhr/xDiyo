"""Offline migration tests cover fidelity, missingness, publication and resume."""
import json
import socket

import pandas as pd
import pyarrow.parquet as pq
import pytest

from scraping.sofascore.audit import hash_file
from scraping.sofascore.export import read_tables
from scraping.sofascore.legacy import load_season
from scraping.sofascore.migrate_legacy import migrate
from scraping.sofascore.validate_legacy_migration import validate
from utils.data_loading import merge_seasons


def fixture(root):
    source=root/"source"; source.mkdir()
    lineups=root/"lineups"; lineups.mkdir()
    output=root/"out"
    rows=[]
    for eid,home,away,hp,ap in ((10,1,2,1,0),(20,2,1,0,1)):
        rows.append({"match_id":eid,"season_id":500,"season_year":"24/25","round":eid//10,
            "home_id":home,"away_id":away,"home_team":f"team{home}","away_team":f"team{away}",
            "home_score":2,"away_score":1,"home_45_min":0,"away_45_min":1,"home_90_min":2,"away_90_min":0,
            "home_standing":2 if eid==10 else 4,"away_standing":4 if eid==10 else 2,
            "home_got_promoted":hp,"away_got_promoted":ap,"home_got_demoted":0,"away_got_demoted":0,
            "home_total_expectedgoals":1.7,"away_total_expectedgoals":None,
            "home_45_min_shotsontarget":0,"home_90_min_passes_value":12,"home_90_min_passes_all":None,
            "away_total_passes_value":13,"away_total_passes_all":20,
            "home_shots":str([{"xg":1.0,"xgot":0.0,"shotType":"miss","time":12},
                              {"xg":0.12,"xgot":0.2,"shotType":"save","time":50}]),
            "away_shots":"[]","home_heatmap":json.dumps({"playerPoints":[{"x":1,"y":2}],"goalkeeperPoints":[]}),
            "away_heatmap":""})
    pd.DataFrame(rows).to_csv(source/"Premier_League_24_25.csv",index=False)
    roster={"formation":(4,3,3,0),"roster":{"keeper":{"position":(1,0,0,0),"substitution":(1,0),
            "statistics":{"rating":7.1,"ratingVersions":{"original":7.2}}},
            "bench":{"position":(0,1,0,0),"substitution":(0,1),"statistics":{}}}}
    # Companion files may have a different row order and a missing team snapshot.
    pd.DataFrame([{"match_id":20,"home":str(roster),"away":""},
                  {"match_id":10,"home":str({**roster,"formation":(3,5,2,0)}),"away":str(roster)}]
                 ).to_csv(lineups/"Premier_League_24_25_lineups.csv",index=False)
    return source,lineups,output


def test_offline_conversion_and_idempotent_resume(tmp_path,monkeypatch):
    source,lineups,output=fixture(tmp_path)
    def forbidden(*args,**kwargs):
        raise AssertionError("Offline conversion must not use a browser or network")
    monkeypatch.setattr(socket,"create_connection",forbidden)
    monkeypatch.setattr("scraping.sofascore.campaign.BrowserClient",forbidden)
    before={p:hash_file(p) for folder in (source,lineups) for p in folder.glob("*.csv")}
    result=migrate(source,lineups,output)
    assert result["status_counts"]=={"complete":1}, result
    assert result["api_requests"]==0 and result["source_files_unchanged"]
    item=result["seasons"][0]["output"]
    assert item["reconciliation"]["validation"]["passed"]
    tables=read_tables(output/item["manifest"])
    matches=tables["matches"]
    assert all(m["kickoff_utc"] is None and m["home_score_period2"] is None and m["status"]=="unknown" for m in matches)
    assert all(p["player_id"] is None for p in tables["lineup_players"])
    assert tables["shots"][0]["xg"] is None and tables["shots"][0]["legacy_xg_stored"]==1
    assert tables["shots"][1]["xg"]==0.12
    assert all(r["shot_id"] is None for r in tables["shots"])
    stat=next(r for r in tables["statistics"] if r["side"]=="home" and r["key"]=="passes")
    assert stat["value"]==12 and stat["total"] is None and stat["period"]=="2ND"
    assert stat["group_name"] is None and stat["observed_at"] is None
    assert [r["got_promoted"] for r in tables["team_seasons"]]==[True,False]
    assert all(r["membership_status"]=="legacy_unverified" for r in tables["team_seasons"])
    assert not any(tables.get(t) for t in ("incidents","availability","timing","comments","passing_edges"))
    path=output/item["season_file"]
    nested=pq.ParquetFile(path).read().to_pylist()
    assert len(nested[0]["lineup_players"])==4 and "team_seasons" not in nested[0]
    frame=load_season(path)
    assert "home_90_min_passes" not in frame
    assert frame.home_90_min.tolist()==[2,2]
    assert frame.home_formation.tolist()==[(3,5,2,0),(4,3,3,0)]
    assert frame.home_got_promoted.tolist()==[1,0]
    normalized,_=merge_seasons([path],data_dir=output)
    assert normalized.home_standing.tolist()==[0.5,1.0]
    assert normalized.away_standing.tolist()==[1.0,0.5]
    assert normalized.normalized_round.tolist()==[0.5,1.0]
    manifest_time=(output/"Premier_League_24_25.manifest.json").stat().st_mtime_ns
    monkeypatch.setattr("scraping.sofascore.migrate_legacy.normalize_legacy",forbidden)
    resumed=migrate(source,lineups,output)
    assert resumed["seasons"][0]["resumed"] is True
    assert manifest_time==(output/"Premier_League_24_25.manifest.json").stat().st_mtime_ns
    assert before=={p:hash_file(p) for p in before}
    audit=validate(output)
    assert audit["passed"] and audit["matches"]==2 and audit["source_files_reverified"]==2
    assert audit["quality_and_storage"]["shots_with_ambiguous_default_xg"]==2
    assert audit["quality_and_storage"]["match_sides_with_stored_aggregate_xg"]==2
    assert audit["quality_and_storage"]["matches_with_both_sides_stored_aggregate_xg"]==0
    # Never silently trust a damaged published output on resume.
    path.write_bytes(path.read_bytes()+b"damaged")
    assert migrate(source,lineups,output)["status_counts"]=={"failed":1}


def test_quarantine_bad_year_and_optional_lineups(tmp_path):
    source,lineups,output=fixture(tmp_path)
    original=source/"Premier_League_24_25.csv"
    bad=source/"Eredivisie_16_17.csv"
    bad.write_bytes(original.read_bytes())
    report=migrate(source,lineups,output,include_lineups=False)
    assert report["status_counts"]=={"quarantined":1,"complete":1}
    assert not (output/"Eredivisie_16_17.parquet").exists()
    item=report["seasons"][1]["output"]
    tables=read_tables(output/item["manifest"])
    assert not tables.get("lineup_players")
    assert all(r["status"]=="not_requested" for r in tables["coverage"] if r["component"]=="lineups")
    assert migrate(source,lineups,output)["status_counts"]=={"quarantined":1,"failed":1}


def test_duplicate_valid_seasons_and_conflicting_flags_are_rejected(tmp_path):
    source,lineups,output=fixture(tmp_path)
    original=source/"Premier_League_24_25.csv"
    duplicate=source/"La_Liga_24_25.csv"
    duplicate.write_bytes(original.read_bytes())
    with pytest.raises(ValueError,match="Duplicate match IDs"):
        migrate(source,lineups,output)
    duplicate.unlink()
    frame=pd.read_csv(original)
    frame.loc[1,"away_got_promoted"]=0
    frame.to_csv(original,index=False)
    report=migrate(source,lineups,output)
    assert report["status_counts"]=={"failed":1}
    assert "Conflicting legacy movement flags" in report["seasons"][0]["error"]


def test_legacy_error_cell_is_retained_as_unavailable(tmp_path):
    source,lineups,output=fixture(tmp_path)
    path=source/"Premier_League_24_25.csv"
    frame=pd.read_csv(path)
    frame.loc[0,"home_heatmap"]=str({"error":"Not Found"})
    frame.to_csv(path,index=False)
    report=migrate(source,lineups,output)
    assert report["status_counts"]=={"complete":1}, report
    item=report["seasons"][0]["output"]
    tables=read_tables(output/item["manifest"])
    evidence=next(r for r in tables["legacy_fields"] if r["event_id"]==10 and r["key"]=="home_heatmap")
    assert json.loads(evidence["value_json"])=={"error":"Not Found"}
    assert not any(r["event_id"]==10 for r in tables["heatmap_points"])
    view=load_season(output/item["season_file"])
    assert not view.home_heatmap_available.iloc[0] and view.home_heatmap.iloc[0] is None
