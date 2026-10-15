"""Compatibility and historical-data tests need no browser or torch."""
import json

import pandas as pd
import pytest

from scraping.sofascore.config import Scope
from scraping.sofascore.export import publish_tables, read_tables
from scraping.sofascore.legacy import decode_nested, read_legacy_csv, normalize_legacy, legacy_view, legacy_lineups_view, read_compatibility_views
from utils.data_loading import merge_seasons


def write_season(root,name="Example_24_25",extra=None):
    rows=[]
    for eid,home,away in [(10,1,2),(20,2,1)]:
        rows.append({"match_id":eid,"season_id":500,"season_year":"/".join(name.split("_")[-2:]),"round":eid//10,
                     "home_id":home,"away_id":away,"home_team":f"team{home}","away_team":f"team{away}",
                     "home_score":1,"away_score":0,"home_45_min":0,"away_45_min":0,"home_90_min":1,"away_90_min":0,
                     "home_standing":2,"away_standing":4,"home_shots":str([{"xg":1.0,"xgot":1.0,"shotType":"miss","situation":"regular","time":12}]),
                     "away_shots":"[]","home_heatmap":json.dumps({"playerPoints":[{"x":1,"y":2},{"x":3,"y":4}],"goalkeeperPoints":[]}),
                     "away_heatmap":"",**(extra or {})})
    path=root/(name+".csv")
    pd.DataFrame(rows).to_csv(path,index=False)
    roster={"formation":(4,3,3,0),"roster":{"keeper":{"position":(1,0,0,0),"substitution":(1,0),"statistics":{"rating":7.1}}}}
    pd.DataFrame([{"match_id":20,"home":str(roster),"away":str({**roster,"formation":(4,4,2,0)})},
                  {"match_id":10,"home":str({**roster,"formation":(3,5,2,0)}),"away":str(roster)}]).to_csv(root/(name+"_lineups.csv"),index=False)
    return path


def test_safe_literals_and_json():
    assert decode_nested('{"x": null}')=={"x":None}
    assert decode_nested("{'formation': (4, 4, 2, 0)}")=={"formation":(4,4,2,0)}
    with pytest.raises((ValueError,SyntaxError)):
        decode_nested("__import__('os').getcwd()")


def test_lineups_join_by_id_and_optional(tmp_path):
    path=write_season(tmp_path)
    df,_=read_legacy_csv(path)
    assert list(df.match_id)==[10,20]
    assert df.home_formation.tolist()==[(3,5,2,0),(4,3,3,0)]
    df,_=read_legacy_csv(path,include_lineups=False)
    assert df.home_formation.isna().all() and not df.home_lineups_available.any()
    path.with_name(path.stem+"_lineups.csv").unlink()
    assert read_legacy_csv(path)[0].home_formation.isna().all()


def test_union_schema_missingness_and_no_lineup_requirement(tmp_path):
    write_season(tmp_path,"Example_23_24")
    write_season(tmp_path,"Example_24_25",extra={"home_new_stat":7})
    merged,mapping=merge_seasons(["Example_23_24","Example_24_25"],data_dir=tmp_path,include_lineups=False)
    assert "home_new_stat" in merged and merged.home_new_stat.iloc[:2].isna().all()
    assert merged.home_new_stat.iloc[2:].tolist()==[7,7]
    assert mapping=={"Example":0}
    assert not merged.attrs["numeric_availability"]["home_new_stat"][0]


def test_legacy_parquet_preserves_data_and_labels_untrusted_xg(tmp_path):
    path=write_season(tmp_path)
    original,lineups=read_legacy_csv(path)
    scope=Scope(17,500,(1,2))
    tables,provenance=normalize_legacy(path,scope)
    assert tables["shots"][0]["xg"] is None
    assert tables["shots"][0]["legacy_xg_stored"]==1
    assert all(r["observed_at"] is None for r in tables["shots"])
    manifest=publish_tables(tables,tmp_path/"prepared",scope=scope,provenance=provenance)
    reloaded=read_tables(manifest)
    view=legacy_view(reloaded,restore_legacy_xg=True)
    assert view.home_standing.tolist()==original.home_standing.tolist()
    assert view.home_heatmap.tolist()==original.home_heatmap.tolist()
    assert view.home_formation.tolist()==original.home_formation.tolist()
    assert [s[0]["xg"] for s in view.home_shots]==[1,1]
    assert legacy_view(reloaded).home_shots.iloc[0][0]["xg"] is None
    lineup_view=legacy_lineups_view(reloaded)
    assert lineup_view.home.tolist()==lineups.home.tolist()
    streamed,streamed_lineups=read_compatibility_views(manifest,include_rosters=True,restore_legacy_xg=True)
    assert streamed.home_heatmap.tolist()==original.home_heatmap.tolist()
    assert streamed_lineups.home.tolist()==lineups.home.tolist()
    assert not any("sofascore|" in r["source_key"] for r in reloaded["shots"])
    with pytest.raises(ValueError): normalize_legacy(path,Scope(17,99,(1,2)))


def test_latest_empty_roster_does_not_copy_old_snapshot():
    context={"event_id":10,"source_key":"test","raw_hash":"test"}
    tables={"matches":[{"event_id":10}],"lineup_teams":[
        {**context,"side":"home","observed_at":10,"formation":"4-4-2","roster_status":"present"},
        {**context,"side":"home","observed_at":20,"formation":None,"roster_status":"empty"}],
        "lineup_players":[{**context,"observed_at":10,"side":"home","source_order":0,"player_id":1,"player_slug":"old","position":"G","phase":"pregame"}]}
    assert legacy_lineups_view(tables).home.iloc[0]["roster"]=={}
