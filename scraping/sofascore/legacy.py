"""Safe import and compatibility views for existing prepared CSVs.

These transforms never create cache entries purporting to be API responses.
Legacy provenance is attached to normalized rows and a source-file manifest.
"""
from __future__ import annotations

import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import pandas as pd

from .config import Profile
from .export import read_tables
from .models import Job
from .parsers import encoded, number, parse_component, parse_match

NESTED = ("home_shots","away_shots","home_heatmap","away_heatmap")


def decode_nested(value):
    if isinstance(value,(dict,list,tuple)):
        return value
    if value is None or not isinstance(value,str):
        return None if value is None or pd.isna(value) else value
    if not value.strip():
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)


def formation_tuple(value):
    if isinstance(value,(tuple,list)):
        return tuple(value)
    if isinstance(value,str):
        try:
            parts=tuple(int(x) for x in value.split("-"))
            return parts + (0,)*max(0,4-len(parts))
        except ValueError:
            return None
    return None


def read_legacy_csv(path, *, include_lineups=True, lineup_path=None):
    path=Path(path)
    df=pd.read_csv(path,converters={c:decode_nested for c in NESTED})
    df=df.copy()  # consolidate pandas blocks after mixed nested converters
    if "match_id" not in df or df.match_id.isna().any() or df.match_id.duplicated().any():
        raise ValueError("Legacy match_id must be present, non-null and unique")
    lineups=None
    lineup_path=Path(lineup_path) if lineup_path is not None else path.with_name(path.stem+"_lineups.csv")
    if include_lineups and lineup_path.exists():
        lineups=pd.read_csv(lineup_path,converters={"home":decode_nested,"away":decode_nested})
        if "match_id" not in lineups or lineups.match_id.isna().any() or lineups.match_id.duplicated().any():
            raise ValueError("Legacy lineup match_id must be non-null and unique")
        orphans=set(lineups.match_id)-set(df.match_id)
        if orphans:
            raise ValueError(f"Orphaned lineup IDs: {sorted(orphans)[:10]}")
        lineups=lineups.set_index("match_id").reindex(df.match_id).reset_index()
    for side in ("home","away"):
        data=lineups[side] if lineups is not None and side in lineups else pd.Series([None]*len(df))
        df[side+"_formation"]=data.map(lambda value: formation_tuple(value.get("formation")) if isinstance(value,dict) else None).to_list()
        df[side+"_lineups_available"]=data.map(lambda value:isinstance(value,dict) and "roster" in value).to_numpy()
        for component in ("shots","heatmap"):
            col=side+"_"+component
            if col not in df:
                df[col]=pd.Series([None]*len(df),dtype=object)
            df[col+"_available"]=df[col].map(lambda value:isinstance(value,list) if component=="shots" else
                isinstance(value,dict) and "playerPoints" in value and "goalkeeperPoints" in value)
    return df,lineups


def normalize_legacy(path, scope, *, include_lineups=True, lineup_path=None, normalize_scalars=False):
    path=Path(path)
    df,lineups=read_legacy_csv(path,include_lineups=include_lineups,lineup_path=lineup_path)
    if "season_id" not in df or set(df.season_id.dropna().astype(int))!={scope.season_id}:
        raise ValueError("Legacy file season IDs do not match the configured scope")
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    lineage={"kind":"legacy_prepared","match_file":str(path.resolve()),"match_sha256":digest,
             "imported_rounds":sorted(int(r) for r in df["round"].dropna().unique()),
             "limitations":["No authentic raw responses or original fetched-at timestamps",
                            "Legacy xG=1 and xGOT=0/1 may be parser defaults; original values retained separately"]}
    lineup_lookup={}
    if lineups is not None:
        lp=Path(lineup_path) if lineup_path is not None else path.with_name(path.stem+"_lineups.csv")
        lineage.update({"lineup_file":str(lp.resolve()),"lineup_sha256":hashlib.sha256(lp.read_bytes()).hexdigest()})
        lineup_lookup=lineups.set_index("match_id").to_dict("index")
    tables=defaultdict(list)
    helper_columns={side+suffix for side in ("home","away") for suffix in
                    ("_formation","_lineups_available","_shots_available","_heatmap_available")}
    for raw in df.to_dict("records"):
        eid=int(raw["match_id"])
        context={"event_id":eid,"source_key":"legacy_prepared:"+path.name,"raw_hash":digest,"observed_at":None}
        event={"id":eid,"customId":raw.get("custom_id"),"status":{"type":"unknown"},
               "isAwarded":raw.get("is_awarded"),  # A reconstructed row supplies no evidence when absent.
               "season":{"id":scope.season_id,"year":raw.get("season_year")},
               "roundInfo":{"round":int(raw["round"]) if pd.notna(raw.get("round")) else None},
               "tournament":{"uniqueTournament":{"id":scope.league_id}}}
        for side in ("home","away"):
            event[side+"Team"]={"id":int(raw[side+"_id"]),"slug":raw.get(side+"_team")}
            event[side+"Score"]={"display":number(raw.get(side+"_score")),"period1":number(raw.get(side+"_45_min"))}
        # The old "90_min" score was derived from display minus first-half score;
        # do not assert that it was an observed period2 score for extra-time games.
        tables["matches"].append(parse_match(event))
        if normalize_scalars:
            append_legacy_scalars(tables, raw, event, context)
        for key,value in raw.items():
            if key in NESTED or key in helper_columns or key.startswith("Unnamed:") or key=="index":
                continue
            if pd.isna(value): value=None
            tables["legacy_fields"].append({**context,"key":key,"value_json":encoded(value)})
        for side in ("home","away"):
            team=event[side+"Team"]["id"]
            for component in ("shots","heatmap"):
                obj=raw[side+"_"+component]
                available=raw[side+"_"+component+"_available"]
                tables["coverage"].append({**context,"component":component,"side":side,
                    "status":"legacy_stored" if available and obj else "legacy_empty_unknown_reason" if available else "legacy_unavailable"})
                if not available:
                    if normalize_scalars and obj is not None:
                        # Some legacy cells retained an error object instead of
                        # points. Preserve it without claiming an empty success.
                        tables["legacy_fields"].append({**context,"key":side+"_"+component,"value_json":encoded(obj)})
                    continue
                if component=="shots":
                    transformed=[{**s,"isHome":side=="home"} for s in obj]
                    job=Job(f"/event/{eid}/shotmap","shotmap",eid)
                    parsed=parse_component(job,{"shotmap":transformed},event,Profile(),observed_at=None,raw_hash=digest)
                    for row,original in zip(parsed.get("shots",[]),obj):
                        row.update(context)
                        row["source_item_json"]=encoded(original)
                        row["legacy_xg_stored"],row["legacy_xgot_stored"]=row["xg"],row["xgot"]
                        row["xg_quality"]="legacy_unverified_one" if row["xg"]==1 else "legacy_stored" if row["xg"] is not None else "unavailable"
                        row["xgot_quality"]="legacy_unverified_default" if row["xgot"] in {0,1} else "legacy_stored" if row["xgot"] is not None else "unavailable"
                        if row["xg"]==1: row["xg"]=None
                        if row["xgot"] in {0,1}: row["xgot"]=None
                        tables["shots"].append(row)
                else:
                    for name,kind in (("playerPoints","player"),("goalkeeperPoints","goalkeeper")):
                        for order,point in enumerate(obj[name]):
                            tables["heatmap_points"].append({**context,"team_id":team,"kind":kind,"point_order":order,
                                "x":number(point.get("x")),"y":number(point.get("y")),"weight":number(point.get("weight")),
                                "source_item_json":encoded(point)})
            data=lineup_lookup.get(eid,{}).get(side)
            if normalize_scalars:
                tables["coverage"].append({**context,"component":"lineups","side":side,
                    "status":"not_requested" if not include_lineups else "legacy_stored" if isinstance(data,dict) else "legacy_unavailable"})
            if not isinstance(data,dict): continue
            lcontext={**context,"source_key":"legacy_prepared:"+Path(lineage["lineup_file"]).name,"raw_hash":lineage["lineup_sha256"]}
            base={"side":side,"team_id":team,"phase":"legacy_unknown"}
            formation=data.get("formation")
            tables["lineup_teams"].append({**lcontext,**base,"formation":formation if isinstance(formation,str) else "-".join(map(str,formation)) if formation else None,
                                          "confirmed":None,"roster_status":"legacy_stored" if data.get("roster") else "legacy_empty"})
            for index,(slug,player) in enumerate(data.get("roster",{}).items()):
                pos=player.get("position")
                position=next((p for p,v in zip("GDMF",pos) if v),None) if isinstance(pos,(list,tuple)) else pos
                substitution=player.get("substitution")
                tables["lineup_players"].append({**lcontext,**base,"source_order":index,"player_id":None,
                    "player_slug":slug,"position":position,
                    "substitute":bool(substitution[1]) if isinstance(substitution,(list,tuple)) and len(substitution)==2 else None,
                    "statistics_status":"legacy_stored" if player.get("statistics") else "legacy_empty_unknown_reason","source_item_json":encoded(player)})
                for key,value in player.get("statistics",{}).items():
                    tables["player_statistics"].append({**lcontext,**base,"source_order":index,"player_id":None,
                        "key":key,"value":number(value),"value_json":encoded(value)})
    if normalize_scalars:
        tables["team_seasons"] = legacy_team_seasons(df, scope, path.name, digest)
        lineage["limitations"].extend([
            "No saved kickoff dates, player/shot IDs, shot coordinates, injury lists, or original fetch timestamps",
            "Statistics retain legacy lowercase keys; source groups, key casing, and formatted displays were discarded",
            "Period labels follow the old scraper's positional ALL/1ST/2ND mapping and cannot be independently revalidated",
            "Stored standings may include the old mid-table fallback for unavailable pregame-form responses",
            "Stored promotion/relegation flags are preserved as legacy reports, not newly verified movement evidence",
            "Derived second-half score stays in legacy_fields; current score/status/actual period2 are not inferred",
        ])
    return dict(tables),lineage


def append_legacy_scalars(tables, raw, event, context):
    """Map flattened cells without inventing the API structure discarded by v1."""
    periods = {"total":"ALL", "45_min":"1ST", "90_min":"2ND"}
    grouped = {}
    for column, value in raw.items():
        match = re.fullmatch(r"(home|away)_(total|45_min|90_min)_(.+)", column)
        if not match:
            continue
        side, period, key = match.groups()
        part = "value"
        if key.endswith("_value") and column[:-6]+"_all" in raw:
            key = key[:-6]
        elif key.endswith("_all") and column[:-4]+"_value" in raw:
            key, part = key[:-4], "total"
        item = grouped.setdefault((side,period,key), {"cells":{}, "value":None, "total":None})
        # Numeric conversion must not silently discard a new textual statistic.
        if value is not None and not pd.isna(value) and number(value) is None:
            raise ValueError(f"Non-numeric legacy statistic: {column}")
        item[part] = number(value)
        item["cells"][column] = None if pd.isna(value) else value
    for order, ((side,period,key), item) in enumerate(grouped.items()):
        tables["statistics"].append({**context, "side":side,"team_id":event[side+"Team"]["id"],
            "period":periods[period],"key":key,"group_name":None,"name":None,"display":None,
            "group_order":None,"stat_order":order,"value":item["value"],"total":item["total"],
            "value_type":"legacy_numeric","statistics_type":None,
            "source_item_json":encoded({"legacy_columns":item["cells"]})})
    for side in ("home","away"):
        position = number(raw.get(side+"_standing"))
        tables["pregame"].append({**context,"side":side,"team_id":event[side+"Team"]["id"],
            "position":position,"value_json":encoded({"legacy_column":side+"_standing",
                "stored_value":position,"quality":"legacy_stored_possible_midtable_fallback"})})
        tables["coverage"].extend([
            {**context,"component":"pregame","side":side,"status":"legacy_stored_unverified" if position is not None else "legacy_unavailable"},
            {**context,"component":"statistics","side":side,"status":"legacy_stored" if any(
                item["value"] is not None or item["total"] is not None for (s,_,_),item in grouped.items() if s==side) else "legacy_unavailable"},
            {**context,"component":"availability","side":side,"status":"not_retained_in_legacy"},
        ])
    for component in ("incidents","goal_actions","comments","timing"):
        tables["coverage"].append({**context,"component":component,"status":"not_requested"})


def legacy_team_seasons(df, scope, filename, digest):
    """Preserve consistent stored flags with explicit, unverified provenance."""
    observations = defaultdict(set)
    for side in ("home","away"):
        columns = [side+"_id", side+"_got_promoted", side+"_got_demoted"]
        if not set(columns).issubset(df):
            continue
        for tid,promoted,demoted in df[columns].itertuples(index=False,name=None):
            if promoted not in (0,1) or demoted not in (0,1) or (promoted and demoted):
                raise ValueError(f"Invalid legacy movement flags for team {tid}")
            observations[int(tid)].add((bool(promoted),bool(demoted)))
    rows = []
    for tid, values in sorted(observations.items()):
        if len(values) != 1:
            raise ValueError(f"Conflicting legacy movement flags for team {tid}")
        promoted,demoted = next(iter(values))
        rows.append({"competition_id":scope.league_id,"season_id":scope.season_id,"team_id":tid,
            "got_promoted":promoted,"got_demoted":demoted,
            "movement":"promoted" if promoted else "relegated" if demoted else "retained",
            "evidence":"legacy_reported_flags:sha256:"+digest,"source":filename,
            "membership_status":"legacy_unverified","derivation_version":"legacy_import_1"})
    return rows


def latest_snapshot_rows(rows, snapshot_context=()):
    latest={}
    for row in [*rows,*snapshot_context]:
        key=(row["event_id"],row.get("source_key"))
        latest[key]=max(latest.get(key,float("-inf")),row.get("observed_at") or float("-inf"))
    return [row for row in rows if (row.get("observed_at") or float("-inf"))==latest[(row["event_id"],row.get("source_key"))]]


def lineup_snapshot_context(tables):
    return [*tables.get("lineup_teams",[]),*(r for r in tables.get("coverage",[]) if r.get("component")=="lineups")]


def legacy_view(tables, *, include_lineups=True, restore_legacy_xg=False):
    """Materialize the match-wide view without conflating null with zero."""
    rows={m["event_id"]:{"match_id":m["event_id"],"custom_id":m.get("custom_id"),"season_id":m.get("season_id"),
        "competition_id":m.get("competition_id"),"season_year":m.get("season_year"),"round":m.get("round"),
        "kickoff_utc":m.get("kickoff_utc"),"event_metadata_observed_at":m.get("event_metadata_observed_at"),"status":m.get("status"),
        "is_awarded":m.get("is_awarded"),
        **{side+suffix:m.get(side+field) for side in ("home","away") for suffix,field in
           (("_id","_id"),("_team","_slug"),("_score","_score_display"),("_45_min","_score_period1"),("_90_min","_score_period2"))}}
        for m in tables.get("matches",[])}
    for field in tables.get("legacy_fields",[]):
        rows[field["event_id"]][field["key"]]=json.loads(field["value_json"]) if field["value_json"] is not None else None
    for row in rows.values():
        for side in ("home","away"):
            row.update({side+"_shots":[],side+"_heatmap":None,side+"_formation":None,
                        side+"_lineups_available":False,side+"_shots_available":False,side+"_heatmap_available":False})
            row.setdefault(side+"_standing",None)
    for item in tables.get("coverage",[]):
        row=rows[item["event_id"]]
        component=item["component"]
        if component not in {"shots","shotmap","heatmap"}: continue
        if item["status"] not in {"present","empty","cached_last_good","legacy_stored","legacy_empty_unknown_reason"}: continue
        side=item.get("side")
        if component=="heatmap" and side not in {"home","away"}:
            side=next((s for s in ("home","away") if str(row[s+"_id"])==side),None)
        sides=[side] if side in {"home","away"} else ["home","away"] if component in {"shots","shotmap"} else []
        for side in sides:
            key=side+"_"+("shots" if component in {"shots","shotmap"} else "heatmap")
            row[key+"_available"]=True
            if component=="heatmap": row[key]={"playerPoints":[],"goalkeeperPoints":[]}
    for r in tables.get("pregame",[]):
        rows[r["event_id"]][r["side"]+"_standing"]=r["position"]
    periods={"ALL":"total","1ST":"45_min","2ND":"90_min"}
    for stat in tables.get("statistics",[]):
        if stat.get("value_type") == "legacy_numeric":
            columns=json.loads(stat["source_item_json"])["legacy_columns"]
            for column in columns:
                rows[stat["event_id"]][column]=stat.get("total") if column.endswith("_all") and len(columns)==2 else stat.get("value")
            continue
        prefix=stat["side"]+"_"+periods.get(stat["period"],stat["period"].lower())+"_"+stat["key"].lower().replace(" ","_")
        row=rows[stat["event_id"]]
        if stat.get("total") is not None:
            row[prefix+"_value"],row[prefix+"_all"]=stat["value"],stat["total"]
        else: row[prefix]=stat["value"]
    for shot in sorted(tables.get("shots",[]),key=lambda s:s["source_order"]):
        if shot.get("side") not in {"home","away"}: continue
        original=json.loads(shot["source_item_json"]) if shot.get("source_item_json") else {}
        original.update({"xg":shot.get("xg"),"xgot":shot.get("xgot"),"xg_quality":shot.get("xg_quality"),
                         "shotType":shot.get("shot_type"),"goalType":shot.get("goal_type"),
                         "time":shot.get("time"),"situation":shot.get("situation")})
        if restore_legacy_xg and shot.get("xg_quality","").startswith("legacy"):
            original.update({"xg":shot.get("legacy_xg_stored"),"xgot":shot.get("legacy_xgot_stored")})
        rows[shot["event_id"]][shot["side"]+"_shots"].append(original)
    for point in sorted(tables.get("heatmap_points",[]),key=lambda p:p["point_order"]):
        row=rows[point["event_id"]]
        side="home" if point["team_id"]==row["home_id"] else "away" if point["team_id"]==row["away_id"] else None
        if side is None: raise ValueError("Heatmap team does not belong to event")
        key=side+"_heatmap"
        if row[key] is None: row[key]={"playerPoints":[],"goalkeeperPoints":[]}
        row[key]["playerPoints" if point["kind"]=="player" else "goalkeeperPoints"].append(
            json.loads(point["source_item_json"]) if point.get("source_item_json") else {"x":point["x"],"y":point["y"]})
    if include_lineups:
        for team in latest_snapshot_rows(tables.get("lineup_teams",[]),lineup_snapshot_context(tables)):
            row=rows[team["event_id"]]
            row[team["side"]+"_formation"]=formation_tuple(team.get("formation"))
            row[team["side"]+"_lineups_available"]=team.get("roster_status") in {"present","empty","legacy_stored","legacy_empty"}
    return pd.DataFrame(rows.values())


def legacy_lineups_view(tables):
    rows={m["event_id"]:{"match_id":m["event_id"],"home":None,"away":None} for m in tables.get("matches",[])}
    snapshot_context=lineup_snapshot_context(tables)
    for team in latest_snapshot_rows(tables.get("lineup_teams",[]),snapshot_context):
        rows[team["event_id"]][team["side"]]={"formation":formation_tuple(team.get("formation")),"roster":{}}
    stats=defaultdict(dict)
    for item in latest_snapshot_rows(tables.get("player_statistics",[]),snapshot_context):
        stats[(item["event_id"],item["side"],item["source_order"])][item["key"]]=json.loads(item["value_json"]) if item["value_json"] else None
    for player in latest_snapshot_rows(tables.get("lineup_players",[]),snapshot_context):
        team=rows[player["event_id"]][player["side"]]
        if team is None: raise ValueError("Roster has no lineup team snapshot")
        key=player.get("player_slug") or str(player.get("player_id") or f"local_{player['source_order']}")
        if key in team["roster"]: raise ValueError("Duplicate player alias in compatibility roster")
        raw=json.loads(player["source_item_json"]) if player.get("source_item_json") else {}
        if player.get("phase")=="legacy_unknown":
            raw["position"]=tuple(raw["position"]) if isinstance(raw.get("position"),list) else raw.get("position")
            raw["substitution"]=tuple(raw["substitution"]) if isinstance(raw.get("substitution"),list) else raw.get("substitution")
            team["roster"][key]=raw
        else:
            team["roster"][key]={"player_id":player.get("player_id"),
                "position":tuple(int(player.get("position")==pos) for pos in "GDMF"),
                "substitution":(0,1) if player.get("substitute") is True else (1,0) if player.get("substitute") is False else None,
                "statistics":raw.get("statistics",stats[(player["event_id"],player["side"],player["source_order"])])}
    return pd.DataFrame(rows.values())


def read_compatibility_views(path, *, include_lineups=True, include_rosters=False, restore_legacy_xg=False):
    """Stream large point tables directly into the requested nested match view.

    The normalized Arrow tables remain the primary interface. This adapter
    avoids making hundreds of thousands of unnecessary provenance dictionaries
    solely to reconstruct the legacy heatmap cells.
    """
    import pyarrow.parquet as pq
    from .audit import hash_file
    path=Path(path)
    manifest=json.loads(path.read_text(encoding="utf-8"))
    if "tables" not in manifest:
        path=path.parent/manifest["manifest"]
        manifest=json.loads(path.read_text(encoding="utf-8"))
    names={"matches","statistics","pregame","legacy_fields","shots","coverage"}
    if include_lineups:
        names.add("lineup_teams")
    if include_rosters:
        names.add("lineup_players")
    tables=read_tables(path,names=names)
    df=legacy_view(tables,include_lineups=include_lineups,restore_legacy_xg=restore_legacy_xg)
    indices={eid:i for i,eid in enumerate(df.match_id)}
    teams={r.match_id:{r.home_id:"home",r.away_id:"away"} for r in df[["match_id","home_id","away_id"]].itertuples(index=False)}
    heatmaps={}
    info=manifest["tables"].get("heatmap_points")
    if info:
        table_path=path.parent/info["file"]
        if hash_file(table_path)!=info["sha256"]:
            raise ValueError("Heatmap table hash mismatch")
        previous_orders={}
        for batch in pq.ParquetFile(table_path).iter_batches(batch_size=8192,columns=["event_id","team_id","kind","point_order","x","y","weight","source_item_json"]):
            columns=batch.to_pydict()
            for eid,tid,kind,order,x,y,weight,source in zip(*(columns[k] for k in columns)):
                side=teams.get(eid,{}).get(tid)
                if side is None: raise ValueError("Orphan heatmap event/team")
                key=(eid,side,kind)
                if order<=previous_orders.get(key,-1): raise ValueError("Heatmap source order is not strictly increasing")
                previous_orders[key]=order
                target=heatmaps.setdefault((eid,side),{"playerPoints":[],"goalkeeperPoints":[]})
                point=json.loads(source) if source else {"x":x,"y":y}
                if source is None and weight is not None: point["weight"]=weight
                target["playerPoints" if kind=="player" else "goalkeeperPoints"].append(point)
    for (eid,side),points in heatmaps.items():
        df.at[indices[eid],side+"_heatmap"]=points
    return df,legacy_lineups_view(tables) if include_rosters else None


def load_season(path, *, include_lineups=True):
    path=Path(path)
    if path.suffix.lower()==".parquet":
        import pyarrow.parquet as pq
        metadata=pq.ParquetFile(path).schema_arrow.metadata or {}
        manifest=metadata.get(b"xdiyo_canonical_manifest")
        if manifest is None:
            raise ValueError("Expected an xDiyo season Parquet with its canonical manifest")
        path=path.parent/manifest.decode("utf-8")
    if path.suffix.lower()==".csv":
        frame = read_legacy_csv(path,include_lineups=include_lineups)[0]
        import re
        label = re.search(r"_(\d{2,4})_(\d{2,4})$", path.stem)
        if label and "season_year" in frame:
            expected = tuple(int(v)+2000 if len(v)==2 else int(v) for v in label.groups())
            for stored in frame.season_year.dropna().unique():
                found = re.fullmatch(r"(\d{2,4})/(\d{2,4})", str(stored))
                if found and tuple(int(v)+2000 if len(v)==2 else int(v) for v in found.groups()) != expected:
                    raise ValueError(f"Filename and stored season identity conflict: {path.name}")
        # The low-level reader remains a faithful forensic reader. The model
        # loader applies the same legacy trust policy as the Parquet importer.
        for side in ("home", "away"):
            key = side+"_shots"
            if key not in frame:
                continue
            def trustworthy_shots(shots):
                if not isinstance(shots, list):
                    return shots
                result = []
                for shot in shots:
                    item = dict(shot)
                    if item.get("xg") == 1:
                        item.update(legacy_xg_stored=1, xg=None, xg_quality="legacy_unverified_one")
                    result.append(item)
                return result
            frame[key] = frame[key].map(trustworthy_shots)
        return frame
    return read_compatibility_views(path,include_lineups=include_lineups)[0]
