"""Offline, resumable conversion of legacy seasons into schema-2 Parquet.

CSV files and the live collection store are read-only. Nothing opens a browser.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import gc
import json
from pathlib import Path
import re
import time

import pandas as pd
import pyarrow.parquet as pq

from .audit import hash_file
from .campaign import publish_season
from .config import Profile, Scope
from .legacy import NESTED, legacy_lineups_view, normalize_legacy, read_legacy_csv
from .store import atomic_write, json_bytes, season_year_pair

ROOT = Path(__file__).resolve().parents[2]
VERSION = "1"
PROFILE = Profile(include_lineups=True, include_injuries=False, include_comments=False,
                  include_goal_actions=False, duration_mode="off")
IDENTITY = {"match_id", "season_id", "season_year", "round", "home_id", "away_id"}


def inspect_source(path, lineup_dir, registry):
    match = re.fullmatch(r"(.+)_(\d{2})_(\d{2})", path.stem)
    if not match or match[1] not in registry:
        raise ValueError(f"Unknown season filename: {path.name}")
    frame = pd.read_csv(path, usecols=lambda c: c in IDENTITY)
    if not IDENTITY.issubset(frame) or frame.empty or frame.isna().any().any() or frame.match_id.duplicated().any():
        raise ValueError(f"Invalid match/season identity: {path.name}")
    for key in IDENTITY-{"season_year"}:
        if not frame[key].eq(frame[key].astype("int64")).all():
            raise ValueError(f"Non-integer {key}: {path.name}")
    seasons = frame.season_id.unique()
    years = (2000+int(match[2]), 2000+int(match[3]))
    stored = {season_year_pair(value) for value in frame.season_year.unique()}
    lineup = Path(lineup_dir)/(path.stem+"_lineups.csv")
    result = {"stem":path.stem,"source":str(path.resolve()),"source_sha256":hash_file(path),
              "lineup_source":str(lineup.resolve()) if lineup.exists() else None,
              "lineup_sha256":hash_file(lineup) if lineup.exists() else None,
              "matches":len(frame),"event_ids":sorted(int(v) for v in frame.match_id),
              "filename_years":years,"stored_years":sorted(stored,key=str)}
    if len(seasons)!=1 or stored != {years}:
        result.update(status="quarantined",reason="filename_and_stored_season_identity_conflict")
        return result
    scope = Scope(registry[match[1]]["competition_id"],int(seasons[0]),
                  tuple(int(r) for r in frame["round"].unique()),match[1],*years)
    return {**result,"status":"ready","scope":asdict(scope)}


def equal_value(left, right):
    if left is None or (not isinstance(left,(list,dict,tuple)) and pd.isna(left)):
        return right is None or (not isinstance(right,(list,dict,tuple)) and pd.isna(right))
    return left == right


def verify_mapping(tables, source, lineup_path, *, include_lineups):
    """Compare every saved scalar, shot, point, roster, and mapped statistic."""
    original,lineups = read_legacy_csv(source,include_lineups=include_lineups,lineup_path=lineup_path)
    matches = {row["event_id"]:row for row in tables["matches"]}
    assert set(matches)==set(original.match_id), "Match ID mismatch"
    fields = {(row["event_id"],row["key"]):json.loads(row["value_json"]) if row["value_json"] else None
              for row in tables["legacy_fields"]}
    shots,points,stats = defaultdict(list),defaultdict(list),{}
    for row in tables.get("shots",[]):
        shots[(row["event_id"],row["side"])].append(json.loads(row["source_item_json"]))
    for row in tables.get("heatmap_points",[]):
        points[(row["event_id"],row["team_id"],row["kind"])].append(json.loads(row["source_item_json"]))
    for row in tables.get("statistics",[]):
        columns = json.loads(row["source_item_json"])["legacy_columns"]
        for column in columns:
            stats[(row["event_id"],column)] = row["total"] if column.endswith("_all") and len(columns)==2 else row["value"]
    pregame={(row["event_id"],row["side"]):row["position"] for row in tables["pregame"]}
    checked = Counter()
    helper={s+x for s in ("home","away") for x in ("_formation","_lineups_available","_shots_available","_heatmap_available")}
    for raw in original.to_dict("records"):
        eid=raw["match_id"]
        for key,value in raw.items():
            if key in NESTED or key in helper or key=="index" or key.startswith("Unnamed:"):
                continue
            assert equal_value(value,fields[(eid,key)]), f"Scalar mismatch {eid}/{key}"
            checked["scalar_cells"]+=1
            if re.fullmatch(r"(home|away)_(total|45_min|90_min)_.+",key):
                assert equal_value(value,stats[(eid,key)]), f"Statistic mismatch {eid}/{key}"
                checked["statistic_cells"]+=1
        for side in ("home","away"):
            assert equal_value(raw.get(side+"_standing"),pregame[(eid,side)])
            assert matches[eid][side+"_id"]==raw[side+"_id"]
            assert equal_value(matches[eid][side+"_score_display"],raw[side+"_score"])
            assert equal_value(matches[eid][side+"_score_period1"],raw[side+"_45_min"])
            if isinstance(raw[side+"_shots"],list):
                assert shots[(eid,side)]==raw[side+"_shots"], f"Shot mismatch {eid}/{side}"
                checked["shots"]+=len(raw[side+"_shots"])
            elif raw[side+"_shots"] is not None:
                assert fields[(eid,side+"_shots")]==raw[side+"_shots"], f"Unavailable shot evidence mismatch {eid}/{side}"
            heatmap=raw[side+"_heatmap"]
            if raw[side+"_heatmap_available"]:
                for key,kind in (("playerPoints","player"),("goalkeeperPoints","goalkeeper")):
                    assert points[(eid,raw[side+"_id"],kind)]==heatmap[key], f"Heatmap mismatch {eid}/{side}"
                    checked["heatmap_points"]+=len(heatmap[key])
            elif heatmap is not None:
                assert fields[(eid,side+"_heatmap")]==heatmap, f"Unavailable heatmap evidence mismatch {eid}/{side}"
                checked["unavailable_component_objects"]+=1
    if lineups is not None:
        rebuilt=legacy_lineups_view(tables).set_index("match_id")
        for row in lineups.to_dict("records"):
            for side in ("home","away"):
                assert equal_value(row[side],rebuilt.at[row["match_id"],side]), f"Lineup mismatch {row['match_id']}/{side}"
                if isinstance(row[side],dict):checked["lineup_players"]+=len(row[side].get("roster",{}))
    assert not any(tables.get(name) for name in ("incidents","comments","goal_sequences","goal_actions","passing_edges","timing","availability"))
    return {"passed":True, **dict(checked)}


def verify_publication(output, result):
    season = Path(output)/result["season_file"]
    if hash_file(season)!=result["sha256"]:
        raise ValueError("Season output hash mismatch")
    manifest=Path(output)/result["manifest"]
    data=json.loads(manifest.read_text())
    for info in data["tables"].values():
        path=manifest.parent/info["file"]
        if hash_file(path)!=info["sha256"] or pq.ParquetFile(path).metadata.num_rows!=info["rows"]:
            raise ValueError("Published table hash or count mismatch")
    return data


def migrate(source, lineup_dir, output, *, seasons=None, include_lineups=True):
    source,lineup_dir,output = Path(source).resolve(),Path(lineup_dir).resolve(),Path(output).resolve()
    if output in (source,lineup_dir):
        raise ValueError("Choose a separate output folder")
    output.mkdir(parents=True,exist_ok=True)
    audit=output/"_migration"; audit.mkdir(exist_ok=True)
    registry=json.loads((Path(__file__).with_name("competition_registry.json")).read_text())
    files=[p for p in sorted(source.glob("*.csv")) if not p.stem.endswith("_lineups")]
    if seasons:
        if set(seasons)-{p.stem for p in files}:raise ValueError("Unknown season selection")
        files=[p for p in files if p.stem in seasons]
    if not files:raise ValueError("No legacy seasons found")
    inventory=[inspect_source(path,lineup_dir,registry) for path in files]
    owners={}
    for item in inventory:
        if item["status"]!="ready":continue
        overlap=set(item["event_ids"]) & owners.keys()
        if overlap:
            raise ValueError(f"Duplicate match IDs in otherwise valid season identities: {item['stem']}")
        owners.update({eid:item["stem"] for eid in item["event_ids"]})
    atomic_write(audit/"inventory.json",json_bytes(inventory))
    profile=Profile(**{**asdict(PROFILE),"include_lineups":include_lineups})
    results=[]; start=time.time()
    for index,item in enumerate(inventory,1):
        summary={k:v for k,v in item.items() if k!="event_ids"}
        if item["status"]=="quarantined":
            results.append(summary)
            print(json.dumps({"index":index,"total":len(inventory),"season":item["stem"],"status":"quarantined"}),flush=True)
            continue
        print(json.dumps({"index":index,"total":len(inventory),"season":item["stem"],"status":"converting"}),flush=True)
        completed=output/(item["stem"]+".manifest.json")
        tables=None
        try:
            if completed.exists():
                existing=json.loads(completed.read_text())
                provenance=existing.get("reconciliation",{})
                expected=(VERSION,item["source_sha256"],item["lineup_sha256"] if include_lineups else None)
                actual=(provenance.get("migration_version"),provenance.get("match_sha256"),provenance.get("lineup_sha256"))
                if expected!=actual or existing.get("profile")!=asdict(profile):
                    raise ValueError("Existing output differs from this migration; choose a new output folder")
                verify_publication(output,existing)
                summary.update(status="complete",resumed=True,output=existing)
            else:
                scope=Scope(**item["scope"])
                tables,lineage=normalize_legacy(item["source"],scope,include_lineups=include_lineups,
                    lineup_path=item["lineup_source"],normalize_scalars=True)
                verified=verify_mapping(tables,item["source"],item["lineup_source"],include_lineups=include_lineups)
                if hash_file(item["source"])!=item["source_sha256"] or (include_lineups and item["lineup_source"] and hash_file(item["lineup_source"])!=item["lineup_sha256"]):
                    raise ValueError("Source changed during conversion")
                lineage.update(collection="legacy_migration",migration_version=VERSION,validation=verified,
                    complete_semantics="All retained legacy values converted; not complete API-field coverage",api_requests=0)
                published=publish_season(tables,output,scope,profile,item["stem"],lineage)
                verify_publication(output,published)
                summary.update(status="complete",resumed=False,output=published)
        except Exception as exc:
            summary.update(status="failed",error_type=type(exc).__name__,error=str(exc))
        finally:
            tables=None;gc.collect()
        results.append(summary)
        print(json.dumps({"index":index,"total":len(inventory),"season":item["stem"],"status":summary["status"],
                          "error":summary.get("error"),"elapsed_seconds":round(time.time()-start,1)}),flush=True)
        atomic_write(audit/"progress.json",json_bytes({"completed_items":len(results),"total_items":len(inventory),"results":results}))
    report={"migration_version":VERSION,"source":str(source),"lineup_source":str(lineup_dir),"output":str(output),
            "api_requests":0,"status_counts":dict(Counter(r["status"] for r in results)),"seasons":results,
            "matches":sum(r["matches"] for r in results if r["status"]=="complete"),"elapsed_seconds":time.time()-start,
            "all_valid_seasons_converted":not any(r["status"]=="failed" for r in results),
            "profile":asdict(profile)}
    changed_sources=[]
    for item in inventory:
        for key,hash_key in (("source","source_sha256"),("lineup_source","lineup_sha256")):
            if item[key] and hash_file(item[key])!=item[hash_key]:
                changed_sources.append(item[key])
    report["source_files_unchanged"] = not changed_sources
    report["changed_sources"] = changed_sources
    report["all_valid_seasons_converted"] &= not changed_sources
    atomic_write(audit/"report.json",json_bytes(report))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=ROOT/"data/xDiyo_data_promotions")
    parser.add_argument("--lineups",type=Path,default=ROOT/"data/xDiyo_data_heatmaps")
    parser.add_argument("--output",type=Path,default=ROOT/"data/xDiyo_data_legacy")
    parser.add_argument("--season",action="append")
    parser.add_argument("--no-lineups",action="store_true")
    args=parser.parse_args()
    result=migrate(args.source,args.lineups,args.output,seasons=args.season,include_lineups=not args.no_lineups)
    print(json.dumps({k:v for k,v in result.items() if k!="seasons"},indent=2))
    if not result["all_valid_seasons_converted"]:raise SystemExit(1)


if __name__=="__main__":main()
