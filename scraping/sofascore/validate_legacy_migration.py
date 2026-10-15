"""Audit published offline seasons and summarize their actual field coverage."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow.compute as pc

from .audit import hash_file
from .migrate_legacy import ROOT, verify_publication
from .schema import arrow_schema
from .store import atomic_write, json_bytes


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(output):
    output=Path(output).resolve()
    report=json.loads((output/"_migration/report.json").read_text())
    require(report["all_valid_seasons_converted"], "The migration has failed seasons or changed sources")
    require(report["source_files_unchanged"] and report["api_requests"]==0, "Invalid migration provenance")
    counts=Counter(); coverage=Counter(); quality=Counter(); leagues=Counter()
    event_ids=set(); sources=set(); scalar_checks=Counter()
    completed=0
    for entry in report["seasons"]:
        for key,hash_key in (("source","source_sha256"),("lineup_source","lineup_sha256")):
            if entry[key]:
                require(hash_file(entry[key])==entry[hash_key], f"Changed source: {entry[key]}")
                sources.add(entry[key])
        if entry["status"]=="quarantined":
            require(not (output/(entry["stem"]+".parquet")).exists(), "Quarantined season was published")
            continue
        result=entry["output"]
        manifest=verify_publication(output,result)
        base=(output/result["manifest"]).parent
        def read(name, columns):
            info=manifest["tables"].get(name)
            return pq.ParquetFile(base/info["file"]).read(columns=columns) if info else None
        def records(name, columns):
            table=read(name,columns)
            return table.to_pylist() if table is not None else []
        for name,info in manifest["tables"].items():
            require(pq.ParquetFile(base/info["file"]).schema_arrow.equals(arrow_schema(name),check_metadata=True),
                    f"Schema mismatch: {entry['stem']}/{name}")
            counts[name]+=info["rows"]
            quality["canonical_bytes"]+=info["bytes"]
        quality["nested_season_bytes"]+=result["bytes"]
        for excluded in ("incidents","comments","availability","timing","goal_sequences","goal_actions","passing_edges"):
            require(manifest["tables"].get(excluded,{}).get("rows",0)==0, f"Unexpected {excluded}")
        matches=records("matches",["event_id","season_id","competition_id","home_id","away_id","kickoff_utc","event_metadata_observed_at","status"])
        local={r["event_id"] for r in matches}
        require(len(local)==len(matches) and not local & event_ids, "Duplicate match ID")
        require(len(local)==entry["matches"]==result["matches"], "Match count mismatch")
        event_ids.update(local)
        team_ids={m[side+"_id"] for m in matches for side in ("home","away")}
        require(all(m["kickoff_utc"] is None and m["event_metadata_observed_at"] is None and m["status"]=="unknown" for m in matches),
                "Legacy import fabricated match dates or status")
        statistics=read("statistics",["event_id","side","period","key","value"])
        if statistics is not None:
            mask=pc.and_(pc.and_(pc.equal(statistics["key"],"expectedgoals"),pc.equal(statistics["period"],"ALL")),
                         pc.is_valid(statistics["value"]))
            aggregate_xg=statistics.filter(mask).select(["event_id","side"]).to_pylist()
            by_event=Counter(r["event_id"] for r in aggregate_xg)
            quality["match_sides_with_stored_aggregate_xg"]+=len(aggregate_xg)
            quality["matches_with_both_sides_stored_aggregate_xg"]+=sum(v==2 for v in by_event.values())
        for table,columns in (("shots",["xg","xgot","legacy_xg_stored","legacy_xgot_stored","xg_quality","shot_id","player_id","player_x"]),):
            for batch in pq.ParquetFile(base/manifest["tables"][table]["file"]).iter_batches(columns=columns,batch_size=8192) if table in manifest["tables"] else []:
                for shot in batch.to_pylist():
                    require(shot["shot_id"] is None and shot["player_id"] is None and shot["player_x"] is None,
                            "Legacy shot IDs or coordinates unexpectedly present")
                    expected=shot["legacy_xg_stored"]
                    require(shot["xg"]==(None if expected==1 else expected), "Shot xG trust-policy mismatch")
                    expected=shot["legacy_xgot_stored"]
                    require(shot["xgot"]==(None if expected in (0,1) else expected), "Shot xGOT trust-policy mismatch")
                    quality["shots_with_usable_stored_xg"]+=shot["xg"] is not None
                    quality["shots_with_ambiguous_default_xg"]+=shot["xg_quality"]=="legacy_unverified_one"
                    quality["shots_with_usable_stored_xgot"]+=shot["xgot"] is not None
        players=read("lineup_players",["player_id","statistics_status"])
        if players is not None:
            require(players["player_id"].null_count==len(players), "Legacy player ID unexpectedly present")
            quality["lineup_players_with_empty_statistics"]+=sum(v=="legacy_empty_unknown_reason" for v in players["statistics_status"].to_pylist())
        for row in records("team_seasons",["team_id","competition_id","season_id","membership_status","got_promoted","got_demoted"]):
            require(row["team_id"] in team_ids and row["membership_status"]=="legacy_unverified", "Invalid legacy flag provenance")
            require(row["season_id"]==entry["scope"]["season_id"] and row["competition_id"]==entry["scope"]["league_id"], "Flag scope mismatch")
            quality["team_seasons_reported_promoted"]+=row["got_promoted"]
            quality["team_seasons_reported_demoted"]+=row["got_demoted"]
        for row in records("coverage",["component","status"]):
            coverage[row["component"]+":"+row["status"]]+=1
        lineage=result["reconciliation"]
        require(lineage["validation"]["passed"] and lineage["collection"]=="legacy_migration", "Missing source-value validation")
        for name,value in lineage["validation"].items():
            if name!="passed":scalar_checks[name]+=value
        leagues[entry["scope"]["league_name"]]+=len(matches)
        completed+=1
    require(len(event_ids)==report["matches"], "Cohort match count mismatch")
    audit={"passed":True,"api_requests":0,"seasons":completed,"matches":len(event_ids),
           "quarantined":[{"season":e["stem"],"reason":e["reason"]} for e in report["seasons"] if e["status"]=="quarantined"],
           "source_files_reverified":len(sources),"source_bytes":sum(Path(p).stat().st_size for p in sources),
           "table_rows":dict(counts),"quality_and_storage":dict(quality),"coverage":dict(coverage),
           "original_value_checks":dict(scalar_checks),"matches_by_league":dict(leagues),
           "data_completeness":"Coverage of the stored legacy cohort only; unavailable API fields are not recovered"}
    atomic_write(output/"_migration/validation.json",json_bytes(audit))
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"data/xDiyo_data_legacy")
    result=validate(parser.parse_args().output)
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
