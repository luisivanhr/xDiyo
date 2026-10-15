from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

from .client import BrowserClient
from .audit import audit_legacy
from .config import Profile, RatePolicy, Scope
from .export import normalize_store, publish_tables
from .legacy import normalize_legacy
from .planner import build_plan
from .runner import Runner
from .store import Store, atomic_write, json_bytes

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = set(data)-{"scope","profile","rate","driver_path","chrome_binary","readiness_pause","output_root"}
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    return (Scope(**data["scope"]),Profile(**data.get("profile",{})),RatePolicy(**data.get("rate",{})),data)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Serial, cached Sofascore collection. Plan/export never open a browser.")
    parser.add_argument("command",choices=["plan","pilot","discover","collect","resume","refresh-selected","export","export-csv","import-legacy","audit","status","clear-block"])
    parser.add_argument("--config",type=Path)
    parser.add_argument("--store",type=Path,default=PROJECT_ROOT/"data"/"collection_state",
                        help="Shared ledger for all seasons; do not use separate stores for concurrent runs")
    parser.add_argument("--event-id",type=int,action="append")
    parser.add_argument("--component",action="append",choices=["pregame","statistics","shotmap","heatmap","lineups","incidents","comments"])
    parser.add_argument("--refresh-discovery",action="store_true")
    parser.add_argument("--input",type=Path,help="Source CSV for import-legacy, or directory for audit")
    parser.add_argument("--output",type=Path,help="Output directory for import-legacy or JSON report path for audit")
    args = parser.parse_args(argv)
    if args.command not in {"status","clear-block","audit","export-csv"} and not args.config:
        parser.error("--config is required")
    if args.command == "refresh-selected" and (not args.event_id or not args.component):
        parser.error("refresh-selected requires --event-id and --component")
    if args.command=="pilot" and (not args.event_id or len(set(args.event_id))>8):
        parser.error("pilot requires one to eight --event-id values")
    if args.command in {"import-legacy","audit","export-csv"} and (args.input is None or args.output is None):
        parser.error("import-legacy/audit requires --input and --output")
    if args.command=="audit":
        report=audit_legacy(args.input)
        atomic_write(args.output,json_bytes(report))
        print(json.dumps({"report":str(args.output.resolve()),"files":len(report["files"]),"match_rows":report["match_rows"]},indent=2))
        return 0
    if args.command=="export-csv":
        from .compatibility import export_csv
        result = export_csv(args.input,args.output)
        print(json.dumps({"csv_manifest":str(result),"api_requests":0},indent=2))
        return 0
    with Store(args.store) as store:
        if args.command == "status":
            result = {"access_block":store.setting("access_block"),"cooldown_until":store.setting("cooldown_until"),
                      "components":dict(store.db.execute("SELECT status,COUNT(*) FROM components GROUP BY status").fetchall()),
                      "events":store.db.execute("SELECT COUNT(*) FROM events").fetchone()[0]}
        elif args.command == "clear-block":
            store.set_setting("access_block",None)
            result = {"access_block":None,"cooldown_until":store.setting("cooldown_until"),
                      "note":"Only the access latch was cleared. Active server cooldown remains enforced."}
        else:
            scope,profile,policy,data = load_config(args.config)
            event_ids = set(args.event_id) if args.event_id else None
            if args.command == "plan":
                result = build_plan(store.events(scope.league_id,scope.season_id),profile,store,time.time(),
                                    refresh_components=args.component or (),event_ids=event_ids,only_components=args.component).as_dict()
                result["discovery_note"] = "Plan uses cached events. Run discover explicitly to populate or update fixtures."
            elif args.command == "import-legacy":
                tables,lineage=normalize_legacy(args.input,scope,include_lineups=profile.include_lineups)
                scope=replace(scope,rounds=tuple(lineage["imported_rounds"]))
                destination=publish_tables(tables,args.output,scope=scope,profile=profile,provenance=lineage)
                result={"manifest":str(destination),"rows":len(tables["matches"]),"api_requests":0}
            elif args.command == "export":
                destination = publish_tables(normalize_store(store,scope,profile),
                    data.get("output_root",PROJECT_ROOT/"data"/"prepared_v2"),scope=scope,profile=profile)
                result = {"manifest":str(destination)}
            else:
                client = BrowserClient(policy,driver_path=data.get("driver_path"),chrome_binary=data.get("chrome_binary"),
                                       locale=profile.locale,readiness_pause=data.get("readiness_pause",False))
                result = Runner(store,client,policy,profile).run(scope,
                    discover=args.command in {"discover","collect"}, collect=args.command != "discover",
                    discovery_refresh=args.refresh_discovery,event_ids=event_ids,
                    refresh_components=(args.component or ()) if args.command=="refresh-selected" else (),
                    only_components=args.component,pilot=args.command=="pilot")
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return 2 if result.get("status") in {"failed","paused","incomplete","interrupted"} else 0


if __name__ == "__main__":
    sys.exit(main())
