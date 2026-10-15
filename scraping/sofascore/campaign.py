"""Serial historical recollection with durable season checkpoints.

The source CSVs define the league/season cohort and regular-round scope only.
All published values come from the new collector's observed API responses.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import time

from .client import BrowserClient
from .config import Profile, RatePolicy, Scope
from .discovery import event_status, round_jobs
from .export import normalize_store, publish_tables
from .models import CollectionPaused, Job
from .planner import component_jobs
from .rate_limit import RateLimiter, retry_after_seconds
from .runner import Runner
from .schema import TABLES, to_arrow
from .store import Store, atomic_write, json_bytes, season_year_pair

ROOT = Path(__file__).resolve().parents[2]
PROFILE = Profile(include_goal_actions=True, include_lineups=True, include_comments=False)
# Explicit user cadence: normal serial navigation, then five minutes per season.
# These are not claims about provider quotas. Retry-After and blocks still apply.
POLICY = RatePolicy(min_interval_seconds=None, per_minute=None, per_hour=None,
                    max_attempts_per_run=None)
TERMINAL = {"success", "empty", "unavailable"}


def inventory(source):
    registry = json.loads((Path(__file__).parent/"competition_registry.json").read_text())
    result = []
    for path in sorted(Path(source).glob("*.csv")):
        match = re.fullmatch(r"(.+)_(\d{2})_(\d{2})", path.stem)
        if not match or match[1] not in registry:
            raise ValueError(f"Unrecognized season filename: {path.name}")
        years = (2000+int(match[2]), 2000+int(match[3]))
        ids, seasons, labels, rounds = set(), set(), set(), set()
        count = 0
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                count += 1
                ids.add(int(row["match_id"]))
                seasons.add(int(row["season_id"]))
                labels.add(season_year_pair(row["season_year"]))
                rounds.add(int(float(row["round"])))
        if len(seasons) != 1 or count != len(ids) or not ids:
            raise ValueError(f"Invalid legacy identity inventory: {path.name}")
        correct = labels == {years}
        result.append({"stem":path.stem, "source":str(path.resolve()),
            "source_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
            "league_id":registry[match[1]]["competition_id"], "league_name":match[1],
            "season_id":next(iter(seasons)) if correct else None,
            "season_start":years[0], "season_end":years[1], "rounds":sorted(rounds),
            "legacy_event_ids":sorted(ids) if correct else [], "legacy_rows":count,
            "legacy_identity_valid":correct, "legacy_stored_season_id":next(iter(seasons))})
    if not result:
        raise ValueError("No legacy seasons were found")
    # Recent data first gives useful xG-rich seasons early; all seasons remain queued.
    return sorted(result, key=lambda r:(-r["season_start"],r["league_name"]))


def publish_season(tables, output, scope, profile, stem, reconciliation):
    """One same-stem typed nested Parquet, plus a lossless relational manifest."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    output = Path(output)
    table_root = output / "_tables"
    if reconciliation.get("archive_cohort", {}).get("shared_provider_identity"):
        # Two short tournaments may share one historical provider identity.
        # Keep their CURRENT pointers separate without inventing provider IDs.
        if not re.fullmatch(r"[A-Za-z0-9_]+_\d{2}_\d{2}", stem):
            raise ValueError("Invalid shared-provider cohort stem")
        table_root = table_root / "cohorts" / stem
    canonical = publish_tables(tables, table_root, scope=scope, profile=profile,
                               provenance={"collection":"historical_campaign", **reconciliation})
    canonical_info = json.loads(canonical.read_text())
    if not canonical_info["requested_components_terminal"]:
        raise ValueError("Refusing to publish an incomplete season entry")
    arrow = to_arrow("matches", tables["matches"])
    event_ids = arrow["event_id"].to_pylist()
    for name in TABLES:
        if name == "matches" or name not in tables or "event_id" not in TABLES[name]:
            continue
        grouped = defaultdict(list)
        for row in tables[name]:
            grouped[row["event_id"]].append(row)
        ordered = [row for eid in event_ids for row in grouped[eid]]
        child = to_arrow(name, ordered).combine_chunks()
        offsets = [0]
        for eid in event_ids:
            offsets.append(offsets[-1]+len(grouped[eid]))
        values = pa.StructArray.from_arrays([c.chunk(0) if len(c.chunks) else pa.array([],type=c.type)
                                            for c in child.columns], fields=list(child.schema))
        nested = pa.ListArray.from_arrays(pa.array(offsets,type=pa.int32()),values)
        arrow = arrow.append_column(name, nested)
    relative = canonical.relative_to(output).as_posix()
    metadata = {**(arrow.schema.metadata or {}), b"xdiyo_canonical_manifest":relative.encode(),
                b"xdiyo_layout":b"one_match_per_row_with_typed_component_lists",
                b"xdiyo_season_stem":stem.encode()}
    arrow = arrow.replace_schema_metadata(metadata)
    path = output/(stem+".parquet")
    temporary = path.with_suffix(".parquet.tmp")
    pq.write_table(arrow,temporary,compression="zstd",row_group_size=64,use_compliant_nested_type=False)
    if not pq.ParquetFile(temporary).read().equals(arrow,check_metadata=True):
        raise ValueError("Season Parquet round-trip mismatch")
    os.replace(temporary,path)
    shots = tables.get("shots",[])
    report = {"manifest":relative, "season_file":path.name, "complete":True,
        "complete_scope":"Saved collection cohort; independent played-fixture coverage is not implied",
        "schema_version":canonical_info["schema_version"], "matches":len(event_ids),
        "sha256":hashlib.sha256(path.read_bytes()).hexdigest(), "bytes":path.stat().st_size,
        "shots":len(shots), "shots_with_xg":sum(r.get("xg") is not None for r in shots),
        "lineup_players":len(tables.get("lineup_players",[])),
        "lineup_events":len({r["event_id"] for r in tables.get("lineup_teams",[])}),
        "heatmap_points":len(tables.get("heatmap_points",[])),
        "coverage":canonical_info["coverage_summary"], "reconciliation":reconciliation,
        "profile":asdict(profile), "completed_at":time.time()}
    atomic_write(output/(stem+".manifest.json"),json_bytes(report))
    return report


class CampaignRunner(Runner):
    def __init__(self, *args, progress, stop_path, keep_client_open=False, **kwargs):
        super().__init__(*args,**kwargs)
        self.progress, self.stop_path = progress, Path(stop_path)
        self.completed = 0
        self.keep_client_open = keep_client_open

    def _fetch(self, job, event=None):
        if self.stop_path.exists():
            raise CollectionPaused("Campaign stop requested")
        result = super()._fetch(job,event)
        self.completed += 1
        if self.completed == 1 or self.completed % 10 == 0 or result == "failed":
            self.progress(last_component=job.component,last_event_id=job.event_id,
                          requests_this_session=self.completed)
        return result

    def cached_fetch(self, job, event=None):
        state = self.store.get(job)
        # A historical campaign does not periodically refetch terminal evidence.
        if state is None or state["status"] not in TERMINAL:
            self._fetch(job,event)
            state = self.store.get(job)
        if state is None or state["status"] not in TERMINAL:
            raise CollectionPaused(f"Component needs inspection: {job.path}")
        return self.store.payload(job)

    def collect_season(self, item, output):
        with self.store.run_lock():
            run_id = self.store.start_run()
            self.limiter = RateLimiter(self.store,self.policy,run_id)
            summary = {"stem":item["stem"],"status":"interrupted","rate_policy":asdict(self.policy)}
            try:
                if item["season_id"] is None:
                    self.progress(phase="resolving_season_identity")
                    job = Job(f"/unique-tournament/{item['league_id']}/seasons","season_registry")
                    payload = self.cached_fetch(job)
                    choices = [s for s in (payload or {}).get("seasons",[])
                               if season_year_pair(s.get("year")) == (item["season_start"],item["season_end"])]
                    if len(choices) != 1:
                        raise ValueError(f"Could not uniquely resolve {item['stem']} in provider season list")
                    item["season_id"] = choices[0]["id"]
                scope = Scope(item["league_id"],item["season_id"],tuple(item["rounds"]),
                              item["league_name"],item["season_start"],item["season_end"])
                self.progress(phase="discovery",season_id=scope.season_id)
                for job in round_jobs(scope):
                    payload = self.cached_fetch(job)
                    if payload is None:
                        raise ValueError(f"Requested round unavailable: {job.path}")
                    self.store.add_events(payload["events"],scope,job,self.store.get(job)["last_success"])
                actual = {e["id"] for e in self.store.events(scope.league_id,scope.season_id)}
                missing = sorted(set(item["legacy_event_ids"])-actual)
                unavailable_events = []
                for eid in missing:
                    job = Job(f"/event/{eid}","event",eid,reasons=("legacy_discovery_reconciliation",))
                    payload = self.cached_fetch(job)
                    if payload is None:
                        unavailable_events.append(eid)
                    else:
                        if payload["event"]["id"] != eid:
                            raise ValueError("Reconciliation event ID mismatch")
                        self.store.add_events([payload["event"]],scope,job,self.store.get(job)["last_success"])
                if unavailable_events:
                    raise ValueError(f"Legacy events no longer retrievable: {unavailable_events}")
                events = self.store.events(scope.league_id,scope.season_id)
                if not events:
                    raise ValueError("Empty season discovery")
                for event in events:
                    if season_year_pair((event.get("season") or {}).get("year")) != (scope.season_start,scope.season_end):
                        raise ValueError("Cached event filename/season-year conflict")
                jobs = [(job,event) for event in events for job in component_jobs(event,self.profile,time.time())]
                self.progress(phase="components",discovered_matches=len(events),
                              requested_components=len(jobs),
                              terminal_components=sum(bool(self.store.get(j)) and self.store.get(j)["status"] in TERMINAL
                                                      for j,_ in jobs))
                for index,(job,event) in enumerate(jobs):
                    self.cached_fetch(job,event)
                    if index % 25 == 0:
                        self.progress(terminal_components=index+1)
                if any(self.store.get(j) is None or self.store.get(j)["status"] not in TERMINAL for j,_ in jobs):
                    raise ValueError("Nonterminal component in completed season")
                reconciliation = {"legacy_source_sha256":item["source_sha256"],
                    "legacy_identity_valid":item["legacy_identity_valid"],
                    "legacy_rows":item["legacy_rows"], "discovered_matches":len(events),
                    "reconciled_missing_from_rounds":missing,
                    "extra_event_ids":sorted({e["id"] for e in events}-set(item["legacy_event_ids"])),
                    "status_counts":dict(Counter(event_status(e) for e in events)),
                    "rounds":list(scope.rounds)}
                self.progress(phase="exporting",terminal_components=len(jobs))
                report = publish_season(normalize_store(self.store,scope,self.profile),output,scope,
                                        self.profile,item["stem"],reconciliation)
                summary.update(status="complete",report=report)
                return report
            except CollectionPaused as exc:
                summary.update(status="paused",reason=str(exc))
                raise
            except Exception as exc:
                summary.update(status="failed",reason=str(exc))
                raise
            finally:
                if not self.keep_client_open:
                    self.client.close()
                summary["attempts"] = self.store.db.execute(
                    "SELECT COUNT(*) FROM attempts WHERE run_id=?",(run_id,)).fetchone()[0]
                self.store.finish_run(run_id,summary["status"],summary)


def prepare_manual_access(store, client, policy, progress, stop_path):
    """Account website traffic and clear a saved block only after user input."""
    with store.run_lock():
        run_id = store.start_run()
        summary = {"mode":"manual_captcha", "status":"interrupted", "attempts":0}

        def account(requests):
            now = time.time()
            with store.db:
                for request in requests:
                    store.db.execute("INSERT INTO attempts(run_id,key,started,status,outcome) VALUES (?,?,?,?,?)",
                        (run_id,"manual_browser:"+request["path"],now,request["status"],"manual_browser_visit"))
            summary["attempts"] += len(requests)
            if requests:
                store.set_setting("last_request_finished",now)
            for request in requests:
                delay = retry_after_seconds(request["retry_after"],now)
                if delay is None and request["status"] == 429:
                    delay = policy.default_cooldown_seconds
                if delay is not None:
                    store.set_setting("cooldown_until",max(store.setting("cooldown_until",0),now+delay))

        try:
            progress(state="waiting_for_user",phase="manual_captcha")
            client.prepare_manual_access(on_requests=account)
            if Path(stop_path).exists():
                raise CollectionPaused("Campaign stop requested")
            store.set_setting("last_manual_access_confirmation",{
                "at":time.time(), "previous_block":store.setting("access_block"),
                "observed_api_requests":summary["attempts"],
                "cooldown_until":store.setting("cooldown_until",0)})
            store.set_setting("access_block",None)
            summary["status"] = "confirmed"
        finally:
            store.finish_run(run_id,summary["status"],summary)


def main(argv=None, *, on_progress=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=ROOT/"data/xDiyo_data_promotions")
    parser.add_argument("--output",type=Path,default=ROOT/"data/xDiyo_data")
    parser.add_argument("--store",type=Path,default=ROOT/"data/collection_state")
    parser.add_argument("--driver-path",type=Path)
    parser.add_argument("--manual-captcha",action="store_true",
                        help="Open Sofascore for manual CAPTCHA completion, then reuse that browser session")
    parser.add_argument("--random-delay-max-seconds",type=float,default=None,
                        help="Add a uniform random delay from 0 to this many seconds between requests")
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--season",action="append",help="Restrict to exact stems for diagnostics")
    parser.add_argument("--pause-on-block",action="store_true",
                        help="Return control on an access block, for manual notebook recovery")
    parser.add_argument("--resume",action="store_true",
                        help="Remove a previous STOP marker after acquiring both collector locks")
    parser.add_argument("--retry-blocked",action="store_true",
                        help="Explicitly retry a previous access block once; retain server cooldowns")
    args = parser.parse_args(argv)
    policy = replace(POLICY,random_delay_max_seconds=args.random_delay_max_seconds)
    args.output.mkdir(parents=True,exist_ok=True)
    control = args.output/"_collection"
    control.mkdir(exist_ok=True)
    campaign_path = control/"inventory.json"
    if campaign_path.exists():
        items = json.loads(campaign_path.read_text())["seasons"]
        observed = {p.stem for p in args.source.glob("*.csv")}
        if observed != {i["stem"] for i in items}:
            raise ValueError("Legacy season cohort changed; rebuild the inventory explicitly")
    else:
        items = inventory(args.source)
        atomic_write(campaign_path,json_bytes({"created_at":time.time(),"seasons":items,
                     "profile":asdict(PROFILE),"rate_policy":asdict(policy),"between_seasons_seconds":300}))
    if args.season:
        selected = set(args.season)
        if selected-{i["stem"] for i in items}:
            raise ValueError("Unknown requested season stem")
        items = [i for i in items if i["stem"] in selected]
    if args.prepare_only:
        print(json.dumps({"seasons":len(items),"output":str(args.output),"api_requests":0}))
        return 0
    status_path = control/"status.json"
    state = {"pid":os.getpid(),"started_at":time.time(),"state":"running",
             "total_seasons":len(items),"completed_seasons":0,"profile":asdict(PROFILE),
             "rate_policy":asdict(policy),"between_seasons_seconds":300,
             "manual_captcha":args.manual_captcha}

    def progress(**updates):
        state.update(updates,updated_at=time.time())
        atomic_write(status_path,json_bytes(state))
        if on_progress is not None:
            on_progress(dict(state))

    def rest(until, reason):
        progress(state="waiting",phase=reason,resume_at=until)
        while time.time()<until:
            if (control/"STOP").exists():
                raise CollectionPaused("Campaign stop requested")
            time.sleep(min(1,until-time.time()) if until>time.time() else 0)

    owns_campaign = False
    manual_client = None
    try:
        # Separate campaign lock guards exports/rest as well as HTTP sessions.
        with Store(control/"lock") as lock, lock.run_lock(), Store(args.store) as store:
            owns_campaign = True
            # Never alter another running collector's stop or recovery controls.
            with store.run_lock():
                if args.resume:
                    (control/"STOP").unlink(missing_ok=True)
                if args.retry_blocked and not args.manual_captcha and not (control/"STOP").exists():
                    previous = store.setting("access_block")
                    if previous:
                        store.set_setting("last_manual_access_retry", {
                            "at":time.time(), "previous_block":previous,
                            "cooldown_until":store.setting("cooldown_until",0)})
                        store.set_setting("access_block",None)
            for item in items:
                done_path = args.output/(item["stem"]+".manifest.json")
                if done_path.exists():
                    done = json.loads(done_path.read_text())
                    result_path=args.output/done["season_file"]
                    if done.get("complete") and result_path.exists() and hashlib.sha256(result_path.read_bytes()).hexdigest()==done["sha256"]:
                        progress(completed_seasons=state["completed_seasons"]+1)
                        last_end=done.get("completed_at",0)
                        if state["completed_seasons"]<len(items) and last_end+300>time.time():
                            rest(last_end+300,"between_seasons")
                        continue
                progress(state="running",current_season=item["stem"],phase="starting_season",
                         season_started_at=time.time(),requests_this_session=0,terminal_components=0)
                while True:
                    if (control/"STOP").exists():
                        raise CollectionPaused("Campaign stop requested")
                    if args.manual_captcha and manual_client is None:
                        cooldown=store.setting("cooldown_until",0)
                        if cooldown>time.time():
                            rest(cooldown,"provider_cooldown")
                        manual_client=BrowserClient(policy,driver_path=args.driver_path)
                        prepare_manual_access(store,manual_client,policy,progress,control/"STOP")
                    block=store.setting("access_block")
                    if block:
                        progress(state="needs_vpn_recovery",phase="access_block",access_block=block)
                        if args.pause_on_block:
                            raise CollectionPaused("Access blocked. Resolve access, then explicitly retry from the notebook.")
                        time.sleep(2)
                        continue
                    cooldown=store.setting("cooldown_until",0)
                    if cooldown>time.time():
                        rest(cooldown,"provider_cooldown")
                    progress(state="running",access_block=None)
                    client=manual_client or BrowserClient(policy,driver_path=args.driver_path)
                    runner=CampaignRunner(store,client,policy,PROFILE,progress=progress,
                                          stop_path=control/"STOP",keep_client_open=args.manual_captcha)
                    try:
                        report=runner.collect_season(item,args.output)
                    except CollectionPaused:
                        if store.setting("access_block") or store.setting("cooldown_until",0)>time.time():
                            continue
                        raise
                    else:
                        progress(completed_seasons=state["completed_seasons"]+1,last_completed=report)
                        print(json.dumps({"completed":item["stem"],"matches":report["matches"],
                                          "shots_with_xg":report["shots_with_xg"]}),flush=True)
                        break
                gc.collect()
                if state["completed_seasons"]<len(items):
                    rest(time.time()+300,"between_seasons")
            progress(state="complete",phase="complete",completed_at=time.time())
    except BaseException as exc:
        if owns_campaign:
            progress(state="paused" if isinstance(exc,(CollectionPaused,KeyboardInterrupt)) else "failed",
                     error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        if manual_client is not None:
            manual_client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
