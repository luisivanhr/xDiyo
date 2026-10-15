"""Durable queue for the Chrome DevTools collector. This module sends no HTTP."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time
from urllib.parse import urlsplit
import uuid

from scraping.sofascore.config import Profile
from scraping.sofascore.discovery import event_status, is_completed_event
from scraping.sofascore.models import Job
from scraping.sofascore.event_round import match_season
from scraping.sofascore.parsers import validate_payload
from scraping.sofascore.planner import component_jobs
from scraping.sofascore.store import atomic_write, json_bytes, season_year_pair
from scraping.sofascore.export import read_tables
from .capture import ROOT, PROFILE, api_path, match_url, reference_from_cache
from .coverage import assess_features
from .import_har import import_har, response_json, response_time, _check_event

BASE = ROOT / "data/website_capture_v3/campaigns"
PROTOCOL_VERSION = 1
TERMINAL = {"success", "empty", "unavailable"}
DEFAULT_POLICY = {"random_action_delay_max_seconds":None, "between_seasons_seconds":300,
                  "match_timeout_seconds":90, "view_wait_seconds":8, "max_api_observations_per_match":120,
                  "max_api_observations_per_run":None, "pause_on_missing_core":True,
                  "memory_reset_every_matches":None, "collection_mode":"page",
                  "session_request_interval_seconds":0.5, "session_challenge_recovery":False,
                  "session_challenge_cooldown_seconds":180}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_policy(policy):
    result = {**DEFAULT_POLICY, **policy}
    if set(result) != set(DEFAULT_POLICY):
        raise ValueError("Unknown browser pacing setting")
    for name, value in result.items():
        if name == "collection_mode":
            if value not in ("page", "session"): raise ValueError("collection_mode must be page or session")
            continue
        if name in {"pause_on_missing_core", "session_challenge_recovery"}:
            if not isinstance(value, bool): raise ValueError(f"{name} must be boolean")
            continue
        if name in {"random_action_delay_max_seconds", "max_api_observations_per_run", "memory_reset_every_matches"} and value is None: continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")
        if name not in {"random_action_delay_max_seconds", "between_seasons_seconds", "session_request_interval_seconds", "session_challenge_cooldown_seconds"} and value <= 0:
            raise ValueError(f"{name} must be positive")
        if (name.startswith("max_api_observations") or name == "memory_reset_every_matches") and type(value) is not int:
            raise ValueError(f"{name} must be an integer")
    return result


def prepare_campaign(event_ids, *, name="pilot", profile=PROFILE, policy=None, base=BASE, cache_root=None, references=None, source_metadata=None):
    """Prepare explicit cached fixture identities; never copy their component values."""
    event_ids = list(event_ids)
    daily = (source_metadata or {}).get("collection") == "daily_update"
    if (not event_ids and not daily) or len(event_ids) != len(set(event_ids)) or any(type(i) is not int or i <= 0 for i in event_ids):
        raise ValueError("Supply distinct positive event IDs")
    if profile.include_comments: raise ValueError("The DevTools campaign does not retain commentary")
    if profile.include_upcoming and not daily: raise ValueError("Upcoming fixtures require a daily-update campaign")
    if daily and validate_policy(policy or {})["collection_mode"] != "session":
        raise ValueError("Daily updates require Chrome session mode")
    items = []
    for eid in event_ids:
        reference = references[eid] if references else reference_from_cache(eid, cache_root)
        event = reference["event"]
        if event["id"] != eid or (not daily and reference.get("kind") != "queue_identity" and not is_completed_event(event)):
            raise ValueError(f"Fixture {eid} must be completed")
        if daily:
            _check_event(event, event, allow_unfinished=True)
        items.append({"event_id":eid, "url":match_url(event), "reference":reference,
                      "season_id":match_season(event)["id"], "season_year":match_season(event)["year"],
                      "competition_id":event["tournament"]["uniqueTournament"]["id"],
                      "label":f"{event['homeTeam']['name']} - {event['awayTeam']['name']}"})
    slug = re.sub(r"[^a-z0-9-]", "-", name.lower()).strip("-")[:40] or "pilot"
    campaign_id = f"{slug}-{uuid.uuid4().hex[:8]}"
    directory = Path(base).resolve() / campaign_id
    directory.mkdir(parents=True)
    config = {"protocol":PROTOCOL_VERSION, "campaign_id":campaign_id, "created_at":time.time(),
              "profile":asdict(profile), "policy":validate_policy(policy or {}), "items":items,
              "scope":"Explicit fixture queue. Does not constitute full-season publication.",
              "source_metadata":source_metadata}
    atomic_write(directory/"config.json", json_bytes(config))
    with Campaign(directory) as campaign:
        campaign.write_status()
    return {"campaign_id":campaign_id, "directory":str(directory), "matches":len(items),
            "event_ids":event_ids, "browser_started":False,"source_metadata":source_metadata}


def request_pause(campaign_id, *, base=BASE):
    path = campaign_path(campaign_id, base)
    if not (path/"config.json").is_file(): raise ValueError("Unknown campaign")
    atomic_write(path/"PAUSE", b"User requested pause")


def campaign_path(campaign_id, base=BASE):
    if not isinstance(campaign_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", campaign_id):
        raise ValueError("Invalid campaign ID")
    return Path(base).resolve() / campaign_id


def status(campaign_id, *, base=BASE):
    return read_json(campaign_path(campaign_id, base)/"status.json")


def update_policy(campaign_id, changes, *, base=BASE, lock_path=None):
    """Change collection mode/pacing/reset controls while no writer owns the lock."""
    path = campaign_path(campaign_id, base)/"config.json"
    if not path.is_file(): raise ValueError("Unknown campaign")
    with ExitStack() as stack:
        try:
            stack.enter_context(collector_lock(lock_path or ROOT/"data/collection_state/collector.lock"))
        except OSError as error:
            raise RuntimeError("Close DevTools and disconnect the collector before updating its saved policy") from error
        config = read_json(path)
        policy = validate_policy({**config["policy"], **changes})
        if policy != config["policy"]:
            atomic_write(path.with_name(f"config.before-policy-{uuid.uuid4().hex}.json"), path.read_bytes())
            config["policy"] = policy
            atomic_write(path, json_bytes(config))
        return policy


@contextmanager
def collector_lock(path):
    """OS lock compatible with the current scraper, without opening its SQLite DB."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if not stream.tell(): stream.write(b"0"); stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: yield
        finally:
            stream.seek(0)
            if os.name == "nt": msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else: fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


from .status_write import atomic_write_status


class Campaign:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.config = read_json(self.directory/"config.json")
        self.profile = Profile(**self.config["profile"])
        self.daily = (self.config.get("source_metadata") or {}).get("collection") == "daily_update"
        self.policy = validate_policy(self.config["policy"])
        self.items = {item["event_id"]:item for item in self.config["items"]}
        self.db = sqlite3.connect(self.directory/"queue.sqlite")
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS queue(event_id INTEGER PRIMARY KEY, ordinal INTEGER, state TEXT,
            lease TEXT, report TEXT, reason TEXT, finished REAL);
          CREATE TABLE IF NOT EXISTS observations(digest TEXT PRIMARY KEY,event_id INTEGER,path TEXT,
            observed REAL,entry_json TEXT);
          CREATE INDEX IF NOT EXISTS observations_event_observed ON observations(event_id,observed);
        """)
        with self.db:
            for index, item in enumerate(self.config["items"]):
                self.db.execute("INSERT OR IGNORE INTO queue VALUES (?,?, 'pending',NULL,NULL,NULL,NULL)", (item["event_id"], index))
        self.current = None
        self.run_observations = 0

    def __enter__(self): return self
    def __exit__(self, *args): self.close()
    def close(self): self.db.close()

    def write_status(self, state=None, reason=None):
        counts=dict(self.db.execute("SELECT state,COUNT(*) FROM queue GROUP BY state"))
        total=sum(counts.values())
        # Large imported queues must not put tens of thousands of rows in every
        # progress response or exceed Chrome's 1 MiB native-message output limit.
        query="SELECT * FROM queue ORDER BY ordinal" if total<=100 else """SELECT * FROM queue
          ORDER BY CASE state WHEN 'running' THEN 0 WHEN 'paused' THEN 1 WHEN 'needs_attention' THEN 2 ELSE 3 END,
                   finished DESC,ordinal LIMIT 100"""
        rows = [dict(row) for row in self.db.execute(query)]
        if state is None:
            state = "ready" if any(counts.get(s,0) for s in ("pending","running","paused")) else "finished_with_gaps" if counts.get("complete",0)!=total else "complete"
        result = {"campaign_id":self.config["campaign_id"], "state":state, "reason":reason,
                  "updated_at":time.time(), "matches":rows, "counts":counts,"total_matches":total,
                  "matches_truncated":total>len(rows),
                  "api_observations_this_connection":self.run_observations,
                  "api_requests_sent_by_python":0}
        atomic_write_status(self.directory/"status.json", json_bytes(result))
        return result

    def next_match(self, *, resume=False):
        if resume:
            (self.directory/"PAUSE").unlink(missing_ok=True)
            with self.db:
                self.db.execute("UPDATE queue SET state='paused' WHERE state='needs_attention'")
        if (self.directory/"PAUSE").exists(): return {"paused":True}
        if self.current: return self.current
        row = self.db.execute("SELECT * FROM queue WHERE state IN ('pending','running','paused') ORDER BY ordinal LIMIT 1").fetchone()
        if row is None: return {"done":True, "status":self.write_status()}
        item = self.items[row["event_id"]]
        previous = self.db.execute("SELECT * FROM queue WHERE ordinal<? AND finished IS NOT NULL ORDER BY ordinal DESC LIMIT 1", (row["ordinal"],)).fetchone()
        wait_until = 0
        if previous:
            prior = self.items[previous["event_id"]]
            if (any(prior[k] != item[k] for k in ("season_id", "competition_id"))
                    or prior["reference"].get("season_stem") != item["reference"].get("season_stem")):
                wait_until = previous["finished"] + self.policy["between_seasons_seconds"]
        lease = uuid.uuid4().hex
        with self.db:
            self.db.execute("UPDATE queue SET state='running',lease=?,reason=NULL WHERE event_id=?", (lease,item["event_id"]))
        self.current = {"event_id":item["event_id"], "lease":lease, "url":item["url"], "label":item["label"],
                        "wait_until":wait_until, "policy":self.policy, "profile":self.config["profile"],
                        "paths":list(self.jobs(item["event_id"]))}
        if self.daily:
            self.current["dynamic_paths"] = True
        self.current["coverage"] = self.coverage(item["event_id"])
        self.write_status("running")
        return self.current

    def jobs(self, event_id):
        event = self.items[event_id]["reference"]["event"]
        metadata = Job(f"/event/{event_id}", "event", event_id)
        if self.daily:
            saved = self.captured_event(event_id)
            if saved is None:
                return {metadata.path: metadata}
            event, observed = saved
            jobs = {job.path: job for job in component_jobs(event, self.profile, observed)}
            return {metadata.path: metadata, **jobs}
        # Plan the requested completed-fixture endpoints, without asserting that
        # a legacy identity is itself a newly observed finished event.
        if self.items[event_id]["reference"].get("kind") == "queue_identity":
            event={**event,"status":{"type":"finished"}}
        jobs = {job.path:job for job in component_jobs(event,self.profile,time.time())}
        jobs[f"/event/{event_id}"] = Job(f"/event/{event_id}","event",event_id)
        return jobs

    def captured_event(self, event_id):
        for row in self.db.execute("SELECT entry_json FROM observations WHERE event_id=? AND path=? ORDER BY observed DESC,rowid DESC",
                                   (event_id, f"/event/{event_id}")):
            entry = json.loads(row[0])
            if entry.get("_xdiyoSelected") and entry.get("response", {}).get("status") in (200, 304):
                try:
                    event = response_json(entry)["event"]
                    self.check_event(event_id, event)
                    return event, response_time(entry)
                except (ValueError, TypeError, KeyError, UnicodeError):
                    continue
        return None

    def verify_lease(self, event_id, lease):
        if not self.current or self.current["event_id"] != event_id or self.current["lease"] != lease:
            raise ValueError("Stale or mismatched match lease")

    def run_limit_reached(self):
        limit = self.policy["max_api_observations_per_run"]
        return limit is not None and self.run_observations >= limit

    def observe(self, event_id, lease, entry):
        self.verify_lease(event_id,lease)
        request = entry.get("request", {})
        url = request.get("url", "")
        endpoint = api_path(url)
        if endpoint is None: raise ValueError("Only Sofascore API observations may be saved")
        observed = response_time(entry)
        parsed = urlsplit(url)
        query = bool(parsed.query or request.get("queryString") or entry.get("_queryPresent"))
        response = entry.get("response", {})
        http_status = response.get("status")
        if type(http_status) is not int or not 0 <= http_status <= 599: raise ValueError("Invalid HTTP status")
        selected = endpoint in self.jobs(event_id) and request.get("method") == "GET" and not query
        clean = {"startedDateTime":entry["startedDateTime"], "time":entry.get("time",0),
                 "request":{"method":request.get("method"), "url":f"https://{parsed.hostname}/api/v1{endpoint}"},
                 "response":{"status":http_status,"content":{}}, "_queryPresent":query,
                 "_xdiyoSelected":selected, "_captureStatus":entry.get("_captureStatus")}
        if selected and self.policy["collection_mode"] == "session" and entry.get("_observationSource") == "chrome_session_request":
            clean["_observationSource"] = "chrome_session_request"
        error = entry.get("_apiError")
        if selected and http_status not in (200,304) and isinstance(error,dict):
            clean["_apiError"] = {key:str(error[key])[:120] for key in ("code","reason")
                                  if isinstance(error.get(key),(str,int)) and not isinstance(error.get(key),bool)}
        content = response.get("content", {})
        if selected and http_status in (200,304):
            for key in ("text","encoding","mimeType"):
                if key in content: clean["response"]["content"][key] = content[key]
            text = clean["response"]["content"].get("text")
            if text is not None and (not isinstance(text,str) or len(text.encode()) > 8*1024*1024):
                raise ValueError("Response body exceeds the capture limit")
        encoded = json_bytes(clean)
        digest = hashlib.sha256(encoded).hexdigest()
        with self.db:
            cursor = self.db.execute("INSERT OR IGNORE INTO observations VALUES (?,?,?,?,?)", (digest,event_id,endpoint,observed,encoded.decode()))
        self.run_observations += cursor.rowcount
        return {"ack":digest, "duplicate":cursor.rowcount == 0,
                "pause_requested":(self.directory/"PAUSE").exists() or self.run_limit_reached(),
                "run_limit_reached":self.run_limit_reached()}

    def entries(self, event_id):
        return [json.loads(row[0]) for row in self.db.execute("SELECT entry_json FROM observations WHERE event_id=? ORDER BY observed,rowid",(event_id,))]

    def coverage(self, event_id):
        jobs = self.jobs(event_id)
        entries = self.entries(event_id)
        latest = {}
        for entry in entries:
            if entry["_xdiyoSelected"]: latest[api_path(entry["request"]["url"])] = entry
        states = []
        metadata = self.captured_event(event_id) if self.daily else None
        for path, job in jobs.items():
            entry = latest.get(path)
            if entry and metadata:
                if job.component != "event" and response_time(entry) < metadata[1]:
                    entry = None  # Never reuse pregame/older components after a new metadata snapshot.
                elif (job.component == "event" and event_status(metadata[0]) != "finished" and
                      time.time() - metadata[1] > self.profile.upcoming_refresh_seconds):
                    entry = None
            state = "not_observed"
            if entry:
                code = entry["response"]["status"]
                if code == 404: state = "unavailable"
                elif code in (401,403): state = "blocked"
                elif code == 429: state = "rate_limited"
                elif code in (200,304):
                    try:
                        payload = response_json(entry)
                        empty = validate_payload(job.component,payload)
                        if job.component == "event": self.check_event(event_id,payload["event"])
                        state = "empty" if empty else "success"
                    except (ValueError,TypeError,KeyError,UnicodeError): state = "invalid_or_missing_body"
                else: state = "failed"
            states.append({"path":path,"component":job.component,"status":state,
                           "http_status":entry["response"]["status"] if entry else None,
                           "observed_at":entry["startedDateTime"] if entry else None})
        identity_ok = any(s["component"] == "event" and s["status"] == "success" for s in states)
        return {"components":states, "complete":identity_ok and all(s["status"] in TERMINAL for s in states),
                "core_complete":identity_ok and all(s["status"] in TERMINAL for s in states if s["component"] != "pregame"),
                "blocked":any(s["status"] in {"blocked","rate_limited"} for s in states),
                "observations":len(entries)}

    def progress(self, event_id, lease):
        self.verify_lease(event_id,lease)
        coverage=self.coverage(event_id)
        self.write_status("running")
        return {**coverage, "pause_requested":(self.directory/"PAUSE").exists(),
                "run_limit_reached":self.run_limit_reached()}

    def awarded_website_deferral(self, event_id, coverage, page_evidence):
        """Keep missing data unresolved when an awarded fixture has no site views.

        This is a queue disposition, not evidence of HTTP 404 or endpoint
        unavailability. Only newly captured metadata can establish isAwarded.
        """
        if not isinstance(page_evidence,dict) or page_evidence.get("recovery_refreshed") is not True:
            return None
        views = page_evidence.get("available_views")
        allowed = {"lineups","statistics","standings","matches","media","ai-insights"}
        if not isinstance(views,list) or not views or any(not isinstance(v,str) or v not in allowed for v in views):
            return None
        if not set(views) & {"standings","matches"}: return None
        missing = [state for state in coverage["components"] if state["status"] not in TERMINAL]
        if not missing or any(state["status"] != "not_observed" for state in missing): return None
        view_for = {"statistics":"statistics","shotmap":"statistics","heatmap":"statistics","lineups":"lineups"}
        if any(state["component"] not in view_for or view_for[state["component"]] in views for state in missing):
            return None
        metadata = [entry for entry in self.entries(event_id) if entry["_xdiyoSelected"] and
                    api_path(entry["request"]["url"]) == f"/event/{event_id}"]
        if not metadata or metadata[-1]["response"]["status"] not in (200,304): return None
        try:
            event = response_json(metadata[-1])["event"]
            self.check_event(event_id,event)
        except (ValueError,TypeError,KeyError,UnicodeError):
            return None
        if event.get("isAwarded") is not True: return None
        return {"reason":"awarded_fixture_without_requested_website_views",
                "available_views":sorted(set(views)), "recovery_refreshed":True,
                "isAwarded":True, "metadata_observed_at":metadata[-1]["startedDateTime"],
                "missing_paths":[state["path"] for state in missing],
                "note":"Deferred for later review; missing responses remain not_observed, not complete or HTTP unavailable."}

    def finish(self, event_id, lease, *, page_event_id, reason="views_exhausted", page_evidence=None):
        self.verify_lease(event_id,lease)
        coverage = self.coverage(event_id)
        session_mode = self.policy["collection_mode"] == "session"
        if session_mode:
            if not isinstance(page_evidence,dict) or page_evidence.get("collection_mode") != "session":
                raise ValueError("Session collection evidence is required")
            # Identity comes from the captured metadata, not a caller-supplied ID
            # or the different anchor match displayed in the collection tab.
            if not coverage["complete"]:
                raise ValueError("Every session component and fixture identity must be verified before publication")
        elif page_event_id != event_id:
            raise ValueError("Displayed match identity did not match the queue")
        if coverage["blocked"]: return self.pause("provider_block")
        har = {"log":{"version":"1.2","creator":{"name":"xDiyo automated DevTools capture","version":"0.2.21"},"entries":self.entries(event_id)}}
        path = self.directory/"archives"/f"{event_id}.har"
        atomic_write(path,json_bytes(har))
        reference=self.items[event_id]["reference"]
        if reference.get("kind") == "queue_identity" or self.daily:
            metadata=[e for e in self.entries(event_id) if e["_xdiyoSelected"] and api_path(e["request"]["url"])==f"/event/{event_id}"]
            if not metadata or metadata[-1]["response"]["status"] not in (200,304):
                raise ValueError("Fresh event metadata is required; legacy routing identities cannot populate match data")
            event=response_json(metadata[-1])["event"]
            self.check_event(event_id,event)
            reference={"event":event,"observed_at":response_time(metadata[-1]),"source":"new Chrome event response checked against original queue identity"}
        if self.daily:
            # A resumed unfinished fixture may have crossed kickoff or finished.
            # Only components from this current metadata snapshot enter its import.
            current_entries = [entry for entry in self.entries(event_id)
                               if response_time(entry) >= reference["observed_at"]]
            atomic_write(path, json_bytes({"log": {**har["log"], "entries": current_entries}}))
        report = import_har(path,event_id=event_id,reference=reference,
                            profile=self.profile,output_root=self.directory/"imports")
        features = assess_features(read_tables(report["manifest"]),self.profile)
        atomic_write(Path(report["directory"])/"feature_coverage.json",json_bytes(features))
        state = "complete" if report["complete"] else "partial" if coverage["core_complete"] else "needs_attention"
        report_path = str(Path(report["directory"])/"report.json")
        deferral = self.awarded_website_deferral(event_id,coverage,page_evidence) if state == "needs_attention" else None
        if isinstance(page_evidence,dict):
            views = page_evidence.get("available_views")
            report["page_evidence"] = {
                "available_views":[v[:128] for v in views[:40] if isinstance(v,str)] if isinstance(views,list) else None,
                "recovery_refreshed":page_evidence.get("recovery_refreshed") is True,
                "memory_reset_before_match":page_evidence.get("memory_reset_before_match") is True,
            }
            if session_mode:
                request_count = page_evidence.get("session_request_count")
                report["page_evidence"].update(collection_mode="session", identity_source="captured_event_metadata",
                    session_renewed_before_match=page_evidence.get("session_renewed_before_match") is True,
                    session_request_count=request_count if type(request_count) is int and 0 <= request_count <= self.policy["max_api_observations_per_match"] else None)
                for name, maximum in (
                    ("session_renewal_count", 2), ("session_challenge_recoveries", 1),
                    ("session_recovery_wait_ms", math.ceil(self.policy["session_challenge_cooldown_seconds"]*1000)),
                    ("session_expiry_wait_ms", 2*math.ceil(self.policy["match_timeout_seconds"]*1000+21000)),
                    ("session_pacing_requested_wait_ms", math.ceil(self.policy["match_timeout_seconds"]*1000)),
                    ("session_pacing_actual_wait_ms", math.ceil(self.policy["match_timeout_seconds"]*1000)),
                ):
                    value = page_evidence.get(name, 0)
                    report["page_evidence"][name] = value if type(value) is int and 0 <= value <= maximum else None
                backend = page_evidence.get("session_pacing_backend")
                report["page_evidence"]["session_pacing_backend"] = backend if backend in ("native", "browser") else None
                # Measured rest can exceed its requested duration after a local
                # scheduling stall. Missing measurements from older versions
                # remain unknown; do not turn them into zero-duration evidence.
                for name in ("session_recovery_actual_wait_ms", "session_expiry_actual_wait_ms",
                             "session_pacing_native_waited_ms", "session_pacing_native_roundtrip_ms",
                             "session_pacing_progress_ms"):
                    value = page_evidence.get(name)
                    report["page_evidence"][name] = value if type(value) is int and 0 <= value <= 9_007_199_254_740_991 else None
            if "historical_link_recovery" in page_evidence:
                report["page_evidence"]["historical_link_recovery"] = page_evidence["historical_link_recovery"] is True
            frame_waits = page_evidence.get("navigation_frame_waits_total")
            if type(frame_waits) is int and 0 <= frame_waits <= 1_000_000:
                report["page_evidence"]["navigation_frame_waits_total"] = frame_waits
        if deferral:
            state, reason = "deferred", deferral["reason"]
            report["collection_disposition"] = {"state":state,**deferral}
        if deferral or isinstance(page_evidence,dict):
            atomic_write(Path(report_path),json_bytes(report))
        with self.db:
            self.db.execute("UPDATE queue SET state=?,report=?,reason=?,finished=? WHERE event_id=?",(state,report_path,reason,time.time(),event_id))
        self.current = None
        stop = state == "needs_attention" and self.policy["pause_on_missing_core"]
        if stop: atomic_write(self.directory/"PAUSE",b"Missing core match data")
        self.write_status("paused" if stop else None,"missing_core_data" if stop else None)
        return {"state":state,"pause_requested":stop,"report":report_path,"feature_coverage":features,
                "deferral":deferral}

    def check_event(self,event_id,event):
        expected=self.items[event_id]["reference"]["event"]
        _check_event(event,expected,allow_unfinished=self.daily)
        if season_year_pair(match_season(event).get("year")) != season_year_pair(match_season(expected).get("year")):
            raise ValueError("Captured event season year conflicts with queue identity")

    def pause(self, reason="user_paused"):
        if self.current:
            with self.db:
                self.db.execute("UPDATE queue SET state='paused',reason=? WHERE event_id=?", (reason,self.current["event_id"]))
        atomic_write(self.directory/"PAUSE", str(reason).encode())
        self.current = None
        return {"paused":True,"status":self.write_status("paused",reason)}
