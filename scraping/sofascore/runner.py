from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
import random

from .discovery import event_status, round_jobs, unresolved_jobs
from .models import CollectionPaused, Job
from .parsers import validate_payload
from .planner import build_plan
from .rate_limit import RateLimiter, retry_after_seconds


class Runner:
    """Serial run owner. Every attempt and endpoint result is checkpointed."""

    def __init__(self, store, client, policy, profile, *, clock=None):
        self.store, self.client, self.policy, self.profile = store, client, policy, profile
        self.clock = clock

    def _next_due(self, job, event, now, status):
        if status == "unavailable":
            return now + (self.profile.negative_cache_seconds if event and event_status(event)=="finished"
                          else self.profile.upcoming_refresh_seconds)
        if job.component in {"event", "discovery"}:
            return now + 86400
        if event and event_status(event) != "finished":
            return now + self.profile.upcoming_refresh_seconds
        kickoff = (event or {}).get("startTimestamp")
        if kickoff and now < kickoff + self.profile.settling_window_hours*3600:
            return kickoff + self.profile.settling_window_hours*3600 + self.profile.settling_delay_seconds
        return None

    def _fetch(self, job, event=None):
        limiter = self.limiter
        for retry in range(self.policy.max_attempts_per_job):
            attempt = limiter.reserve(job)
            try:
                response = self.client.fetch(job)
            except Exception as exc:
                # Browser startup/programming failures should be visible, not retried as noncoverage.
                limiter.complete(attempt, None, "client_exception")
                self.store.record(job, "failed", limiter.clock.time(), error=type(exc).__name__)
                raise
            now = limiter.clock.time()
            if response.additional_api_requests:
                with self.store.db:
                    for _ in range(response.additional_api_requests):
                        self.store.db.execute("INSERT INTO attempts(run_id,key,started,outcome) VALUES (?,?,?,?)",
                            (limiter.run_id,job.key,now,"unexpected_browser_api_request"))
                self.store.set_setting("access_block", {"at":now,"reason":"unexpected_browser_api_requests"})
                limiter.complete(attempt, response.status, "unexpected_browser_api_requests")
                self.store.record(job, "blocked", now, http_status=response.status,
                                  error="Browser issued extra API requests; inspect before resuming")
                raise CollectionPaused("Unexpected browser API traffic was counted; run paused for inspection")
            if response.status in {401,403} or response.error in {"non_json_response","blocked_by_client"}:
                advertised = retry_after_seconds(response.headers.get("retry-after"), now)
                if advertised is not None:
                    limiter.cooldown(advertised)
                self.store.set_setting("access_block", {"at":now,"status":response.status,"reason":response.error})
                limiter.complete(attempt,response.status,"blocked")
                self.store.record(job,"blocked",now,http_status=response.status,error=response.error or "access_denied")
                raise CollectionPaused("Access denied or non-JSON page; the shared collector is paused")
            if response.status == 429:
                seconds = retry_after_seconds(response.headers.get("retry-after"), now)
                limiter.cooldown(max(seconds or 0,self.policy.default_cooldown_seconds if seconds is None else 0))
                limiter.complete(attempt,429,"rate_limited")
                self.store.record(job,"rate_limited",now,http_status=429,
                                  next_due=self.store.setting("cooldown_until"))
                raise CollectionPaused("Rate limited; persisted Retry-After cooldown and stopped")
            if response.status == 404:
                limiter.complete(attempt,404,"unavailable")
                self.store.record(job,"unavailable",now,http_status=404,
                                  next_due=self._next_due(job,event,now,"unavailable"))
                return "unavailable"
            error = response.error
            if response.status == 200 and not error:
                try:
                    empty = validate_payload(job.component,response.payload)
                except (ValueError, TypeError, KeyError) as exc:
                    error = f"schema_error:{exc}"
                    diagnostic = self.store.archive_diagnostic(job,response.payload,now,error)
                    error += f";diagnostic:{diagnostic}"
                else:
                    status = "empty" if empty else "success"
                    self.store.record(job,status,now,payload=response.payload,http_status=200,
                                      next_due=self._next_due(job,event,now,status),event_context=event)
                    limiter.complete(attempt,200,status)
                    return status
            retryable = response.status is None or response.status in {408,500,502,503,504} or (
                response.error in {"navigation_timeout","navigation_failed","unmatched_response","invalid_response_body"})
            error = error or f"http_{response.status}"
            limiter.complete(attempt,response.status,"failed")
            self.store.record(job,"failed",now,http_status=response.status,error=error)
            if not retryable or retry+1 == self.policy.max_attempts_per_job:
                return "failed"
            delay = min(self.policy.max_backoff_seconds,self.policy.backoff_seconds*(2**retry))
            advertised = retry_after_seconds(response.headers.get("retry-after"),now)
            if advertised is not None:
                limiter.cooldown(advertised)
                raise CollectionPaused("Server requested a retry delay; cooldown checkpointed")
            limiter.sleep(delay + random.uniform(0,delay*.1))
        return "failed"

    def run(self, scope, *, discover=True, discovery_refresh=False, collect=True,
            event_ids=None, refresh_components=(), only_components=None, pilot=False):
        summary = {"outcomes":{},"planned_requests":0,"cache_hits":0,"limitations":[]}
        summary.update(scope=asdict(scope),profile=asdict(self.profile))
        counts = Counter()
        with self.store.run_lock():
            if pilot:
                if not event_ids or len(event_ids)>8:
                    raise ValueError("Pilot requires one to eight explicitly selected event IDs")
                cohort=set(self.store.setting("pilot_event_ids",[])) | set(event_ids)
                if len(cohort)>8:
                    raise ValueError("The cumulative pilot cohort cannot exceed eight events")
                self.store.set_setting("pilot_event_ids",sorted(cohort))
                prior=self.store.setting("pilot_run_ids",[])
                used=sum(self.store.db.execute("SELECT COUNT(*) FROM attempts WHERE run_id=?",(rid,)).fetchone()[0] for rid in prior)
                if used>=40:
                    raise CollectionPaused("The cumulative capability pilot has used its 40-attempt allowance")
                self.policy=replace(self.policy,max_attempts_per_run=min(self.policy.max_attempts_per_run or 40,40-used))
            run_id = self.store.start_run(self.clock.time() if self.clock else None)
            if pilot:
                self.store.set_setting("pilot_run_ids",[*prior,run_id])
            self.limiter = RateLimiter(self.store,self.policy,run_id,self.clock)
            started = self.limiter.clock.time()
            summary["rate_policy"] = asdict(self.policy)
            status = "failed"
            try:
                if pilot:
                    for eid in sorted(event_ids):
                        job=Job(f"/event/{eid}","event",eid,locale=self.profile.locale,reasons=("pilot_event_discovery",))
                        if self.store.due(job,self.limiter.clock.time(),discovery_refresh):
                            counts[self._fetch(job)] += 1
                        payload=self.store.payload(job)
                        if payload:
                            if payload["event"].get("id")!=eid:
                                raise ValueError("Event detail ID does not match the selected pilot event")
                            self.store.add_events([payload["event"]],scope,job,self.store.get(job)["last_success"])
                elif discover:
                    for job in round_jobs(scope,self.profile.locale):
                        if self.store.due(job,self.limiter.clock.time(),discovery_refresh):
                            counts[self._fetch(job)] += 1
                        payload = self.store.payload(job)
                        if payload:
                            cached = self.store.get(job)
                            self.store.add_events(payload["events"],scope,job,cached["last_success"])
                    for job in unresolved_jobs(self.store,scope,self.profile,self.limiter.clock.time()):
                        counts[self._fetch(job)] += 1
                        payload = self.store.payload(job)
                        if payload:
                            self.store.add_events([payload["event"]],scope,job,self.store.get(job)["last_success"])
                events = self.store.events(scope.league_id,scope.season_id)
                plan = build_plan(events,self.profile,self.store,self.limiter.clock.time(),
                                  refresh_components=refresh_components,event_ids=event_ids,only_components=only_components)
                summary.update({k:v for k,v in plan.as_dict().items() if k != "jobs"})
                summary["component_selection"]=sorted(only_components) if only_components is not None else "profile"
                if collect:
                    by_id = {e["id"]:e for e in events}
                    for job in plan.jobs:
                        counts[self._fetch(job,by_id[job.event_id])] += 1
                status = "incomplete" if counts["failed"] or plan.limitations or plan.deferred else "complete"
            except CollectionPaused as exc:
                status = "paused"
                summary["pause_reason"] = str(exc)
            except BaseException:
                status = "interrupted"
                raise
            finally:
                try:
                    self.client.close()
                except Exception as exc:
                    summary["cleanup_error"]=type(exc).__name__
                    status="failed"
                summary["outcomes"] = dict(counts)
                summary["attempts"] = self.store.db.execute("SELECT COUNT(*) FROM attempts WHERE run_id=?",(run_id,)).fetchone()[0]
                summary["attempts_by_component"] = dict(self.store.db.execute(
                    "SELECT COALESCE(c.component,'unrecorded'),COUNT(*) FROM attempts a LEFT JOIN components c ON a.key=c.key WHERE a.run_id=? GROUP BY c.component",(run_id,)).fetchall())
                summary["elapsed_seconds"] = self.limiter.clock.time()-started
                summary["run_id"],summary["status"] = run_id,status
                self.store.finish_run(run_id,status,summary,self.limiter.clock.time())
        return summary
