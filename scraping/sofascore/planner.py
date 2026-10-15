from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .discovery import event_status
from .models import Job


def component_jobs(event, profile, now):
    eid = event["id"]
    status = event_status(event)
    finished = status == "finished"
    kickoff = event.get("startTimestamp")
    upcoming = profile.include_upcoming and status == "notstarted" and kickoff is not None and (
        0 <= kickoff-now <= profile.lineup_window_hours*3600)
    if profile.include_upcoming and status in {"inprogress", "interrupted"}:
        upcoming = True  # Capture available lineups with an explicit in-progress phase.
    if not finished and not upcoming:
        return []
    jobs = []

    def add(endpoint, component, reasons=(), team_id=None):
        jobs.append(Job(f"/event/{eid}/{endpoint}", component, eid, team_id, profile.locale, tuple(reasons)))

    if profile.include_pregame and status not in {"inprogress", "interrupted"}:
        add("pregame-form", "pregame", ("match_specific_pregame_positions",))
    if finished and profile.include_statistics:
        add("statistics", "statistics")
    if finished and profile.include_shotmap:
        add("shotmap", "shotmap", ("individual_shots_and_xg",))
    if finished and profile.include_heatmaps:
        for side in ("home", "away"):
            team_id = (event.get(side+"Team") or {}).get("id")
            if team_id is None:
                raise ValueError(f"Event {eid} has no {side} team ID for heatmap collection")
            add(f"heatmap/{team_id}", "heatmap", (side,), team_id)
    reasons = []
    if profile.include_lineups:
        reasons.append("lineups")
    if profile.include_injuries:
        reasons.append("availability")
    if reasons:
        add("lineups", "lineups", reasons)
    reasons = []
    if profile.duration_mode == "incidents":
        reasons.append("timing_evidence")
    if profile.include_injuries and profile.injury_scope == "availability_and_match_incidents":
        reasons.append("injury_incidents")
    if profile.include_goal_actions:
        reasons.append("bundled_goal_action_sequences")
    if finished and reasons:
        add("incidents", "incidents", reasons)
    if finished and profile.include_comments:
        add("comments", "comments")
    return jobs


@dataclass
class Plan:
    jobs: list[Job] = field(default_factory=list)
    cache_hits: int = 0
    deferred: int = 0
    event_count: int = 0
    limitations: list[str] = field(default_factory=list)

    def as_dict(self):
        return {"events":self.event_count, "planned_requests":len(self.jobs),
                "by_component":dict(Counter(j.component for j in self.jobs)),
                "cache_hits":self.cache_hits,"deferred":self.deferred, "limitations":self.limitations,
                "jobs":[{"key":j.key,"path":j.path,"component":j.component,"reasons":j.reasons} for j in self.jobs]}


def build_plan(events, profile, store, now, *, refresh_components=(), event_ids=None, only_components=None):
    # Latest duplicate object wins; discovery retains separate membership references.
    selected = {e["id"]:e for e in events if event_ids is None or e["id"] in event_ids}
    plan = Plan(event_count=len(selected))
    if event_ids is not None and set(event_ids)-selected.keys():
        plan.limitations.append(f"Selected event metadata unavailable: {sorted(set(event_ids)-selected.keys())}")
    for event in sorted(selected.values(), key=lambda e:(e.get("startTimestamp") or float("inf"),e["id"])):
        for job in component_jobs(event, profile, now):
            if only_components is not None and job.component not in only_components:
                continue
            if store.due(job, now, job.component in refresh_components):
                plan.jobs.append(job)
            else:
                state=store.get(job)
                if state and state["status"] in {"success","empty","unavailable"}:
                    plan.cache_hits += 1
                else:
                    plan.deferred += 1
    return plan
