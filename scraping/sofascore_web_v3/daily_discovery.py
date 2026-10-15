"""Discover played history plus the current/next regular round, per league."""
from __future__ import annotations

import json
import time

from scraping.sofascore.config import Profile
from scraping.sofascore.discovery import event_status
from scraping.sofascore.store import atomic_write, json_bytes, season_year_pair
from .campaign import BASE, prepare_campaign
from .discovery import Discovery, positive, round_numbers, select_regular_records

SETTLED = {"finished", "awarded", "cancelled", "abandoned"}
UNFINISHED = {"notstarted", "postponed", "inprogress", "interrupted"}


def selected_records(rows):
    """Resolve one scheduled replacement too, without joining distinct fixtures."""
    selected, superseded = select_regular_records(rows)
    groups = {}
    for row in selected:
        event = json.loads(row["payload"])
        key = (row["competition"], row["season"], row["round"], event["homeTeam"]["id"], event["awayTeam"]["id"])
        groups.setdefault(key, []).append((row, event))
    omitted = set()
    for values in groups.values():
        active = [(r, e) for r, e in values if event_status(e) in {"notstarted", "inprogress", "finished"}]
        if len(active) != 1:
            continue
        replacement, current = active[0]
        for row, event in values:
            if event_status(event) in {"postponed", "abandoned"} and event["startTimestamp"] < current["startTimestamp"]:
                omitted.add(row["id"])
                superseded.append({"event_id": row["id"], "replacement_event_id": replacement["id"], "status": event_status(event)})
    return [row for row in selected if row["id"] not in omitted], superseded


class DailyDiscovery(Discovery):
    def put(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO state VALUES(?,?)", (key, json.dumps(value)))

    def get(self, key, default=None):
        row = self.db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def process(self, job, payload, observed):
        cid, sid, kind = job["competition"], job["season"], job["kind"]
        if kind == "seasons":
            values = payload.get("seasons")
            if not isinstance(values, list):
                raise ValueError("Missing season list")
            found = [s for s in values if isinstance(s, dict) and season_year_pair(s.get("year")) == tuple(self.config["years"])]
            if not found:
                self.put(f"availability:{cid}", "season_not_listed")
                return
            if len(found) != 1:
                raise ValueError("Requested season is ambiguous")
            sid = positive(found[0].get("id"))
            self.put(f"availability:{cid}", "available")
            self.add_job(f"/unique-tournament/{cid}/season/{sid}/rounds", cid, "rounds", sid)
            self.add_job(f"/unique-tournament/{cid}/season/{sid}/standings/total", cid, "standings", sid)
        elif kind == "rounds":
            if payload.get("rounds") == []:
                self.put(f"availability:{cid}", "schedule_not_listed")
                return
            numbers = round_numbers(payload)
            anchor = (payload.get("currentRound") or {}).get("round")
            if type(anchor) is not int:
                anchor = numbers[0]
            anchor = min(max(numbers[0], anchor), numbers[-1])
            self.put(f"rounds:{cid}", numbers)
            self.put(f"anchor:{cid}", anchor)
            # Past rounds are refreshed as compact inventories, not per-match
            # components. This also rediscovers old postponements and corrections.
            for number in numbers:
                if number <= anchor:
                    self.add_job(f"/unique-tournament/{cid}/season/{sid}/events/round/{number}", cid, "events", sid, number)
        elif kind == "standings" and payload.get("standings") == []:
            self.put(f"availability:{cid}", "standings_not_listed")
        else:
            super().process(job, payload, observed)
            if kind != "events" or job["round"] < self.get(f"anchor:{cid}", 1):
                return
            rows, _ = selected_records(self.db.execute("SELECT * FROM events WHERE competition=? AND round=?", (cid, job["round"])))
            events = [json.loads(row["payload"]) for row in rows]
            # An old postponed fixture does not hold back the next scheduled
            # round. Its identity stays in the inventory and is revisited next run.
            can_advance = all(event_status(e) in SETTLED or
                (event_status(e) == "postponed" and e["startTimestamp"] <= observed) for e in events)
            if can_advance:
                following = [n for n in self.get(f"rounds:{cid}", []) if n > job["round"]]
                if following:
                    number = following[0]
                    self.add_job(f"/unique-tournament/{cid}/season/{sid}/events/round/{number}", cid, "events", sid, number)

    def season_selection(self, cid):
        rows = list(self.db.execute("SELECT * FROM events WHERE competition=?", (cid,)))
        records, superseded = selected_records(rows)
        events = [json.loads(row["payload"]) for row in records]
        teams = set(self.get(f"teams:{cid}", []))
        jobs = list(self.db.execute("SELECT * FROM jobs WHERE competition=?", (cid,)))
        per_round = {}
        for event in events:
            per_round.setdefault(event["roundInfo"]["round"], []).append(event)
        issues = []
        for number, fixtures in per_round.items():
            participants = [e[side]["id"] for e in fixtures for side in ("homeTeam", "awayTeam")]
            if (len(teams) < 2 or len(fixtures) != len(teams) // 2 or
                    len(participants) != len(set(participants)) or not set(participants) <= teams):
                issues.append(f"Round {number} has incomplete or conflicting team coverage")
        if any(event_status(e) not in SETTLED | UNFINISHED for e in events):
            issues.append("Unknown match status")
        anchor = self.get(f"anchor:{cid}", 1)
        open_rounds = [n for n, fixtures in per_round.items() if n >= anchor and
                       any(event_status(e) in UNFINISHED and not
                           (event_status(e) == "postponed" and e["startTimestamp"] <= time.time()) for e in fixtures)]
        target = min(open_rounds) if open_rounds else None
        finished = [e for e in events if e.get("status", {}).get("type") == "finished"]
        unfinished = [e for e in events if event_status(e) in UNFINISHED and
                      (target is None or e["roundInfo"]["round"] <= target)]
        wanted = {e["id"] for e in finished + unfinished}
        selected = [r for r in records if r["id"] in wanted]
        selected.sort(key=lambda r: (json.loads(r["payload"])["startTimestamp"], r["id"]))
        completed_rounds = [n for n, fixtures in per_round.items() if all(event_status(e) in SETTLED for e in fixtures)]
        season_job = next((job for job in jobs if job["kind"] == "rounds"), None)
        availability = self.get(f"availability:{cid}", "discovering")
        complete = bool(jobs) and all(job["state"] == "complete" for job in jobs)
        if not complete:
            state = "discovering"
        elif availability != "available":
            state = availability
        elif issues:
            state = "needs_review"
        elif target is not None:
            state = "round_ready"
        elif unfinished:
            state = "waiting_for_rescheduled_matches"
        else:
            state = "season_complete"
        summary = {"league": self.leagues[cid]["name"], "competition_id": cid,
            "season_id": season_job["season"] if season_job else None, "state": state,
            "last_completed_round": max(completed_rounds, default=None), "target_round": target,
            "history_matches": len(finished), "unfinished_matches": len(unfinished),
            "in_progress": sum(event_status(e) in {"inprogress", "interrupted"} for e in unfinished),
            "prediction_candidates": sum(event_status(e) == "notstarted" and e["startTimestamp"] > time.time() for e in unfinished),
            "enumerated_rounds": sorted(per_round), "excluded_stage_matches": sum(not r["regular"] for r in rows),
            "non_collectable_terminal_matches": [{"event_id": e["id"], "status": event_status(e)} for e in events
                if e.get("status", {}).get("type") != "finished" and event_status(e) in SETTLED],
            "superseded_fixtures": superseded, "issues": issues}
        return summary, selected

    def status(self):
        counts = dict(self.db.execute("SELECT state,COUNT(*) FROM jobs GROUP BY state"))
        seasons = [self.season_selection(cid)[0] for cid in self.leagues]
        complete = bool(counts) and set(counts) == {"complete"}
        return {"discovery_id": self.config["discovery_id"], "directory": str(self.directory), "mode": "daily",
            "jobs": counts, "requests": self.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0],
            "discovery_complete": complete, "ready_to_collect": complete and not any(s["issues"] for s in seasons),
            "seasons": seasons, "campaign": self.get("campaign"), "api_requests_sent_by_python": 0}

    def prepare_collection(self, *, campaign_base=BASE):
        snapshot = self.status()
        if snapshot["campaign"]:
            return snapshot["campaign"]
        if not snapshot["ready_to_collect"]:
            raise ValueError("Finish daily discovery and review the per-round coverage first")
        from .daily_storage import plan_reuse
        years = self.config["years"]
        cohorts, references, queued = [], {}, []
        for cid, league in self.leagues.items():
            summary, records = self.season_selection(cid)
            stem = f"{league['name']}_{years[0]%100:02d}_{years[1]%100:02d}"
            events = [json.loads(r["payload"]) for r in records]
            reuse = plan_reuse(stem, events, self.config)
            by_id = {r["id"]: r for r in records}
            for event in events:
                eid = event["id"]
                if eid in reuse["reused_ids"]:
                    continue
                references[eid] = {"kind": "daily_identity", "event": event,
                    "observed_at": by_id[eid]["observed"], "source": str(self.directory / "discovery.sqlite"),
                    "season_stem": stem, "route_source": "chrome_daily_discovery"}
                queued.append(eid)
            cohorts.append({**summary, "stem": stem, "event_ids": [e["id"] for e in events], **reuse})
        atomic_write(self.directory / "cohort.json", json_bytes({"discovery_id": self.config["discovery_id"],
            "created_at": time.time(), "profile": self.config["profile"], "years": years, "seasons": cohorts,
            "queued_ids": queued, "output_root": self.config["output_root"]}))
        result = prepare_campaign(queued, references=references, base=campaign_base,
            name=f"daily-{years[0]}-{years[1]}", profile=Profile(**self.config["profile"]), policy=self.config["policy"],
            source_metadata={"collection": "daily_update", "discovery_id": self.config["discovery_id"],
                             "discovery_directory": str(self.directory), "output_root": self.config["output_root"]})
        result.pop("event_ids", None)
        result["reused_matches"] = sum(len(s["reused_ids"]) for s in cohorts)
        result["output_root"] = self.config["output_root"]
        with self.db:
            self.put("campaign", result)
        return result
