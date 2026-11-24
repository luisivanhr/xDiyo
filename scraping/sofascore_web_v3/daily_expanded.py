"""Mixed-season daily discovery using the existing Chrome-only protocol.

Enumerate compact regular-season inventories, reconcile played counts against
the corresponding standings, then queue history and a dated fixture window.
No archive settings, raw observations, or running queues are modified.
"""
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import re
import time
import uuid

from scraping.sofascore.discovery import event_status
from scraping.sofascore.event_round import match_round, match_season
from scraping.sofascore.store import atomic_write, json_bytes, season_year_pair
from .archive_formats import EXCLUDED, excluded_stage
from .campaign import campaign_path, validate_policy
from .capture import match_url
from .daily_discovery import DailyDiscovery, SETTLED, UNFINISHED, selected_records
from .discovery import positive

POST_STAGE = re.compile(EXCLUDED.pattern + r"|\b(?:hexagonal|cuadrangular|quadrangular)\b|"
    r"\b(?:round[ -]+of[ -]+(?:16|32|64)|last[ -]+(?:16|32|64)|knockout)\b|"
    r"\b(?:\d+(?:st|nd|rd|th)[ -]+\d+(?:st|nd|rd|th)[ -]+places?|championship[ -]+play)", re.I)
PHASE = re.compile(r"\b(apertura|clausura|finalizaci[oó]n)\b", re.I)
GROUP = re.compile(r"\b(?:group|zone|zona)[ -]?[ab]\b", re.I)
CONFERENCE = re.compile(r"\b(?:eastern?|western?)[ -]+conference\b", re.I)


def labels(value):
    return " ".join(str(value.get(k, "")) for k in ("name", "slug", "prefix"))


def phase(value, spec):
    """Keep provider rounds unchanged; distinguish ordinary short tournaments."""
    if spec["schedule"] != "stages":
        return "regular"
    match = PHASE.search(labels(value.get("tournament", value)) + " " + labels(value.get("roundInfo", {})))
    if not match:
        return "regular"
    return "clausura" if match[1].lower().startswith(("claus", "final")) else "apertura"


def regular(value, spec):
    tournament = value.get("tournament", value)
    text = labels(tournament) + " " + labels(value.get("roundInfo", {}))
    if POST_STAGE.search(text) or excluded_stage(value):
        return False
    # Colombian named groups are the second phase (cuadrangulares), not the
    # ordinary Apertura/Finalizacion round robin. Peru's normal halves do not
    # have subsidiary groups either. Keep their source bodies, exclude selection.
    if spec["competition_id"] in {11539, 406} and GROUP.search(text):
        return False
    group = tournament.get("isGroup")
    if type(group) is not bool:
        raise ValueError("Missing tournament stage flag")
    if group:
        if spec["competition_id"] == 242 and CONFERENCE.search(text):
            return True
        if spec["competition_id"] == 155 and GROUP.search(text) and PHASE.search(text):
            return True
        # Ordinary short-tournament tables may themselves carry isGroup.
        if spec["competition_id"] in {11539, 406} and PHASE.search(text) and not GROUP.search(text):
            return True
        raise ValueError("Unreviewed grouped stage; retain evidence before selecting fixtures")
    return True


def prepare_expanded_daily(*, years, leagues, profile, random_delay_max_seconds,
                           history_root, output_root, base, fixture_window_days):
    if type(fixture_window_days) is not int or not 1 <= fixture_window_days <= 60:
        raise ValueError("fixture_window_days must be between 1 and 60")
    policy = validate_policy({"collection_mode": "session", "memory_reset_every_matches": 50,
        "session_challenge_recovery": True, "random_action_delay_max_seconds": random_delay_max_seconds})
    discovery_id = f"discover-daily-{years[0]}-{years[1]}-{uuid.uuid4().hex[:8]}"
    directory = campaign_path(discovery_id, base)
    config = dict(discovery_id=discovery_id, years=years, leagues=leagues, mode="daily", daily_schema=2,
        created_at=time.time(), scope="regular_season", profile=asdict(profile), policy=policy,
        fixture_window_days=fixture_window_days, history_root=str(Path(history_root).resolve()),
        output_root=str(Path(output_root).resolve()))
    atomic_write(directory / "config.json", json_bytes(config))
    with ExpandedDailyDiscovery(directory) as discovery:
        return discovery.status()


class ExpandedDailyDiscovery(DailyDiscovery):
    def __init__(self, directory, *, read_only=False):
        super().__init__(directory, read_only=read_only)
        self.leagues = {s["competition_id"]: s for s in self.config["leagues"]}
        self._daily_summaries = {}

    def round_limit(self, cid):
        spec = self.leagues[cid]
        if spec["schedule"] != "split":
            return None
        teams = self.get(f"teams:{cid}", [])
        # Current regular stage before a split: three meetings in Scotland and
        # Switzerland, one in Egypt, two in the other selected split leagues.
        legs = 3 if cid in {36, 215} else 1 if cid == 808 else 2
        return legs * (len(teams) - 1) if teams else None

    def process(self, job, payload, observed):
        cid, sid, kind = job["competition"], job["season"], job["kind"]
        self._daily_summaries.pop(cid, None)
        spec = self.leagues[cid]
        if kind == "seasons":
            values = payload.get("seasons")
            if not isinstance(values, list):
                raise ValueError("Missing season list")
            found = [s for s in values if isinstance(s, dict) and season_year_pair(s.get("year")) == tuple(spec["years"])]
            if not found:
                self.put(f"availability:{cid}", "season_not_listed")
                return
            if len(found) != 1:
                raise ValueError("Requested season is ambiguous; no latest-season fallback")
            sid = positive(found[0].get("id"))
            self.put(f"season:{cid}", sid)
            self.put(f"availability:{cid}", "available")
            # Standings first: establish regular participants and pre-split limit.
            self.add_job(f"/unique-tournament/{cid}/season/{sid}/standings/total", cid, "standings", sid)
            return
        if kind == "standings":
            values = payload.get("standings")
            if not isinstance(values, list):
                raise ValueError("Missing standings inventory")
            if not values:
                self.put(f"availability:{cid}", "standings_not_listed")
                return
            tables = defaultdict(dict)
            for table in values:
                if table.get("type") != "total" or not regular(table, spec):
                    continue
                if table["tournament"].get("uniqueTournament", {}).get("id") != cid:
                    raise ValueError("Standings belong to another competition")
                stage = phase(table, spec)
                rows = table.get("rows", [])
                seen = set()
                for row in rows:
                    team, count = positive(row.get("team", {}).get("id")), row.get("matches")
                    if team in seen or type(count) is not int or count < 0:
                        raise ValueError("Missing, duplicate or invalid standings played counts")
                    seen.add(team)
                    prior = tables[stage].get(str(team))
                    if prior is not None and prior != count:
                        raise ValueError("Conflicting overlapping standings tables")
                    tables[stage][str(team)] = count
                if len(seen) < 2:
                    raise ValueError("Missing regular standings teams")
            if len(tables) > 1 and "regular" in tables and spec["schedule"] == "stages":
                # A cumulative annual table must not be added to both half-year
                # counts. Each ordinary stage gets its own independent check.
                tables.pop("regular")
            if not tables:
                raise ValueError("Regular standings are missing")
            self.put(f"standings:{cid}", dict(tables))
            self.put(f"teams:{cid}", sorted({int(t) for counts in tables.values() for t in counts}))
            self.add_job(f"/unique-tournament/{cid}/season/{sid}/rounds", cid, "rounds", sid)
            return
        if kind == "rounds":
            values = payload.get("rounds")
            if not isinstance(values, list):
                raise ValueError("Missing round inventory")
            if not values:
                self.put(f"availability:{cid}", "schedule_not_listed")
                return
            numbers = set()
            for item in values:
                if not isinstance(item, dict):
                    raise ValueError("Malformed round inventory")
                text, number = labels(item), item.get("round")
                if POST_STAGE.search(text) or number == 0:
                    continue
                if type(number) is not int or not 1 <= number <= 300:
                    raise ValueError("Unknown regular round identity")
                # Stage labels are recognized only for explicitly staged leagues.
                for key in ("name", "slug", "prefix"):
                    label = str(item.get(key, "")).strip()
                    if label and not re.fullmatch(r"(?:round[ -]?)?\d+", label, re.I):
                        if spec["schedule"] != "stages" or not PHASE.search(label):
                            raise ValueError("Unreviewed named round")
                limit = self.round_limit(cid)
                if limit is None or number <= limit:
                    numbers.add(number)
            if not numbers or min(numbers) != 1 or numbers != set(range(1, max(numbers) + 1)):
                raise ValueError("Regular round inventory must be contiguous from 1")
            self.put(f"rounds:{cid}", sorted(numbers))
            for number in sorted(numbers):
                self.add_job(f"/unique-tournament/{cid}/season/{sid}/events/round/{number}", cid, "events", sid, number)
            return
        if kind != "events":
            raise ValueError("Unsupported daily job")
        events = payload.get("events")
        if not isinstance(events, list) or not events or payload.get("hasNextPage") is True:
            raise ValueError("Empty or paginated round response; cannot assert complete discovery")
        for event in events:
            if not isinstance(event, dict):
                raise ValueError("Malformed fixture")
            eid = positive(event.get("id"))
            tournament, season = event.get("tournament", {}), match_season(event)
            sport = tournament.get("category", {}).get("sport", {})
            if sport and sport.get("id") != 1:
                raise ValueError("Fixture is not football")
            if (tournament.get("uniqueTournament", {}).get("id") != cid or season.get("id") != sid
                    or season_year_pair(season.get("year")) != tuple(spec["years"])):
                raise ValueError(f"Fixture {eid} belongs to another competition or season")
            number = match_round(event)
            if job["round"] not in ((event.get("roundInfo") or {}).get("round"), number) or type(number) is not int or number < 1:
                raise ValueError(f"Fixture {eid} has a conflicting round")
            home, away = (positive(event.get(side, {}).get("id")) for side in ("homeTeam", "awayTeam"))
            if home == away or type(event.get("startTimestamp")) is not int:
                raise ValueError("Invalid fixture teams or kickoff")
            match_url(event)
            selected = regular(event, spec)
            limit = self.round_limit(cid)
            selected = selected and (limit is None or number <= limit)
            old = self.db.execute("SELECT * FROM events WHERE id=?", (eid,)).fetchone()
            if old:
                before = json.loads(old["payload"])
                if (old["competition"], old["season"], old["round"], old["regular"], before["homeTeam"]["id"], before["awayTeam"]["id"], phase(before, spec)) != (
                        cid, sid, job["round"], int(selected), home, away, phase(event, spec)):
                    raise ValueError(f"Conflicting duplicate fixture identity: {eid}")
            self.db.execute("INSERT OR REPLACE INTO events VALUES(?,?,?,?,?,?,?)", (eid, cid, sid,
                job["round"], int(selected), json.dumps(event, ensure_ascii=False, allow_nan=False), observed))

    def season_selection(self, cid):
        if cid not in self._daily_summaries:
            self._daily_summaries[cid] = self._season_selection(cid)
        return self._daily_summaries[cid]

    def prepare_collection(self, **kwargs):
        self._daily_summaries.clear()
        return super().prepare_collection(**kwargs)

    def _season_selection(self, cid):
        spec = self.leagues[cid]
        rows = list(self.db.execute("SELECT * FROM events WHERE competition=?", (cid,)))
        # Identical ordered teams and round numbers in different ordinary stages
        # are distinct fixtures, not replacement identities.
        by_phase = defaultdict(list)
        for row in rows:
            by_phase[phase(json.loads(row["payload"]), spec)].append(row)
        records, superseded = [], []
        for group in by_phase.values():
            selected, replaced = selected_records(group)
            records.extend(selected); superseded.extend(replaced)
        events = [json.loads(r["payload"]) for r in records]
        tables = self.get(f"standings:{cid}", {})
        teams = set(self.get(f"teams:{cid}", []))
        issues, per_round = [], defaultdict(list)
        played = defaultdict(Counter)
        for event in events:
            stage = phase(event, spec)
            per_round[stage, match_round(event)].append(event)
            if event_status(event) not in SETTLED | UNFINISHED:
                issues.append(f"Unknown status for fixture {event['id']}")
            if not {event['homeTeam']['id'], event['awayTeam']['id']} <= teams:
                issues.append(f"Unknown standings team in fixture {event['id']}")
            if stage not in tables:
                issues.append(f"No matching regular standings for stage {stage}")
            if event_status(event) in {"finished", "awarded"}:
                played[stage].update(str(event[side]["id"]) for side in ("homeTeam", "awayTeam"))
        for (stage, number), fixtures in per_round.items():
            participants = [e[s]["id"] for e in fixtures for s in ("homeTeam", "awayTeam")]
            identities = [(e["homeTeam"]["id"], e["awayTeam"]["id"]) for e in fixtures]
            if len(identities) != len(set(identities)):
                issues.append(f"Conflicting fixtures in {stage} round {number}")
            if spec["schedule"] in {"round_robin", "split"} and (
                    len(fixtures) != len(teams) // 2 or len(participants) != len(set(participants))):
                issues.append(f"Round {number} has incomplete or conflicting team coverage")
        jobs = list(self.db.execute("SELECT state FROM jobs WHERE competition=?", (cid,)))
        complete = bool(jobs) and all(r[0] == "complete" for r in jobs)
        if complete:
            for stage, counts in tables.items():
                if any(played[stage][team] != count for team, count in counts.items()):
                    issues.append(f"Played fixtures do not reconcile with {stage} standings")
        now = self.config["created_at"]  # Stable selection across reconnect/publication.
        end = now + self.config["fixture_window_days"] * 86400
        finished = [e for e in events if e.get("status", {}).get("type") == "finished"]
        unfinished = [e for e in events if event_status(e) in UNFINISHED and
            (event_status(e) != "notstarted" or now <= e["startTimestamp"] < end)]
        wanted = {e["id"] for e in finished + unfinished}
        selected = sorted((r for r in records if r["id"] in wanted), key=lambda r: (json.loads(r["payload"])["startTimestamp"], r["id"]))
        availability = self.get(f"availability:{cid}", "discovering")
        if complete and availability != "available":
            issues.append(f"Requested coverage is unavailable: {availability}")
        state = ("discovering" if not complete else availability if availability != "available" else
                 "needs_review" if issues else "round_ready" if unfinished else
                 "outside_fixture_window" if any(event_status(e) == "notstarted" for e in events) else "season_complete")
        targets = {stage: min(match_round(e) for e in unfinished if phase(e, spec) == stage)
                   for stage in {phase(e, spec) for e in unfinished}}
        years = spec["years"]
        summary = dict(league=spec["name"], competition_id=cid, season_id=self.get(f"season:{cid}"),
            years=years, stem=f"{spec['name']}_{years[0]%100:02d}_{years[1]%100:02d}", state=state,
            last_completed_round=max((n for (_, n), group in per_round.items() if all(event_status(e) in SETTLED for e in group)), default=None),
            target_round=min(targets.values(), default=None), target_rounds=targets,
            history_matches=len(finished), unfinished_matches=len(unfinished),
            prediction_candidates=sum(event_status(e) == "notstarted" for e in unfinished),
            in_progress=sum(event_status(e) in {"inprogress", "interrupted"} for e in unfinished),
            enumerated_rounds=self.get(f"rounds:{cid}", []), excluded_stage_matches=sum(not r["regular"] for r in rows),
            superseded_fixtures=superseded, issues=list(dict.fromkeys(issues)),
            selection_start=now, selection_end=end, schedule_validation="regular_inventory_and_standings")
        return summary, selected
