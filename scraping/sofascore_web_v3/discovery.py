"""Resumable league/season discovery. Chrome supplies responses; Python sends no HTTP.

Discovery is separate from immutable collection campaigns. Every advertised
numbered round is enumerated before a regular-season campaign can be prepared.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
import uuid

from scraping.sofascore.discovery import event_status
from scraping.sofascore.event_round import match_round
from scraping.sofascore.store import atomic_write, json_bytes, season_year_pair
from .campaign import ROOT, BASE, campaign_path, prepare_campaign, read_json, validate_policy
from .capture import PROFILE, match_url
from .import_har import response_json, response_time
from .historical_formats import curtailed_rule, regular_round_limit, validate_curtailed_season, verified_unplayed_award, verified_replacement_id

DISCOVERY_BASE = ROOT / "data/website_capture_v3/discovery"
STAGE = re.compile(r"play[ -]?offs?|relegation|promotion|championship[ -](?:round|group)|"
                   r"qualification|qualifying|quarter.?final|semi.?final|\bfinal\b", re.I)


def positive(value):
    if type(value) is not int or not 0 < value < 2**53:
        raise ValueError("Expected a positive integer provider ID")
    return value


def league_catalog():
    registry = read_json(ROOT / "scraping/sofascore/competition_registry.json")
    return [{"name": name, "competition_id": row["competition_id"],
             "meetings_per_pair": 3 if name == "Premiership" else 2}
            for name, row in registry.items()]


def normalize_leagues(leagues):
    if not isinstance(leagues, list) or not 1 <= len(leagues) <= 100:
        raise ValueError("Select between 1 and 100 leagues")
    result = []
    for league in leagues:
        name = league.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,60}", name):
            raise ValueError("League filename names must contain letters, digits and underscores")
        meetings = league.get("meetings_per_pair", 2)
        if type(meetings) is not int or not 1 <= meetings <= 4:
            raise ValueError("Specify 1 to 4 regular-season meetings per team pair")
        result.append({"name": name, "competition_id": positive(league.get("competition_id")),
                       "meetings_per_pair": meetings})
    if len({v["name"] for v in result}) != len(result) or len({v["competition_id"] for v in result}) != len(result):
        raise ValueError("Duplicate league name or competition ID")
    return result


def prepare_discovery(*, season="25/26", leagues=None, include_lineups=True,
                      random_delay_max_seconds=None, collection_profile=None,
                      collection_policy=None, base=DISCOVERY_BASE, mode="season",
                      history_root=None, output_root=None):
    years = season_year_pair(season)
    if years is None or years[1] != years[0] + 1:
        raise ValueError("Choose an explicit season such as 25/26 or 2025/2026")
    if type(include_lineups) is not bool:
        raise ValueError("include_lineups must be boolean")
    if mode not in {"season", "daily"}:
        raise ValueError("Discovery mode must be season or daily")
    leagues = normalize_leagues(league_catalog() if leagues is None else leagues)
    policy = validate_policy(collection_policy if collection_policy is not None else {
        "collection_mode": "session", "memory_reset_every_matches": 50,
        "session_challenge_recovery": True, "random_action_delay_max_seconds": random_delay_max_seconds})
    profile = collection_profile or replace(PROFILE, include_lineups=include_lineups, include_upcoming=mode == "daily")
    if profile.include_comments or (profile.include_upcoming and mode != "daily"):
        raise ValueError("Season discovery collection supports completed fixtures without commentary")
    if mode == "daily" and (not profile.include_upcoming or policy["collection_mode"] != "session"):
        raise ValueError("Daily discovery requires upcoming fixtures and Chrome session mode")
    discovery_id = f"discover-{years[0]}-{years[1]}-{uuid.uuid4().hex[:8]}"
    directory = campaign_path(discovery_id, base)
    directory.mkdir(parents=True)
    atomic_write(directory / "config.json", json_bytes({"discovery_id": discovery_id,
        "years": years, "leagues": leagues, "created_at": time.time(), "scope": "regular_season",
        "profile": asdict(profile), "policy": policy, "mode": mode,
        "history_root": str(Path(history_root or ROOT / "data/xDiyo_data").resolve()),
        "output_root": str(Path(output_root or ROOT / "data/xDiyo_data").resolve())}))
    with open_discovery(directory) as discovery:
        return discovery.status()


def open_discovery(directory, *, read_only=False):
    mode = read_json(Path(directory) / "config.json").get("mode")
    if mode == "archive":
        from .archive_discovery import ArchiveDiscovery
        return ArchiveDiscovery(directory, read_only=read_only)
    if mode == "daily":
        from .daily_discovery import DailyDiscovery
        return DailyDiscovery(directory, read_only=read_only)
    return Discovery(directory, read_only=read_only)


def round_numbers(payload):
    rounds = payload.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise ValueError("Provider returned no round inventory; discovery is incomplete")
    selected = set()
    for row in rounds:
        if not isinstance(row, dict):
            raise ValueError("Malformed round inventory")
        # Named knockout rounds can share a number with regular league rounds.
        labels = [str(row.get(k, "")).strip() for k in ("name", "slug", "prefix")]
        if any(STAGE.search(label) for label in labels):
            continue
        number = row.get("round")
        # Some older inventories advertise a round 0 alongside the regular
        # schedule. Our collection scope starts at round 1; retain the raw
        # inventory but do not request that pre-round bucket.
        if type(number) is int and number == 0:
            continue
        if type(number) is not int or not 1 <= number <= 300:
            raise ValueError("Unknown round identity; inspect the provider's season format")
        if any(label and not re.fullmatch(r"(?:round[ -]?)?\d+", label, re.I) for label in labels):
            raise ValueError("Unrecognized named round; review stage selection before collecting")
        selected.add(number)
    if not selected:
        raise ValueError("No regular numbered rounds were advertised")
    return sorted(selected)


def regular_stage(event):
    tournament = event.get("tournament", {})
    info = event.get("roundInfo") or {}
    labels = " ".join(str(obj.get(key, "")) for obj in (tournament, info)
                      for key in ("name", "slug", "prefix"))
    if tournament.get("isGroup") is True or STAGE.search(labels):
        return False
    limit = regular_round_limit(tournament.get("uniqueTournament", {}).get("id"),
                                season_year_pair(event.get("season", {}).get("year")))
    if limit is not None and type(info.get("round")) is int and info["round"] > limit:
        return False
    if tournament.get("isGroup") is not False:
        raise ValueError("Missing tournament stage flag; regular-season membership is unverified")
    return True


def select_regular_records(rows):
    """Resolve an unplayed/abandoned record against one later finished match.

    The original provider IDs and bodies remain in SQLite. Apart from explicitly
    reviewed identities, the normalized round and ordered teams must agree;
    ambiguous completed duplicates remain unresolved and fail reconciliation.
    """
    records = [(row, json.loads(row["payload"])) for row in rows if row["regular"]]
    candidates = {row["id"]: event for row, event in records}
    grouped = defaultdict(list)
    selected, superseded = [], []
    for row, event in records:
        reviewed = verified_replacement_id(event, candidates)
        if reviewed is not None:
            superseded.append({"event_id": row["id"], "replacement_event_id": reviewed,
                               "status": event_status(event)})
            continue
        key = (row["competition"], row["season"], match_round(event),
               event["homeTeam"]["id"], event["awayTeam"]["id"])
        grouped[key].append((row, event))
    for group in grouped.values():
        completed = [(r, e) for r, e in group if e.get("status", {}).get("type") == "finished"]
        replacement = completed[0] if len(completed) == 1 else None
        for row, event in group:
            if (replacement and event_status(event) in {"postponed", "abandoned", "cancelled"}
                    and event["startTimestamp"] < replacement[1]["startTimestamp"]):
                superseded.append({"event_id": row["id"], "replacement_event_id": replacement[0]["id"],
                                   "status": event_status(event)})
            else:
                selected.append(row)
    return selected, sorted(superseded, key=lambda r: r["event_id"])


class Discovery:
    def __init__(self, directory, *, read_only=False):
        self.directory = Path(directory).resolve()
        self.config = read_json(self.directory / "config.json")
        self.leagues = {x["competition_id"]: x for x in normalize_leagues(self.config["leagues"])}
        if read_only:
            self.db = sqlite3.connect((self.directory / "discovery.sqlite").as_uri()+"?mode=ro", uri=True)
            self.db.row_factory = sqlite3.Row
            self.current = None
            return
        self.db = sqlite3.connect(self.directory / "discovery.sqlite")
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS jobs(path TEXT PRIMARY KEY, competition INTEGER, kind TEXT,
            season INTEGER, round INTEGER, state TEXT, lease TEXT, error TEXT);
          CREATE TABLE IF NOT EXISTS responses(path TEXT PRIMARY KEY, observed REAL, body TEXT, sha256 TEXT);
          CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, competition INTEGER, season INTEGER,
            round INTEGER, regular INTEGER, payload TEXT, observed REAL);
          CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY,value TEXT);
          CREATE TABLE IF NOT EXISTS attempts(id INTEGER PRIMARY KEY, path TEXT, observed REAL,
            http_status INTEGER, error TEXT);
        """)
        with self.db:
            for cid in self.leagues:
                self.add_job(f"/unique-tournament/{cid}/seasons", cid, "seasons")
        self.current = None

    def __enter__(self): return self
    def __exit__(self, *args): self.close()
    def close(self): self.db.close()

    def add_job(self, path, competition, kind, season=None, round_number=None):
        self.db.execute("INSERT OR IGNORE INTO jobs VALUES(?,?,?,?,?,'pending',NULL,NULL)",
                        (path, competition, kind, season, round_number))

    def curtailed_rule(self, competition):
        if self.config.get("mode", "season") == "season":
            return curtailed_rule(competition, self.config["years"])
        return None

    def next_job(self, *, resume=False):
        if resume:
            with self.db:
                self.db.execute("DELETE FROM state WHERE key='pause'")
        pause = self.db.execute("SELECT value FROM state WHERE key='pause'").fetchone()
        if pause:
            return {"paused": True, "reason": pause[0]}
        if self.current:
            return self.current
        row = self.db.execute("SELECT * FROM jobs WHERE state!='complete' ORDER BY rowid LIMIT 1").fetchone()
        if row is None:
            return {"done": True, "summary": self.status()}
        self.current = {"path": row["path"], "competition_id": row["competition"],
                        "season_id": row["season"], "lease": uuid.uuid4().hex,
                        "inspection_only": row["kind"] == "round_inspection",
                        "policy": self.config["policy"], "label": self.leagues[row["competition"]]["name"]}
        with self.db:
            self.db.execute("UPDATE jobs SET state='running',lease=? WHERE path=?",
                            (self.current["lease"], row["path"]))
        return self.current

    def pause(self, reason="Paused by user"):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO state VALUES('pause',?)", (str(reason)[:300],))
        self.current = None
        return self.status()

    def observe(self, path, lease, entry):
        if not self.current or (path, lease) != (self.current["path"], self.current["lease"]):
            raise ValueError("Stale discovery lease")
        request = entry.get("request", {})
        if request.get("method") != "GET" or request.get("url") != f"https://www.sofascore.com/api/v1{path}" or request.get("queryString"):
            raise ValueError("Response does not match the exact leased discovery path")
        observed = response_time(entry)
        if observed > time.time() + 60:
            raise ValueError("Discovery response timestamp is in the future")
        code = entry.get("response", {}).get("status")
        if type(code) is not int or not 100 <= code <= 599:
            raise ValueError("Invalid response status")
        with self.db:
            self.db.execute("INSERT INTO attempts(path,observed,http_status) VALUES(?,?,?)", (path, observed, code))
        try:
            if code != 200:
                raise ValueError(f"HTTP {code}: discovery is incomplete; resume after checking access")
            payload = response_json(entry)
            body = json.dumps(payload, ensure_ascii=False, allow_nan=False)
            if not isinstance(payload, dict) or len(body.encode()) > 8 * 1024 * 1024:
                raise ValueError("Invalid or oversized discovery response")
            # Retain readable provider evidence even if a new league's shape or
            # identities fail validation. That page never becomes complete.
            with self.db:
                self.db.execute("INSERT OR REPLACE INTO responses VALUES(?,?,?,?)",
                    (path, observed, body, hashlib.sha256(body.encode()).hexdigest()))
            row = self.db.execute("SELECT * FROM jobs WHERE path=?", (path,)).fetchone()
            with self.db:
                self.process(row, payload, observed)
                self.db.execute("UPDATE jobs SET state='complete',error=NULL WHERE path=?", (path,))
            self.current = None
            return {"saved": True, "summary": self.status()}
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            with self.db:
                self.db.execute("UPDATE jobs SET state='failed',error=? WHERE path=?", (str(error)[:300], path))
                self.db.execute("UPDATE attempts SET error=? WHERE id=(SELECT MAX(id) FROM attempts)", (str(error)[:300],))
            self.pause(str(error))
            raise ValueError(str(error)) from error

    def process(self, job, payload, observed):
        cid, sid = job["competition"], job["season"]
        if job["kind"] == "seasons":
            values = payload.get("seasons")
            if not isinstance(values, list):
                raise ValueError("Missing season list")
            found = [s for s in values if isinstance(s, dict) and season_year_pair(s.get("year")) == tuple(self.config["years"])]
            if len(found) != 1:
                raise ValueError("Requested season is missing or ambiguous; never use the latest season implicitly")
            sid = positive(found[0].get("id"))
            self.add_job(f"/unique-tournament/{cid}/season/{sid}/rounds", cid, "rounds", sid)
            self.add_job(f"/unique-tournament/{cid}/season/{sid}/standings/total", cid, "standings", sid)
        elif job["kind"] == "rounds":
            limit = regular_round_limit(cid, self.config["years"])
            for number in round_numbers(payload):
                if limit is not None and number > limit:
                    continue
                self.add_job(f"/unique-tournament/{cid}/season/{sid}/events/round/{number}", cid, "events", sid, number)
        elif job["kind"] == "standings":
            standings = payload.get("standings")
            if not isinstance(standings, list):
                raise ValueError("Missing standings inventory")
            selected = [s for s in standings if s.get("type") == "total" and regular_stage(s)]
            if len(selected) != 1:
                raise ValueError("Regular-season standings are missing or ambiguous")
            standing = selected[0]
            if standing["tournament"].get("uniqueTournament", {}).get("id") != cid:
                raise ValueError("Standings belong to another competition")
            teams = [positive(r.get("team", {}).get("id")) for r in standing.get("rows", [])]
            if len(teams) < 2 or len(set(teams)) != len(teams):
                raise ValueError("Missing or duplicate standings team IDs")
            self.db.execute("INSERT OR REPLACE INTO state VALUES(?,?)", (f"teams:{cid}", json.dumps(sorted(teams))))
            if self.curtailed_rule(cid):
                counts = {str(r["team"]["id"]): r.get("matches") for r in standing["rows"]}
                if any(type(n) is not int or n < 0 for n in counts.values()):
                    raise ValueError("Curtailed-season final standings need each team's played-match count")
                self.db.execute("INSERT OR REPLACE INTO state VALUES(?,?)", (f"played_counts:{cid}", json.dumps(counts)))
        elif job["kind"] == "round_inspection":
            # Explicit operator-requested inspection of an anomalous bucket.
            # observe() keeps its response and checksum; no fixtures enter the
            # regular schedule or campaign without a separate reviewed repair.
            if not isinstance(payload.get("events"), list) or payload.get("hasNextPage") is True:
                raise ValueError("Malformed or paginated round inspection response")
        else:
            if job["kind"] == "recovered_event":
                recovered = payload.get("event")
                if not isinstance(recovered, dict) or job["path"] != f"/event/{recovered.get('id')}":
                    raise ValueError("Recovered metadata does not match its event path")
                events = [recovered]
            else:
                events = payload.get("events")
            if (not isinstance(events, list) or (not events and not self.curtailed_rule(cid))
                    or payload.get("hasNextPage") is True):
                raise ValueError("Empty or paginated round response; cannot assert complete discovery")
            for event in events:
                if not isinstance(event, dict):
                    raise ValueError("Malformed fixture")
                eid = positive(event.get("id"))
                tournament, season = event.get("tournament", {}), event.get("season", {})
                sport = tournament.get("category", {}).get("sport", {})
                if sport and sport.get("id") != 1:
                    raise ValueError("The selected competition is not football")
                if tournament.get("uniqueTournament", {}).get("id") != cid or season.get("id") != sid or season_year_pair(season.get("year")) != tuple(self.config["years"]):
                    raise ValueError(f"Fixture {eid} belongs to another competition or season")
                provider_round = (event.get("roundInfo") or {}).get("round")
                if job["round"] not in (provider_round, match_round(event)):
                    raise ValueError(f"Fixture {eid} has a conflicting round")
                home, away = positive(event.get("homeTeam", {}).get("id")), positive(event.get("awayTeam", {}).get("id"))
                if home == away or type(event.get("startTimestamp")) is not int:
                    raise ValueError(f"Fixture {eid} has invalid teams or kickoff")
                match_url(event)
                regular = regular_stage(event)
                if job["kind"] == "recovered_event" and not regular:
                    raise ValueError("Recovered event is outside the regular-season scope")
                old = self.db.execute("SELECT * FROM events WHERE id=?", (eid,)).fetchone()
                if old:
                    previous = json.loads(old["payload"])
                    if (old["competition"], old["season"], old["round"], old["regular"], previous["homeTeam"]["id"], previous["awayTeam"]["id"]) != (cid, sid, job["round"], int(regular), home, away):
                        raise ValueError(f"Conflicting duplicate fixture identity: {eid}")
                self.db.execute("INSERT OR REPLACE INTO events VALUES(?,?,?,?,?,?,?)", (eid, cid, sid,
                    job["round"], int(regular), json.dumps(event, ensure_ascii=False, allow_nan=False), observed))

    def admit_recovered_event(self, entry, *, event_id, competition_id, season_id, round_number):
        """Admit explicitly reviewed Chrome HAR metadata after round enumeration.

        Call under the collector lock with Chrome disconnected. No requests are
        sent and no raw round response is replaced. Normal fixture validation
        and the complete season pairing checks remain mandatory.
        """
        snapshot = self.status()
        if self.config.get("mode", "season") not in {"season", "archive"} or snapshot["campaign"] or not snapshot["discovery_complete"]:
            raise ValueError("Recovery requires completed season discovery without a collection campaign")
        path = f"/event/{positive(event_id)}"
        round_path = f"/unique-tournament/{positive(competition_id)}/season/{positive(season_id)}/events/round/{positive(round_number)}"
        source = self.db.execute("SELECT state FROM jobs WHERE path=? AND kind='events'", (round_path,)).fetchone()
        if not source or source[0] != "complete" or self.db.execute("SELECT 1 FROM events WHERE id=?", (event_id,)).fetchone():
            raise ValueError("Recovery must add a missing event in an already enumerated round")
        # Validate the original exact-route response before creating its job.
        if (entry.get("request", {}).get("url") != f"https://www.sofascore.com/api/v1{path}"
                or response_json(entry).get("event", {}).get("id") != event_id):
            raise ValueError("Recovered metadata does not match the requested event")
        with self.db:
            self.add_job(path, competition_id, "recovered_event", season_id, round_number)
        job = self.next_job(resume=True)
        return self.observe(job["path"], job["lease"], entry)

    def revalidate_saved_failures(self):
        """After a parser repair, validate retained 200 evidence without new requests.

        Use under the collector lock with Chrome disconnected. Access denials,
        missing bodies and newer failed attempts can never be satisfied this way.
        """
        repaired = []
        for job in list(self.db.execute("SELECT * FROM jobs WHERE state='failed'")):
            saved = self.db.execute("SELECT * FROM responses WHERE path=?", (job["path"],)).fetchone()
            latest = self.db.execute("SELECT * FROM attempts WHERE path=? ORDER BY id DESC LIMIT 1", (job["path"],)).fetchone()
            if not saved or not latest or latest["http_status"] != 200 or latest["observed"] != saved["observed"]:
                continue
            if hashlib.sha256(saved["body"].encode()).hexdigest() != saved["sha256"]:
                raise ValueError("Saved discovery response checksum mismatch")
            with self.db:
                self.process(job, json.loads(saved["body"]), saved["observed"])
                self.db.execute("UPDATE jobs SET state='complete',error=NULL,lease=NULL WHERE path=?", (job["path"],))
            repaired.append(job["path"])
        return repaired

    def status(self):
        counts = dict(self.db.execute("SELECT state,COUNT(*) FROM jobs GROUP BY state"))
        seasons = []
        for cid, league in self.leagues.items():
            rows = list(self.db.execute("SELECT * FROM events WHERE competition=?", (cid,)))
            selected, superseded = select_regular_records(rows)
            regular = [json.loads(r["payload"]) for r in selected]
            season_job = self.db.execute("SELECT season FROM jobs WHERE competition=? AND kind='rounds'", (cid,)).fetchone()
            pairs = Counter(tuple(sorted((e["homeTeam"]["id"], e["awayTeam"]["id"]))) for e in regular)
            team_record = self.db.execute("SELECT value FROM state WHERE key=?", (f"teams:{cid}",)).fetchone()
            teams = set(json.loads(team_record[0])) if team_record else set()
            expected = len(teams) * (len(teams) - 1) * league["meetings_per_pair"] // 2
            balanced = (len(teams) >= 2 and {t for p in pairs for t in p} == teams
                and len(pairs) == len(teams) * (len(teams) - 1) // 2
                and all(n == league["meetings_per_pair"] for n in pairs.values()))
            if balanced and league["meetings_per_pair"] % 2 == 0:
                directed = Counter((e["homeTeam"]["id"], e["awayTeam"]["id"]) for e in regular)
                balanced = all(directed[a, b] == league["meetings_per_pair"] // 2
                               for a in teams for b in teams if a != b)
            statuses = Counter(event_status(e) for e in regular)
            finished = sum(e.get("status", {}).get("type") == "finished" for e in regular)
            terminal_exclusions = [{"event_id": e["id"], "status": "awarded" if verified_unplayed_award(e) else event_status(e)} for e in regular
                if e.get("status", {}).get("type") != "finished"
                and (event_status(e) in {"abandoned", "cancelled", "awarded"} or verified_unplayed_award(e))]
            rule = self.curtailed_rule(cid)
            validation = {"schedule_verified": balanced, "schedule_validation": "full_pairings"}
            if rule:
                record = self.db.execute("SELECT value FROM state WHERE key=?", (f"played_counts:{cid}",)).fetchone()
                issues = validate_curtailed_season(regular, teams, json.loads(record[0]) if record else None,
                                                   league["meetings_per_pair"], rule)
                validation = {"schedule_verified": not issues, "schedule_validation": "curtailed_final_standings",
                    "expected_played_matches": rule["matches"], "validation_issues": issues,
                    "curtailment_source": rule["source"],
                    "unplayed_after_curtailment": [{"event_id": e["id"], "status": event_status(e)} for e in regular
                                                  if e.get("status", {}).get("type") != "finished"]}
            seasons.append({"league": league["name"], "competition_id": cid,
                "season_id": season_job[0] if season_job else None, "regular_matches": len(regular),
                "excluded_stage_matches": sum(not r["regular"] for r in rows),
                "superseded_fixtures": superseded, "teams": len(teams),
                "expected_matches_from_format": expected, "schedule_balanced": balanced,
                "finished": finished, "statuses": dict(statuses),
                "non_collectable_terminal_matches": terminal_exclusions, **validation})
        complete = bool(counts) and set(counts) == {"complete"}
        ready = complete and all(s["schedule_verified"] and s["finished"] > 0
            and (s["schedule_validation"] == "curtailed_final_standings"
                 or s["finished"] + len(s["non_collectable_terminal_matches"]) == s["regular_matches"]) for s in seasons)
        saved = self.db.execute("SELECT value FROM state WHERE key='campaign'").fetchone()
        return {"discovery_id": self.config["discovery_id"], "directory": str(self.directory),
                "jobs": counts, "requests": self.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0],
                "discovery_complete": complete, "ready_to_collect": ready, "seasons": seasons,
                "campaign": json.loads(saved[0]) if saved else None, "api_requests_sent_by_python": 0}

    def prepare_collection(self, *, campaign_base=BASE):
        snapshot = self.status()
        if snapshot["campaign"]:
            return snapshot["campaign"]
        if not snapshot["ready_to_collect"]:
            raise ValueError("Finish discovery and resolve incomplete schedules/statuses before creating a season campaign")
        ids, references = [], {}
        years = self.config["years"]
        for cid, league in self.leagues.items():
            records, _ = select_regular_records(self.db.execute("SELECT * FROM events WHERE competition=? AND regular=1", (cid,)))
            records.sort(key=lambda r: (json.loads(r["payload"])["startTimestamp"], r["id"]))
            for record in records:
                eid, event = record["id"], json.loads(record["payload"])
                if event.get("status", {}).get("type") != "finished":
                    continue  # Terminal unplayed/abandoned evidence remains in the discovery inventory.
                # Fresh collection still verifies provider metadata and every component.
                references[eid] = {"kind": "queue_identity", "event": event,
                    "observed_at": record["observed"], "source": str(self.directory / "discovery.sqlite"),
                    "season_stem": f"{league['name']}_{years[0]%100:02d}_{years[1]%100:02d}",
                    "route_source": "chrome_season_discovery"}
                ids.append(eid)
        from scraping.sofascore.config import Profile
        result = prepare_campaign(ids, references=references, base=campaign_base,
            name=f"chrome-{years[0]}-{years[1]}", profile=Profile(**self.config["profile"]),
            policy=self.config["policy"], source_metadata={"collection": "season_discovery",
                "discovery_id": self.config["discovery_id"], "scope": "regular_season",
                "discovery_requests": snapshot["requests"], "seasons": snapshot["seasons"]})
        # Keep native messages small; thousands of IDs remain in the saved campaign.
        result.pop("event_ids", None)
        with self.db:
            self.db.execute("INSERT INTO state VALUES('campaign',?)", (json.dumps(result),))
        return result
