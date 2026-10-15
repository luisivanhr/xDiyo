"""Many league-seasons, one discovery ID and one immutable collection campaign.

Uses the existing extension protocol and durable response/lease implementation.
Only Chrome requests provider data. No new endpoints or browser reloads are needed.
"""
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

from scraping.sofascore.config import Profile
from scraping.sofascore.discovery import event_status, is_completed_event
from scraping.sofascore.event_round import match_round, match_season
from scraping.sofascore.store import atomic_write, json_bytes, season_year_pair
from .archive_formats import EXCLUDED, check_kickoff, is_regular, reviewed_standings, validate_schedule
from .campaign import ROOT, BASE, Campaign, campaign_path, collector_lock, prepare_campaign, read_json, validate_policy
from .capture import PROFILE, match_url
from .discovery import DISCOVERY_BASE, Discovery, normalize_leagues, positive, round_numbers, select_regular_records

ARCHIVE_BASE = ROOT / "data/website_capture_v3/batches"


def _mexico_window(spec):
    """The saved short-tournament windows, never a generic date-only fallback."""
    years = season_year_pair(spec.get("cycle"))
    kind = {11621: "Apertura", 11620: "Clausura"}.get(spec["competition_id"])
    if not years or years[1] != years[0] + 1 or spec.get("tournament") != kind or kind is None:
        return None
    year = years[0]
    dates = (f"{year}-07-01", f"{year+1}-01-01", f"{year+1}-07-01")
    selected = dates[:2] if kind == "Apertura" else dates[1:]
    if (spec.get("start"), spec.get("end")) != selected:
        raise ValueError("Mexican tournament window differs from its saved cycle")
    return tuple(datetime.fromisoformat(day).replace(tzinfo=timezone.utc).timestamp() for day in dates)


def validate_specs(cohorts):
    if not isinstance(cohorts, list) or not 1 <= len(cohorts) <= 200:
        raise ValueError("Select 1 to 200 explicit league-season cohorts")
    identities, stems = set(), set()
    for spec in cohorts:
        normalize_leagues([spec])
        if type(spec.get("snapshot")) is not bool:
            raise ValueError("Specify whether the cohort is a current-season snapshot")
        for key in ("expected_teams", "round_limit"):
            if spec[key] is not None and (type(spec[key]) is not int or spec[key] < 1):
                raise ValueError(f"Invalid {key}")
        if type(spec.get("groups")) is not int or spec["groups"] not in {1, 2, 4}:
            raise ValueError("Unsupported number of regular-season groups")
        if spec["groups"] > 1 and (not spec["expected_teams"] or spec["expected_teams"] % spec["groups"]):
            raise ValueError("Grouped competition needs a divisible expected team count")
        if not isinstance(spec.get("aliases"), list) or not spec["aliases"]:
            raise ValueError("Each cohort needs explicit provider season aliases")
        for alias in spec["aliases"]:
            years = season_year_pair(alias)
            if years is None or years[1] not in (years[0], years[0]+1):
                raise ValueError("Invalid calendar-year or two-year season")
            identity = spec["competition_id"], years
            stem = f"{spec['name']}_{years[0]%100:02d}_{years[1]%100:02d}"
            if identity in identities or stem in stems:
                raise ValueError("Overlapping season selectors or duplicate output stems")
            identities.add(identity); stems.add(stem)
    return cohorts


def prepare_archive(name, cohorts, *, output=ROOT / "data/xDiyo_data", include_lineups=True,
                    random_delay_max_seconds=None, base=ARCHIVE_BASE, discovery_base=DISCOVERY_BASE):
    """Idempotent local preparation, safe while another capture is running."""
    validate_specs(cohorts)
    if type(include_lineups) is not bool:
        raise ValueError("include_lineups must be boolean")
    profile = asdict(replace(PROFILE, include_lineups=include_lineups, include_upcoming=False))
    policy = validate_policy({"collection_mode": "session", "memory_reset_every_matches": 50,
        "session_challenge_recovery": True, "random_action_delay_max_seconds": random_delay_max_seconds})
    directory = campaign_path("archive-" + name, base)
    settings = dict(name=name, cohorts=cohorts, profile=profile, policy=policy,
        output_root=str(Path(output).resolve()), discovery_base=str(Path(discovery_base).resolve()))
    with collector_lock(directory / "prepare.lock"):
        path = directory / "batch.json"
        if path.exists():
            saved = read_json(path)
            if saved["settings"] != settings:
                raise ValueError("Saved archive settings differ; never rewrite an existing discovery/queue")
            config = read_json(Path(saved["directory"]) / "config.json")
            if (config.get("mode") != "archive" or config.get("discovery_id") != saved["discovery_id"]
                    or any(config[k] != settings[k] for k in ("name", "cohorts", "profile", "policy", "output_root"))):
                raise ValueError("Saved discovery differs from the archive selection")
            return {**saved, "path": str(path)}
        for spec in cohorts:
            for alias in spec["aliases"]:
                years = season_year_pair(alias)
                stem = f"{spec['name']}_{years[0]%100:02d}_{years[1]%100:02d}"
                if any((Path(output) / (stem + suffix)).exists() for suffix in (".parquet", ".manifest.json")):
                    raise ValueError(f"Existing output for {stem}; use its original campaign or a reviewed update")
        discovery_id = f"discover-{name}-{uuid.uuid4().hex[:8]}"
        discovery_dir = campaign_path(discovery_id, discovery_base)
        leagues = {}
        for spec in cohorts:
            # Display label only. Per-cohort names and formats stay in cohorts.
            leagues.setdefault(spec["competition_id"], {k: spec[k] for k in ("name", "competition_id", "meetings_per_pair")})
        config = {**settings, "mode": "archive", "discovery_id": discovery_id,
                  "leagues": list(leagues.values()), "created_at": time.time(), "scope": "regular_season"}
        atomic_write(discovery_dir / "config.json", json_bytes(config))
        with ArchiveDiscovery(discovery_dir) as discovery:
            discovery.status()
        saved = {"settings": settings, "discovery_id": discovery_id, "directory": str(discovery_dir)}
        atomic_write(path, json_bytes(saved))
    return {**saved, "path": str(path)}


class ArchiveDiscovery(Discovery):
    def __init__(self, directory, *, read_only=False):
        super().__init__(directory, read_only=read_only)
        self.specs = validate_specs(self.config["cohorts"])
        self._summaries = {}

    def _value(self, key):
        row = self.db.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def _save(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO state VALUES(?,?)", (key, json.dumps(value)))

    def _resolved(self, index):
        resolved = self._value(f"resolved:{index}")
        if resolved:
            return resolved
        spec = self.specs[index]
        # Chrome's exact 2770 response contains both 2010/11 short tournaments.
        # Clausura 2011 is absent from its separate inventory. Reuse the captured
        # source identity, not a fabricated season or a rewritten event body.
        if (spec["name"] == "Liga_MX_Clausura" and spec["competition_id"] == 11620 and spec["season"] == "2010/2011"
                and spec.get("tournament") == "Clausura" and _mexico_window(spec)
                and spec["expected_teams"] == 18 and spec["round_limit"] == 17
                and spec["meetings_per_pair"] == 1 and not spec["snapshot"]
                and self._value(f"missing:{index}")):
            sources = [self._value(f"resolved:{i}") for i, other in enumerate(self.specs)
                       if other["name"] == "Liga_MX_Apertura" and other["competition_id"] == 11621 and other["season"] == "2010/2011"
                       and other.get("tournament") == "Apertura" and _mexico_window(other)]
            if len(sources) == 1 and sources[0] and sources[0]["season_id"] == 2770:
                source = sources[0]
                if tuple(source["years"]) == (2010, 2011):
                    return {**source, "stem": "Liga_MX_Clausura_10_11", "source_competition_id": 11621,
                            "source_review": "https://www.chivasdecorazon.com.mx/es/noticias/14641/Anuario-Chivas-2011"}
        return None

    def _cohort(self, cid, sid):
        found = [(i, s, self._resolved(i)) for i, s in enumerate(self.specs) if s["competition_id"] == cid]
        found = [(i, s, r) for i, s, r in found if r and r["season_id"] == sid]
        if len(found) != 1:
            raise ValueError("Missing or ambiguous resolved cohort identity")
        return found[0]

    def next_job(self, *, resume=False):
        pending = [s["name"] for s in self.specs if s.get("scope_pending")]
        if pending:
            return {"paused": True, "reason": "Scope decision required for " + ", ".join(pending)}
        result = super().next_job(resume=resume)
        if result.get("season_id"):
            _, spec, _ = self._cohort(result["competition_id"], result["season_id"])
            result["label"] = f"{spec['name']} {spec['season']}"
        return result

    def process(self, job, payload, observed):
        # A reviewed shared provider season can feed two distinct cohort views.
        self._summaries.clear()
        cid, sid = job["competition"], job["season"]
        if job["kind"] == "seasons":
            values = payload.get("seasons")
            if not isinstance(values, list):
                raise ValueError("Missing season inventory")
            for i, spec in enumerate(self.specs):
                if spec["competition_id"] != cid:
                    continue
                aliases = {season_year_pair(x) for x in spec["aliases"]}
                found = [s for s in values if isinstance(s, dict) and season_year_pair(s.get("year")) in aliases]
                if not found:
                    # Finish enumerating the other requested cohorts. This gap
                    # remains visible and prevents campaign creation; no season
                    # is dropped, substituted or declared successfully collected.
                    self._save(f"missing:{i}", "Requested season is absent from the captured provider inventory")
                    self._summaries.pop(i, None)
                    continue
                if len(found) != 1:
                    raise ValueError(f"{spec['name']} {spec['season']}: season ambiguous; no implicit fallback")
                self.db.execute("DELETE FROM state WHERE key=?", (f"missing:{i}",))
                sid, year = positive(found[0].get("id")), found[0]["year"]
                years = season_year_pair(year)
                self._save(f"resolved:{i}", dict(season_id=sid, year=year, years=years,
                    stem=f"{spec['name']}_{years[0]%100:02d}_{years[1]%100:02d}"))
                self._summaries.pop(i, None)
                self.add_job(f"/unique-tournament/{cid}/season/{sid}/rounds", cid, "rounds", sid)
                if spec["expected_teams"] is None or spec["snapshot"]:
                    self.add_job(f"/unique-tournament/{cid}/season/{sid}/standings/total", cid, "standings", sid)
            return
        index, spec, resolved = self._cohort(cid, sid)
        self._summaries.pop(index, None)
        if job["kind"] == "rounds":
            values = payload.get("rounds")
            if not isinstance(values, list):
                raise ValueError("Missing round inventory")
            # Named play-in/placement rounds are also excluded, without opening
            # any new endpoint or silently treating a named stage as a league round.
            values = [r for r in values if not isinstance(r, dict) or not EXCLUDED.search(
                " ".join(str(r.get(k, "")) for k in ("name", "slug", "prefix")))]
            advertised = round_numbers({"rounds": values})
            selected = [n for n in advertised if spec["round_limit"] is None or n <= spec["round_limit"]]
            # One delayed regular fixture was misfiled after the Scottish split.
            # Read its source bucket; match_round admits only its exact identity.
            # All actual post-split events remain regular=False and never queue.
            if (cid == 36 and sid == 8190 and tuple(resolved["years"]) == (2014, 2015)
                    and spec["round_limit"] == 33):
                if 37 not in advertised:
                    raise ValueError("Missing reviewed source bucket for Inverness-Dundee United")
                selected = sorted({*selected, 37})
            if not selected or selected[0] != 1:
                raise ValueError("The regular-season inventory must begin at round 1")
            self._save(f"rounds:{index}", selected)
            for number in selected:
                self.add_job(f"/unique-tournament/{cid}/season/{sid}/events/round/{number}", cid, "events", sid, number)
        elif job["kind"] == "standings":
            standings = payload.get("standings")
            if not isinstance(standings, list):
                raise ValueError("Missing standings inventory")
            selected = [s for s in standings if s.get("type") == "total"
                and s.get("tournament", {}).get("isGroup") is False
                and not EXCLUDED.search(str(s.get("tournament", {}).get("name", "")))]
            if len(selected) != 1:
                raise ValueError("Regular-season standings missing or ambiguous")
            standing = selected[0]
            if standing["tournament"].get("uniqueTournament", {}).get("id") != cid:
                raise ValueError("Standings belong to another competition")
            rows = standing.get("rows", [])
            teams = {positive(r.get("team", {}).get("id")): r.get("matches") for r in rows}
            if len(teams) != len(rows) or len(teams) < 2:
                raise ValueError("Missing or duplicate standings teams")
            if spec["snapshot"] and any(type(n) is not int or n < 0 for n in teams.values()):
                raise ValueError("Current-season snapshot needs standings played counts")
            self._save(f"standings:{index}", teams)
        elif job["kind"] in {"events", "recovered_event"}:
            if job["kind"] == "recovered_event":
                recovered = payload.get("event")
                if not isinstance(recovered, dict) or job["path"] != f"/event/{recovered.get('id')}":
                    raise ValueError("Recovered metadata does not match its event path")
                events = [recovered]
            else:
                events = payload.get("events")
            if not isinstance(events, list) or not events or payload.get("hasNextPage") is True:
                raise ValueError("Empty or paginated round response; discovery is incomplete")
            for event in events:
                if not isinstance(event, dict):
                    raise ValueError("Malformed fixture")
                eid = positive(event.get("id"))
                tournament, season = event.get("tournament", {}), match_season(event)
                sport = tournament.get("category", {}).get("sport", {})
                if sport and sport.get("id") != 1:
                    raise ValueError("The selected fixture is not football")
                if (tournament.get("uniqueTournament", {}).get("id") != cid or season.get("id") != sid
                        or season_year_pair(season.get("year")) != tuple(resolved["years"])):
                    raise ValueError(f"Fixture {eid} belongs to another competition or season")
                number = match_round(event)
                provider_round = (event.get("roundInfo") or {}).get("round")
                if job["round"] not in (provider_round, number) or type(number) is not int or number < 1:
                    raise ValueError(f"Fixture {eid} has a conflicting round")
                home, away = positive(event.get("homeTeam", {}).get("id")), positive(event.get("awayTeam", {}).get("id"))
                if home == away:
                    raise ValueError("Fixture has identical teams")
                regular = is_regular(event, spec)
                if job["kind"] == "recovered_event" and not regular:
                    raise ValueError("Recovered event is outside the regular-season scope")
                if regular:
                    window = _mexico_window(spec)
                    if window:
                        stamp = event.get("startTimestamp")
                        if type(stamp) is not int or not window[0] <= stamp < window[2]:
                            raise ValueError("Mexican fixture kickoff is outside its football cycle")
                        selected = stamp < window[1] if spec["tournament"] == "Apertura" else stamp >= window[1]
                        shared = any(r and r.get("source_competition_id") == cid and r["season_id"] == sid
                                     for i in range(len(self.specs)) if (r := self._resolved(i)))
                        if not selected and not shared:
                            # The original round response still retains this
                            # opposite-half fixture. Its own endpoint selects it.
                            continue
                    else:
                        check_kickoff(event, spec)
                match_url(event)
                old = self.db.execute("SELECT * FROM events WHERE id=?", (eid,)).fetchone()
                if old:
                    previous = json.loads(old["payload"])
                    if (old["competition"], old["season"], old["round"], old["regular"], previous["homeTeam"]["id"], previous["awayTeam"]["id"]) != (cid, sid, job["round"], int(regular), home, away):
                        raise ValueError(f"Conflicting duplicate fixture identity: {eid}")
                self.db.execute("INSERT OR REPLACE INTO events VALUES(?,?,?,?,?,?,?)", (eid, cid, sid, job["round"],
                    int(regular), json.dumps(event, ensure_ascii=False, allow_nan=False), observed))
        else:
            raise ValueError("Unsupported archive discovery job")

    def _records(self, spec, resolved):
        if not resolved:
            return [], []
        records = self.db.execute("SELECT * FROM events WHERE competition=? AND season=?",
            (resolved.get("source_competition_id", spec["competition_id"]), resolved["season_id"]))
        window = _mexico_window(spec)
        if window:
            start, end = window[:2] if spec["tournament"] == "Apertura" else window[1:]
            records = [r for r in records if start <= json.loads(r["payload"])["startTimestamp"] < end]
        return select_regular_records(records)

    def _eligible_records(self, spec, resolved):
        rows, _ = self._records(spec, resolved)
        rows = [r for r in rows if is_completed_event(json.loads(r["payload"]))]
        return sorted(rows, key=lambda r: (json.loads(r["payload"])["startTimestamp"], r["id"]))

    def status(self):
        counts = dict(self.db.execute("SELECT state,COUNT(*) FROM jobs GROUP BY state"))
        complete = bool(counts) and set(counts) == {"complete"}
        seasons = []
        provider_identities = Counter((r.get("source_competition_id", spec["competition_id"]), r["season_id"])
            for i, spec in enumerate(self.specs) if (r := self._resolved(i)))
        for i, spec in enumerate(self.specs):
            if i in self._summaries:
                seasons.append(self._summaries[i])
                continue
            resolved = self._resolved(i)
            records, superseded = self._records(spec, resolved)
            events = [json.loads(r["payload"]) for r in records]
            raw_standings = self._value(f"standings:{i}")
            standings = {int(t): n for t, n in raw_standings.items()} if raw_standings else None
            if resolved and spec["competition_id"] == 38:
                response = self.db.execute("SELECT body FROM responses WHERE path=?",
                    (f"/unique-tournament/38/season/{resolved['season_id']}/standings/total",)).fetchone()
                standings = reviewed_standings(events, spec, standings, json.loads(response[0]) if response else None)
            issues = validate_schedule(events, spec, standings)
            if self._value(f"missing:{i}") and not resolved:
                issues.insert(0, self._value(f"missing:{i}"))
            if spec.get("scope_pending"):
                issues.insert(0, "Mixed J2/J3 scope decision is pending")
            eligible = []
            if not issues:
                try:
                    eligible = self._eligible_records(spec, resolved)
                except ValueError as error:
                    issues.append(str(error))
            source_cid = resolved.get("source_competition_id", spec["competition_id"]) if resolved else spec["competition_id"]
            summary = dict(league=spec["name"], season=spec["season"], competition_id=source_cid,
                season_id=resolved["season_id"] if resolved else None, stem=resolved["stem"] if resolved else None,
                regular_matches=len(events), finished=sum(is_completed_event(e) for e in events),
                selected_matches=len(eligible), statuses=dict(Counter(event_status(e) for e in events)),
                schedule_verified=not issues, validation_issues=issues, superseded_fixtures=superseded,
                snapshot=spec["snapshot"], selection=spec.get("selection", "all"), format_source=spec.get("source"))
            if resolved and resolved.get("source_competition_id"):
                summary.update(requested_competition_id=spec["competition_id"], format_source=resolved["source_review"])
            if resolved and provider_identities[source_cid, resolved["season_id"]] > 1:
                summary["shared_provider_identity"] = True
            self._summaries[i] = summary
            seasons.append(summary)
        return dict(discovery_id=self.config["discovery_id"], directory=str(self.directory), jobs=counts,
            requests=self.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0],
            discovery_complete=complete, ready_to_collect=complete and all(s["schedule_verified"] for s in seasons),
            seasons=seasons, campaign=self._value("campaign"), api_requests_sent_by_python=0)

    def prepare_collection(self, *, campaign_base=BASE):
        snapshot = self.status()
        if snapshot["campaign"]:
            return snapshot["campaign"]
        if not snapshot["ready_to_collect"]:
            raise ValueError("Resolve every requested season and its schedule before creating the campaign")
        ids, references = [], {}
        for i, spec in enumerate(self.specs):
            resolved = self._resolved(i)
            for record in self._eligible_records(spec, resolved):
                eid, event = record["id"], json.loads(record["payload"])
                if eid in references:
                    raise ValueError("One event was selected into two season cohorts")
                references[eid] = dict(kind="queue_identity", event=event, observed_at=record["observed"],
                    source=str(self.directory / "discovery.sqlite"), season_stem=resolved["stem"],
                    route_source="chrome_archive_discovery")
                ids.append(eid)
        # Recover a campaign created just before a power failure prevented the
        # final discovery-state write. Never create a second queue for that ID.
        candidates = []
        prefix = ("chrome-" + self.config["name"]).strip("-")[:40]
        for path in Path(campaign_base).glob(prefix + "-*/config.json"):
            candidate = read_json(path)
            if (candidate.get("source_metadata") or {}).get("discovery_id") == self.config["discovery_id"]:
                candidates.append((path.parent, candidate))
        if len(candidates) > 1:
            raise ValueError("Multiple campaigns refer to this discovery; inspect before proceeding")
        if candidates:
            directory, candidate = candidates[0]
            if (candidate["source_metadata"].get("collection") != "archive_discovery"
                    or candidate["source_metadata"].get("seasons") != snapshot["seasons"]
                    or [item["event_id"] for item in candidate["items"]] != ids
                    or [item["reference"] for item in candidate["items"]] != [references[eid] for eid in ids]
                    or candidate["profile"] != self.config["profile"] or candidate["policy"] != self.config["policy"]):
                raise ValueError("Existing archive campaign differs from the verified discovery")
            with Campaign(directory) as campaign:
                if not (directory / "status.json").exists():
                    campaign.write_status()
            result = dict(campaign_id=candidate["campaign_id"], directory=str(directory.resolve()),
                          matches=len(ids), browser_started=False, source_metadata=candidate["source_metadata"])
        else:
            result = prepare_campaign(ids, name="chrome-" + self.config["name"], base=campaign_base,
                references=references, profile=Profile(**self.config["profile"]), policy=self.config["policy"],
                source_metadata=dict(collection="archive_discovery", discovery_id=self.config["discovery_id"],
                    scope="regular_season", prepared_at=time.time(), discovery_requests=snapshot["requests"],
                    seasons=snapshot["seasons"]))
        result.pop("event_ids", None)
        with self.db:
            self._save("campaign", result)
        return result
