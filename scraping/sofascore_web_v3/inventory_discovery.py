"""Chrome-captured season/stage evidence before an archive selection is known.

This discovery cannot prepare a collection campaign. It retains season lists,
round lists and standings without interpreting unfamiliar competition formats.
The reviewed archive still has to pass its own fixture validation afterwards.
"""
from pathlib import Path
import json
import time
import uuid
import hashlib

from scraping.sofascore.store import atomic_write, json_bytes, season_year_pair
from .campaign import campaign_path, collector_lock, read_json, validate_policy
from .discovery import DISCOVERY_BASE, Discovery, normalize_leagues, positive
from .event_pages import validate_event_page, validate_event_pages, review_boundary_overlap


def prepare_inventory(name, leagues, *, first_year, last_year, base=DISCOVERY_BASE):
    """Prepare an idempotent local inventory; only the extension sends requests."""
    leagues = [{k: row[k] for k in ("name", "competition_id")}
               for row in normalize_leagues(leagues)]
    if (type(first_year) is not int or type(last_year) is not int
            or not 1900 <= first_year <= last_year <= 2100):
        raise ValueError("Choose an explicit inclusive season-start year range")
    # Validate the operator-supplied name before using it in either path.
    campaign_path("inventory-" + name, base)
    settings = dict(name=name, leagues=leagues, first_year=first_year, last_year=last_year,
                    mode="inventory", scope="season_and_stage_review_only",
                    policy=validate_policy({"collection_mode": "session",
                        "session_challenge_recovery": True, "memory_reset_every_matches": 50}))
    pointer = campaign_path("inventory-" + name, base) / "prepared.json"
    with collector_lock(pointer.parent / "prepare.lock"):
        if pointer.exists():
            saved = read_json(pointer)
            config = read_json(Path(saved["directory"]) / "config.json")
            if any(config.get(k) != v for k, v in settings.items()):
                raise ValueError("Saved inventory settings differ; preserve its captured evidence")
            return saved
        discovery_id = "discover-inventory-" + uuid.uuid4().hex[:12]
        directory = campaign_path(discovery_id, base)
        config = dict(settings, discovery_id=discovery_id, created_at=time.time())
        atomic_write(directory / "config.json", json_bytes(config))
        with InventoryDiscovery(directory) as discovery:
            result = discovery.status()
        saved = {k: result[k] for k in ("discovery_id", "directory")}
        atomic_write(pointer, json_bytes(saved))
        return saved


class InventoryDiscovery(Discovery):
    # Only a reviewed metadata-only 404 can become unavailable. Collection
    # discovery never treats an unavailable job as completed fixture evidence.
    terminal_job_states = ("complete", "unavailable")

    def review_page_overlap(self, competition_id, season_id, *, event_id, reason):
        """Review retained evidence while disconnected; never rewrite responses."""
        cid, sid = positive(competition_id), positive(season_id)
        if self.current or cid not in self.leagues or not isinstance(reason, str) or len(reason.strip()) < 20:
            raise ValueError("Disconnect and provide an explicit page-boundary review reason")
        rows = list(self.db.execute("SELECT * FROM jobs WHERE competition=? AND season=? AND kind='event_pages_last' ORDER BY round",
                                   (cid, sid)))
        pages = []
        for job in rows:
            saved = self.db.execute("SELECT * FROM responses WHERE path=?", (job["path"],)).fetchone()
            if not saved or hashlib.sha256(saved["body"].encode()).hexdigest() != saved["sha256"]:
                raise ValueError("Missing or changed original page evidence")
            pages.append({"page": job["round"], "payload": json.loads(saved["body"])})
        review = review_boundary_overlap(pages, competition_id=cid, season_id=sid, event_id=event_id)
        validate_event_pages(pages, competition_id=cid, season_id=sid, reviewed_overlaps=[review])
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO state VALUES(?,?)", (
                f"page_overlap_review:{cid}:{sid}:last", json.dumps({"reviews": [review],
                    "reason": reason, "reviewed_at": time.time(), "not_fixture_validation": True})))
        return review

    def acknowledge_unavailable_rounds(self, competition_id, season_id, *, reason):
        """Retain an observed rounds 404 while allowing other inventory evidence.

        Call while disconnected under the collector lock. This cannot acknowledge
        access denials, event requests, or missing capture bodies as successful.
        """
        path = f"/unique-tournament/{positive(competition_id)}/season/{positive(season_id)}/rounds"
        job = self.db.execute("SELECT * FROM jobs WHERE path=?", (path,)).fetchone()
        last = self.db.execute("SELECT * FROM attempts WHERE path=? ORDER BY id DESC LIMIT 1", (path,)).fetchone()
        if (self.current or not isinstance(reason, str) or len(reason.strip()) < 20
                or not job or job["kind"] != "rounds" or job["state"] not in {"failed", "unavailable"}
                or not last or last["http_status"] != 404):
            raise ValueError("Only a retained rounds-metadata HTTP 404 may be acknowledged")
        with self.db:
            self.db.execute("UPDATE jobs SET state='unavailable',lease=NULL WHERE path=?", (path,))
            self.db.execute("INSERT OR REPLACE INTO state VALUES(?,?)", ("unavailable:" + path,
                json.dumps({"path": path, "http_status": 404, "observed": last["observed"],
                            "reviewed_at": time.time(), "reason": reason, "not_fixture_validation": True})))

    def inspect_event_pages(self, competition_id, season_id, *, directions=("last",)):
        """Queue bounded page inspection for an already captured exact season.

        The original metadata and its selection stay unchanged. All returned
        events remain evidence only; this mode cannot create a collection queue.
        """
        cid, sid = positive(competition_id), positive(season_id)
        if self.current or not directions or len(set(directions)) != len(directions) or set(directions) - {"last", "next"}:
            raise ValueError("Disconnect before selecting explicit page directions")
        row = self.db.execute("SELECT value FROM state WHERE key=?", (f"inventory:{cid}",)).fetchone()
        if cid not in self.leagues or not row or sum(s["id"] == sid for s in json.loads(row[0])) != 1:
            raise ValueError("Page inspection requires one exact captured in-scope season")
        with self.db:
            for direction in directions:
                self.add_job(f"/unique-tournament/{cid}/season/{sid}/events/{direction}/0",
                             cid, "event_pages_" + direction, sid, 0)

    def observe(self, path, lease, entry):
        leased = bool(self.current and (path, lease) == (self.current["path"], self.current["lease"]))
        try:
            return super().observe(path, lease, entry)
        except ValueError:
            # MLS's site uses season pages while older seasons have no rounds
            # endpoint. Only explicit evidence inspection may continue after
            # this metadata 404. The attempt stays failed/unavailable, and no
            # event or collection validation is waived.
            job = self.db.execute("SELECT * FROM jobs WHERE path=?", (path,)).fetchone()
            if (not leased or entry.get("response", {}).get("status") != 404 or not job
                    or job["kind"] != "rounds" or job["competition"] != 242
                    or job["state"] != "failed" or job["lease"] != lease
                    or not self.db.execute("SELECT 1 FROM jobs WHERE competition=242 AND season=? AND kind='event_pages_last'",
                                           (job["season"],)).fetchone()):
                raise
            self.acknowledge_unavailable_rounds(242, job["season"], reason=
                "MLS rounds endpoint returned HTTP 404 during explicit season-page inspection; original failure retained. Fixture validation still required.")
            with self.db:
                self.db.execute("DELETE FROM state WHERE key='pause'")
            return {"saved": False, "metadata_unavailable": True, "summary": self.status()}

    def process(self, job, payload, observed):
        cid, sid, kind = job["competition"], job["season"], job["kind"]
        if kind == "seasons":
            values = payload.get("seasons")
            if not isinstance(values, list) or not values:
                raise ValueError("Missing season inventory")
            selected, seen = [], set()
            for value in values:
                if not isinstance(value, dict):
                    raise ValueError("Malformed season inventory entry")
                season_id = positive(value.get("id"))
                if season_id in seen:
                    raise ValueError("Duplicate provider season identity")
                seen.add(season_id)
                years = season_year_pair(value.get("year"))
                if (years and years[1] in (years[0], years[0] + 1)
                        and self.config["first_year"] <= years[0] <= self.config["last_year"]):
                    selected.append(value)
                    self.add_job(f"/unique-tournament/{cid}/season/{season_id}/rounds",
                                 cid, "rounds", season_id)
                    self.add_job(f"/unique-tournament/{cid}/season/{season_id}/standings/total",
                                 cid, "standings", season_id)
            self.db.execute("INSERT OR REPLACE INTO state VALUES(?,?)",
                            (f"inventory:{cid}", json.dumps(selected)))
        elif kind == "rounds":
            if not isinstance(payload.get("rounds"), list):
                raise ValueError("Missing round inventory")
        elif kind == "standings":
            if not isinstance(payload.get("standings"), list):
                raise ValueError("Missing standings inventory")
            for table in payload["standings"]:
                if (not isinstance(table, dict)
                        or table.get("tournament", {}).get("uniqueTournament", {}).get("id") != cid):
                    raise ValueError("Standings belong to another competition")
        elif kind in {"event_pages_last", "event_pages_next"}:
            direction, page = kind.removeprefix("event_pages_"), job["round"]
            if (cid not in self.leagues or type(sid) is not int or sid <= 0
                    or type(page) is not int or not 0 <= page < 4096
                    or job["path"] != f"/unique-tournament/{cid}/season/{sid}/events/{direction}/{page}"):
                raise ValueError("Invalid season-event page identity")
            values = validate_event_page(payload, competition_id=cid, season_id=sid)
            if payload["hasNextPage"]:
                if page == 4095:
                    raise ValueError("Excessive event page chain")
                self.add_job(f"/unique-tournament/{cid}/season/{sid}/events/{direction}/{page + 1}",
                             cid, kind, sid, page + 1)
            else:
                pages = []
                for number in range(page + 1):
                    path = f"/unique-tournament/{cid}/season/{sid}/events/{direction}/{number}"
                    saved = self.db.execute("SELECT * FROM responses WHERE path=?", (path,)).fetchone()
                    if not saved or hashlib.sha256(saved["body"].encode()).hexdigest() != saved["sha256"]:
                        raise ValueError("Missing or changed original page evidence")
                    pages.append({"page": number, "payload": json.loads(saved["body"])})
                # A terminal empty first page is legitimate evidence of no
                # upcoming fixtures, never a complete historical season.
                saved_review = self.db.execute("SELECT value FROM state WHERE key=?",
                    (f"page_overlap_review:{cid}:{sid}:{direction}",)).fetchone()
                reviews = json.loads(saved_review[0])["reviews"] if saved_review else []
                captured = validate_event_pages(pages, competition_id=cid, season_id=sid,
                    reviewed_overlaps=reviews) if values or page else []
                self.db.execute("INSERT OR REPLACE INTO state VALUES(?,?)", (
                    f"page_inventory:{cid}:{sid}:{direction}", json.dumps({"competition_id": cid,
                        "season_id": sid, "direction": direction, "pages": page + 1,
                        "events": len(captured), "complete": True, "observed": observed,
                        "reviewed_overlaps": reviews})))
        elif kind == "round_inspection":
            # Explicit diagnostics retain an anomalous source bucket without
            # adding its fixtures to any collectible schedule.
            if (cid not in self.leagues or type(sid) is not int or sid <= 0
                    or type(job["round"]) is not int or not 0 <= job["round"] <= 300
                    or job["path"] != f"/unique-tournament/{cid}/season/{sid}/events/round/{job['round']}"):
                raise ValueError("Invalid round inspection identity")
            super().process(job, payload, observed)
        else:
            raise ValueError("Inventory permits only season, round and standings metadata")

    def status(self):
        counts = dict(self.db.execute("SELECT state,COUNT(*) FROM jobs GROUP BY state"))
        seasons = []
        for cid, league in self.leagues.items():
            row = self.db.execute("SELECT value FROM state WHERE key=?", (f"inventory:{cid}",)).fetchone()
            for season in json.loads(row[0]) if row else []:
                seasons.append(dict(league=league["name"], competition_id=cid,
                    season=season.get("year"), season_id=season["id"],
                    schedule_verified=False, selected_matches=0,
                    validation_issues=["Metadata inventory only; explicit format and fixture review required"]))
        return dict(discovery_id=self.config["discovery_id"], directory=str(self.directory),
            jobs=counts, requests=self.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0],
            discovery_complete=bool(counts) and set(counts) <= set(self.terminal_job_states),
            ready_to_collect=False, campaign=None, seasons=seasons,
            api_requests_sent_by_python=0, scope="season_and_stage_review_only",
            unavailable_metadata=[json.loads(r[0]) for r in self.db.execute("SELECT value FROM state WHERE key LIKE 'unavailable:%'")],
            event_page_inventories=[json.loads(r[0]) for r in self.db.execute("SELECT value FROM state WHERE key LIKE 'page_inventory:%'")])

    def prepare_collection(self, *, campaign_base=None):
        raise ValueError("Metadata inventory cannot create collection; prepare and validate the reviewed archive")
