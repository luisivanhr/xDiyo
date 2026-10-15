from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid
from contextlib import contextmanager

from .models import Job
from .event_round import match_season


def season_year_pair(value):
    if value is None:
        return None
    import re
    # Calendar-year competitions (for example Japan through 2026) occupy one
    # year. Keep the existing two-year identity used by queues and manifests.
    if re.fullmatch(r"(?:19|20)\d{2}", str(value).strip()):
        year = int(str(value).strip())
        return year, year
    match = re.search(r"(\d{2,4})\s*[/\u2013-]\s*(\d{2,4})", str(value))
    if not match:
        return None
    result = []
    for item in match.groups():
        year = int(item)
        result.append(year + 2000 if year < 100 else year)
    return tuple(result)


def json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def atomic_write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class Store:
    """Raw cache, event registry and shared request ledger.

    All network commands must hold run_lock for their entire run, using the same
    store root across seasons. A lock is held by the OS, never inferred from a
    leftover file. Read-only planning and export do not need the run lock.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "manifest.sqlite", timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS components (
              key TEXT PRIMARY KEY, path TEXT NOT NULL, component TEXT NOT NULL,
              event_id INTEGER, team_id INTEGER, locale TEXT NOT NULL,
              status TEXT NOT NULL, last_attempt REAL, http_status INTEGER,
              error TEXT, attempts INTEGER NOT NULL DEFAULT 0,
              raw_path TEXT, raw_hash TEXT, last_success REAL, next_due REAL
            );
            CREATE TABLE IF NOT EXISTS attempts (
              id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, key TEXT NOT NULL,
              started REAL NOT NULL, status INTEGER, outcome TEXT
            );
            CREATE INDEX IF NOT EXISTS attempts_started ON attempts(started);
            CREATE INDEX IF NOT EXISTS attempts_run ON attempts(run_id);
            CREATE TABLE IF NOT EXISTS response_versions (
              id INTEGER PRIMARY KEY, key TEXT NOT NULL, fetched REAL NOT NULL,
              raw_path TEXT NOT NULL, raw_hash TEXT NOT NULL, event_json TEXT
            );
            CREATE INDEX IF NOT EXISTS response_key ON response_versions(key, fetched);
            CREATE TABLE IF NOT EXISTS events (
              event_id INTEGER PRIMARY KEY, payload TEXT NOT NULL,
              observed REAL NOT NULL, locale TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memberships (
              event_id INTEGER NOT NULL REFERENCES events(event_id),
              league_id INTEGER NOT NULL, season_id INTEGER NOT NULL,
              discovery_key TEXT NOT NULL, stage_json TEXT NOT NULL,
              PRIMARY KEY(event_id, league_id, season_id, discovery_key, stage_json)
            );
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS runs (
              run_id TEXT PRIMARY KEY, started REAL NOT NULL, ended REAL,
              status TEXT NOT NULL, summary TEXT
            );
        """)
        self.db.commit()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.db.close()

    @contextmanager
    def run_lock(self):
        stream = (self.root / "collector.lock").open("a+b")
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            stream.close()
            raise RuntimeError("Another collector holds this store's request ledger") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()

    def get(self, job: Job):
        row = self.db.execute("SELECT * FROM components WHERE key=?", (job.key,)).fetchone()
        return dict(row) if row else None

    def payload(self, job: Job):
        row = self.get(job)
        if not row or not row["raw_path"]:
            return None
        data = gzip.decompress((self.root / row["raw_path"]).read_bytes())
        if hashlib.sha256(data).hexdigest() != row["raw_hash"]:
            raise ValueError(f"Raw cache integrity failure: {job.key}")
        return json.loads(data)

    def due(self, job: Job, now: float, refresh: bool = False) -> bool:
        if refresh:
            return True
        row = self.get(job)
        if not row:
            return True
        if row["next_due"] is not None and now < row["next_due"]:
            return False
        # A failed refresh is due even when last-good data remain available.
        return row["status"] not in {"success", "empty", "unavailable"} or row["next_due"] is not None

    def record(self, job: Job, status: str, now: float, *, payload=None,
               http_status=None, error=None, next_due=None, event_context=None):
        raw_path = raw_hash = None
        if status in {"success", "empty"}:
            body = json_bytes(payload)
            raw_hash = hashlib.sha256(body).hexdigest()
            raw_path = f"raw/{raw_hash[:2]}/{raw_hash}.json.gz"
            target = self.root / raw_path
            if not target.exists():
                atomic_write(target, gzip.compress(body, mtime=0))
        with self.db:
            if raw_path:
                self.db.execute("INSERT INTO response_versions(key,fetched,raw_path,raw_hash,event_json) VALUES (?,?,?,?,?)",
                    (job.key,now,raw_path,raw_hash,json_bytes(event_context).decode() if event_context is not None else None))
            self.db.execute("""
                INSERT INTO components
                  (key,path,component,event_id,team_id,locale,status,last_attempt,
                   http_status,error,attempts,raw_path,raw_hash,last_success,next_due)
                VALUES (?,?,?,?,?,?,?,?,?,?,1,?,?,?,?)
                ON CONFLICT(key) DO UPDATE SET status=excluded.status,
                  last_attempt=excluded.last_attempt,http_status=excluded.http_status,
                  error=excluded.error,attempts=components.attempts+1,
                  raw_path=COALESCE(excluded.raw_path,components.raw_path),
                  raw_hash=COALESCE(excluded.raw_hash,components.raw_hash),
                  last_success=COALESCE(excluded.last_success,components.last_success),
                  next_due=excluded.next_due
                """, (job.key,job.path,job.component,job.event_id,job.team_id,job.locale,
                       status,now,http_status,error,raw_path,raw_hash,
                       now if raw_path else None,next_due))

    def snapshots(self, job):
        rows=self.db.execute("SELECT * FROM response_versions WHERE key=? ORDER BY fetched,id",(job.key,)).fetchall()
        result=[]
        for row in rows:
            data=gzip.decompress((self.root/row["raw_path"]).read_bytes())
            if hashlib.sha256(data).hexdigest()!=row["raw_hash"]:
                raise ValueError(f"Snapshot hash mismatch: {job.key}")
            result.append({"payload":json.loads(data),"fetched":row["fetched"],"raw_hash":row["raw_hash"],
                           "event":json.loads(row["event_json"]) if row["event_json"] else None})
        return result

    def archive_diagnostic(self, job, payload, now, error):
        """Retain a changed JSON shape without replacing the last-good cache."""
        body = json_bytes(payload)
        digest = hashlib.sha256(body).hexdigest()
        directory = self.root/"diagnostics"
        raw = directory/(digest+".json.gz")
        if not raw.exists():
            atomic_write(raw, gzip.compress(body, mtime=0))
        record = directory/(uuid.uuid4().hex+".json")
        atomic_write(record, json_bytes({"source_key":job.key,"observed_at":now,"error":error,
            "raw_file":raw.name,"raw_sha256":digest}))
        return str(record.relative_to(self.root))

    def add_events(self, events: list[dict], scope, job: Job, now: float):
        with self.db:
            for event in events:
                if not isinstance(event.get("id"), int):
                    raise ValueError("Discovered event has no integer ID")
                season=match_season(event).get("id")
                season_year=match_season(event).get("year")
                competition=((event.get("tournament") or {}).get("uniqueTournament") or {}).get("id")
                if season is not None and season!=scope.season_id:
                    raise ValueError("Discovered event belongs to another season")
                if competition is not None and competition!=scope.league_id:
                    raise ValueError("Discovered event belongs to another competition")
                if scope.season_start is not None and scope.season_end is not None:
                    expected=(scope.season_start,scope.season_end)
                    if season_year_pair(season_year) != expected:
                        raise ValueError(
                            f"Discovered event season year {season_year!r} does not match {expected[0]}/{expected[1]}")
                old=self.db.execute("SELECT payload,observed FROM events WHERE event_id=?",(event["id"],)).fetchone()
                if old and now >= old["observed"]:
                    previous=json.loads(old["payload"])
                    relevant=("status","startTimestamp","homeTeam","awayTeam","homeScore","awayScore","time")
                    if any(previous.get(k)!=event.get(k) for k in relevant):
                        self.db.execute("UPDATE components SET next_due=? WHERE event_id=? AND component!='event'",(now,event["id"]))
                self.db.execute("""INSERT INTO events VALUES (?,?,?,?)
                    ON CONFLICT(event_id) DO UPDATE SET payload=excluded.payload,
                    observed=excluded.observed,locale=excluded.locale
                    WHERE excluded.observed >= events.observed""",
                    (event["id"], json_bytes(event).decode(), now, job.locale))
                stage = {k:event[k] for k in ("roundInfo", "tournament", "season") if k in event}
                self.db.execute("INSERT OR IGNORE INTO memberships VALUES (?,?,?,?,?)",
                                (event["id"],scope.league_id,scope.season_id,job.key,json_bytes(stage).decode()))

    def events(self, league_id=None, season_id=None):
        if (league_id is None) != (season_id is None):
            raise ValueError("Filter by both competition and season")
        if league_id is None:
            rows = self.db.execute("SELECT * FROM events")
        else:
            rows = self.db.execute("""SELECT DISTINCT e.* FROM events e JOIN memberships m
                ON e.event_id=m.event_id WHERE m.league_id=? AND m.season_id=?""", (league_id,season_id))
        return sorted((json.loads(r["payload"]) for r in rows),
                      key=lambda e:(e.get("startTimestamp") or float("inf"), e["id"]))

    def event_observed_at(self, event_id):
        row = self.db.execute("SELECT observed FROM events WHERE event_id=?", (event_id,)).fetchone()
        return row["observed"] if row else None

    def freeze_unavailable(self, league_id, season_id):
        """Make observed historical 404s terminal without changing their status."""
        with self.db:
            self.db.execute("""UPDATE components SET next_due=NULL
                WHERE status='unavailable' AND event_id IN
                (SELECT event_id FROM memberships WHERE league_id=? AND season_id=?)""",
                (league_id,season_id))

    def setting(self, key, default=None):
        row = self.db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_setting(self, key, value):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (key,json.dumps(value)))

    def start_run(self, now=None):
        run_id = uuid.uuid4().hex
        with self.db:
            self.db.execute("INSERT INTO runs VALUES (?,?,NULL,'running',NULL)",
                            (run_id,time.time() if now is None else now))
        return run_id

    def finish_run(self, run_id, status, summary, now=None):
        with self.db:
            self.db.execute("UPDATE runs SET ended=?,status=?,summary=? WHERE run_id=?",
                            (time.time() if now is None else now,status,json.dumps(summary),run_id))
