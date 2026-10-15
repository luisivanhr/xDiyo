"""Read awarded flags from saved Chrome observations, without provider requests."""
from __future__ import annotations

from contextlib import ExitStack, closing
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sqlite3

from scraping.sofascore.parsers import awarded_flag, validate_payload
from .capture import api_path
from .import_har import response_json, response_time


@dataclass(frozen=True)
class AwardedEvidence:
    value: bool | None = None
    observed_at: float | None = None
    source: str | None = None
    digest: str | None = None


class SavedAwardedFlags:
    """Use the latest valid event body across read-only campaign databases.

    A later HTTP failure does not erase an earlier successful observation.
    Missing bodies remain unknown; conflicting identities or equally recent
    flags fail closed. Nothing in the capture databases is changed.
    """

    def __init__(self, databases):
        self.paths = sorted({Path(path).resolve() for path in databases})
        self.stack = ExitStack()
        self.connections = []

    def __enter__(self):
        try:
            for path in self.paths:
                connection = self.stack.enter_context(closing(sqlite3.connect(
                    path.as_uri() + "?mode=ro", uri=True)))
                self.connections.append((path, connection))
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)

    def get(self, match):
        eid = match["event_id"]
        endpoint = f"/event/{eid}"
        expected = tuple(match[key] for key in
                         ("event_id", "competition_id", "season_id", "home_id", "away_id"))
        best = AwardedEvidence()
        for path, connection in self.connections:
            candidate = None
            for digest, observed, encoded in connection.execute(
                    "SELECT digest,observed,entry_json FROM observations "
                    "WHERE event_id=? AND path=? ORDER BY observed DESC", (eid, endpoint)):
                if candidate is not None and observed < candidate.observed_at:
                    break
                if hashlib.sha256(encoded.encode("utf-8")).hexdigest() != digest:
                    raise ValueError(f"Captured metadata hash mismatch for {eid}")
                entry = json.loads(encoded)
                request = entry.get("request", {})
                if (request.get("method") != "GET" or api_path(request.get("url", "")) != endpoint or
                        entry.get("_queryPresent") or entry.get("_xdiyoSelected") is False or
                        entry.get("response", {}).get("status") not in (200, 304)):
                    continue
                try:
                    payload = response_json(entry)
                    validate_payload("event", payload)
                except (ValueError, TypeError, KeyError):
                    continue
                event = payload["event"]
                actual = (event["id"], ((event.get("tournament") or {}).get("uniqueTournament") or {}).get("id"),
                          (event.get("season") or {}).get("id"), (event.get("homeTeam") or {}).get("id"),
                          (event.get("awayTeam") or {}).get("id"))
                if actual != expected or any(type(value) is not int for value in actual):
                    raise ValueError(f"Captured fixture identity mismatch for {eid}")
                if not math.isfinite(observed) or abs(response_time(entry) - observed) > 0.000001:
                    raise ValueError(f"Captured observation time mismatch for {eid}")
                current = AwardedEvidence(awarded_flag(event), observed, str(path), digest)
                if candidate is not None and current.value != candidate.value:
                    raise ValueError(f"Conflicting equally recent awarded flags for {eid}")
                candidate = current
            if candidate is not None:
                if best.observed_at == candidate.observed_at and best.value != candidate.value:
                    raise ValueError(f"Conflicting equally recent awarded flags for {eid}")
                if best.observed_at is None or candidate.observed_at > best.observed_at:
                    best = candidate
        return best
