"""Exact archive identities and phase tables, using captured metadata only."""
import json

from scraping.sofascore.store import season_year_pair
from .discovery import positive


def validate_selector_identities(cohorts):
    """Allow distinct pinned seasons in one year; reject overlapping selectors."""
    identities, pins = {}, set()
    for spec in cohorts:
        cid = positive(spec["competition_id"])
        pin = positive(spec["provider_season_id"]) if "provider_season_id" in spec else None
        if pin is not None:
            if (cid, pin) in pins:
                raise ValueError("Duplicate exact provider season selector")
            pins.add((cid, pin))
        own = set()
        for alias in spec["aliases"]:
            years = season_year_pair(alias)
            if years is None or years[1] not in (years[0], years[0] + 1):
                raise ValueError("Invalid calendar-year or two-year season")
            key = cid, years
            if key in own:
                raise ValueError("Overlapping aliases within one season selector")
            own.add(key)
            previous = identities.get(key, [])
            if previous and (pin is None or None in previous or pin in previous):
                raise ValueError("Overlapping season selectors; pin every same-year tournament explicitly")
            identities.setdefault(key, []).append(pin)


def resolve_captured_season(values, spec):
    """A saved pin never falls back to another season with the same year label."""
    if not isinstance(values, list):
        raise ValueError("Missing season inventory")
    aliases = {season_year_pair(a) for a in spec["aliases"]}
    pin = positive(spec["provider_season_id"]) if "provider_season_id" in spec else None
    if pin is None:
        found = [s for s in values if isinstance(s, dict) and season_year_pair(s.get("year")) in aliases]
    else:
        found = [s for s in values if isinstance(s, dict) and s.get("id") == pin]
        if found and any(season_year_pair(s.get("year")) not in aliases for s in found):
            raise ValueError("Exact provider season has a conflicting year")
    if not found:
        return None
    if len(found) != 1:
        raise ValueError(f"{spec['name']} {spec['season']}: season ambiguous; no implicit fallback")
    positive(found[0].get("id"))
    return found[0]


def select_captured_tables(payload, *, competition_id, selectors):
    """Audit exact table identities, without flattening ordinary repeated phases.

    This helper does not classify fixture events or approve collection. A table
    name, ID or tournament mismatch requires review rather than fuzzy matching.
    """
    positive(competition_id)
    tables = payload.get("standings")
    if not isinstance(tables, list) or not selectors:
        raise ValueError("Missing standings or explicit table selectors")
    selected, fingerprints, ids = [], {}, set()
    for selector in selectors:
        sid, tid = positive(selector["standings_id"]), positive(selector["tournament_id"])
        if sid in ids or not isinstance(selector.get("name"), str):
            raise ValueError("Duplicate or malformed table selector")
        ids.add(sid)
        matches = [t for t in tables if t.get("id") == sid]
        if len(matches) != 1:
            raise ValueError("Selected ordinary table missing or duplicated")
        table = matches[0]
        tournament = table.get("tournament", {})
        if (table.get("name") != selector["name"] or table.get("type") != "total"
                or tournament.get("id") != tid
                or tournament.get("uniqueTournament", {}).get("id") != competition_id):
            raise ValueError("Selected table has a conflicting identity")
        rows = table.get("rows")
        if not isinstance(rows, list) or len(rows) < 2:
            raise ValueError("Selected table has no complete club roster")
        by_id = {}
        for row in rows:
            team = positive(row.get("team", {}).get("id"))
            if team in by_id or type(row.get("matches")) is not int or row["matches"] < 0:
                raise ValueError("Invalid or duplicate standings club")
            by_id[team] = {k: row.get(k) for k in
                          ("matches", "wins", "draws", "losses", "scoresFor", "scoresAgainst", "points")}
        # A duplicate must be explicitly reviewed. Identical stage totals alone
        # do not authorize merging separately selected tables.
        fingerprint = tid, json.dumps(by_id, sort_keys=True)
        duplicate_of = selector.get("duplicate_of")
        if duplicate_of is not None:
            positive(duplicate_of)
            if duplicate_of not in fingerprints or fingerprints[duplicate_of] != fingerprint:
                raise ValueError("Reviewed duplicate table does not match its selected original")
        else:
            selected.append(table)
        fingerprints[sid] = fingerprint
    return selected
