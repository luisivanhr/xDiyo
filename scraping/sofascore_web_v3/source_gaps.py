"""Exact reviewed source gaps, with separately identified external observations.

These are played fixtures, not awards or synthetic Sofascore events. They add
only schedule counters during discovery. Only real provider IDs enter Chrome's
queue. The publisher appends the pinned observations with internal dataset IDs.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

from scraping.sofascore.discovery import is_completed_event
from scraping.sofascore.schema import TABLES
from scraping.sofascore.statistics_backfill import _row
from scraping.sofascore.store import json_bytes

STEM = "Greek_Super_League_20_21"
REVIEW = Path(__file__).with_name("reviewed_sources") / "greek_2020_21_flashscore.json"
REVIEW_SHA256 = "025b40b7520be01671f3b2f01c3e2f687266e45c8a1a91b558d56c3f619614e1"
IDENTITIES = {
    "z99UJF7d": (3248, 3241, "2021-01-23T17:30:00Z", 2, 0),
    "2F1GsJa3": (6349, 267459, "2021-01-24T15:15:00Z", 0, 1),
    "dIAQKehj": (3251, 3250, "2021-01-24T17:30:00Z", 2, 2),
}
NAMES = {
    3248: ("Panathinaikos",), 3241: ("OFI Crete", "OFI"),
    6349: ("Giannina", "PAS Giannina"), 267459: ("Volos",),
    3251: ("PAOK",), 3250: ("AEK",),
}
# Keep provider-specific definitions distinct where no native equivalence has
# been reviewed. In particular do not reinterpret free kicks as foul counts.
LABELS = {
    "Posesión": ("ballPossession", "Ball possession", "Match overview"),
    "Remates totales": ("totalShotsOnGoal", "Total shots", "Shots"),
    "Remates a puerta": ("shotsOnGoal", "Shots on target", "Shots"),
    "Remates fuera": ("shotsOffGoal", "Shots off target", "Shots"),
    "Córneres": ("cornerKicks", "Corner kicks", "Match overview"),
    "Tarjetas amarillas": ("yellowCards", "Yellow cards", "Match overview"),
    "Tarjetas rojas": ("redCards", "Red cards", "Match overview"),
    "Fueras de juego": ("offsides", "Offsides", "Attack"),
    "Faltas": ("fouls", "Fouls", "Match overview"),
    "Paradas": ("goalkeeperSaves", "Total saves", "Goalkeeping"),
    "Tiros libres": ("flashscoreFreeKicks", "Free kicks (Flashscore)", "Attack"),
    "Saques de banda": ("flashscoreThrowIns", "Throw-ins (Flashscore)", "Attack"),
}
for english, spanish in {
    "Ball possession": "Posesión", "Total shots": "Remates totales",
    "Shots on target": "Remates a puerta", "Shots off target": "Remates fuera",
    "Corner kicks": "Córneres", "Yellow cards": "Tarjetas amarillas",
    "Red cards": "Tarjetas rojas", "Offsides": "Fueras de juego",
    "Fouls": "Faltas", "Goalkeeper saves": "Paradas",
    "Free kicks": "Tiros libres", "Throw-ins": "Saques de banda",
}.items():
    LABELS[english] = LABELS[spanish]


def flashscore_event_id(match_id):
    """Reversible eight-character base62 ID in a separate int64 namespace.

    This is an internal dataset identity, never a Sofascore event ID or URL.
    It is disjoint from the existing [10**12, 2*10**12) workbook namespace.
    """
    if not isinstance(match_id, str) or not re.fullmatch(r"[0-9A-Za-z]{8}", match_id):
        raise ValueError("Expected the observed eight-character Flashscore match ID")
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    value = 0
    for char in match_id:
        value = value * 62 + alphabet.index(char)
    return 10**15 + value


def _stamp(value):
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("Source timestamps require a timezone")
    return stamp.timestamp()


def _validate_observation(observation):
    sid = observation["source_match_id"]
    home, away, kickoff, hs, aws = IDENTITIES[sid]
    url = urlparse(observation["url"])
    if (url.scheme != "https" or url.netloc not in {"www.flashscore.es", "www.flashscore.com"}
            or not any(p in url.path for p in ("/partido/futbol/", "/match/football/"))
            or parse_qs(url.query).get("mid") != [sid]):
        raise ValueError("Flashscore observation URL does not identify the reviewed match")
    if (observation.get("source") != "flashscore" or observation.get("capture_method") != "visible_browser_dom"
            or observation.get("round") != 18 or observation.get("competition") != "Superliga"
            or observation.get("status_text", "").casefold() not in {"finished", "finalizado"}
            or observation.get("home_name") not in NAMES[home] or observation.get("away_name") not in NAMES[away]
            or (observation.get("home_score"), observation.get("away_score")) != (hs, aws)
            or observation.get("kickoff_utc") != kickoff):
        raise ValueError("Flashscore observation differs from the independently reviewed fixture")
    local = datetime.strptime(observation["displayed_kickoff"], "%d.%m.%Y %H:%M").replace(
        tzinfo=ZoneInfo(observation["display_timezone"]))
    if local.timestamp() != _stamp(kickoff) or _stamp(observation["observed_at"]) <= _stamp(kickoff):
        raise ValueError("Source observation time or displayed kickoff is inconsistent")
    scores = observation.get("period_scores", {})
    if scores:
        if (set(scores) != {"1ST", "2ND"} or any(len(s) != 2 or any(type(n) is not int or n < 0 for n in s)
                                                 for s in scores.values())
                or [sum(scores[p][i] for p in scores) for i in (0, 1)] != [hs, aws]):
            raise ValueError("Observed period scores do not reconcile with the final score")
    return home, away


def observation_tables(observation):
    """Normalize observed values only; duplicates must agree, absent values stay absent."""
    home, away = _validate_observation(observation)
    sid = observation["source_match_id"]
    eid = flashscore_event_id(sid)
    digest = hashlib.sha256(json_bytes(observation)).hexdigest()
    observed = _stamp(observation["observed_at"])
    context = dict(source_key=observation["url"], raw_hash=digest, observed_at=observed)
    match = dict.fromkeys(TABLES["matches"])
    match.update(event_id=eid, competition_id=185, season_id=32553, season_year="20/21",
        status="finished", status_code=100, round=18, home_id=home, away_id=away,
        home_name=observation["home_name"], away_name=observation["away_name"],
        kickoff_utc=_stamp(observation["kickoff_utc"]), event_metadata_observed_at=observed)
    for i, side in enumerate(("home", "away")):
        match[side + "_score_current"] = float(observation[side + "_score"])
        match[side + "_score_display"] = float(observation[side + "_score"])
        for period, field in (("1ST", "period1"), ("2ND", "period2")):
            if period in observation.get("period_scores", {}):
                match[side + "_score_" + field] = float(observation["period_scores"][period][i])
    rows, seen = [], {}
    periods = observation["periods"]
    if not periods or not set(periods) <= {"ALL", "1ST", "2ND"} or "ALL" not in periods:
        raise ValueError("Unrecognized or missing observed statistics periods")
    for period, raw_rows in periods.items():
        if not raw_rows:
            raise ValueError("An observed statistics period cannot be empty")
        for raw in raw_rows:
            parsed = re.fullmatch(r"(\d+%?)\s*([^\d]+?)\s*(\d+%?)", raw)
            if not parsed or parsed[2].strip() not in LABELS:
                raise ValueError(f"Unreviewed Flashscore statistic: {raw}")
            hd, label, ad = parsed[1], parsed[2].strip(), parsed[3]
            key, name, group = LABELS[label]
            values = tuple(int(v.rstrip("%")) for v in (hd, ad))
            if key == "ballPossession":
                if not hd.endswith("%") or not ad.endswith("%") or sum(values) != 100:
                    raise ValueError("Possession must be an observed complementary percentage")
            elif "%" in hd + ad:
                raise ValueError("Count statistic cannot be a percentage")
            identity = (period, key)
            if identity in seen:
                if seen[identity] != values:
                    raise ValueError("Conflicting duplicated Flashscore statistic")
                continue
            seen[identity] = values
            for side, value, display in zip(("home", "away"), values, (hd, ad)):
                row = _row(match, side, key, name, group, value, context, display=display,
                    source=dict(provider="flashscore", source_match_id=sid, period=period, raw_text=raw,
                                capture_method="visible_browser_dom"))
                row["period"] = period
                rows.append(row)
    for (period, key), values in seen.items():
        if period == "ALL" and key != "ballPossession" and ("1ST", key) in seen and ("2ND", key) in seen:
            if values != tuple(a+b for a, b in zip(seen["1ST", key], seen["2ND", key])):
                raise ValueError("Observed half statistics do not reconcile with full-match statistics")
    coverage = [dict(event_id=eid, **context, component=c, status="success", latest_attempt_status="success")
                for c in ("event", "statistics")]
    # Explicit provider gaps; no shots, player identities or heatmaps inferred
    # from aggregate statistics, and no claim of a successful Sofa capture.
    coverage += [dict(event_id=eid, source_key="sofascore:missing-event:" + sid,
                     component=c, status="unavailable", latest_attempt_status="unavailable")
                 for c in ("sofascore_event", "pregame", "shotmap", "heatmap", "lineups", "availability",
                           "incidents", "goal_actions")]
    coverage = [{**dict.fromkeys(TABLES["coverage"]), **row} for row in coverage]
    return {"matches": [match], "statistics": rows, "coverage": coverage}


def review_metadata():
    raw = REVIEW.read_bytes()
    if hashlib.sha256(raw).hexdigest() != REVIEW_SHA256:
        raise ValueError("Reviewed supplemental-source file hash mismatch")
    review = json.loads(raw)
    observations = review["observations"]
    if (len(observations) != 3 or {o["source_match_id"] for o in observations} != set(IDENTITIES)):
        raise ValueError("The review must contain exactly the three authorized fixtures")
    for observation in observations:
        observation_tables(observation)
    return dict(review_id="greek-2020-21-three-missing-2026-11-11", review_sha256=REVIEW_SHA256,
        source_capture_complete=False, missing_sofascore_fixtures=3, verified_sofascore_event_ids=[],
        supplemental_event_id_namespace="internal_flashscore_base62_10e15", review=review)


def reviewed_gaps(events, spec, standings):
    """Permit this exact incomplete provider inventory; every other defect fails normally."""
    if (spec.get("competition_id") != 185 or spec.get("season") != "2020/2021"
            or spec.get("snapshot") is not False or spec.get("groups") != 1
            or spec.get("meetings_per_pair") != 2 or spec.get("expected_teams") not in (None, 14)
            or spec.get("round_limit") not in (None, 26) or len(events) != 179
            or standings is None or len(standings) != 14 or set(standings.values()) != {26}):
        return []
    if any(not is_completed_event(e) or e.get("season", {}).get("id") != 32553
           or e.get("tournament", {}).get("uniqueTournament", {}).get("id") != 185 for e in events):
        return []
    occupied = {(e["homeTeam"]["id"], e["awayTeam"]["id"], e["roundInfo"]["round"]) for e in events}
    gaps = [(18, h, a) for h, a, *_ in IDENTITIES.values()]
    if any((h, a, n) in occupied for n, h, a in gaps):
        return []
    review_metadata()  # Missing, tampered or invalid source evidence never opens the gate.
    return gaps


def campaign_supplements(config, stem, items):
    cohorts = (config.get("source_metadata") or {}).get("seasons", [])
    matches = [c for c in cohorts if c.get("stem") == stem and c.get("reviewed_source_gaps")]
    if not matches:
        if any(c.get("stem") == stem and (c.get("missing_sofascore_fixtures")
                   or c.get("expected_dataset_matches", c.get("selected_matches")) != c.get("selected_matches"))
               for c in cohorts):
            raise ValueError("Missing supplemental evidence for the approved provider gaps")
        return None
    if len(matches) != 1 or stem != STEM or matches[0]["reviewed_source_gaps"] != review_metadata():
        raise ValueError("Campaign supplemental provenance differs from its pinned review")
    cohort = matches[0]
    if (cohort.get("selected_matches") != 179 or cohort.get("expected_dataset_matches") != 182
            or not cohort.get("schedule_verified") or len(items) != 179
            or any((i["competition_id"], i["season_id"]) != (185, 32553) for i in items)):
        raise ValueError("Supplemental publication requires the exact 179-capture Greek cohort")
    events = [i["reference"]["event"] for i in items]
    teams = {e[s]["id"] for e in events for s in ("homeTeam", "awayTeam")}
    spec = dict(competition_id=185, season="2020/2021", snapshot=False, groups=1,
                meetings_per_pair=2, expected_teams=14, round_limit=26)
    from .archive_formats import validate_schedule
    if validate_schedule(events, spec, {t: 26 for t in teams}):
        raise ValueError("Queued Greek fixture identities no longer reconcile with the reviewed gaps")
    return cohort["reviewed_source_gaps"]


def supplemental_tables(metadata):
    if metadata != review_metadata():
        raise ValueError("Unreviewed supplemental metadata")
    result = {"matches": [], "statistics": [], "coverage": []}
    for observation in metadata["review"]["observations"]:
        for name, rows in observation_tables(observation).items():
            result[name].extend(rows)
    return result


def publication_expected(expected, stem, reconciliation):
    """Expand a native queue identity inventory by exactly the reviewed supplements."""
    metadata = reconciliation.get("reviewed_source_gaps")
    if metadata is None:
        return expected
    cohort = reconciliation.get("archive_cohort", {})
    if (stem != STEM or cohort.get("stem") != STEM or cohort.get("selected_matches") != 179
            or cohort.get("expected_dataset_matches") != 182 or not cohort.get("schedule_verified")
            or cohort.get("reviewed_source_gaps") != metadata):
        raise ValueError("Supplemental publication has inconsistent archive provenance")
    rows = supplemental_tables(metadata)["matches"]
    additions = {r["event_id"]: tuple(r[k] for k in ("event_id", "competition_id", "season_id", "home_id", "away_id"))
                 for r in rows}
    native = {k: v for k, v in expected.items() if k not in additions}
    if (len(native) != 179 or any(v[1:3] != (185, 32553) for v in native.values())
            or (set(expected) & set(additions) and not set(additions) <= set(expected))
            or any(k in expected and expected[k] != v for k, v in additions.items())):
        raise ValueError("Supplemental publication differs from the native queue inventory")
    return {**native, **additions}


def verify_supplemental_tables(saved, metadata):
    """Reject changed, missing or fabricated external rows even on watcher restart."""
    from scraping.sofascore.schema import to_arrow
    intended = supplemental_tables(metadata)
    ids = {r["event_id"] for r in intended["matches"]}
    for name in set(saved) | set(intended):
        actual = [r for r in saved.get(name, []) if r.get("event_id") in ids]
        expected = to_arrow(name, intended.get(name, [])).to_pylist()
        if actual != expected:
            raise ValueError(f"Published supplemental {name} differs from the pinned observations")
