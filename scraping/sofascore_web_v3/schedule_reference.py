"""Offline comparison with an independent, exact ordinary-fixture reference.

Not wired into archive approval: real event identities and round contracts must
be reviewed before any season can use this path. Original events are read only.
"""
from collections import Counter
import re

from scraping.sofascore.discovery import is_completed_event
from scraping.sofascore.event_round import match_round
from scraping.sofascore.store import season_year_pair
from .archive_formats import excluded_stage
from .discovery import positive

NAMED_KNOCKOUT = re.compile(r"\bround[ -]of[ -](?:16|32|64)\b", re.I)


def _natural(value):
    if type(value) is not int or value < 0:
        raise ValueError("Expected a nonnegative integer score or count")
    return value


def validate_fixture_reference(events, reference):
    """Check exact pairs/results and stage-round occupancy, without global relaxations."""
    if reference.get("scope") != "full_ordinary_schedule":
        raise ValueError("An explicit full ordinary schedule reference is required")
    if (not isinstance(reference.get("source_sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", reference["source_sha256"])
            or not str(reference.get("source_url", "")).startswith("https://")):
        raise ValueError("Missing independent reference provenance")
    cid, sid = positive(reference["competition_id"]), positive(reference["season_id"])
    years = season_year_pair(reference["season_year"])
    if years is None:
        raise ValueError("Invalid reference season year")
    rounds = reference.get("rounds_by_tournament")
    if not isinstance(rounds, dict) or not rounds:
        raise ValueError("Missing explicit stage-round contract")
    allowed_rounds = {}
    for tournament, values in rounds.items():
        if not isinstance(tournament, str) or not re.fullmatch(r"[1-9][0-9]*", tournament):
            raise ValueError("Invalid reference tournament identity")
        tid = positive(int(tournament))
        if (not isinstance(values, list) or not values
                or any(type(n) is not int or n < 1 for n in values)
                or len(values) != len(set(values)) or min(values) != 1):
            raise ValueError("Invalid explicit stage-round contract")
        allowed_rounds[tid] = set(values)
    expected = Counter()
    for row in reference.get("directed_opponents", []):
        tid = positive(row["tournament_id"])
        home, away, count = positive(row["home"]), positive(row["away"]), positive(row["count"])
        if tid not in allowed_rounds or home == away or (tid, home, away) in expected:
            raise ValueError("Invalid or duplicate expected opponent pair")
        expected[tid, home, away] = count
    if not expected:
        raise ValueError("Empty expected opponent matrix")
    expected_clubs, expected_stage_clubs = Counter(), Counter()
    for (tid, home, away), count in expected.items():
        expected_clubs[home] += count
        expected_clubs[away] += count
        expected_stage_clubs[tid, home] += count
        expected_stage_clubs[tid, away] += count
    played = {positive(int(k)): _natural(v) for k,v in reference.get("played_per_club", {}).items()}
    if dict(expected_clubs) != played:
        raise ValueError("Reference matrix and club played totals disagree")
    expected_results = Counter()
    for row in reference.get("results", []):
        tid = positive(row["tournament_id"])
        home, away = positive(row["home"]), positive(row["away"])
        score = row.get("score")
        if not isinstance(score, list) or len(score) != 2:
            raise ValueError("Missing reference score")
        hg, ag = map(_natural, score)
        expected_results[tid, home, away, hg, ag] += 1
    result_pairs = Counter()
    for (tid, home, away, _, _), count in expected_results.items():
        result_pairs[tid, home, away] += count
    if result_pairs != expected:
        raise ValueError("Reference result list does not match its opponent matrix")

    if not isinstance(events, list):
        raise ValueError("Expected an explicit captured event list")
    ids, directed, appearances, results = set(), Counter(), Counter(), Counter()
    occupancy, stage_appearances, covered, issues = Counter(), Counter(), set(), []
    for event in events:
        try:
            eid = positive(event["id"])
            if eid in ids:
                raise ValueError("Duplicate fixture ID")
            ids.add(eid)
            tournament, season = event["tournament"], event["season"]
            tid = positive(tournament["id"])
            if (tournament.get("uniqueTournament", {}).get("id") != cid
                    or season.get("id") != sid or season_year_pair(season.get("year")) != years):
                raise ValueError("Foreign competition or season")
            labels = " ".join(str(event.get("roundInfo", {}).get(k, "")) for k in ("name", "slug", "prefix"))
            if excluded_stage(event) or NAMED_KNOCKOUT.search(labels):
                raise ValueError("Knockout or later-stage fixture in ordinary selection")
            number = match_round(event)
            if tid not in allowed_rounds or type(number) is not int or number not in allowed_rounds[tid]:
                raise ValueError("Unknown ordinary stage or round")
            if not is_completed_event(event) or event.get("isAwarded") is True:
                raise ValueError("Unplayed or unresolved match status")
            home, away = positive(event["homeTeam"]["id"]), positive(event["awayTeam"]["id"])
            if home == away or home not in played or away not in played:
                raise ValueError("Foreign or identical fixture clubs")
            hg, ag = _natural(event["homeScore"]["current"]), _natural(event["awayScore"]["current"])
            directed[tid, home, away] += 1
            appearances.update((home, away))
            stage_appearances.update(((tid, home), (tid, away)))
            results[tid, home, away, hg, ag] += 1
            occupancy.update(((tid, number, home), (tid, number, away)))
            covered.add((tid, number))
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            issues.append(f"Invalid fixture: {error}")
    if directed != expected:
        issues.append("Fixture pairings differ from the independent opponent matrix")
    if dict(appearances) != played:
        issues.append("Fixture club totals differ from the independent final table")
    if stage_appearances != expected_stage_clubs:
        issues.append("Ordinary-stage club totals differ from the reference")
    if results != expected_results:
        issues.append("Fixture results differ from the independent result reference")
    if any(count > 1 for count in occupancy.values()):
        issues.append("A club appears twice in the same ordinary stage and round")
    if covered != {(tid, number) for tid, values in allowed_rounds.items() for number in values}:
        issues.append("Ordinary stage-round coverage is incomplete")
    return issues
