"""Reconcile explicit ordinary stages without rewriting captured fixture data.

Administrative results contribute only to the standings calculation. Unplayed
awards never become capture targets, and played-but-annulled games remain
separately identified. Callers must also validate the complete page chain and
the phase's format-specific pairing contract before approving an archive.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import combinations
import re

from scraping.sofascore.discovery import is_completed_event
from scraping.sofascore.event_round import match_round
from scraping.sofascore.event_scores import match_scores
from scraping.sofascore.event_sides import match_team_ids, reviewed_reversed_sides

FIELDS = ("matches", "wins", "draws", "losses", "scoresFor", "scoresAgainst")
DISPOSITIONS = {"exclude_friendly", "exclude_unplayed", "exclude_unfinished",
                "retain_played", "retain_annulled"}


def fixture_identity(event):
    return dict(id=event["id"], home=event["homeTeam"]["id"], away=event["awayTeam"]["id"],
        round=match_round(event), kickoff=event["startTimestamp"], status=event.get("status"),
        score=[event.get(s, {}).get("current") for s in ("homeScore", "awayScore")],
        isAwarded=event.get("isAwarded"), competition_id=event["tournament"]["uniqueTournament"]["id"],
        season_id=event["season"]["id"], tournament_id=event["tournament"]["id"])


def checked_review(event, reviews):
    found = [r for r in reviews if r["event_id"] == event["id"]]
    if not found:
        return None
    if len(found) != 1 or found[0]["guard"] != fixture_identity(event):
        raise ValueError(f"Reviewed fixture identity changed: {event['id']}")
    review = found[0]
    if review.get("disposition") not in DISPOSITIONS or not review.get("reason"):
        raise ValueError("Missing explicit fixture disposition")
    if review["disposition"].startswith("retain_") and not is_completed_event(event):
        raise ValueError("A retained played fixture must have a completed provider status")
    if "standings_score" in review:
        score = review["standings_score"]
        if not isinstance(score, list) or len(score) != 2 or any(type(v) is not int or v < 0 for v in score):
            raise ValueError("Invalid reviewed administrative score")
    if review.get("standings_winner") not in (None, "home", "away"):
        raise ValueError("Invalid reviewed administrative winner")
    return review


def checked_table_rows(tables, reviews):
    rows = {}
    for table in tables:
        values = deepcopy(table["rows"])
        cid = table["tournament"]["uniqueTournament"]["id"]
        found = [r for r in reviews if r["competition_id"] == cid and r["table_id"] == table["id"]]
        if len(found) > 1:
            raise ValueError("Duplicate reviewed table")
        if found:
            review = found[0]
            guarded = {str(r["team"]["id"]): {k: r[k] for k in FIELDS} for r in values
                       if str(r["team"]["id"]) in review["guard_rows"]}
            if guarded != review["guard_rows"] or set(guarded) != set(review["replacement_rows"]):
                raise ValueError("Reviewed provider standings changed")
            for row in values:
                row.update(review["replacement_rows"].get(str(row["team"]["id"]), {}))
        for row in values:
            team = row["team"]["id"]
            if type(team) is not int or team < 1 or team in rows:
                raise ValueError("Duplicate or invalid ordinary-stage club")
            if any(type(row.get(k)) is not int or row[k] < 0 for k in FIELDS):
                raise ValueError("Invalid standings count")
            if row["matches"] != row["wins"] + row["draws"] + row["losses"]:
                raise ValueError("Standings played total differs from W/D/L")
            rows[team] = row
    return rows


def reconcile_phase(events, tables, *, event_reviews=(), table_reviews=()):
    expected = checked_table_rows(tables, table_reviews)
    actual, selected, nominal, annulled, excluded, ids = defaultdict(Counter), [], [], [], [], set()
    for event in events:
        eid = event["id"]
        if eid in ids:
            raise ValueError("Duplicate fixture ID in ordinary phase")
        ids.add(eid)
        review = checked_review(event, event_reviews)
        if event.get("isAwarded") and review is None:
            raise ValueError(f"Awarded fixture needs an explicit played/unplayed review: {eid}")
        disposition = review["disposition"] if review else None
        if is_completed_event(event) and disposition not in {"exclude_friendly", "exclude_unplayed", "exclude_unfinished"}:
            selected.append(eid)
        if disposition == "retain_annulled":
            annulled.append(eid)
            continue
        if disposition == "exclude_friendly":
            excluded.append(eid)
            continue
        if not is_completed_event(event) and (not review or "standings_score" not in review):
            continue
        home, away = event["homeTeam"]["id"], event["awayTeam"]["id"]
        if home == away or home not in expected or away not in expected:
            raise ValueError(f"Fixture {eid} has a foreign or identical club")
        score = review.get("standings_score") if review else None
        if score is None:
            score = [match_scores(event, side).get("current") for side in ("home", "away")]
        if any(type(n) is not int or n < 0 for n in score):
            raise ValueError(f"Fixture {eid} has no final score")
        nominal.append(eid)
        hg, ag = score
        winner = review.get("standings_winner") if review else None
        for side, team, gf, ga in [("home", home, hg, ag), ("away", away, ag, hg)]:
            won = (winner == side) if winner else gf > ga
            draw = False if winner else gf == ga
            lost = (winner != side) if winner else gf < ga
            actual[team].update(matches=1, wins=int(won), draws=int(draw), losses=int(lost),
                                scoresFor=gf, scoresAgainst=ga)
    differences = [dict(team=team, name=row["team"]["name"], field=field,
                        expected=row[field], actual=actual[team][field])
                   for team, row in expected.items() for field in FIELDS if row[field] != actual[team][field]]
    return dict(selected_ids=selected, standings_ids=nominal, annulled_played_ids=annulled,
                excluded_friendly_ids=excluded, table_differences=differences,
                expected_rows=expected, actual_rows=dict(actual))


def validate_phase_format(events, tables, contract, reconciliation):
    """Check independently specified pair multiplicities and per-stage rounds.

    Nominal fixtures include documented administrative results for schedule
    reconciliation. Their disposition still independently excludes capture.
    A current snapshot is bounded by its captured table, never a full-season
    completion claim. Unbalanced formats require explicit club groups/rules.
    """
    expected = reconciliation["expected_rows"]
    teams = set(expected)
    nominal = [e for e in events if e["id"] in set(reconciliation["standings_ids"])]
    pairs = Counter(tuple(sorted(match_team_ids(e))) for e in nominal)
    directed = Counter(match_team_ids(e) for e in nominal)
    appearances = Counter(t for pair in pairs.elements() for t in pair)
    cid, sid, tid = contract["competition_id"], contract["season_id"], contract["tournament_id"]
    if any((e["tournament"]["uniqueTournament"]["id"], e["season"]["id"], e["tournament"]["id"]) != (cid, sid, tid) for e in events):
        raise ValueError("Foreign fixture in phase contract")
    if len(teams) != contract["teams"] or not nominal:
        raise ValueError("Ordinary-phase membership changed or is empty")
    snapshot = contract["snapshot"]
    if type(snapshot) is not bool:
        raise ValueError("Explicit snapshot decision required")
    groups = contract.get("club_groups", [sorted(teams)])
    if set(t for g in groups for t in g) != teams or sum(map(len, groups)) != len(teams):
        raise ValueError("Group membership does not partition the ordinary clubs")
    group = {t: i for i, members in enumerate(groups) for t in members}
    rule = contract["pair_rule"]
    if rule == "round_robin":
        meetings = contract["meetings"]
        if type(meetings) is not int or not 1 <= meetings <= 3:
            raise ValueError("Invalid round-robin multiplicity")
        allowed = {tuple(sorted(p)) for g in groups for p in combinations(g, 2)}
        if set(pairs) - allowed or any(n > meetings for n in pairs.values()):
            raise ValueError("Foreign or repeated ordinary-stage opponent pair")
        if not snapshot and (set(pairs) != allowed or set(pairs.values()) != {meetings}):
            raise ValueError("Incomplete round-robin opponent matrix")
        if meetings == 2 and any(n > 1 for n in directed.values()):
            raise ValueError("Repeated home allocation in double round robin")
    elif rule == "round_robin_extra":
        # Colombia's one extra derby and Argentina/Peru's ordinary crossovers.
        intra = {tuple(sorted(p)) for g in groups for p in combinations(g, 2)}
        extra = Counter(tuple(sorted(p)) for p in contract["extra_pairs"])
        allowed = Counter({p: 1 for p in intra}) + extra
        if set(pairs) - set(allowed) or any(n > allowed[p] for p, n in pairs.items()):
            raise ValueError("Pair differs from the reviewed derby/crossover schedule")
        if not snapshot and pairs != allowed:
            raise ValueError("Missing ordinary or designated extra fixture")
        extra_counts = Counter(t for p in extra.elements() for t in p)
        if set(extra_counts) != teams or set(extra_counts.values()) != {contract["extra_per_club"]}:
            raise ValueError("Unbalanced designated derby/crossover reference")
        if len(groups) > 1 and any(group[a] == group[b] for a, b in extra):
            raise ValueError("Ordinary crossover lies inside one group")
    elif rule == "a_league":
        if cid != 136 or tid != 144:
            raise ValueError("A-League pair rules cannot authorize another competition")
        if any(n > 3 for n in pairs.values()) or any(n > 2 for n in directed.values()):
            raise ValueError("Excess ordinary A-League opponent meetings")
        if not snapshot and (len(pairs) != len(teams)*(len(teams)-1)//2 or min(pairs.values()) < 2):
            raise ValueError("Incomplete A-League opponent coverage")
        if not snapshot and set(appearances.values()) != {contract["played_per_club"]}:
            raise ValueError("Wrong A-League season length")
    elif rule == "mls":
        if cid != 242 or tid != 18 or len(groups) != 2:
            raise ValueError("MLS conference rules require the ordinary MLS stage")
        if any(n > contract["max_meetings"] for n in pairs.values()):
            raise ValueError("Excess MLS opponent meetings")
        if contract["season_id"] != 26780 and set(appearances.values()) != {34}:
            raise ValueError("MLS completed season must contain 34 games per club")
        # These seasons have exact home/away intra-conference double round robins.
        if sid in {20108, 40071, 70158, 86668}:
            for g in groups:
                if any(directed[a, b] != 1 for a in g for b in g if a != b):
                    raise ValueError("MLS conference home/away matrix is incomplete")
            if any(n != 1 for (a, b), n in pairs.items() if group[a] != group[b]):
                raise ValueError("Repeated MLS cross-conference pairing")
        if sid != 26780 and any(sum(n for (h, a), n in directed.items() if h == t) != 17 for t in teams):
            raise ValueError("MLS home allocation differs from the 17-game schedule")
    else:
        raise ValueError("Unreviewed ordinary phase format")

    occupancy = Counter()
    mode = contract["round_mode"]
    special = {(136,34942,144), (242,12825,18), (242,15813,18)}
    if mode == "provider_week_buckets" and (cid,sid,tid) not in special:
        raise ValueError("Week-bucket exception is not valid for this exact season")
    if mode == "absent" and (cid != 242 or tid != 18 or sid not in {20108,26780,35964,40071,47955,57317,70158,86668}):
        raise ValueError("Absent-round exception is not valid for this exact season")
    if mode not in {"strict", "provider_week_buckets", "absent"}:
        raise ValueError("Unreviewed round policy")
    covered = set()
    for e in nominal:
        n = match_round(e)
        labels = " ".join(str((e.get("roundInfo") or {}).get(k, "")) for k in ("name","slug","prefix"))
        if re.search(r"play.?off|final|round[ -]of[ -](?:16|32|64)", labels, re.I):
            raise ValueError("Knockout fixture inside ordinary stage")
        if (mode == "absent" and n is not None) or (mode != "absent" and (type(n) is not int or n not in contract["rounds"] or n < 1)):
            raise ValueError(f"Unexpected source round: {cid}/{sid}/{tid} event {e['id']} round {n}")
        covered.add(n)
        occupancy.update((n, e[s]["id"]) for s in ("homeTeam","awayTeam"))
    if mode == "strict" and any(n > 1 for n in occupancy.values()):
        raise ValueError("A club plays twice in one ordinary phase round")
    if mode != "absent" and covered != set(contract["rounds"]):
        raise ValueError("Ordinary round coverage differs from the reviewed schedule")

    if contract.get("independent_results"):
        reference = Counter(tuple(r) for r in contract["independent_results"])
        actual = Counter((e["homeTeam"]["id"], e["awayTeam"]["id"],
                          match_scores(e,"home")["current"], match_scores(e,"away")["current"]) for e in nominal)
        if actual != reference:
            raise ValueError("Fixtures differ from the independent official result matrix")
    return dict(matches=len(nominal), clubs=len(teams), pairs=len(pairs),
                pair_multiplicities=dict(Counter(pairs.values())), round_mode=mode)
