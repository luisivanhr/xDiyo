"""Explicit format rules for the three requested archive campaigns.

Keep original provider seasons and rounds. Calendar years use YY_YY in filenames
so the existing publisher can retain its filename/identity contract.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import re

from scraping.sofascore.discovery import event_status, is_completed_event
from scraping.sofascore.event_round import match_round
from scraping.sofascore.store import season_year_pair
from .discovery import STAGE

J_HISTORY = "https://www.jleague.jp/corporate/about_competitions/tournament_history/"
J_TRANSITION = "https://www.jleague.jp/news/article/29712/?mode=pc"
J_NEW = "https://www.jleague.jp/corporate/about_competitions/season/"
MX_2020 = "https://www.chivasdecorazon.com.mx/es/estadisticas/tabla_general/Clausura-2020"
EUROPE_LEAGUES = (("Bundesliga", 35), ("Bundesliga_2", 44), ("Championship", 18),
    ("Premier_League", 17), ("La_Liga", 8), ("La_Liga_2", 54), ("Ligue_1", 34),
    ("Ligue_2", 182), ("Serie_A", 23), ("Serie_B", 53), ("Eredivisie", 37),
    ("Premiership", 36), ("Pro_League", 38))
EXCLUDED = re.compile(STAGE.pattern
    + r"|championship[ -]stage|placement|ranking[ -]match|play[ -]?in|liguilla|repechaje"
    + r"|\bmatch[ -]for[ -]\d+(?:st|nd|rd|th)[ -]place\b", re.I)


def _spec(name, cid, season, *, meetings=2, teams=None, rounds=None,
          groups=1, snapshot=False, source=None, start=None, end=None, **extra):
    return dict(name=name, competition_id=cid, season=season, aliases=[season],
                meetings_per_pair=meetings, expected_teams=teams, round_limit=rounds,
                groups=groups, snapshot=snapshot, source=source, start=start, end=end, **extra)


def requested_archives(*, japan_transition=None):
    """Frozen request: Europe 2010/11-2014/15; Japan to 26/27; MX A2010-A2026.

    None leaves the mixed J2/J3 scope explicitly unresolved. Preparation may
    save that question; discovery will not run it until the scope is settled.
    """
    if japan_transition not in {None, "all", "skip"}:
        raise ValueError("Use all or skip for the mixed J2/J3 tournament")
    europe = []
    for year in range(2010, 2015):
        for name, cid in EUROPE_LEAGUES:
            scottish = cid == 36
            europe.append(_spec(name, cid, f"{year}/{year+1}",
                meetings=3 if scottish else 2, rounds=33 if scottish else None,
                start=f"{year}-07-01", end=f"{year+1}-08-01"))
    japan = []
    for year in range(2010, 2026):
        for name, cid in (("J1", 196), ("J2", 402)):
            teams = (20 if year == 2021 or year >= 2024 else 18) if cid == 196 else (
                19 if year == 2010 else 20 if year == 2011 or year >= 2024 else 22)
            rounds = (teams - 1 if teams % 2 == 0 else teams) * 2
            japan.append(_spec(name, cid, str(year), teams=teams, rounds=rounds, source=J_HISTORY,
                               start=f"{year}-01-01", end=f"{year+1}-01-01"))
    japan.append(_spec("J1_Transition", 196, "2026", teams=20, groups=2, rounds=18,
                       source=J_TRANSITION, start="2026-01-01", end="2026-07-01"))
    if japan_transition != "skip":
        japan.append(_spec("J2_J3_Transition", 402, "2026", teams=40, groups=4, rounds=18,
            source=J_TRANSITION, start="2026-01-01", end="2026-07-01",
            scope_pending=japan_transition is None, selection=japan_transition))
    for name, cid in (("J1", 196), ("J2", 402)):
        japan.append(_spec(name, cid, "2026/2027", teams=20, rounds=38, snapshot=True,
                           source=J_NEW, start="2026-07-01", end="2027-08-01"))
    mexico = []
    for year in range(2010, 2027):
        apertura = _spec("Liga_MX_Apertura", 11621, f"{year}/{year+1}", meetings=1,
            teams=19 if year == 2019 else 18, rounds=19 if year == 2019 else 17,
            snapshot=year == 2026, start=f"{year}-07-01", end=f"{year+1}-01-01",
            cycle=f"{year}/{year+1}", tournament="Apertura")
        # The current provider selector uses 2026; older entries use YY/YY.
        # Accept either explicit spelling, but never two matching season IDs.
        if year == 2026:
            apertura["aliases"].append("2026")
        mexico.append(apertura)
        if year < 2026:
            mexico.append(_spec("Liga_MX_Clausura", 11620, f"{year}/{year+1}", meetings=1,
                teams=18, rounds=10 if year == 2019 else 17,
                start=f"{year+1}-01-01", end=f"{year+1}-07-01",
                cycle=f"{year}/{year+1}", tournament="Clausura",
                curtailed=year == 2019, source=MX_2020 if year == 2019 else None))
    return {"europe-2010-2015": europe, "japan-2010-2027": japan, "mexico-2010-2026": mexico}


def is_regular(event, spec):
    tournament, info = event.get("tournament", {}), event.get("roundInfo", {})
    labels = " ".join(str(obj.get(k, "")) for obj in (tournament, info) for k in ("name", "slug", "prefix"))
    if EXCLUDED.search(labels):
        return False
    number = match_round(event)
    if spec["round_limit"] is not None and number > spec["round_limit"]:
        return False
    if type(tournament.get("isGroup")) is not bool:
        raise ValueError("Missing tournament stage flag")
    # Only the explicitly selected Japanese regional competition may admit
    # grouped events. Their complete within-group schedule is verified below.
    return tournament["isGroup"] is False or spec["groups"] > 1


def validate_schedule(events, spec, standings=None):
    """Validate the complete inventory or a standings-reconciled current snapshot."""
    issues = []
    teams = {e[side]["id"] for e in events for side in ("homeTeam", "awayTeam")}
    if standings is not None and teams != set(standings):
        issues.append("Fixture teams differ from the regular-season standings")
    expected_teams = spec["expected_teams"] or (len(standings) if standings else 0)
    if len(teams) != expected_teams or expected_teams < 2:
        issues.append(f"Expected {expected_teams} teams; found {len(teams)}")
    meetings = spec["meetings_per_pair"]
    pairs = Counter(tuple(sorted((e["homeTeam"]["id"], e["awayTeam"]["id"]))) for e in events)
    directed = Counter((e["homeTeam"]["id"], e["awayTeam"]["id"]) for e in events)
    schedule_rounds = {e["id"]: match_round(e) for e in events}
    if (spec["competition_id"] == 196 and spec["season"] == "2016"
            and expected_teams == 18 and spec["round_limit"] == 34
            and meetings == 2 and spec["groups"] == 1):
        # The captured 2016 inventory reuses 1-17 for both ordinary stages.
        # Validate each complete round robin separately, without renumbering
        # provider payloads or exported rounds. Dates are the official windows:
        # https://www.jleague.jp/special/meijiyasudaj1_1st_winner/2016/
        # https://www.vissel-kobe.co.jp/match/?y=2016
        stage = {e["id"]: _j1_2016_stage(e) for e in events}
        if any(v is None for v in stage.values()):
            issues.append("J1 2016 fixture has an unverified stage identity or date")
        for number in (1, 2):
            stage_events = [e for e in events if stage[e["id"]] == number]
            stage_pairs = Counter(tuple(sorted((e["homeTeam"]["id"], e["awayTeam"]["id"])))
                                  for e in stage_events)
            if len(stage_pairs) != 153 or any(n != 1 for n in stage_pairs.values()):
                issues.append(f"J1 2016 stage {number} needs all 153 distinct pairings")
        schedule_rounds = {eid: n + 17 * (stage[eid] - 1) if stage[eid] else None
                           for eid, n in schedule_rounds.items()}
    round_teams = Counter((schedule_rounds[e["id"]], e[side]["id"])
                          for e in events for side in ("homeTeam", "awayTeam"))
    if (any(n > meetings for n in pairs.values()) or any(n > (meetings+1)//2 for n in directed.values())
            or any(n > 1 for n in round_teams.values())):
        issues.append("Duplicate pairings or a team appears twice in one round")
    finished = [e for e in events if is_completed_event(e)]
    if not finished:
        issues.append("No completed regular-season fixtures")
    unknown = [e["id"] for e in events if not is_completed_event(e)
               and event_status(e) not in {"notstarted", "scheduled", "inprogress", "postponed", "abandoned", "cancelled", "awarded"}]
    if unknown:
        issues.append("Unknown unfinished fixture statuses")
    if spec["snapshot"]:
        counts = Counter(e[s]["id"] for e in finished for s in ("homeTeam", "awayTeam"))
        if (standings is None or len(standings) != expected_teams
                or any(type(n) is not int or n < 0 for n in standings.values())
                or any(counts[t] != n for t, n in standings.items())):
            issues.append("Completed fixtures do not reconcile with current standings played counts")
        return issues
    if spec.get("curtailed"):
        # Clausura 2020: 18 clubs with ten played matches each, 90 games.
        counts = Counter(e[s]["id"] for e in finished for s in ("homeTeam", "awayTeam"))
        if len(finished) != 90 or set(counts) != teams or any(n != 10 for n in counts.values()):
            issues.append("Clausura 2020 needs exactly 90 completed games and ten per club")
    else:
        adjacency = defaultdict(set)
        for a, b in pairs:
            adjacency[a].add(b); adjacency[b].add(a)
        components, remaining = [], set(teams)
        while remaining:
            stack, group = [next(iter(remaining))], set()
            while stack:
                t = stack.pop()
                if t in group:
                    continue
                group.add(t); stack.extend(adjacency[t] - group)
            components.append(group); remaining -= group
        if len(components) != spec["groups"] or any(len(g) * spec["groups"] != expected_teams for g in components):
            issues.append("Missing teams or incorrect number/size of regular-season groups")
        if any(pairs[tuple(sorted((a, b)))] != meetings for group in components for a in group for b in group if a != b):
            issues.append("Incomplete regular-season pairings")
        if meetings % 2 == 0 and any(directed[a, b] != meetings//2 for group in components for a in group for b in group if a != b):
            issues.append("Incomplete home/away pairings")
    rounds = spec["round_limit"] or ((expected_teams-1 if expected_teams % 2 == 0 else expected_teams) * meetings)
    if set(schedule_rounds.values()) != set(range(1, rounds+1)):
        issues.append("Regular-season round coverage is incomplete or outside the selected range")
    if any(not is_completed_event(e) and event_status(e) not in {"abandoned", "cancelled", "awarded"} for e in events):
        issues.append("Historical season still contains unresolved unfinished fixtures")
    return issues


def _j1_2016_stage(event):
    """Only classify the verified 2016 provider season's two ordinary stages."""
    number, season = match_round(event), event.get("season", {})
    if (event.get("tournament", {}).get("uniqueTournament", {}).get("id") != 196
            or season.get("id") != 11307 or season.get("year") != "2016"
            or type(number) is not int or not 1 <= number <= 17
            or type(event.get("startTimestamp")) is not int):
        return None
    day = datetime.fromtimestamp(event["startTimestamp"], timezone.utc).date().isoformat()
    if "2016-02-27" <= day <= "2016-06-25":
        return 1
    if "2016-07-02" <= day <= "2016-11-03":
        return 2
    return None


def reviewed_standings(events, spec, standings, payload):
    """Reconcile one captured Mons ID disagreement for three exact seasons.

    Only validation uses the alias. Fixture IDs, captured standings and output
    payloads stay untouched; this is not a global mapping between old/new clubs.
    """
    seasons = {"2011/2012": 3414, "2012/2013": 4676, "2013/2014": 6241}
    sid = seasons.get(spec["season"])
    if spec["competition_id"] != 38 or sid is None or not standings or not payload:
        return standings
    selected = [s for s in payload.get("standings", []) if s.get("type") == "total"
        and s.get("tournament", {}).get("isGroup") is False
        and not EXCLUDED.search(str(s.get("tournament", {}).get("name", "")))]
    if (len(selected) != 1
            or selected[0].get("tournament", {}).get("uniqueTournament", {}).get("id") != 38):
        return standings
    rows = selected[0].get("rows", [])
    mons = [r for r in rows if r.get("team", {}).get("id") == 2906]
    if (len(rows) != 16 or len(mons) != 1 or mons[0]["team"].get("name") != "RAEC Mons"
            or mons[0].get("matches") != 30 or len(standings) != 16
            or set(standings.values()) != {30} or 2906 not in standings or 549294 in standings
            or {r.get("team", {}).get("id"): r.get("matches") for r in rows} != standings):
        return standings
    appearances = Counter(e[s]["id"] for e in events for s in ("homeTeam", "awayTeam"))
    if (len(events) != 240 or appearances.get(549294) != 30
            or set(appearances) != (set(standings) - {2906}) | {549294}
            or set(appearances.values()) != {30}):
        return standings
    for event in events:
        if (event.get("tournament", {}).get("uniqueTournament", {}).get("id") != 38
                or event.get("season", {}).get("id") != sid
                or season_year_pair(event.get("season", {}).get("year")) != season_year_pair(spec["season"])):
            return standings
        if any(event[s]["id"] == 549294 and event[s].get("name") != "RAEC Mons" for s in ("homeTeam", "awayTeam")):
            return standings
    return {549294 if t == 2906 else t: n for t, n in standings.items()}


def check_kickoff(event, spec):
    timestamp = event.get("startTimestamp")
    if type(timestamp) is not int:
        raise ValueError("Missing or invalid fixture kickoff")
    if spec.get("start") and timestamp < datetime.fromisoformat(spec["start"]).replace(tzinfo=timezone.utc).timestamp():
        raise ValueError("Fixture kickoff precedes the selected tournament")
    if spec.get("end") and timestamp >= datetime.fromisoformat(spec["end"]).replace(tzinfo=timezone.utc).timestamp():
        raise ValueError("Fixture kickoff falls outside the selected tournament")
