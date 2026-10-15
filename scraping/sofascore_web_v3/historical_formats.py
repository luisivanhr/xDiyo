"""Verified stage boundaries and officially curtailed historical seasons.

These rules do not relax discovery for any other league or season. Every
advertised round must still be acknowledged, and played matches must reconcile
with the final standings and the independently recorded season totals.
"""
from collections import Counter
from datetime import datetime, timezone

from scraping.sofascore.discovery import event_status
from scraping.sofascore.store import season_year_pair


CURTAILED_SEASONS = {
    (38, 2019, 2020): {
        "teams": 16, "matches": 232, "meetings": 2, "last_played_round": 29,
        "source": "https://www.belgiancompetition.be/sites/default/files/content/download/files/abc-2020-vm-26_pub.pdf",
    },
    (36, 2019, 2020): {
        "teams": 12, "matches": 179, "meetings": 3, "last_played_round": 30,
        "source": "https://spfl.co.uk/news/ladbrokes-premiership-and-spfl-season-201920-cur",
    },
}
LAST_PLAYED_BEFORE = datetime(2020, 3, 14, tzinfo=timezone.utc).timestamp()


# These old fixture stubs cannot use the ordinary later/same-round rule.
# Blackpool's opener and Montpellier-Lyon reversed venue; Chelsea-Villa moved earlier.
# Original and replacement: id, competition, season, years, round, home, away,
# kickoff, status. A replacement must also retain its verified final score.
# https://www.premierleague.com/ar/news/698090
# https://www.standard.co.uk/sport/football/chelseas-premier-league-fixture-with-aston-villa-brought-forward-8732292.html
# https://mhscfoot.com/articles/2015-2016/equipe-pro/retrospective-2014-2015-octobre-2014/
REVIEWED_REPLACEMENTS = {
    # Reviewed Mexican resumptions and the Atlas-America venue reversal.
    # Raw records remain intact; exact original/replacement identities and
    # scores are required. Official reports and links are in the README.
    6133946: (
        (6133946, 11621, 8176, (2014, 2015), 8, 1932, 1938, 1416092400, "cancelled"),
        (5559934, 11621, 8176, (2014, 2015), 8, 1932, 1938, 1410652800, "finished"), (1, 0)),
    7450393: (
        (7450393, 11621, 13413, (2017, 2018), 11, 1932, 1948, 1506564000, "finished"),
        (7636813, 11621, 13413, (2017, 2018), 11, 1932, 1948, 1510282800, "finished"), (1, 1)),
    11341785: (
        (11341785, 11621, 52052, (2023, 2024), 4, 1949, 1939, 1692396000, "postponed"),
        (11557348, 11621, 52052, (2023, 2024), 4, 1939, 1949, 1692587400, "finished"), (1, 1)),
    11528701: (
        (11528701, 11621, 52052, (2023, 2024), 4, 1949, 1939, 1692586800, "cancelled"),
        (11557348, 11621, 52052, (2023, 2024), 4, 1939, 1949, 1692587400, "finished"), (1, 1)),
    # Two obsolete Clausura 2024 round-15 stubs; the clubs' actual matches
    # were played in rounds 2 and 7, with the exact identities below.
    11911065: (
        (11911065, 11620, 57315, (2023, 2024), 15, 1933, 1950, 1713139200, "cancelled"),
        (11911187, 11620, 57315, (2023, 2024), 2, 1950, 1933, 1705719600, "finished"), (3, 1)),
    11911068: (
        (11911068, 11620, 57315, (2023, 2024), 15, 1949, 36534, 1713132000, "cancelled"),
        (11911139, 11620, 57315, (2023, 2024), 7, 1949, 36534, 1708293600, "finished"), (0, 1)),
    # Exact 2026 postponed stubs with officially announced future replacements.
    # Both replacements must still be unplayed and scoreless. They validate
    # the fixture inventory but never enter the completed-match snapshot.
    16317921: (
        (16317921, 11621, 96191, (2026, 2026), 7, 1939, 36525, 1788656400, "postponed"),
        (17025169, 11621, 96191, (2026, 2026), 7, 1939, 36525, 1793242800, "notstarted"), (None, None)),
    16317922: (
        (16317922, 11621, 96191, (2026, 2026), 7, 1951, 1932, 1788649200, "postponed"),
        (17020784, 11621, 96191, (2026, 2026), 7, 1951, 1932, 1794697200, "notstarted"), (None, None)),
    # Exact stale Montedio Yamagata 2021 records, reconciled against the club's
    # official match archive. These have the wrong round as well as date.
    # Sources and the individual pair mappings are documented in the README.
    9309687: (
        (9309687, 402, 35274, (2021, 2021), 16, 21902, 34034, 1622343600, "cancelled"),
        (9309823, 402, 35274, (2021, 2021), 5, 21902, 34034, 1616907600, "finished"), (0, 1)),
    9309763: (
        (9309763, 402, 35274, (2021, 2021), 11, 40029, 21902, 1619845200, "cancelled"),
        (9309827, 402, 35274, (2021, 2021), 6, 40029, 21902, 1617426000, "finished"), (0, 0)),
    9309829: (
        (9309829, 402, 35274, (2021, 2021), 5, 21901, 21902, 1616907600, "cancelled"),
        (9309720, 402, 35274, (2021, 2021), 19, 21901, 21902, 1624179600, "finished"), (0, 1)),
    9309840: (
        (9309840, 402, 35274, (2021, 2021), 6, 21902, 44523, 1617512400, "cancelled"),
        (9309979, 402, 35274, (2021, 2021), 33, 21902, 44523, 1633842000, "finished"), (2, 1)),
    9309863: (
        (9309863, 402, 35274, (2021, 2021), 9, 8060, 21902, 1618999200, "cancelled"),
        (9309950, 402, 35274, (2021, 2021), 31, 8060, 21902, 1632564000, "finished"), (0, 4)),
    9309864: (
        (9309864, 402, 35274, (2021, 2021), 21, 21902, 3139, 1625302800, "cancelled"),
        (9310055, 402, 35274, (2021, 2021), 37, 21902, 3139, 1635915600, "finished"), (1, 2)),
    9309903: (
        (9309903, 402, 35274, (2021, 2021), 38, 21902, 5128, 1636282800, "cancelled"),
        (9309844, 402, 35274, (2021, 2021), 7, 21902, 5128, 1618117200, "finished"), (0, 2)),
    9309913: (
        (9309913, 402, 35274, (2021, 2021), 27, 5130, 21902, 1630231200, "cancelled"),
        (9309875, 402, 35274, (2021, 2021), 21, 5130, 21902, 1625306400, "finished"), (1, 3)),
    9309941: (
        (9309941, 402, 35274, (2021, 2021), 29, 3139, 21902, 1631354400, "cancelled"),
        (9309884, 402, 35274, (2021, 2021), 2, 3139, 21902, 1615096800, "finished"), (0, 2)),
    9309957: (
        (9309957, 402, 35274, (2021, 2021), 31, 21902, 40029, 1632650400, "cancelled"),
        (9632202, 402, 35274, (2021, 2021), 42, 21902, 40029, 1638676800, "finished"), (5, 1)),
    9309976: (
        (9309976, 402, 35274, (2021, 2021), 33, 21902, 26009, 1633860000, "cancelled"),
        (9309819, 402, 35274, (2021, 2021), 4, 21902, 26009, 1616302800, "finished"), (1, 2)),
    9310005: (
        (9310005, 402, 35274, (2021, 2021), 34, 34034, 21902, 1634464800, "cancelled"),
        (9309894, 402, 35274, (2021, 2021), 38, 34034, 21902, 1636257600, "finished"), (1, 0)),
    9310053: (
        (9310053, 402, 35274, (2021, 2021), 25, 21902, 3149, 1628931600, "postponed"),
        (9309949, 402, 35274, (2021, 2021), 30, 21902, 3149, 1631959200, "finished"), (0, 2)),
    9310077: (
        (9310077, 402, 35274, (2021, 2021), 26, 21902, 8060, 1629622800, "cancelled"),
        (9309797, 402, 35274, (2021, 2021), 14, 21902, 8060, 1621141200, "finished"), (1, 0)),
    9310099: (
        (9310099, 402, 35274, (2021, 2021), 39, 5762, 21902, 1636887600, "cancelled"),
        (9309959, 402, 35274, (2021, 2021), 32, 5762, 21902, 1633150800, "finished"), (2, 0)),
    # Official replay/resumption; both provider records are labelled finished.
    # https://www.jleague.jp/news/article/22666/
    # https://www.jleague.jp/match/j2/2022/083101/summary/
    10094765: (
        (10094765, 402, 40276, (2022, 2022), 8, 21902, 34034, 1648962000, "finished"),
        (10648806, 402, 40276, (2022, 2022), 8, 21902, 34034, 1661940000, "finished"), (0, 2)),
    # https://www.jleague.jp/news/article/26000/
    # https://www.jleague.jp/match/j2/2023/092702/summary/
    11019080: (
        (11019080, 402, 48054, (2023, 2023), 31, 21906, 38138, 1692525600, "finished"),
        (11680527, 402, 48054, (2023, 2023), 31, 21906, 38138, 1695808800, "finished"), (1, 1)),
    # J1 2021 fixtures moved earlier for the ACL schedule; retain the stale stubs.
    # https://www.jleague.jp/news/article/19907/
    # https://www.jleague.jp/match/j1/2021/050803/livetxt/
    # https://www.frontale.co.jp/goto_game/2021/j_league1/13.html
    9309246: (
        (9309246, 196, 35273, (2021, 2021), 20, 5127, 3148, 1624701600, "cancelled"),
        (9486403, 196, 35273, (2021, 2021), 20, 5127, 3148, 1620813600, "finished"), (2, 2)),
    9309261: (
        (9309261, 196, 35273, (2021, 2021), 20, 3138, 3135, 1624784400, "cancelled"),
        (9486404, 196, 35273, (2021, 2021), 20, 3138, 3135, 1620810000, "finished"), (1, 2)),
    9309348: (
        (9309348, 196, 35273, (2021, 2021), 13, 3136, 3141, 1625049000, "postponed"),
        (9486368, 196, 35273, (2021, 2021), 13, 3136, 3141, 1620453600, "finished"), (1, 0)),
    9309349: (
        (9309349, 196, 35273, (2021, 2021), 13, 3138, 5127, 1625047200, "postponed"),
        (9486384, 196, 35273, (2021, 2021), 13, 3138, 5127, 1620464400, "finished"), (0, 2)),
    1379596: (
        (1379596, 17, 2746, (2010, 2011), 1, 67, 49, 1281794400, "postponed"),
        (1380091, 17, 2746, (2010, 2011), 1, 49, 67, 1281794400, "finished"), (0, 4)),
    3964835: (
        (3964835, 17, 6311, (2013, 2014), 3, 38, 40, 1377957600, "postponed"),
        (4166893, 17, 6311, (2013, 2014), 3, 38, 40, 1377110700, "finished"), (2, 1)),
    8060497: (
        (8060497, 34, 8122, (2014, 2015), 10, 1642, 1649, 1413745200, "cancelled"),
        (5509696, 34, 8122, (2014, 2015), 10, 1649, 1642, 1413745200, "finished"), (5, 1)),
    # https://spfl.co.uk/league/premiership/archive/298?page=2
    1380321: (
        (1380321, 36, 2752, (2010, 2011), 33, 2348, 2352, 1302962400, "postponed"),
        (1694223, 36, 2752, (2010, 2011), 33, 2348, 2352, 1302627600, "finished"), (0, 1)),
    # https://www.skysports.com/football/hearts-vs-st-mirren/report/289467
    3971519: (
        (3971519, 36, 6309, (2013, 2014), 25, 2353, 2359, 1391024700, "postponed"),
        (4934258, 36, 6309, (2013, 2014), 25, 2353, 2359, 1391024700, "finished"), (2, 1)),
}


def verified_replacement_id(event, candidates):
    """Recognize a reviewed duplicate only with its exact replacement present."""
    rule = REVIEWED_REPLACEMENTS.get(event.get("id"))
    if rule is None:
        return None
    original, played, score = rule
    original_scores = {10094765: (0, 1), 11019080: (1, 1), 7450393: (1, 1),
        6133946: (None, None), 11341785: (None, None), 11528701: (None, None),
        16317921: (None, None), 16317922: (None, None),
        11911065: (None, None), 11911068: (None, None)}
    if (original[0] in original_scores and tuple((event.get(side) or {}).get("current")
            for side in ("homeScore", "awayScore")) != original_scores[original[0]]):
        return None
    replacement = candidates.get(played[0])
    if replacement is None:
        return None
    for value, expected in ((event, original[-1]), (replacement, played[-1])):
        accepted = {expected, "canceled"} if expected == "cancelled" else {expected}
        if (value.get("status") or {}).get("type") not in accepted:
            return None

    def identity(e):
        season = e.get("season") or {}
        return (e.get("id"), ((e.get("tournament") or {}).get("uniqueTournament") or {}).get("id"),
            season.get("id"), season_year_pair(season.get("year")), (e.get("roundInfo") or {}).get("round"),
            (e.get("homeTeam") or {}).get("id"), (e.get("awayTeam") or {}).get("id"),
            e.get("startTimestamp"), event_status(e))

    if (identity(event) == original and identity(replacement) == played
            and tuple((replacement.get(side) or {}).get("current") for side in ("homeScore", "awayScore")) == score):
        return played[0]
    return None


def verified_unplayed_award(event):
    """Resolve one verified contradictory status without changing provider data.

    Cosenza-Verona was never played and was awarded 0-3. Sofascore retains
    status=postponed alongside isAwarded=true. Other fixtures stay unresolved.
    https://www.figc.it/media/65942/com-uff-047-sez-unite-testi-delle-decisioni-riunione-del-18102018-cu.pdf
    """
    return (event.get("id") == 7930529
        and event.get("tournament", {}).get("uniqueTournament", {}).get("id") == 53
        and event.get("season", {}).get("id") == 18325
        and event.get("season", {}).get("year") in {"18/19", "2018/19", "2018/2019"}
        and event.get("roundInfo", {}).get("round") == 2
        and event.get("homeTeam", {}).get("id") == 2716
        and event.get("awayTeam", {}).get("id") == 2701
        and event.get("startTimestamp") == 1535817600
        and event.get("isAwarded") is True
        and event_status(event) == "postponed"
        and event.get("homeScore", {}).get("current") == 0
        and event.get("awayScore", {}).get("current") == 3)


def regular_round_limit(competition_id, years):
    # Older Scottish responses label post-split games as ordinary Premiership
    # events (isGroup=False). The requested 2015/16-2019/20 scope ends at round
    # 33, before the split. Do not infer this format for unverified seasons.
    # https://spfl.co.uk/admin/filemanager/files/shares/SPFL%20Rules%20and%20Regulations%2018-May-20%20%28MASTER%20COPY%20B%29%20CLEAN.pdf
    if (competition_id == 36 and years and len(years) == 2
            and 2015 <= years[0] <= 2019 and years[1] == years[0] + 1):
        return 33
    return None


def curtailed_rule(competition_id, years):
    return CURTAILED_SEASONS.get((competition_id, *years))


def validate_curtailed_season(events, teams, played_counts, meetings, rule):
    """Return review issues, never fabricate fixtures or their final status."""
    issues = []
    finished = [e for e in events if e.get("status", {}).get("type") == "finished"]
    appearances = Counter(e[side]["id"] for e in finished for side in ("homeTeam", "awayTeam"))
    pairs = Counter(tuple(sorted((e["homeTeam"]["id"], e["awayTeam"]["id"]))) for e in finished)
    directed = Counter((e["homeTeam"]["id"], e["awayTeam"]["id"]) for e in finished)
    round_teams = Counter((e["roundInfo"]["round"], e[side]["id"])
                          for e in finished for side in ("homeTeam", "awayTeam"))
    if len(teams) != rule["teams"] or meetings != rule["meetings"]:
        issues.append("Curtailed-season team count or format differs from the verified season")
    if len(finished) != rule["matches"]:
        issues.append(f"Expected {rule['matches']} played matches in the curtailed season; found {len(finished)}")
    if set(appearances) != teams or any(e[side]["id"] not in teams for e in events for side in ("homeTeam", "awayTeam")):
        issues.append("Curtailed-season fixture teams differ from the final standings")
    if (not isinstance(played_counts, dict) or set(played_counts) != {str(t) for t in teams}
            or any(type(n) is not int or n < 0 for n in played_counts.values())):
        issues.append("Final standings must include a valid played-match count for every team")
    elif any(appearances[t] != played_counts[str(t)] for t in teams):
        issues.append("Played fixtures do not reconcile with each team's final standings count")
    if (any(n > meetings for n in pairs.values())
            or any(n > (meetings + 1) // 2 for n in directed.values())
            or any(n > 1 for n in round_teams.values())):
        issues.append("Duplicate or conflicting played fixture pairings")
    if any(e["roundInfo"]["round"] > rule["last_played_round"]
           or e["startTimestamp"] >= LAST_PLAYED_BEFORE for e in finished):
        issues.append("Played fixture falls after the verified season curtailment")
    if any(e.get("status", {}).get("type") != "finished"
           and event_status(e) not in {"notstarted", "postponed", "cancelled", "abandoned", "awarded"}
           for e in events):
        issues.append("Unexpected active or unknown status in a curtailed historical season")
    return issues
