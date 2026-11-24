"""Verified stage boundaries and officially curtailed historical seasons.

These rules do not relax discovery for any other league or season. Every
advertised round must still be acknowledged, and played matches must reconcile
with the final standings and the independently recorded season totals.
"""
from collections import Counter
from datetime import datetime, timezone

from scraping.sofascore.discovery import event_status, verified_finished_event
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


EGYPT_2017_HOME_ALLOCATION_SOURCES = (
    "https://img.youm7.com/ArticleImgs/2017/8/30/94461-9.jpg",
    "https://media.dotmsr.com/ArticleImgs/2018/4/15/111126-جدول-الدوري.jpg",
)


def reviewed_home_allocation(events, spec, standings):
    """Recognize the two Raja home listings on the EFA's 2017/18 sheets.

    Round 17 (fixture 145) and round 34 (fixture 303) both list Raja first.
    Preserve the original provider sides, scores and team-attributed stats.
    Only the expected directed counts change; all full-season pairing, round,
    team and status checks still apply to the entire inventory.
    """
    if (spec.get("competition_id") != 808 or season_year_pair(spec.get("season")) != (2017, 2018)
            or spec.get("snapshot") is not False or spec.get("expected_teams") != 18
            or spec.get("round_limit") != 34 or spec.get("meetings_per_pair") != 2
            or spec.get("groups") != 1 or len(events) != 306):
        return {}
    appearances = Counter(e[s]["id"] for e in events for s in ("homeTeam", "awayTeam"))
    if (len(appearances) != 18 or set(appearances.values()) != {34}
            or (standings is not None and standings != dict(appearances))):
        return {}
    expected = {7675984: (17, 1515069000, 0, 0), 7779213: (34, 1524836700, 2, 2)}
    pair = [e for e in events if {(e.get(s) or {}).get("id") for s in ("homeTeam", "awayTeam")}
            == {134186, 35187}]
    if len(pair) != 2 or {e.get("id") for e in pair} != set(expected):
        return {}
    for event in pair:
        tournament, season, status = (event.get(k) or {} for k in ("tournament", "season", "status"))
        if ((tournament.get("uniqueTournament") or {}).get("id") != 808
                or tournament.get("id") != 38612 or tournament.get("isGroup") is not False
                or season.get("id") != 14262 or season_year_pair(season.get("year")) != (2017, 2018)
                or (event.get("homeTeam") or {}).get("id") != 134186
                or (event.get("awayTeam") or {}).get("id") != 35187
                or status.get("type") != "finished" or status.get("code") != 100
                or event.get("isAwarded", False) is not False
                or ((event.get("roundInfo") or {}).get("round"), event.get("startTimestamp"),
                    (event.get("homeScore") or {}).get("current"), (event.get("awayScore") or {}).get("current"))
                    != expected[event["id"]]):
            return {}
    return {(134186, 35187): 2, (35187, 134186): 0}


TURKEY_2022_WITHDRAWAL_SOURCE = "https://www.tff.org/default.aspx?ftxtID=39572&pageID=285"
TURKEY_2022_FIXTURE_SOURCE = "https://www.tff.org/default.aspx?pageID=1628"
# TFF's exact round/home/away list of unplayed awards after the earthquake.
# These are schedule evidence, not synthetic provider events or capture rows.
TURKEY_2022_AWARDS = (
    (24, 3101, 3091), (24, 5138, 3061),
    (25, 55625, 5138), (25, 3091, 3085),
    (26, 5138, 3051), (26, 3072, 3091),
    (27, 3091, 3076), (27, 5138, 6063),
    (28, 6362, 3091), (28, 3101, 5138),
    (29, 5138, 3085), (29, 3091, 4954),
    (30, 3050, 3091),
    (31, 3091, 3103), (31, 5138, 6362),
    (32, 3072, 5138), (32, 3086, 3091),
    (33, 5138, 4954), (33, 3091, 3066),
    (34, 3050, 5138), (34, 5765, 3091),
    (35, 3091, 3052), (35, 5138, 5765),
    (36, 3086, 5138), (36, 3091, 55625),
    (37, 3056, 3091), (37, 5138, 3066),
    (38, 3091, 3061), (38, 3052, 5138),
)
TURKEY_2022_TEAMS = frozenset((3050, 3051, 3052, 3056, 3061, 3066, 3072,
    3076, 3085, 3086, 3091, 3101, 3103, 4954, 5138, 5765, 6063, 6362, 55625))


def reviewed_withdrawal_awards(events, spec, standings):
    """Recognize only the exact 2022/23 Turkish withdrawal schedule.

    TFF retained previous results and awarded all remaining Hatayspor/Gaziantep
    games from 6 February. Sofascore omits those 29 unplayed games. Require the
    exact missing directed pairings, all 313 played games, all 19 clubs and
    final 36-game standings before using official exclusions in validation.
    """
    if (spec.get("competition_id") != 52 or season_year_pair(spec.get("season")) != (2022, 2023)
            or spec.get("snapshot") or spec.get("groups") != 1
            or spec.get("meetings_per_pair") != 2 or spec.get("round_limit") not in (None, 38)
            or len(events) != 313 or standings != {t: 36 for t in TURKEY_2022_TEAMS}):
        return ()
    cutoff = datetime(2023, 2, 6, tzinfo=timezone.utc).timestamp()
    start = datetime(2022, 8, 1, tzinfo=timezone.utc).timestamp()
    end = datetime(2023, 6, 9, tzinfo=timezone.utc).timestamp()
    for event in events:
        season, tournament, status = event.get("season") or {}, event.get("tournament") or {}, event.get("status") or {}
        home, away = ((event.get(s) or {}).get("id") for s in ("homeTeam", "awayTeam"))
        stamp, number = event.get("startTimestamp"), (event.get("roundInfo") or {}).get("round")
        if (season.get("id") != 42632 or season_year_pair(season.get("year")) != (2022, 2023)
                or (tournament.get("uniqueTournament") or {}).get("id") != 52
                or tournament.get("isGroup") is not False
                or home not in TURKEY_2022_TEAMS or away not in TURKEY_2022_TEAMS or home == away
                or status.get("type") != "finished" or status.get("code") != 100
                or event.get("isAwarded") is True or type(stamp) is not int or not start <= stamp < end
                or type(number) is not int or not 1 <= number <= 38):
            return ()
        if {home, away} & {3091, 5138} and (stamp >= cutoff or number > 23):
            return ()
    directed = Counter((e["homeTeam"]["id"], e["awayTeam"]["id"]) for e in events)
    missing = {(a, b) for a in TURKEY_2022_TEAMS for b in TURKEY_2022_TEAMS if a != b and not directed[a, b]}
    if missing != {(a, b) for _, a, b in TURKEY_2022_AWARDS} or any(n != 1 for n in directed.values()):
        return ()
    return TURKEY_2022_AWARDS


# These old fixture stubs cannot use the ordinary later/same-round rule.
# Blackpool's opener and Montpellier-Lyon reversed venue; Chelsea-Villa moved earlier.
# Original and replacement: id, competition, season, years, round, home, away,
# kickoff, status. A replacement must also retain its verified final score.
# https://www.premierleague.com/ar/news/698090
# https://www.standard.co.uk/sport/football/chelseas-premier-league-fixture-with-aston-villa-brought-forward-8732292.html
# https://mhscfoot.com/articles/2015-2016/equipe-pro/retrospective-2014-2015-octobre-2014/
REVIEWED_REPLACEMENTS = {
    # Coverage wave 3: original Chrome payloads and independent sources are
    # retained in coverage_europe3_reviewed_fixtures.json.
    9348229: (
        (9348229, 218, 29387, (2020, 2021), 15, 3309, 200123, 1613908800, 'cancelled'),
        (8853426, 218, 29387, (2020, 2021), 15, 200123, 3309, 1613908800, 'finished'), (2, 2)),
    8853492: (
        (8853492, 218, 29387, (2020, 2021), 7, 25480, 43876, 1603537200, 'postponed'),
        (8853543, 218, 29387, (2020, 2021), 7, 43876, 25480, 1603548000, 'finished'), (0, 2)),
    7441786: (
        (7441786, 131, 13401, (2017, 2018), 34, 46160, 2962, 1523037600, 'finished'),
        (7775419, 131, 13401, (2017, 2018), 34, 46160, 2962, 1523901600, 'finished'), (1, 0)),
    7528258: (
        (7528258, 131, 13401, (2017, 2018), 5, 2978, 46161, 1505498400, 'finished'),
        (7620751, 131, 13401, (2017, 2018), 5, 2978, 46161, 1508781600, 'finished'), (2, 1)),
    8246895: (
        (8246895, 131, 23874, (2019, 2020), 23, 46161, 2990, 1580151600, 'finished'),
        (8593372, 131, 23874, (2019, 2020), 23, 46161, 2990, 1583175600, 'finished'), (3, 1)),
    9567709: (
        (9567709, 131, 36893, (2021, 2022), 13, 2976, 2954, 1635530400, 'abandoned'),
        (9985017, 131, 36893, (2021, 2022), 13, 2976, 2954, 1638187200, 'finished'), (1, 1)),
    12196211: (
        (12196211, 218, 52774, (2023, 2024), 17, 284427, 291318, 1711197000, 'finished'),
        (11408637, 218, 52774, (2023, 2024), 17, 284427, 291318, 1711200600, 'finished'), (1, 2)),
    16412900: (
        (16412900, 218, 97214, (2026, 2027), 16, 316055, 3313, 1785769200, 'postponed'),
        (16693841, 218, 97214, (2026, 2027), 16, 316055, 3313, 1797084000, 'notstarted'), (None, None)),
    16412938: (
        (16412938, 218, 97214, (2026, 2027), 6, 294048, 266889, 1789215600, 'postponed'),
        (17524909, 218, 97214, (2026, 2027), 6, 294048, 266889, 1795604400, 'notstarted'), (None, None)),
    16412944: (
        (16412944, 218, 97214, (2026, 2027), 21, 388209, 258536, 1789257600, 'postponed'),
        (17085331, 218, 97214, (2026, 2027), 21, 388209, 258536, 1806757200, 'notstarted'), (None, None)),
    16412985: (
        (16412985, 218, 97214, (2026, 2027), 26, 138564, 3313, 1793376000, 'postponed'),
        (17400310, 218, 97214, (2026, 2027), 26, 138564, 3313, 1809781200, 'notstarted'), (None, None)),
    16413004: (
        (16413004, 218, 97214, (2026, 2027), 13, 388209, 3313, 1795356000, 'notstarted'),
        (16413128, 218, 97214, (2026, 2027), 13, 3313, 388209, 1795190400, 'notstarted'), (None, None)),
    16317481: (
        (16317481, 131, 96187, (2026, 2027), 14, 2958, 2949, 1794136500, 'postponed'),
        (17540733, 131, 96187, (2026, 2027), 14, 2958, 2949, 1795460400, 'notstarted'), (None, None)),
    16472617: (
        (16472617, 239, 97470, (2026, 2027), 11, 3000, 74467, 1794756600, 'postponed'),
        (17525221, 239, 97470, (2026, 2027), 11, 3000, 74467, 1796242500, 'notstarted'), (None, None)),
    16472618: (
        (16472618, 239, 97470, (2026, 2027), 11, 74465, 483088, 1794756600, 'postponed'),
        (17525218, 239, 97470, (2026, 2027), 11, 74465, 483088, 1796234400, 'notstarted'), (None, None)),
    16472623: (
        (16472623, 239, 97470, (2026, 2027), 11, 5136, 4501, 1794756600, 'postponed'),
        (17510693, 239, 97470, (2026, 2027), 11, 5136, 4501, 1795356000, 'notstarted'), (None, None)),
    16472627: (
        (16472627, 239, 97470, (2026, 2027), 11, 74469, 25787, 1794756600, 'postponed'),
        (17532374, 239, 97470, (2026, 2027), 11, 74469, 25787, 1798990200, 'notstarted'), (None, None)),
    16362250: (
        (16362250, 206, 96614, (2026, 2027), 12, 2374, 2364, 1792850400, 'postponed'),
        (17517605, 206, 96614, (2026, 2027), 12, 2374, 2364, 1796154300, 'notstarted'), (None, None)),
    16362265: (
        (16362265, 206, 96614, (2026, 2027), 15, 2350, 2366, 1794668400, 'postponed'),
        (17502226, 206, 96614, (2026, 2027), 15, 2350, 2366, 1796154300, 'notstarted'), (None, None)),
    # Coverage wave 2: exact captured duplicates and official schedules/results
    # are retained in coverage_europe2_reviewed_fixtures.json and the README.
    10539414: (
        (10539414, 247, 42282, (2022, 2023), 22, 3277, 3275, 1677857400, "cancelled"),
        (11074941, 247, 42282, (2022, 2023), 22, 3277, 252080, 1677857400, "finished"), (0, 6)),
    16295834: (
        (16295834, 247, 95923, (2026, 2027), 10, 3273, 43840, 1789925400, "postponed"),
        (17285568, 247, 95923, (2026, 2027), 10, 3273, 43840, 1796315400, "postponed"), (None, None)),
    16295875: (
        (16295875, 247, 95923, (2026, 2027), 16, 3286, 3275, 1795966200, "postponed"),
        (17485700, 247, 95923, (2026, 2027), 16, 3286, 3275, 1796142600, "notstarted"), (None, None)),
    16559945: (
        (16559945, 185, 98659, (2026, 2027), 7, 5062, 3250, 1792254600, "postponed"),
        (17486940, 185, 98659, (2026, 2027), 7, 5062, 3250, 1797440400, "notstarted"), (None, None)),
    11351404: (
        (11351404, 202, 52176, (2023, 2024), 3, 3108, 3117, 1691326800, "postponed"),
        (11351555, 202, 52176, (2023, 2024), 3, 3117, 3108, 1693413000, "finished"), (0, 2)),
    9723710: (
        (9723710, 152, 37234, (2021, 2022), 6, 7731, 116223, 1629732600, "cancelled"),
        (9625801, 152, 37234, (2021, 2022), 6, 7731, 312033, 1629732600, "finished"), (1, 0)),
    16326495: (
        (16326495, 210, 96249, (2026, 2027), 14, 25737, 282203, 1795626000, "notstarted"),
        (17479954, 210, 96249, (2026, 2027), 14, 25737, 282203, 1795618800, "notstarted"), (None, None)),
    16357129: (
        (16357129, 215, 96589, (2026, 2027), 14, 2463, 2446, 1795361400, "cancelled"),
        (17526778, 215, 96589, (2026, 2027), 14, 2446, 2463, 1795361400, "notstarted"), (None, None)),
    # Coverage wave 1: official reports/schedules and captured identities are
    # documented in ARCHIVE_EXPANSION_README.md and the regression fixture.
    # Resumed Slovenian matches retain the original consolidated final result.
    # Future/postponed replacements remain outside the completed-match snapshot.
    7649803: (
        (7649803, 212, 13457, (2017, 2018), 16, 2412, 2409, 1511186400, "cancelled"),
        (7457453, 212, 13457, (2017, 2018), 16, 2412, 2409, 1511032800, "finished"), (2, 0)),
    9764637: (
        (9764637, 212, 37326, (2021, 2022), 5, 213357, 37737, 1630681200, "cancelled"),
        (9614355, 212, 37326, (2021, 2022), 5, 213357, 37737, 1629039600, "finished"), (4, 2)),
    7885037: (
        (7885037, 955, 17906, (2018, 2019), 4, 168084, 34469, 1538234700, "cancelled"),
        (7885113, 955, 17906, (2018, 2019), 4, 34469, 168084, 1538242500, "finished"), (2, 0)),
    7885066: (
        (7885066, 955, 17906, (2018, 2019), 10, 168086, 168094, 1542901800, "postponed"),
        (7885186, 955, 17906, (2018, 2019), 10, 168094, 168086, 1542898200, "finished"), (1, 1)),
    9688833: (
        (9688833, 955, 37597, (2021, 2022), 4, 168094, 21895, 1632842700, "postponed"),
        (10000039, 955, 37597, (2021, 2022), 4, 21895, 168094, 1639147200, "finished"), (0, 0)),
    10913432: (
        (10913432, 955, 44908, (2022, 2023), 14, 23400, 34318, 1681498800, "cancelled"),
        (10665958, 955, 44908, (2022, 2023), 14, 23400, 34318, 1674408600, "finished"), (1, 0)),
    11347303: (
        (11347303, 170, 52147, (2023, 2024), 4, 36246, 2032, 1691859600, "postponed"),
        (11459684, 170, 52147, (2023, 2024), 4, 2032, 36246, 1706025600, "finished"), (0, 3)),
    16275179: (
        (16275179, 170, 95727, (2026, 2027), 2, 2039, 2032, 1786201200, "postponed"),
        (17184945, 170, 95727, (2026, 2027), 2, 2039, 2032, 1796317200, "notstarted"), (None, None)),
    15260931: (
        (15260931, 20, 87809, (2026, 2026), 26, 957, 1165, 1794063600, "postponed"),
        (17505158, 20, 87809, (2026, 2026), 26, 957, 1165, 1794247200, "notstarted"), (None, None)),
    16653115: (
        (16653115, 955, 99275, (2026, 2027), 10, 168090, 269199, 1792423200, "postponed"),
        (17356156, 955, 99275, (2026, 2027), 10, 168090, 269199, 1796650800, "notstarted"), (None, None)),
    16653146: (
        (16653146, 955, 99275, (2026, 2027), 13, 34469, 56021, 1793898000, "postponed"),
        (17413886, 955, 99275, (2026, 2027), 13, 34469, 56021, 1794072600, "postponed"), (None, None)),
    16653103: (
        (16653103, 955, 99275, (2026, 2027), 9, 34315, 34313, 1792173600, "postponed"),
        (17329607, 955, 99275, (2026, 2027), 9, 34315, 34313, 1797701400, "notstarted"), (None, None)),
    16653113: (
        (16653113, 955, 99275, (2026, 2027), 9, 168088, 21895, 1792162800, "postponed"),
        (17328039, 955, 99275, (2026, 2027), 9, 168088, 21895, 1797691500, "notstarted"), (None, None)),
    16653114: (
        (16653114, 955, 99275, (2026, 2027), 9, 34469, 23400, 1792087200, "postponed"),
        (17328047, 955, 99275, (2026, 2027), 9, 34469, 23400, 1804359600, "notstarted"), (None, None)),
    16653119: (
        (16653119, 955, 99275, (2026, 2027), 10, 34318, 34315, 1792519200, "postponed"),
        (17291748, 955, 99275, (2026, 2027), 10, 34318, 34315, 1792519200, "finished"), (1, 4)),
    16378721: (
        (16378721, 212, 96864, (2026, 2027), 6, 2413, 37737, 1787422500, "postponed"),
        (17391680, 212, 96864, (2026, 2027), 6, 2413, 37737, 1796229000, "notstarted"), (None, None)),
    16378725: (
        (16378725, 212, 96864, (2026, 2027), 16, 2417, 2424, 1787940900, "postponed"),
        (17488629, 212, 96864, (2026, 2027), 16, 2417, 2424, 1795451400, "notstarted"), (None, None)),
    16890867: (
        (16890867, 212, 96864, (2026, 2027), 16, 2417, 2424, 1795276800, "postponed"),
        (17488629, 212, 96864, (2026, 2027), 16, 2417, 2424, 1795451400, "notstarted"), (None, None)),
    # Chapecoense-Vasco, Brazil 2026 round 21, was rescheduled to 14 November.
    # Keep the exact scoreless replacement in the schedule, outside the played
    # snapshot. CBF's alteration notice and both raw records are in the README.
    15235440: (
        (15235440, 325, 87678, (2026, 2026), 21, 21845, 1974, 1785434400, "postponed"),
        (17449066, 325, 87678, (2026, 2026), 21, 21845, 1974, 1794699000, "notstarted"), (None, None)),
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


# Exact Egyptian reschedules/obsolete home allocations. These retain both raw
# records; only an identity- and score-matched played replacement supersedes
# its unfinished placeholder. Sources are in the Egypt fixture review receipt.
EGYPT4_REPLACEMENTS = {
    7711950: (
        (7711950, 808, 14262, (2017, 2018), 23, 43564, 43559, 1518283800, "cancelled"),
        (7757187, 808, 14262, (2017, 2018), 23, 43559, 43564, 1522425600, "finished"), (0, 0)),
    7771281: (
        (7771281, 808, 14262, (2017, 2018), 32, 43560, 43564, 1526482800, "cancelled"),
        (7787588, 808, 14262, (2017, 2018), 32, 43560, 43564, 1525271400, "finished"), (1, 1)),
    7925353: (
        (7925353, 808, 18033, (2018, 2019), 16, 43565, 42368, 1543236300, "cancelled"),
        (8070670, 808, 18033, (2018, 2019), 16, 42368, 43565, 1543255200, "finished"), (3, 0)),
    9278104: (
        (9278104, 808, 34930, (2020, 2021), 7, 35187, 43564, 1610973000, "cancelled"),
        (9289854, 808, 34930, (2020, 2021), 7, 35187, 43564, 1614610800, "finished"), (0, 1)),
    10316706: (
        (10316706, 808, 39373, (2021, 2022), 27, 371622, 58319, 1658853000, "cancelled"),
        (10502103, 808, 39373, (2021, 2022), 27, 58319, 371622, 1658853000, "finished"), (0, 0)),
    10824342: (
        (10824342, 808, 46713, (2022, 2023), 7, 209028, 371622, 1671108300, "postponed"),
        (10936272, 808, 46713, (2022, 2023), 7, 371622, 209028, 1671108300, "finished"), (3, 2)),
}
REVIEWED_REPLACEMENTS.update(EGYPT4_REPLACEMENTS)


# Additional code/award guards for the newly reviewed exact records.
COVERAGE3_REPLACEMENT_STATUS_CODES = {
    9348229: (70, 100),
    8853492: (60, 100),
    7441786: (100, 100),
    7528258: (100, 100),
    8246895: (100, 100),
    9567709: (90, 90),
    12196211: (100, 100),
    16412900: (60, 0),
    16412938: (60, 0),
    16412944: (60, 0),
    16412985: (60, 0),
    16413004: (0, 0),
    16317481: (60, 0),
    16472617: (60, 0),
    16472618: (60, 0),
    16472623: (60, 0),
    16472627: (60, 0),
    16362250: (60, 0),
    16362265: (60, 0),
}


def verified_replacement_id(event, candidates):
    """Recognize a reviewed duplicate only with its exact replacement present."""
    rule = REVIEWED_REPLACEMENTS.get(event.get("id"))
    if rule is None:
        return None
    original, played, score = rule
    if event.get("id") in EGYPT4_REPLACEMENTS:
        replacement = candidates.get(played[0])
        expected_code = 60 if original[-1] == "postponed" else 70
        if (replacement is None
                or (event.get("status") or {}).get("code") != expected_code
                or (replacement.get("status") or {}).get("code") != 100
                or any(value.get("isAwarded", False) is not False for value in (event, replacement))
                or any((event.get(side) or {}).get("current") is not None for side in ("homeScore", "awayScore"))):
            return None
    original_scores = {
        9348229: (None, None),
        8853492: (None, None),
        7441786: (2, 0),
        7528258: (0, 2),
        8246895: (3, 3),
        9567709: (0, 0),
        12196211: (1, 2),
        16412900: (None, None),
        16412938: (None, None),
        16412944: (None, None),
        16412985: (None, None),
        16413004: (None, None),
        16317481: (None, None),
        16472617: (None, None),
        16472618: (None, None),
        16472623: (None, None),
        16472627: (None, None),
        16362250: (None, None),
        16362265: (None, None),
        10094765: (0, 1), 11019080: (1, 1), 7450393: (1, 1),
        6133946: (None, None), 11341785: (None, None), 11528701: (None, None),
        16317921: (None, None), 16317922: (None, None), 15235440: (None, None),
        11911065: (None, None), 11911068: (None, None),
        7649803: (None, None),
        9764637: (None, None),
        7885037: (None, None),
        7885066: (None, None),
        9688833: (None, None),
        10913432: (None, None),
        11347303: (None, None),
        16275179: (None, None),
        15260931: (None, None),
        16653115: (None, None),
        16653146: (None, None),
        16653103: (None, None),
        16653113: (None, None),
        16653114: (None, None),
        16653119: (None, None),
        16378721: (None, None),
        16378725: (None, None),
        16890867: (None, None),
        10539414: (None, None), 16295834: (None, None), 16295875: (None, None),
        16559945: (None, None), 11351404: (None, None), 9723710: (None, None),
        16326495: (None, None), 16357129: (None, None)
    }
    if (original[0] in original_scores and tuple((event.get(side) or {}).get("current")
            for side in ("homeScore", "awayScore")) != original_scores[original[0]]):
        return None
    replacement = candidates.get(played[0])
    if replacement is None:
        return None
    if codes := COVERAGE3_REPLACEMENT_STATUS_CODES.get(original[0]):
        if any((value.get("status") or {}).get("code") != code
               or value.get("isAwarded", False) is not False
               for value, code in zip((event, replacement), codes)):
            return None
    for value, expected in ((event, original[-1]), (replacement, played[-1])):
        accepted = {expected, "canceled"} if expected == "cancelled" else {expected}
        if expected == "finished" and verified_finished_event(value):
            continue
        if (expected == "abandoned" and (value.get("status") or {}) ==
                {"type": "canceled", "code": 90, "description": "Abandoned"}):
            continue
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


REVIEWED_UNPLAYED_AWARDS = {
    # Dynamo did not play at Mariupol on 27 August 2017. UAF awarded 3-0;
    # this exact score-bearing postponement is schedule evidence only.
    # https://uaf.ua/article/31997
    7431119: ((7431119, 218, 13336, (2017, 2018), 7, 3319, 3305, 1503842400, "postponed"), (3, 0)),
    # Piast-Gornik was abandoned, then awarded 0-3 (league decision upheld
    # by PZPN on 23 March 2018). Its raw record lacks isAwarded and retains
    # postponed status; keep it outside completed-match captures.
    7433450: ((7433450, 202, 13350, (2017, 2018), 26, 7918, 3110, 1520105400, "postponed"), (0, 3)),
    # RFU disciplinary decisions: 211886, 211964, 211977, 212338.
    # These exact postponed records represent unplayed awarded results.
    8680226: ((8680226, 203, 23682, (2019, 2020), 25, 24118, 7529, 1593604800, "postponed"), (0, 3)),
    8680347: ((8680347, 203, 23682, (2019, 2020), 29, 107213, 2334, 1594920600, "postponed"), (3, 0)),
    8680361: ((8680361, 203, 23682, (2019, 2020), 30, 2322, 2334, 1595433600, "postponed"), (3, 0)),
    8809808: ((8809808, 203, 29200, (2020, 2021), 8, 2326, 2318, 1600531200, "postponed"), (3, 0)),
}


def verified_unplayed_award(event):
    """Recognize exact unplayed/abandoned awards without changing provider data.

    Cosenza-Verona was never played and was awarded 0-3. Sofascore retains
    status=postponed alongside isAwarded=true. Four Russian forfeits have the
    same contradiction; their full identities and RFU decisions are recorded
    in REVIEWED_UNPLAYED_AWARDS. The abandoned Piast-Gornik match has no
    isAwarded flag; only its exact reviewed identity accepts that absence.
    Other fixtures stay unresolved.
    https://www.figc.it/media/65942/com-uff-047-sez-unite-testi-delle-decisioni-riunione-del-18102018-cu.pdf
    """
    if rule := REVIEWED_UNPLAYED_AWARDS.get(event.get("id")):
        expected, scores = rule
        season = event.get("season") or {}
        identity = (event.get("id"), (event.get("tournament") or {}).get("uniqueTournament", {}).get("id"),
            season.get("id"), season_year_pair(season.get("year")), (event.get("roundInfo") or {}).get("round"),
            (event.get("homeTeam") or {}).get("id"), (event.get("awayTeam") or {}).get("id"),
            event.get("startTimestamp"), event_status(event))
        awarded_flag = None if event.get("id") in {7433450, 7431119} else True
        return (identity == expected and event.get("isAwarded") is awarded_flag
            and (event.get("status") or {}).get("type") == "postponed"
            and (event.get("status") or {}).get("code") == 60
            and tuple((event.get(s) or {}).get("current") for s in ("homeScore", "awayScore")) == scores)
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
           or e["startTimestamp"] >= rule.get("last_played_before", LAST_PLAYED_BEFORE) for e in finished):
        issues.append("Played fixture falls after the verified season curtailment")
    if any(e.get("status", {}).get("type") != "finished"
           and event_status(e) not in {"notstarted", "postponed", "cancelled", "abandoned", "awarded"}
           for e in events):
        issues.append("Unexpected active or unknown status in a curtailed historical season")
    return issues
