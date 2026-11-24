"""Exact archive cohorts ended by official decisions, not incomplete scrapes.

All advertised rounds must still be captured by ArchiveDiscovery. These four
reviews reconcile only actually completed games with the official final table;
cancelled future fixture stubs remain in the raw discovery and are never queued.
"""
from datetime import datetime, timezone

from scraping.sofascore.store import season_year_pair
from .historical_formats import validate_curtailed_season


ARCHIVE_CURTAILMENTS = {
    "dutch_2019_end": {
        "competition_id": 131, "season_id": 23874, "years": (2019, 2020),
        "teams": 20, "matches": 290, "meetings": 2, "last_played_round": 29,
        "cutoff": "2020-03-14",
        "played_counts": dict.fromkeys((2970, 2949, 2968, 46160, 2947, 2979,
            2967, 2962, 2990, 2972, 2963, 53067, 2994, 46161, 2976, 2978,
            2954, 46168, 2973, 2974), 29),
        "source": "https://www.knvb.nl/nieuws/betaald-voetbal/competitiezaken/59905/afwikkeling-voetbalseizoen-2019%E2%80%9920-een-feit",
    },
    "portugal_2019_end": {
        "competition_id": 239, "season_id": 24151, "years": (2019, 2020),
        "teams": 18, "matches": 216, "meetings": 2, "last_played_round": 24,
        "cutoff": "2020-03-13",
        "played_counts": dict.fromkeys((3013, 2998, 4501, 38744, 4500, 3005,
            3015, 3022, 3018, 3037, 3025, 21831, 74469, 74465, 3017, 207129,
            35184, 36365), 24),
        "source": "https://www.tribunalarbitraldesporto.pt/files/decisoes/TAD_23-2020-2.pdf",
    },
    "scotland_2019_end": {
        "competition_id": 206, "season_id": 23988, "years": (2019, 2020),
        "teams": 10, "matches": 137, "meetings": 4, "last_played_round": 28,
        "cutoff": "2020-03-14",
        "played_counts": {2349: 28, 2362: 27, 2357: 27, 2364: 27, 2350: 28,
            2366: 26, 2374: 28, 2369: 28, 2368: 28, 2358: 27},
        "source": "https://spfl.co.uk/admin/filemanager/files/shares/SPFL%20Rules%20and%20Regulations%2018-May-20%20%28MASTER%20COPY%20B%29%20CLEAN.pdf",
    },
    "ukraine_2021_end": {
        "competition_id": 218, "season_id": 37040, "years": (2021, 2022),
        "teams": 16, "matches": 143, "meetings": 2, "last_played_round": 18,
        "cutoff": "2022-02-24",
        "played_counts": {3313: 18, 3305: 18, 258537: 18, 312677: 18,
            3309: 18, 25479: 18, 25480: 18, 200776: 18, 200773: 18,
            266889: 18, 229644: 17, 23790: 18, 3310: 18, 200123: 17,
            291318: 18, 3319: 18},
        "source": "https://upl.ua/ua/news/view/7075",
    },
}


def validate_archive_curtailment(events, spec, standings):
    """Return issues for an explicitly selected review, or None for other scopes."""
    key = spec.get("historical_curtailment_review")
    if key is None:
        return None
    rule = ARCHIVE_CURTAILMENTS.get(key)
    if rule is None:
        return ["Unknown historical curtailment review"]
    first, last = rule["years"]
    full_rounds = (rule["teams"] - 1) * rule["meetings"]
    if (spec.get("competition_id") != rule["competition_id"]
            or season_year_pair(spec.get("season")) != rule["years"]
            or spec.get("snapshot") is not False or spec.get("groups") != 1
            or spec.get("meetings_per_pair") != rule["meetings"]
            or spec.get("expected_teams") not in (None, rule["teams"])
            or spec.get("round_limit") not in (None, full_rounds)
            or (spec.get("start"), spec.get("end")) != (f"{first}-07-01", f"{last}-08-01")
            or standings != rule["played_counts"]):
        return ["Curtailed archive scope or final standings differs from the exact reviewed season"]
    start = datetime(first, 7, 1, tzinfo=timezone.utc).timestamp()
    end = datetime(last, 8, 1, tzinfo=timezone.utc).timestamp()
    for event in events:
        tournament, season = event.get("tournament") or {}, event.get("season") or {}
        number, stamp = (event.get("roundInfo") or {}).get("round"), event.get("startTimestamp")
        status = event.get("status") or {}
        if ((tournament.get("uniqueTournament") or {}).get("id") != rule["competition_id"]
                or tournament.get("isGroup") is not False
                or season.get("id") != rule["season_id"]
                or season_year_pair(season.get("year")) != rule["years"]
                or type(number) is not int or not 1 <= number <= full_rounds
                or type(stamp) is not int or not start <= stamp < end
                or event.get("isAwarded", False) is not False
                or event.get("homeTeam", {}).get("id") == event.get("awayTeam", {}).get("id")
                or (status.get("type") == "finished" and (status.get("code") != 100
                    or any(type((event.get(side) or {}).get("current")) is not int
                           or event[side]["current"] < 0 for side in ("homeScore", "awayScore"))))):
            return ["Curtailed archive fixture has an unreviewed identity, date, round or result"]
    if len({e["id"] for e in events}) != len(events):
        return ["Duplicate event ID in the curtailed archive"]
    cutoff = datetime.fromisoformat(rule["cutoff"]).replace(tzinfo=timezone.utc).timestamp()
    return validate_curtailed_season(events, set(standings),
        {str(t): n for t, n in standings.items()}, rule["meetings"],
        {**rule, "last_played_before": cutoff})
