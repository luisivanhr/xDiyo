"""Exact reviewed home/away metadata corrections; provider payloads stay intact.

The Saudi league's official results list Hazem-Wehda (30 August 2018, 0-0)
and Ettifaq-Batin (13 September 2018, 3-2). Sofascore reverses the complete
side presentation for these two IDs, including scores and component labels.
Keep team/player attribution and values; reverse only normalized side labels.

https://www.spl.com.sa/en/match/8be3d4d9ac2a452d9091acd697c8f462/al-hazem-vs-al-wehda
https://www.spl.com.sa/ar/match/2f8848b94eaa4b0d856a18c4bc5f068f/

The same exact correction applies to Shakhtar-Vorskla, 24 October 2020,
1-1. The return fixture on 4 April 2021 was correctly Vorskla-Shakhtar.
https://upl.ua/ua/news/view/4609
"""


def reviewed_reversed_sides(event):
    rule = {
        # 2022 CSL venue swaps were not applied to these provider side labels.
        # Exact source notices/results are retained in the remaining review.
        10536153: (649, 41892, {"2022"}, (13, 221241, 3373, 1660728600, 2, 0)),
        10536163: (649, 41892, {"2022"}, (14, 3367, 3373, 1661081400, 1, 1)),
        10943287: (649, 41892, {"2022"}, (32, 3373, 66440, 1671796800, 3, 0)),
        7884024: (955, 17906, {"18/19", "2018/19", "2018/2019"},
                  (1, 32994, 168086, 1535645100, 0, 0)),
        7885025: (955, 17906, {"18/19", "2018/19", "2018/2019"},
                  (2, 168084, 34318, 1536852300, 2, 3)),
        8853483: (218, 29387, {"20/21", "2020/21", "2020/2021"},
                  (7, 3309, 3313, 1603537200, 1, 1)),
        # Reviewed Egyptian venue/home-allocation changes. Sources and raw
        # identities are recorded in coverage_egypt4_reviewed_fixtures.json.
        7618577: (808, 14262, {"17/18", "2017/18", "2017/2018"},
                  (6, 43559, 43564, 1508425200, 1, 2)),
        7589258: (808, 14262, {"17/18", "2017/18", "2017/2018"},
                  (4, 43559, 58317, 1506784500, 1, 2)),
        7925326: (808, 18033, {"18/19", "2018/19", "2018/2019"},
                  (12, 43560, 42366, 1541257200, 0, 0)),
        9231878: (808, 34930, {"20/21", "2020/21", "2020/2021"},
                  (7, 209028, 58319, 1610627400, 1, 1)),
        9905528: (808, 39373, {"21/22", "2021/22", "2021/2022"},
                  (1, 42368, 43564, 1635271200, 2, 0)),
        12097995: (808, 55005, {"23/24", "2023/24", "2023/2024"},
                   (12, 6910, 58317, 1718985600, 2, 1)),
    }.get(event.get("id"))
    if rule is None:
        return False
    competition_id, season_id, years, expected = rule
    season, status = event.get("season") or {}, event.get("status") or {}
    actual = ((event.get("roundInfo") or {}).get("round"),
              (event.get("homeTeam") or {}).get("id"), (event.get("awayTeam") or {}).get("id"),
              event.get("startTimestamp"), (event.get("homeScore") or {}).get("current"),
              (event.get("awayScore") or {}).get("current"))
    return (actual == expected
        and (event.get("tournament") or {}).get("uniqueTournament", {}).get("id") == competition_id
        and season.get("id") == season_id and season.get("year") in years
        and status.get("type") == "finished" and status.get("code") == 100
        and event.get("isAwarded", False) is False)


def match_team_ids(event):
    """Canonical ordered IDs for validation and publication, never raw mutation."""
    pair = tuple((event.get(s + "Team") or {}).get("id") for s in ("home", "away"))
    return pair[::-1] if reviewed_reversed_sides(event) else pair


def opposite_side(side):
    return {"home": "away", "away": "home"}.get(side, side)
