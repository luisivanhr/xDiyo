"""Narrow corrections for verified provider round-label errors.

Nine Serie B 2015/16 fixtures belong to round 4 but are labelled round 0;
the rescheduled Lecce-Ascoli 2018/19 fixture has no round label. Wigan-Doncaster
2013/14 belongs to round 2 but is labelled round 3. Competition,
season, ordered teams and kickoff must all match before correcting the
normalized round. Original event payloads and stage_json remain unchanged.
"""

def reviewed_j2_2018_metadata(event):
    """Identify two exact round-39 responses with missing season/round fields.

    Official match reports establish 2018 J2 round 39, not inferred pairings:
    https://www.rbomiya.com/match/368/
    https://www.ventforet.jp/games/pdf/game2018.pdf
    Raw provider payloads remain unchanged, including their absent fields.
    """
    expected = {8036367: (5130, 3149, 1, 2), 8036368: (5762, 38157, 1, 0)}.get(event.get("id"))
    tournament = event.get("tournament") or {}
    return bool(expected and event.get("season") is None and event.get("roundInfo") is None
        and tournament.get("id") == 3034 and tournament.get("isGroup") is False
        and (tournament.get("uniqueTournament") or {}).get("id") == 402
        and ((tournament.get("category") or {}).get("sport") or {}).get("id") == 1
        and event.get("startTimestamp") == 1540710000
        and (event.get("status") or {}).get("type") == "finished"
        and (event.get("status") or {}).get("code") == 100
        and ((event.get("homeTeam") or {}).get("id"), (event.get("awayTeam") or {}).get("id"),
             (event.get("homeScore") or {}).get("current"), (event.get("awayScore") or {}).get("current")) == expected)


def match_season(event):
    """Read season identity with the documented exact historical correction."""
    if reviewed_j2_2018_metadata(event):
        return {"id": 16005, "year": "2018"}
    return event.get("season") or {}


SERIE_B_2015_ROUND_FOUR = {
    6881870: (378962, 2722, 1442946600),
    6881871: (2718, 2712, 1442946600),
    6881872: (7935, 2710, 1442946600),
    6881874: (2708, 2726, 1442860200),
    6881875: (2735, 2698, 1442946600),
    6881876: (2730, 2767, 1442946600),
    6881877: (2704, 43686, 1442946600),
    6881878: (2742, 2706, 1442946600),
    6881879: (2691, 2769, 1442946600),
}


def match_round(event):
    """Return the provider round, or the verified correction for exact identity."""
    if reviewed_j2_2018_metadata(event):
        return 39
    number = (event.get("roundInfo") or {}).get("round")
    # The rescheduled Lecce-Ascoli match has no roundInfo in Sofascore.
    # It completed round 22 on 23 March 2019; preserve the raw missing label.
    # https://uslecce.it/il-presidente-balata-consegna-due-targhe-al-lecce-e-allascoli/
    season = event.get("season") or {}
    cid = ((event.get("tournament") or {}).get("uniqueTournament") or {}).get("id")
    # Official J.League reports identify these rescheduled games' original rounds.
    # https://www.jleague.jp/news/article/6012/?mode=pc
    # https://www.jleague.jp/sp/match/j1/2020/110321/live/
    japanese = {
        7071662: (11307, "2016", 1, 21904, 3146, 1464861600, 7),
        9117396: (27042, "2020", 25, 3137, 3134, 1604376000, 31),
    }.get(event.get("id"))
    if (japanese and cid == 196
            and (season.get("id"), season.get("year"), number,
                 (event.get("homeTeam") or {}).get("id"),
                 (event.get("awayTeam") or {}).get("id"), event.get("startTimestamp"))
            == japanese[:-1]):
        return japanese[-1]
    # This game in hand was played after the split, but completed round 29.
    # The separate 16 May post-split meeting (6740413) remains excluded.
    # https://www.skysports.com/football/inverness-ct-vs-dundee-united/report/315594
    if (number == 37 and event.get("id") == 6726416 and cid == 36
            and season.get("id") == 8190 and season.get("year") in {"14/15", "2014/15", "2014/2015"}
            and ((event.get("homeTeam") or {}).get("id"),
                 (event.get("awayTeam") or {}).get("id"), event.get("startTimestamp"))
            == (2362, 2349, 1430851500)):
        return 29
    # The delayed opening fixture is misfiled with the April round-32 meeting.
    # https://www.celticfc.com/news/7222
    if (number == 32 and event.get("id") == 5587698 and cid == 36
            and season.get("id") == 8190 and season.get("year") in {"14/15", "2014/15", "2014/2015"}
            and ((event.get("homeTeam") or {}).get("id"),
                 (event.get("awayTeam") or {}).get("id"), event.get("startTimestamp"))
            == (2352, 2358, 1417635900)):
        return 1
    # The only missing pairing in round 2; both clubs already play in round 3.
    # The FA fixture list confirms the rearranged 20 August kickoff:
    # https://www.thefa.com/~/media/Files/PDF/Leagues/FLC-fixtures1314
    if (number == 3 and event.get("id") == 3967161 and cid == 18
            and season.get("id") == 6315 and season.get("year") in {"13/14", "2013/14", "2013/2014"}
            and ((event.get("homeTeam") or {}).get("id"),
                 (event.get("awayTeam") or {}).get("id"), event.get("startTimestamp"))
            == (49, 84, 1377024300)):
        return 2
    if (number is None and event.get("id") == 8168679 and cid == 53
            and season.get("id") == 18325 and season.get("year") in {"18/19", "2018/19", "2018/2019"}
            and ((event.get("homeTeam") or {}).get("id"),
                 (event.get("awayTeam") or {}).get("id"), event.get("startTimestamp"))
            == (2689, 2707, 1553360400)):
        return 22
    if type(number) is not int or number != 0:
        return number
    cid = ((event.get("tournament") or {}).get("uniqueTournament") or {}).get("id")
    season = event.get("season") or {}
    expected = SERIE_B_2015_ROUND_FOUR.get(event.get("id"))
    if (expected and cid == 53 and season.get("id") == 10753
            and season.get("year") in {"15/16", "2015/16", "2015/2016"}
            and ((event.get("homeTeam") or {}).get("id"),
                 (event.get("awayTeam") or {}).get("id"), event.get("startTimestamp")) == expected):
        return 4
    return number
