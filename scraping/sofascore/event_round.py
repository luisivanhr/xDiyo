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


def reviewed_egypt_2019_round(event):
    """Eight COVID-restart fixtures lack roundInfo in the original response.

    Captured original Chrome HARs establish the event identity; contemporary
    match reports establish round 14, 15, 18 or 19. Sources are retained with
    the Egypt review receipt. No provider field is changed in raw storage.
    """
    rule = {
        8816749: (42368, 43560, 1596736800, 1, 0, 14),
        8822046: (43559, 209028, 1596985200, 2, 1, 18),
        8822047: (6910, 43564, 1596996000, 2, 0, 15),
        8823198: (175002, 216375, 1597077000, 3, 0, 19),
        8831404: (209028, 373168, 1597327200, 1, 1, 19),
        8831405: (43563, 42365, 1597336200, 1, 1, 19),
        8835330: (43565, 35187, 1597413600, 1, 1, 19),
        8835333: (43562, 6910, 1597431600, 1, 1, 19),
    }.get(event.get("id"))
    season, status = event.get("season") or {}, event.get("status") or {}
    tournament = event.get("tournament") or {}
    actual = ((event.get("homeTeam") or {}).get("id"), (event.get("awayTeam") or {}).get("id"),
              event.get("startTimestamp"), (event.get("homeScore") or {}).get("current"),
              (event.get("awayScore") or {}).get("current"))
    if (rule and actual == rule[:-1] and event.get("roundInfo") is None
            and tournament.get("id") == 38612 and tournament.get("isGroup") is False
            and (tournament.get("uniqueTournament") or {}).get("id") == 808
            and season.get("id") == 25455 and season.get("year") in {"19/20", "2019/20", "2019/2020"}
            and status.get("type") == "finished" and status.get("code") == 100
            and event.get("isAwarded", False) is False):
        return rule[-1]
    return None


def reviewed_egypt_2020_round(event):
    """Match reports identify exact postponed games; never rewrite raw data.

    https://www.youm7.com/story/2021/1/18/-/5163983
    https://www.mahttmsr.com/8567
    https://www.btolat.com/news/250957
    https://english.ahram.org.eg/NewsContent/6/51/418503/Sports/Egyptian-Football/Egyptian-Premier-League-results--scorers-th-matchd.aspx
    """
    rule = {
        9278105: (7, 42366, 42365, 1610982000, 0, 1, 6),
        9289854: (7, 35187, 43564, 1614610800, 0, 1, 4),
        9689650: (29, 6910, 371622, 1628362800, 4, 0, 23),
        9689656: (30, 43559, 371622, 1628614800, 1, 2, 29),
        9689657: (30, 43565, 266626, 1628442000, 1, 1, 29),
        9689658: (30, 43560, 42365, 1628449200, 3, 1, 29),
        9689659: (30, 43561, 175002, 1628449200, 1, 1, 29),
        9689660: (30, 35187, 42368, 1628622000, 1, 2, 29),
        9689663: (30, 58319, 43568, 1628614800, 2, 2, 29),
        9689664: (30, 43564, 43563, 1628528400, 0, 0, 29),
    }.get(event.get("id"))
    season, status = event.get("season") or {}, event.get("status") or {}
    actual = ((event.get("roundInfo") or {}).get("round"),
              (event.get("homeTeam") or {}).get("id"), (event.get("awayTeam") or {}).get("id"),
              event.get("startTimestamp"), (event.get("homeScore") or {}).get("current"),
              (event.get("awayScore") or {}).get("current"))
    if (rule and actual == rule[:-1]
            and (event.get("tournament") or {}).get("uniqueTournament", {}).get("id") == 808
            and season.get("id") == 34930 and season.get("year") in {"20/21", "2020/21", "2020/2021"}
            and status.get("type") == "finished" and status.get("code") == 100
            and event.get("isAwarded", False) is False):
        return rule[-1]
    return None


def reviewed_remaining_round(event):
    """Exact postponed-game labels, checked against the original fixture lists.

    Colo-Colo/Wanderers: ANFP's rescheduled date-30 notice, 6 November 2021.
    Alianza Petrolera/DIM: the 26 October 2021 date-16 match report.
    Tecnico/Mushuc Runa: LigaPro's official second-stage 2024 calendar, date 8.
    Sources and original response hashes are retained in the remaining review.
    """
    rule = {
        10829731: (155, 41884, 68, 3205, 3202, 1666288800, 1, 2, None, 23),
        9927907: (11653, 36048, 64475, 3155, 3150, 1636232400, 3, 0, 31, 30),
        9652499: (11536, 37473, 19236, 89156, 6115, 1635289200, 3, 0, 15, 16),
        12014145: (240, 58043, 1387, 21941, 134976, 1727728200, 2, 2, 6, 8),
    }.get(event.get("id"))
    tournament, status = event.get("tournament") or {}, event.get("status") or {}
    actual = ((tournament.get("uniqueTournament") or {}).get("id"),
              (event.get("season") or {}).get("id"), tournament.get("id"),
              (event.get("homeTeam") or {}).get("id"), (event.get("awayTeam") or {}).get("id"),
              event.get("startTimestamp"), (event.get("homeScore") or {}).get("current"),
              (event.get("awayScore") or {}).get("current"), (event.get("roundInfo") or {}).get("round"))
    if (rule and actual == rule[:-1] and status.get("type") == "finished"
            and status.get("code") == 100 and not event.get("isAwarded", False)):
        return rule[-1]
    return None


def match_round(event):
    """Return the provider round, or the verified correction for exact identity."""
    if reviewed_j2_2018_metadata(event):
        return 39
    if (remaining_round := reviewed_remaining_round(event)) is not None:
        return remaining_round
    if (egypt_round := reviewed_egypt_2019_round(event)) is not None:
        return egypt_round
    if (egypt_round := reviewed_egypt_2020_round(event)) is not None:
        return egypt_round
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
