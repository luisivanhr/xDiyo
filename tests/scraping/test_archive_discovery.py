"""Offline contract tests for archive discovery, special formats and publication."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import socket

import pytest

from scraping.sofascore.export import read_tables
from scraping.sofascore.store import season_year_pair
from scraping.sofascore_web_v3.archive_discovery import prepare_archive
from scraping.sofascore_web_v3.archive_formats import _spec, requested_archives, validate_schedule, is_regular
from scraping.sofascore_web_v3.campaign import Campaign
from scraping.sofascore_web_v3.discovery import open_discovery
from scraping.sofascore_web_v3.native_host import Host
from scraping.sofascore_web_v3.publish import publish_captured_season

FIXTURES = Path(__file__).parents[1] / "fixtures/sofascore"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", lambda *a, **kw: pytest.fail("Python must remain offline"))
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **kw: pytest.fail("Python must remain offline"))


def fixture(name):
    return json.loads((FIXTURES / f"12436870_{name}.json").read_text(encoding="utf-8"))


def events(teams, *, year="2010", cid=196, sid=10, meetings=2, group=False, id_base=100):
    ring = list(teams) + ([None] if len(teams) % 2 else [])
    first = []
    for number in range(1, len(ring)):
        for i in range(len(ring)//2):
            home, away = ring[i], ring[-1-i]
            if home is not None and away is not None:
                first.append((number, home, away))
        ring = [ring[0], ring[-1], *ring[1:-1]]
    schedule = first + ([(n + len(ring)-1, a, h) for n, h, a in first] if meetings == 2 else [])
    result = []
    for i, (n, h, a) in enumerate(schedule):
        event = deepcopy(fixture("event")["event"])
        event.update(id=id_base+i, season={"id": sid, "year": year}, roundInfo={"round": n},
            startTimestamp=int(datetime(season_year_pair(year)[0], 3, 1, tzinfo=timezone.utc).timestamp()) + n*86400,
            status={"type": "finished", "description": "Ended", "code": 100})
        event["tournament"] = dict(id=cid, name="Example League", slug="example-league", isGroup=group,
                                     uniqueTournament={"id": cid})
        event["homeTeam"].update(id=h, name=f"Team {h}", slug=f"team-{h}")
        event["awayTeam"].update(id=a, name=f"Team {a}", slug=f"team-{a}")
        result.append(event)
    return result


def prepare(tmp_path, specs):
    return prepare_archive("test", specs, base=tmp_path / "batches", discovery_base=tmp_path / "discoveries",
                           output=tmp_path / "output")


def observe(discovery, job, payload, code=200):
    entry = {"startedDateTime": datetime.now(timezone.utc).isoformat(), "time": 10,
        "request": {"method": "GET", "url": "https://www.sofascore.com/api/v1" + job["path"]},
        "response": {"status": code, "content": {"text": json.dumps(payload)}}}
    return discovery.observe(job["path"], job["lease"], entry)


def drain(discovery, values, standings=None):
    paths = []
    while True:
        job = discovery.next_job(resume=not paths)
        if job.get("done"):
            return paths
        assert not job.get("paused"), job
        paths.append(job["path"])
        cid, sid = job["competition_id"], job["season_id"]
        selected = [e for e in values if e["tournament"]["uniqueTournament"]["id"] == cid
                    and (sid is None or e["season"]["id"] == sid)]
        if job["path"].endswith("/seasons"):
            payload = {"seasons": list({e["season"]["id"]: e["season"] for e in selected}.values())}
        elif job["path"].endswith("/rounds"):
            payload = {"rounds": [{"round": n} for n in sorted({e["roundInfo"]["round"] for e in selected})]
                       + [{"round": 0}, {"round": 100, "name": "Final"}]}
        elif job["path"].endswith("/standings/total"):
            played = standings or {t: sum(t in (e["homeTeam"]["id"], e["awayTeam"]["id"])
                                          and e["status"]["type"] == "finished" for e in selected)
                                    for e in selected for t in (e["homeTeam"]["id"], e["awayTeam"]["id"])}
            payload = {"standings": [{"type": "total", "tournament": selected[0]["tournament"],
                                      "rows": [{"team": {"id": t}, "matches": n} for t, n in played.items()]}]}
        else:
            number = int(job["path"].rsplit("/", 1)[1])
            payload = {"events": [e for e in selected if e["roundInfo"]["round"] == number]}
        observe(discovery, job, payload)


@pytest.mark.parametrize("label,expected", [("2010", (2010, 2010)), ("2026", (2026, 2026)),
    ("26/27", (2026, 2027)), ("2010/2011", (2010, 2011)), ("2010 garbage", None), (None, None)])
def test_calendar_identity_does_not_alias_two_year_season(label, expected):
    assert season_year_pair(label) == expected


def test_reviewed_wigan_round_keeps_source_bucket_and_normalizes_export(tmp_path):
    from scraping.sofascore.parsers import parse_match
    spec = _spec("Championship", 18, "2013/2014", teams=4, rounds=6)
    values = events([49, 1, 84, 2], year="13/14", cid=18, sid=6315)
    changed = next(e for e in values if e["homeTeam"]["id"] == 49 and e["awayTeam"]["id"] == 84)
    assert changed["roundInfo"]["round"] == 2
    changed.update(id=3967161, roundInfo={"round": 3}, startTimestamp=1377024300)
    before = deepcopy(changed)
    saved = prepare(tmp_path, [spec])
    with open_discovery(saved["directory"]) as discovery:
        drain(discovery, values)
        assert discovery.status()["ready_to_collect"]
        row = discovery.db.execute("SELECT * FROM events WHERE id=3967161").fetchone()
        assert row["round"] == 3 and json.loads(row["payload"]) == before
    assert parse_match(changed)["round"] == 2 and changed == before
    # A different match cannot use this correction to bypass round validation.
    bad = deepcopy(changed); bad["id"] += 1
    assert validate_schedule([bad if e is changed else e for e in values], spec)


@pytest.mark.parametrize("field", ["id", "competition", "season", "year", "round", "home", "away", "kickoff"])
def test_wigan_round_correction_rejects_changed_identity(field):
    from scraping.sofascore.event_round import match_round
    e = dict(id=3967161, tournament={"uniqueTournament": {"id": 18}},
        season={"id": 6315, "year": "13/14"}, roundInfo={"round": 3},
        homeTeam={"id": 49}, awayTeam={"id": 84}, startTimestamp=1377024300)
    assert match_round(e) == 2
    if field == "id": e["id"] += 1
    elif field == "competition": e["tournament"]["uniqueTournament"]["id"] += 1
    elif field == "season": e["season"]["id"] += 1
    elif field == "year": e["season"]["year"] = "14/15"
    elif field == "round": e["roundInfo"]["round"] = 4
    elif field == "home": e["homeTeam"]["id"] += 1
    elif field == "away": e["awayTeam"]["id"] += 1
    else: e["startTimestamp"] += 1
    assert match_round(e) == e["roundInfo"]["round"]


@pytest.mark.parametrize("original,played,expected_score,original_status", [
    ((9309246, 196, 35273, "2021", 20, 5127, 3148, 1624701600),
     (9486403, 196, 35273, "2021", 20, 5127, 3148, 1620813600), (2, 2), "canceled"),
    ((9309261, 196, 35273, "2021", 20, 3138, 3135, 1624784400),
     (9486404, 196, 35273, "2021", 20, 3138, 3135, 1620810000), (1, 2), "canceled"),
    ((9309348, 196, 35273, "2021", 13, 3136, 3141, 1625049000),
     (9486368, 196, 35273, "2021", 13, 3136, 3141, 1620453600), (1, 0), "postponed"),
    ((9309349, 196, 35273, "2021", 13, 3138, 5127, 1625047200),
     (9486384, 196, 35273, "2021", 13, 3138, 5127, 1620464400), (0, 2), "postponed"),
    ((1379596, 17, 2746, "10/11", 1, 67, 49, 1281794400),
     (1380091, 17, 2746, "10/11", 1, 49, 67, 1281794400), (0, 4), "postponed"),
    ((3964835, 17, 6311, "13/14", 3, 38, 40, 1377957600),
     (4166893, 17, 6311, "13/14", 3, 38, 40, 1377110700), (2, 1), "postponed"),
    ((8060497, 34, 8122, "14/15", 10, 1642, 1649, 1413745200),
     (5509696, 34, 8122, "14/15", 10, 1649, 1642, 1413745200), (5, 1), "canceled"),
    ((1380321, 36, 2752, "10/11", 33, 2348, 2352, 1302962400),
     (1694223, 36, 2752, "10/11", 33, 2348, 2352, 1302627600), (0, 1), "postponed"),
    ((3971519, 36, 6309, "13/14", 25, 2353, 2359, 1391024700),
     (4934258, 36, 6309, "13/14", 25, 2353, 2359, 1391024700), (2, 1), "postponed"),
])
def test_reviewed_replacements_require_both_exact_records(original, played, expected_score, original_status):
    from scraping.sofascore_web_v3.discovery import select_regular_records
    from scraping.sofascore_web_v3.historical_formats import verified_replacement_id

    def row(identity, status):
        eid, cid, sid, year, number, home, away, kickoff = identity
        e = dict(id=eid, tournament={"uniqueTournament": {"id": cid}},
            season={"id": sid, "year": year}, roundInfo={"round": number},
            homeTeam={"id": home}, awayTeam={"id": away}, startTimestamp=kickoff,
            status={"type": status})
        if status == "finished":
            e.update(homeScore={"current": expected_score[0]}, awayScore={"current": expected_score[1]})
        return dict(id=eid, competition=cid, season=sid, round=number, regular=1, payload=json.dumps(e))

    old, new = row(original, original_status), row(played, "finished")
    before = deepcopy([old, new])
    selected, superseded = select_regular_records(iter([old, new]))
    assert selected == [new]
    assert superseded == [dict(event_id=original[0], replacement_event_id=played[0],
        status="cancelled" if original_status == "canceled" else original_status)]
    assert [old, new] == before
    assert select_regular_records([old]) == ([old], [])
    for index in range(len(played)):
        for target in (0, 1):
            identity = list(original if target == 0 else played)
            identity[index] = "15/16" if index == 3 else identity[index] + 1
            pair = [row(identity, original_status), new] if target == 0 else [old, row(identity, "finished")]
            assert verified_replacement_id(json.loads(pair[0]["payload"]),
                {pair[1]["id"]: json.loads(pair[1]["payload"])}) is None
    wrong_score = deepcopy(new)
    payload = json.loads(wrong_score["payload"])
    payload["awayScore"]["current"] += 1
    wrong_score["payload"] = json.dumps(payload)
    assert len(select_regular_records([old, wrong_score])[0]) == 2


@pytest.mark.parametrize("pair", json.loads((FIXTURES / "mexico_reviewed_duplicates.json").read_text(encoding="utf-8"))["pairs"],
                         ids=lambda p: str(p["original"]["id"]))
def test_mexico_reviewed_duplicates_preserve_raw_and_require_exact_evidence(pair):
    from scraping.sofascore_web_v3.discovery import select_regular_records
    from scraping.sofascore_web_v3.historical_formats import verified_replacement_id
    from scraping.sofascore.discovery import is_completed_event
    before = deepcopy(pair)
    old, new = pair["original"], pair["replacement"]

    def row(e):
        return dict(id=e["id"], competition=e["tournament"]["uniqueTournament"]["id"],
                    season=e["season"]["id"], round=e["roundInfo"]["round"],
                    regular=1, payload=json.dumps(e))

    chosen, superseded = select_regular_records([row(old), row(new)])
    assert chosen == [row(new)]
    assert superseded[0]["event_id"] == old["id"]
    assert superseded[0]["replacement_event_id"] == new["id"]
    assert select_regular_records([row(old)]) == ([row(old)], [])
    if new["status"]["type"] == "notstarted":
        assert not is_completed_event(new)
    for side in ("original", "replacement"):
        for path in (("id",), ("season", "id"), ("season", "year"), ("roundInfo", "round"),
                     ("tournament", "uniqueTournament", "id"), ("homeTeam", "id"),
                     ("awayTeam", "id"), ("startTimestamp",), ("status", "type")):
            changed = deepcopy(pair)
            node = changed[side]
            for key in path[:-1]:
                node = node[key]
            value = node[path[-1]]
            node[path[-1]] = value + 1 if isinstance(value, int) else "wrong"
            assert verified_replacement_id(changed["original"], {new["id"]: changed["replacement"]}) is None
        changed = deepcopy(pair)
        changed[side].setdefault("awayScore", {})["current"] = 99
        assert verified_replacement_id(changed["original"], {new["id"]: changed["replacement"]}) is None
    assert pair == before


def test_requested_scope_counts_and_chronology():
    all_specs = requested_archives(japan_transition="all")
    eu, jp, mx = all_specs.values()
    assert [len(x) for x in (eu, jp, mx)] == [65, 36, 33]
    assert len({s["competition_id"] for s in eu}) == 13
    assert {s["season"] for s in eu} == {f"{y}/{y+1}" for y in range(2010, 2015)}
    assert all(s["round_limit"] == 33 for s in eu if s["competition_id"] == 36)
    assert jp[1]["expected_teams"] == 19 and jp[1]["round_limit"] == 38
    assert [s["groups"] for s in jp[-4:-2]] == [2, 4]
    assert [s["season"] for s in jp[-4:]] == ["2026", "2026", "2026/2027", "2026/2027"]
    assert mx[0]["tournament"] == "Apertura" and mx[1]["tournament"] == "Clausura"
    assert mx[0]["season"] == mx[1]["season"] == "2010/2011"
    assert mx[-1]["season"] == "2026/2027" and mx[-1]["snapshot"]
    assert next(s for s in mx if s["tournament"] == "Apertura" and s["season"] == "2019/2020")["expected_teams"] == 19
    assert next(s for s in mx if s["tournament"] == "Clausura" and s["season"] == "2019/2020")["round_limit"] == 10
    assert len(requested_archives(japan_transition="skip")["japan-2010-2027"]) == 35


@pytest.mark.parametrize("pair", json.loads((FIXTURES / "j2_reviewed_duplicates.json").read_text(encoding="utf-8"))["pairs"],
                         ids=lambda p: str(p["original"]["id"]))
def test_j2_reviewed_stubs_and_replays_require_exact_provider_records(pair):
    from scraping.sofascore_web_v3.discovery import select_regular_records
    from scraping.sofascore_web_v3.historical_formats import verified_replacement_id
    old, new = pair["original"], pair["played"]
    before = deepcopy(pair)
    assert verified_replacement_id(old, {new["id"]: new}) == new["id"]
    def row(e):
        return dict(id=e["id"], competition=402, season=e["season"]["id"],
                    round=e["roundInfo"]["round"], regular=1, payload=json.dumps(e))
    chosen, superseded = select_regular_records([row(old), row(new)])
    assert chosen == [row(new)] and superseded[0]["event_id"] == old["id"]
    assert select_regular_records([row(old)]) == ([row(old)], [])
    for side in ("original", "played"):
        for path in (("id",), ("season", "id"), ("season", "year"), ("roundInfo", "round"),
                     ("tournament", "uniqueTournament", "id"), ("homeTeam", "id"),
                     ("awayTeam", "id"), ("startTimestamp",), ("status", "type")):
            changed = deepcopy(pair)
            node = changed[side]
            for key in path[:-1]: node = node[key]
            value = node[path[-1]]
            node[path[-1]] = value + 1 if isinstance(value, int) else "wrong"
            assert verified_replacement_id(changed["original"], {new["id"]: changed["played"]}) is None
        if pair[side]["status"]["type"] == "finished":
            changed = deepcopy(pair)
            changed[side]["awayScore"]["current"] += 1
            assert verified_replacement_id(changed["original"], {new["id"]: changed["played"]}) is None
    assert pair == before


@pytest.mark.parametrize("mutation", [None, "season", "competition", "year", "round", "stage", "event", "url"])
def test_archive_missing_fixture_recovery_preserves_round_evidence(tmp_path, mutation):
    values = events([1, 2, 3, 4])
    missing = values[-1]
    config = prepare(tmp_path, [_spec("J1", 196, "2010", teams=4, rounds=6)])
    with open_discovery(config["directory"]) as discovery:
        drain(discovery, values[:-1])
        assert discovery.status()["discovery_complete"]
        assert not discovery.status()["ready_to_collect"]
        before = list(discovery.db.execute("SELECT path,body,sha256 FROM responses ORDER BY path"))
        value = deepcopy(missing)
        if mutation == "season": value["season"]["id"] += 1
        if mutation == "competition": value["tournament"]["uniqueTournament"]["id"] += 1
        if mutation == "year": value["season"]["year"] = "2011"
        if mutation == "round": value["roundInfo"]["round"] -= 1
        if mutation == "stage": value["tournament"]["isGroup"] = True
        if mutation == "event": value["id"] += 1
        entry = {"startedDateTime": datetime.now(timezone.utc).isoformat(), "time": 10,
            "request": {"method": "GET", "url": f"https://www.sofascore.com/api/v1/event/{missing['id']}"},
            "response": {"status": 200, "content": {"text": json.dumps({"event": value})}}}
        if mutation == "url": entry["request"]["url"] += "?extra=1"
        def recover():
            return discovery.admit_recovered_event(entry, event_id=missing["id"], competition_id=196,
                season_id=10, round_number=6)
        if mutation is None:
            assert recover()["summary"]["ready_to_collect"]
            assert discovery.prepare_collection(campaign_base=tmp_path / "campaigns")["matches"] == 12
            with pytest.raises(ValueError, match="without a collection campaign"): recover()
        else:
            with pytest.raises(ValueError): recover()
            assert not discovery.status()["ready_to_collect"]
            assert discovery.db.execute("SELECT COUNT(*) FROM events WHERE id=?", (missing["id"],)).fetchone()[0] == 0
        for path,body,sha in before:
            assert tuple(discovery.db.execute("SELECT body,sha256 FROM responses WHERE path=?", (path,)).fetchone()) == (body,sha)


def carpi_siena_event():
    event = deepcopy(fixture("event")["event"])
    event.update(id=4167965, season={"id": 6821, "year": "13/14"},
        roundInfo={"round": 15}, startTimestamp=1385301600,
        status={"type": "postponed", "code": 60, "description": "Postponed"},
        homeScore={"current": 0}, awayScore={"current": 1})
    event["tournament"].update(uniqueTournament={"id": 53}, isGroup=False)
    event["homeTeam"]["id"], event["awayTeam"]["id"] = 43685, 170650
    return event


def test_reviewed_celtic_round_is_exact_and_preserves_raw():
    from scraping.sofascore.parsers import parse_match
    from scraping.sofascore.event_round import match_round
    event = deepcopy(fixture("event")["event"])
    event.update(id=5587698, season={"id": 8190, "year": "14/15"},
        roundInfo={"round": 32}, startTimestamp=1417635900)
    event["tournament"]["uniqueTournament"]["id"] = 36
    event["homeTeam"]["id"], event["awayTeam"]["id"] = 2352, 2358
    before = deepcopy(event)
    assert parse_match(event)["round"] == 1 and event == before
    for path in ("id", "season.id", "season.year", "roundInfo.round", "startTimestamp",
                 "tournament.uniqueTournament.id", "homeTeam.id", "awayTeam.id"):
        changed = deepcopy(event)
        target = changed
        parts = path.split(".")
        for key in parts[:-1]:
            target = target[key]
        key = parts[-1]
        target[key] = target[key] + 1 if isinstance(target[key], int) else "unknown"
        assert match_round(changed) == changed["roundInfo"]["round"]


def test_standard_zulte_abandonment_is_terminal_but_not_completed():
    from scraping.sofascore.discovery import event_status, is_completed_event, verified_abandoned_event
    event = carpi_siena_event()
    event.update(id=5555478, season={"id": 8168, "year": "14/15"},
        roundInfo={"round": 11}, startTimestamp=1413721800,
        homeScore={"current": 1}, awayScore={"current": 2})
    event["tournament"]["uniqueTournament"]["id"] = 38
    event["homeTeam"]["id"], event["awayTeam"]["id"] = 2901, 2933
    before = deepcopy(event)
    assert event_status(event) == "abandoned" and not is_completed_event(event) and event == before
    for path in ("id", "season.id", "season.year", "roundInfo.round", "startTimestamp",
                 "tournament.uniqueTournament.id", "homeTeam.id", "awayTeam.id",
                 "status.type", "status.code", "homeScore.current", "awayScore.current"):
        changed, target = deepcopy(event), None
        target = changed
        parts = path.split(".")
        for key in parts[:-1]:
            target = target[key]
        key = parts[-1]
        target[key] = target[key] + 1 if isinstance(target[key], int) else "unknown"
        assert not verified_abandoned_event(changed)


def test_inverness_game_in_hand_recovered_without_selecting_post_split(tmp_path):
    from scraping.sofascore.event_round import match_round
    from scraping.sofascore.parsers import parse_match
    spec = _spec("Premiership", 36, "2014/2015", teams=12, meetings=3, rounds=33)
    values = events(range(1, 13), year="14/15", cid=36, sid=8190)
    third = deepcopy(values[:66])
    for event in third:
        event["id"] += 1000
        event["roundInfo"]["round"] += 22
    values += third
    stub = next(e for e in values if e["roundInfo"]["round"] == 29)
    mapping = {stub["homeTeam"]["id"]: 2362, stub["awayTeam"]["id"]: 2349}
    for event in values:
        for side in ("homeTeam", "awayTeam"):
            event[side]["id"] = mapping.get(event[side]["id"], event[side]["id"])
    stub.update(id=5587666, startTimestamp=1426345200, status={"type": "postponed", "code": 60},
                homeScore={}, awayScore={})
    played = deepcopy(stub)
    played.update(id=6726416, roundInfo={"round": 37}, startTimestamp=1430851500,
        status={"type": "finished", "code": 100}, homeScore={"current": 2}, awayScore={"current": 1})
    post_split = deepcopy(played)
    post_split.update(id=6740413, startTimestamp=1431784800,
        homeScore={"current": 3}, awayScore={"current": 0})
    before = deepcopy(played)
    saved = prepare(tmp_path, [spec])
    with open_discovery(saved["directory"]) as discovery:
        paths = drain(discovery, values + [played, post_split])
        assert any(p.endswith("/round/37") for p in paths)
        status = discovery.status()
        assert status["ready_to_collect"] and status["seasons"][0]["selected_matches"] == 198
        row = discovery.db.execute("SELECT * FROM events WHERE id=6726416").fetchone()
        assert row["round"] == 37 and row["regular"] == 1 and json.loads(row["payload"]) == before
        assert discovery.db.execute("SELECT regular FROM events WHERE id=6740413").fetchone()[0] == 0
        assert discovery.db.execute("SELECT COUNT(*) FROM events WHERE id=5587666").fetchone()[0] == 1
        campaign = discovery.prepare_collection(campaign_base=tmp_path / "campaigns")
    with Campaign(campaign["directory"]) as capture:
        assert 6726416 in capture.items and 6740413 not in capture.items and 5587666 not in capture.items
    assert parse_match(played)["round"] == 29 and played == before
    for path in ("id", "season.id", "season.year", "roundInfo.round", "startTimestamp",
                 "tournament.uniqueTournament.id", "homeTeam.id", "awayTeam.id"):
        changed = deepcopy(played)
        target = changed
        parts = path.split(".")
        for key in parts[:-1]:
            target = target[key]
        key = parts[-1]
        target[key] = target[key] + 1 if isinstance(target[key], int) else "unknown"
        assert match_round(changed) == changed["roundInfo"]["round"]
        assert not is_regular(changed, spec)


@pytest.mark.parametrize("year,sid", [("2011/2012", 3414), ("2012/2013", 4676), ("2013/2014", 6241)])
def test_reviewed_mons_standings_alias_leaves_provider_data_unchanged(year, sid):
    from scraping.sofascore_web_v3.archive_formats import reviewed_standings
    values = events([549294, *range(1, 16)], year=year, cid=38, sid=sid)
    for event in values:
        for side in ("homeTeam", "awayTeam"):
            if event[side]["id"] == 549294:
                event[side]["name"] = "RAEC Mons"
    spec = _spec("Pro_League", 38, year)
    standing = {t: 30 for t in [2906, *range(1, 16)]}
    payload = {"standings": [{"type": "total", "tournament": values[0]["tournament"],
        "rows": [{"team": {"id": t, "name": "RAEC Mons" if t == 2906 else str(t)}, "matches": n}
                 for t, n in standing.items()]}]}
    before = deepcopy((values, standing, payload))
    assert validate_schedule(values, spec, standing)
    assert validate_schedule(values, spec, reviewed_standings(values, spec, standing, payload)) == []
    assert (values, standing, payload) == before
    bad = deepcopy(payload); bad["standings"][0]["rows"][0]["team"]["name"] = "Different club"
    assert reviewed_standings(values, spec, standing, bad) == standing
    assert reviewed_standings(values[:-1], spec, standing, payload) == standing
    for path in ("season.id", "season.year", "tournament.uniqueTournament.id", "homeTeam.name"):
        bad = deepcopy(values)
        target = bad[0]
        parts = path.split(".")
        for key in parts[:-1]:
            target = target[key]
        key = parts[-1]
        target[key] = target[key] + 1 if isinstance(target[key], int) else "unknown"
        assert reviewed_standings(bad, spec, standing, payload) == standing
    assert reviewed_standings(values, {**spec, "season": "2014/2015"}, standing, payload) == standing
    assert reviewed_standings(values, {**spec, "competition_id": 36}, standing, payload) == standing


@pytest.mark.parametrize("path", ["id", "tournament.uniqueTournament.id", "season.id", "season.year",
    "roundInfo.round", "homeTeam.id", "awayTeam.id", "startTimestamp", "status.type", "status.code",
    "homeScore.current", "awayScore.current"])
def test_stale_status_correction_requires_every_verified_field(path):
    from scraping.sofascore.discovery import event_status, verified_finished_event
    from scraping.sofascore_web_v3.import_har import _check_event
    event = carpi_siena_event()
    original = deepcopy(event)
    assert event_status(event) == "finished" and verified_finished_event(event)
    target = event
    parts = path.split(".")
    for key in parts[:-1]:
        target = target[key]
    key = parts[-1]
    target[key] = target[key] + 1 if isinstance(target[key], int) else "unknown"
    assert not verified_finished_event(event) and event_status(event) != "finished"
    with pytest.raises(ValueError, match="identity_or_status"):
        _check_event(event, original)


def test_verified_stale_status_included_in_archive_without_mutating_response(tmp_path):
    values = events(range(1, 23), year="13/14", cid=53, sid=6821)
    changed = next(e for e in values if e["roundInfo"]["round"] == 15)
    mapping = {changed["homeTeam"]["id"]: 43685, changed["awayTeam"]["id"]: 170650}
    for event in values:
        event["startTimestamp"] += 180 * 86400
        for side in ("homeTeam", "awayTeam"):
            event[side]["id"] = mapping.get(event[side]["id"], event[side]["id"])
    changed.update(carpi_siena_event())
    before = deepcopy(changed)
    spec = _spec("Serie_B", 53, "2013/2014", teams=22, rounds=42)
    saved = prepare(tmp_path, [spec])
    with open_discovery(saved["directory"]) as discovery:
        drain(discovery, values)
        status = discovery.status()
        assert status["ready_to_collect"]
        assert status["seasons"][0]["finished"] == 462
        prepared = discovery.prepare_collection(campaign_base=tmp_path / "campaigns")
        assert json.loads(discovery.db.execute("SELECT payload FROM events WHERE id=4167965").fetchone()[0]) == before
    with Campaign(prepared["directory"]) as campaign:
        assert campaign.items[4167965]["reference"]["event"] == before


def test_verified_stale_status_passes_capture_and_normalization(tmp_path):
    from scraping.sofascore_web_v3.campaign import prepare_campaign
    event = carpi_siena_event()
    before, eid = deepcopy(event), event["id"]
    prepared = prepare_campaign([eid], base=tmp_path / "campaigns",
        references={eid: {"event": event, "observed_at": 1700000000}},
        policy={"collection_mode": "session"})
    responses = [("", {"event": event}), ("/pregame-form", {"homeTeam": {"position": 1}, "awayTeam": {"position": 2}}),
        ("/statistics", {"statistics": []}), ("/shotmap", {"shotmap": []}),
        ("/lineups", {"home": {"players": []}, "away": {"players": []}}), ("/incidents", {"incidents": []})]
    responses.extend((f"/heatmap/{event[s]['id']}", {"playerPoints": [], "goalkeeperPoints": []})
        for s in ("homeTeam", "awayTeam"))
    with Campaign(prepared["directory"]) as campaign:
        job = campaign.next_match()
        for suffix, payload in responses:
            campaign.observe(eid, job["lease"], {
                "startedDateTime": datetime.now(timezone.utc).isoformat(), "time": 10,
                "request": {"method": "GET", "url": f"https://www.sofascore.com/api/v1/event/{eid}{suffix}"},
                "response": {"status": 200, "content": {"text": json.dumps(payload)}}})
        result = campaign.finish(eid, job["lease"], page_event_id=None,
            page_evidence={"collection_mode": "session", "session_request_count": len(responses)})
        assert result["state"] == "complete"
        report = json.loads(Path(result["report"]).read_text())
        match = read_tables(report["manifest"])["matches"][0]
        assert match["status"] == "finished" and match["status_code"] == 60
        assert match["home_score_current"] == 0 and match["away_score_current"] == 1
        assert event == before
        raw = json.loads(campaign.entries(eid)[0]["response"]["content"]["text"])["event"]
        assert raw == before


def test_multi_season_one_inventory_one_campaign_and_idempotent_resume(tmp_path):
    specs = [_spec("J1", 196, year, teams=2, rounds=2) for year in ("2010", "2011")]
    saved = prepare(tmp_path, specs)
    assert prepare(tmp_path, specs)["discovery_id"] == saved["discovery_id"]
    values = events([1, 2]) + events([3, 4], year="2011", sid=11, id_base=200)
    with open_discovery(saved["directory"]) as discovery:
        first = discovery.next_job()
        observe(discovery, first, {"seasons": [e["season"] for e in (values[0], values[2])]})
    with open_discovery(saved["directory"]) as discovery:
        paths = drain(discovery, values)
        assert not any(p.endswith("/seasons") for p in paths)
        assert discovery.status()["requests"] == 7
        assert discovery.status()["ready_to_collect"]
        result = discovery.prepare_collection(campaign_base=tmp_path / "campaigns")
        assert discovery.prepare_collection(campaign_base=tmp_path / "campaigns") == result
        # Simulate a power failure between durable campaign creation and the
        # discovery's final checkpoint. Recovery must reuse the same queue.
        with discovery.db:
            discovery.db.execute("DELETE FROM state WHERE key='campaign'")
        assert discovery.prepare_collection(campaign_base=tmp_path / "campaigns") == result
        assert len(list((tmp_path / "campaigns").iterdir())) == 1
        with Campaign(result["directory"]) as campaign:
            assert list(campaign.items) == [100, 101, 200, 201]
            assert {x["reference"]["season_stem"] for x in campaign.items.values()} == {"J1_10_10", "J1_11_11"}
            assert campaign.profile.include_heatmaps and campaign.profile.include_lineups and campaign.profile.include_shotmap
            assert not campaign.profile.include_comments
    specs[0]["round_limit"] = 3
    with pytest.raises(ValueError, match="settings differ"):
        prepare(tmp_path, specs)


def test_ambiguous_season_never_falls_back(tmp_path):
    saved = prepare(tmp_path, [_spec("J1", 196, "2010", teams=2, rounds=2)])
    with open_discovery(saved["directory"]) as discovery:
        job = discovery.next_job()
        with pytest.raises(ValueError, match="ambiguous"):
            observe(discovery, job, {"seasons": [{"id": 1, "year": "2010"}, {"id": 2, "year": "2010"}]})
        assert discovery.next_job()["paused"]
        assert not discovery.status()["ready_to_collect"]
        assert discovery.db.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 1


def test_missing_season_remains_a_gap_while_other_seasons_are_discovered(tmp_path):
    specs = [_spec("J1", 196, year, teams=2, rounds=2) for year in ("2010", "2011")]
    saved = prepare(tmp_path, specs)
    with open_discovery(saved["directory"]) as discovery:
        drain(discovery, events([1, 2]))
        status = discovery.status()
        assert status["discovery_complete"] and not status["ready_to_collect"]
        assert status["seasons"][0]["schedule_verified"]
        assert "absent" in status["seasons"][1]["validation_issues"][0]
        with pytest.raises(ValueError, match="every requested season"):
            discovery.prepare_collection(campaign_base=tmp_path / "campaigns")


def mexico_halves(year=2010, sid=2770):
    values = events(range(1, 19), year=f"{year%100:02d}/{(year+1)%100:02d}",
                    cid=11621, sid=sid, meetings=1)
    first = int(datetime(year, 7, 24, tzinfo=timezone.utc).timestamp())
    second = int(datetime(year+1, 1, 8, tzinfo=timezone.utc).timestamp())
    for event in values:
        event["startTimestamp"] = first + event["roundInfo"]["round"] * 7 * 86400
    other = deepcopy(values)
    for event in other:
        event["id"] += 1000
        event["startTimestamp"] = second + event["roundInfo"]["round"] * 7 * 86400
    return values, other


def test_mexico_shared_2010_source_has_two_complete_disjoint_cohorts(tmp_path):
    specs = requested_archives()["mexico-2010-2026"][:2]
    apertura, clausura = mexico_halves()
    original = deepcopy(apertura + clausura)
    saved = prepare(tmp_path, specs)
    with open_discovery(saved["directory"]) as discovery:
        drain(discovery, apertura + clausura)
        status = discovery.status()
        assert status["ready_to_collect"]
        assert [s["selected_matches"] for s in status["seasons"]] == [153, 153]
        assert all(s["shared_provider_identity"] for s in status["seasons"])
        assert [s["competition_id"] for s in status["seasons"]] == [11621, 11621]
        assert status["seasons"][1]["requested_competition_id"] == 11620
        result = discovery.prepare_collection(campaign_base=tmp_path / "campaigns")
        assert result["matches"] == 306
        with Campaign(result["directory"]) as campaign:
            for half, stem in ((apertura, "Liga_MX_Apertura_10_11"), (clausura, "Liga_MX_Clausura_10_11")):
                assert {i["event_id"] for i in campaign.items.values() if i["reference"]["season_stem"] == stem} == {e["id"] for e in half}
            assert all(i["competition_id"] == 11621 and i["season_id"] == 2770 for i in campaign.items.values())
            assert [i["reference"]["event"] for i in campaign.items.values()] == original
        # Omitting a January fixture still blocks the full campaign.
        with discovery.db:
            discovery.db.execute("DELETE FROM events WHERE id=?", (clausura[0]["id"],))
        discovery._summaries.clear()
        assert not discovery.status()["ready_to_collect"]
    assert apertura + clausura == original


@pytest.mark.parametrize("sid", [2771, 9999])
def test_mexico_missing_clausura_never_aliases_an_unreviewed_source(tmp_path, sid):
    specs = requested_archives()["mexico-2010-2026"][:2]
    apertura, _ = mexico_halves(sid=sid)
    saved = prepare(tmp_path, specs)
    with open_discovery(saved["directory"]) as discovery:
        drain(discovery, apertura)
        status = discovery.status()
        assert not status["ready_to_collect"]
        assert status["seasons"][1]["season_id"] is None


def test_mexico_opposite_half_stays_in_raw_response_without_duplicate_queue(tmp_path):
    specs = requested_archives()["mexico-2010-2026"][2:4]
    apertura, clausura = mexico_halves(2011, 3400)
    values = apertura + clausura
    counterpart = deepcopy(values)
    for event in counterpart:
        event["tournament"]["uniqueTournament"]["id"] = 11620
    saved = prepare(tmp_path, specs)
    with open_discovery(saved["directory"]) as discovery:
        drain(discovery, values + counterpart)
        status = discovery.status()
        assert status["ready_to_collect"]
        assert [s["selected_matches"] for s in status["seasons"]] == [153, 153]
        assert all(not s.get("shared_provider_identity") for s in status["seasons"])
        assert discovery.db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 306
        body = discovery.db.execute("SELECT body FROM responses WHERE path=?",
            ("/unique-tournament/11621/season/3400/events/round/1",)).fetchone()[0]
        assert len(json.loads(body)["events"]) == 18


def test_mexico_shared_publication_has_independent_current_pointers(tmp_path):
    from scraping.sofascore.campaign import publish_season
    from scraping.sofascore.config import Profile, Scope
    from scraping.sofascore.parsers import parse_match
    from scraping.sofascore_web_v3.reconcile_publication import _bundle
    halves = mexico_halves()
    manifests = []
    for half, stem in zip(halves, ("Liga_MX_Apertura_10_11", "Liga_MX_Clausura_10_11")):
        tables = {"matches": [parse_match(event) for event in half]}
        scope = Scope(11621, 2770, tuple(range(1, 18)), league_name=stem.rsplit("_", 2)[0],
                      season_start=2010, season_end=2011)
        result = publish_season(tables, tmp_path, scope, Profile(), stem,
            {"archive_cohort": {"shared_provider_identity": True, "stem": stem,
             "competition_id": 11621, "season_id": 2770, "schedule_verified": True}})
        manifest = tmp_path / result["manifest"]
        manifests.append(manifest)
        assert read_tables(manifest)["matches"] == tables["matches"]
        expected = {e["id"]: (e["id"], 11621, 2770, e["homeTeam"]["id"], e["awayTeam"]["id"]) for e in half}
        assert _bundle(tmp_path, stem, expected)["matches"] == 153
    for manifest in manifests:
        partition = manifest.parent.parent.parent
        current = json.loads((partition / "CURRENT.json").read_text())
        assert partition / current["manifest"] == manifest
    assert manifests[0].parent.parent.parent != manifests[1].parent.parent.parent


def test_shared_provider_cohorts_keep_the_configured_transition_rest(tmp_path):
    from scraping.sofascore_web_v3.campaign import prepare_campaign
    apertura, clausura = mexico_halves()
    selected = [apertura[0], clausura[0]]
    references = {e["id"]: {"event": e, "season_stem": stem} for e, stem in zip(selected,
        ("Liga_MX_Apertura_10_11", "Liga_MX_Clausura_10_11"))}
    result = prepare_campaign(references, references=references, base=tmp_path,
                              policy={"between_seasons_seconds": 300})
    with Campaign(result["directory"]) as campaign:
        with campaign.db:
            campaign.db.execute("UPDATE queue SET state='complete',finished=1000 WHERE event_id=?", (selected[0]["id"],))
        job = campaign.next_match()
        assert job["event_id"] == selected[1]["id"] and job["wait_until"] == 1300


def test_scope_pending_does_not_request_anything(tmp_path):
    spec = _spec("J2_J3_Transition", 402, "2026", teams=40, rounds=18, groups=4, scope_pending=True)
    saved = prepare(tmp_path, [spec])
    with open_discovery(saved["directory"]) as discovery:
        assert discovery.next_job(resume=True)["paused"]
        assert discovery.status()["requests"] == 0


def test_saved_archive_opens_through_existing_extension_native_protocol(tmp_path):
    saved = prepare(tmp_path, [_spec("J1", 196, "2010", teams=2, rounds=2)])
    host = Host(base=tmp_path / "campaigns", discovery_base=tmp_path / "discoveries", lock_path=tmp_path / "collector.lock")
    try:
        summary = host.dispatch({"operation": "discovery_open", "discovery_id": saved["discovery_id"]})
        assert summary["requests"] == 0
        job = host.dispatch({"operation": "discovery_next", "resume": True})
        assert job["path"] == "/unique-tournament/196/seasons" and job["competition_id"] == 196
    finally:
        host.close()


def test_regional_groups_include_regular_games_exclude_placement():
    values = events([1, 2], group=True) + events([3, 4], group=True, id_base=200)
    spec = _spec("Transition", 196, "2010", teams=4, rounds=2, groups=2)
    assert all(is_regular(e, spec) for e in values)
    assert validate_schedule(values, spec) == []
    damaged = deepcopy(values)
    damaged[-1]["homeTeam"]["id"] = 1
    assert validate_schedule(damaged, spec)
    final = deepcopy(values[0]); final["tournament"]["name"] = "J1 League, Championship stage"
    assert not is_regular(final, spec)
    final["tournament"]["name"] = "Placement matches"
    assert not is_regular(final, spec)
    assert not is_regular(values[0], _spec("Normal", 196, "2010", teams=4, rounds=2))


@pytest.mark.parametrize("unknown_stage", [False, True])
def test_j1_transition_captured_round_inventory_excludes_placement(tmp_path, unknown_stage):
    spec = next(s for s in requested_archives(japan_transition="skip")["japan-2010-2027"]
                if s["name"] == "J1_Transition")
    saved = prepare(tmp_path, [spec])
    payload = json.loads((FIXTURES / "j1_transition_2026_rounds.json").read_text(encoding="utf-8"))
    if unknown_stage:
        payload["rounds"].append({"round": 53, "name": "Unreviewed special stage"})
    with open_discovery(saved["directory"]) as discovery:
        observe(discovery, discovery.next_job(resume=True), {"seasons": [{"id": 87931, "year": "2026"}]})
        job = discovery.next_job()
        if unknown_stage:
            with pytest.raises(ValueError, match="Unrecognized named round"):
                observe(discovery, job, payload)
            assert discovery.next_job()["paused"]
            assert not discovery.status()["ready_to_collect"]
        else:
            observe(discovery, job, payload)
            planned = [r[0] for r in discovery.db.execute(
                "SELECT round FROM jobs WHERE round IS NOT NULL ORDER BY round")]
            assert planned == list(range(1, 19))
        retained = discovery.db.execute("SELECT body FROM responses WHERE path=?", (job["path"],)).fetchone()
        assert json.loads(retained[0]) == payload


@pytest.mark.parametrize("name", ["Match for 3rd place", "Match for 5th place"])
@pytest.mark.parametrize("field", ["name", "slug", "prefix"])
def test_named_placement_fixture_cannot_enter_a_regular_round(name, field):
    spec = _spec("Transition", 196, "2026", teams=4, rounds=2, groups=2)
    event = events([1, 2], year="2026", group=True)[0]
    assert is_regular(event, spec)
    event["roundInfo"][field] = name.lower().replace(" ", "-") if field == "slug" else name
    assert not is_regular(event, spec)


def test_captured_j1_2016_validates_both_stages_and_retains_source_rounds(tmp_path):
    from scraping.sofascore.event_round import match_round
    values = json.loads((FIXTURES / "j1_2016_schedule.json").read_text(encoding="utf-8"))["events"]
    before = deepcopy(values)
    spec = next(s for s in requested_archives(japan_transition="skip")["japan-2010-2027"]
                if s["name"] == "J1" and s["season"] == "2016")
    saved = prepare(tmp_path, [spec])
    with open_discovery(saved["directory"]) as discovery:
        drain(discovery, values)
        status = discovery.status()
        assert status["ready_to_collect"]
        season = status["seasons"][0]
        assert season["selected_matches"] == 306
        assert season["superseded_fixtures"] == [
            dict(event_id=7003240, replacement_event_id=7076718, status="cancelled"),
            dict(event_id=7003252, replacement_event_id=7071662, status="cancelled")]
        records, _ = discovery._records(spec, discovery._resolved(0))
        selected = [json.loads(r["payload"]) for r in records]
        assert validate_schedule(selected, spec) == []
        assert validate_schedule(selected[:-1], spec)
        assert validate_schedule(selected + [selected[-1]], spec)
        altered = deepcopy(selected)
        altered[-1]["startTimestamp"] = int(datetime(2016, 12, 1, tzinfo=timezone.utc).timestamp())
        assert validate_schedule(altered, spec)
        altered = deepcopy(selected)
        altered[-1]["season"]["id"] += 1
        assert validate_schedule(altered, spec)
        assert validate_schedule(selected, {**spec, "season": "2017"})
        campaign = discovery.prepare_collection(campaign_base=tmp_path / "campaigns")
        assert campaign["matches"] == 306
    assert values == before
    second = next(e for e in values if e["startTimestamp"] >= 1467417600 and e["roundInfo"]["round"] == 1)
    assert match_round(second) == 1


@pytest.mark.parametrize("identity,expected", [
    ((7071662, 196, 11307, "2016", 1, 21904, 3146, 1464861600), 7),
    ((9117396, 196, 27042, "2020", 25, 3137, 3134, 1604376000), 31),
])
def test_japan_reviewed_rounds_require_exact_identity_and_preserve_raw(identity, expected):
    from scraping.sofascore.event_round import match_round
    from scraping.sofascore.parsers import parse_match
    def event(identity):
        eid, cid, sid, year, number, home, away, kickoff = identity
        e = deepcopy(fixture("event")["event"])
        e.update(id=eid, season={"id": sid, "year": year}, roundInfo={"round": number},
                 startTimestamp=kickoff)
        e["tournament"].update(uniqueTournament={"id": cid}, isGroup=False)
        e["homeTeam"]["id"], e["awayTeam"]["id"] = home, away
        return e
    e = event(identity)
    before = deepcopy(e)
    assert match_round(e) == expected and parse_match(e)["round"] == expected
    assert e == before
    for i in range(len(identity)):
        changed = list(identity)
        changed[i] = "2022" if i == 3 else changed[i] + 1
        other = event(changed)
        assert match_round(other) == other["roundInfo"]["round"]


def test_tokushima_yamagata_stale_status_requires_exact_verified_identity():
    from scraping.sofascore.discovery import event_status, is_completed_event
    from scraping.sofascore.parsers import parse_match
    e = deepcopy(fixture("event")["event"])
    e.update(id=3506406, season={"id": 5918, "year": "2013"}, roundInfo={"round": 33},
             startTimestamp=1379325600, status={"type": "postponed", "code": 60},
             homeScore={"current": 2}, awayScore={"current": 2})
    e["tournament"].update(uniqueTournament={"id": 402}, isGroup=False)
    e["homeTeam"]["id"], e["awayTeam"]["id"] = 21907, 21902
    before = deepcopy(e)
    assert is_completed_event(e) and event_status(e) == "finished"
    parsed = parse_match(e)
    assert parsed["status"] == "finished" and parsed["status_code"] == 60 and e == before
    for path in ("id", "season.id", "season.year", "roundInfo.round", "startTimestamp",
                 "tournament.uniqueTournament.id", "homeTeam.id", "awayTeam.id",
                 "status.type", "status.code", "homeScore.current", "awayScore.current"):
        changed = deepcopy(e)
        target = changed
        parts = path.split(".")
        for key in parts[:-1]:
            target = target[key]
        key = parts[-1]
        target[key] = target[key] + 1 if isinstance(target[key], int) else "unknown"
        assert not is_completed_event(changed)


def test_odd_team_byes_are_not_missing_matches():
    spec = _spec("J2", 402, "2010", teams=19, rounds=38)
    values = events(range(1, 20), cid=402)
    assert len(values) == 342 and validate_schedule(values, spec) == []
    assert validate_schedule(values[:-1], spec)


def test_clausura_2020_accepts_only_verified_shortened_played_schedule():
    spec = next(s for s in requested_archives()["mexico-2010-2026"] if s.get("curtailed"))
    values = [e for e in events(range(1, 19), year="19/20", cid=11620, meetings=1) if e["roundInfo"]["round"] <= 10]
    assert len(values) == 90 and validate_schedule(values, spec) == []
    assert validate_schedule(values[:-1], spec)


def test_current_snapshot_reconciles_played_counts_and_queues_finished_only(tmp_path):
    spec = _spec("J1", 196, "2026/2027", teams=2, rounds=2, snapshot=True)
    values = events([1, 2], year="26/27")
    values[-1]["status"] = {"type": "notstarted"}
    saved = prepare(tmp_path, [spec])
    with open_discovery(saved["directory"]) as discovery:
        drain(discovery, values)
        assert discovery.status()["ready_to_collect"]
        result = discovery.prepare_collection(campaign_base=tmp_path / "campaigns")
        assert result["matches"] == 1
    assert validate_schedule(values, spec, {1: 2, 2: 2})


def test_calendar_capture_and_publication_preserve_provider_year(tmp_path):
    values = events([42, 38])
    saved = prepare(tmp_path, [_spec("J1", 196, "2010", teams=2, rounds=2)])
    with open_discovery(saved["directory"]) as discovery:
        drain(discovery, values)
        prepared = discovery.prepare_collection(campaign_base=tmp_path / "campaigns")
    with Campaign(prepared["directory"]) as campaign:
        for event in values:
            eid = event["id"]
            job = campaign.next_match()
            responses = [("", {"event": event}), ("/pregame-form", {"homeTeam": {"position": 1}, "awayTeam": {"position": 2}}), ("/statistics", {"statistics": []}),
                         ("/shotmap", fixture("shotmap")), ("/lineups", fixture("lineups")),
                         ("/incidents", fixture("incidents"))]
            responses.extend((f"/heatmap/{event[s]['id']}", {"playerPoints": [{"x": 1, "y": 2}], "goalkeeperPoints": []})
                             for s in ("homeTeam", "awayTeam"))
            for suffix, payload in responses:
                campaign.observe(eid, job["lease"], {
                    "startedDateTime": datetime.now(timezone.utc).isoformat(), "time": 10,
                    "request": {"method": "GET", "url": f"https://www.sofascore.com/api/v1/event/{eid}{suffix}"},
                    "response": {"status": 200, "content": {"text": json.dumps(payload)}}})
            assert campaign.finish(eid, job["lease"], page_event_id=None,
                page_evidence={"collection_mode": "session", "session_request_count": len(responses)})["state"] == "complete"
    published = publish_captured_season(prepared["directory"], "J1_10_10", output=tmp_path / "output")
    assert published["matches"] == 2
    tables = read_tables(tmp_path / "output" / published["manifest"])
    assert {r["season_year"] for r in tables["matches"]} == {"2010"}
    assert published["reconciliation"]["archive_cohort"]["schedule_verified"]


@pytest.mark.parametrize("index", [0, 1])
def test_recovered_j2_2018_metadata_retains_raw_and_validates_capture(tmp_path, index):
    from scraping.sofascore.event_round import match_round, match_season, reviewed_j2_2018_metadata
    from scraping.sofascore.parsers import parse_match
    from scraping.sofascore_web_v3.campaign import prepare_campaign
    from scraping.sofascore_web_v3.import_har import import_har, _check_event
    entry = json.loads((FIXTURES / "j2_2018_recovered_metadata.json").read_text(encoding="utf-8"))[index]
    event = json.loads(entry["response"]["content"]["text"])["event"]
    before = deepcopy(event)
    assert "season" not in event and "roundInfo" not in event
    assert match_round(event) == 39 and match_season(event) == {"id": 16005, "year": "2018"}
    assert parse_match(event)["season_id"] == 16005
    assert parse_match(event)["round"] == 39
    reference = {"event": event, "observed_at": 1791416400, "source": "original Chrome HAR"}
    prepared = prepare_campaign([event["id"]], references={event["id"]: reference}, base=tmp_path / "campaigns")
    with Campaign(prepared["directory"]) as campaign:
        campaign.check_event(event["id"], event)
        assert campaign.items[event["id"]]["season_id"] == 16005
    har = tmp_path / "original.har"
    har.write_text(json.dumps({"log": {"entries": [entry]}}), encoding="utf-8")
    result = import_har(har, event_id=event["id"], output_root=tmp_path / "imports")
    assert result["event_metadata_source"] == "captured_network_export"
    match = read_tables(result["manifest"])["matches"][0]
    assert (match["event_id"], match["season_id"], match["season_year"], match["round"]) == (event["id"], 16005, "2018", 39)
    assert json.loads((Path(result["directory"]) / "reference.json").read_text(encoding="utf-8"))["event"] == before
    assert event == before
    for path in ("id", "homeTeam.id", "awayTeam.id", "homeScore.current", "awayScore.current",
                 "tournament.id", "tournament.uniqueTournament.id", "tournament.category.sport.id",
                 "tournament.isGroup", "startTimestamp", "status.type", "status.code"):
        changed = deepcopy(event)
        target = changed
        keys = path.split(".")
        for key in keys[:-1]:
            target = target[key]
        key = keys[-1]
        target[key] = not target[key] if type(target[key]) is bool else target[key] + 1 if type(target[key]) is int else "unknown"
        assert not reviewed_j2_2018_metadata(changed)
        assert match_season(changed) == {} and match_round(changed) is None
        with pytest.raises(ValueError):
            _check_event(changed, event)
    for key, value in (("season", {"id": 999, "year": "2018"}), ("roundInfo", {"round": 1})):
        changed = {**event, key: value}
        assert not reviewed_j2_2018_metadata(changed)
        with pytest.raises(ValueError):
            _check_event(changed, event)
