"""Offline acceptance checks for the requested historical extension setup."""
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import sqlite3

import pytest

from scraping.sofascore_web_v3 import historical_batch
from scraping.sofascore_web_v3.campaign import Campaign, read_json
from scraping.sofascore_web_v3.discovery import Discovery, prepare_discovery, regular_stage
from scraping.sofascore_web_v3.historical_formats import curtailed_rule
from scraping.sofascore.event_round import SERIE_B_2015_ROUND_FOUR, match_round
from scraping.sofascore.parsers import parse_match

ROOT = Path(__file__).resolve().parents[2]
REQUESTS = read_json(ROOT / "scraping/historical_expansion_2015_2020.json")
FIXTURE = read_json(ROOT / "tests/fixtures/sofascore/12436870_event.json")["event"]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: pytest.fail("No provider requests allowed"))


def prepare(tmp_path, requests=REQUESTS, **kwargs):
    return historical_batch.prepare_historical_batch(requests, name="historical-test",
        output=tmp_path / "output", base=tmp_path / "batches", discovery_base=tmp_path / "discovery", **kwargs)


def test_requested_21_seasons_prepare_once_and_resume_identical_ids(tmp_path):
    first = prepare(tmp_path)
    assert [len(r["stems"]) for r in first["runs"]] == [7, 6, 3, 3, 2]
    expected = {f"{r['name']}_{s.replace('/', '_')}" for r in REQUESTS for s in r["seasons"]}
    assert {stem for r in first["runs"] for stem in r["stems"]} == expected
    assert len(expected) == 21
    original = Path(first["path"]).read_bytes()
    second = prepare(tmp_path)
    assert first == second and Path(first["path"]).read_bytes() == original
    assert len(list((tmp_path / "discovery").iterdir())) == 5
    for run in first["runs"]:
        config = read_json(tmp_path / "discovery" / run["discovery_id"] / "config.json")
        assert config["mode"] == "season" and config["output_root"] == str((tmp_path / "output").resolve())
        assert all(config["profile"][key] for key in ("include_lineups", "include_shotmap", "include_heatmaps"))
        assert not config["profile"]["include_comments"]
        assert config["policy"]["collection_mode"] == "session"
        assert config["policy"]["random_action_delay_max_seconds"] is None


def test_resume_rejects_changed_settings_without_creating_more_ids(tmp_path):
    batch = prepare(tmp_path)
    original = Path(batch["path"]).read_bytes()
    with pytest.raises(ValueError, match="settings differ"):
        prepare(tmp_path, include_lineups=False)
    assert len(list((tmp_path / "discovery").iterdir())) == 5
    assert Path(batch["path"]).read_bytes() == original


def test_existing_output_is_not_recollected_or_overwritten(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    existing = output / "Eredivisie_16_17.parquet"
    existing.write_bytes(b"preserved")
    with pytest.raises(ValueError, match="Existing output for Eredivisie_16_17"):
        prepare(tmp_path)
    assert existing.read_bytes() == b"preserved" and not (tmp_path / "discovery").exists()


def test_semantically_duplicate_season_is_rejected(tmp_path):
    requests = deepcopy(REQUESTS)
    requests[0]["seasons"] = ["15/16", "2015/2016"]
    with pytest.raises(ValueError, match="Duplicate league/season"):
        prepare(tmp_path, requests)
    assert not (tmp_path / "discovery").exists()


def attach_campaign(batch, run, tmp_path):
    directory = Path(batch["settings"]["discovery_base"]) / run["discovery_id"]
    campaign = {"campaign_id": f"capture-{run['season'].replace('/', '-')}",
                "directory": str(tmp_path / run["discovery_id"])}
    with sqlite3.connect(directory / "discovery.sqlite") as db:
        db.execute("INSERT OR REPLACE INTO state VALUES('campaign',?)", (json.dumps(campaign),))
    return campaign


def test_watcher_waits_for_chrome_then_converts_and_advances(tmp_path, monkeypatch):
    batch = prepare(tmp_path)
    calls, waits = [], []

    def wait(seconds):
        index = len(waits)
        waits.append(seconds)
        attach_campaign(batch, batch["runs"][index], tmp_path)

    def publish(directory, **kwargs):
        run = batch["runs"][len(calls)]
        assert directory == tmp_path / run["discovery_id"]
        assert kwargs["output"] == tmp_path / "output" and kwargs["watch"] is True
        calls.append(directory)
        return {"state": "finished", "verified_in_this_run": dict.fromkeys(run["stems"], {})}

    monkeypatch.setattr(historical_batch, "watch_publications", publish)
    result = historical_batch.watch_historical_batch(batch, sleep=wait, emit=lambda _: None)
    assert len(result) == len(calls) == len(waits) == 5


@pytest.mark.parametrize("state", ["stopped", "finished_with_gaps"])
def test_interrupted_converter_does_not_advance_to_another_year(tmp_path, monkeypatch, state):
    batch = prepare(tmp_path)
    attach_campaign(batch, batch["runs"][0], tmp_path)
    calls = []

    def publish(directory, **kwargs):
        calls.append(directory)
        return {"state": state, "verified_in_this_run": {}}

    monkeypatch.setattr(historical_batch, "watch_publications", publish)
    result = historical_batch.watch_historical_batch(batch, emit=lambda _: None,
        sleep=lambda _: pytest.fail("Must stop with the converter"))
    assert len(calls) == len(result) == 1


def test_once_checks_without_starting_capture_or_waiting(tmp_path):
    batch = prepare(tmp_path)
    assert historical_batch.watch_historical_batch(batch, once=True, emit=lambda _: None,
        sleep=lambda _: pytest.fail("Status check must not wait")) == []
    assert len(list((tmp_path / "discovery").iterdir())) == 5


@pytest.mark.parametrize("season", ["15/16", "16/17", "17/18", "18/19", "19/20"])
def test_old_scottish_unlabelled_post_split_rounds_are_not_queued(tmp_path, season):
    config = prepare_discovery(base=tmp_path / "discovery", season=season,
        leagues=[{"name": "Premiership", "competition_id": 36, "meetings_per_pair": 3}])
    with Discovery(config["directory"]) as discovery:
        with discovery.db:
            discovery.process({"competition": 36, "season": 900, "kind": "rounds"},
                {"rounds": [{"round": n} for n in range(39)]}, 0)
        numbers = [r[0] for r in discovery.db.execute("SELECT round FROM jobs WHERE kind='events' ORDER BY round")]
        assert numbers == list(range(1, 34))
    e = {"season": {"year": season}, "roundInfo": {"round": 33},
         "tournament": {"name": "Scottish Premiership", "isGroup": False, "uniqueTournament": {"id": 36}}}
    assert regular_stage(e)
    e["roundInfo"]["round"] = 34
    assert not regular_stage(e)
    e["tournament"]["uniqueTournament"]["id"] = 18
    assert regular_stage(e)


@pytest.mark.parametrize("event_id", list(SERIE_B_2015_ROUND_FOUR))
def test_verified_mislabelled_round_four_fixtures_keep_raw_stage(event_id):
    home, away, kickoff = SERIE_B_2015_ROUND_FOUR[event_id]
    event = {"id": event_id, "tournament": {"uniqueTournament": {"id": 53}},
             "season": {"id": 10753, "year": "15/16"}, "roundInfo": {"round": 0},
             "homeTeam": {"id": home}, "awayTeam": {"id": away}, "startTimestamp": kickoff}
    before = deepcopy(event)
    row = parse_match(event)
    assert row["round"] == 4 and json.loads(row["stage_json"])["round"] == 0
    assert event == before
    for change in ({"id": 999}, {"season": {"id": 10753, "year": "16/17"}},
                   {"tournament": {"uniqueTournament": {"id": 36}}},
                   {"homeTeam": {"id": away}}, {"startTimestamp": kickoff + 1}):
        assert match_round({**event, **change}) == 0
    assert match_round({**event, "roundInfo": {"round": 7}}) == 7


def test_verified_lecce_ascoli_missing_round_keeps_provider_payload():
    event = {"id": 8168679, "tournament": {"uniqueTournament": {"id": 53}, "isGroup": False},
             "season": {"id": 18325, "year": "18/19"}, "roundInfo": None,
             "homeTeam": {"id": 2689}, "awayTeam": {"id": 2707}, "startTimestamp": 1553360400}
    before = deepcopy(event)
    row = parse_match(event)
    assert row["round"] == 22 and row["stage_json"] is None
    assert regular_stage(event) and event == before
    for change in ({"id": 999}, {"season": {"id": 18325, "year": "19/20"}},
                   {"tournament": {"uniqueTournament": {"id": 36}}},
                   {"homeTeam": {"id": 2707}}, {"startTimestamp": 1553360401}):
        assert match_round({**event, **change}) is None
    assert match_round({**event, "roundInfo": {"round": 23}}) == 23


def test_verified_cosenza_unplayed_award_is_narrow_and_preserves_raw_status():
    from scraping.sofascore_web_v3.historical_formats import verified_unplayed_award
    event = {"id": 7930529, "tournament": {"uniqueTournament": {"id": 53}},
             "season": {"id": 18325, "year": "18/19"}, "roundInfo": {"round": 2},
             "homeTeam": {"id": 2716}, "awayTeam": {"id": 2701}, "startTimestamp": 1535817600,
             "status": {"type": "postponed"}, "isAwarded": True,
             "homeScore": {"current": 0}, "awayScore": {"current": 3}}
    before = deepcopy(event)
    assert verified_unplayed_award(event) and event == before
    for change in ({"id": 999}, {"isAwarded": False}, {"status": {"type": "finished"}},
                   {"awayScore": {"current": 2}}, {"startTimestamp": 1535817601},
                   {"season": {"id": 18325, "year": "19/20"}}):
        assert not verified_unplayed_award({**event, **change})


def curtailed_fixture(cid):
    rule = curtailed_rule(cid, [2019, 2020])
    rotation = list(range(1, rule["teams"] + 1))
    base_rounds = []
    for _ in range(len(rotation) - 1):
        base_rounds.append(list(zip(rotation[:len(rotation)//2], reversed(rotation[len(rotation)//2:]))))
        rotation = [rotation[0], rotation[-1], *rotation[1:-1]]
    events = defaultdict(list)
    for cycle in range(rule["meetings"]):
        for index, pairs in enumerate(base_rounds):
            number = cycle * len(base_rounds) + index + 1
            for slot, (home, away) in enumerate(pairs):
                if cycle % 2:
                    home, away = away, home
                e = deepcopy(FIXTURE)
                e.update(id=cid * 100000 + number * 10 + slot, season={"id": cid * 100, "year": "19/20"},
                    roundInfo={"round": number}, startTimestamp=1564617600 + number * 86400)
                e["tournament"] = {"id": cid, "isGroup": False, "name": "Example League", "uniqueTournament": {"id": cid}}
                e["homeTeam"] = {"id": home, "name": f"Team {home}"}
                e["awayTeam"] = {"id": away, "name": f"Team {away}"}
                played = (number - 1) * (rule["teams"] // 2) + slot < rule["matches"]
                e["status"] = {"type": "finished" if played else "postponed"}
                events[number].append(e)
    appearances = Counter(e[side]["id"] for rows in events.values() for e in rows
                          if e["status"]["type"] == "finished" for side in ("homeTeam", "awayTeam"))
    standings = {"standings": [{"type": "total", "tournament": e["tournament"],
        "rows": [{"team": {"id": t}, "matches": appearances[t]} for t in sorted(appearances)]}]}
    return events, standings, rule


def collect_inventory(tmp_path, cid, mutate=None):
    events, standings, rule = curtailed_fixture(cid)
    if mutate:
        mutate(events, standings)
    config = prepare_discovery(season="19/20", leagues=[{"name": f"League_{cid}",
        "competition_id": cid, "meetings_per_pair": rule["meetings"]}], base=tmp_path / "discovery")
    with Discovery(config["directory"]) as discovery:
        while True:
            job = discovery.next_job()
            if job.get("done"):
                return config, job["summary"]
            if job["path"].endswith("/seasons"):
                data = {"seasons": [{"id": cid * 100, "year": "19/20"}]}
            elif job["path"].endswith("/rounds"):
                data = {"rounds": [{"round": n} for n in events]}
            elif job["path"].endswith("/standings/total"):
                data = standings
            else:
                data = {"events": events[int(job["path"].rsplit("/", 1)[1])]}
            entry = {"startedDateTime": datetime.now(timezone.utc).isoformat(), "time": 1,
                "request": {"method": "GET", "url": "https://www.sofascore.com/api/v1" + job["path"]},
                "response": {"status": 200, "content": {"text": json.dumps(data)}}}
            discovery.observe(job["path"], job["lease"], entry)


@pytest.mark.parametrize("cid", [36, 38])
@pytest.mark.parametrize("empty_unplayed_rounds", [False, True])
def test_officially_curtailed_season_collects_only_played_matches(tmp_path, cid, empty_unplayed_rounds):
    def mutate(events, standings):
        if empty_unplayed_rounds:
            for number, rows in events.items():
                if all(e["status"]["type"] != "finished" for e in rows):
                    events[number] = []
    config, snapshot = collect_inventory(tmp_path, cid, mutate)
    assert snapshot["ready_to_collect"]
    season = snapshot["seasons"][0]
    assert season["schedule_verified"] and season["schedule_validation"] == "curtailed_final_standings"
    assert not season["validation_issues"]
    expected = 179 if cid == 36 else 232
    with Discovery(config["directory"]) as discovery:
        campaign = discovery.prepare_collection(campaign_base=tmp_path / "campaigns")
    with Campaign(campaign["directory"]) as captured:
        assert len(captured.items) == expected
        assert all(i["reference"]["event"]["status"]["type"] == "finished" for i in captured.items.values())


@pytest.mark.parametrize("mutation", ["missing_match", "standing_count", "late_match", "live_match", "extra_match"])
def test_curtailed_season_never_approves_missing_or_conflicting_data(tmp_path, mutation):
    def mutate(events, standings):
        if mutation == "missing_match":
            events[1].pop()
        elif mutation == "standing_count":
            standings["standings"][0]["rows"][0]["matches"] -= 1
        elif mutation == "late_match":
            events[1][0]["startTimestamp"] = 1700000000
        elif mutation == "live_match":
            events[33][0]["status"] = {"type": "inprogress"}
        else:
            extra = deepcopy(events[1][0])
            extra["id"] += 1_000_000
            events[1].append(extra)
    config, snapshot = collect_inventory(tmp_path, 36, mutate)
    assert snapshot["discovery_complete"] and not snapshot["ready_to_collect"]
    assert snapshot["seasons"][0]["validation_issues"]
    with Discovery(config["directory"]) as discovery:
        with pytest.raises(ValueError, match="resolve incomplete"):
            discovery.prepare_collection(campaign_base=tmp_path / "campaigns")
    assert not (tmp_path / "campaigns").exists()


def test_curtailed_standings_without_played_counts_are_rejected(tmp_path):
    def mutate(events, standings):
        del standings["standings"][0]["rows"][0]["matches"]
    with pytest.raises(ValueError, match="played-match count"):
        collect_inventory(tmp_path, 38, mutate)


def test_curtailment_exceptions_do_not_apply_to_other_seasons_or_daily_mode(tmp_path):
    assert curtailed_rule(36, [2018, 2019]) is None
    assert curtailed_rule(37, [2019, 2020]) is None
    config = prepare_discovery(season="19/20", mode="daily", leagues=[REQUESTS[0]], base=tmp_path / "discovery")
    with Discovery(config["directory"]) as discovery:
        assert discovery.curtailed_rule(38) is None
