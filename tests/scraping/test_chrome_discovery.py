from datetime import datetime, timezone
import json
from pathlib import Path
import socket

import pytest

from scraping.sofascore_web_v3.campaign import Campaign
from scraping.sofascore_web_v3.discovery import Discovery, prepare_discovery, round_numbers, regular_stage
from scraping.sofascore_web_v3.native_host import Host

FIXTURE = Path(__file__).parents[1] / "fixtures/sofascore/12436870_event.json"
LEAGUES = [{"name": "Example_League", "competition_id": 17, "meetings_per_pair": 2}]


def event(eid=101, round_number=1, group=False):
    value = json.loads(FIXTURE.read_text())["event"]
    value.update(id=eid, season={"id": 900, "year": "25/26"}, roundInfo={"round": round_number},
                 status={"type": "finished", "description": "Ended"}, startTimestamp=1760000000+round_number)
    value["tournament"] = {"id": 1, "name": "Example League", "slug": "example-league", "isGroup": group,
                            "uniqueTournament": {"id": 17}}
    value["homeTeam"] = {"id": 1, "name": "Home", "slug": "home"}
    value["awayTeam"] = {"id": 2, "name": "Away", "slug": "away"}
    if round_number == 2:
        value["homeTeam"], value["awayTeam"] = value["awayTeam"], value["homeTeam"]
    return value


def entry(job, payload, code=200):
    return {"startedDateTime": datetime.now(timezone.utc).isoformat(), "time": 10,
        "request": {"method": "GET", "url": "https://www.sofascore.com/api/v1" + job["path"],
                    "headers": [{"name": "Cookie", "value": "PRIVATE_SENTINEL"}]},
        "response": {"status": code, "content": {"text": json.dumps(payload)}}}


def payload(job):
    path = job["path"]
    if path.endswith("/seasons"):
        return {"seasons": [{"id": 999, "year": "26/27"}, {"id": 900, "year": "25/26"}]}
    if path.endswith("/rounds"):
        return {"rounds": [{"round": 1}, {"round": 2}, {"round": 3, "name": "Final", "slug": "final"}]}
    if path.endswith("/standings/total"):
        return {"standings": [{"type": "total", "tournament": event()["tournament"],
                                "rows": [{"team": {"id": 1}}, {"team": {"id": 2}}]}]}
    n = int(path.rsplit("/", 1)[1])
    return {"events": [event(100+n, n), event(200+n, n, group=True)]}


def create(tmp_path, **kwargs):
    return prepare_discovery(base=tmp_path / "discovery", leagues=LEAGUES, **kwargs)


def drain(discovery):
    paths = []
    while True:
        job = discovery.next_job(resume=not paths)
        if job.get("done"):
            return paths
        paths.append(job["path"])
        discovery.observe(job["path"], job["lease"], entry(job, payload(job)))


def test_missing_fixture_recovered_from_exact_chrome_metadata(tmp_path):
    config = create(tmp_path)
    with Discovery(config["directory"]) as discovery:
        while True:
            job = discovery.next_job()
            if job.get("done"):
                break
            data = payload(job)
            if job["path"].endswith("/events/round/2"):
                data["events"] = [event(202, 2, group=True)]
            discovery.observe(job["path"], job["lease"], entry(job, data))
        assert not discovery.status()["ready_to_collect"]
        archived_round = discovery.db.execute("SELECT body FROM responses WHERE path LIKE '%/round/2'").fetchone()[0]
        recovered = entry({"path": "/event/102"}, {"event": event(102, 2)})
        result = discovery.admit_recovered_event(recovered, event_id=102, competition_id=17, season_id=900, round_number=2)
        assert result["summary"]["ready_to_collect"]
        assert discovery.db.execute("SELECT body FROM responses WHERE path LIKE '%/round/2'").fetchone()[0] == archived_round
        assert json.loads(discovery.db.execute("SELECT body FROM responses WHERE path='/event/102'").fetchone()[0]) == {"event": event(102, 2)}
        with pytest.raises(ValueError, match="missing event"):
            discovery.admit_recovered_event(recovered, event_id=102, competition_id=17, season_id=900, round_number=2)


@pytest.mark.parametrize("mutation", ["round", "season", "stage", "event", "url"])
def test_recovered_metadata_does_not_bypass_identity_or_scope(tmp_path, mutation):
    config = create(tmp_path)
    with Discovery(config["directory"]) as discovery:
        drain(discovery)
        value = event(103, 2)
        if mutation == "round": value["roundInfo"]["round"] = 1
        if mutation == "season": value["season"]["id"] = 901
        if mutation == "stage": value["tournament"]["isGroup"] = True
        if mutation == "event": value["id"] = 104
        recovered = entry({"path": "/event/103"}, {"event": value})
        if mutation == "url": recovered["request"]["url"] += "?extra=1"
        with pytest.raises(ValueError):
            discovery.admit_recovered_event(recovered, event_id=103, competition_id=17, season_id=900, round_number=2)
        assert discovery.db.execute("SELECT COUNT(*) FROM events WHERE id=103").fetchone()[0] == 0


def test_discovery_resolves_exact_season_and_prepares_existing_campaign_offline(tmp_path, monkeypatch):
    monkeypatch.setattr(socket, "create_connection", lambda *a, **kw: pytest.fail("No Python HTTP allowed"))
    config = create(tmp_path)
    with Discovery(config["directory"]) as discovery:
        paths = drain(discovery)
        assert len(paths) == 5
        assert all("999" not in path for path in paths)
        status = discovery.status()
        assert status["ready_to_collect"]
        assert status["seasons"][0]["regular_matches"] == 2
        assert status["seasons"][0]["excluded_stage_matches"] == 2
        result = discovery.prepare_collection(campaign_base=tmp_path / "campaigns")
        assert result == discovery.prepare_collection(campaign_base=tmp_path / "campaigns")
        with Campaign(result["directory"]) as campaign:
            assert list(campaign.items) == [101, 102]
            assert campaign.profile.include_lineups and campaign.profile.include_shotmap
            assert campaign.profile.include_heatmaps and not campaign.profile.include_comments
            first = campaign.next_match()
            assert first["policy"]["collection_mode"] == "session"
            assert "/event/101/shotmap" in first["paths"]
            assert campaign.items[101]["reference"]["season_stem"] == "Example_League_25_26"
        assert all("PRIVATE_SENTINEL" not in row[0] for row in discovery.db.execute("SELECT body FROM responses"))


def test_resume_reuses_acknowledged_pages_and_releases_stale_lease(tmp_path):
    config = create(tmp_path)
    with Discovery(config["directory"]) as discovery:
        job = discovery.next_job()
        discovery.observe(job["path"], job["lease"], entry(job, payload(job)))
        pending = discovery.next_job()
    with Discovery(config["directory"]) as resumed:
        fresh = resumed.next_job(resume=True)
        assert fresh["path"] == pending["path"] and fresh["lease"] != pending["lease"]
        with pytest.raises(ValueError, match="lease"):
            resumed.observe(pending["path"], pending["lease"], entry(pending, payload(pending)))
        assert not any(p.endswith("/seasons") for p in drain(resumed))
        assert resumed.status()["requests"] == 5


@pytest.mark.parametrize("code", [401, 403, 404, 429, 500])
def test_http_errors_pause_and_never_count_as_complete(tmp_path, code):
    config = create(tmp_path)
    with Discovery(config["directory"]) as discovery:
        job = discovery.next_job()
        with pytest.raises(ValueError, match=f"HTTP {code}"):
            discovery.observe(job["path"], job["lease"], entry(job, {"error": "blocked"}, code))
        assert discovery.next_job()["paused"]
        assert not discovery.status()["ready_to_collect"]
        assert discovery.db.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 0
        retry = discovery.next_job(resume=True)
        assert retry["path"] == job["path"]


@pytest.mark.parametrize("years", [[], [{"id": 999, "year": "26/27"}], [{"id": 900, "year": "25/26"}, {"id": 901, "year": "2025/2026"}]])
def test_missing_or_ambiguous_season_never_selects_latest(tmp_path, years):
    config = create(tmp_path)
    with Discovery(config["directory"]) as discovery:
        job = discovery.next_job()
        with pytest.raises(ValueError, match="missing or ambiguous"):
            discovery.observe(job["path"], job["lease"], entry(job, {"seasons": years}))


@pytest.mark.parametrize("mutation", ["season", "competition", "round", "team", "incomplete", "duplicate"])
def test_conflicting_fixture_or_truncated_round_rolls_back_whole_page(tmp_path, mutation):
    config = create(tmp_path)
    with Discovery(config["directory"]) as discovery:
        while True:
            job = discovery.next_job()
            data = payload(job)
            if "/events/round/" in job["path"]: break
            discovery.observe(job["path"], job["lease"], entry(job, data))
        bad = data["events"][1]
        if mutation == "season": bad["season"]["id"] = 901
        if mutation == "competition": bad["tournament"]["uniqueTournament"]["id"] = 18
        if mutation == "round": bad["roundInfo"]["round"] = 99
        if mutation == "team": bad["awayTeam"]["id"] = bad["homeTeam"]["id"]
        if mutation == "incomplete": data["hasNextPage"] = True
        if mutation == "duplicate": bad["id"] = data["events"][0]["id"]
        with pytest.raises(ValueError): discovery.observe(job["path"], job["lease"], entry(job, data))
        assert discovery.db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0


@pytest.mark.parametrize("issue", ["missing_team", "missing_match", "not_finished", "wrong_home_away"])
def test_complete_enumeration_with_bad_schedule_cannot_create_collection(tmp_path, issue):
    config = create(tmp_path)
    with Discovery(config["directory"]) as discovery:
        while True:
            job = discovery.next_job()
            if job.get("done"): break
            data = payload(job)
            if issue == "missing_team" and "standings/total" in job["path"]:
                data["standings"][0]["rows"].append({"team": {"id": 3}})
            if "/events/round/2" in job["path"]:
                if issue == "missing_match": data["events"] = [event(202, 2, group=True)]
                if issue == "not_finished": data["events"][0]["status"] = {"type": "notstarted"}
                if issue == "wrong_home_away":
                    e = data["events"][0]
                    e["homeTeam"], e["awayTeam"] = e["awayTeam"], e["homeTeam"]
            discovery.observe(job["path"], job["lease"], entry(job, data))
        assert discovery.status()["discovery_complete"]
        assert not discovery.status()["ready_to_collect"]
        with pytest.raises(ValueError, match="incomplete schedules"):
            discovery.prepare_collection(campaign_base=tmp_path / "campaigns")
        assert not (tmp_path / "campaigns").exists()


def test_native_discovery_and_collection_cannot_run_together(tmp_path):
    host = Host(base=tmp_path / "campaigns", discovery_base=tmp_path / "discovery", lock_path=tmp_path / "lock")
    try:
        assert len(host.dispatch({"operation": "discovery_catalog"})["leagues"]) == 13
        result = host.dispatch({"operation": "discovery_create", "season": "25/26", "leagues": LEAGUES})
        with pytest.raises(ValueError, match="already attached"):
            host.dispatch({"operation": "open", "campaign_id": "old-campaign"})
        job = host.dispatch({"operation": "discovery_next"})
        host.dispatch({"operation": "discovery_observe", "path": job["path"], "lease": job["lease"], "entry": entry(job, payload(job))})
        assert host.dispatch({"operation": "discovery_status"})["requests"] == 1
        assert result["api_requests_sent_by_python"] == 0
    finally: host.close()


def test_named_stages_and_group_events_excluded_but_championship_league_allowed():
    assert round_numbers({"rounds": [{"round": 1}, {"round": 1}, {"round": 1, "name": "Playoffs"}]}) == [1]
    e = event(); e["tournament"]["name"] = "Championship"
    assert regular_stage(e)
    e["roundInfo"]["name"] = "Promotion Playoffs"
    assert not regular_stage(e)
    with pytest.raises(ValueError, match="stage flag"):
        regular_stage({"tournament": {"name": "Unknown format"}})


def test_dutch_provider_round_prefix_excludes_playoffs_without_dropping_regular_round_one():
    rounds = [{"round": 1}, {"round": 1, "name": "Round 1", "slug": "round-1", "prefix": "Relegation-Promotion"},
              {"round": 34}, {"round": 29, "name": "Final", "slug": "final", "prefix": "Europa Playoffs"}]
    assert round_numbers({"rounds": rounds}) == [1, 34]
    assert round_numbers({"rounds": [{"round": 1, "name": "Round 1", "slug": "round-1"}]}) == [1]
    e = event(); e["roundInfo"].update(name="Round 1", slug="round-1", prefix="Relegation-Promotion")
    assert not regular_stage(e)


def test_historical_round_zero_is_not_requested_and_named_finals_stay_excluded(tmp_path):
    config = create(tmp_path)
    with Discovery(config["directory"]) as discovery:
        paths = []
        while True:
            job = discovery.next_job()
            if job.get("done"):
                break
            paths.append(job["path"])
            data = payload(job)
            if job["path"].endswith("/rounds"):
                data["rounds"] = [{"round": 0}, {"round": 1}, {"round": 2},
                    {"round": 3, "name": "Quarterfinals", "slug": "quarterfinals"},
                    {"round": 2, "name": "Semifinals", "slug": "semifinals"},
                    {"round": 1, "name": "Final", "slug": "final"}]
            discovery.observe(job["path"], job["lease"], entry(job, data))
        assert [p.rsplit("/", 1)[-1] for p in paths if "/events/round/" in p] == ["1", "2"]
        assert discovery.status()["ready_to_collect"]
    with pytest.raises(ValueError, match="No regular numbered rounds"):
        round_numbers({"rounds": [{"round": 0}, {"round": 1, "name": "Final"}]})


def test_parser_repair_can_revalidate_saved_success_without_new_provider_attempt(tmp_path, monkeypatch):
    config = create(tmp_path)
    with Discovery(config["directory"]) as discovery:
        job = discovery.next_job()
        original = Discovery.process
        def broken(*args): raise ValueError("unsupported format")
        monkeypatch.setattr(Discovery, "process", broken)
        with pytest.raises(ValueError): discovery.observe(job["path"], job["lease"], entry(job, payload(job)))
        monkeypatch.setattr(Discovery, "process", original)
        assert discovery.revalidate_saved_failures() == [job["path"]]
        assert discovery.status()["requests"] == 1
        assert discovery.next_job(resume=True)["path"].endswith("/rounds")


def test_round_inspection_preserves_evidence_without_adding_fixtures(tmp_path):
    config = create(tmp_path)
    with Discovery(config["directory"]) as discovery:
        drain(discovery)
        before = discovery.status()
        with discovery.db:
            discovery.add_job("/unique-tournament/17/season/900/events/round/0", 17, "round_inspection", 900, 0)
        job = discovery.next_job()
        assert job["inspection_only"] is True
        discovery.observe(job["path"], job["lease"], entry(job, {"events": [event(999, 0)]}))
        assert discovery.db.execute("SELECT 1 FROM responses WHERE path=?", (job["path"],)).fetchone()
        assert not discovery.db.execute("SELECT 1 FROM events WHERE id=999").fetchone()
        after = discovery.status()
        assert after["seasons"] == before["seasons"]
        assert after["requests"] == before["requests"] + 1


def test_discovery_controls_and_input_validation(tmp_path):
    result = create(tmp_path, include_lineups=False, random_delay_max_seconds=2.5)
    config = json.loads((Path(result["directory"]) / "config.json").read_text())
    assert not config["profile"]["include_lineups"]
    assert config["policy"]["random_action_delay_max_seconds"] == 2.5
    for leagues in ([LEAGUES[0], LEAGUES[0]], [{"name": "../escape", "competition_id": 17}], []):
        with pytest.raises(ValueError): prepare_discovery(leagues=leagues, base=tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_rearranged_rounds_are_queued_by_actual_kickoff_and_preview_is_read_only(tmp_path):
    config = create(tmp_path)
    with Discovery(config["directory"]) as discovery:
        while True:
            job = discovery.next_job()
            if job.get("done"): break
            data = payload(job)
            if job["path"].endswith("/events/round/1"):
                data["events"][0]["startTimestamp"] += 10000
            discovery.observe(job["path"], job["lease"], entry(job, data))
        before = discovery.status()
        result = discovery.prepare_collection(campaign_base=tmp_path / "campaigns")
    with Campaign(result["directory"]) as campaign:
        assert list(campaign.items) == [102, 101]
    with Discovery(config["directory"], read_only=True) as preview:
        assert preview.status()["requests"] == before["requests"]
        with pytest.raises(Exception, match="readonly"):
            preview.db.execute("DELETE FROM events")


@pytest.mark.parametrize("case", ["replacement", "abandoned", "two_finished", "wrong_direction", "earlier_finish", "cancelled"])
def test_fixture_replacement_requires_unique_later_match_and_retains_evidence(tmp_path, case):
    config = create(tmp_path)
    with Discovery(config["directory"]) as discovery:
        while True:
            job = discovery.next_job()
            if job.get("done"): break
            data = payload(job)
            if job["path"].endswith("/events/round/1"):
                old = event(301, 1)
                old["startTimestamp"] -= 1000
                old["status"] = {"type": "postponed"}
                if case == "abandoned": old["status"] = {"type": "canceled", "description": "Abandoned", "code": 90}
                if case == "two_finished": old["status"] = {"type": "finished"}
                if case == "wrong_direction": old["homeTeam"], old["awayTeam"] = old["awayTeam"], old["homeTeam"]
                if case == "earlier_finish": old["startTimestamp"] += 2000
                if case == "cancelled": old["status"] = {"type": "canceled"}
                data["events"].append(old)
            discovery.observe(job["path"], job["lease"], entry(job, data))
        status = discovery.status()
        assert discovery.db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 5
        assert json.loads(discovery.db.execute("SELECT payload FROM events WHERE id=301").fetchone()[0]) == old
        if case in {"replacement", "abandoned", "cancelled"}:
            assert status["ready_to_collect"]
            assert status["seasons"][0]["superseded_fixtures"] == [{"event_id": 301, "replacement_event_id": 101,
                "status": case if case in {"abandoned", "cancelled"} else "postponed"}]
            result = discovery.prepare_collection(campaign_base=tmp_path / "campaigns")
            with Campaign(result["directory"]) as campaign:
                assert list(campaign.items) == [101, 102]
        else:
            assert not status["ready_to_collect"]
            with pytest.raises(ValueError, match="incomplete schedules"):
                discovery.prepare_collection(campaign_base=tmp_path / "campaigns")


@pytest.mark.parametrize("terminal_status", ["abandoned", "cancelled"])
def test_unreplaced_terminal_match_remains_in_inventory_without_blocking_finished_collection(tmp_path, terminal_status):
    config = create(tmp_path)
    with Discovery(config["directory"]) as discovery:
        while True:
            job = discovery.next_job()
            if job.get("done"): break
            data = payload(job)
            if job["path"].endswith("/events/round/2"):
                data["events"][0]["status"] = {"type": "canceled", "description": "Abandoned", "code": 90} if terminal_status == "abandoned" else {"type": "canceled", "description": "Canceled", "code": 70}
            discovery.observe(job["path"], job["lease"], entry(job, data))
        status = discovery.status()
        assert status["ready_to_collect"]
        season = status["seasons"][0]
        assert season["regular_matches"] == 2 and season["finished"] == 1
        assert season["non_collectable_terminal_matches"] == [{"event_id": 102, "status": terminal_status}]
        result = discovery.prepare_collection(campaign_base=tmp_path / "campaigns")
        with Campaign(result["directory"]) as campaign:
            assert list(campaign.items) == [101]
        assert discovery.db.execute("SELECT 1 FROM events WHERE id=102").fetchone()
