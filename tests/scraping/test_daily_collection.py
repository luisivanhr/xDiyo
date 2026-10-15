"""Offline daily discovery, update, publication and restart acceptance tests."""
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import time

import pytest

from scraping.sofascore.config import Scope
from scraping.sofascore.campaign import publish_season
from scraping.sofascore.export import read_tables
from scraping.sofascore.parsers import parse_match
from scraping.sofascore.schema import to_arrow
from scraping.sofascore_web_v3.campaign import Campaign, prepare_campaign
from scraping.sofascore_web_v3.capture import PROFILE
from scraping.sofascore_web_v3.daily import prepare_daily_run, daily_status, watch_daily_run
from scraping.sofascore_web_v3.daily_discovery import DailyDiscovery
from scraping.sofascore_web_v3.daily_storage import (merge_capture, publish_daily_campaign,
    read_daily_tables, source_snapshot)
from scraping.sofascore_web_v3.discovery import prepare_discovery, open_discovery
from scraping.sofascore_web_v3.native_host import Host
from scraping.sofascore_web_v3.publish import hash_file, snapshot_season, completed_season_stems

FIXTURES = Path(__file__).parents[1] / "fixtures/sofascore"
NOW = int(time.time())
LEAGUE = {"name": "Example_League", "competition_id": 17, "meetings_per_pair": 1}
PAIRS = {1: [(1, 2), (3, 4)], 2: [(1, 3), (2, 4)], 3: [(1, 4), (2, 3)]}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: pytest.fail("Python sent a network request"))


def event(eid=101, number=1, status="finished", teams=(1, 2), cid=17, kickoff=None):
    value = json.loads((FIXTURES / "12436870_event.json").read_text())["event"]
    value.update(id=eid, season={"id": cid * 100, "year": "26/27"}, roundInfo={"round": number},
                 status={"type": status}, startTimestamp=int(kickoff or (NOW + (3600 if status == "notstarted" else -86400))))
    value["tournament"] = {"id": cid, "name": "Example League", "slug": "example-league", "isGroup": False,
                           "uniqueTournament": {"id": cid}}
    for side, tid in zip(("homeTeam", "awayTeam"), teams):
        value[side] = {"id": tid, "name": f"Team {tid}", "slug": f"team-{tid}"}
    if status != "finished":
        value["homeScore"] = value["awayScore"] = {}
    return value


def entry(path, payload=None, code=200, observed=None):
    return {"startedDateTime": datetime.fromtimestamp(observed or time.time(), timezone.utc).isoformat(), "time": 0,
        "request": {"method": "GET", "url": "https://www.sofascore.com/api/v1" + path},
        "response": {"status": code, "content": {"text": json.dumps(payload)}},
        "_observationSource": "chrome_session_request"}


def new_discovery(tmp_path, **options):
    return prepare_discovery(season="26/27", leagues=[LEAGUE], mode="daily", base=tmp_path / "discovery",
                             history_root=tmp_path / "history", output_root=tmp_path / "daily", **options)


def finish_discovery(discovery, *, statuses=None, anchor=2, ended=(), absent=(), change=None):
    paths = []
    while True:
        job = discovery.next_job(resume=True)
        if job.get("done"):
            return paths
        path, cid = job["path"], job["competition_id"]
        paths.append(path)
        if path.endswith("/seasons"):
            payload = {"seasons": [] if cid in absent else [{"id": cid * 100, "year": "26/27"}]}
        elif path.endswith("/rounds"):
            payload = {"currentRound": {"round": anchor}, "rounds": [{"round": n} for n in (1, 2, 3)]}
        elif path.endswith("/standings/total"):
            payload = {"standings": [{"type": "total", "tournament": event(cid=cid)["tournament"],
                                     "rows": [{"team": {"id": n}} for n in range(1, 5)]}]}
        else:
            number = int(path.rsplit("/", 1)[1])
            states = (statuses or {}).get(number, ["finished", "finished"] if number == 1 or cid in ended else ["finished", "notstarted"])
            payload = {"events": [event(cid * 1000 + 10 * number + i, number, state, pair, cid)
                                  for i, (state, pair) in enumerate(zip(states, PAIRS[number]))]}
        if change:
            payload = change(job, payload)
        discovery.observe(path, job["lease"], entry(path, payload))


def collect(campaign, event_override=None, lineup_payload=None):
    """Supply deterministic network observations through the real native queue."""
    paths = []
    while True:
        job = campaign.next_match(resume=True)
        if job.get("done"):
            return paths
        eid, lease = job["event_id"], job["lease"]
        assert job["dynamic_paths"]
        metadata = deepcopy(campaign.items[eid]["reference"]["event"])
        if event_override:
            metadata = event_override(metadata)
        origin = time.time() - 1
        index = 0
        for _ in range(20):
            coverage = campaign.coverage(eid)
            pending = next((c for c in coverage["components"] if c["status"] not in {"success", "empty", "unavailable"}), None)
            if pending is None:
                break
            path = pending["path"]
            paths.append(path)
            body = {"event": metadata} if pending["component"] == "event" else None
            if pending["component"] == "lineups" and lineup_payload is not None:
                body = lineup_payload
            campaign.observe(eid, lease, entry(path, body, 200 if body else 404, origin + index / 1000))
            index += 1
        else:
            pytest.fail("Daily queue did not reach terminal coverage")
        assert campaign.finish(eid, lease, page_event_id=None, page_evidence={"collection_mode": "session"})["state"] == "complete"


def prepare_collected(tmp_path, **discovery_options):
    prepared = new_discovery(tmp_path)
    with open_discovery(prepared["directory"]) as discovery:
        finish_discovery(discovery, **discovery_options)
        campaign = discovery.prepare_collection(campaign_base=tmp_path / "campaigns")
    with Campaign(campaign["directory"]) as collection:
        collect(collection)
    return prepared, campaign


def test_mixed_round_includes_entire_round_and_seasons_end_independently(tmp_path):
    leagues = [LEAGUE, {**LEAGUE, "name": "Ended", "competition_id": 18}, {**LEAGUE, "name": "Unlisted", "competition_id": 19}]
    prepared = prepare_discovery(season="26/27", leagues=leagues, mode="daily", base=tmp_path / "discovery",
        history_root=tmp_path / "history", output_root=tmp_path / "daily")
    with open_discovery(prepared["directory"]) as discovery:
        assert isinstance(discovery, DailyDiscovery)
        paths = finish_discovery(discovery, ended=[18], absent=[19])
        status = discovery.status()
        assert status["ready_to_collect"]
        active, ended, missing = status["seasons"]
        assert (active["last_completed_round"], active["target_round"]) == (1, 2)
        assert (active["history_matches"], active["unfinished_matches"]) == (3, 1)
        assert ended["state"] == "season_complete" and ended["target_round"] is None
        assert missing["state"] == "season_not_listed"
        assert not any("/17/" in p and p.endswith("/round/3") for p in paths)
        result = discovery.prepare_collection(campaign_base=tmp_path / "campaigns")
        assert result == discovery.prepare_collection(campaign_base=tmp_path / "campaigns")
        assert result["matches"] == 10


def test_completed_anchor_advances_once_to_next_round_and_handles_old_postponement(tmp_path):
    prepared = new_discovery(tmp_path)
    with open_discovery(prepared["directory"]) as discovery:
        finish_discovery(discovery, statuses={1: ["finished", "postponed"], 2: ["finished", "finished"], 3: ["notstarted", "notstarted"]})
        status, records = discovery.season_selection(17)
        assert status["target_round"] == 3
        assert status["unfinished_matches"] == 3
        assert len(records) == 6


@pytest.mark.parametrize("states", [["inprogress", "notstarted"], ["interrupted", "finished"], ["notstarted", "notstarted"]])
def test_started_unstarted_and_interrupted_rounds_remain_selected(tmp_path, states):
    prepared = new_discovery(tmp_path)
    with open_discovery(prepared["directory"]) as discovery:
        finish_discovery(discovery, statuses={2: states})
        summary, records = discovery.season_selection(17)
        assert summary["target_round"] == 2 and len(records) == 4
        assert summary["history_matches"] == 2 + states.count("finished")


def test_postponed_last_round_is_revisited_without_inventing_next_round(tmp_path):
    prepared = new_discovery(tmp_path)
    with open_discovery(prepared["directory"]) as discovery:
        finish_discovery(discovery, anchor=3, statuses={2: ["finished", "finished"], 3: ["finished", "postponed"]})
        season = discovery.status()["seasons"][0]
        assert season["state"] == "waiting_for_rescheduled_matches" and season["target_round"] is None
        assert season["unfinished_matches"] == 1


@pytest.mark.parametrize("damage", ["truncated", "duplicate_teams", "unknown_status"])
def test_incomplete_round_is_not_publishable(tmp_path, damage):
    def change(job, payload):
        if job["path"].endswith("/round/2"):
            if damage == "truncated": payload["events"].pop()
            elif damage == "duplicate_teams": payload["events"][1]["homeTeam"] = payload["events"][0]["homeTeam"]
            else: payload["events"][1]["status"] = {"type": "new_provider_state"}
        return payload
    prepared = new_discovery(tmp_path)
    with open_discovery(prepared["directory"]) as discovery:
        finish_discovery(discovery, change=change)
        assert not discovery.status()["ready_to_collect"]
        with pytest.raises(ValueError, match="coverage"):
            discovery.prepare_collection(campaign_base=tmp_path / "campaigns")


def test_rescheduled_identity_replaces_obsolete_postponed_fixture_only(tmp_path):
    def change(job, payload):
        if job["path"].endswith("/round/2"):
            old = deepcopy(payload["events"][1])
            old.update(id=999, status={"type": "postponed"}, startTimestamp=100)
            payload["events"].append(old)
        return payload
    prepared = new_discovery(tmp_path)
    with open_discovery(prepared["directory"]) as discovery:
        finish_discovery(discovery, change=change)
        summary, records = discovery.season_selection(17)
        assert discovery.status()["ready_to_collect"]
        assert [r["id"] for r in records].count(17021) == 1 and all(r["id"] != 999 for r in records)
        assert summary["superseded_fixtures"][0]["replacement_event_id"] == 17021
        assert discovery.db.execute("SELECT 1 FROM events WHERE id=999").fetchone()


@pytest.mark.parametrize("status, expected", [("notstarted", {"event", "pregame", "lineups"}),
    ("inprogress", {"event", "lineups"}), ("interrupted", {"event", "lineups"}), ("postponed", {"event"})])
def test_fresh_metadata_drives_upcoming_paths_and_valid_unavailable_responses(tmp_path, status, expected):
    e = event(status=status)
    config = prepare_campaign([101], references={101: {"event": e, "observed_at": time.time()}},
        profile=replace(PROFILE, include_upcoming=True), policy={"collection_mode": "session"}, base=tmp_path,
        source_metadata={"collection": "daily_update"})
    with Campaign(config["directory"]) as campaign:
        job = campaign.next_match()
        assert job["paths"] == ["/event/101"]
        campaign.observe(101, job["lease"], entry("/event/101", {"event": e}))
        assert {j.component for j in campaign.jobs(101).values()} == expected
        for path in list(campaign.jobs(101)):
            if path != "/event/101": campaign.observe(101, job["lease"], entry(path, code=404))
        result = campaign.finish(101, job["lease"], page_event_id=None, page_evidence={"collection_mode": "session"})
        tables = read_tables(json.loads(Path(result["report"]).read_text())["manifest"])
        assert tables["matches"][0]["status"] == status
        assert any(r["status"] == "not_applicable_before_completion" for r in tables["coverage"])
        assert result["state"] == "complete"


def test_resuming_old_pregame_snapshot_refreshes_metadata_and_fetches_final_components(tmp_path):
    e = event(status="notstarted")
    config = prepare_campaign([101], references={101: {"event": e, "observed_at": time.time()}},
        profile=replace(PROFILE, include_upcoming=True), policy={"collection_mode": "session"}, base=tmp_path,
        source_metadata={"collection": "daily_update"})
    with Campaign(config["directory"]) as campaign:
        job = campaign.next_match()
        campaign.observe(101, job["lease"], entry("/event/101", {"event": e}, observed=time.time() - 4000))
        campaign.observe(101, job["lease"], entry("/event/101/lineups", code=404, observed=time.time() - 3999))
        campaign.pause()
    with Campaign(config["directory"]) as campaign:
        job = campaign.next_match(resume=True)
        assert campaign.coverage(101)["components"][0]["status"] == "not_observed"
        def finished(metadata):
            metadata["status"] = {"type": "finished"}
            return metadata
        paths = collect(campaign, finished)
        assert "/event/101/statistics" in paths and "/event/101/lineups" in paths
        report = json.loads(Path(campaign.db.execute("SELECT report FROM queue").fetchone()[0]).read_text())
        assert read_tables(report["manifest"])["matches"][0]["status"] == "finished"


def test_repeated_publication_is_idempotent_and_new_run_only_queues_unfinished(tmp_path):
    prepared, campaign = prepare_collected(tmp_path)
    output = tmp_path / "daily"
    first = publish_daily_campaign(campaign["directory"])
    stem = "Example_League_26_27"
    assert len(first["seasons"][0]["history_ids"]) == 3 and first["seasons"][0]["prediction_ids"] == [17021]
    assert first == publish_daily_campaign(campaign["directory"])
    original = source_snapshot(output, stem)[0]
    second = new_discovery(tmp_path)
    with open_discovery(second["directory"]) as discovery:
        finish_discovery(discovery)
        refreshed = discovery.prepare_collection(campaign_base=tmp_path / "campaigns")
        assert refreshed["matches"] == 1 and refreshed["reused_matches"] == 3
    with Campaign(refreshed["directory"]) as collection:
        collect(collection)
    publish_daily_campaign(refreshed["directory"])
    all_rows = read_daily_tables(output, view="all")[stem]["matches"]
    assert len(all_rows) == len({r["event_id"] for r in all_rows}) == 4
    assert hash_file(original["manifest"]) == original["sha256"]
    assert read_daily_tables(output, now=time.time() + 86400)[stem]["matches"] == []
    latest = (output / "LATEST.json").read_bytes()
    publish_daily_campaign(campaign["directory"])
    assert (output / "LATEST.json").read_bytes() == latest
    assert completed_season_stems(refreshed["directory"]) == []
    with pytest.raises(ValueError, match="Daily updates"):
        snapshot_season(refreshed["directory"], stem)


def test_published_daily_run_survives_capture_cleanup(tmp_path):
    import shutil

    _, campaign = prepare_collected(tmp_path)
    published = publish_daily_campaign(campaign["directory"])
    output = tmp_path / "daily"
    before = read_daily_tables(output, view="all")
    shutil.rmtree(Path(campaign["directory"]) / "imports")
    (output / "LATEST.json").unlink()

    assert publish_daily_campaign(campaign["directory"]) == published
    assert read_daily_tables(output, view="all") == before
    # A new discovery reuses saved finished fixtures without old raw imports.
    prepared = new_discovery(tmp_path)
    with open_discovery(prepared["directory"]) as discovery:
        finish_discovery(discovery)
        refreshed = discovery.prepare_collection(campaign_base=tmp_path / "campaigns")
        assert refreshed["reused_matches"] == 3 and refreshed["matches"] == 1


def test_default_daily_run_publishes_and_reuses_the_canonical_folder(tmp_path, monkeypatch):
    from scraping.sofascore_web_v3 import daily, discovery as discovery_module

    output = tmp_path / "data/xDiyo_data"
    monkeypatch.setattr(daily, "OUTPUT", output)
    monkeypatch.setattr(discovery_module, "ROOT", tmp_path)
    # Both the script helper and discovery initiated by the extension use it.
    manual = prepare_discovery(season="26/27", leagues=[LEAGUE], mode="daily", base=tmp_path / "manual")
    manual_config = json.loads((Path(manual["directory"]) / "config.json").read_text())
    assert Path(manual_config["output_root"]) == Path(manual_config["history_root"]) == output

    settings = dict(season="26/27", leagues=[LEAGUE], discovery_base=tmp_path / "discovery")
    for iteration in range(2):
        prepared = prepare_daily_run(**settings)
        with open_discovery(prepared["directory"]) as discovery:
            assert Path(discovery.config["output_root"]) == Path(discovery.config["history_root"]) == output
            finish_discovery(discovery)
            campaign = discovery.prepare_collection(campaign_base=tmp_path / "campaigns")
        assert campaign["reused_matches"] == (3 if iteration else 0)
        with Campaign(campaign["directory"]) as collection:
            collect(collection)
        publish_daily_campaign(campaign["directory"])
    rows = read_daily_tables(output, view="all")["Example_League_26_27"]["matches"]
    assert len(rows) == len({r["event_id"] for r in rows}) == 4
    assert (output / "_collection/last_run.json").is_file()
    assert not (tmp_path / "data/xDiyo_data_daily").exists()


def test_completed_publication_can_be_relocated_without_rewriting_its_frozen_plan(tmp_path):
    import shutil

    prepared, campaign = prepare_collected(tmp_path)
    saved = publish_daily_campaign(campaign["directory"])
    source, target = tmp_path / "daily", tmp_path / "history"
    expected = read_daily_tables(source, view="all")
    cohort = Path(prepared["directory"]) / "cohort.json"
    frozen_hash = hash_file(cohort)
    shutil.copytree(source, target)
    for season in saved["seasons"]:
        for field in ("manifest", "season_file"):
            season[field] = str(target / Path(season[field]).relative_to(source))
    (Path(prepared["directory"]) / "publication.json").write_text(json.dumps(saved))
    config_path = Path(campaign["directory"]) / "config.json"
    config = json.loads(config_path.read_text())
    config["source_metadata"]["output_root"] = str(target)
    config_path.write_text(json.dumps(config))

    assert publish_daily_campaign(campaign["directory"]) == saved
    assert read_daily_tables(target, view="all") == expected
    assert hash_file(cohort) == frozen_hash == saved["cohort_sha256"]
    # The next run finds the moved completed matches in the shared folder.
    prepared = prepare_discovery(season="26/27", leagues=[LEAGUE], mode="daily", base=tmp_path / "discovery",
        history_root=target, output_root=target)
    with open_discovery(prepared["directory"]) as discovery:
        finish_discovery(discovery)
        refreshed = discovery.prepare_collection(campaign_base=tmp_path / "campaigns")
        assert refreshed["reused_matches"] == 3 and refreshed["matches"] == 1


@pytest.mark.parametrize("published", [False, True])
def test_changing_output_alone_cannot_redirect_a_daily_run(tmp_path, published):
    _, campaign = prepare_collected(tmp_path)
    if published:
        publish_daily_campaign(campaign["directory"])
    config_path = Path(campaign["directory"]) / "config.json"
    config = json.loads(config_path.read_text())
    config["source_metadata"]["output_root"] = str(tmp_path / "other")
    config_path.write_text(json.dumps(config))
    message = "inside its capture directory" if published else "configuration disagree"
    with pytest.raises(ValueError, match=message):
        publish_daily_campaign(campaign["directory"])


def test_finished_update_replaces_fixture_once_and_next_round_is_added(tmp_path):
    _, campaign = prepare_collected(tmp_path)
    initial = publish_daily_campaign(campaign["directory"])
    original_manifest = initial["seasons"][0]["manifest"]
    prepared = new_discovery(tmp_path)
    with open_discovery(prepared["directory"]) as discovery:
        finish_discovery(discovery, statuses={2: ["finished", "finished"], 3: ["notstarted", "notstarted"]})
        updated = discovery.prepare_collection(campaign_base=tmp_path / "campaigns")
        assert updated["matches"] == 3 and updated["reused_matches"] == 3
    with Campaign(updated["directory"]) as queue:
        collect(queue)
    latest = publish_daily_campaign(updated["directory"])
    assert len(latest["seasons"][0]["history_ids"]) == 4
    assert latest["seasons"][0]["prediction_ids"] == [17030, 17031]
    rows = read_daily_tables(tmp_path / "daily", view="all")["Example_League_26_27"]["matches"]
    assert [r["event_id"] for r in rows].count(17021) == 1
    assert next(r for r in rows if r["event_id"] == 17021)["status"] == "finished"
    assert next(r for r in read_tables(original_manifest)["matches"] if r["event_id"] == 17021)["status"] == "notstarted"


def test_empty_refresh_reuses_all_finished_and_publishes_no_prediction_candidates(tmp_path):
    _, campaign = prepare_collected(tmp_path, ended=[17])
    publish_daily_campaign(campaign["directory"])
    prepared = new_discovery(tmp_path)
    with open_discovery(prepared["directory"]) as discovery:
        finish_discovery(discovery, ended=[17])
        updated = discovery.prepare_collection(campaign_base=tmp_path / "campaigns")
        assert updated["matches"] == 0 and updated["reused_matches"] == 6
    messages = []
    result = watch_daily_run(prepared["discovery_id"], discovery_base=tmp_path / "discovery", once=True, emit=messages.append)
    assert result["seasons"][0]["state"] == "unchanged"
    assert result["seasons"][0]["prediction_ids"] == []
    assert read_daily_tables(tmp_path / "daily")["Example_League_26_27"]["matches"] == []


def test_changed_score_forces_refresh(tmp_path):
    _, campaign = prepare_collected(tmp_path)
    publish_daily_campaign(campaign["directory"])
    prepared = new_discovery(tmp_path)
    def corrected(job, payload):
        if job["path"].endswith("/round/1"):
            payload["events"][0]["homeScore"]["current"] = 50
        return payload
    with open_discovery(prepared["directory"]) as discovery:
        finish_discovery(discovery, change=corrected)
        updated = discovery.prepare_collection(campaign_base=tmp_path / "campaigns")
        assert updated["matches"] == 2 and updated["reused_matches"] == 2


def test_publication_rejects_concurrent_changed_output_and_repairs_lost_latest_pointer(tmp_path):
    _, campaign = prepare_collected(tmp_path)
    result = publish_daily_campaign(campaign["directory"])
    output = tmp_path / "daily"
    (output / "LATEST.json").unlink()
    assert publish_daily_campaign(campaign["directory"]) == result
    assert (output / "LATEST.json").exists()
    _, a = prepare_collected(tmp_path)
    _, b = prepare_collected(tmp_path)
    publish_daily_campaign(b["directory"])
    with pytest.raises(ValueError, match="changed since"):
        publish_daily_campaign(a["directory"])


def test_hash_tamper_is_not_reused_or_silently_overwritten(tmp_path):
    _, campaign = prepare_collected(tmp_path)
    result = publish_daily_campaign(campaign["directory"])
    manifest = Path(result["seasons"][0]["manifest"])
    specification = json.loads(manifest.read_text())
    (manifest.parent / specification["tables"]["coverage"]["file"]).write_bytes(b"damaged")
    with pytest.raises(ValueError, match="hash mismatch"):
        publish_daily_campaign(campaign["directory"])


def test_snapshot_merge_deduplicates_and_keeps_last_good_data_per_heatmap_side():
    match = parse_match(event())
    before = {"matches": [match], "lineup_players": [{"event_id": 101, "phase": "pregame", "observed_at": 1}],
              "heatmap_points": [{"event_id": 101, "team_id": 1, "x": 10}, {"event_id": 101, "team_id": 2, "x": 20}]}
    after = {"matches": [{**match, "status": "finished"}],
             "lineup_players": [{"event_id": 101, "phase": "postgame", "observed_at": 2}] * 2,
             "heatmap_points": [{"event_id": 101, "team_id": 2, "x": 30}],
             "coverage": [{"event_id": 101, "component": "heatmap", "side": "1", "latest_attempt_status": "unavailable", "status": "unavailable"}]}
    merge_capture(before, after)
    merge_capture(before, after)
    assert len(before["matches"]) == 1 and len(before["lineup_players"]) == 2
    assert before["heatmap_points"] == [{"event_id": 101, "team_id": 1, "x": 10}, {"event_id": 101, "team_id": 2, "x": 30}]
    assert before["coverage"][0]["status"] == "cached_last_good"


def test_minimal_runner_resumes_same_unfinished_discovery_and_native_opens_daily_mode(tmp_path):
    settings = dict(season="26/27", leagues=[LEAGUE], discovery_base=tmp_path / "discovery",
                    history_root=tmp_path / "history", output_root=tmp_path / "daily")
    first = prepare_daily_run(**settings)
    second = prepare_daily_run(**settings)
    assert first["discovery_id"] == second["discovery_id"]
    host = Host(base=tmp_path / "campaigns", discovery_base=tmp_path / "discovery", lock_path=tmp_path / "writer.lock")
    try:
        host.dispatch({"operation": "discovery_open", "discovery_id": first["discovery_id"]})
        assert isinstance(host.discovery, DailyDiscovery)
        assert host.dispatch({"operation": "hello", "protocol": 1})["version"] == "0.4.1"
        read_only = daily_status(first["discovery_id"], discovery_base=tmp_path / "discovery")
        assert read_only["phase"] == "discovery"
    finally:
        host.close()


def test_optional_lineups_can_be_added_without_mistaking_an_old_profile_for_complete(tmp_path):
    prepared = new_discovery(tmp_path, collection_profile=replace(PROFILE, include_upcoming=True, include_lineups=False))
    with open_discovery(prepared["directory"]) as discovery:
        finish_discovery(discovery)
        first = discovery.prepare_collection(campaign_base=tmp_path / "campaigns")
    with Campaign(first["directory"]) as queue:
        collect(queue)
    publish_daily_campaign(first["directory"])
    prepared = new_discovery(tmp_path)
    with open_discovery(prepared["directory"]) as discovery:
        finish_discovery(discovery)
        second = discovery.prepare_collection(campaign_base=tmp_path / "campaigns")
        assert second["reused_matches"] == 0 and second["matches"] == 4


def test_saved_history_reused_without_modifying_any_source_and_retaining_other_fields(tmp_path):
    _, campaign = prepare_collected(tmp_path)
    result = publish_daily_campaign(campaign["directory"])
    manifest = Path(result["seasons"][0]["manifest"])
    tables = read_tables(manifest)
    # Representative additional statistics already repaired in the saved dataset.
    tables["statistics"] = to_arrow("statistics", [{"event_id": 17010, "key": "cornerKicks", "value": 7.0}]).to_pylist()
    stem = "Example_League_26_27"
    history = tmp_path / "history"
    publish_season(tables, history, Scope(17, 1700, (1, 2)), replace(PROFILE, include_upcoming=True), stem, {"test": "saved history"})
    before = {str(p.relative_to(history)): hash_file(p) for p in history.rglob("*") if p.is_file()}
    prepared = prepare_discovery(season="26/27", leagues=[LEAGUE], mode="daily", base=tmp_path / "discovery",
        history_root=history, output_root=tmp_path / "new_daily")
    with open_discovery(prepared["directory"]) as discovery:
        finish_discovery(discovery)
        second = discovery.prepare_collection(campaign_base=tmp_path / "campaigns")
        assert second["reused_matches"] == 3 and second["matches"] == 1
    with Campaign(second["directory"]) as queue:
        collect(queue)
    published = publish_daily_campaign(second["directory"])
    assert read_tables(published["seasons"][0]["manifest"])["statistics"] == tables["statistics"]
    assert before == {str(p.relative_to(history)): hash_file(p) for p in history.rglob("*") if p.is_file()}


@pytest.mark.parametrize("status, phase", [("notstarted", "pregame"), ("inprogress", "inprogress"), ("finished", "postgame")])
def test_lineup_observation_phase_follows_fresh_metadata(tmp_path, status, phase):
    metadata = event(status="notstarted")
    config = prepare_campaign([101], references={101: {"event": metadata, "observed_at": time.time()}},
        profile=replace(PROFILE, include_upcoming=True), policy={"collection_mode": "session"}, base=tmp_path,
        source_metadata={"collection": "daily_update"})
    def current(e):
        e["status"] = {"type": status}
        return e
    lineup = {"confirmed": True, "home": {"players": [{"player": {"id": 7, "name": "Player"}, "statistics": {"totalPass": 3}}]},
              "away": {"players": []}}
    with Campaign(config["directory"]) as queue:
        collect(queue, current, lineup)
        report = json.loads(Path(queue.db.execute("SELECT report FROM queue").fetchone()[0]).read_text())
    rows = read_tables(report["manifest"])
    assert {r["phase"] for r in rows["lineup_players"]} == {phase}
    assert {r["phase"] for r in rows["player_statistics"]} == {phase}


def test_publication_failure_restores_current_files_and_can_resume(tmp_path, monkeypatch):
    from scraping.sofascore_web_v3 import daily_storage
    _, first = prepare_collected(tmp_path)
    published = publish_daily_campaign(first["directory"])
    output = tmp_path / "daily"
    mutable = [output / "Example_League_26_27.parquet", output / "Example_League_26_27.manifest.json",
               output / "_tables/competition=17/season=1700/CURRENT.json", output / "LATEST.json"]
    before = {p: hash_file(p) for p in mutable}
    _, second = prepare_collected(tmp_path)
    atomic, fail = daily_storage.atomic_write_status, [True]
    def interrupted(path, contents):
        if Path(path) == mutable[1] and fail[0]:
            fail[0] = False
            raise OSError("Simulated interrupted public-file replacement")
        return atomic(path, contents)
    monkeypatch.setattr(daily_storage, "atomic_write_status", interrupted)
    with pytest.raises(OSError, match="Simulated interrupted"):
        publish_daily_campaign(second["directory"])
    assert before == {p: hash_file(p) for p in mutable}
    resumed = publish_daily_campaign(second["directory"])
    assert resumed["campaign_id"] == second["campaign_id"]
    assert hash_file(published["seasons"][0]["manifest"]) == published["seasons"][0]["manifest_sha256"]


def test_all_unlisted_seasons_publish_empty_current_selection_and_new_runner_run(tmp_path):
    settings = dict(season="26/27", leagues=[LEAGUE], discovery_base=tmp_path / "discovery",
                    history_root=tmp_path / "history", output_root=tmp_path / "daily")
    run = prepare_daily_run(**settings)
    with open_discovery(run["directory"]) as discovery:
        finish_discovery(discovery, absent=[17])
        campaign = discovery.prepare_collection(campaign_base=tmp_path / "campaigns")
        assert campaign["matches"] == 0
    publish_daily_campaign(campaign["directory"])
    assert read_daily_tables(tmp_path / "daily") == {"Example_League_26_27": {"matches": []}}
    assert prepare_daily_run(**settings)["discovery_id"] != run["discovery_id"]
