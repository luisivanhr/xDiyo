"""Metadata can be captured without asserting unfamiliar formats are valid."""
from datetime import datetime, timezone
from pathlib import Path
import json
import socket

import pytest

from scraping.sofascore_web_v3.discovery import open_discovery
from scraping.sofascore_web_v3.inventory_discovery import prepare_inventory


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", lambda *a: pytest.fail("No Python HTTP"))


def prepare(tmp_path, **extra):
    return prepare_inventory("format-review", [{"name": "Example", "competition_id": 242}],
        first_year=2017, last_year=2026, base=tmp_path, **extra)


def observe(discovery, payload):
    job = discovery.next_job()
    return discovery.observe(job["path"], job["lease"], {
        "startedDateTime": datetime.now(timezone.utc).isoformat(), "time": 1,
        "request": {"method": "GET", "url": "https://www.sofascore.com/api/v1" + job["path"]},
        "response": {"status": 200, "content": {"text": json.dumps(payload)}}})


def test_captures_ambiguous_seasons_and_stages_but_never_prepares_collection(tmp_path):
    saved = prepare(tmp_path)
    assert prepare(tmp_path) == saved
    inventory = [{"id": 1, "year": "2016/2017"}, {"id": 2, "year": "2017/2018"},
                 {"id": 3, "year": "2026"}, {"id": 4, "year": "2026"},
                 {"id": 5, "year": "2027"}]
    with open_discovery(saved["directory"]) as discovery:
        observe(discovery, {"seasons": inventory})
        for _ in range(3):
            observe(discovery, {"rounds": [{"name": "Unreviewed stage", "round": 1}]})
            observe(discovery, {"standings": [{"tournament": {"isGroup": True,
                "uniqueTournament": {"id": 242}}, "rows": []}]})
        state = discovery.status()
        assert state["discovery_complete"] and not state["ready_to_collect"]
        assert {s["season_id"] for s in state["seasons"]} == {2, 3, 4}
        assert state["jobs"] == {"complete": 7} and state["requests"] == 7
        assert discovery.db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
        assert discovery.db.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 7
        with pytest.raises(ValueError, match="cannot create collection"):
            discovery.prepare_collection()
        assert discovery.next_job()["done"]
    with open_discovery(saved["directory"], read_only=True) as discovery:
        assert discovery.status() == state


def test_empty_in_range_inventory_is_evidence_not_readiness(tmp_path):
    saved = prepare(tmp_path)
    with open_discovery(saved["directory"]) as discovery:
        observe(discovery, {"seasons": [{"id": 1, "year": "2015"}]})
        state = discovery.status()
        assert state["discovery_complete"] and state["seasons"] == []
        assert not state["ready_to_collect"]


@pytest.mark.parametrize("payload", [{}, {"seasons": []},
    {"seasons": [{"id": 1, "year": "2020"}, {"id": 1, "year": "2021"}]},
    {"seasons": [{"id": "1", "year": "2020"}]}])
def test_bad_inventory_pauses_and_retains_raw_evidence(tmp_path, payload):
    saved = prepare(tmp_path)
    with open_discovery(saved["directory"]) as discovery:
        with pytest.raises(ValueError):
            observe(discovery, payload)
        assert discovery.next_job()["paused"]
        assert discovery.status()["jobs"] == {"failed": 1}
        assert discovery.db.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 1


def test_foreign_standings_fail_and_injected_event_jobs_cannot_queue_matches(tmp_path):
    saved = prepare(tmp_path)
    with open_discovery(saved["directory"]) as discovery:
        observe(discovery, {"seasons": [{"id": 1, "year": "2020"}]})
        observe(discovery, {"rounds": []})
        with pytest.raises(ValueError, match="another competition"):
            observe(discovery, {"standings": [{"tournament": {"uniqueTournament": {"id": 136}}}]})
        with pytest.raises(ValueError, match="only season"):
            discovery.process({"competition": 242, "season": 1, "kind": "events"}, {"events": []}, 0)
        assert discovery.db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0


def test_changed_preparation_preserves_saved_configuration(tmp_path):
    saved = prepare(tmp_path)
    path = Path(saved["directory"]) / "config.json"
    original = path.read_bytes()
    with pytest.raises(ValueError, match="settings differ"):
        prepare_inventory("format-review", [{"name": "Example", "competition_id": 242}],
            first_year=2018, last_year=2026, base=tmp_path)
    assert path.read_bytes() == original


def test_round_inspection_retains_only_raw_evidence(tmp_path):
    saved = prepare(tmp_path)
    with open_discovery(saved["directory"]) as discovery:
        observe(discovery, {"seasons": [{"id": 1, "year": "2020"}]})
        observe(discovery, {"rounds": []})
        observe(discovery, {"standings": []})
        path = "/unique-tournament/242/season/1/events/round/0"
        with discovery.db:
            discovery.add_job(path, 242, "round_inspection", 1, 0)
        assert discovery.next_job()["inspection_only"]
        observe(discovery, {"events": [{"id": 123, "roundInfo": {"round": 0}}]})
        assert discovery.status()["discovery_complete"]
        assert not discovery.status()["ready_to_collect"]
        assert discovery.db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
        assert discovery.db.execute("SELECT body FROM responses WHERE path=?", (path,)).fetchone()
        with pytest.raises(ValueError, match="Invalid round inspection"):
            discovery.process({"competition": 242, "season": 1, "kind": "round_inspection",
                               "round": 0, "path": "/event/123"}, {"events": []}, 0)


def page_event(eid):
    return {"id": eid, "season": {"id": 1}, "tournament": {"uniqueTournament": {"id": 242}}}


def ready_inventory(tmp_path):
    saved = prepare(tmp_path)
    with open_discovery(saved["directory"]) as discovery:
        observe(discovery, {"seasons": [{"id": 1, "year": "2020"}]})
        observe(discovery, {"rounds": []})
        observe(discovery, {"standings": []})
    return saved


def test_complete_page_chain_remains_inspection_and_reuses_saved_progress(tmp_path):
    saved = ready_inventory(tmp_path)
    with open_discovery(saved["directory"]) as d:
        d.inspect_event_pages(242, 1)
        job = d.next_job()
        assert job["event_page_number"] == 0 and job["event_page_direction"] == "last" and job["inspection_only"]
        observe(d, {"events": [page_event(10)], "hasNextPage": True})
    with open_discovery(saved["directory"]) as d:
        d.inspect_event_pages(242, 1)
        assert d.next_job()["event_page_number"] == 1
        observe(d, {"events": [page_event(11)], "hasNextPage": False})
        state = d.status()
        assert state["discovery_complete"] and not state["ready_to_collect"]
        assert state["event_page_inventories"][0]["events"] == 2
        assert d.db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
        with pytest.raises(ValueError, match="cannot create collection"):
            d.prepare_collection()


@pytest.mark.parametrize("mutation", ["repeat", "foreign", "bad_flag", "empty_intermediate", "missing_page", "changed_hash"])
def test_partial_or_conflicting_page_evidence_cannot_finish(tmp_path, mutation):
    saved = ready_inventory(tmp_path)
    with open_discovery(saved["directory"]) as d:
        d.inspect_event_pages(242, 1)
        observe(d, {"events": [page_event(10)], "hasNextPage": True})
        payload = {"events": [page_event(11)], "hasNextPage": False}
        if mutation == "repeat": payload["events"][0]["id"] = 10
        elif mutation == "foreign": payload["events"][0]["season"]["id"] = 2
        elif mutation == "bad_flag": payload["hasNextPage"] = 0
        elif mutation == "empty_intermediate": payload = {"events": [], "hasNextPage": True}
        elif mutation == "missing_page":
            with d.db: d.db.execute("DELETE FROM responses WHERE path LIKE '%/events/last/0'")
        else:
            with d.db: d.db.execute("UPDATE responses SET sha256='changed' WHERE path LIKE '%/events/last/0'")
        with pytest.raises(ValueError): observe(d, payload)
        assert not d.status()["discovery_complete"]
        assert d.status()["event_page_inventories"] == []


def test_empty_terminal_page_and_foreign_scope_remain_noncollectible(tmp_path):
    saved = ready_inventory(tmp_path)
    with open_discovery(saved["directory"]) as d:
        for cid, sid in [(242, 2), (17, 1)]:
            with pytest.raises(ValueError, match="exact captured"):
                d.inspect_event_pages(cid, sid)
        d.inspect_event_pages(242, 1, directions=("next",))
        observe(d, {"events": [], "hasNextPage": False})
        assert d.status()["event_page_inventories"][0]["events"] == 0
        assert not d.status()["ready_to_collect"]


@pytest.mark.parametrize("code", [404, 403, 500])
def test_only_observed_rounds_404_can_be_acknowledged_without_claiming_fixtures(tmp_path, code):
    saved = prepare(tmp_path)
    with open_discovery(saved["directory"]) as d:
        observe(d, {"seasons": [{"id": 1, "year": "2020"}]})
        job = d.next_job()
        with pytest.raises(ValueError):
            d.observe(job["path"], job["lease"], {
                "startedDateTime": datetime.now(timezone.utc).isoformat(),
                "request": {"method": "GET", "url": "https://www.sofascore.com/api/v1" + job["path"]},
                "response": {"status": code}})
        if code != 404:
            with pytest.raises(ValueError): d.acknowledge_unavailable_rounds(242, 1, reason="Review found no numbered-round endpoint")
            assert d.status()["jobs"]["failed"] == 1
        else:
            d.acknowledge_unavailable_rounds(242, 1, reason="Review found no numbered-round endpoint; inspect paginated events separately")
            assert d.next_job(resume=True)["path"].endswith("standings/total")
            observe(d, {"standings": []})
            state = d.status()
            assert state["jobs"] == {"complete": 2, "unavailable": 1}
            assert state["unavailable_metadata"][0]["http_status"] == 404
            assert not state["ready_to_collect"]


def test_selected_mls_page_inspection_retains_rounds_404_and_continues_metadata(tmp_path):
    saved = prepare(tmp_path)
    with open_discovery(saved["directory"]) as d:
        observe(d, {"seasons": [{"id": 1, "year": "2020"}]})
        d.inspect_event_pages(242, 1)
        job = d.next_job()
        result = d.observe(job["path"], job["lease"], {
            "startedDateTime": datetime.now(timezone.utc).isoformat(),
            "request": {"method": "GET", "url": "https://www.sofascore.com/api/v1" + job["path"]},
            "response": {"status": 404}})
        assert result["metadata_unavailable"] and not result["saved"]
        assert d.next_job()["path"].endswith("standings/total")
        assert d.db.execute("SELECT http_status FROM attempts ORDER BY id DESC LIMIT 1").fetchone()[0] == 404
        assert not d.status()["ready_to_collect"]


def test_reviewed_boundary_overlap_preserves_originals_and_remains_noncollectible(tmp_path):
    saved = ready_inventory(tmp_path)
    with open_discovery(saved["directory"]) as d:
        d.inspect_event_pages(242, 1)
        first = page_event(10)
        first["homeTeam"] = {"id": 100, "userCount": 4}
        observe(d, {"events": [first, page_event(11)], "hasNextPage": True})
        second = json.loads(json.dumps(first))
        second["homeTeam"]["userCount"] = 5
        with pytest.raises(ValueError, match="Repeated fixture"):
            observe(d, {"events": [page_event(12), second], "hasNextPage": False})
        before = list(d.db.execute("SELECT path,body,sha256 FROM responses ORDER BY path"))
        d.review_page_overlap(242, 1, event_id=10,
            reason="Same fixture at adjacent page boundary; only follower count changed; schedule review still required")
        assert len(d.revalidate_saved_failures()) == 1
        assert list(d.db.execute("SELECT path,body,sha256 FROM responses ORDER BY path")) == before
        assert d.status()["event_page_inventories"][0]["events"] == 3
        assert not d.status()["ready_to_collect"]


@pytest.mark.parametrize("change", ["score", "round", "club", "position", "season", "unused"])
def test_overlap_review_never_hides_fixture_conflicts(change):
    from copy import deepcopy
    from scraping.sofascore_web_v3.event_pages import review_boundary_overlap, validate_event_pages
    event = dict(page_event(10), homeTeam={"id": 100}, homeScore={"current": 2}, roundInfo={"round": 3})
    pages = [{"page": 0, "payload": {"events": [event, page_event(11)], "hasNextPage": True}},
             {"page": 1, "payload": {"events": [page_event(12), deepcopy(event)], "hasNextPage": False}}]
    review = review_boundary_overlap(pages, competition_id=242, season_id=1, event_id=10)
    changed = pages[1]["payload"]["events"][-1]
    if change == "score": changed["homeScore"]["current"] = 3
    elif change == "round": changed["roundInfo"]["round"] = 4
    elif change == "club": changed["homeTeam"]["id"] = 101
    elif change == "season": changed["season"]["id"] = 2
    elif change == "position": pages[1]["payload"]["events"].reverse()
    else: changed["id"] = 13
    with pytest.raises(ValueError):
        validate_event_pages(pages, competition_id=242, season_id=1, reviewed_overlaps=[review])
