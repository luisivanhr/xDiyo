"""Offline browser event replay. No HTTP calls, browser launch, or old cache writes."""
import base64
from dataclasses import replace
import json
from pathlib import Path
import time

import pytest

from scraping.sofascore.export import read_tables
from scraping.sofascore.models import CollectionPaused
from scraping.sofascore.store import Store
from scraping.sofascore_web_v3.capture import (
    PROFILE, WebsiteCapture, api_path, match_url, reference_from_cache,
)

FIXTURES = Path(__file__).parents[1]/"fixtures/sofascore"


def fixture(component):
    return json.loads((FIXTURES/f"12436870_{component}.json").read_text(encoding="utf-8"))


def log(method, **params):
    return {"message":json.dumps({"message":{"method":"Network."+method,"params":params}})}


class BrowserReplay:
    """Only implements observation and ordinary navigation; any injected fetch fails."""
    def __init__(self):
        self.logs = []
        self.bodies = {}
        self.calls = []
        self.current_url = "about:blank"
        self.closed = False

    def get(self, url):
        assert "/api/" not in url
        self.current_url = url
        self.calls.append(("navigate",url))

    def get_log(self, name):
        assert name == "performance"
        logs, self.logs = self.logs, []
        return logs

    def execute_cdp_cmd(self, command, params):
        assert command == "Network.getResponseBody"
        self.calls.append((command,params["requestId"]))
        return self.bodies[params["requestId"]]

    def save_screenshot(self, path):
        return False

    def quit(self):
        self.closed = True

    def response(self, path, payload=None, *, status=200, finish=True, encoded=False,
                 query="", method="GET", headers=None, from_cache=False):
        request_id = str(len(self.bodies)+1)
        body = json.dumps(payload)
        self.bodies[request_id] = {"body":base64.b64encode(body.encode()).decode() if encoded else body,
                                   "base64Encoded":encoded}
        self.logs.extend([
            log("requestWillBeSent",requestId=request_id,type="Fetch",
                request={"url":"https://www.sofascore.com/api/v1"+path+query,"method":method}),
            log("responseReceived",requestId=request_id,response={"status":status,
                "mimeType":"application/json","headers":headers or {},"fromDiskCache":from_cache}),
        ])
        if finish:
            self.logs.append(log("loadingFinished",requestId=request_id,encodedDataLength=len(body)))
        return request_id


@pytest.fixture
def capture(tmp_path):
    browser = BrowserReplay()
    reference = {"event":fixture("event")["event"],"observed_at":1700000000,"source":"frozen fixture"}
    instance = WebsiteCapture(reference,output_root=tmp_path/"out",ledger_root=tmp_path/"ledger",
                              driver_factory=lambda:browser).start()
    instance.open_match()
    yield instance
    instance.close()


def test_async_body_capture_is_selected_and_passive(capture):
    browser = capture.driver
    path = f"/event/{capture.event['id']}/shotmap"
    rid = browser.response(path,fixture("shotmap"),finish=False,encoded=True)
    browser.response("/event/999/shotmap",{"shotmap":[]})
    browser.response(f"/event/{capture.event['id']}/comments",{"comments":[]})
    browser.response(path,{"shotmap":[]},query="?period=1")
    capture.poll()
    assert capture.store.get(capture.jobs[path]) is None
    assert not any(call[0]=="Network.getResponseBody" for call in browser.calls)
    browser.logs.append(log("loadingFinished",requestId=rid,encodedDataLength=100))
    capture.poll()
    assert capture.store.payload(capture.jobs[path]) == fixture("shotmap")
    assert [call for call in browser.calls if call[0]=="Network.getResponseBody"] == [("Network.getResponseBody",rid)]
    assert capture.report()["observed_api_requests"] == 4
    assert capture.report()["target_api_requests"] == 1
    assert capture.report()["direct_api_requests_by_collector"] == 0


def test_end_to_end_normalized_export_preserves_xg_lineups_and_goal_actions(capture):
    browser = capture.driver
    for path,job in capture.jobs.items():
        payload = fixture(job.component) if job.component in {"event","shotmap","lineups","incidents"} else (
            {"statistics":[]} if job.component=="statistics" else
            {"playerPoints":[{"x":25,"y":75}],"goalkeeperPoints":[]} if job.component=="heatmap" else None)
        browser.response(path,payload,status=404 if job.component=="pregame" else 200)
    result = capture.finish()
    assert result["complete"] and result["closed"] and browser.closed
    tables = read_tables(result["manifest"])
    assert len(tables["shots"]) == len(fixture("shotmap")["shotmap"])
    assert any(row["xg"] is not None for row in tables["shots"])
    assert len(tables["lineup_players"]) == sum(len(fixture("lineups")[side]["players"]) for side in ("home","away"))
    assert len(tables["heatmap_points"]) == 2  # Synthetic heatmap fixture, not a live claim.
    assert len(tables["goal_actions"]) == 7
    assert "comments" not in tables
    assert result["event_metadata_source"] == "captured_website_response"
    assert json.loads(Path(result["manifest"]).read_text())["requested_components_terminal"]
    assert capture.report() == result


@pytest.mark.parametrize("status,expected",[(403,"blocked"),(404,"unavailable"),(429,"rate_limited")])
def test_failure_coverage_is_not_empty_data(capture,status,expected):
    path = f"/event/{capture.event['id']}/shotmap"
    capture.driver.response(path,status=status)
    result = capture.finish()
    assert not result["complete"]
    assert next(row["status"] for row in result["coverage"] if row["component"]=="shotmap") == expected
    assert "shots" not in result["table_rows"]
    if status==429:
        with Store(capture.ledger_root) as ledger:
            assert ledger.setting("cooldown_until") > time.time()+3500


def test_missing_metadata_and_components_cannot_be_published_as_complete(capture):
    result = capture.finish()
    assert not result["complete"]
    manifest = json.loads(Path(result["manifest"]).read_text())
    assert not manifest["requested_components_terminal"]
    assert manifest["provenance"]["event_metadata_source"] == "reference_from_existing_cache"
    assert read_tables(result["manifest"])["matches"][0]["event_metadata_observed_at"] == 1700000000


@pytest.mark.parametrize("fragment",["#id:999999","#id:999999,tab:lineups"])
def test_wrong_historical_fixture_stops_browsing(capture,fragment):
    browser = capture.driver
    browser.current_url = capture.url.split("#")[0]+fragment
    browser.response("/event/999999",{"event":{"id":999999}})
    capture.poll()
    assert capture.stop_reason == "event_identity_mismatch"
    assert capture.report()["page_event_id"] == 999999
    assert browser.current_url == "about:blank"
    assert capture.store.get(capture.event_job) is None
    with pytest.raises(CollectionPaused,match="identity"):
        capture.open_match()


def test_wrong_id_in_selected_response_is_rejected(capture):
    payload = fixture("event")
    payload["event"]["id"] = 999
    capture.driver.response(capture.event_job.path,payload)
    capture.poll()
    assert capture.store.get(capture.event_job)["status"] == "failed"
    assert capture.event_metadata_source == "reference_from_existing_cache"


def test_query_failure_does_not_poison_full_component_and_network_failure_preserves_last_good(capture):
    path = f"/event/{capture.event['id']}/shotmap"
    browser = capture.driver
    browser.response(path,fixture("shotmap"))
    capture.poll()
    browser.response(path,status=404,query="?period=1")
    capture.poll()
    assert capture.store.get(capture.jobs[path])["status"] == "success"
    rid = browser.response(path,finish=False,status=503)
    browser.logs.append(log("loadingFailed",requestId=rid))
    capture.poll()
    assert capture.store.get(capture.jobs[path])["status"] == "failed"
    assert capture.store.payload(capture.jobs[path]) == fixture("shotmap")


def test_idle_polls_do_not_rewrite_last_request_time_and_repeats_are_counted(capture):
    path = f"/event/{capture.event['id']}/shotmap"
    capture.driver.response(path,fixture("shotmap"),from_cache=True)
    capture.poll()
    last_finished = capture.ledger.setting("last_request_finished")
    capture.poll()
    assert capture.ledger.setting("last_request_finished") == last_finished
    capture.driver.response(path,fixture("shotmap"))
    capture.poll()
    assert capture.report()["duplicate_api_requests"] == 1
    assert capture.report()["browser_cache_responses"] == 1
    assert capture.ledger.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 2


def test_limit_stops_page_and_releases_shared_lock(capture):
    capture.max_api_requests = 1
    capture.driver.response("/unrelated",{})
    capture.poll()
    assert capture.stop_reason == "observed_api_request_limit"
    capture.finish()
    with Store(capture.ledger_root) as ledger, ledger.run_lock():
        assert ledger.db.execute("SELECT status FROM runs").fetchone()[0] == "incomplete"


def test_shared_access_latch_and_component_cache_are_preserved(tmp_path):
    ledger_root = tmp_path/"ledger"
    with Store(ledger_root) as ledger:
        ledger.set_setting("access_block",{"reason":"original block"})
    browser = BrowserReplay()
    instance = WebsiteCapture({"event":fixture("event")["event"],"observed_at":1700000000},
                              output_root=tmp_path/"out",ledger_root=ledger_root,driver_factory=lambda:browser)
    instance.start()
    instance.open_homepage()
    browser.response("/some/homepage/request",status=403)
    instance.poll()
    assert instance.stop_reason is None  # Human can complete the visible challenge during warmup.
    instance.open_match()
    browser.response(instance.event_job.path,status=403)
    instance.poll()
    assert instance.stop_reason == "access_denied"
    instance.close()
    with Store(ledger_root) as ledger:
        assert ledger.setting("access_block") == {"reason":"original block"}
        assert ledger.db.execute("SELECT COUNT(*) FROM components").fetchone()[0] == 0
        assert ledger.db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0


def test_active_cooldown_prevents_browser_start(tmp_path):
    with Store(tmp_path/"ledger") as ledger:
        ledger.set_setting("cooldown_until",time.time()+300)
    def unexpected_browser():
        raise AssertionError("No browser may launch during cooldown")
    instance = WebsiteCapture({"event":fixture("event")["event"],"observed_at":1700000000},
        output_root=tmp_path/"out",ledger_root=tmp_path/"ledger",driver_factory=unexpected_browser)
    with pytest.raises(CollectionPaused,match="cooldown"):
        instance.start()
    with Store(tmp_path/"ledger") as ledger, ledger.run_lock():
        assert ledger.setting("cooldown_until") > time.time()


def test_url_filter_and_optional_lineups(tmp_path):
    assert api_path("https://www.sofascore.com/api/v1/event/1?x=2") == "/event/1"
    assert api_path("https://unrelated.example/api/v1/event/1") is None
    assert api_path("http://www.sofascore.com/api/v1/event/1") is None
    event = fixture("event")["event"]
    assert match_url(event).endswith("#id:12436870")
    instance = WebsiteCapture({"event":event,"observed_at":1700000000},output_root=tmp_path,
                              profile=replace(PROFILE,include_lineups=False,include_injuries=False))
    assert not any(job.component=="lineups" for job in instance.jobs.values())
    with pytest.raises(ValueError,match="commentary"):
        WebsiteCapture({"event":event,"observed_at":1700000000},profile=replace(PROFILE,include_comments=True))


def test_reference_lookup_is_read_only(capture):
    reference = reference_from_cache(capture.event["id"],capture.store.root)
    assert reference["observed_at"] == 1700000000
    assert reference["event"] == capture.event
    with pytest.raises(ValueError,match="not in"):
        reference_from_cache(999,capture.store.root)


def test_closed_browser_exports_partial_checkpoint(capture):
    from selenium.common.exceptions import InvalidSessionIdException
    def lost_session(name):
        raise InvalidSessionIdException("Browser closed by user")
    capture.driver.get_log = lost_session
    result = capture.finish()
    assert result["stop_reason"] == "browser_unavailable"
    assert result["closed"] and not result["complete"]
    assert Path(result["manifest"]).is_file()
    with Store(capture.ledger_root) as ledger, ledger.run_lock():
        assert ledger.db.execute("SELECT status FROM runs").fetchone()[0] == "incomplete"


def test_reference_teams_must_match_fresh_event_metadata(capture):
    payload = fixture("event")
    payload["event"]["homeTeam"]["id"] = 999
    capture.driver.response(capture.event_job.path,payload)
    capture.poll()
    assert capture.store.get(capture.event_job)["status"] == "failed"
    assert capture.event == fixture("event")["event"]
