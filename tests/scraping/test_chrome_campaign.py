"""The Chrome writer must be durable, scoped, and offline, including failures."""
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import socket
import struct

import pytest

from scraping.sofascore.export import read_tables
from scraping.sofascore_web_v3.campaign import Campaign, prepare_campaign, campaign_path, validate_policy, collector_lock, request_pause, update_policy
from scraping.sofascore_web_v3.capture import PROFILE
from scraping.sofascore_web_v3.native_host import Host, read_message, write_message, serve, MAX_INPUT

FIXTURES=Path(__file__).parents[1]/"fixtures/sofascore"
EID=12436870
def fixture(name): return json.loads((FIXTURES/f"{EID}_{name}.json").read_text(encoding="utf-8"))
def entry(suffix="", payload=None, code=304, seconds=0):
    return {"startedDateTime":f"2026-09-12T02:00:{seconds:02d}Z","time":10,
            "request":{"method":"GET","url":f"https://www.sofascore.com/api/v1/event/{EID}{suffix}","headers":[{"name":"cookie","value":"SECRET_SENTINEL"}]},
            "response":{"status":code,"content":{"text":json.dumps(payload)} if payload is not None else {}}}
def entries():
    ev=fixture("event")["event"]
    return [entry(payload=fixture("event")), entry("/pregame-form",code=404),
            entry("/statistics",{"statistics":[]}),entry("/shotmap",fixture("shotmap")),
            entry("/lineups",fixture("lineups")),entry("/incidents",fixture("incidents")),
            *[entry(f"/heatmap/{ev[s]['id']}",{"playerPoints":[{"x":1,"y":2}],"goalkeeperPoints":[]}) for s in ("homeTeam","awayTeam")]]
def prepare(tmp_path, **kwargs):
    ref={"event":fixture("event")["event"],"observed_at":1700000000}
    return prepare_campaign([EID],base=tmp_path,references={EID:ref},**kwargs)


def test_native_delay_is_bounded_leased_and_offline(tmp_path):
    config=prepare(tmp_path)
    clock=[0.0]; sleeps=[]
    def sleep(seconds):
        sleeps.append(seconds); clock[0]+=seconds
    host=Host(base=tmp_path,lock_path=tmp_path/"delay.lock",monotonic=lambda:clock[0],sleep=sleep)
    try:
        host.dispatch({"operation":"open","campaign_id":config["campaign_id"]})
        job=host.dispatch({"operation":"next"})
        message={"operation":"delay","event_id":EID,"lease":job["lease"]}
        for value in (None,True,0,-1,1001,"50",1.5,float("nan")):
            with pytest.raises(ValueError,match="integer"):
                host.dispatch({**message,"milliseconds":value})
        with pytest.raises(ValueError,match="lease"):
            host.dispatch({**message,"lease":"stale","milliseconds":500})
        assert sleeps==[]
        result=host.dispatch({**message,"milliseconds":500})
        assert result["waited_ms"]>=500 and not result["pause_requested"]
        assert all(value<=0.1 for value in sleeps)
        assert host.campaign.db.execute("SELECT COUNT(*) FROM observations").fetchone()[0]==0
    finally:
        host.close()


def test_native_delay_observes_pause_during_wait(tmp_path):
    config=prepare(tmp_path)
    clock=[0.0]
    def sleep(seconds):
        clock[0]+=seconds
        if clock[0]>=0.2:
            (Path(config["directory"])/"PAUSE").write_text("requested during local pacing")
    host=Host(base=tmp_path,lock_path=tmp_path/"delay.lock",monotonic=lambda:clock[0],sleep=sleep)
    try:
        host.dispatch({"operation":"open","campaign_id":config["campaign_id"]})
        job=host.dispatch({"operation":"next"})
        result=host.dispatch({"operation":"delay","event_id":EID,"lease":job["lease"],"milliseconds":1000})
        assert result["pause_requested"] and 100<=result["waited_ms"]<1000
        assert host.campaign.db.execute("SELECT COUNT(*) FROM observations").fetchone()[0]==0
    finally:
        host.close()


@pytest.mark.parametrize("changes", [{"collection_mode":"unknown"},{"session_request_interval_seconds":-1},{"session_request_interval_seconds":True}])
def test_bad_session_policy_rejected(changes):
    with pytest.raises(ValueError): validate_policy(changes)


def test_session_finish_verifies_metadata_instead_of_anchor_page(tmp_path):
    config=prepare(tmp_path,policy={"collection_mode":"session"})
    with Campaign(config["directory"]) as queue:
        job=queue.next_match()
        for e in entries():
            e["_observationSource"]="chrome_session_request"
            queue.observe(EID,job["lease"],e)
        with pytest.raises(ValueError,match="evidence"):
            queue.finish(EID,job["lease"],page_event_id=EID)
        result=queue.finish(EID,job["lease"],page_event_id=None,page_evidence={"collection_mode":"session","session_request_count":8})
        assert result["state"]=="complete"
        report=json.loads(Path(result["report"]).read_text())
        assert report["page_evidence"]["identity_source"]=="captured_event_metadata"
        assert report["page_evidence"]["session_request_count"]==8
        assert read_tables(report["manifest"])["shots"]
        assert all(e["_observationSource"]=="chrome_session_request" for e in queue.entries(EID))


@pytest.mark.parametrize("metadata", [None,{"event":{"id":999}}])
def test_session_cannot_publish_missing_or_wrong_metadata(tmp_path,metadata):
    config=prepare(tmp_path,policy={"collection_mode":"session"})
    with Campaign(config["directory"]) as queue:
        job=queue.next_match()
        for e in entries()[1:]: queue.observe(EID,job["lease"],e)
        if metadata: queue.observe(EID,job["lease"],entry(payload=metadata))
        with pytest.raises(ValueError,match="verified"):
            queue.finish(EID,job["lease"],page_event_id=EID,page_evidence={"collection_mode":"session"})
        assert not (Path(config["directory"])/"imports").exists()


def test_session_error_retains_reason_but_discards_tokens(tmp_path):
    config=prepare(tmp_path,policy={"collection_mode":"session"})
    with Campaign(config["directory"]) as queue:
        job=queue.next_match()
        response=entry(code=403,payload={"error":{"token":"SECRET_SENTINEL"}})
        response["_apiError"]={"code":403,"reason":"challenge","token":"SECRET_SENTINEL"}
        queue.observe(EID,job["lease"],response)
        saved=queue.entries(EID)[0]
        assert saved["_apiError"]=={"code":"403","reason":"challenge"}
        assert "SECRET_SENTINEL" not in json.dumps(saved)

@pytest.mark.parametrize("policy", [{"session_challenge_recovery":1},{"session_challenge_recovery":"true"},
    {"session_challenge_cooldown_seconds":-1},{"session_challenge_cooldown_seconds":True},
    {"session_challenge_cooldown_seconds":float("inf")}])
def test_invalid_session_recovery_policy_rejected(policy):
    with pytest.raises(ValueError): validate_policy(policy)

def test_same_lease_challenge_recovery_preserves_failure_and_publishes_new_success(tmp_path):
    config=prepare(tmp_path,policy={"collection_mode":"session","session_challenge_recovery":True})
    with Campaign(config["directory"]) as queue:
        job=queue.next_match()
        for observation in entries():
            if observation["request"]["url"].endswith("/shotmap"): continue
            queue.observe(EID,job["lease"],observation)
        queue.observe(EID,job["lease"],entry("/shotmap",code=403,seconds=2))
        assert queue.coverage(EID)["blocked"]
        queue.observe(EID,job["lease"],entry("/shotmap",fixture("shotmap"),code=200,seconds=3))
        result=queue.finish(EID,job["lease"],page_event_id=None,page_evidence={"collection_mode":"session",
            "session_request_count":9,"session_challenge_recoveries":1,"session_recovery_wait_ms":180000,
            "session_expiry_wait_ms":0,"session_renewal_count":1})
        assert result["state"]=="complete"
        evidence=json.loads(Path(result["report"]).read_text())["page_evidence"]
        assert evidence["session_challenge_recoveries"]==1
        assert evidence["session_recovery_wait_ms"]==180000
        assert evidence["session_recovery_actual_wait_ms"] is None
        assert [e["response"]["status"] for e in queue.entries(EID) if e["request"]["url"].endswith("/shotmap")]==[403,200]


@pytest.mark.parametrize("measured", [281000, None, -1, True, "281000", float("inf"), 9_007_199_254_740_992])
def test_session_wait_measurements_are_retained_without_confusing_requested_rest(tmp_path, measured):
    config=prepare(tmp_path,policy={"collection_mode":"session"})
    with Campaign(config["directory"]) as queue:
        job=queue.next_match()
        for observation in entries(): queue.observe(EID,job["lease"],observation)
        fields=("session_recovery_actual_wait_ms", "session_expiry_actual_wait_ms",
                "session_pacing_native_waited_ms", "session_pacing_native_roundtrip_ms", "session_pacing_progress_ms")
        result=queue.finish(EID,job["lease"],page_event_id=None,page_evidence={"collection_mode":"session",
            "session_request_count":8,"session_recovery_wait_ms":180000, **dict.fromkeys(fields,measured)})
        evidence=json.loads(Path(result["report"]).read_text())["page_evidence"]
        assert evidence["session_recovery_wait_ms"]==180000
        assert all(evidence[name]==(281000 if type(measured) is int and measured==281000 else None) for name in fields)

@pytest.mark.parametrize("value", [0,-1,True,1.5,float("inf"),"50"])
def test_bad_memory_reset_interval_rejected(value):
    with pytest.raises(ValueError): validate_policy({"memory_reset_every_matches":value})

def test_update_policy_preserves_queue_and_requires_writer_disconnect(tmp_path):
    config=prepare(tmp_path)
    path=Path(config["directory"])/"config.json"
    before=json.loads(path.read_text())
    assert validate_policy({})["memory_reset_every_matches"] is None
    lock=tmp_path/"writer.lock"
    with collector_lock(lock):
        with pytest.raises(RuntimeError,match="Close DevTools"):
            update_policy(config["campaign_id"],{"memory_reset_every_matches":50},base=tmp_path,lock_path=lock)
    assert json.loads(path.read_text())==before
    updated=update_policy(config["campaign_id"],{"memory_reset_every_matches":50},base=tmp_path,lock_path=lock)
    assert updated["memory_reset_every_matches"]==50
    after=json.loads(path.read_text())
    assert {k:v for k,v in before.items() if k!="policy"}=={k:v for k,v in after.items() if k!="policy"}
    assert json.loads(next(path.parent.glob("config.before-policy-*.json")).read_text())==before
    with Campaign(config["directory"]) as queue:
        assert queue.policy["memory_reset_every_matches"]==50
        assert queue.next_match()["event_id"]==EID

@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def reject(*args,**kwargs): raise AssertionError("The native writer must not connect to the internet")
    monkeypatch.setattr(socket,"create_connection",reject)

def test_ack_survives_disconnect_resume_and_deduplicates(tmp_path):
    config=prepare(tmp_path)
    with Campaign(config["directory"]) as queue:
        job=queue.next_match()
        assert not queue.observe(EID,job["lease"],entries()[0])["duplicate"]
        queue.pause("connection closed")
    with Campaign(config["directory"]) as queue:
        assert queue.next_match()["paused"]
        resumed=queue.next_match(resume=True)
        assert resumed["lease"]!=job["lease"]
        with pytest.raises(ValueError,match="lease"): queue.observe(EID,job["lease"],entries()[0])
        assert queue.observe(EID,resumed["lease"],entries()[0])["duplicate"]
        assert len(queue.entries(EID))==1

@pytest.mark.parametrize("historical_recovery", [True,False,"true"])
def test_complete_campaign_exports_typed_data_and_distinguishes_features(tmp_path,historical_recovery):
    config=prepare(tmp_path)
    with Campaign(config["directory"]) as queue:
        job=queue.next_match()
        for e in entries(): queue.observe(EID,job["lease"],e)
        assert queue.coverage(EID)["complete"]
        with pytest.raises(ValueError,match="identity"): queue.finish(EID,job["lease"],page_event_id=999)
        result=queue.finish(EID,job["lease"],page_event_id=EID,
                            page_evidence={"memory_reset_before_match":True,"historical_link_recovery":historical_recovery})
        assert result["state"]=="complete"
        assert result["feature_coverage"]["features"]["normalized_standings"]["status"]=="not_verified"
        report=json.loads(Path(result["report"]).read_text())
        assert report["page_evidence"]["memory_reset_before_match"] is True
        assert report["page_evidence"]["historical_link_recovery"] is (historical_recovery is True)
        tables=read_tables(report["manifest"])
        assert tables["shots"] and tables["lineup_players"] and "comments" not in tables
        assert queue.next_match()["done"]
    for path in Path(config["directory"]).rglob("*"):
        if path.is_file(): assert b"SECRET_SENTINEL" not in path.read_bytes()

@pytest.mark.parametrize("metadata",[entry(code=404),entry(),entry(payload={"event":{"id":999}})])
def test_missing_invalid_or_404_identity_is_never_complete(tmp_path,metadata):
    config=prepare(tmp_path)
    with Campaign(config["directory"]) as queue:
        job=queue.next_match()
        for e in [metadata,*entries()[1:]]: queue.observe(EID,job["lease"],e)
        assert not queue.coverage(EID)["core_complete"]
        result=queue.finish(EID,job["lease"],page_event_id=EID)
        assert result["state"]=="needs_attention" and result["pause_requested"]
        assert queue.next_match(resume=True)["event_id"]==EID

def test_missing_pregame_is_partial_not_claimed_normalized(tmp_path):
    config=prepare(tmp_path)
    with Campaign(config["directory"]) as queue:
        job=queue.next_match()
        for e in entries():
            if "pregame-form" not in e["request"]["url"]: queue.observe(EID,job["lease"],e)
        result=queue.finish(EID,job["lease"],page_event_id=EID)
        assert result["state"]=="partial" and not result["pause_requested"]
        assert queue.next_match()["status"]["state"]=="finished_with_gaps"


def awarded_entries(awarded=True):
    payload=deepcopy(fixture("event"))
    payload["event"]["isAwarded"]=awarded
    return [entry(payload=payload),entry("/pregame-form",code=404),
            entry("/lineups",code=404),entry("/incidents",fixture("incidents"))]


def test_awarded_fixture_without_views_is_deferred_without_faking_completion(tmp_path):
    config=prepare(tmp_path)
    with Campaign(config["directory"]) as queue:
        job=queue.next_match()
        for observation in awarded_entries(): queue.observe(EID,job["lease"],observation)
        result=queue.finish(EID,job["lease"],page_event_id=EID,
                            page_evidence={"available_views":["standings","matches"],"recovery_refreshed":True})
        assert result["state"]=="deferred" and not result["pause_requested"]
        report=json.loads(Path(result["report"]).read_text(encoding="utf-8"))
        assert report["complete"] is False
        assert report["collection_disposition"]["state"]=="deferred"
        assert report["page_evidence"]=={"available_views":["standings","matches"],"recovery_refreshed":True,"memory_reset_before_match":False}
        assert len(report["collection_disposition"]["missing_paths"])==4
        assert all(row["status"]=="not_observed" for row in report["coverage"]
                   if row["component"] in {"statistics","shotmap","heatmap"})
        tables=read_tables(report["manifest"],verify_hashes=True)
        assert "shots" not in tables and "heatmap_points" not in tables
        resumed=queue.next_match(resume=True)
        assert resumed["done"] and resumed["status"]["state"]=="finished_with_gaps"
        assert resumed["status"]["counts"]=={"deferred":1}


@pytest.mark.parametrize("awarded,evidence",[
    (False,{"available_views":["standings","matches"],"recovery_refreshed":True}),
    (True,None),
    (True,{"available_views":["standings","matches"],"recovery_refreshed":False}),
    (True,{"available_views":["standings","statistics"],"recovery_refreshed":True}),
    (True,{"available_views":[],"recovery_refreshed":True}),
    (True,{"available_views":["standings",{}],"recovery_refreshed":True}),
])
def test_awarded_deferral_requires_fresh_status_and_absent_views_after_recovery(tmp_path,awarded,evidence):
    config=prepare(tmp_path)
    with Campaign(config["directory"]) as queue:
        job=queue.next_match()
        for observation in awarded_entries(awarded): queue.observe(EID,job["lease"],observation)
        result=queue.finish(EID,job["lease"],page_event_id=EID,page_evidence=evidence)
        assert result["state"]=="needs_attention" and result["pause_requested"]
        assert result["deferral"] is None
        if isinstance(evidence,dict):
            report=json.loads(Path(result["report"]).read_text(encoding="utf-8"))
            assert "page_evidence" in report


def test_awarded_fixture_does_not_defer_provider_denials(tmp_path):
    config=prepare(tmp_path)
    with Campaign(config["directory"]) as queue:
        job=queue.next_match()
        for observation in awarded_entries(): queue.observe(EID,job["lease"],observation)
        queue.observe(EID,job["lease"],entry("/statistics",code=403,seconds=2))
        result=queue.finish(EID,job["lease"],page_event_id=EID,
                            page_evidence={"available_views":["standings","matches"],"recovery_refreshed":True})
        assert result["paused"]
        assert result["status"]["counts"]=={"paused":1}


def test_awarded_deferral_cannot_use_reference_metadata_alone(tmp_path):
    config=prepare(tmp_path)
    with Campaign(config["directory"]) as queue:
        queue.items[EID]["reference"]["event"]["isAwarded"]=True
        job=queue.next_match()
        for observation in awarded_entries()[1:]: queue.observe(EID,job["lease"],observation)
        result=queue.finish(EID,job["lease"],page_event_id=EID,
                            page_evidence={"available_views":["standings","matches"],"recovery_refreshed":True})
        assert result["state"]=="needs_attention" and result["pause_requested"]


def test_awarded_deferral_does_not_hide_missing_incidents(tmp_path):
    config=prepare(tmp_path)
    with Campaign(config["directory"]) as queue:
        job=queue.next_match()
        for observation in awarded_entries()[:-1]: queue.observe(EID,job["lease"],observation)
        result=queue.finish(EID,job["lease"],page_event_id=EID,
                            page_evidence={"available_views":["standings","matches"],"recovery_refreshed":True})
        assert result["state"]=="needs_attention" and result["pause_requested"]

@pytest.mark.parametrize("code", [403,429])
def test_later_block_and_bodyless_response_cannot_use_older_success(tmp_path,code):
    config=prepare(tmp_path)
    with Campaign(config["directory"]) as queue:
        job=queue.next_match()
        for e in entries(): queue.observe(EID,job["lease"],e)
        queue.observe(EID,job["lease"],entry("/shotmap",code=code,seconds=2))
        assert queue.coverage(EID)["blocked"]
        assert queue.finish(EID,job["lease"],page_event_id=EID)["paused"]
        resumed=queue.next_match(resume=True)
        queue.observe(EID,resumed["lease"],entry("/shotmap",seconds=3))
        assert not queue.coverage(EID)["complete"]

def test_resumed_saved_denial_needs_a_newer_response_and_retains_evidence(tmp_path):
    config=prepare(tmp_path)
    with Campaign(config["directory"]) as queue:
        job=queue.next_match()
        queue.observe(EID,job["lease"],entry("/shotmap",code=403,seconds=2))
        queue.pause("provider_block")
    with Campaign(config["directory"]) as queue:
        job=queue.next_match(resume=True)
        queue.observe(EID,job["lease"],entry("/shotmap",fixture("shotmap"),seconds=1))
        coverage=queue.progress(EID,job["lease"])
        blocked=next(c for c in coverage["components"] if c["component"]=="shotmap")
        assert coverage["blocked"]
        assert blocked["http_status"]==403 and blocked["observed_at"]=="2026-09-12T02:00:02Z"
        queue.observe(EID,job["lease"],entry("/shotmap",fixture("shotmap"),seconds=3))
        assert not queue.progress(EID,job["lease"])["blocked"]
        assert [e["response"]["status"] for e in queue.entries(EID)]==[304,403,304]

def test_disabled_lineups_preserves_absences_without_roster(tmp_path):
    config=prepare(tmp_path,profile=replace(PROFILE,include_lineups=False))
    with Campaign(config["directory"]) as queue:
        job=queue.next_match()
        assert any(p.endswith("/lineups") for p in job["paths"])
        for e in entries(): queue.observe(EID,job["lease"],e)
        result=queue.finish(EID,job["lease"],page_event_id=EID)
        report=json.loads(Path(result["report"]).read_text())
        assert "lineup_players" not in report["table_rows"]
        assert result["feature_coverage"]["features"]["lineups"]["status"]=="not_requested"

def test_native_writer_independently_strips_commentary_queries_and_unselected_bodies(tmp_path):
    config=prepare(tmp_path)
    with Campaign(config["directory"]) as queue:
        job=queue.next_match()
        for suffix in ("/comments","/shotmap?token=SECRET_SENTINEL"):
            queue.observe(EID,job["lease"],entry(suffix,{"value":"SECRET_SENTINEL"}))
        assert "SECRET_SENTINEL" not in json.dumps(queue.entries(EID))
        bad=entry(); bad["request"]["url"]="https://evil.example/api/v1/event/1"
        with pytest.raises(ValueError,match="Sofascore"): queue.observe(EID,job["lease"],bad)

def test_pause_and_budgets_are_durable_local_controls(tmp_path):
    config=prepare(tmp_path,policy={"max_api_observations_per_run":1})
    with Campaign(config["directory"]) as queue:
        job=queue.next_match()
        result=queue.observe(EID,job["lease"],entries()[0])
        assert result["pause_requested"] and result["run_limit_reached"]
        assert queue.progress(EID,job["lease"])["run_limit_reached"]
        request_pause(config["campaign_id"],base=tmp_path)
        assert queue.progress(EID,job["lease"])["pause_requested"]

@pytest.mark.parametrize("policy", [{},{"max_api_observations_per_run":None}])
def test_unlimited_run_passes_old_cap_but_honors_manual_pause(tmp_path,policy):
    config=prepare(tmp_path,policy=policy)
    with Campaign(config["directory"]) as queue:
        assert queue.policy["max_api_observations_per_run"] is None
        job=queue.next_match()
        queue.run_observations=399  # Prior observations from this connection.
        for e in entries()[:3]:
            assert not queue.observe(EID,job["lease"],e)["pause_requested"]
            assert not queue.progress(EID,job["lease"])["run_limit_reached"]
        assert queue.run_observations==402
        request_pause(config["campaign_id"],base=tmp_path)
        result=queue.observe(EID,job["lease"],entries()[3])
        assert result["pause_requested"] and not result["run_limit_reached"]
        progress=queue.progress(EID,job["lease"])
        assert progress["pause_requested"] and not progress["run_limit_reached"]

@pytest.mark.parametrize("value", [0,-1,1.5,float("nan"),True,"400"])
def test_invalid_run_limit_rejected(value):
    with pytest.raises(ValueError): validate_policy({"max_api_observations_per_run":value})

@pytest.mark.parametrize("value",[-1,float("nan"),True,"5"])
def test_bad_jitter_rejected(value):
    with pytest.raises(ValueError): validate_policy({"random_action_delay_max_seconds":value})

def test_native_framing_handles_unicode_fragmentation_eof_and_limits():
    encoded=io.BytesIO(); write_message(encoded,{"hello":"Mâ€™gladbach"})
    class Fragmented(io.BytesIO):
        def read(self,n=-1): return super().read(min(2,n))
    assert read_message(Fragmented(encoded.getvalue()))=={"hello":"Mâ€™gladbach"}
    assert read_message(io.BytesIO()) is None
    for raw in (b"\x01",struct.pack("=I",0),struct.pack("=I",MAX_INPUT+1),struct.pack("=I",10)+b"{}",struct.pack("=I",2)+b"[]"):
        with pytest.raises(ValueError): read_message(io.BytesIO(raw))

def test_host_whitelist_lock_and_disconnect_pause(tmp_path):
    config=prepare(tmp_path)
    host=Host(base=tmp_path,lock_path=tmp_path/"shared.lock")
    try:
        with pytest.raises(ValueError): host.dispatch({"operation":"hello","protocol":99})
        with pytest.raises(ValueError): host.dispatch({"operation":"open","campaign_id":"../escape"})
        host.dispatch({"operation":"open","campaign_id":config["campaign_id"]})
        host.dispatch({"operation":"next"})
        with pytest.raises((OSError,BlockingIOError)):
            with collector_lock(tmp_path/"shared.lock"): pass
    finally: host.close()
    assert json.loads((Path(config["directory"])/"status.json").read_text())["state"]=="paused"
    with collector_lock(tmp_path/"shared.lock"): pass

def test_native_stream_is_only_framed_messages(tmp_path):
    source=io.BytesIO(); output=io.BytesIO()
    write_message(source,{"id":1,"operation":"hello","protocol":1}); source.seek(0)
    serve(source,output,base=tmp_path,lock_path=tmp_path/"lock")
    output.seek(0)
    assert read_message(output)=={"id":1,"ok":True,"result":{"protocol":1,"version":"0.4.1","host":"com.xdiyo.sofascore"}}
    assert read_message(output) is None
