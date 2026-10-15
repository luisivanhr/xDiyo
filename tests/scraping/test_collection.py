"""Synthetic endpoint fixtures exercise contracts, not current API coverage."""
from dataclasses import replace
import json

import pytest

from scraping.sofascore.client import matching_response
from scraping.sofascore.config import Profile, RatePolicy, Scope
from scraping.sofascore.discovery import event_status, round_jobs, unresolved_jobs
from scraping.sofascore.export import normalize_store, publish_tables, read_tables
from scraping.sofascore.models import BudgetExhausted, CollectionPaused, Job, Response
from scraping.sofascore.parsers import parse_component, validate_payload
from scraping.sofascore.planner import build_plan, component_jobs
from scraping.sofascore.rate_limit import RateLimiter, retry_after_seconds
from scraping.sofascore.runner import Runner
from scraping.sofascore.store import Store
from scraping.sofascore.timing import timing_evidence
from scraping.sofascore.goal_actions import derive_pass_edges
from utils.team_seasons import Membership, derive_team_seasons, join_team_season_flags


class FakeClock:
    def __init__(self):
        self.wall = 2000000000.0
        self.mono = 1000.0
    def time(self): return self.wall
    def monotonic(self): return self.mono
    def sleep(self, seconds):
        self.wall += seconds
        self.mono += seconds


@pytest.fixture
def clock(): return FakeClock()


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path/"cache") as value:
        yield value


@pytest.fixture
def event():
    return {"id":100,"startTimestamp":1000000000,"status":{"type":"finished","code":100},
            "homeTeam":{"id":1,"name":"Home","slug":"home"},"awayTeam":{"id":2,"name":"Away","slug":"away"},
            "season":{"id":500,"year":"2024/2025"},"roundInfo":{"round":1},
            "tournament":{"id":30,"uniqueTournament":{"id":17}},
            "homeScore":{"display":1,"period1":0,"period2":1},"awayScore":{"display":0,"period1":0,"period2":0}}


def payload(component, event):
    return {"discovery":{"events":[event]},"event":{"event":event},
            "pregame":{"homeTeam":{"position":1},"awayTeam":{"position":2}},
            "statistics":{"statistics":[]},"shotmap":{"shotmap":[]},
            "heatmap":{"playerPoints":[{"x":1.5,"y":2}],"goalkeeperPoints":[]},
            "lineups":{"home":{"players":[],"missingPlayers":[]},"away":{"players":[]}},
            "incidents":{"incidents":[]},"comments":{"comments":[]}}[component]


class FakeClient:
    def __init__(self,event, responses=None):
        self.event,self.responses = event,responses or {}
        self.calls=[]
        self.closed=False
    def fetch(self,job):
        self.calls.append(job)
        values = self.responses.get(job.path)
        if values:
            return values.pop(0)
        return Response(200,payload(job.component,self.event))
    def close(self): self.closed=True


def seed(store,event,clock):
    scope = Scope(17,500,(1,))
    job = round_jobs(scope)[0]
    store.record(job,"success",clock.time(),payload={"events":[event]})
    store.add_events([event],scope,job,clock.time())
    return scope


def test_cold_six_requests_and_dedup(store,event,clock):
    plan=build_plan([event,event],Profile(),store,clock.time())
    assert len(plan.jobs)==6
    assert len({j.key for j in plan.jobs})==6
    lineup=next(j for j in plan.jobs if j.component=="lineups")
    assert lineup.reasons==("lineups","availability")


def test_random_delay_validation():
    assert RatePolicy().random_delay_max_seconds is None
    for value in (None,0,2.5):
        assert RatePolicy(random_delay_max_seconds=value).random_delay_max_seconds==value
    for value in (-1,float("nan"),float("inf"),float("-inf"),True,"2"):
        with pytest.raises(ValueError,match="random_delay_max_seconds"):
            RatePolicy(random_delay_max_seconds=value)


def test_random_delay_draws_once_per_attempt_after_required_pacing(store,clock,monkeypatch):
    draws=[]
    values=iter([0,2.5,5])
    def uniform(low,high):
        draws.append((low,high))
        return next(values)
    monkeypatch.setattr("scraping.sofascore.rate_limit.random.uniform",uniform)
    policy=RatePolicy(per_minute=None,per_hour=None,random_delay_max_seconds=5)
    job=Job("/event/100/shotmap","shotmap",100)
    limiter=RateLimiter(store,policy,store.start_run(clock.time()),clock)
    initial=clock.time()
    attempt=limiter.reserve(job)
    assert draws==[] and clock.time()==initial
    clock.sleep(4)  # A slow response must finish before spacing starts.
    limiter.complete(attempt,200,"success")
    for random_delay in (0,2.5,5):
        before=clock.time()
        # The persisted completion also provides spacing after a restart.
        if random_delay==5:
            limiter=RateLimiter(store,policy,store.start_run(clock.time()),clock)
        attempt=limiter.reserve(job)
        assert clock.time()-before==10+random_delay
        limiter.complete(attempt,200,"success")
    assert draws==[(0,5)]*3


@pytest.mark.parametrize("maximum",[None,0])
def test_disabled_random_delay_does_not_draw_or_sleep(store,clock,monkeypatch,maximum):
    def unexpected_draw(*args):
        raise AssertionError("Random delay is disabled")
    monkeypatch.setattr("scraping.sofascore.rate_limit.random.uniform",unexpected_draw)
    policy=RatePolicy(min_interval_seconds=None,per_minute=None,per_hour=None,
                      random_delay_max_seconds=maximum)
    limiter=RateLimiter(store,policy,store.start_run(clock.time()),clock)
    job=Job("/event/100/shotmap","shotmap",100)
    before=clock.time()
    for _ in range(3):
        limiter.complete(limiter.reserve(job),200,"success")
    assert clock.time()==before


def test_random_delay_preserves_cooldown_and_block_without_reserving(store,clock,monkeypatch):
    def unexpected_draw(*args):
        raise AssertionError("Blocked requests must not draw a random delay")
    monkeypatch.setattr("scraping.sofascore.rate_limit.random.uniform",unexpected_draw)
    limiter=RateLimiter(store,RatePolicy(random_delay_max_seconds=5),store.start_run(clock.time()),clock)
    job=Job("/event/100/shotmap","shotmap",100)
    limiter.complete(limiter.reserve(job),200,"success")
    limiter.cooldown(60)
    before=clock.time()
    with pytest.raises(CollectionPaused,match="cooldown"):
        limiter.reserve(job)
    assert store.setting("cooldown_until")==before+60
    clock.sleep(60)
    store.set_setting("access_block",{"status":403})
    with pytest.raises(CollectionPaused,match="block"):
        limiter.reserve(job)
    assert store.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]==1


def test_random_delay_applies_to_retry_and_not_cached_resume(store,event,clock,monkeypatch):
    draws=[]
    def uniform(low,high):
        draws.append(high)
        return high/2
    monkeypatch.setattr("scraping.sofascore.rate_limit.random.uniform",uniform)
    scope=seed(store,event,clock)
    policy=RatePolicy(min_interval_seconds=None,per_minute=None,per_hour=None,
                      random_delay_max_seconds=4,backoff_seconds=10)
    client=FakeClient(event,{"/event/100/shotmap":[Response(503),Response(200,{"shotmap":[]})]})
    before=clock.time()
    result=Runner(store,client,policy,Profile(),clock=clock).run(scope,discover=False,only_components={"shotmap"})
    assert result["status"]=="complete" and len(client.calls)==2
    assert draws==[1,4]  # Existing retry backoff jitter, then request delay.
    assert clock.time()-before==12.5
    before=clock.time()
    result=Runner(store,FakeClient(event),policy,Profile(),clock=clock).run(
        scope,discover=False,only_components={"shotmap"})
    assert result["status"]=="complete" and clock.time()==before and draws==[1,4]


@pytest.mark.parametrize("lineups,injuries,count",[(False,False,0),(False,True,1),(True,False,1),(True,True,1)])
def test_shared_lineup_dependency(event,clock,lineups,injuries,count):
    jobs=component_jobs(event,Profile(include_lineups=lineups,include_injuries=injuries),clock.time())
    assert sum(j.component=="lineups" for j in jobs)==count


def test_shared_incidents_and_no_comments(event,clock):
    profile=Profile(duration_mode="incidents",injury_scope="availability_and_match_incidents")
    jobs=component_jobs(event,profile,clock.time())
    assert [j.reasons for j in jobs if j.component=="incidents"]==[("timing_evidence","injury_incidents")]
    assert not any(j.component in {"comments","event"} for j in jobs)


def test_unplayed_and_postponed_no_postmatch_requests(event,clock):
    event["status"]={"type":"notstarted"}
    event["startTimestamp"]=clock.time()+3600
    assert component_jobs(event,Profile(),clock.time())==[]
    assert {j.component for j in component_jobs(event,Profile(include_upcoming=True),clock.time())}=={"pregame","lineups"}
    event["status"]={"type":"postponed"}
    assert component_jobs(event,Profile(include_upcoming=True),clock.time())==[]


def test_round_one_and_explicit_gaps():
    assert len(round_jobs(Scope(17,500,(1,1))))==1
    with pytest.raises(ValueError): Scope(17,500,(0,1))


def test_cache_resume_only_failed_away_heatmap(store,event,clock):
    scope=seed(store,event,clock)
    policy=RatePolicy(max_attempts_per_job=1)
    client=FakeClient(event,{"/event/100/heatmap/2":[Response(503)]})
    report=Runner(store,client,policy,Profile(),clock=clock).run(scope,discover=False)
    assert report["status"]=="incomplete" and len(client.calls)==6 and client.closed
    next_client=FakeClient(event)
    report=Runner(store,next_client,policy,Profile(),clock=clock).run(scope,discover=False)
    assert report["status"]=="complete"
    assert [j.path for j in next_client.calls]==["/event/100/heatmap/2"]
    assert not build_plan([event],Profile(),store,clock.time()).jobs


def test_selected_refresh_does_not_fetch_other_missing_components(store,event,clock):
    scope=seed(store,event,clock)
    client=FakeClient(event)
    report=Runner(store,client,RatePolicy(),Profile(),clock=clock).run(scope,discover=False,
        only_components={"shotmap"},refresh_components={"shotmap"},event_ids={100})
    assert [j.component for j in client.calls]==["shotmap"]
    assert report["component_selection"]==["shotmap"]


def test_failed_refresh_preserves_last_good(store,event,clock):
    job=Job("/event/100/shotmap","shotmap",100)
    store.record(job,"success",clock.time(),payload={"shotmap":[{"xg":0}]})
    original=store.get(job)
    clock.sleep(60)
    store.record(job,"failed",clock.time(),http_status=503)
    assert store.payload(job)=={"shotmap":[{"xg":0}]}
    assert store.get(job)["last_success"]==original["last_success"]
    assert store.due(job,clock.time())


def test_persisted_hour_budget_and_monotonic_spacing(store,clock):
    policy=RatePolicy(per_hour=2,per_minute=2,min_interval_seconds=10,max_attempts_per_run=5)
    job=Job("/event/100/statistics","statistics",100)
    start=clock.time()
    limiter=RateLimiter(store,policy,store.start_run(clock.time()),clock)
    limiter.reserve(job); limiter.reserve(job)
    assert clock.time()==start+10
    restarted=RateLimiter(store,policy,store.start_run(clock.time()),clock)
    restarted.reserve(job)
    assert clock.time()==start+3600


def test_budget_counts_attempts_including_retries(store,event,clock):
    scope=seed(store,event,clock)
    client=FakeClient(event,{"/event/100/pregame-form":[Response(503),Response(503)]})
    report=Runner(store,client,RatePolicy(max_attempts_per_run=2),Profile(),clock=clock).run(scope,discover=False)
    assert report["status"]=="paused" and report["attempts"]==2 and len(client.calls)==2


def test_429_survives_restart_and_stops_all_jobs(store,event,clock):
    scope=seed(store,event,clock)
    client=FakeClient(event,{"/event/100/pregame-form":[Response(429,headers={"retry-after":"600"})]})
    report=Runner(store,client,RatePolicy(),Profile(),clock=clock).run(scope,discover=False)
    assert report["status"]=="paused" and len(client.calls)==1
    next_client=FakeClient(event)
    next_report=Runner(store,next_client,RatePolicy(),Profile(),clock=clock).run(scope,discover=False)
    assert next_report["status"]=="paused" and next_client.calls==[]
    assert store.setting("cooldown_until")==clock.time()+600


def test_403_latches_and_client_closes(store,event,clock):
    scope=seed(store,event,clock)
    client=FakeClient(event,{"/event/100/pregame-form":[Response(403)]})
    report=Runner(store,client,RatePolicy(),Profile(),clock=clock).run(scope,discover=False)
    assert report["status"]=="paused" and client.closed
    assert store.setting("access_block")


def test_os_lock_rejects_second_collector(store):
    with store.run_lock():
        with pytest.raises(RuntimeError):
            with store.run_lock(): pass
    with store.run_lock(): pass


def test_response_association_rejects_previous_navigation():
    url="https://www.sofascore.com/api/v1/event/2/statistics"
    old=url.replace("/2/","/1/")
    logs=[{"method":"Network.requestWillBeSent","params":{"requestId":"new","request":{"url":url}}},
          {"method":"Network.responseReceived","params":{"requestId":"old","response":{"url":old,"status":200}}}]
    assert matching_response(logs,url)==(None,0)


def test_pacing_includes_slow_browser_start(store,clock):
    limiter=RateLimiter(store,RatePolicy(),store.start_run(clock.time()),clock)
    job=Job("/event/100/statistics","statistics",100)
    attempt=limiter.reserve(job)
    clock.sleep(100)
    limiter.complete(attempt,200,"success")
    finished=clock.time()
    limiter.reserve(job)
    assert clock.time()>=finished+10


def test_lineup_snapshots_keep_observation_phase(store,event,clock):
    scope=seed(store,event,clock)
    job=Job("/event/100/lineups","lineups",100)
    pre={**event,"status":{"type":"notstarted"}}
    data=payload("lineups",event)
    store.record(job,"success",clock.time(),payload=data,event_context=pre)
    clock.sleep(100)
    store.record(job,"success",clock.time(),payload=data,event_context=event)
    tables=normalize_store(store,scope,Profile(),now=clock.time())
    assert {r["phase"] for r in tables["lineup_teams"]}=={"pregame","postgame"}
    assert len(store.snapshots(job))==2


def test_finished_status_invalidates_pregame_lineup_cache(store,event,clock):
    pre={**event,"status":{"type":"notstarted"}}
    scope=seed(store,pre,clock)
    job=Job("/event/100/lineups","lineups",100)
    store.record(job,"success",clock.time(),payload=payload("lineups",event),next_due=clock.time()+1000)
    clock.sleep(10)
    store.add_events([event],scope,round_jobs(scope)[0],clock.time())
    assert store.due(job,clock.time())


def test_shots_keep_missing_xg_zero_and_own_goal_metadata(event,clock):
    job=Job("/event/100/shotmap","shotmap",100)
    data={"shotmap":[{"id":10,"isHome":True,"shotType":"miss"},
                     {"id":11,"isHome":False,"xg":0,"xgot":0,"shotType":"goal","goalType":"own"}]}
    rows=parse_component(job,data,event,Profile(),observed_at=clock.time(),raw_hash="fixture")["shots"]
    assert len(rows)==2 and rows[0]["xg"] is None and rows[0]["xgot"] is None
    assert rows[0]["shot_type"]=="miss" and rows[1]["xg"]==0 and rows[1]["goal_type"]=="own"


def test_statistics_periods_use_labels(event,clock):
    job=Job("/event/100/statistics","statistics",100)
    data={"statistics":[{"period":p,"groups":[{"statisticsItems":[{"key":"shots","homeValue":n,"awayValue":0}]}]}
                        for p,n in [("2ND",7),("1ST",3),("EXTRA1",2)]]}
    rows=parse_component(job,data,event,Profile(),observed_at=clock.time(),raw_hash="fixture")["statistics"]
    assert [(r["period"],r["value"]) for r in rows if r["side"]=="home"]==[("2ND",7),("1ST",3),("EXTRA1",2)]


def test_lineup_without_formation_and_independent_injury_flag(event,clock):
    data={"home":{"players":[{"player":{"id":99,"position":"G","slug":"keeper"}}],
                  "missingPlayers":[{"player":{"id":98},"reason":3,"type":"doubtful"}]},
          "away":{"players":[],"formation":"4-4-2"}}
    job=Job("/event/100/lineups","lineups",100)
    parsed=parse_component(job,data,event,Profile(),observed_at=clock.time(),raw_hash="fixture")
    assert parsed["lineup_players"][0]["player_id"]==99
    assert parsed["lineup_teams"][0]["formation"] is None
    assert parsed["availability"][0]["classification"]=="unclassified_reported_absence"
    assert any(r["component"]=="availability" and r["side"]=="away" and r["status"]=="unavailable" for r in parsed["coverage"])
    parsed=parse_component(job,data,event,Profile(include_lineups=False),observed_at=clock.time(),raw_hash="fixture")
    assert "lineup_players" not in parsed and "lineup_teams" not in parsed and "availability" in parsed


def test_timing_sentinel_and_announcements_never_exact(event):
    event["time"]={"injuryTime1":1,"injuryTime2":7}
    rows=timing_evidence(event,{"incidents":[{"incidentType":"period","text":"FT","time":90,"addedTime":999}]})
    assert all(r["elapsed_seconds"] is None for r in rows)
    assert rows[-1]["quality"]=="invalid_or_sentinel"
    assert rows[1]["candidate_added_minutes"]==7 and rows[1]["announced_added_minutes"] is None


def test_goal_edges_require_explicit_recipient_and_completed_pass():
    base={"event_id":100,"goal_id":1,"actor_id":9,"action_type":"pass"}
    actions=[{**base,"action_order":0,"completed":True},
             {**base,"action_order":1,"recipient_id":10,"completed":False},
             {**base,"action_order":2,"recipient_id":10,"completed":True}]
    assert derive_pass_edges(actions)==[{"event_id":100,"goal_id":1,"action_order":2,"from_player_id":9,"to_player_id":10}]


def test_parquet_export_nulls_optional_lineups_and_no_network(store,event,clock,tmp_path):
    scope=seed(store,event,clock)
    profile=Profile(include_lineups=False)
    client=FakeClient(event)
    Runner(store,client,RatePolicy(),profile,clock=clock).run(scope,discover=False)
    tables=normalize_store(store,scope,profile,now=clock.time())
    assert "lineup_players" not in tables and "lineup_teams" not in tables
    manifest=publish_tables(tables,tmp_path/"prepared",scope=scope,profile=profile)
    loaded=read_tables(manifest)
    assert loaded["matches"][0]["home_score_period2"]==1
    assert loaded["matches"][0]["home_score_extra1"] is None
    assert loaded["heatmap_points"][0]["x"]==1.5
    assert loaded["timing"][0]["elapsed_seconds"] is None


def test_failed_publication_preserves_active_version(tmp_path):
    root=tmp_path/"prepared"
    scope=Scope(17,500,(1,))
    manifest=publish_tables({"matches":[{"event_id":100}]},root,scope=scope)
    pointer=manifest.parents[2]/"CURRENT.json"
    before=pointer.read_bytes()
    with pytest.raises(ValueError):
        publish_tables({"matches":[{"event_id":100},{"event_id":100}]},root,scope=scope)
    assert pointer.read_bytes()==before
    assert read_tables(pointer)["matches"][0]["event_id"]==100


def member(comp,season,year,tier,teams,complete=True):
    return Membership(comp,season,year,year+1,tier,frozenset(teams),complete,"test membership")


def test_promotion_id_transitions_and_unknown_newcomer():
    data=[member(1,10,2023,1,[1,2]),member(2,20,2023,2,[3,4]),
          member(1,11,2024,1,[1,3]),member(2,21,2024,2,[2,4,5])]
    rows=derive_team_seasons(data)
    actual={(r["season_id"],r["team_id"]):r for r in rows}
    assert actual[11,3]["movement"]=="promoted"
    assert actual[21,2]["movement"]=="relegated"
    assert actual[11,1]["movement"]=="retained"
    assert actual[21,5]["movement"]=="unknown" and actual[21,5]["got_promoted"] is None


def test_promotion_gaps_incomplete_and_seed_binding():
    data=[member(1,10,2022,1,[1]),member(1,12,2024,1,[2]),member(2,21,2023,2,[2])]
    assert next(r for r in derive_team_seasons(data) if r["season_id"]==12)["movement"]=="unknown"
    override={"competition_id":1,"season_id":12,"team_id":2,"movement":"other_entry","evidence":"explicit case"}
    rows=derive_team_seasons(data,[override])
    assert next(r for r in rows if r["season_id"]==12)["movement"]=="other_entry"
    with pytest.raises(ValueError): derive_team_seasons(data,[{**override,"season_id":99}])
    partial=[member(1,10,2023,1,[1],False),member(1,11,2024,1,[2]),member(2,20,2023,2,[2])]
    assert next(r for r in derive_team_seasons(partial) if r["season_id"]==11)["movement"]=="unknown"


def test_flag_join_uses_ids_preserves_rows_and_unknown():
    import pandas as pd
    rows=derive_team_seasons([member(1,10,2023,1,[1]),member(1,11,2024,1,[1])])
    matches=pd.DataFrame([{"event_id":3,"competition_id":1,"season_id":11,"home_id":1,"away_id":99}])
    result=join_team_season_flags(matches,rows)
    assert len(result)==1 and result.iloc[0].home_got_promoted==False
    assert pd.isna(result.iloc[0].away_got_promoted)
    with pytest.raises(ValueError): join_team_season_flags(matches,rows+rows)


def test_pilot_budget_and_cohort_survive_restarts(store,event,clock):
    prior = store.start_run(clock.time())
    store.set_setting("pilot_run_ids",[prior])
    store.set_setting("pilot_event_ids",[100])
    with store.db:
        for _ in range(39):
            store.db.execute("INSERT INTO attempts(run_id,key,started,outcome) VALUES (?,?,?,?)",
                             (prior,"old",clock.time()-7200,"success"))
    client = FakeClient(event)
    report = Runner(store,client,RatePolicy(),Profile(),clock=clock).run(Scope(17,500,(1,)),pilot=True,event_ids={100})
    assert report["attempts"] == 1 and report["status"] == "paused"
    assert len(client.calls) == 1
    with pytest.raises(CollectionPaused,match="40-attempt"):
        Runner(store,FakeClient(event),RatePolicy(),Profile(),clock=clock).run(Scope(17,500,(1,)),pilot=True,event_ids={100})
    with pytest.raises(ValueError,match="eight events"):
        Runner(store,FakeClient(event),RatePolicy(),Profile(),clock=clock).run(Scope(17,500,(1,)),pilot=True,event_ids=set(range(1,9)))


def test_schema_diagnostic_preserves_previous_success(store,event,clock):
    scope = seed(store,event,clock)
    job = Job("/event/100/shotmap","shotmap",100)
    store.record(job,"success",clock.time(),payload={"shotmap":[]})
    client = FakeClient(event,{job.path:[Response(200,{"newShape":[]})]})
    report = Runner(store,client,RatePolicy(),Profile(),clock=clock).run(scope,discover=False,
        event_ids={100},only_components={"shotmap"},refresh_components={"shotmap"})
    assert report["status"] == "incomplete" and report["attempts"] == 1
    assert store.payload(job) == {"shotmap":[]}
    assert "diagnostic:" in store.get(job)["error"]
    assert len(list((store.root/"diagnostics").glob("*.json.gz"))) == 1


def test_stage_duplicates_and_postponed_fixture_outside_current_round(store,event,clock):
    old = {**event,"status":{"type":"postponed"},"roundInfo":{"round":1,"name":"First stage"}}
    scope = seed(store,old,clock)
    clock.sleep(21601)
    later = Scope(17,500,(12,))
    assert [j.event_id for j in unresolved_jobs(store,later,Profile(),clock.time())] == [100]
    stage = {**event,"roundInfo":{"round":12,"name":"Second stage"}}
    store.add_events([stage],later,round_jobs(later)[0],clock.time())
    assert len(store.events(17,500)) == 1
    assert store.db.execute("SELECT COUNT(*) FROM memberships WHERE event_id=100").fetchone()[0] == 2
    assert store.events(17,500)[0]["roundInfo"]["name"] == "Second stage"
    assert unresolved_jobs(store,later,Profile(),clock.time()) == []
