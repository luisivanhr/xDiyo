import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from scraping.sofascore.campaign import CampaignRunner, POLICY, PROFILE, inventory
from scraping.sofascore.config import RatePolicy, Scope
from scraping.sofascore.legacy import load_season
from scraping.sofascore.models import CollectionPaused, Job, Response
from scraping.sofascore.rate_limit import RateLimiter
from scraping.sofascore.store import Store


EVENT = {"id":100,"startTimestamp":1720000000,"status":{"type":"finished"},
         "season":{"id":500,"year":"24/25"},"roundInfo":{"round":1},
         "tournament":{"id":1,"uniqueTournament":{"id":17}},
         "homeTeam":{"id":1,"name":"H"},"awayTeam":{"id":2,"name":"A"},
         "homeScore":{"current":0},"awayScore":{"current":0}}


class Client:
    def __init__(self,block=None): self.calls=[]; self.block=block
    def close(self): pass
    def fetch(self,job):
        self.calls.append(job)
        if job.component==self.block:
            return Response(403)
        if job.component=="pregame":
            return Response(404)
        payloads={"discovery":{"events":[EVENT]},"statistics":{"statistics":[]},
                  "shotmap":{"shotmap":[{"id":4,"isHome":True,"shotType":"save","time":7}]},
                  "heatmap":{"playerPoints":[{"x":1,"y":2}],"goalkeeperPoints":[]},
                  "lineups":{"home":{"players":[],"formation":"4-4-2"},"away":{"players":[]}},
                  "incidents":{"incidents":[]}}
        return Response(200,payloads[job.component])


def item():
    return {"stem":"Premier_League_24_25","source_sha256":"test","league_id":17,
            "season_id":500,"rounds":[1],"league_name":"Premier_League",
            "season_start":2024,"season_end":2025,"legacy_event_ids":[100],
            "legacy_rows":1,"legacy_identity_valid":True}


def runner(store,client,tmp_path):
    return CampaignRunner(store,client,POLICY,PROFILE,progress=lambda **kw:None,stop_path=tmp_path/"STOP")


def test_campaign_resumes_without_duplicate_requests_and_preserves_nested_data(tmp_path):
    output=tmp_path/"out"; output.mkdir()
    with Store(tmp_path/"store") as store:
        first=Client("shotmap")
        with pytest.raises(CollectionPaused): runner(store,first,tmp_path).collect_season(item(),output)
        assert not (output/"Premier_League_24_25.parquet").exists()
        assert store.setting("access_block")["status"]==403
        store.set_setting("access_block",None)  # test simulates separately authorized recovery
        second=Client()
        report=runner(store,second,tmp_path).collect_season(item(),output)
        assert not any(j.component in {"discovery","pregame","statistics"} for j in second.calls)
        assert not any(j.component=="comments" for j in first.calls+second.calls)
        path=output/"Premier_League_24_25.parquet"
        row=pq.ParquetFile(path).read().to_pylist()[0]
        assert row["shots"][0]["xg"] is None
        assert row["lineup_teams"][0]["formation"]=="4-4-2"
        assert len(row["heatmap_points"])==2
        assert report["matches"]==1 and report["shots_with_xg"]==0
        assert load_season(path).match_id.tolist()==[100]
        third=Client()
        runner(store,third,tmp_path).collect_season(item(),output)
        assert third.calls==[]


def test_disabled_campaign_caps_still_enforce_cooldown_and_access_block(tmp_path):
    assert RatePolicy().per_hour==120 and POLICY.per_hour is None
    with Store(tmp_path/"store") as store:
        run=store.start_run()
        limiter=RateLimiter(store,POLICY,run)
        job=Job("/event/100/shotmap","shotmap",100)
        for _ in range(45): limiter.complete(limiter.reserve(job),200,"success")
        limiter.cooldown(60)
        with pytest.raises(CollectionPaused,match="cooldown"): limiter.reserve(job)
        store.set_setting("cooldown_until",0)
        store.set_setting("access_block",{"status":403})
        with pytest.raises(CollectionPaused,match="block"): limiter.reserve(job)


def test_bad_legacy_label_is_resolved_instead_of_copying_wrong_season(tmp_path):
    (tmp_path/"Eredivisie_16_17.csv").write_text(
        "match_id,season_id,season_year,round\n1,10370,15/16,1\n",encoding="utf-8")
    entry=inventory(tmp_path)[0]
    assert entry["season_id"] is None and entry["legacy_event_ids"]==[]
    assert entry["season_start"]==2016


def test_store_rejects_wrong_year_even_when_ids_match(tmp_path):
    with Store(tmp_path/"store") as store:
        scope=Scope(17,500,(1,),"Premier_League",2023,2024)
        with pytest.raises(ValueError,match="year"):
            store.add_events([EVENT],scope,Job("/round/1","discovery"),0)


def test_access_denial_retains_server_retry_after(tmp_path):
    class Denied(Client):
        def fetch(self,job): return Response(403,headers={"retry-after":"600"})
    with Store(tmp_path/"store") as store:
        with pytest.raises(CollectionPaused):
            runner(store,Denied(),tmp_path).collect_season(item(),tmp_path)
        assert store.setting("cooldown_until") > store.setting("last_request_finished")+590


def campaign_args(tmp_path):
    source=tmp_path/"source"; source.mkdir()
    (source/"Premier_League_24_25.csv").write_text(
        "match_id,season_id,season_year,round\n100,500,24/25,1\n",encoding="utf-8")
    return ["--source",str(source),"--output",str(tmp_path/"out"),
            "--store",str(tmp_path/"store"),"--pause-on-block"]


def test_manual_campaign_returns_on_block_and_retries_only_when_requested(tmp_path,monkeypatch):
    from scraping.sofascore import campaign
    args=campaign_args(tmp_path)+["--random-delay-max-seconds","2"]
    monkeypatch.setattr("scraping.sofascore.rate_limit.random.uniform",lambda low,high:0)
    clients=[]
    def factory(*args,**kwargs):
        assert args[0].random_delay_max_seconds==2
        client=Client("shotmap" if not clients else None)
        clients.append(client)
        return client
    monkeypatch.setattr(campaign,"BrowserClient",factory)
    updates=[]
    with pytest.raises(CollectionPaused,match="Access blocked"):
        campaign.main(args,on_progress=updates.append)
    assert updates[-1]["state"]=="paused"
    assert updates[-1]["rate_policy"]["random_delay_max_seconds"]==2
    assert POLICY.random_delay_max_seconds is None
    assert not (tmp_path/"out/Premier_League_24_25.parquet").exists()
    with pytest.raises(CollectionPaused,match="Access blocked"):
        campaign.main(args)
    assert len(clients)==1
    assert campaign.main(args+["--retry-blocked"])==0
    assert not any(j.component in {"discovery","pregame","statistics"} for j in clients[1].calls)
    with Store(tmp_path/"store") as store:
        assert store.setting("last_manual_access_retry")["previous_block"]["status"]==403
    def unexpected_wait(seconds):
        raise AssertionError("An already completed selection should return immediately")
    monkeypatch.setattr(campaign.time,"sleep",unexpected_wait)
    assert campaign.main(args)==0
    assert len(clients)==2


def test_manual_resume_retains_cooldown_and_requires_collector_locks(tmp_path,monkeypatch):
    from scraping.sofascore import campaign
    args=campaign_args(tmp_path)
    control=tmp_path/"out/_collection"; control.mkdir(parents=True)
    stop=control/"STOP"; stop.write_text("paused")
    original_status='{"state":"running","pid":123}'
    (control/"status.json").write_text(original_status)
    with Store(control/"lock") as lock, lock.run_lock():
        with pytest.raises(RuntimeError,match="Another collector"):
            campaign.main(args+["--resume","--retry-blocked"])
    assert stop.exists()
    assert (control/"status.json").read_text()==original_status
    until=campaign.time.time()+3600
    with Store(tmp_path/"store") as store:
        store.set_setting("access_block",{"status":403})
        store.set_setting("cooldown_until",until)
    def interrupt_wait(seconds):
        raise KeyboardInterrupt
    monkeypatch.setattr(campaign.time,"sleep",interrupt_wait)
    with pytest.raises(KeyboardInterrupt):
        campaign.main(args+["--resume","--retry-blocked"])
    assert not stop.exists()
    with Store(tmp_path/"store") as store:
        assert store.setting("cooldown_until")==until
        assert store.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]==0
    assert json.loads((control/"status.json").read_text())["state"]=="paused"


class ManualClient(Client):
    def __init__(self,block=None,cancel=False,requests=None):
        super().__init__(block)
        self.cancel=cancel
        self.requests=requests or []
        self.confirmations=0
        self.closes=0
    def prepare_manual_access(self,*,on_requests):
        self.confirmations+=1
        on_requests(self.requests)
        if self.cancel:
            raise KeyboardInterrupt
    def fetch(self,job):
        assert self.confirmations==1
        response=super().fetch(job)
        if job.component=="discovery" and "/season/501/" in job.path:
            response.payload["events"]=[{**EVENT,"id":101,"season":{"id":501,"year":"23/24"}}]
        return response
    def close(self):
        self.closes+=1


def test_captcha_browser_is_reused_across_seasons_and_closes_at_end(tmp_path,monkeypatch):
    from scraping.sofascore import campaign
    args=campaign_args(tmp_path)+["--manual-captcha"]
    (tmp_path/"source/Premier_League_23_24.csv").write_text(
        "match_id,season_id,season_year,round\n101,501,23/24,1\n",encoding="utf-8")
    client=ManualClient(requests=[{"path":"/sport/football/events","status":200,"retry_after":None}])
    created=[]
    def factory(*args,**kwargs):
        created.append(client)
        return client
    monkeypatch.setattr(campaign,"BrowserClient",factory)
    now=[campaign.time.time()]
    monkeypatch.setattr(campaign.time,"time",lambda:now[0])
    monkeypatch.setattr(campaign.time,"sleep",lambda seconds:now.__setitem__(0,now[0]+seconds))
    with Store(tmp_path/"store") as store:
        store.set_setting("access_block",{"status":403})
    assert campaign.main(args)==0
    assert len(created)==1 and client.confirmations==1 and client.closes==1
    assert len(client.calls)==16
    for stem in ("Premier_League_24_25","Premier_League_23_24"):
        assert (tmp_path/"out"/(stem+".parquet")).exists()
    with Store(tmp_path/"store") as store:
        assert store.setting("access_block") is None
        assert store.setting("last_manual_access_confirmation")["previous_block"]["status"]==403
        assert store.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]==17
    assert campaign.main(args)==0
    assert len(created)==1  # Already complete selections never open the homepage.


@pytest.mark.parametrize("cancel",[False,True])
def test_manual_captcha_cancel_or_repeat_block_stays_paused(tmp_path,monkeypatch,cancel):
    from scraping.sofascore import campaign
    args=campaign_args(tmp_path)+["--manual-captcha","--retry-blocked"]
    client=ManualClient(block="shotmap",cancel=cancel)
    monkeypatch.setattr(campaign,"BrowserClient",lambda *args,**kwargs:client)
    original={"status":403,"at":1}
    with Store(tmp_path/"store") as store:
        store.set_setting("access_block",original)
    with pytest.raises(KeyboardInterrupt if cancel else CollectionPaused):
        campaign.main(args)
    assert client.confirmations==1 and client.closes==1
    assert not (tmp_path/"out/Premier_League_24_25.parquet").exists()
    with Store(tmp_path/"store") as store:
        if cancel:
            assert store.setting("access_block")==original
            assert client.calls==[]
        else:
            assert store.setting("access_block")["status"]==403
    assert json.loads((tmp_path/"out/_collection/status.json").read_text())["state"]=="paused"


def test_manual_homepage_retry_after_remains_enforced(tmp_path):
    from scraping.sofascore import campaign
    client=ManualClient(requests=[{"path":"/sport/football/events","status":429,"retry_after":"60"}])
    with Store(tmp_path/"store") as store:
        campaign.prepare_manual_access(store,client,POLICY,lambda **kw:None,tmp_path/"STOP")
        assert store.setting("cooldown_until") >= campaign.time.time()+59
        limiter=RateLimiter(store,POLICY,store.start_run())
        with pytest.raises(CollectionPaused,match="cooldown"):
            limiter.reserve(Job("/event/100/shotmap","shotmap",100))
        assert store.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]==1


def test_existing_cooldown_precedes_manual_browser(tmp_path,monkeypatch):
    from scraping.sofascore import campaign
    args=campaign_args(tmp_path)+["--manual-captcha","--retry-blocked"]
    def forbidden(*args,**kwargs):
        raise AssertionError("The browser must not open during a server cooldown")
    def interrupt_wait(seconds):
        raise KeyboardInterrupt
    monkeypatch.setattr(campaign,"BrowserClient",forbidden)
    monkeypatch.setattr(campaign.time,"sleep",interrupt_wait)
    with Store(tmp_path/"store") as store:
        store.set_setting("access_block",{"status":403})
        store.set_setting("cooldown_until",campaign.time.time()+60)
    with pytest.raises(KeyboardInterrupt):
        campaign.main(args)
    with Store(tmp_path/"store") as store:
        assert store.setting("access_block")["status"]==403
