"""Fixture-date identity and statistic alignment regression from the 5de29058 review."""
from copy import deepcopy
import json
import pandas as pd
import pytest
from xdiyo_analytics.data.catalogue import Registry,publish_dataset,load_snapshot
from xdiyo_analytics.data.catalogue.native import seed_comprehensive
from xdiyo_analytics.data.loading import SeasonData


def dated_input(tmp_path,monkeypatch,days,*,reverse=False,bad_stat=False):
    from xdiyo_analytics.data.catalogue import native
    rows=[];stats=[]
    for i,day in enumerate(days):
        event=999+i;home,away=(2,22) if reverse else (22,2)
        rows.append(dict(event_id=event,home_id=home,away_id=away,
            home_name='Beta' if reverse else 'Shared Name',away_name='Shared Name' if reverse else 'Beta',
            kickoff_utc=None if day is None else pd.Timestamp(day,tz='UTC').timestamp(),
            status='finished',is_awarded=False,home_score_current=7,away_score_current=0,round=i+1))
        for side,team,value in [('home',home,11+i),('away',away,4+i)]:
            stats.append(dict(event_id=event,team_id=123 if bad_stat and i==0 and side=='home' else team,
                side=side,period='ALL',group_name='Match overview',key='cornerKicks',value=value))
    frame=pd.DataFrame(rows);statistics=pd.DataFrame(stats)
    manifest=dict(scope=dict(league_id=17,season_id=10,season_start=2020,season_end=2021),
        tables={'matches':{'sha256':'0'*64},'statistics':{'sha256':'1'*64}})
    (tmp_path/'meta').mkdir(exist_ok=True)
    (tmp_path/'meta/source.json').write_text(json.dumps(manifest),encoding='utf-8')
    (tmp_path/'Premier_League_20_21.manifest.json').write_text(json.dumps({'manifest':'meta/source.json'}),encoding='utf-8')
    monkeypatch.setattr(native,'load_season',lambda *a,**kw:SeasonData({'matches':frame,'statistics':statistics},{}))
    r=Registry();r.seed('team',1,name='Old Club',entity_kind='club');r.seed('team',22,name='New Club',entity_kind='club')
    r.alias('team',1,'Shared Name',scope='17',end='2020-07-31',decision='reviewed old club interval')
    r.alias('team',22,'Shared Name',scope='17',start='2020-08-01',decision='reviewed new club interval')
    return r,frame,statistics


def test_exact_audit_reproduction_uses_september_not_january(tmp_path,monkeypatch):
    r,_,_=dated_input(tmp_path,monkeypatch,['2020-09-01'])
    original=deepcopy(r.payload['aliases'])
    _,tables,_=seed_comprehensive(tmp_path,registry=r)
    assert tables['matches'].home_id.tolist()==[22]
    assert tables['matches'].home_name.tolist()==['New Club']
    assert tables['matches'].home_score_current.tolist()==[7]
    assert [a for a in r.payload['aliases'] if a['name']=='Shared Name']==original
    assert r.resolve('team','Shared Name','17','2020-07-31')==1
    assert r.resolve('team','Shared Name','17','2020-08-01')==22


@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('order',[(0,1),(1,0)])
def test_midseason_native_id_reuse_maps_matches_and_statistics_per_fixture(tmp_path,monkeypatch,reverse,order):
    days=['2020-07-31','2020-08-01']
    r,frame,stats=dated_input(tmp_path,monkeypatch,[days[i] for i in order],reverse=reverse)
    original=deepcopy(r.payload['aliases'])
    _,tables,_=seed_comprehensive(tmp_path,registry=r)
    expected=[1 if i==0 else 22 for i in order]
    side='away' if reverse else 'home'
    assert tables['matches'][side+'_id'].tolist()==expected
    assert tables['statistics'].loc[tables['statistics'].side.eq(side),'team_id'].tolist()==expected
    assert tables['statistics'].value.tolist()==stats.value.tolist()
    assert [a for a in r.payload['aliases'] if a['name']=='Shared Name']==original
    added=[a for a in r.payload['aliases'] if a['kind']=='team' and a['name']=='Beta']
    assert added and all(a['start']==a['end'] for a in added)
    version=publish_dataset(tmp_path/'catalogue','comprehensive_dataset',tables,r)
    saved,_,registry=load_snapshot(tmp_path/'catalogue','comprehensive_dataset',version)
    assert saved['matches'][side+'_id'].tolist()==expected
    assert registry.resolve('team','Shared Name','17','2020-07-31')==1
    assert registry.resolve('team','Shared Name','17','2020-08-01')==22
    # Re-importing is idempotent and never broadens the reviewed intervals.
    payload=deepcopy(r.payload);_,again,_=seed_comprehensive(tmp_path,registry=r)
    assert r.payload==payload
    pd.testing.assert_frame_equal(again['matches'],tables['matches'])
    pd.testing.assert_frame_equal(again['statistics'],tables['statistics'])


@pytest.mark.parametrize('value',[None,float('nan'),float('inf'),'not a timestamp','2020-09-01',True])
def test_missing_fixture_date_is_not_a_season_date_fallback(tmp_path,monkeypatch,value):
    r,frame,_=dated_input(tmp_path,monkeypatch,[None])
    frame['kickoff_utc']=pd.Series([value],dtype=object)
    aliases=deepcopy(r.payload['aliases'])
    with pytest.raises(ValueError,match='fixture 999.*date.*review'):
        seed_comprehensive(tmp_path,registry=r)
    assert [a for a in r.payload['aliases'] if a['kind']=='team']==aliases
    assert not r.payload['entities'].get('fixture')


def test_overlapping_aliases_fail_closed(tmp_path,monkeypatch):
    r,_,_=dated_input(tmp_path,monkeypatch,['2020-09-01'])
    r.alias('team',1,'Shared Name',scope='17',start='2020-08-01',decision='contradictory review')
    with pytest.raises(ValueError,match='Ambiguous canonical team'):
        seed_comprehensive(tmp_path,registry=r)


def test_statistic_team_must_match_native_fixture_side(tmp_path,monkeypatch):
    r,_,_=dated_input(tmp_path,monkeypatch,['2020-09-01'],bad_stat=True)
    with pytest.raises(ValueError,match='Statistic.*999.*home'):
        seed_comprehensive(tmp_path,registry=r)
