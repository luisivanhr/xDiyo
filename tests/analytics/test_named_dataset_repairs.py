"""Public-boundary reproductions from the d21eb134 audit."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import pandas as pd
import pytest
from test_named_datasets import sample
from xdiyo_analytics.data.catalogue import Registry,publish_dataset,select_data
from xdiyo_analytics.data.catalogue.native import seed_comprehensive
from xdiyo_analytics.data.loading import SeasonData


@pytest.mark.parametrize('mode',['union','extend_history'])
def test_cross_snapshot_reserve_correction_cannot_restore_fixture(tmp_path,mode):
    r,f=sample();old=publish_dataset(tmp_path,'international_dataset',{'matches':f},r)
    changed=deepcopy(r.payload)
    for decision in changed['squads']:
        if decision['team']==1:decision['classification']='reserve_or_development_squad'
    newer=publish_dataset(tmp_path,'comprehensive_dataset',{'matches':f},Registry(changed))
    with pytest.raises(ValueError,match='Incompatible squad classifications: team 1'):
        select_data(tmp_path,snapshots={'international_dataset':old,'comprehensive_dataset':newer},mode=mode,train_ids=[1],evaluation_ids=[4])


def test_classification_checks_respect_effective_dates(tmp_path):
    r,f=sample();old=publish_dataset(tmp_path,'international_dataset',{'matches':f},r)
    changed=deepcopy(r.payload)
    for d in changed['squads']:
        if d['team']==1:d['end']='2020-12-31'
    newer=Registry(changed);newer.classify(1,'reserve_or_development_squad',start='2021-01-01',decision='later squad change')
    version=publish_dataset(tmp_path,'comprehensive_dataset',{'matches':f},newer)
    selection=select_data(tmp_path,snapshots={'international_dataset':old,'comprehensive_dataset':version},mode='union')
    assert selection['target_ids']==[1,2,3,4]


def native_input(tmp_path,monkeypatch,*,event=999,day='2020-09-01',home_name='Alpha',reverse=False):
    from xdiyo_analytics.data.catalogue import native
    table=pd.DataFrame([dict(event_id=event,home_id=2 if reverse else 1,away_id=1 if reverse else 2,
        home_name='Beta' if reverse else home_name,away_name=home_name if reverse else 'Beta',
        kickoff_utc=pd.Timestamp(day,tz='UTC').timestamp(),status='finished',is_awarded=False,home_score_current=7,away_score_current=0,round=1)])
    manifest=dict(scope=dict(league_id=17,season_id=10,season_start=2020,season_end=2021),tables={'matches':{'sha256':'0'*64}})
    (tmp_path/'meta').mkdir(exist_ok=True);(tmp_path/'meta/source.json').write_text(json.dumps(manifest))
    (tmp_path/'Premier_League_20_21.manifest.json').write_text(json.dumps({'manifest':'meta/source.json'}))
    monkeypatch.setattr(native,'load_season',lambda *a,**kw:SeasonData({'matches':table},{}))


@pytest.mark.parametrize('kind,identity,facts',[
 ('team',1,dict(name='International-only Club',entity_kind='club')),
 ('season',10,dict(competition=999,label='Other edition',start_year=2010,end_year=2011)),
 ('competition',17,dict(name='Other League',region='Elsewhere',entity_kind='club_league'))])
def test_native_incremental_entity_collisions_fail(tmp_path,monkeypatch,kind,identity,facts):
    native_input(tmp_path,monkeypatch);r=Registry();r.seed(kind,identity,**facts)
    with pytest.raises(ValueError,match=f'native {kind} ID collision'):
        seed_comprehensive(tmp_path,registry=r)
    assert r.payload['entities'][kind][str(identity)]==facts


def test_reviewed_native_entity_remap_updates_all_fixture_keys(tmp_path,monkeypatch):
    native_input(tmp_path,monkeypatch);r=Registry()
    r.seed('team',1,name='Different entity',entity_kind='club');r.seed('team',22,name='Alpha',entity_kind='club')
    r.alias('team',22,'Alpha',scope='17',decision='reviewed distinct identity')
    _,tables,_=seed_comprehensive(tmp_path,registry=r)
    assert tables['matches'].home_id.tolist()==[22]
    assert r.payload['entities']['fixture']['999']['occurrences'][0][2]==22
    assert r.payload['entities']['team']['1']['name']=='Different entity'


@pytest.mark.parametrize('reverse',[False,True])
def test_native_revised_fixture_requires_review_and_keeps_identity(tmp_path,monkeypatch,reverse):
    native_input(tmp_path,monkeypatch);r,tables,_=seed_comprehensive(tmp_path)
    native_input(tmp_path,monkeypatch,day='2020-09-02',reverse=reverse)
    with pytest.raises(ValueError,match='map_occurrence'):
        seed_comprehensive(tmp_path,registry=r)
    assert set(r.payload['entities']['fixture'])=={'999'}
    r.map_occurrence(999,17,10,2 if reverse else 1,1 if reverse else 2,'2020-09-02',decision='reviewed correction')
    _,tables,_=seed_comprehensive(tmp_path,registry=r)
    assert tables['matches'].event_id.tolist()==[999]


def test_native_nearby_repeat_requires_explicit_distinct_decision(tmp_path,monkeypatch):
    native_input(tmp_path,monkeypatch);r,_,_=seed_comprehensive(tmp_path)
    native_input(tmp_path,monkeypatch,event=1001,day='2020-09-02')
    with pytest.raises(ValueError,match='reviewed distinct-fixture'):
        seed_comprehensive(tmp_path,registry=r)
    other=r.fixture(17,10,1,2,'2020-09-02')
    r.map_occurrence(other,17,10,1,2,'2020-09-02',decision='reviewed genuine distinct match')
    _,tables,_=seed_comprehensive(tmp_path,registry=r)
    assert tables['matches'].event_id.tolist()==[other] and other!=999


def test_installed_wheel_includes_catalogue_and_legacy_imports(tmp_path):
    root=Path(__file__).resolve().parents[2];wheels=tmp_path/'wheels';installed=tmp_path/'installed'
    env=dict(os.environ);env.pop('PYTHONPATH',None)
    subprocess.run([sys.executable,'-m','pip','wheel',str(root),'--no-deps','--no-build-isolation','--wheel-dir',str(wheels)],check=True,capture_output=True,env=env)
    wheel=next(wheels.glob('*.whl'))
    subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-compile','--target',str(installed),str(wheel)],check=True,capture_output=True,env=env)
    code="import sys;sys.path.insert(0,sys.argv[1]);import xdiyo_analytics.data as d;from xdiyo_analytics.data.catalogue import native,load_team_identity_decisions,load_reviewed_team_directory;from xdiyo_analytics.data.catalogue.migration import plan_recipe_dataset_migration;from xdiyo_analytics.reporting import TeamCatalog;assert callable(d.load_season) and callable(d.select_data);assert d.__file__.startswith(sys.argv[1]);assert len(load_team_identity_decisions()['records'])>5000;assert len(load_reviewed_team_directory()['teams'])==1204;assert TeamCatalog.builtin().display(36836,badges=True)[1].startswith('data:image/svg+xml;base64,');assert TeamCatalog.builtin().display(22,badges=True)[1].startswith('data:image/svg+xml;base64,');print(d.__file__)"
    result=subprocess.run([sys.executable,'-I','-c',code,str(installed)],cwd=tmp_path,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr



def test_reviewed_date_correction_cannot_cross_train_test(tmp_path,monkeypatch):
    native_input(tmp_path,monkeypatch);r,tables,_=seed_comprehensive(tmp_path)
    root=tmp_path/'catalogue';old=publish_dataset(root,'international_dataset',tables,r)
    native_input(tmp_path,monkeypatch,day='2020-09-02')
    with pytest.raises(ValueError,match='map_occurrence'):seed_comprehensive(tmp_path,registry=r)
    r.map_occurrence(999,17,10,1,2,'2020-09-02',decision='same physical fixture, corrected date')
    _,tables,_=seed_comprehensive(tmp_path,registry=r)
    new=publish_dataset(root,'comprehensive_dataset',tables,r)
    with pytest.raises(ValueError,match='both training and evaluation'):
        select_data(root,snapshots={'comprehensive_dataset':new},train_ids=[999],evaluation_ids=[999])
    with pytest.raises(ValueError,match='Incompatible canonical registries'):
        select_data(root,snapshots={'comprehensive_dataset':new,'international_dataset':old},mode='union')
