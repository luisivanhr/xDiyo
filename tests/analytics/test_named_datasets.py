"""Canonical dataset contract regressions. No original data or model fitting."""
from copy import deepcopy
import json
import pandas as pd
import numpy as np
import pytest
from xdiyo_analytics.data.catalogue import *
from xdiyo_analytics.data.catalogue.storage import STAT_COLUMNS, ODDS_COLUMNS
from xdiyo_analytics.features import RollingMean,MatchScore,Stat,IsHome
from xdiyo_analytics.features import evaluate_features


def sample():
    r=Registry();r.seed('competition',17,name='Premier_League',region='England',entity_kind='club_league')
    r.alias('competition',17,'ENG1',scope='historical_league_code',decision='reviewed')
    r.alias('competition',17,'Premier League',scope='England',decision='reviewed')
    r.seed('season',1,competition=17,label='2020/2021',start_year=2020,end_year=2021)
    for name in ['20-21','2020/21']:r.alias('season',1,name,scope='17',decision='reviewed')
    for team,name in enumerate(['Alpha','Beta','Willem II','Reserve B'],1):
        r.seed('team',team,name=name,entity_kind='club');r.alias('team',team,name,scope='17',decision='reviewed')
        r.classify(team,'reserve_or_development_squad' if team==4 else 'not_reserve',decision='reviewed squad identity')
    rows=[]
    for day,home,away,score in [(1,1,2,1),(5,2,1,2),(9,1,2,3),(13,2,1,4)]:
        date=f'2020-09-{day:02}';event=r.fixture(17,1,home,away,date)
        rows.append(dict(event_id=event,competition_id=17,season_id=1,season_year='2020/2021',stage='',round=None,
            home_id=home,away_id=away,home_name=['','Alpha','Beta'][home],away_name=['','Alpha','Beta'][away],
            date=date,local_datetime=None,time_precision='date',timezone='unknown',kickoff_at=None,result_available_at=None,
            status='reported_result',score_basis='reported_unknown',home_score_current=score,away_score_current=0,
            identity_state='resolved',conflict=False,membership_verified=False,schedule_complete=False))
    return r,pd.DataFrame(rows)


def published(tmp_path):
    r,f=sample();version=publish_dataset(tmp_path,'international_dataset',{'matches':f},r)
    return r,f,{'international_dataset':version}


def test_snapshot_repeat_zstd_reload_and_tamper(tmp_path):
    import pyarrow.parquet as pq
    r,f,s=published(tmp_path);v=s['international_dataset']
    assert publish_dataset(tmp_path,'international_dataset',{'matches':f},r)==v
    loaded,manifest,_=load_snapshot(tmp_path,'international_dataset',v)
    assert list(loaded['matches'].event_id)==list(f.event_id)
    path=tmp_path/'international_dataset'/v/'matches.parquet'
    assert pq.ParquetFile(path).metadata.row_group(0).column(0).compression=='ZSTD'
    path.write_bytes(path.read_bytes()+b'bad')
    with pytest.raises(ValueError,match='payload changed'):load_snapshot(tmp_path,'international_dataset',v)


def test_selection_freeze_reload_and_new_snapshot_does_not_expand(tmp_path):
    r,f,s=published(tmp_path);sel=select_data(tmp_path,snapshots=s)
    save_selection(sel,tmp_path/'selection.json')
    data=resolve_selection(tmp_path,tmp_path/'selection.json')
    assert data.provenance['data_selection']==sel
    changed=f.copy();changed.loc[0,'home_score_current']=9
    v=publish_dataset(tmp_path,'international_dataset',{'matches':changed},r)
    assert v!=s['international_dataset']
    assert resolve_selection(tmp_path,sel).matches.home_score_current.iloc[0]==1
    bad=deepcopy(sel);bad['history_ids'].pop()
    with pytest.raises(ValueError,match='changed'):resolve_selection(tmp_path,bad)


@pytest.mark.parametrize('side',['home','away','both'])
def test_reserve_filter_before_history_and_no_score_influence(tmp_path,side):
    r,f=sample();row=f.iloc[0].to_dict();row['date']='2020-08-01'
    for field in (['home_id','away_id'] if side=='both' else [side+'_id']):row[field]=4
    if side=='both':
        r.seed('team',5,name='Reserve II',entity_kind='club');r.classify(5,'reserve_or_development_squad',decision='reviewed');row['away_id']=5
    row['event_id']=r.fixture(17,1,row['home_id'],row['away_id'],row['date'])
    raw=pd.concat([pd.DataFrame([row]),f],ignore_index=True)
    v=publish_dataset(tmp_path,'international_dataset',{'matches':raw},r)
    sel=select_data(tmp_path,snapshots={'international_dataset':v});data=resolve_selection(tmp_path,sel)
    assert set(data.matches.event_id)==set(f.event_id)
    history=selected_history(data)
    a=evaluate_features(history,{'mean':RollingMean(MatchScore(),window=20)})
    raw.loc[0,'home_score_current']=999
    v2=publish_dataset(tmp_path,'international_dataset',{'matches':raw},r)
    h2=selected_history(resolve_selection(tmp_path,select_data(tmp_path,snapshots={'international_dataset':v2})))
    pd.testing.assert_frame_equal(a,evaluate_features(h2,{'mean':RollingMean(MatchScore(),window=20)}))
    assert inspect_coverage(tmp_path,'international_dataset',v)['excluded']=={'reserve_excluded':1}
    assert not inspect_coverage(tmp_path,'international_dataset',v)['real_schedule_complete']


def test_squad_unknown_false_positives_effective_aliases():
    r,_=sample()
    assert r.squad(3,'2020-01-01')=='not_reserve'
    assert r.squad(99,'2020-01-01')=='needs_review'
    r.alias('team',4,'Jong Alpha',scope='17',decision='reviewed reserve alias')
    assert r.squad(r.resolve('team','Jong Alpha','17','2020-01-01'),'2020-01-01')=='reserve_or_development_squad'
    r.alias('team',1,'Same name',scope='17',start='2000-01-01',end='2010-01-01',decision='reviewed')
    r.alias('team',2,'Same name',scope='17',start='2011-01-01',decision='reviewed')
    assert r.resolve('team','Same name','17','2005-01-01')==1
    assert r.resolve('team','Same name','17','2012-01-01')==2
    assert r.resolve('team','Same name','other-country','2012-01-01') is None


def test_fixed_targets_history_and_same_batch_causality(tmp_path):
    r,f,s=published(tmp_path)
    sel=select_data(tmp_path,snapshots=s,mode='extend_history',train_ids=[3],evaluation_ids=[4])
    assert sel['target_ids']==[3,4] and sel['history_ids']==[1,2,3]
    history=selected_history(resolve_selection(tmp_path,sel));out=evaluate_features(history,{'mean':RollingMean(MatchScore())})
    assert out.iloc[0,0]!=out.iloc[0,0]
    assert history.loc[history.event_id.eq(4),'catalogue_history_available_at'].isna().all()
    shuffled=history.sample(frac=1,random_state=3)
    other=evaluate_features(shuffled,{'mean':RollingMean(MatchScore())}).sort_index()
    pd.testing.assert_frame_equal(out,other)
    with pytest.raises(ValueError,match='cutoff'):evaluate_features(history,{'mean':RollingMean(MatchScore())},cutoffs=history.kickoff_at+pd.Timedelta(days=1))


def test_future_mutation_truncation_and_requirements(tmp_path):
    r,f,s=published(tmp_path);sel=select_data(tmp_path,snapshots=s)
    h=selected_history(resolve_selection(tmp_path,sel));out=evaluate_features(h,{'mean':RollingMean(MatchScore())})
    later=h.copy();later.loc[later.event_id.eq(4),'goals_for']=999
    pd.testing.assert_frame_equal(out,evaluate_features(later,{'mean':RollingMean(MatchScore())}))
    short=h.loc[h.event_id.lt(4)].copy()
    pd.testing.assert_frame_equal(out.loc[short.index],evaluate_features(short,{'mean':RollingMean(MatchScore())}))
    with pytest.raises(ValueError,match='requirements'):evaluate_features(h,{'shots':RollingMean(Stat('ALL','Shots','shotsOnGoal'))})
    for requirements in [('goals','membership'),('goals','odds'),('regulation_goals',)]:
        rejected=select_data(tmp_path,snapshots=s,requirements=requirements)
        assert not rejected['target_ids'] and len(rejected['exclusions'])==4
    assert requirement_closure([{'name':'nested','dependencies':[{'requires':['stat:ALL:cornerKicks']}]}])==['goals','stat:ALL:cornerKicks']


def test_strict_rejects_unknown_time_and_accepts_aware_evidence(tmp_path):
    r,f,s=published(tmp_path)
    assert not select_data(tmp_path,snapshots=s,profile='strict_as_of')['target_ids']
    f['kickoff_at']=[x+'T12:00:00Z' for x in f.date];f['result_available_at']=[x+'T15:00:00Z' for x in f.date]
    f['status']='finished';f['score_basis']='regulation'
    v=publish_dataset(tmp_path,'international_dataset',{'matches':f},r)
    assert len(select_data(tmp_path,snapshots={'international_dataset':v},profile='strict_as_of')['target_ids'])==4
    f['kickoff_at']=f.date+'T12:00:00'
    v=publish_dataset(tmp_path,'international_dataset',{'matches':f},r)
    with pytest.raises(ValueError,match='timezone-aware'):select_data(tmp_path,snapshots={'international_dataset':v})


def test_coherent_preference_and_conflict(tmp_path):
    r,f,s=published(tmp_path);native=f.copy();native.loc[0,'home_score_current']=np.nan
    v=publish_dataset(tmp_path,'comprehensive_dataset',{'matches':native},r)
    sel=select_data(tmp_path,snapshots={**s,'comprehensive_dataset':v},mode='union')
    assert sel['records'][0]['dataset']=='international_dataset'
    assert sel['records'][1]['dataset']=='comprehensive_dataset'
    native.loc[0,'home_score_current']=9
    v=publish_dataset(tmp_path,'comprehensive_dataset',{'matches':native},r)
    sel=select_data(tmp_path,snapshots={**s,'comprehensive_dataset':v},mode='union')
    assert 1 not in sel['target_ids']
    with pytest.raises(ValueError,match='both training'):select_data(tmp_path,snapshots=s,train_ids=[1],evaluation_ids=[1])


def test_no_provenance_and_atomic_interruption(tmp_path,monkeypatch):
    import xdiyo_analytics.data.catalogue.storage as module
    r,f=sample();bad=f.assign(provider='forbidden')
    with pytest.raises(ValueError,match='Noncanonical'):publish_dataset(tmp_path,'international_dataset',{'matches':bad},r)
    original=module.os.replace
    def fail(*a):raise OSError('interrupted publication')
    monkeypatch.setattr(module.os,'replace',fail)
    with pytest.raises(OSError):publish_dataset(tmp_path,'international_dataset',{'matches':f},r)
    assert list_datasets(tmp_path)==[]
    monkeypatch.setattr(module.os,'replace',original)
    assert publish_dataset(tmp_path,'international_dataset',{'matches':f},r)


def test_txt_syntax_revision_and_unknown_scores(tmp_path):
    r,_=sample();p=tmp_path/'results.txt';p.write_text('Sea\tLge\tDate\tHT\tAT\tHS\tAS\tGD\tWDL\n20-21\tENG1\t01/09/2020\tAlpha\tBeta\t1\t0\t1\tW\n')
    a=read_historical_results(p,r);assert len(a.matches)==1 and a.matches.kickoff_at.isna().all()
    assert a.matches.score_basis.iloc[0]=='reported_unknown'
    with pytest.raises(ValueError,match='v2'):read_historical_results(p,r,revision='v1')
    p.write_text(p.read_text().replace('\t1\tW','\t2\tW'))
    with pytest.raises(ValueError,match='difference'):read_historical_results(p,r)


def test_fixture_identity_not_score_and_reviewed_date_shift():
    r,f=sample();assert r.fixture(17,1,1,2,'2020-09-01')==1
    assert r.fixture(17,1,2,1,'2020-09-01')!=1
    assert r.fixture(17,1,1,2,'2020-10-01')!=1
    r.map_occurrence(1,17,1,1,2,'2020-09-02',decision='reviewed postponement')
    assert r.fixture(17,1,1,2,'2020-09-02')==1

def workbook(path, *, conflict=False, bad_goal=False):
    from openpyxl import Workbook
    w=Workbook();w.remove(w.active)
    identity=dict(match_id='1',start_datetime='2020-09-01 12:00:00',competition_type='League',country='England',league='Premier League',league_id='123',season='2020/21',home_team_id='999',home_team='Alpha',away_team_id='888',away_team='Beta')
    result={**identity,**{f'{side}_goals_{period}':value for side,period,value in [('home','ft',1),('away','ft',0),('home','1h',1),('away','1h',0),('home','2h',0),('away','2h',0)]}}
    if bad_goal:result['home_goals_ft']=2
    for name in ['Matches_Results','Statistics_FT','Statistics_1H','Statistics_2H','Odds']:
        values=result if name=='Matches_Results' else dict(identity)
        if name=='Statistics_FT':values.update(home_corners_ft=-1,away_corners_ft=0,home_ball_possession_ft=55,away_ball_possession_ft=45)
        if name=='Odds':values.update(home_win_opening_odds=1,away_win_opening_odds=2)
        if conflict and name=='Statistics_2H':values['home_team']='wrong'
        ws=w.create_sheet(name);ws.append(list(values));ws.append(list(values.values()));ws.cell(100,40).value=None
    w.save(path)


def test_full_workbook_joins_optional_invalid_and_missing(tmp_path):
    r,_=sample();path=tmp_path/'input.xlsx';workbook(path)
    imported=read_results_workbook(path,r)
    assert imported.summary['inspected']==1
    assert imported.summary['statistics_2h_rows']==1
    assert imported.summary['invalid_statistics_cells']==1
    assert imported.summary['invalid_odds_cells']==1
    assert imported.matches.kickoff_at.isna().all()
    assert imported.matches.score_basis.iloc[0]=='regulation'
    assert imported.statistics.query("key=='cornerKicks' and period=='ALL' and side=='away'").value.iloc[0]==0
    assert imported.statistics.query("key=='cornerKicks' and period=='ALL' and side=='home'").validity.iloc[0]=='invalid'
    assert imported.odds.available_at.isna().all()
    version=publish_dataset(tmp_path/'published','international_dataset',dict(matches=imported.matches,statistics=imported.statistics,odds=imported.odds),r)
    sel=select_data(tmp_path/'published',snapshots={'international_dataset':version},requirements=['stat:ALL:cornerKicks'])
    assert not sel['target_ids']
    workbook(path,conflict=True)
    with pytest.raises(ValueError,match='identity descriptors'):read_results_workbook(path,r)
    workbook(path,bad_goal=True)
    with pytest.raises(ValueError,match='reconcile'):read_results_workbook(path,r)


def test_native_recipe_roundtrip_and_fixed_target_preparation(tmp_path):
    from xdiyo_analytics.ui import default_recipe,prepare_recipe
    from xdiyo_analytics.ui.recipe import node,export_python,read_recipe
    r,f,s=published(tmp_path)
    sel=select_data(tmp_path,snapshots=s,mode='extend_history',train_ids=[3],evaluation_ids=[4])
    recipe=default_recipe();recipe['data_selection']=dict(root=str(tmp_path),selection=sel)
    recipe['features']={'goals':node('features.RollingMean',source=node('features.MatchScore'),window=2),'home':node('features.IsHome')}
    recipe['labels']={'outcome':node('labels.Outcome',perspective='home')};recipe['target']='outcome'
    prepared=prepare_recipe(recipe)
    assert list(prepared.dataset.metadata.event_id)==[3,4]
    assert len(prepared.split_plan.folds)==1
    assert list(prepared.split_plan.folds[0].train)==[0] and list(prepared.split_plan.folds[0].test)==[1]
    assert prepared.dataset.definitions['data_selection']==sel
    path=tmp_path/'recipe.json';path.write_text(json.dumps(recipe));reloaded=prepare_recipe(read_recipe(path))
    pd.testing.assert_frame_equal(prepared.dataset.X,reloaded.dataset.X)
    source=export_python(recipe).split('result = run_recipe')[0]
    namespace={};exec(compile(source,'exported recipe','exec'),namespace)
    pd.testing.assert_frame_equal(namespace['prepared'].dataset.X,prepared.dataset.X)


def test_missing_same_date_and_delayed_statistic(tmp_path):
    r,f=sample();second=f.iloc[1].to_dict();second.update(date=f.iloc[0]['date'],event_id=r.fixture(17,1,2,1,f.iloc[0]['date']))
    f=pd.concat([f.iloc[[0]],pd.DataFrame([second]),f.iloc[2:]],ignore_index=True)
    v=publish_dataset(tmp_path,'international_dataset',{'matches':f},r)
    h=selected_history(resolve_selection(tmp_path,select_data(tmp_path,snapshots={'international_dataset':v})))
    result=evaluate_features(h,{'mean':RollingMean(MatchScore())})
    assert result.loc[h.date.eq('2020-09-01') if 'date' in h else h.kickoff_at.eq(h.kickoff_at.min()),'mean'].isna().all()


def test_registry_policy_change_requires_new_snapshot(tmp_path):
    r,f,s=published(tmp_path);sel=select_data(tmp_path,snapshots=s)
    path=tmp_path/'international_dataset'/s['international_dataset']/'registry.json'
    value=json.loads(path.read_text());value['revision']+=1;path.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='Registry changed'):resolve_selection(tmp_path,sel)

def test_statistic_availability_is_a_dependency_bound(tmp_path):
    r,f=sample();stats=[]
    for row in f.to_dict('records'):
        for side in ('home','away'):
            stats.append(dict(event_id=row['event_id'],team_id=row[side+'_id'],side=side,period='ALL',group_name='Match overview',key='cornerKicks',value=2.,validity='observed',definition='regulation_period',available_at='2020-09-12T00:00:00Z' if row['event_id']==1 else None))
    v=publish_dataset(tmp_path,'international_dataset',dict(matches=f,statistics=pd.DataFrame(stats)),r)
    sel=select_data(tmp_path,snapshots={'international_dataset':v},requirements=['stat:ALL:cornerKicks'])
    h=selected_history(resolve_selection(tmp_path,sel))
    output=evaluate_features(h,{'corners':RollingMean(Stat('ALL','Match overview','cornerKicks'))})
    assert output.loc[h.event_id.eq(2),'corners'].isna().all()
    assert output.loc[h.event_id.eq(3),'corners'].eq(2).all()


def test_partial_score_is_ineligible_not_zero(tmp_path):
    r,f=sample();f['home_score_current']=f.home_score_current.astype('Int64');f.loc[0,'home_score_current']=pd.NA
    v=publish_dataset(tmp_path,'international_dataset',dict(matches=f),r)
    selected=select_data(tmp_path,snapshots={'international_dataset':v})
    assert selected['target_ids']==[2,3,4]


def test_publication_row_order_invariance_and_unknown_fields(tmp_path):
    r,f=sample();a=publish_dataset(tmp_path,'international_dataset',dict(matches=f),r)
    assert a==publish_dataset(tmp_path,'international_dataset',dict(matches=f.sample(frac=1,random_state=6)),r)
    with pytest.raises(ValueError,match='Noncanonical summary'):publish_dataset(tmp_path,'international_dataset',dict(matches=f),r,summary={'provider':'x'})
    with pytest.raises(ValueError,match='Noncanonical entity'):r.allocate('team',name='A',entity_kind='club',external_id=1)
    with pytest.raises(ValueError,match='positive integers'):r.seed('team',float(2**53+1),name='A')


def test_near_date_or_reversed_fixture_is_quarantined_until_review():
    from xdiyo_analytics.data.catalogue.importers import canonical_row
    r,_=sample()
    for home,away,day in [('Alpha','Beta','2020-09-02'),('Beta','Alpha','2020-09-01')]:
        row,reason=canonical_row(r,competition=17,season=1,home=home,away=away,day=day,score=(1,0),basis='reported_unknown')
        assert row is None and reason=='fixture_identity_needs_review'


def test_txt_revision_audit_does_not_restore_v1_rows(tmp_path):
    from xdiyo_analytics.data.catalogue.importers import inspect_historical_revisions
    header='Sea\tLge\tDate\tHT\tAT\tHS\tAS\tGD\tWDL\n'
    a=tmp_path/'one.txt';b=tmp_path/'two.txt'
    a.write_text(header+'16-17\tGER3\t20/03/2017\tZwickau\tHansa Rostock\t0\t0\t0\tD\n'+
                 '15-16\tGER3\t20/03/2017\tZwickau\tHansa Rostock\t0\t0\t0\tD\n')
    b.write_text(header+'16-17\tGER3\t20/03/2017\tZwickau\tHansa Rostock\t2\t2\t0\tD\n')
    review=inspect_historical_revisions(a,b)
    assert review['duplicate_excess']==[1,0] and review['score_revisions']==1
    assert review['supplements_admitted']==0


def test_ambiguous_alias_and_new_competition_identity():
    r,_=sample();new=r.allocate('competition',name='Premier League',region='Other country',entity_kind='club_league')
    assert new!=17
    r.alias('competition',new,'Premier League',scope='Other country',decision='reviewed separate competition')
    assert r.resolve('competition','Premier League','England','2020-01-01')==17
    r.alias('team',2,'Alpha',scope='17',decision='conflicting review')
    with pytest.raises(ValueError,match='Ambiguous'):r.resolve('team','Alpha','17','2020-01-01')


def test_legacy_loading_unchanged(ui_recipe):
    from xdiyo_analytics.data import load_seasons
    options=ui_recipe['data'];before=load_seasons(**options)
    # Merely importing the new APIs/creating an independent registry has no effect.
    Registry()
    after=load_seasons(**options)
    for key in before.tables:pd.testing.assert_frame_equal(before[key],after[key])
    assert before.provenance==after.provenance

from test_ui_workflow import ui_recipe

def test_combined_import_preserves_auxiliary_identity_dtypes(tmp_path):
    from xdiyo_analytics.data.catalogue.importers import empty_result
    r,_=sample();path=tmp_path/'input.xlsx';workbook(path);book=read_results_workbook(path,r)
    text=empty_result([],[],[],{},path)
    tables,summary=combine_imports([text,book])
    assert pd.api.types.is_integer_dtype(tables['statistics'].event_id)
    assert publish_dataset(tmp_path/'catalogue','international_dataset',tables,r,summary=summary)


@pytest.mark.parametrize('identity',[1.5,1.0,True,'1.0',0])
def test_target_ids_never_coerce_or_round(tmp_path,identity):
    _,_,snapshots=published(tmp_path)
    with pytest.raises(ValueError,match='Fixture IDs'):
        select_data(tmp_path,snapshots=snapshots,train_ids=[identity])


def test_retrospective_profile_cannot_hide_contradictory_known_release(tmp_path):
    r,f=sample();f.loc[0,'kickoff_at']='2020-09-01T12:00:00Z';f.loc[0,'result_available_at']='2020-09-01T11:00:00Z'
    version=publish_dataset(tmp_path,'international_dataset',{'matches':f},r)
    with pytest.raises(ValueError,match='Result predates kickoff'):
        select_data(tmp_path,snapshots={'international_dataset':version})


def test_reviewed_fixture_mapping_reapplication_is_idempotent():
    r,f=sample();r.map_occurrence(1,17,1,1,2,'2020-08-31',decision='reviewed civil-date correction')
    before=deepcopy(r.payload)
    r.map_occurrence(1,17,1,1,2,'2020-08-31',decision='reviewed civil-date correction')
    assert r.payload==before


def test_native_review_retains_accented_canonical_reserve_name():
    from xdiyo_analytics.data.catalogue.native import review_native_reserves
    r=Registry();r.seed('team',24343,name='Barcelona Atl\u00e8tic',entity_kind='club')
    review_native_reserves(r)
    assert r.squad(24343,'2020-01-01')=='reserve_or_development_squad'
    before=deepcopy(r.payload);review_native_reserves(r)
    assert r.payload==before
