"""Movement enrichment uses reviewed evidence, never imported roster absence."""
import json
import pandas as pd
import pytest
from test_named_datasets import sample
from xdiyo_analytics.data.movements import derive_movements
from xdiyo_analytics.data.catalogue import (
    publish_dataset, load_snapshot, enrich_movements, select_data, resolve_selection, selected_history)
from xdiyo_analytics.data.catalogue.integration import feature_requirements
from xdiyo_analytics.features import TeamMovement, evaluate_features


def flags():
    return derive_movements([dict(league='league',year=2020,stem='league20',competition_id=17,season_id=1,
                                 teams={'1':'Alpha','2':'Beta'})],
                            {'league':dict(system='country',tier=1)},
                            boundaries=[dict(stem='league20',promoted_ids=[1],sources=['reviewed boundary'],complete_review=True)])


def enriched(tmp_path):
    registry, frame = sample()
    old = publish_dataset(tmp_path,'international_dataset',{'matches':frame},registry)
    new = enrich_movements(tmp_path,'international_dataset',old,flags())
    return old,new


def test_enrichment_immutable_zstd_history_and_native_recipe(tmp_path):
    import pyarrow.parquet as pq
    from xdiyo_analytics.ui import default_recipe,prepare_recipe
    from xdiyo_analytics.ui.recipe import node,export_python,read_recipe
    old,new=enriched(tmp_path)
    assert old != new
    before,_,_=load_snapshot(tmp_path,'international_dataset',old)
    after,manifest,_=load_snapshot(tmp_path,'international_dataset',new)
    pd.testing.assert_frame_equal(before['matches'],after['matches'][before['matches'].columns])
    assert 'team_seasons' not in before
    assert pq.ParquetFile(tmp_path/'international_dataset'/new/'team_seasons.parquet').metadata.row_group(0).column(0).compression=='ZSTD'
    sel=select_data(tmp_path,snapshots={'international_dataset':new},requirements=['movements'],train_ids=[3],evaluation_ids=[4])
    history=selected_history(resolve_selection(tmp_path,sel))
    out=evaluate_features(history,{'promoted':TeamMovement('promoted'),'relegated':TeamMovement('relegated')})
    assert out.promoted.tolist()==history.team_id.eq(1).astype(float).tolist()
    assert out.relegated.eq(0).all()
    assert feature_requirements({'p':TeamMovement()})=={'goals','movements'}
    recipe=default_recipe();recipe['data_selection']=dict(root=str(tmp_path),selection=sel)
    recipe['features']={'promoted':node('features.TeamMovement',movement='promoted'),
                        'relegated':node('features.TeamMovement',movement='relegated')}
    recipe['labels']={'outcome':node('labels.Outcome',perspective='home')};recipe['target']='outcome'
    prepared=prepare_recipe(recipe)
    assert len(prepared.dataset.X.columns)==4
    assert prepared.dataset.X.notna().all().all()
    path=tmp_path/'recipe.json';path.write_text(json.dumps(recipe))
    pd.testing.assert_frame_equal(prepared.dataset.X,prepare_recipe(read_recipe(path)).dataset.X)
    namespace={};exec(compile(export_python(recipe).split('result = run_recipe')[0],'export','exec'),namespace)
    pd.testing.assert_frame_equal(namespace['prepared'].dataset.X,prepared.dataset.X)
    assert enrich_movements(tmp_path,'international_dataset',old,flags())==new


@pytest.mark.parametrize('mutation', ['missing','duplicate','fractional','contradiction','no_evidence'])
def test_invalid_evidence_rejected(tmp_path,mutation):
    registry,frame=sample();f=flags()
    if mutation=='missing':f=f.iloc[:1]
    elif mutation=='duplicate':f=pd.concat([f,f.iloc[:1]])
    elif mutation=='fractional':f['team_id']=f.team_id.astype(float)
    elif mutation=='contradiction':f.loc[0,'got_promoted']=False
    else:f.loc[0,'evidence']=''
    with pytest.raises(ValueError):publish_dataset(tmp_path,'international_dataset',{'matches':frame,'team_seasons':f},registry)


def test_unknown_partial_roster_and_capability_gate(tmp_path):
    registry,frame=sample()
    f=derive_movements([dict(league='league',year=2020,stem='s',competition_id=17,season_id=1,teams={'1':'A','2':'B'})],
                       {'league':dict(system='country',tier=1)})
    assert f.movement.eq('unknown').all() and f.got_promoted.isna().all()
    version=publish_dataset(tmp_path,'international_dataset',{'matches':frame,'team_seasons':f},registry)
    sel=select_data(tmp_path,snapshots={'international_dataset':version},requirements=['movements'])
    assert sel['records']==[] and {r['reason'] for r in sel['exclusions']}=={'unknown_season_movement'}
    ordinary=select_data(tmp_path,snapshots={'international_dataset':version})
    assert len(ordinary['records'])==4
    with pytest.raises(ValueError,match='Rebuild frozen selection'):
        evaluate_features(selected_history(resolve_selection(tmp_path,ordinary)),{'p':TeamMovement()})


def test_strict_movement_availability_not_invented(tmp_path):
    registry,frame=sample()
    frame['status']='finished';frame['score_basis']='regulation'
    frame['kickoff_at']=pd.to_datetime(frame.date,utc=True)
    frame['result_available_at']=frame.kickoff_at+pd.Timedelta(hours=3)
    version=publish_dataset(tmp_path,'international_dataset',{'matches':frame,'team_seasons':flags()},registry)
    sel=select_data(tmp_path,snapshots={'international_dataset':version},requirements=['movements'],profile='strict_as_of')
    assert {r['reason'] for r in sel['exclusions']}=={'unsupported_movement_availability'}


def test_subset_selection_keeps_evidence_and_conflicting_union_rejected(tmp_path):
    old,new=enriched(tmp_path)
    tables,_,registry=load_snapshot(tmp_path,'international_dataset',new)
    # Match metadata cannot disagree with the evidence even if boolean equality is nullable.
    tables['matches'].loc[0,'home_got_promoted']=pd.NA
    with pytest.raises(ValueError,match='Conflicting materialized'):
        publish_dataset(tmp_path,'international_dataset',tables,registry)
    tables,_,registry=load_snapshot(tmp_path,'international_dataset',new)
    f=flags();f.loc[0,['movement','got_promoted']]=['retained',False]
    raw=load_snapshot(tmp_path,'international_dataset',old)[0]['matches']
    other=publish_dataset(tmp_path,'comprehensive_dataset',{'matches':raw,'team_seasons':f},registry)
    sel=select_data(tmp_path,snapshots={'international_dataset':new,'comprehensive_dataset':other},mode='union',requirements=['movements'])
    assert not sel['records'] and all(e['reason']=='material_conflict' for e in sel['exclusions'])
    sel=select_data(tmp_path,snapshots={'international_dataset':new},target_ids=[1],history_ids=[] ,requirements=['movements'])
    data=resolve_selection(tmp_path,sel)
    assert len(data.matches)==1 and len(data['team_seasons'])==2


def test_payload_tampering_rejected(tmp_path):
    _,new=enriched(tmp_path)
    path=tmp_path/'international_dataset'/new/'team_seasons.parquet'
    f=pd.read_parquet(path);f.loc[0,'got_promoted']=False;f.to_parquet(path,index=False,compression='zstd')
    with pytest.raises(ValueError,match='payload changed'):
        load_snapshot(tmp_path,'international_dataset',new)


def test_unknown_alternate_is_not_a_contradiction(tmp_path):
    old,new=enriched(tmp_path)
    tables,_,registry=load_snapshot(tmp_path,'international_dataset',old)
    f=flags();f['movement']='unknown';f['got_promoted']=pd.array([None,None],dtype='boolean');f['got_demoted']=pd.array([None,None],dtype='boolean')
    alternate=publish_dataset(tmp_path,'comprehensive_dataset',dict(matches=tables['matches'],team_seasons=f),registry)
    sel=select_data(tmp_path,snapshots={'international_dataset':new,'comprehensive_dataset':alternate},mode='union',requirements=['movements'])
    assert len(sel['records'])==4
    assert {r['dataset'] for r in sel['records']}=={'international_dataset'}
    assert resolve_selection(tmp_path,sel)['team_seasons'].movement.tolist()==['promoted','retained']
