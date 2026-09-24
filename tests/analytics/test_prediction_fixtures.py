from copy import deepcopy
import warnings
import numpy as np
import pandas as pd
import pytest
from prediction_persistence_samples import publish,snapshot,prepared,EVENT,HOME,AWAY,BASE,event_mask
from xdiyo_analytics.data import load_prediction_fixtures,select_prediction_fixtures

@pytest.fixture
def batch(tmp_path):
    root=publish(tmp_path/'data/xDiyo_data')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        return load_prediction_fixtures(root,verify_hashes=True)

def test_ordinary_and_daily_names_have_identical_populations(tmp_path):
    import shutil
    root=publish(tmp_path/'data/xDiyo_data');daily=tmp_path/'daily/current';shutil.copytree(root,daily);before=snapshot(root)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning);ordinary=load_prediction_fixtures(root);other=load_prediction_fixtures(daily)
    pd.testing.assert_frame_equal(ordinary.fixtures,other.fixtures)
    assert len(ordinary.data.matches)==14 and len(ordinary.completed_matches)==4
    assert ordinary.fixtures.event_id.tolist()==[EVENT+3,EVENT+3,EVENT+2,EVENT+2,EVENT+5,EVENT+5]
    assert ordinary.fixtures.status.eq('notstarted').all()
    assert ordinary.completed_matches.event_id.tolist()==[EVENT,EVENT+1,EVENT,EVENT+1]
    assert snapshot(root)==before and snapshot(daily)==before

@pytest.mark.parametrize('layout',['match','team_match'])
def test_align_keeps_exact_ids_null_labels_and_full_upstream_history(batch,layout):
    dataset,history=prepared(batch,layout);original=deepcopy(dataset);source=deepcopy(batch.data.tables)
    aligned=batch.align(dataset);factor=1 if layout=='match' else 2
    assert len(dataset.X)==14*factor and len(aligned.X)==6*factor
    assert aligned.X.index.equals(pd.RangeIndex(6*factor)) and aligned.y.index.equals(aligned.metadata.index)
    assert aligned.y.isna().all().all()
    assert aligned.metadata.event_id.tolist()==[event for event in batch.fixtures.event_id for _ in range(factor)]
    if layout=='team_match':
        assert aligned.metadata.side.tolist()==['home','away']*6 and aligned.metadata.team_id.tolist()==[HOME,AWAY]*6
    else:assert aligned.metadata.home_id.eq(HOME).all() and aligned.metadata.away_id.eq(AWAY).all()
    known=~event_mask(aligned.metadata,EVENT+5);recent=[c for c in aligned.X if c.endswith('recent')]
    assert aligned.X.loc[known,recent].notna().all().all() and aligned.X.loc[~known,recent].isna().all().all()
    for name in ['X','y','metadata']:pd.testing.assert_frame_equal(getattr(dataset,name),getattr(original,name))
    for name,frame in source.items():pd.testing.assert_frame_equal(batch.data.tables[name],frame)
    aligned.definitions['new']='local';assert 'new' not in dataset.definitions

@pytest.mark.parametrize('rounds,expected',[(None,6),(5,2),([5,6],4),({'Alpha':5,'Beta':[6,7]},3),({'Alpha':[5,99]},1),([],0),({},0),([99],0)])
@pytest.mark.parametrize('layout',['match','team_match'])
def test_round_selection_at_select_and_align_preserves_history_and_empty_schema(batch,rounds,expected,layout):
    dataset,_=prepared(batch,layout);before=deepcopy(dataset);selected=select_prediction_fixtures(batch.data,rounds=rounds)
    aligned=batch.align(dataset,rounds=rounds);other=selected.align(dataset);factor=1 if layout=='match' else 2
    assert len(selected.fixtures)==expected and len(aligned.X)==expected*factor
    assert selected.data is batch.data and len(selected.data.matches)==14
    pd.testing.assert_frame_equal(aligned.X,other.X);pd.testing.assert_frame_equal(aligned.metadata,other.metadata)
    for name in ['X','y','metadata']:
        assert getattr(aligned,name).dtypes.equals(getattr(dataset,name).dtypes)
        pd.testing.assert_frame_equal(getattr(dataset,name),getattr(before,name))

def test_load_rounds_then_alignment_only_narrows_existing_batch(tmp_path):
    root=publish(tmp_path/'ordinary')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning);initial=load_prediction_fixtures(root,rounds={'Alpha':[5,6],'Beta':7})
    dataset,_=prepared(initial)
    assert len(initial.data.matches)==14 and len(initial.completed_matches)==4 and len(initial.fixtures)==3
    final=initial.align(dataset,rounds=[5,7])
    assert final.metadata.source_league.tolist()==['Alpha','Beta'] and final.metadata.event_id.tolist()==[EVENT+2,EVENT+5]
    assert initial.align(dataset,rounds={'Beta':5}).X.empty

def test_status_and_clock_filters_are_independent_of_round_numbers(batch):
    selected=select_prediction_fixtures(batch.data,statuses=['notstarted','postponed'],rounds=[5,8],as_of=pd.Timestamp(BASE+3*86400,unit='s'))
    assert selected.fixtures.event_id.tolist()==[EVENT+2,EVENT+2,EVENT+7,EVENT+7]
    assert select_prediction_fixtures(batch.data,as_of='2099-01-01').fixtures.empty and len(batch.fixtures)==6

def test_nullable_rounds_retained_by_none_excluded_from_explicit_requests(batch):
    data=deepcopy(batch.data);data.matches.loc[event_mask(data.matches,EVENT+2),'round']=pd.NA
    assert len(select_prediction_fixtures(data).fixtures)==6 and len(select_prediction_fixtures(data,rounds=[5,6]).fixtures)==2

@pytest.mark.parametrize('rounds',[True,-1,1.5,'5',[5.0],[True],{'Alpha':None},{'':5},{1:5}])
def test_invalid_round_requests_fail_without_mutation(batch,rounds):
    before=deepcopy(batch.data.matches)
    with pytest.raises((TypeError,ValueError)):select_prediction_fixtures(batch.data,rounds=rounds)
    pd.testing.assert_frame_equal(batch.data.matches,before)

@pytest.mark.parametrize('problem',['missing_match','duplicate_match','wrong_home','missing_identity'])
def test_align_rejects_incomplete_or_wrong_match_identity(batch,problem):
    dataset,_=prepared(batch);row=int(np.flatnonzero(event_mask(dataset.metadata,EVENT+2))[0])
    if problem=='missing_match':
        for name in ['X','y','metadata']:setattr(dataset,name,getattr(dataset,name).drop(index=row).reset_index(drop=True))
    elif problem=='duplicate_match':
        for name in ['X','y','metadata']:setattr(dataset,name,pd.concat([getattr(dataset,name),getattr(dataset,name).iloc[[row]]],ignore_index=True))
    elif problem=='wrong_home':dataset.metadata.loc[row,'home_id']=HOME+10
    else:dataset.metadata.loc[row,'event_id']=pd.NA
    with pytest.raises(ValueError):batch.align(dataset)

@pytest.mark.parametrize('problem',['missing_side','duplicate_side','wrong_team','wrong_opponent'])
def test_team_layout_requires_complete_matching_home_away_pair(batch,problem):
    dataset,_=prepared(batch,'team_match');rows=np.flatnonzero(event_mask(dataset.metadata,EVENT+2)&dataset.metadata.source_league.eq('Alpha'))
    first=int(rows[0]);second=int(rows[1])
    if problem=='missing_side':
        for name in ['X','y','metadata']:setattr(dataset,name,getattr(dataset,name).drop(index=first).reset_index(drop=True))
    elif problem=='duplicate_side':dataset.metadata.loc[first,'side']=dataset.metadata.side.iloc[second]
    elif problem=='wrong_team':dataset.metadata.loc[first,'team_id']=HOME+90
    else:dataset.metadata.loc[first,'opponent_id']=AWAY+90
    with pytest.raises(ValueError):batch.align(dataset)

def test_filtered_fixture_prediction_uses_saved_pipeline_without_labels_or_refitting(batch,tmp_path,monkeypatch):
    from sklearn.linear_model import Ridge
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xdiyo_analytics.training import EstimatorAdapter,refit_model,save_model,load_model
    dataset,_=prepared(batch)
    train=np.flatnonzero(dataset.metadata.status.eq('finished').to_numpy()&dataset.y.notna().all(axis=1).to_numpy())
    assert len(train)==2
    fitted=refit_model(dataset,lambda:EstimatorAdapter(make_pipeline(SimpleImputer(keep_empty_features=True),StandardScaler(),Ridge())),train_positions=train)
    future=batch.align(dataset,rounds={'Alpha':5,'Beta':6})
    expected=fitted.predict(future);path=save_model(fitted,tmp_path/'deployment')
    monkeypatch.setattr(Ridge,'fit',lambda *args:pytest.fail('prediction must not refit'))
    actual=load_model(path).predict(future)
    pd.testing.assert_frame_equal(actual['predict'],expected['predict'])
    assert len(actual['predict'])==2 and future.y.isna().all().all()
    assert future.metadata.source_league.tolist()==['Beta','Alpha']

@pytest.mark.parametrize('options',[{'statuses':[]},{'statuses':['']},{'as_of':pd.NaT},{'as_of':'not a date'}])
def test_invalid_status_or_time_selection_is_explicit(batch,options):
    with pytest.raises(ValueError):select_prediction_fixtures(batch.data,**options)

def test_round_filter_requires_round_metadata_only_when_requested(batch):
    data=deepcopy(batch.data);data.tables['matches']=data.matches.drop(columns='round')
    assert len(select_prediction_fixtures(data).fixtures)==6
    with pytest.raises(KeyError,match='round'):select_prediction_fixtures(data,rounds=5)
