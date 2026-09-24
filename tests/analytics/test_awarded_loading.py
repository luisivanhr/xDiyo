import json,warnings
import numpy as np
import pandas as pd
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from prediction_persistence_samples import publish,snapshot,EVENT,prepared,event_mask
from xdiyo_analytics.data import load_season,load_season_table,load_seasons,load_prediction_fixtures

@pytest.mark.parametrize('include',[False,True])
def test_awarded_flag_filters_all_event_tables_preserving_unknowns_and_sources(tmp_path,include):
    root=publish(tmp_path/'ordinary');before=snapshot(root)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        data=load_season(root,'Alpha_26_27',tables=['shots','catalog','statistics','matches','pregame'],include_awarded=include,verify_hashes=True)
    assert tuple(data.tables)==('shots','catalog','statistics','matches','pregame')
    for name,frame in data.tables.items():
        if 'event_id' in frame:
            assert event_mask(frame,EVENT+4).any()==include
            assert event_mask(frame,EVENT+1).any() and event_mask(frame,EVENT+5).any()
        else:assert frame.name.tolist()==['constant']
    assert data.provenance['excluded_awarded_event_ids']==([] if include else [EVENT+4])
    assert data.provenance['awarded_flag_available'] and data.matches.is_awarded.isna().sum()==2
    assert snapshot(root)==before

def test_child_only_default_reads_matches_explicit_include_is_selected_only(tmp_path):
    root=publish(tmp_path/'ordinary')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        data=load_season(root,'Alpha_26_27',tables=['statistics','shots'])
    assert data.matches is None and set(data.tables)=={'statistics','shots'}
    assert not event_mask(data.statistics,EVENT+4).any()
    manifest=next((root/'_tables/competition=17').rglob('manifest.json'))
    payload=json.loads(manifest.read_text());payload['tables']['matches']['file']='unavailable.parquet';manifest.write_text(json.dumps(payload))
    included=load_season(root,'Alpha_26_27',tables='shots',include_awarded=True)
    assert len(included.shots)==16 and included.matches is None
    with pytest.raises(FileNotFoundError):load_season(root,'Alpha_26_27',tables='shots')

@pytest.mark.parametrize('api',['shortcut','multiple'])
def test_default_exclusion_reaches_shortcut_and_multiseason_loaders(tmp_path,api):
    root=publish(tmp_path/'ordinary')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        frame=load_season_table(root,'Alpha_26_27',table='shots') if api=='shortcut' else load_seasons(root,'26_27',tables='shots').shots
    assert not event_mask(frame,EVENT+4).any() and len(frame)==(14 if api=='shortcut' else 28)

@pytest.mark.parametrize('value',[None,0,1,'false',np.bool_(True)])
def test_include_awarded_requires_boolean_before_source_reads(tmp_path,value):
    with pytest.raises(TypeError,match='include_awarded'):load_season(tmp_path,'absent',include_awarded=value)
    with pytest.raises(TypeError,match='include_awarded'):load_seasons(tmp_path,'26_27',include_awarded=value)

@pytest.mark.parametrize('layout',['match','team_match'])
def test_awarded_metadata_survives_history_labels_and_assembly(tmp_path,layout):
    root=publish(tmp_path/'ordinary')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning);batch=load_prediction_fixtures(root,include_awarded=True)
    dataset,history=prepared(batch,layout)
    assert history.loc[event_mask(history,EVENT+4),'is_awarded'].eq(True).all()
    assert dataset.metadata.loc[event_mask(dataset.metadata,EVENT+4),'is_awarded'].eq(True).all()
    assert dataset.metadata.loc[event_mask(dataset.metadata,EVENT+1),'is_awarded'].isna().all()

@pytest.mark.parametrize('dtype,large',[('Int64',2**53+1),('UInt64',2**63+9),('int64[pyarrow]',2**53+1),('uint64[pyarrow]',2**63+9)])
def test_event_membership_retains_signed_unsigned_nullable_identity(dtype,large):
    from xdiyo_analytics.data.loading import _event_membership
    values=pd.Series([large,None,7],dtype=dtype,index=[5,1,5]);before=values.copy()
    result=_event_membership(values,{large})
    assert result.tolist()==[True,False,False] and result.index.tolist()==[5,1,5]
    pd.testing.assert_series_equal(values,before)
    assert _event_membership(values.iloc[:0],{large}).empty

def test_null_child_event_is_retained_while_explicit_awarded_event_is_excluded(tmp_path):
    root=publish(tmp_path/'ordinary');manifest=next((root/'_tables/competition=17').rglob('manifest.json'))
    file=manifest.parent/'shots.parquet'
    table=pa.table({'event_id':pa.array([EVENT+4,None,EVENT],type=pa.uint64()),'shot_id':[1,2,3]});pq.write_table(table,file)
    body=json.loads(manifest.read_text());payload=file.read_bytes()
    body['tables']['shots']={'file':file.name,'rows':3,'bytes':len(payload),'sha256':__import__('hashlib').sha256(payload).hexdigest()}
    manifest.write_text(json.dumps(body))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning);shots=load_season(root,'Alpha_26_27',tables='shots').shots
    assert shots.shot_id.tolist()==[2,3] and pd.isna(shots.event_id.iloc[0]) and shots.event_id.iloc[1]==EVENT
