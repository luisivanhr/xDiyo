from datetime import date, datetime, timezone
import inspect
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from xdiyo_analytics.experiments.recovery import dump_bundle,load_bundle,pack,unpack,_pack_dtype

@pytest.mark.parametrize('dtype,values',[
    ('Int64',[2**53+1,None,-3]),('UInt64',[2**63+9,None,7]),
    ('boolean',[True,None,False]),('Float64',[1.25,None,-2.]),
    ('string[python]',['home',None,'away']),('string[pyarrow]',['home',None,'away']),
    ('object',[('target',1),None,'draw']),
])
def test_nullable_labels_ids_and_values_roundtrip_exactly(tmp_path,dtype,values):
    frame=pd.DataFrame({'value':pd.Series(values,dtype=dtype)})
    frame.index=pd.Index([2**53+1,5,3],dtype='int64',name='row')
    path=tmp_path/'bundle.json';dump_bundle(path,frame)
    pd.testing.assert_frame_equal(load_bundle(path),frame)
    assert 'NaN' not in path.read_text()

@pytest.mark.parametrize('shape',['ordinary','empty','unused_classes','nullable_class_level'])
def test_probability_multiindex_schema_and_class_identity_roundtrip(shape):
    classes=pd.Index([0,1,2],name='class',dtype='Int64' if shape=='nullable_class_level' else 'int64')
    columns=pd.MultiIndex.from_product([['outcome'],classes],names=['target','class'])
    frame=pd.DataFrame([[.2,.3,.5],[.4,.5,.1]],columns=columns,index=pd.Index([17,5],name='original_row'))
    if shape=='empty':frame=frame.iloc[:0]
    if shape=='unused_classes':frame=frame.iloc[:,:2]
    result=unpack(pack(frame))
    pd.testing.assert_frame_equal(result,frame)
    for actual,expected in zip(result.columns.levels,frame.columns.levels):
        pd.testing.assert_index_equal(actual,expected)

def test_categorical_timezone_arrays_and_plotly_artifacts(tmp_path):
    frame=pd.DataFrame({'label':pd.Series(pd.Categorical(['away',None,'home'],categories=['home','draw','away'],ordered=True)),
        'time':pd.Series(pd.to_datetime(['2026-01-01',None,'2026-01-03'],utc=True))})
    figure=go.Figure(go.Scatter(x=[1,2],y=[3,4]))
    payload={'frame':frame,'empty':np.empty((0,2),dtype='float32'),
        'array':np.array([[2**63+9]],dtype='uint64'),'figure':figure,'missing':(pd.NA,pd.NaT,np.inf,-np.inf,np.nan)}
    path=tmp_path/'artifact.json';dump_bundle(path,payload);result=load_bundle(path)
    pd.testing.assert_frame_equal(result['frame'],frame)
    np.testing.assert_array_equal(result['array'],payload['array'])
    assert result['empty'].shape==(0,2) and result['empty'].dtype=='float32'
    assert json.loads(result['figure'].to_json())==json.loads(figure.to_json())
    assert result['missing'][0] is pd.NA and result['missing'][1] is pd.NaT
    assert np.isinf(result['missing'][2]) and np.isnan(result['missing'][4])

def test_data_only_bundle_rejects_unknown_objects_schema_and_foreign_record(tmp_path):
    with pytest.raises(TypeError,match='data-only'):pack(object())
    path=tmp_path/'bad.json';path.write_text(json.dumps({'schema':99,'payload':None}))
    with pytest.raises(ValueError,match='schema'):load_bundle(path)
    with pytest.raises(ValueError,match='module'):
        unpack({'@':'record','module':'pathlib','name':'Path','fields':{}})

@pytest.mark.parametrize('storage',['python','pyarrow'])
def test_string_index_and_multiindex_levels_preserve_storage(storage):
    index=pd.Index(['home','away'],dtype=pd.StringDtype(storage=storage),name='class')
    pd.testing.assert_index_equal(unpack(pack(index)),index)
    multi=pd.MultiIndex.from_product([['outcome'],index],names=['target','class'])
    actual=unpack(pack(multi))
    pd.testing.assert_index_equal(actual,multi)
    pd.testing.assert_index_equal(actual.levels[1],multi.levels[1])

def test_previous_tuple_multiindex_and_dtype_string_bundles_remain_readable(tmp_path):
    old_index={'@':'rangeindex','start':0,'stop':2,'step':1,'name':None}
    old_series={'@':'series','index':old_index,'name':'label','values':['home',None],'dtype':'string'}
    old_categories={'@':'series','index':old_index,'name':'result','values':['home','away'],
        'dtype':'category','categories':{'@':'index','values':['away','draw','home'],'dtype':'object','name':None},'ordered':True}
    old_multi={'@':'multiindex','values':[{'@':'tuple','values':['outcome',0]},{'@':'tuple','values':['outcome',1]}],
        'names':['target','class']}
    path=tmp_path/'old.json';path.write_text(json.dumps({'schema':1,'payload':[old_series,old_categories,old_multi]}))
    series,categories,multi=load_bundle(path)
    pd.testing.assert_series_equal(series,pd.Series(['home',None],dtype='string',name='label'))
    assert categories.cat.categories.tolist()==['away','draw','home'] and categories.cat.ordered
    pd.testing.assert_index_equal(multi,pd.MultiIndex.from_tuples([('outcome',0),('outcome',1)],names=['target','class']))


@pytest.mark.parametrize('storage',['python','pyarrow'])
@pytest.mark.parametrize('sentinel',['NA','nan'])
def test_string_sentinel_values_mask_and_storage_roundtrip(tmp_path,storage,sentinel):
    if storage=='pyarrow':pytest.importorskip('pyarrow')
    supports_nan='na_value' in inspect.signature(pd.StringDtype).parameters
    if sentinel=='nan' and not supports_nan:
        pytest.skip('This pandas version cannot represent the NaN string sentinel.')
    dtype=(pd.StringDtype(storage=storage,na_value=np.nan) if sentinel=='nan' else
           pd.StringDtype(storage=storage))
    series=pd.Series(['home',None,'away'],dtype=dtype,name='label')
    payload={'series':series,'frame':series.to_frame(),'index':pd.Index(series.array,name='class')}
    path=tmp_path/'string-sentinel.json';dump_bundle(path,payload)
    restored=load_bundle(path)
    pd.testing.assert_series_equal(restored['series'],series)
    pd.testing.assert_frame_equal(restored['frame'],payload['frame'])
    pd.testing.assert_index_equal(restored['index'],payload['index'])
    for actual in (restored['series'],restored['frame']['label'],restored['index']):
        assert actual.dtype.storage==storage
        np.testing.assert_array_equal(pd.isna(actual),[False,True,False])
        if sentinel=='NA':assert actual.dtype.na_value is pd.NA
        else:assert np.isnan(actual.dtype.na_value)


def archived_string_series(storage,sentinel='nan'):
    dtype=(f'string[{storage}]' if sentinel=='legacy' else
           {'kind':'string','storage':storage,'na_value':{'@':'NA'} if sentinel=='NA' else
            {'@':'float','value':'nan'}})
    return {'@':'series','index':{'@':'rangeindex','start':0,'stop':3,'step':1,'name':None},
            'name':'label','values':['home',{'@':'float','value':'nan'},'away'],'dtype':dtype}


@pytest.mark.parametrize('storage',['python','pyarrow'])
@pytest.mark.parametrize('sentinel',['nan','NA','legacy'])
def test_archived_string_descriptors_load_across_supported_pandas(tmp_path,storage,sentinel):
    if storage=='pyarrow':pytest.importorskip('pyarrow')
    path=tmp_path/'archived-string.json'
    path.write_text(json.dumps({'schema':1,'payload':archived_string_series(storage,sentinel)}))
    restored=load_bundle(path)
    assert restored.name=='label' and restored.dtype.storage==storage
    assert restored.dropna().tolist()==['home','away']
    np.testing.assert_array_equal(restored.isna(),[False,True,False])
    if sentinel=='nan' and 'na_value' in inspect.signature(pd.StringDtype).parameters:
        assert np.isnan(restored.iloc[1]) and np.isnan(restored.dtype.na_value)
    else:
        assert restored.iloc[1] is pd.NA and restored.dtype.na_value is pd.NA


@pytest.mark.parametrize('storage',['python','pyarrow'])
def test_nan_string_descriptor_adapts_to_pandas2_constructor(monkeypatch,storage):
    if storage=='pyarrow':pytest.importorskip('pyarrow')
    original=pd.StringDtype
    class Pandas2StringDtype(original):
        def __init__(self,storage=None):
            super().__init__(storage=storage)
    with monkeypatch.context() as patch:
        patch.setattr(pd,'StringDtype',Pandas2StringDtype)
        restored=unpack(archived_string_series(storage))
    assert restored.dtype.storage==storage and restored.dtype.na_value is pd.NA
    assert restored.iloc[1] is pd.NA
    assert restored.dropna().tolist()==['home','away']
    np.testing.assert_array_equal(restored.isna(),[False,True,False])


def test_encoding_string_dtype_without_explicit_missing_sentinel_defaults_to_na():
    class LegacyStringDtype(pd.StringDtype):
        @property
        def na_value(self):
            raise AttributeError('Legacy dtype exposes only its default missing sentinel')
    assert _pack_dtype(LegacyStringDtype(storage='python'))=={
        'kind':'string','storage':'python','na_value':{'@':'NA'}}


@pytest.mark.parametrize('value',[date.min,date(2024,2,29),date.max])
def test_plain_date_uses_distinct_data_only_tag(value):
    encoded=pack(value)
    assert encoded=={'@':'date','value':value.isoformat()}
    restored=unpack(json.loads(json.dumps(encoded)))
    assert type(restored) is date and restored==value


def date_container_payload():
    value=date(2024,2,29)
    series=pd.Series([value,None,date(2025,1,3)],dtype=object,name=value)
    series.attrs={'source':{'as_of':value},value:'date key'}
    index=pd.Index([value,None,date(2025,1,3)],dtype=object,name=value)
    frame=series.to_frame()
    frame.index=index
    frame.attrs={'window':(value,date(2025,1,3))}
    return {'scalar':value,'series':series,'frame':frame,'index':index}


def assert_date_container_payload(actual,expected):
    assert type(actual['scalar']) is date and actual['scalar']==expected['scalar']
    pd.testing.assert_series_equal(actual['series'],expected['series'])
    pd.testing.assert_frame_equal(actual['frame'],expected['frame'])
    pd.testing.assert_index_equal(actual['index'],expected['index'])
    for values in (actual['series'],actual['frame'].iloc[:,0],actual['frame'].index,actual['index']):
        assert [type(item) for item in values]==[date,type(None),date]
    assert type(actual['series'].name) is date
    assert type(actual['index'].name) is date
    assert type(actual['frame'].columns[0]) is date
    assert actual['series'].attrs==expected['series'].attrs
    assert type(actual['series'].attrs['source']['as_of']) is date
    assert type(next(key for key in actual['series'].attrs if key!='source')) is date
    assert actual['frame'].attrs==expected['frame'].attrs
    assert all(type(item) is date for item in actual['frame'].attrs['window'])


def test_plain_dates_survive_object_containers_axes_and_attrs(tmp_path):
    expected=date_container_payload()
    path=tmp_path/'dates.json';dump_bundle(path,expected)
    assert_date_container_payload(load_bundle(path),expected)


def test_plain_dates_survive_saved_football_experiment_outputs(tmp_path):
    from football_experiment_samples import prepared,ridge
    from xdiyo_analytics.experiments import FootballExperiment
    preparation=prepared(holdout=True)
    expected=date_container_payload()
    preparation.outputs['dates']=expected
    experiment=FootballExperiment('Date output fidelity',output_dir=tmp_path)
    result=experiment.run(preparation,model=ridge())
    loaded=experiment.load(result.record['run_id'])
    assert_date_container_payload(loaded.prepared.outputs['dates'],expected)


@pytest.mark.parametrize('value',[datetime(2025,1,2,3,4,5),
                                  datetime(2025,1,2,3,4,5,tzinfo=timezone.utc),
                                  pd.Timestamp('2025-01-02 03:04:05.123456789',tz='Asia/Tokyo')])
def test_datetime_and_timestamp_keep_existing_timestamp_encoding(value):
    encoded=pack(value)
    assert encoded=={'@':'timestamp','value':value.isoformat()}
    restored=unpack(encoded)
    assert isinstance(restored,pd.Timestamp) and restored==pd.Timestamp(value)


def test_legacy_date_only_timestamp_remains_readable():
    restored=unpack({'@':'timestamp','value':'2024-02-29'})
    assert isinstance(restored,pd.Timestamp) and restored==pd.Timestamp('2024-02-29')



def object_array_payload():
    matches=np.empty(2,dtype=object)
    matches[0]=(17,101);matches[1]=(18,102)
    nested=np.empty(3,dtype=object)
    nested[0]=['goals',[1,2]]
    nested[1]={'match':(17,101),'values':np.array([3,5],dtype='int16')}
    nested[2]=np.array([[7,9]],dtype='float32')
    matrix=np.empty((2,2),dtype=object)
    matrix[0,0]=(1,2);matrix[0,1]=[3,4]
    matrix[1,0]=();matrix[1,1]=[]
    payload={'matches':matches,'nested':nested,'matrix':matrix,
             'empty_rows':np.empty((0,3),dtype=object),
             'empty_middle':np.empty((2,0,3),dtype=object)}
    for name,cell in [('tuple',(17,101)),('list',[1,[2,3]]),
                      ('dict',{'match':(17,101)}),('array',matches)]:
        scalar=np.empty((),dtype=object);scalar[()]=cell
        payload['scalar_'+name]=scalar
    return payload


def assert_recovered_array_cells(actual,expected):
    assert type(actual) is type(expected)
    if isinstance(expected,np.ndarray):
        assert actual.shape==expected.shape and actual.dtype==expected.dtype
        if expected.dtype==object:
            for index in np.ndindex(expected.shape):
                assert_recovered_array_cells(actual[index],expected[index])
        else:
            np.testing.assert_array_equal(actual,expected)
    elif isinstance(expected,dict):
        assert actual.keys()==expected.keys()
        for key in expected:assert_recovered_array_cells(actual[key],expected[key])
    elif isinstance(expected,(tuple,list)):
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected):assert_recovered_array_cells(a,b)
    else:
        assert actual==expected


@pytest.mark.parametrize('name',list(object_array_payload()))
def test_object_arrays_keep_cell_types_and_recorded_rank(tmp_path,name):
    expected=object_array_payload()[name]
    path=tmp_path/'objects.json';dump_bundle(path,expected)
    restored=load_bundle(path)
    assert_recovered_array_cells(restored,expected)
    assert json.loads(path.read_text())['payload']['shape']==list(expected.shape)


def test_archived_tuple_object_array_uses_existing_nested_values_format():
    archived={'@':'array','dtype':'object','shape':[2],
              'values':[{'@':'tuple','values':[17,101]},{'@':'tuple','values':[18,102]}]}
    expected=object_array_payload()['matches']
    assert pack(expected)==archived
    assert_recovered_array_cells(unpack(archived),expected)


@pytest.mark.parametrize('values',[[1],[1,2,3]])
def test_object_array_rejects_inconsistent_recorded_shape(values):
    with pytest.raises(ValueError,match='recorded shape'):
        unpack({'@':'array','dtype':'object','shape':[2],'values':values})


@pytest.mark.parametrize('value',[np.array([[1,2],[3,4]],dtype='float32'),
                                 np.array(7,dtype='int16'),np.empty((0,2),dtype='uint64')])
def test_nonobject_array_decoding_keeps_dtype_shape_and_values(value):
    assert_recovered_array_cells(unpack(pack(value)),value)


def test_object_arrays_survive_saved_football_experiment_outputs(tmp_path):
    from football_experiment_samples import prepared,ridge
    from xdiyo_analytics.experiments import FootballExperiment
    preparation=prepared(holdout=True)
    expected=object_array_payload()
    preparation.outputs['object_arrays']=expected
    experiment=FootballExperiment('Object array output fidelity',output_dir=tmp_path)
    result=experiment.run(preparation,model=ridge())
    loaded=experiment.load(result.record['run_id'])
    assert_recovered_array_cells(loaded.prepared.outputs['object_arrays'],expected)
