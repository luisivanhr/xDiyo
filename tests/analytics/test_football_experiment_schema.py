import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from xdiyo_analytics.experiments.recovery import dump_bundle,load_bundle,pack,unpack

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
