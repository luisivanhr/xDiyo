"""Native unary absolute value: composition, causality, lineage and recipes."""
from copy import deepcopy
import json
import numpy as np
import pandas as pd
import pytest

from transition_samples import STAT,history_from_games
from test_point_geometry import maps_for
from test_spatial_arithmetic import report
from test_ui_workflow import ui_recipe
from xdiyo_analytics.features import (Abs,Column,Constant,Difference,Product,Ratio,Sum,
    SpatialPointSummary as Point,RollingMean,RollingStd,RollingZScore,Lag,EMA,H2H,
    combine_features,evaluate_features)
from xdiyo_analytics.ui import catalog_for_ui,prepare_recipe
from xdiyo_analytics.ui.recipe import node,export_python
from xdiyo_analytics.experiments.recovery import dump_bundle,load_bundle,signature,pack


def test_abs_values_nullable_zeros_and_nonfinite_inputs():
    frame=pd.DataFrame({'x':pd.array([-3,0,-0.,pd.NA,np.inf,-np.inf,1e308],dtype='Float64')},index=[9,2,2,1,8,4,3])
    frame.attrs={'provenance':{'source':'fixture'}};before=deepcopy(frame)
    result=combine_features(frame,{'abs':Abs(Column('x')),'nested':Sum(Abs(Column('x')),1),'zero':Abs(Constant(-0.))})
    np.testing.assert_allclose(result['abs'],[3,0,0,np.nan,np.nan,np.nan,1e308],equal_nan=True)
    assert not np.signbit(result['abs'].to_numpy()[1:3]).any()
    assert result['zero'].eq(0).all() and result.index.equals(frame.index)
    assert result.attrs['provenance']==frame.attrs['provenance']
    pd.testing.assert_frame_equal(frame,before)


def test_inside_and_outside_mean_have_different_order():
    h=history_from_games([{}, {}, {}]);p=maps_for(h,[(10,20)])
    p.loc[p.event_id==h.event_id.iloc[2],'x']=90
    source=Difference(Point(),Constant(50))
    f=evaluate_features(h,{'inside':RollingMean(Abs(source),20),'outside':Abs(RollingMean(source,20))},heatmaps=p,keyed=True)
    np.testing.assert_allclose(f.inside,[np.nan,np.nan,40,40,40,40],equal_nan=True)
    np.testing.assert_allclose(f.outside,[np.nan,np.nan,40,40,0,0],equal_nan=True)
    inside=f.attrs['spatial_features']['inside']['calculation']
    outside=f.attrs['spatial_features']['outside']['calculation']
    assert inside['operator']=='RollingMean' and inside['source']['operator']=='Abs'
    assert outside['operator']=='Abs' and outside['source']['operator']=='RollingMean'
    assert 'Absolute value' in next(iter(report(f,h).artifacts[0].data.maps.values()))['spec']['calculation_label']


@pytest.mark.parametrize('wrapper',[lambda x:Lag(x),lambda x:EMA(x,span=2),lambda x:RollingStd(x,2),lambda x:RollingMean(x,2)])
def test_temporal_composition_missing_windows(wrapper):
    h=history_from_games([{}, {}, {}, {}]);p=maps_for(h,[(20,30)])
    missing_ids={int(value) for value in h.event_id.iloc[2:6]}
    p=p[~p.event_id.map(lambda value:int(value) in missing_ids)]
    expr=wrapper(Abs(Difference(Point(),50)))
    f=evaluate_features(h,{'a':expr},heatmaps=p)
    assert np.isnan(f.iloc[0,0])
    if not isinstance(expr,EMA):assert np.isnan(f.iloc[-1,0])
    audit=f.attrs['spatial_point_audit']['features']['a']
    assert len(audit['maps'])==8 and len(audit['outputs'])==8
    assert len(audit['history'])==8


def test_cutoff_availability_h2h_safety_and_single_output():
    h=history_from_games([dict(day=0),dict(day=2),dict(day=4),dict(day=4),dict(day=6)])
    p=maps_for(h,[(10,20)])
    releases=h.kickoff_at+pd.Timedelta(hours=3)
    releases.iloc[2:4]=h.kickoff_at.iloc[-1]+pd.Timedelta(days=1)
    settings=dict(cutoffs=h.kickoff_at-pd.Timedelta(hours=1),available_at=releases)
    expr=H2H(RollingMean(Abs(Difference(Point(),50)),3))
    before=evaluate_features(h,{'a':expr},heatmaps=p,**settings)
    changed_ids={int(value) for value in h.event_id.iloc[2:]}
    q=p.copy();q.loc[q.event_id.map(lambda value:int(value) in changed_ids),'x']=99
    after=evaluate_features(h,{'a':expr},heatmaps=q,**settings)
    np.testing.assert_allclose(before.iloc[:8],after.iloc[:8],equal_nan=True)
    for expr in (Abs(Point()),Abs(STAT),Abs(Sum(Point(),1))):
        with pytest.raises(ValueError,match='before prediction'):
            evaluate_features(h,{'unsafe':expr},heatmaps=p)
    with pytest.raises(ValueError,match='one output column'):
        evaluate_features(h,{'multi':Abs(Lag(Point(side='both')))},heatmaps=p)
    with pytest.raises(ValueError,match='same H2H'):
        evaluate_features(h,{'scopes':Sum(Abs(H2H(Lag(Point()))),Lag(Point()))},heatmaps=p)
    with pytest.raises(ValueError,match='reference must be historical'):
        evaluate_features(h,{'unsafe_ref':RollingZScore(Point(),2,reference=Abs(Point()))},heatmaps=p)


def test_spatial_lineage_cache_order_and_persistence(tmp_path):
    h=history_from_games([{}, {}, {}]);p=maps_for(h,[(10,20),(30,40)])
    base=RollingMean(Difference(Point(),50),2)
    features={'base':base,'abs':Abs(base),'repeat':Sum(Abs(base),Abs(base)),
              'missing':Abs(Ratio(base,Constant(0))),'overflow':Abs(Product(base,Constant(1e308)))}
    f=evaluate_features(h,features,heatmaps=p,keyed=True)
    reverse=evaluate_features(h,dict(reversed(list(features.items()))),heatmaps=p,keyed=True)
    assert signature(f.attrs)==signature(reverse.attrs)
    for name in features:
        audit=f.attrs['spatial_point_audit']['features'][name]
        assert len(audit['maps'])==6 and len(audit['history'])==6
    assert f['missing'].isna().all() and f['overflow'].isna().all()
    spec=f.attrs['spatial_features']['abs']
    assert spec['derived'] and spec['calculation']['operator']=='Abs'
    assert f.attrs['spatial_features']['base']['calculation']['operator']=='RollingMean'
    report(f,h)
    f.to_parquet(tmp_path/'abs.parquet',compression='zstd')
    restored=pd.read_parquet(tmp_path/'abs.parquet')
    assert pack(restored.attrs)==pack(json.loads(json.dumps(f.attrs)))
    pd.testing.assert_frame_equal(restored,f,check_index_type=False)
    dump_bundle(tmp_path/'abs.json',f);recovered=load_bundle(tmp_path/'abs.json')
    assert signature(recovered)==signature(f)
    assert signature(Abs(base))!=signature(base)


def test_native_builder_exports_and_prepared_wrapper(ui_recipe,monkeypatch):
    from xdiyo_analytics.data import load_seasons
    from xdiyo_analytics.histories import build_team_history
    loaded=load_seasons(**ui_recipe['data']);h=build_team_history(loaded)
    loaded.tables['heatmap_points']=maps_for(h,[(10,20),(30,40)])
    monkeypatch.setattr('xdiyo_analytics.data.load_seasons',lambda **kwargs:deepcopy(loaded))
    recipe=deepcopy(ui_recipe)
    expr=RollingMean(Abs(Difference(Point(),50)),20)
    recipe['features']={'abs':catalog_for_ui().encode(expr)}
    recipe['derived_features']={'magnitude':node('prepared.Abs',source=node('prepared.Column',name='home::abs'))}
    recipe=json.loads(json.dumps(recipe))
    prepared=prepare_recipe(recipe)
    scope={};exec(export_python(recipe).split('result = run_recipe')[0],scope)
    pd.testing.assert_frame_equal(prepared.dataset.X,scope['prepared'].dataset.X)
    assert prepared.dataset.definitions==scope['prepared'].dataset.definitions
    assert prepared.dataset.X['magnitude'].equals(prepared.dataset.X['home::abs'].rename('magnitude'))
    schemas={s['id']:s for s in catalog_for_ui().schema()}
    assert schemas['features.Abs']['fields'][0]['primary']
    assert schemas['prepared.Abs']['fields'][0]['categories']==['derived_feature']


def test_abs_preserves_fixture_scope():
    from test_match_score import score_data
    from xdiyo_analytics.histories import build_team_history
    from xdiyo_analytics.features import BayesianFixture
    from xdiyo_analytics.labels import Outcome,create_labels
    from xdiyo_analytics.datasets import assemble_dataset
    h=build_team_history(score_data(3))
    raw=Difference(BayesianFixture(fields=('p_draw',)),.5)
    f=evaluate_features(h,{'signed':raw,'magnitude':Abs(raw)},keyed=True)
    label=create_labels(h,{'target':Outcome(perspective='home')})['target']
    dataset=assemble_dataset(f,label,layout='match')
    assert list(dataset.X)==['fixture::signed','fixture::magnitude']
    np.testing.assert_allclose(dataset.X['fixture::magnitude'],np.abs(dataset.X['fixture::signed']))
