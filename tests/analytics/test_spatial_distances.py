from copy import deepcopy
from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import jensenshannon
from transition_samples import history_from_games, STAT
from test_point_geometry import maps_for
from test_spatial_arithmetic import report
from test_ui_workflow import ui_recipe
from xdiyo_analytics.features import (Heatmap, SpatialHistoricalDeviation as Deviation,
    SpatialFixtureDistance as Distance, SpatialEntropy, SpatialConcentration,
    RollingMean, Lag, Abs, RegionMass, evaluate_features)
from xdiyo_analytics.features.spatial_distance import js_distance, unit_maps
from xdiyo_analytics.labels import create_labels, MatchTotal
from xdiyo_analytics.datasets import assemble_dataset
from xdiyo_analytics.ui import catalog_for_ui, prepare_recipe
from xdiyo_analytics.ui.recipe import export_python


def sample():
    h=history_from_games([{}, {}, {}, {}, {}])
    p=maps_for(h,[(10,20)])
    # Keep exact large integer keys when choosing maps.
    for i,event in enumerate(dict.fromkeys(h.event_id.tolist())):
        mask=p.event_id.map(lambda v:int(v)==int(event)) & p.kind.eq('player')
        p.loc[mask,'x']=10 if i==0 else 90
    return h,p


@pytest.mark.parametrize('n',[2,3,6,7,10,13])
def test_grid_resolutions_and_causal_deviation(n):
    h,p=sample(); source=Heatmap(grid_size=n,kinds=('player',),orientation='team')
    f=evaluate_features(h,{'d':Lag(Deviation(source,window=2)),
        'entropy':RollingMean(SpatialEntropy(source),2),
        'hhi':RollingMean(SpatialConcentration(source),2),
        'region':RollingMean(RegionMass(source,'attacking_third'),2)},heatmaps=p,keyed=True)
    np.testing.assert_allclose(f.d,[np.nan]*4+[1,1]+[jensenshannon([0,1],[.5,.5],base=2)]*2+[0,0],equal_nan=True)
    np.testing.assert_allclose(f.entropy.iloc[2:],0,atol=1e-15)
    np.testing.assert_allclose(f.hhi.iloc[2:],1)
    assert len(report(f,h).artifacts[0].data.maps)==4


def test_js_analytic_symmetry_rotation_and_validation():
    p=np.array([[1,0,0,0],[.1,.2,.3,.4],[np.nan]*4])
    q=np.array([[0,0,1,0],[.4,.3,.2,.1],[1,0,0,0]])
    np.testing.assert_allclose(js_distance(p,q)[:2],[1,jensenshannon(p[1],q[1],base=2)])
    np.testing.assert_allclose(js_distance(p,q),js_distance(q,p),equal_nan=True)
    np.testing.assert_allclose(js_distance(p,q),js_distance(p[:,::-1],q[:,::-1]),equal_nan=True)
    assert np.isnan(js_distance(p,q)[2])
    for values in ([np.nan,0,0,0],[1,1,0,0],[-1,2,0,0],[np.inf,0,0,0],[0,0,0,0]):
        with pytest.raises(ValueError):
            unit_maps(pd.DataFrame([values]),dict(grid_size=2,columns=[0,1,2,3],normalization='mass'),1e-10)


@pytest.mark.parametrize('n',[2,5,12])
def test_all_concentration_normalizers_follow_actual_resolution(n):
    from xdiyo_analytics.features.spatial_distribution import distribution_values
    k=n*n;frame=pd.DataFrame([np.full(k,1/k)])
    frame.attrs['spatial_features']={'uniform':dict(kind='grid',grid_size=n,
        columns=list(frame),normalization='mass',calculation={},operators=[])}
    original=deepcopy(frame)
    expected={'effective_cells':k,'effective_fraction':1,'hhi':1/k,'normalized_hhi':0,
              'largest_cell_share':1/k,'occupied_fraction':1}
    for metric,value in expected.items():
        actual=distribution_values(frame,SpatialConcentration(Heatmap(grid_size=n),metric)).iloc[0,0]
        np.testing.assert_allclose(actual,value,atol=1e-14)
    for normalized,value in ((True,1),(False,np.log(k))):
        np.testing.assert_allclose(distribution_values(frame,SpatialEntropy(Heatmap(grid_size=n),normalized)).iloc[0,0],value)
    pd.testing.assert_frame_equal(frame,original)


def test_missing_slots_cutoffs_late_releases_and_safety():
    h,p=sample()
    times=h.kickoff_at-pd.Timedelta(hours=1)
    available=h.kickoff_at+pd.Timedelta(hours=3)
    available.iloc[2:4]=h.kickoff_at.iloc[-1]+pd.Timedelta(days=1)
    expr=RollingMean(Deviation(window=2),window=2)
    before=evaluate_features(h,{'d':expr},heatmaps=p,cutoffs=times,available_at=available)
    changed=p.copy();event=int(h.event_id.iloc[2])
    changed.loc[changed.event_id.map(lambda x:int(x)==event),'x']=50
    after=evaluate_features(h,{'d':expr},heatmaps=changed,cutoffs=times,available_at=available)
    np.testing.assert_allclose(before,after,equal_nan=True)
    # A missing most recent map occupies the inner slot; don't reach farther back.
    last=int(h.event_id.iloc[4]); missing=p[~p.event_id.map(lambda x:int(x)==last)]
    result=evaluate_features(h,{'d':Lag(Deviation(window=1))},heatmaps=missing)
    assert result.d.iloc[6:].isna().all()
    for expr in (Deviation(),Abs(Deviation())):
        with pytest.raises(ValueError,match='before prediction'):
            evaluate_features(h,{'d':expr},heatmaps=p)
    frozen=times.copy(); frozen.iloc[4:8]=times.iloc[4]
    before=evaluate_features(h,{'d':RollingMean(Deviation(),20)},heatmaps=p,cutoffs=frozen)
    changed=p.copy();future={int(x) for x in h.event_id.iloc[4:]}
    changed.loc[changed.event_id.map(lambda x:int(x) in future),'x']=40
    after=evaluate_features(h,{'d':RollingMean(Deviation(),20)},heatmaps=changed,cutoffs=frozen)
    np.testing.assert_allclose(before.iloc[:8],after.iloc[:8],equal_nan=True)


@pytest.mark.parametrize('n',[3,6,11])
def test_fixture_frames_scope_metadata_and_report(n):
    h,p=sample()
    source=Heatmap(grid_size=n,kinds=('player',),orientation='home')
    f=evaluate_features(h,{k:Distance(source,comparison=k) for k in ('style','home_context','away_context')},heatmaps=p,keyed=True)
    # Both teams' for histories are identical; each against is already reversed.
    np.testing.assert_allclose(f.iloc[2:],0,atol=1e-15)
    label=create_labels(h,{'y':MatchTotal(STAT)})['y']
    ds=assemble_dataset(f.iloc[::-1],label,layout='match')
    assert list(ds.X)==['fixture::style','fixture::home_context','fixture::away_context']
    assert all(s['scope']=='fixture' and set(s['columns'])<=set(ds.X) for s in ds.definitions['spatial_features'].values())
    data=report(f,h).artifacts[0].data
    for key in data.maps:
        panels=data.panel_pair('4',key)['panels']
        assert len(panels)==1 and panels[0]['side']=='fixture'
    cutoffs=h.kickoff_at.copy();cutoffs.iloc[1]-=pd.Timedelta(hours=1)
    with pytest.raises(ValueError,match='identical.*cutoffs'):
        evaluate_features(h,{'d':Distance()},heatmaps=p,cutoffs=cutoffs)
    shuffled=h.iloc[[3,7,2,0,9,6,1,8,4,5]]
    actual=evaluate_features(shuffled,{'style':Distance(source)},heatmaps=p,keyed=True)
    pd.testing.assert_series_equal(actual['style'].sort_index(),f['style'].sort_index())


def test_defaults_exports_and_implicit_heatmap_loading(ui_recipe,monkeypatch):
    from xdiyo_analytics.data import load_seasons
    from xdiyo_analytics.histories import build_team_history
    data=load_seasons(**ui_recipe['data']); h=build_team_history(data)
    data.tables['heatmap_points']=maps_for(h,[(10,20),(80,50)])
    calls=[]
    def load(**kwargs):
        calls.append(kwargs);return deepcopy(data)
    monkeypatch.setattr('xdiyo_analytics.data.load_seasons',load)
    recipe=deepcopy(ui_recipe)
    recipe['features']={'style':{'component':'features.SpatialFixtureDistance','params':{}},
        'deviation':catalog_for_ui().encode(RollingMean(Deviation(),20))}
    recipe['data']['tables']=['matches','statistics','pregame']
    prepared=prepare_recipe(json.loads(json.dumps(recipe)))
    assert 'heatmap_points' in calls[-1]['tables']
    scope={};exec(export_python(recipe).split('result = run_recipe')[0],scope)
    pd.testing.assert_frame_equal(prepared.dataset.X,scope['prepared'].dataset.X)
    assert prepared.dataset.definitions==scope['prepared'].dataset.definitions
    assert Distance().source.grid_size==6 and Heatmap().grid_size==10
    assert catalog_for_ui().build(catalog_for_ui().encode(Distance()))==Distance()


@pytest.mark.parametrize('normalization',['mass','density'])
@pytest.mark.parametrize('n',[3,7])
def test_gaussian_resolution_and_exact_external_js(normalization,n):
    from xdiyo_analytics.features.spatial import heatmap_values
    h,p=sample();p.loc[p.team_id.map(int)==int(h.team_id.iloc[1]),'y']=80
    source=Heatmap(grid_size=n,resolution=n*4,method='gaussian',kinds=('player',),
                   normalization=normalization,orientation='team')
    frame=heatmap_values(h,p,source);spec=next(iter(frame.attrs['spatial_features'].values()))
    maps=unit_maps(frame,spec,1e-10)
    expected=jensenshannon(maps[0],maps[1],base=2)
    f=evaluate_features(h,{'d':Distance(source,window=1)},heatmaps=p)
    np.testing.assert_allclose(f.d.iloc[2:4],expected)
    with pytest.raises(ValueError,match='multiple'):
        replace(source,resolution=n*4+1)


def test_directional_opponent_contexts_and_separate_leagues():
    from transition_samples import A,B,C,D
    h=history_from_games([dict(home=A,away=C,day=0),dict(home=B,away=D,day=1),
        dict(home=A,away=B,day=3),dict(home=A,away=B,day=4,competition=20)])
    p=maps_for(h,[(10,20)])
    p.loc[p.team_id.map(lambda v:int(v)==B)&p.kind.eq('player'),'x']=90
    f=evaluate_features(h,{k:Distance(comparison=k) for k in ('style','home_context','away_context')},heatmaps=p)
    np.testing.assert_allclose(f.iloc[4:6],[[1,0,1],[1,0,1]])
    assert f.iloc[6:].isna().all().all()
    # One absent fixture row is unavailable, never paired with an unrelated row.
    partial=evaluate_features(h.drop(index=5),{'d':Distance()},heatmaps=p)
    assert np.isnan(partial.loc[4,'d'])


def test_temporal_fixture_scope_and_arithmetic_reporter_metadata():
    h,p=sample()
    f=evaluate_features(h,{'lagged':Lag(Distance()),'absolute':Abs(Distance())},heatmaps=p,keyed=True)
    data=report(f,h).artifacts[0].data
    assert len(data.panel_pair('4','lagged')['panels'])==2
    assert all(not v['missing'] for v in data.panel_pair('4','lagged')['panels'])
    assert len(data.panel_pair('4','absolute')['panels'])==1


@pytest.mark.parametrize('method',['pca','clusters'])
def test_native_embedding_recipe_fit_save_and_export(ui_recipe,monkeypatch,tmp_path,method):
    from xdiyo_analytics.data import load_seasons
    from xdiyo_analytics.histories import build_team_history
    from xdiyo_analytics.ui import run_recipe
    from xdiyo_analytics.ui.recipe import export_notebook
    from xdiyo_analytics.training import save_model,load_model
    from pathlib import Path
    from runpy import run_path
    example=run_path(str(Path(__file__).resolve().parents[2]/'examples/spatial_extensions.py'))
    embedding_recipe_step,keeper_definitions=example['embedding_recipe_step'],example['keeper_definitions']
    data=load_seasons(**ui_recipe['data']);h=build_team_history(data)
    data.tables['heatmap_points']=maps_for(h,[(10,20),(80,50)])
    points=data.tables['heatmap_points']
    points['x']=points.event_id.map(lambda v:10+int(v)%5*18)
    monkeypatch.setattr('xdiyo_analytics.data.load_seasons',lambda **kwargs:deepcopy(data))
    recipe=deepcopy(ui_recipe)
    recipe['features']={'map':catalog_for_ui().encode(RollingMean(Heatmap(grid_size=3,kinds=('player',),orientation='team'),20)),
        **{k:catalog_for_ui().encode(v) for k,v in keeper_definitions(3).items()}}
    prepared=prepare_recipe(recipe)
    recipe['preprocessors'].insert(0,embedding_recipe_step(prepared,'map',method,dimensions=2))
    result=run_recipe(recipe,prepared=prepared)
    fitted=result.training.folds[0].model.estimator.named_steps['prepare_0']
    fold=result.training.folds[0]
    assert fitted.n_training_maps_<=2*len(fold.train_positions)
    save_model(result.training,tmp_path/'saved')
    loaded=load_model(tmp_path/'saved')
    np.testing.assert_allclose(loaded.model.estimator.named_steps['prepare_0'].transform(prepared.dataset.X),fitted.transform(prepared.dataset.X))
    scope={};exec(export_python(recipe).split('result = run_recipe')[0],scope)
    assert scope['recipe']['preprocessors']==recipe['preprocessors']
    for cell in export_notebook(recipe)['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),'exported','exec')
