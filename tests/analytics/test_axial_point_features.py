"""Per-match axial encoding, causal native recipes and optional reporting."""
from copy import deepcopy
import json
import numpy as np
import pandas as pd
import pytest

from transition_samples import history_from_games
from test_point_geometry import maps_for, observed
from test_spatial_arithmetic import report
from test_ui_workflow import ui_recipe
from xdiyo_analytics.features import SpatialPointSummary as Point, RollingMean, Product, Constant, evaluate_features
from xdiyo_analytics.reporting import axial_direction_summary
from xdiyo_analytics.ui import prepare_recipe
from xdiyo_analytics.ui.recipe import export_python
from pathlib import Path
from runpy import run_path

_example=run_path(str(Path(__file__).resolve().parents[2]/'examples/axial_point_features.py'))
feature_specs,recipe_features=_example['feature_specs'],_example['recipe_features']


def line(degrees):
    direction=np.array([np.cos(np.deg2rad(degrees)),np.sin(np.deg2rad(degrees))])
    return 50+np.array([-20.,20.])[:,None]*direction


def angular_history(angles):
    h=history_from_games([{} for _ in angles])
    p=pd.concat([maps_for(h[h.event_id==event],line(angle))
                 for event,angle in zip(h.event_id.unique(),angles)],ignore_index=True)
    return h,p


@pytest.mark.parametrize('angles,r,mean', [([1,179],np.cos(np.deg2rad(2)),0),([32,32],1,32),([0,90],0,None)])
def test_axial_means_and_consistency(angles,r,mean):
    h,p=angular_history([*angles,45])
    f=evaluate_features(h,feature_specs(),heatmaps=p)
    c,s=f.to_numpy()[-1]
    got=axial_direction_summary(c,s)
    assert got['consistency']==pytest.approx(r,abs=1e-14)
    if mean is None: assert np.isnan(got['mean_axis'])
    else: assert abs(np.angle(np.exp(2j*(got['mean_axis']-np.deg2rad(mean))))) < 1e-13
    if angles==[1,179]:
        assert .9993 < np.hypot(c,s) < 1  # Never normalize the average pair.
        old=evaluate_features(h,{'raw_mean':RollingMean(Point('axis_angle'),20)},heatmaps=p)
        assert old.iloc[-1,0]==pytest.approx(np.pi/2)  # Compatibility retained.


@pytest.mark.parametrize('angle', [0,1,44,90,123,179,180,359,-1])
def test_encoding_angle_formula_period_and_coordinate_transforms(angle):
    h,p=angular_history([angle])
    def channels(points):
        return np.array([observed(h,points,field=f).iloc[0,0] for f in ('axis_cos2','axis_sin2')])
    base=channels(p)
    np.testing.assert_allclose(base,[np.cos(2*np.deg2rad(angle)),np.sin(2*np.deg2rad(angle))],atol=1e-14)
    _,shift=angular_history([angle+180])
    np.testing.assert_allclose(base,channels(shift),atol=1e-14)
    for transform,expected in [('rotation',base),('reflect_x',base*[1,-1]),('reflect_y',base*[1,-1]),('swap',base*[-1,1])]:
        q=p.copy()
        if transform=='rotation':q[['x','y']]=100-q[['x','y']]
        elif transform=='reflect_x':q['x']=100-q.x
        elif transform=='reflect_y':q['y']=100-q.y
        else:q[['x','y']]=q[['y','x']].to_numpy()
        np.testing.assert_allclose(channels(q),expected,atol=1e-14)
    np.testing.assert_allclose(channels(p.sample(frac=1,random_state=12)),base,atol=1e-14)
    for field,value in zip(('axis_cos2','axis_sin2'),base):
        np.testing.assert_allclose(observed(h,p,field=field,side='against'),value,atol=1e-14)


def test_boundary_continuity():
    pairs=[]
    for angle in (-1e-7,1e-7,180-1e-7,180+1e-7):
        h,p=angular_history([angle])
        pairs.append([observed(h,p,field=f).iloc[0,0] for f in ('axis_cos2','axis_sin2')])
    np.testing.assert_allclose(pairs,np.tile([1,0],(4,1)),atol=4e-9)


@pytest.mark.parametrize('delta,valid', [(0,False),(1e-15,False),(1e-10,True)])
def test_near_isotropy_uses_existing_gap_rule(delta,valid):
    h=history_from_games([{}]);xy=50+np.array([[-20*(1+delta),0],[20*(1+delta),0],[0,-20],[0,20]])
    p=maps_for(h,xy)
    masks=[np.isfinite(observed(h,p,field=f).to_numpy()) for f in ('axis_angle','axis_cos2','axis_sin2')]
    assert all((m==valid).all() for m in masks)


@pytest.mark.parametrize('xy,min_points', [([(.1,3.4)]*7,1), ([(10,20)],1), (line(20),3),
    ([(20,20),(20,80),(80,20),(80,80)],1)])
def test_undefined_axes_are_missing_in_both_channels(xy,min_points):
    h=history_from_games([{}]);p=maps_for(h,xy)
    for field in ('axis_cos2','axis_sin2'):
        assert observed(h,p,field=field,min_points=min_points).isna().all().all()


def test_missing_masks_fixed_windows_and_min_periods():
    h,p=angular_history([10]*5)
    events=h.event_id.unique()
    p=p[p.event_id!=events[1]]
    p=p[~((p.event_id==events[2]) & (p.kind=='player'))]
    for minimum in (1,2):
        features={f:RollingMean(Point(f),2,min_periods=minimum) for f in ('axis_cos2','axis_sin2')}
        out=evaluate_features(h,features,heatmaps=p)
        assert out.iloc[:,0].isna().equals(out.iloc[:,1].isna())
        assert out.iloc[6:8].isna().all().all() # Empty/missing maps occupy last two slots.
        assert out.iloc[8:].isna().all().all()==(minimum==2)
        summary=axial_direction_summary(*out.to_numpy().T)
        assert np.isnan(summary['consistency']).tolist()==out.iloc[:,0].isna().tolist()
        assert np.isnan(summary['mean_axis']).tolist()==out.iloc[:,0].isna().tolist()


def test_raw_safety_future_mutation_and_lineage():
    h,p=angular_history([10,30,50,70])
    for field in ('axis_cos2','axis_sin2'):
        with pytest.raises(ValueError,match='before prediction'):
            evaluate_features(h,{'raw':Product(Point(field),Constant(2))},heatmaps=p)
    cutoffs=h.kickoff_at-pd.Timedelta(hours=1)
    release=h.kickoff_at+pd.Timedelta(hours=3);release.iloc[2:4]=h.kickoff_at.iloc[-1]+pd.Timedelta(days=1)
    features={**feature_specs(),'derived':Product(feature_specs()['activity_axis_cos2_r20'],Constant(2))}
    a=evaluate_features(h,features,heatmaps=p,cutoffs=cutoffs,available_at=release,keyed=True)
    q=p.copy();q.loc[q.event_id.isin(h.event_id.iloc[4:]),'y']=50
    b=evaluate_features(h,features,heatmaps=q,cutoffs=cutoffs,available_at=release,keyed=True)
    np.testing.assert_allclose(a.to_numpy()[:6],b.to_numpy()[:6],equal_nan=True)
    report(a,h)
    assert a.attrs['spatial_features']['derived']['derived']
    assert a.attrs['spatial_point_audit']['timing_sha256']==b.attrs['spatial_point_audit']['timing_sha256']


def test_native_recipe_fixture_export_and_persistence(ui_recipe,monkeypatch,tmp_path):
    from xdiyo_analytics.data import load_seasons
    from xdiyo_analytics.histories import build_team_history
    from xdiyo_analytics.experiments.recovery import dump_bundle,load_bundle
    loaded=load_seasons(**ui_recipe['data']);h=build_team_history(loaded)
    loaded.tables['heatmap_points']=maps_for(h,line(1))
    monkeypatch.setattr('xdiyo_analytics.data.load_seasons',lambda **kwargs:deepcopy(loaded))
    recipe=deepcopy(ui_recipe);recipe['features']=json.loads(json.dumps(recipe_features()))
    prepared=prepare_recipe(recipe);ds=prepared.dataset
    assert list(ds.X)==[f'{side}::activity_axis_{field}_r20' for side in ('home','away') for field in ('cos2','sin2')]
    scope={};exec(export_python(recipe).split('result = run_recipe')[0],scope)
    pd.testing.assert_frame_equal(ds.X,scope['prepared'].dataset.X)
    assert ds.definitions==scope['prepared'].dataset.definitions
    f=evaluate_features(h,feature_specs(),heatmaps=loaded.tables['heatmap_points'],keyed=True)
    f.to_parquet(tmp_path/'channels.parquet',compression='zstd')
    restored=pd.read_parquet(tmp_path/'channels.parquet')
    pd.testing.assert_frame_equal(f,restored,check_index_type=False)
    assert restored.attrs==json.loads(json.dumps(f.attrs))
    dump_bundle(tmp_path/'channels.json',f)
    recovered=load_bundle(tmp_path/'channels.json')
    pd.testing.assert_frame_equal(f,recovered);assert f.attrs==recovered.attrs
