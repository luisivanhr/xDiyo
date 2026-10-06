"""Paired historical interactions: numeric, causal and native integration contracts."""
from copy import deepcopy
import json
import numpy as np
import pandas as pd
import pytest

from transition_samples import STAT, OWN, OTHER, A, B, C, history_from_games
from test_point_geometry import maps_for
from test_spatial_arithmetic import report
from test_ui_workflow import ui_recipe
from xdiyo_analytics.features import (RollingWeightedMean as Weighted, RollingMean, RollingStd, Lag,
    Stat, ForAgainst, H2H, Product, Sum, Abs, Difference, MatchScore, SpatialPointSummary as Point,
    RegionMass, Heatmap, evaluate_features)
from xdiyo_analytics.experiments.recovery import signature, pack, dump_bundle, load_bundle
from xdiyo_analytics.ui import catalog_for_ui, prepare_recipe
from xdiyo_analytics.ui.recipe import export_python, export_notebook

WEIGHTS = Stat('1ST', 'Match overview', 'cornerKicks')


def sample(values, weights):
    return history_from_games([dict(values=(z,z), half=(w,w)) for z,w in zip(values, weights)])


def final(values, weights, **kwargs):
    h = sample([*values, 999], [*weights, 999])
    f = evaluate_features(h, {'w':Weighted(STAT, WEIGHTS, window=len(values), **kwargs)})
    return f.w.iloc[-1], f.attrs['weighted_history_audit']['w'][0]['records'][-1]


def test_interaction_order_and_missing_times_zero():
    h = sample([0,10,0], [10,0,0])
    f = evaluate_features(h, {'inside':RollingMean(Product(STAT, WEIGHTS),2),
                             'outside':Product(RollingMean(STAT,2),RollingMean(WEIGHTS,2))})
    assert f.inside.iloc[-1] == 0 and f.outside.iloc[-1] == 25
    h = sample([-2,np.nan,0], [3,0,0])
    f = evaluate_features(h, {'p':Lag(Product(STAT,WEIGHTS))})
    assert f.p.iloc[2] == -6 and np.isnan(f.p.iloc[-1])


@pytest.mark.parametrize('z,w,expected', [([0,10],[1,3],7.5), ([0,10],[2,2],5),
    ([0,np.nan],[1,100],0), ([0,10],[0,0],np.nan), ([-10,10],[1,3],5),
    ([np.inf,10],[100,1],10), ([0,10],[np.inf,1],10)])
def test_weighted_pairs(z,w,expected):
    value, _ = final(z,w)
    np.testing.assert_allclose(value,expected,equal_nan=True)


def test_minimum_slots_and_audit():
    value,audit = final([9,np.nan,10,20], [1,8,0,2], min_periods=3)
    assert np.isnan(value)
    assert audit['paired_finite_count']==3 and audit['positive_weight_count']==2
    assert audit['zero_weight_count']==1 and audit['missing_value_count']==1
    assert audit['total_weight']==3 and audit['effective_sample_size']==pytest.approx(1.8)
    h = sample([8,np.nan,10,0], [1,2,0,1])
    f = evaluate_features(h, {'w':Weighted(STAT, WEIGHTS,window=2)})
    assert np.isnan(f.w.iloc[-1])  # no replacement of missing/zero slots


@pytest.mark.parametrize('z', [1,np.nan,np.inf])
def test_negative_available_weights_raise_even_without_value(z):
    with pytest.raises(ValueError,match='finite negative weight'):
        final([z],[-1])


def test_negative_only_selected_available_window_and_cutoff_safety():
    h=sample([np.nan,10,20,999],[-1,2,3,-999])
    # First result is released late: native eligibility excludes it until the
    # last cutoff, where it is outside the fixed two-slot window.
    releases=h.kickoff_at.copy();releases.iloc[:2]=h.kickoff_at.iloc[-1]
    expr=Weighted(STAT,WEIGHTS,window=2)
    f=evaluate_features(h,{'w':expr},available_at=releases)
    assert f.w.iloc[-1]==16
    h['weight_release']=h.kickoff_at
    h.loc[:1,'weight_release']=h.kickoff_at.iloc[-1]+pd.Timedelta(days=1)
    h['value_release']=h.kickoff_at
    h.loc[2:3,'value_release']=h.kickoff_at.iloc[-1]+pd.Timedelta(days=1)
    expr=Weighted(STAT,WEIGHTS,window=3, source_available_at='value_release',weights_available_at='weight_release')
    f=evaluate_features(h,{'w':expr},cutoffs=h.kickoff_at-pd.Timedelta(hours=1))
    assert f.w.iloc[-1]==20
    audit=f.attrs['weighted_history_audit']['w'][0]['records'][-1]
    assert audit['window_matches']==3 and audit['positive_weight_count']==1
    assert audit['unavailable_weight_count']==1 and audit['unavailable_value_count']==1
    changed=h.copy();changed.loc[:1,OWN]=12345;changed.loc[:1,OTHER]=54321
    changed.loc[2:3,OWN]=12345;changed.loc[2:3,OTHER]=54321
    changed.loc[6:,OWN]=-54321
    after=evaluate_features(changed,{'w':expr},cutoffs=h.kickoff_at-pd.Timedelta(hours=1))
    np.testing.assert_allclose(f,after,equal_nan=True)
    assert signature(f.attrs)!=signature(after.attrs)
    h['weight_release']=pd.NaT
    assert evaluate_features(h,{'w':expr}).isna().all().all()


def test_invariances_and_extreme_magnitudes():
    for factor in (0.01,1,100,1e300,1e-300):
        value,_=final([-10,20,30],np.array([1.,2.,3.])*factor)
        assert value==pytest.approx(20)
        shifted,_=final([3,33,43],np.array([1.,2.,3.])*factor)
        assert shifted==pytest.approx(value+13)
        assert -10<=value<=30
    for z,w,expected in [([1e308,1e308],[1e308,1e308],1e308),
                         ([-1e308,1e308],[1e308,1e308],0),
                         ([1e308,0],[1e-308,1e308],1e-308)]:
        value,audit=final(z,w)
        assert value==pytest.approx(expected,rel=1e-14,abs=0)
        if w==[1e308,1e308]:assert audit['total_weight'] is None and audit['total_weight_overflow']


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('spatial', [False, True])
def test_shared_ordinary_and_h2h_audits_are_isolated(reverse, spatial):
    h=history_from_games([dict(home=A,away=B),dict(home=A,away=C),dict(home=A,away=B)])
    points=maps_for(h,[(10,20),(30,40)]) if spatial else None
    shared=Weighted(Point() if spatial else STAT,1,3)
    definitions={'ordinary':shared,'h2h':H2H(shared),
                 'wrapped':Sum(Lag(shared),1), 'wrapped_h2h':H2H(Sum(Lag(shared),1))}
    if reverse:definitions=dict(reversed(list(definitions.items())))
    together=evaluate_features(h,definitions,heatmaps=points,keyed=True)
    for name,expr in definitions.items():
        alone=evaluate_features(h,{name:expr},heatmaps=points,keyed=True)
        np.testing.assert_allclose(together[name],alone[name],equal_nan=True)
        assert together.attrs['weighted_history_audit'][name]==alone.attrs['weighted_history_audit'][name]
        entries=together.attrs['weighted_history_audit'][name]
        assert len(entries)==1 and entries[0]['definition']['h2h']==('h2h' in name)
        if spatial:
            assert together.attrs['spatial_point_audit']['features'][name]==alone.attrs['spatial_point_audit']['features'][name]
    assert together.attrs['weighted_history_audit']['ordinary'][0]['records'][-2]['window_matches']==2
    assert together.attrs['weighted_history_audit']['h2h'][0]['records'][-2]['window_matches']==1
    if spatial:
        table=report(together,h).tables['weighted_history_coverage']
        for name in definitions:
            assert len(table[table.feature==name])==len(h)


@pytest.mark.parametrize('power', [-300,0,300])
def test_signed_underflow_retains_representable_mean_under_weight_rescaling(power):
    from fractions import Fraction
    z=np.ldexp(np.array([1.,-1.,1.]),[400,400,-400])
    w=np.ldexp(np.ones(3),np.array([300,300,-300])+power)
    expected=float(sum((Fraction(float(a))*Fraction(float(b)) for a,b in zip(z,w)),Fraction(0)) /
                   sum(map(lambda b:Fraction(float(b)),w),Fraction(0)))
    value,_=final(z,w)
    assert expected>0 and value==pytest.approx(expected,rel=1e-14,abs=0)


@pytest.mark.parametrize('permutation', [(0,1,2),(2,0,1),(1,2,0)])
@pytest.mark.parametrize('small', [-350,-400])
def test_scaled_subnormal_terms_and_cancellation_are_order_independent(permutation, small):
    from fractions import Fraction
    from xdiyo_analytics.features.weighted import stable_mean
    z=np.ldexp(np.array([1.,-1.,1.5]),[400,400,small])
    w=np.ldexp(np.ones(3),[160,160,-160])
    expected=float(sum((Fraction(float(a))*Fraction(float(b)) for a,b in zip(z,w)),Fraction(0)) /
                   sum((Fraction(float(b)) for b in w),Fraction(0)))
    assert expected>0
    for shift in (-400,0,400):
        values=z[list(permutation)]
        weights=np.ldexp(w[list(permutation)],shift)
        # Powers-of-two rescaling introduces no input rounding in these cases.
        assert stable_mean(values,weights)==expected


def test_ordinary_weighted_fast_path_is_preserved(monkeypatch):
    import fractions
    from xdiyo_analytics.features.weighted import stable_mean
    def fail(*args,**kwargs):raise AssertionError('Ordinary values should not need exact arithmetic')
    monkeypatch.setattr(fractions,'Fraction',fail)
    assert stable_mean(np.array([0.,10.]),np.array([1.,3.]))==7.5
    from math import fsum
    values,weights=np.array([-10.,10.]),np.array([1.,3.])
    scaled=weights/weights.max()
    previous=fsum(float(a)*float(b) for a,b in zip(scaled,values/10))/fsum(scaled)*10
    assert stable_mean(values,weights)==previous
    assert previous==pytest.approx(5)


def test_scope_identity_shuffles_and_asof_state():
    h=history_from_games([dict(home=A,away=B,values=(2,20)),
        dict(home=B,away=A,values=(30,3)),dict(home=A,away=C,values=(4,40)),dict(home=A,away=B)])
    definitions={'w':Weighted(STAT,1,3), 'against':Weighted(ForAgainst(STAT,'against'),1,3),
                 'same':Weighted(STAT,1,3,venue='same'), 'h2h':H2H(Weighted(STAT,1,3)),
                 'nested':Weighted(Lag(STAT),1,3)}
    f=evaluate_features(h,definitions,keyed=True)
    np.testing.assert_allclose(f.iloc[-2].to_numpy(),[3,30,3,2.5,2.5])
    shuffled=h.sample(frac=1,random_state=4)
    g=evaluate_features(shuffled,definitions,keyed=True).reindex(f.index)
    pd.testing.assert_frame_equal(f,g)
    with pytest.raises(ValueError,match='same H2H'):
        evaluate_features(h,{'bad':Weighted(H2H(STAT),1)})
    with pytest.raises(ValueError,match='one output column'):
        evaluate_features(h,{'bad':Weighted(Stat(None,STAT.group,STAT.key),1)})
    with pytest.raises(ValueError,match='unique'):
        evaluate_features(pd.concat([h,h.iloc[:1]],ignore_index=True),{'w':Weighted(STAT,1)})
    # Colliding event IDs in another source/competition remain distinct.
    other=h.copy();other['source_league']='Other';other['competition_id']=99;other[OWN]=100
    h['source_league']='First'
    both=pd.concat([h,other],ignore_index=True);both.attrs=h.attrs
    out=evaluate_features(both,{'w':Weighted(STAT,1,3)},keyed=True)
    assert out.w.iloc[-2]==100 and out.w.iloc[6]==3


def test_spatial_centroids_masks_lineage_reporting_and_persistence(tmp_path):
    h=sample([1,3,1],[1,1,1]);p=maps_for(h,[(10,20)])
    last=set(h.event_id.iloc[2:4].tolist());p.loc[p.event_id.isin(last),'x']=90
    # Unequal sample volume must not add an implicit observation weight.
    extra=p[p.event_id.isin(last)].copy();extra.point_order+=10
    p=pd.concat([p,extra],ignore_index=True)
    expr=Weighted(Point('mean_x'),STAT,window=2)
    f=evaluate_features(h,{'weighted':expr,'twice':Sum(expr,expr)},heatmaps=p,keyed=True)
    assert f.weighted.iloc[-1]==70 and f.twice.iloc[-1]==140
    assert len(f.attrs['spatial_point_audit']['features']['twice']['maps'])==6
    assert len(f.attrs['weighted_history_audit']['twice'])==1
    assert f.attrs['spatial_features']['weighted']['calculation']['operator']=='RollingWeightedMean'
    study=report(f,h)
    assert 'weighted_history_coverage' in study.tables
    assert 'weights:' in next(iter(study.artifacts[0].data.maps.values()))['spec']['calculation_label']
    reflected=p.copy();reflected.x=100-reflected.x
    g=evaluate_features(h,{'weighted':expr},heatmaps=reflected)
    np.testing.assert_allclose(g.weighted,100-f.weighted.to_numpy(),equal_nan=True)
    f.to_parquet(tmp_path/'f.parquet',compression='zstd')
    loaded=pd.read_parquet(tmp_path/'f.parquet')
    assert pack(loaded.attrs)==pack(json.loads(json.dumps(f.attrs)))
    dump_bundle(tmp_path/'f.json',f)
    assert signature(load_bundle(tmp_path/'f.json'))==signature(f)


def test_spatial_missing_axes_and_signed_transform():
    from xdiyo_analytics.reporting.heatmaps import axial_direction_summary
    h=sample([1,1,1],[1,1,1]);p=maps_for(h,[(0,0),(100,0)])
    defs={field:Weighted(Point(field),STAT,2) for field in ('axis_cos2','axis_sin2')}
    f=evaluate_features(h,defs,heatmaps=p)
    summary=axial_direction_summary(f.axis_cos2,f.axis_sin2)
    assert summary['consistency'][-1]==pytest.approx(1)
    missing=p[~p.event_id.isin(set(h.event_id.iloc[2:4]))]
    f=evaluate_features(h,{k:Weighted(v.source,STAT,2,min_periods=2) for k,v in defs.items()},heatmaps=missing)
    assert f.iloc[-1].isna().all()
    constant=maps_for(h,[(10,10),(10,10)])
    assert evaluate_features(h,defs,heatmaps=constant).isna().all().all()
    assert evaluate_features(h,{'ok':Weighted(Point(),Abs(Difference(STAT,10)),2)},heatmaps=p).iloc[-1,0]==50
    with pytest.raises(ValueError,match='before prediction'):
        evaluate_features(h,{'bad':Product(Point(),STAT)},heatmaps=p)


@pytest.mark.parametrize('angles,r,mean', [([1,179],np.cos(np.deg2rad(2)),0),([32,32],1,32),([0,90],0,None)])
def test_weighted_axial_wrap_and_zero_resultant(angles,r,mean):
    from test_axial_point_features import angular_history
    from xdiyo_analytics.reporting import axial_direction_summary
    h,p=angular_history([*angles,45])
    features={field:Weighted(Point(field),1,2) for field in ('axis_cos2','axis_sin2')}
    f=evaluate_features(h,features,heatmaps=p)
    got=axial_direction_summary(*f.iloc[-1].to_numpy())
    assert got['consistency']==pytest.approx(r,abs=1e-14)
    if mean is None:assert np.isnan(got['mean_axis'])
    else:assert abs(np.angle(np.exp(2j*(got['mean_axis']-np.deg2rad(mean)))))<1e-13
    reflected=p.copy();reflected.x=100-reflected.x
    g=evaluate_features(h,features,heatmaps=reflected)
    np.testing.assert_allclose(g.axis_cos2,f.axis_cos2,equal_nan=True,atol=1e-14)
    np.testing.assert_allclose(g.axis_sin2,-f.axis_sin2,equal_nan=True,atol=1e-14)


def test_equal_weight_support_percentage_scale_and_zero_value():
    h=sample([0,10,np.nan,20,99],[1,3,999,2,99])
    definitions={'w':Weighted(STAT,WEIGHTS,4), 'fraction':Weighted(STAT,Product(WEIGHTS,.01),4),
        'mean':RollingMean(STAT,4),'equal':Weighted(STAT,1,4),
        'interaction':RollingMean(Product(STAT,WEIGHTS),4),
        'fraction_interaction':RollingMean(Product(STAT,Product(WEIGHTS,.01)),4)}
    f=evaluate_features(h,definitions)
    np.testing.assert_allclose(f.w,f.fraction,equal_nan=True)
    np.testing.assert_allclose(f['mean'],f.equal,equal_nan=True)
    np.testing.assert_allclose(f.interaction*.01,f.fraction_interaction,equal_nan=True)
    assert f.w.iloc[2]==0  # a zero value with positive weight contributes


def test_simultaneous_and_late_inputs_do_not_replace_window_slots():
    h=history_from_games([dict(day=0),dict(day=2),dict(day=2),dict(day=4)])
    h['weight_release']=h.kickoff_at
    h.loc[2:5,'weight_release']=h.kickoff_at.iloc[-1]+pd.Timedelta(days=1)
    expr=Weighted(STAT,WEIGHTS,2,weights_available_at='weight_release')
    f=evaluate_features(h,{'w':expr})
    assert f.w.iloc[2]==2 and f.w.iloc[4]==2 and np.isnan(f.w.iloc[-2])
    changed=h.copy();changed.loc[2:5,OWN]=1e12
    changed.loc[2:,'team::1ST::Match overview::cornerKicks::value']=-1e12
    g=evaluate_features(changed,{'w':expr})
    np.testing.assert_allclose(f,g,equal_nan=True)


def test_historical_rating_uses_its_own_cutoff():
    from xdiyo_analytics.features import MatchResultGlicko
    h=sample([1,3,1,3],[1,1,1,1])
    rating=MatchResultGlicko(side='for',fields=('rating',))
    settings=dict(cutoffs=h.kickoff_at-pd.Timedelta(hours=1),available_at=h.kickoff_at+pd.Timedelta(hours=3))
    f=evaluate_features(h,{'rating':rating,'weighted':Weighted(rating,STAT,2)},**settings)
    assert f.weighted.iloc[-2]==pytest.approx((f.rating.iloc[2]*3+f.rating.iloc[4])/4)
    changed=h.copy();changed.loc[6:,'result']=['L','W']
    g=evaluate_features(changed,{'rating':rating,'weighted':Weighted(rating,STAT,2)},**settings)
    np.testing.assert_allclose(f,g,equal_nan=True)


@pytest.mark.parametrize('bad', ['coordinates','version','identity'])
def test_native_source_validation_is_not_suppressed(bad):
    h=sample([1,1],[1,1]);p=maps_for(h,[(10,20),(30,40)])
    if bad=='coordinates':p.loc[0,'x']=np.inf
    elif bad=='version':p.loc[0,'raw_hash']='conflicting'
    else:p=pd.concat([p,p.iloc[:1]],ignore_index=True)
    with pytest.raises(ValueError):
        evaluate_features(h,{'w':Weighted(Point(),1)},heatmaps=p)


def test_generic_recipe_example_and_nested_wrappers():
    from pathlib import Path
    from runpy import run_path
    example=run_path(str(Path(__file__).resolve().parents[2]/'examples/weighted_feature_interactions.py'))
    h=sample([1,3,1,3],[1,1,1,1]);h['goals_for']=h[OWN];h['goals_against']=h[OTHER]
    points=maps_for(h,[(10,20),(70,80)])
    definitions=example['feature_definitions'](window=3,grid_size=7)
    encoded=example['recipe_features'](window=3,grid_size=7)
    assert {k:catalog_for_ui().build(v) for k,v in json.loads(json.dumps(encoded)).items()}==definitions
    f=evaluate_features(h,definitions,heatmaps=points,keyed=True)
    assert f.iloc[-1].notna().all()
    expr=Weighted(Point(),1,2)
    nested=evaluate_features(h,{'nested':RollingStd(expr,2,min_periods=2),'lag':Lag(expr)},heatmaps=points)
    assert nested.nested.iloc[-1]==0 and nested.lag.iloc[-1]==40
    assert len(nested.attrs['weighted_history_audit']['nested'])==1


def test_ui_venue_visible_for_either_spatial_operand():
    import shutil, subprocess
    from pathlib import Path
    executable=shutil.which('node')
    if executable is None:pytest.skip('Node required to exercise native form visibility')
    schema=next(s for s in catalog_for_ui().schema() if s['id']=='features.RollingWeightedMean')
    venue=next(f for f in schema['fields'] if f['name']=='venue')
    script=r"""
    const fs=require('fs'),vm=require('vm'),assert=require('assert/strict');
    const text=fs.readFileSync(process.argv[1],'utf8');
    const fn=text.slice(text.indexOf('function containsComponent('),text.indexOf('export function fieldsEditor('));
    const context={};vm.createContext(context);vm.runInContext(fn,context);
    const f=JSON.parse(process.argv[2]);
    const stat={component:'features.Stat'},spatial={component:'features.Abs',params:{source:{component:'features.SpatialPointSummary'}}};
    assert.equal(context.visible(f,{source:stat,weights:stat},[f]),false);
    assert.equal(context.visible(f,{source:spatial,weights:stat},[f]),true);
    assert.equal(context.visible(f,{source:stat,weights:spatial},[f]),true);
    assert.equal(context.visible(f,{source:spatial,weights:spatial},[f]),true);
    // Existing source-only conditions keep their previous semantics.
    const old={visible_when_contains:{source:['features.Heatmap']}};
    assert.equal(context.visible(old,{source:stat},[]),false);
    assert.equal(context.visible(old,{source:{component:'features.Heatmap'}},[]),true);
    """
    path=Path(__file__).resolve().parents[2]/'src/xdiyo_analytics/ui/static/forms.js'
    subprocess.run([executable,'-e',script,str(path),json.dumps(venue)],check=True,capture_output=True,text=True)


def test_builder_exports_assembly_and_cache_identity(ui_recipe,monkeypatch):
    from xdiyo_analytics.data import load_seasons
    from xdiyo_analytics.histories import build_team_history
    loaded=load_seasons(**ui_recipe['data']);h=build_team_history(loaded)
    loaded.tables['heatmap_points']=maps_for(h,[(10,20),(30,40)])
    monkeypatch.setattr('xdiyo_analytics.data.load_seasons',lambda **kwargs:deepcopy(loaded))
    catalog=catalog_for_ui();expr=Weighted(Point('mean_x'),STAT,20)
    encoded=json.loads(json.dumps(catalog.encode(expr)))
    assert catalog.build(encoded)==expr
    recipe=deepcopy(ui_recipe);recipe['features']={'weighted':encoded,'interaction':catalog.encode(RollingMean(Product(STAT,Point()),20))}
    prepared=prepare_recipe(recipe)
    scope={};exec(export_python(recipe).split('result = run_recipe')[0],scope)
    pd.testing.assert_frame_equal(prepared.dataset.X,scope['prepared'].dataset.X)
    assert prepared.dataset.definitions==scope['prepared'].dataset.definitions
    notebook=export_notebook(recipe)
    notebook_scope={};exec(''.join(notebook['cells'][1]['source']),notebook_scope)
    pd.testing.assert_frame_equal(prepared.dataset.X,notebook_scope['prepared'].dataset.X)
    assert list(prepared.dataset.X)==['home::weighted','home::interaction','away::weighted','away::interaction']
    assert 'weighted_history_audit' in prepared.dataset.definitions
    schemas={s['id']:s for s in catalog.schema()}
    fields={f['name']:f for f in schemas['features.RollingWeightedMean']['fields']}
    assert fields['weights']['kind']=='component' and fields['weights']['categories']==['feature']
    assert 'positive' in fields['min_periods']['help']
    assert signature(expr)!=signature(Weighted(Point(),Sum(STAT,1),20))


@pytest.mark.parametrize('kwargs',[dict(window=0),dict(window=True),dict(min_periods=0),dict(window=2,min_periods=3),dict(venue='other'),dict(source_available_at=[])])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):Weighted(STAT,1,**kwargs)
