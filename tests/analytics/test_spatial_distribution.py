"""Phase B: independent probability/area arithmetic and causal recipe integration."""
from copy import deepcopy
from types import SimpleNamespace
import json
import math

import numpy as np
import pandas as pd
import pytest

from test_point_geometry import maps_for
from test_ui_workflow import ui_recipe
from transition_samples import history_from_games, STAT, A, B
from xdiyo_analytics.features import (Heatmap, RegionMass, SpatialEntropy, SpatialConcentration,
    RollingMean, Lag, H2H, Product, Constant, RollingZScore, evaluate_features)
from xdiyo_analytics.features.spatial import heatmap_values, region_values, REGIONS
from xdiyo_analytics.features.spatial_distribution import distribution_values
from xdiyo_analytics.ui import prepare_recipe, catalog_for_ui
from xdiyo_analytics.ui.recipe import node, export_python
from xdiyo_analytics.datasets import assemble_dataset
from xdiyo_analytics.labels import MatchTotal, create_labels
from xdiyo_analytics.reporting import HeatmapReporter


def grid(**kwargs):
    return Heatmap(grid_size=6, kinds=('player',), orientation='team', **kwargs)


def frame(values, normalization='mass'):
    result=pd.DataFrame([values], columns=[f'c{i}' for i in range(len(values))])
    result.attrs['spatial_features']={'g':dict(kind='grid',columns=list(result),grid_size=6,
        normalization=normalization,orientation='team',side='for',operators=[])}
    return result


@pytest.mark.parametrize('distribution,h,hhi,occupied', [
    ([1.]+[0.]*35,0,1,1/36), ([1/36]*36,math.log(36),1/36,1),
    ([.5,.5]+[0.]*34,math.log(2),.5,2/36)])
def test_entropy_and_all_concentrations(distribution,h,hhi,occupied):
    f=frame(distribution)
    assert distribution_values(f,SpatialEntropy(None,normalized=False)).iloc[0,0]==pytest.approx(h)
    assert distribution_values(f,SpatialEntropy(None,normalized=False,base=2)).iloc[0,0]==pytest.approx(h/math.log(2))
    assert distribution_values(f,SpatialEntropy(None)).iloc[0,0]==pytest.approx(h/math.log(36))
    expected={'effective_cells':math.exp(h),'effective_fraction':math.exp(h)/36,
        'hhi':hhi,'normalized_hhi':(hhi-1/36)/(1-1/36),'largest_cell_share':max(distribution),
        'occupied_fraction':occupied}
    for metric,value in expected.items():
        assert distribution_values(f,SpatialConcentration(None,metric)).iloc[0,0]==pytest.approx(value)


@pytest.mark.parametrize('bad', [[0.]*36,[-.1,1.1]+[0.]*34,[np.nan,1]+[0.]*34,
    [np.inf]+[0.]*35,[.99]+[0.]*35,[1.01]+[0.]*35])
def test_corrupt_distributions_raise_without_clipping_or_repair(bad):
    with pytest.raises(ValueError,match='spatial grid|Spatial grid'):
        distribution_values(frame(bad),SpatialEntropy(None))


def test_wholly_missing_density_drift_and_count_contract():
    assert distribution_values(frame([np.nan]*36),SpatialEntropy(None)).isna().all().all()
    values=np.full(36,1/36)*(1+1e-12)
    assert distribution_values(frame(values),SpatialEntropy(None)).iloc[0,0]==pytest.approx(1)
    assert distribution_values(frame(values/(100/6)**2,'density'),SpatialEntropy(None)).iloc[0,0]==pytest.approx(1)
    with pytest.raises(ValueError,match='sum to one'):
        distribution_values(frame(values),SpatialEntropy(None,mass_tolerance=0))
    with pytest.raises(ValueError,match='mass or density'):
        distribution_values(frame([1]*36,'count'),SpatialEntropy(None))


@pytest.mark.parametrize('kwargs',[dict(base=1),dict(base=np.inf),dict(normalized='yes'),dict(mass_tolerance=-1),dict(mass_tolerance=.1)])
def test_entropy_parameters(kwargs):
    with pytest.raises(ValueError):SpatialEntropy(grid(),**kwargs)


@pytest.mark.parametrize('normalization,total',[('mass',1),('density',1),('count',36)])
@pytest.mark.parametrize('method',['grid','gaussian'])
def test_regions_partition_grid_mass_density_counts_and_rectangles(normalization,total,method):
    h=history_from_games([{}])
    xy=[((x+.5)*100/6,(y+.5)*100/6) for y in range(6) for x in range(6)]
    # A constant fine-grid density remains uniform under reflected smoothing.
    raw=heatmap_values(h,maps_for(h,xy),grid(normalization=normalization,method=method,resolution=6))
    values={region:region_values(raw,region).iloc[0,0] for region in REGIONS}
    assert values['own_half']+values['opponent_half']==pytest.approx(total)
    assert sum(values[r] for r in ('defensive_third','middle_third','attacking_third'))==pytest.approx(total)
    assert sum(values[r] for r in ('low_y_wide','central_channel','high_y_wide'))==pytest.approx(total)
    assert values['central_channel']+values['wide_channels']==pytest.approx(total)
    assert values['central_attacking_third']==pytest.approx(total/9)
    # Area integration cuts cells; it is not exact point membership.
    assert region_values(raw,'rectangle',(0,10,0,20)).iloc[0,0]==pytest.approx(total*.02)


@pytest.mark.parametrize('rectangle', [None,(0,1,2),(10,0,0,100),(-1,100,0,100),(0,100,0,np.inf)])
def test_rectangle_validation(rectangle):
    with pytest.raises(ValueError): RegionMass(grid(),'rectangle',rectangle)


def test_boundaries_native_reversal_not_rotated_points_rebinning():
    h=history_from_games([{}]);p=maps_for(h,[(50,20),(51,20)])
    own=heatmap_values(h,p,grid())
    against=heatmap_values(h,p,grid(side='against'))
    assert distribution_values(own,SpatialEntropy(None)).iloc[0,0]==0
    assert distribution_values(against,SpatialEntropy(None)).iloc[0,0]==0
    assert region_values(own,'attacking_third').iloc[0,0]==0
    assert region_values(against,'low_y_wide').iloc[0,0]==0
    assert region_values(against,'high_y_wide').iloc[0,0]==pytest.approx(1)
    p.loc[p.kind=='player',['x','y']]=100-p.loc[p.kind=='player',['x','y']]
    rebinned=heatmap_values(h,p,grid())
    assert distribution_values(rebinned,SpatialEntropy(None,normalized=False)).iloc[0,0]==pytest.approx(math.log(2))
    edges=maps_for(h,[(0,0),(100/3,100/3),(50,50),(200/3,200/3),(100,100)])
    raw=heatmap_values(h,edges,grid())
    assert region_values(raw,'defensive_third').iloc[0,0]==pytest.approx(1/5)
    assert region_values(raw,'attacking_third').iloc[0,0]==pytest.approx(2/5)


def test_new_regions_fail_on_partial_maps_legacy_halves_stay_compatible():
    f=frame([1]+[0]*34+[np.nan])
    with pytest.raises(ValueError,match='wholly missing'):
        region_values(f,'attacking_third')
    assert region_values(f,'own_half').iloc[0,0]==pytest.approx(1)
    with pytest.raises(ValueError,match='positive counts or unit mass'):
        region_values(frame([0.]*36),'wide_channels')


def test_per_match_entropy_differs_from_entropy_of_mixture_and_linear_region_commutes():
    h=history_from_games([{}, {}, {}, {}]);p=maps_for(h,[(10,10)])
    p.loc[(p.event_id==h.event_id.iloc[2]) & (p.kind=='player'),'x']=90
    specs={'typical':RollingMean(SpatialEntropy(grid()),2),
           'mixture':SpatialEntropy(RollingMean(grid(),2)),
           'region_before':RollingMean(RegionMass(grid(),'attacking_third'),2),
           'region_after':RegionMass(RollingMean(grid(),2),'attacking_third')}
    result=evaluate_features(h,specs,heatmaps=p)
    assert result.typical.iloc[4]==0
    assert result.mixture.iloc[4]==pytest.approx(math.log(2)/math.log(36))
    np.testing.assert_allclose(result.region_before,result.region_after,equal_nan=True)
    assert result.attrs['spatial_features']['typical']['calculation']['operator']=='RollingMean'
    assert result.attrs['spatial_features']['mixture']['calculation']['operator']=='SpatialEntropy'
    # Missing observations stay inside the fixed window, without refilling.
    p=p[p.event_id!=h.event_id.iloc[2]]
    missing=evaluate_features(h,specs,heatmaps=p)
    np.testing.assert_allclose(missing.region_before,missing.region_after,equal_nan=True)


@pytest.mark.parametrize('expression',[SpatialEntropy(grid()),H2H(SpatialEntropy(grid())),
    Product(SpatialConcentration(grid()),Constant(2)),RegionMass(grid(),'wide_channels')])
def test_recursive_safety(expression):
    h=history_from_games([{},{}])
    with pytest.raises(ValueError,match='before prediction'):
        evaluate_features(h,{'unsafe':expression},heatmaps=maps_for(h))


def test_identity_duplicates_known_reference_and_future_cutoffs():
    h=history_from_games([{}, {},dict(day=2),dict(day=6,status='notstarted')]);p=maps_for(h)
    spec={'e':RollingMean(SpatialEntropy(grid()),2)}
    with pytest.raises(ValueError,match='both history'):
        evaluate_features(h,spec,heatmaps=p.drop(columns='source_season'))
    with pytest.raises(ValueError,match='point_order'):
        evaluate_features(h,spec,heatmaps=pd.concat([p,p]))
    with pytest.raises(ValueError,match='reference must be historical'):
        evaluate_features(h,{'z':RollingZScore(SpatialEntropy(grid()),2,reference=SpatialEntropy(grid()))},heatmaps=p)
    kwargs=dict(available_at=h.kickoff_at+pd.Timedelta(hours=3),cutoffs=h.kickoff_at-pd.Timedelta(hours=1))
    result=evaluate_features(h,spec,heatmaps=p,**kwargs)
    p.loc[p.event_id.isin(h.event_id.iloc[2:]),'x']=20
    changed=evaluate_features(h,spec,heatmaps=p,**kwargs)
    np.testing.assert_allclose(result.iloc[:6],changed.iloc[:6],equal_nan=True)
    assert result.attrs['spatial_distribution_audit']['sources'][0]['source_sha256'] != changed.attrs['spatial_distribution_audit']['sources'][0]['source_sha256']


def test_ui_recipe_exports_assembly_and_report_captions(ui_recipe,monkeypatch,tmp_path):
    from xdiyo_analytics.data import load_seasons
    from xdiyo_analytics.histories import build_team_history
    from xdiyo_analytics.reporting.spatial_view import render_spatial
    data=load_seasons(**ui_recipe['data']);data.tables['heatmap_points']=maps_for(build_team_history(data))
    calls=[]
    def load(**kw):calls.append(kw);return deepcopy(data)
    monkeypatch.setattr('xdiyo_analytics.data.load_seasons',load)
    source=node('features.Heatmap',grid_size=6,kinds=['player'],orientation='team')
    recipe=deepcopy(ui_recipe)
    recipe['features']={
        'entropy':node('features.RollingMean',source=node('features.SpatialEntropy',source=source),window=20),
        'mixture':node('features.SpatialEntropy',source=node('features.RollingMean',source=source,window=20)),
        'region':node('features.RollingMean',source=node('features.RegionMass',source=source,region='rectangle',rectangle=[20,80,25,75]),window=20),
        'hhi':node('features.RollingMean',source=node('features.SpatialConcentration',source=source,metric='hhi'),window=20)}
    prepared=prepare_recipe(json.loads(json.dumps(recipe)))
    assert 'heatmap_points' in calls[0]['tables']
    assert prepared.dataset.X.shape[1]==8
    scope={};exec(export_python(recipe).split('result = run_recipe')[0],scope)
    pd.testing.assert_frame_equal(scope['prepared'].dataset.X,prepared.dataset.X)
    ds=prepared.dataset
    context=SimpleNamespace(X=ds.X,metadata=ds.metadata,definitions=ds.definitions,
        layout='match',match_columns=ds.match_columns)
    report=HeatmapReporter(type='overall',partition='train').run(context)
    store=report.artifacts[0].data
    assert store.maps['entropy']['spec']['calculation_label'].startswith('Rolling mean (20 matches) of Normalized entropy')
    assert store.maps['mixture']['spec']['calculation_label'].startswith('Normalized entropy of Rolling mean')
    html=render_spatial(store)
    assert 'grid_area_overlap' in html and 'Normalized entropy of Rolling mean' in html
    assert store.maps['region']['spec']['rectangles']==[[20.,80.,25.,75.]]
    assert ds.definitions['spatial_distribution_audit']['sources'][0]['provenance']==data.provenance
    fields={f['name']:f for f in next(s for s in catalog_for_ui().schema() if s['id']=='features.SpatialEntropy')['fields']}
    assert fields['source']['initial']['params']['grid_size']==6
    assert fields['source']['initial']['params']['kinds']==['player']


def test_distribution_metadata_parquet_and_shared_grid_cache(monkeypatch,tmp_path):
    import xdiyo_analytics.features.evaluation as evaluator
    original=evaluator.heatmap_values;calls=[]
    def count(*args):calls.append(1);return original(*args)
    monkeypatch.setattr(evaluator,'heatmap_values',count)
    h=history_from_games([{}, {},dict(home=B,away=A)])
    p=maps_for(h,[(20,30),(40,70)])
    source=grid()
    specs={'e':RollingMean(SpatialEntropy(source),2),
           'hhi':RollingMean(SpatialConcentration(source),2),
           'region':RollingMean(RegionMass(source,'central_channel'),2)}
    values=evaluate_features(h,specs,heatmaps=p,keyed=True)
    assert len(calls)==1
    path=tmp_path/'features.parquet';values.to_parquet(path,compression='zstd')
    restored=pd.read_parquet(path)
    np.testing.assert_allclose(values,restored,equal_nan=True)
    assert restored.attrs['spatial_features']==values.attrs['spatial_features']
    assert restored.attrs['spatial_distribution_audit']['sources']==values.attrs['spatial_distribution_audit']['sources']
    # Same historical content on away rows does not undergo a second scalar rotation.
    np.testing.assert_allclose(values.iloc[-2],values.iloc[-1])
