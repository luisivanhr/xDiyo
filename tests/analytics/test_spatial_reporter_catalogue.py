"""Every implemented spatial family reaches a native prepared-value renderer."""
from types import SimpleNamespace
import json
import numpy as np
import pytest
from transition_samples import history_from_games,STAT
from test_point_geometry import maps_for
from xdiyo_analytics.features import (Heatmap,RegionMass,SpatialEntropy,SpatialConcentration,
    SpatialPointSummary,RollingMean,Product,Constant,evaluate_features)
from xdiyo_analytics.features.point_geometry import POINT_FIELDS
from xdiyo_analytics.features.spatial import REGIONS
from xdiyo_analytics.features.spatial_distribution import CONCENTRATION_METRICS
from xdiyo_analytics.labels import create_labels,MatchTotal
from xdiyo_analytics.datasets import assemble_dataset
from xdiyo_analytics.reporting import HeatmapReporter
from xdiyo_analytics.reporting.spatial_view import render_spatial


def cases():
    from xdiyo_analytics.features import SpatialHistoricalDeviation, SpatialFixtureDistance
    yield 'deviation',RollingMean(SpatialHistoricalDeviation(),2),'scalar'
    for comparison in ('style','home_context','away_context'):
        yield comparison,SpatialFixtureDistance(comparison=comparison),'scalar'
    yield 'keeper',RollingMean(SpatialPointSummary('mean_x',kinds=('goalkeeper',)),2),'scalar'
    for method in ('grid','gaussian'):
        for normalization in ('mass','density','count'):
            yield f'{method}_{normalization}',RollingMean(Heatmap(grid_size=2,method=method,normalization=normalization),2),'grid'
    for field in POINT_FIELDS:
        yield field,RollingMean(SpatialPointSummary(field),2),'scalar'
    for region in (*REGIONS,'rectangle'):
        yield region,RollingMean(RegionMass(Heatmap(grid_size=2,kinds=('player',)),region,
                                           rectangle=(10,80,20,70) if region=='rectangle' else None),2),'region'
    for normalized in (False,True):
        yield f'entropy_{normalized}',RollingMean(SpatialEntropy(Heatmap(grid_size=2),normalized=normalized),2),'scalar'
    for metric in CONCENTRATION_METRICS:
        yield metric,RollingMean(SpatialConcentration(Heatmap(grid_size=2),metric),2),'scalar'
    yield 'arithmetic',Product(RollingMean(SpatialPointSummary('axis_sin2'),2),Constant(2)),'scalar'


@pytest.mark.parametrize('name,expression,kind',list(cases()),ids=lambda x:x if isinstance(x,str) else None)
def test_all_spatial_fields_in_both_fixture_panels(name,expression,kind):
    h=history_from_games([{}, {}, {}]);p=maps_for(h,[(20,30),(40,60),(80,70)])
    f=evaluate_features(h,{name:expression},heatmaps=p,keyed=True)
    label=create_labels(h,{'corners':MatchTotal(STAT)})['corners']
    ds=assemble_dataset(f,label,layout='match')
    report=HeatmapReporter(type='overall',partition='train').run(SimpleNamespace(
        X=ds.X,metadata=ds.metadata,layout=ds.layout,match_columns=ds.match_columns,definitions=ds.definitions))
    data=report.artifacts[0].data
    payload=data.offline()
    json.dumps(payload,allow_nan=False)
    assert len(data.maps)==1
    pair=data.panel_pair('2',next(iter(data.maps)))
    for panel in pair['panels']:
        assert panel['spec']['kind']==kind and not panel['missing']
        columns=[c for c in ds.X if c.startswith(panel['side']+'::')]
        np.testing.assert_allclose(panel['values'],ds.X[columns].to_numpy()[-1])
        if kind=='region':assert panel['spec']['rectangles']
        if kind=='scalar':assert len(panel['values'])==1
    html=render_spatial(data)
    assert 'spatial-panels' in html and 'spatial-spec' in html


def test_javascript_routes_grid_region_scalar_and_missing_panels():
    """Execute renderer branches with a DOM/Plotly test double, no browser/server."""
    import shutil
    import subprocess
    from xdiyo_analytics.reporting.spatial_view import JS
    node=shutil.which('node')
    if node is None:pytest.skip('Node is optional for the renderer execution check')
    # Full catalogue payload tests above cover numerical values. Here the actual
    # shipped JavaScript must dispatch those kinds to traces/shapes/annotations.
    harness=r'''
const assert=require('node:assert/strict');
class Element {
 constructor(){this.value='';this.options=[];this.textContent='';this.hidden=false;}
 replaceChildren(...items){this.options=items;if(items.length)this.value=items[0].value;}
 append(){}
}
global.document={createElement:()=>new Element(),createTextNode:x=>x};
async function run(kind,missing=false,fixture=false){
 const calls=[];
 global.Plotly={react:async(host,traces,layout)=>calls.push({traces,layout}),purge:()=>{}};
 const spec={kind,grid_size:2,orientation:'team',field_label:'Axial channel',units:'dimensionless',rectangles:[[0,50,0,100]]};
 const panels=['home','away'].map((side,i)=>({side,team_id:String(i),spec,values:kind==='grid'?[.1,.2,.3,.4]:[.5],missing}));
 if(fixture){panels.splice(1,1);panels[0].side='fixture';panels[0].opponent_id='1';}
 const cfg={manifest:{fixtures:[{id:'0',label:'Home vs Away'}],maps:[{id:'map',label:'Map'}],teams:{'0':{name:'Home'},'1':{name:'Away'}},cache_size:8,included_fixtures:1,total_fixtures:1},pairs:{'0':{map:{panels,scale:[0,1]}}}};
 const els={};for(const name of ['spec','fixture','feature','status','search','panels','download'])els['.spatial-'+name]=new Element();
 els['.spatial-spec'].textContent=JSON.stringify(cfg);
 const hosts=[0,1].map(()=>{const children={};return {querySelector:key=>children[key]??=(new Element())};});
 const root={dataset:{},querySelector:key=>els[key],querySelectorAll:key=>key==='.spatial-panel'?hosts:[]};
 initializeSpatial(root);
 await new Promise(resolve=>setImmediate(resolve));
 assert.equal(calls.length,fixture?1:2);
 assert.equal(hosts[1].hidden,fixture);
 for(const call of calls){
  if(missing)assert.equal(call.layout.annotations[0].text,'Feature unavailable');
  else if(kind==='grid'){assert.equal(call.traces[0].type,'heatmap');assert.deepEqual(call.traces[0].z,[[.1,.2],[.3,.4]]);}
  else {assert.match(call.layout.annotations[0].text,/Axial channel: 0.50000/);if(kind==='region')assert.ok(call.layout.shapes[0].fillcolor);}
 }
 root.spatialDispose();
}
(async()=>{for(const kind of ['grid','region','scalar']){await run(kind);await run(kind,true);}await run('scalar',false,true);await run('scalar',true,true);console.log('All renderer branches passed');})().catch(e=>{console.error(e);process.exitCode=1;});
'''
    result=subprocess.run([node,'-'],input=JS+harness,text=True,capture_output=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
