import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { bundleStatistics, expandFeatureBundle, statSources } from '../../src/xdiyo_analytics/ui/static/feature-bundles.js';

const components=Object.values(JSON.parse(readFileSync(new URL('../../src/xdiyo_analytics/ui/inventory.json',import.meta.url),'utf8')).components);
const stat=(period='ALL')=>({component:'features.Stat',params:{period,group:'Overview',key:'cornerKicks',field:'total'}});
const template=()=>({component:'features.RollingMean',params:{source:{component:'features.ForAgainst',params:{source:stat(),side:'both'}},window:7,min_periods:2}});
const stats=[{period:'ALL',group_name:'Overview',key:'cornerKicks'},{period:'ALL',group_name:'Attack',key:'totalShots'},{period:'1ST',group_name:'Overview',key:'cornerKicks'},{period:'ALL',group_name:'Defence',key:'totalShots'}];

test('one template expands multiple stats with shared settings and independent clones',()=>{
 const source=template(),before=structuredClone(source),chosen=[stats[0],stats[1]],beforeStats=structuredClone(chosen);
 const result=expandFeatureBundle(source,chosen,[],components);
 assert.equal(result.length,2);
 for(const item of result){assert.equal(item.expression.params.window,7);assert.equal(item.expression.params.min_periods,2);assert.equal(item.expression.params.source.params.side,'both');assert.equal(statSources(item.expression)[0].params.period,'ALL');assert.equal(statSources(item.expression)[0].params.field,'total');}
 assert.equal(statSources(result[1].expression)[0].params.key,'totalShots');
 result[0].expression.params.window=99;
 assert.equal(result[1].expression.params.window,7);
 assert.deepEqual(source,before);assert.deepEqual(chosen,beforeStats);
});
test('inventory window, span and lag defaults are materialized and named',()=>{
 for(const [component,field] of [['features.RollingMean','window'],['features.EMA','span'],['features.Lag','periods']]){
  const input={component,params:{source:stat()}};
  const expected=components.find(c=>c.id===component).fields.find(f=>f.name===field).default;
  const [result]=expandFeatureBundle(input,[stats[0]],[],components);
  assert.equal(result.expression.params[field],expected);
  if(expected!==null)assert.match(result.name,new RegExp(field+'_'+expected));
  assert.equal(Object.hasOwn(input.params,field),false);
 }
});
test('period filtering, all-period union and same-key groups remain distinguishable',()=>{
 const input=template();
 assert.deepEqual(bundleStatistics(input,stats),[stats[0],stats[1],stats[3]]);
 const full=expandFeatureBundle(input,bundleStatistics(input,stats),[],components);
 assert.equal(new Set(full.map(x=>x.name)).size,3);
 assert.deepEqual(full.map(x=>statSources(x.expression)[0].params.group),['Overview','Attack','Defence']);
 statSources(input)[0].params.period='1ST';
 assert.deepEqual(bundleStatistics(input,stats),[stats[2]]);
 statSources(input)[0].params.period=null;
 const union=bundleStatistics(input,stats);
 assert.equal(union.length,3);
 const [all]=expandFeatureBundle(input,[union[0]],[],components);
 assert.equal(statSources(all.expression)[0].params.period,null);assert.match(all.name,/all_periods/);
});
test('collisions suffix names without modifying existing feature names',()=>{
 const [first]=expandFeatureBundle(template(),[stats[0]],[],components);
 const used=[first.name,first.name+'_2'];
 const [next]=expandFeatureBundle(template(),[stats[0]],used,components);
 assert.equal(next.name,first.name+'_3');assert.deepEqual(used,[first.name,first.name+'_2']);
});
test('ambiguous templates and duplicate statistics are rejected',()=>{
 assert.throws(()=>expandFeatureBundle({component:'features.IsHome',params:{}},stats),/exactly one Stat/);
 assert.throws(()=>expandFeatureBundle({left:stat(),right:stat()},stats),/exactly one Stat/);
 assert.throws(()=>expandFeatureBundle(template(),[stats[0],stats[0]]),/each statistic once/);
 assert.deepEqual(bundleStatistics({left:stat(),right:stat()},stats),[]);
});
