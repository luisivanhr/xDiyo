// Exercise the real form renderer and event handlers without launching a server.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {configure, componentEditor, makeNode, statisticsOutputPreview} from '../../src/xdiyo_analytics/ui/static/forms.js';

class Element {
 constructor(tag) { this.tag=tag; this.nodeType=1; this.children=[]; this.value=''; this.checked=false; this.type=''; this.events={}; }
 append(...items) { this.children.push(...items); }
 prepend(...items) { this.children.unshift(...items); }
 replaceChildren(...items) { this.children=items; }
 setAttribute(key,value) { this[key]=value; }
 addEventListener(key,handler) { this.events[key]=handler; }
}
globalThis.document={createElement:tag=>new Element(tag),createTextNode:text=>({nodeType:3,text})};
const components=Object.values(JSON.parse(readFileSync(new URL('../../src/xdiyo_analytics/ui/inventory.json',import.meta.url),'utf8')).components);
configure(components,()=>{});
const walk=node=>[node,...(node.children||[]).flatMap(walk)];
const find=(root,label,tag)=>walk(root).find(n=>n['aria-label']===label&&(!tag||n.tag===tag));

test('GAT deterministic mode is an editable opt-in and survives JSON export',async()=>{
 let recipe=makeNode('graphs.TemporalGraphConfig');
 const root=componentEditor(recipe,v=>{recipe=v;},null,['graphs.TemporalGraphConfig']);
 const field=find(root,'Deterministic','input');
 assert.equal(field.checked,false);
 field.checked=true;field.events.change({target:field});await Promise.resolve();
 assert.equal(JSON.parse(JSON.stringify(recipe)).params.deterministic,true);
 field.checked=false;field.events.change({target:field});await Promise.resolve();
 assert.equal(recipe.params.deterministic,false);
});

test('native goal predictors can be disabled and exported as false',async()=>{
 let recipe=makeNode('graphs.TemporalGraphConfig');
 const root=componentEditor(recipe,v=>{recipe=v;},null,['graphs.TemporalGraphConfig']);
 const goals=find(root,'Include goal features','input');
 assert.equal(goals.checked,true);
 goals.checked=false;goals.events.change({target:goals});await Promise.resolve();
 assert.equal(JSON.parse(JSON.stringify(recipe)).params.include_goal_features,false);
 goals.checked=true;goals.events.change({target:goals});await Promise.resolve();
 assert.equal(recipe.params.include_goal_features,true);
});


test('graph CNN and auxiliary reconstruction are independent opt-ins',async()=>{
 let recipe=makeNode('graphs.TemporalGraphConfig');
 const root=componentEditor(recipe,v=>{recipe=v;},null,['graphs.TemporalGraphConfig']);
 let enabled=find(root,'Enable Heatmaps','input');
 assert.equal(enabled.checked,false);
 enabled.checked=true;enabled.events.change({target:enabled});await Promise.resolve();
 assert.equal(recipe.params.heatmaps.component,'graphs.GraphHeatmaps');
 const aux=find(root,'Enable Reconstruction','input');
 assert.equal(aux.checked,false);
 aux.checked=true;aux.events.change({target:aux});await Promise.resolve();
 assert.equal(recipe.params.heatmaps.params.encoder.params.reconstruction.component,'neural.Reconstruction');
 const loss=find(root,'Loss','select');assert.equal(loss.value,'0');
 loss.value='1';loss.onchange();await Promise.resolve();
 assert.equal(recipe.params.heatmaps.params.encoder.params.reconstruction.params.loss,'kl');
 const weight=find(root,'Weight','input');weight.value='0.25';weight.events.change({target:weight});await Promise.resolve();
 assert.equal(recipe.params.heatmaps.params.encoder.params.reconstruction.params.weight,0.25);
 aux.checked=false;aux.events.change({target:aux});await Promise.resolve();
 assert.equal(recipe.params.heatmaps.params.encoder.params.reconstruction,undefined);
 enabled.checked=false;enabled.events.change({target:enabled});await Promise.resolve();
 assert.equal(recipe.params.heatmaps,undefined);
});

test('standalone task, architecture and masks have typed editable form fields',async()=>{
 let recipe=makeNode('neural.HeatmapCNNModel');
 const root=componentEditor(recipe,v=>{recipe=v;},'model');
 const task=find(root,'Task','select');task.value='1';task.onchange();await Promise.resolve();
 assert.equal(recipe.params.task,'regression');
 const width=find(root,'Embedding dim','input');width.value='12';width.events.change({target:width});await Promise.resolve();
 assert.equal(recipe.params.encoder.params.embedding_dim,12);
 assert.equal(find(root,'Enable Reconstruction','input').checked,false);
 assert.ok(find(root,'Pool size','input'));
 assert.ok(find(root,'Kernel size','input'));
});
// Opt-in temporal memory and ordered statistics controls use the native editor.
test('memory and statistics are independent contextual opt-ins',async()=>{
 let recipe=makeNode('graphs.TemporalGraphConfig');
 const root=componentEditor(recipe,v=>{recipe=v;},null,['graphs.TemporalGraphConfig']);
 for(const name of ['Boundary memory','Statistics forecast']) {
  const field=find(root,'Enable '+name,'input');assert.equal(field.checked,false);
  field.checked=true;field.events.change({target:field});await Promise.resolve();
 }
 assert.equal(recipe.params.boundary_memory.component,'graphs.BoundaryMemory');
 assert.equal(recipe.params.statistics_forecast.component,'graphs.StatisticsForecast');
 const route=find(root,'Route','select');route.value='2';route.onchange();await Promise.resolve();
 assert.equal(recipe.params.statistics_forecast.params.route,'bottleneck');
 const side=find(root,'Side','select');side.value='1';side.onchange();await Promise.resolve();
 assert.equal(recipe.params.statistics_forecast.params.targets[0].params.side,'away');
 const family=find(root,'Family','select');family.value='2';family.onchange();await Promise.resolve();
 assert.equal(recipe.params.statistics_forecast.params.targets[0].params.family,'negative_binomial');
 assert.deepEqual(JSON.parse(JSON.stringify(recipe)),recipe);
});


test('quantile controls expose editable levels and learned output count',async()=>{
 let value=makeNode('graphs.StatisticTarget');const root=componentEditor(value,v=>{value=v;},null,['graphs.StatisticTarget']);
 assert.ok(!find(root,'Quantiles'));
 const family=find(root,'Family','select');family.value='4';family.onchange();await Promise.resolve();
 assert.equal(value.params.family,'quantile');
 assert.match(statisticsOutputPreview(value),/3 learned statistic outputs/);
 const numbers=walk(root).filter(n=>n.tag==='input'&&n.type==='number');
 const level=numbers.find(n=>n.value==.25);assert.ok(level);level.value='0.1';level.events.change({target:level});await Promise.resolve();
 assert.equal(value.params.quantiles[0],.1);
 value.params.quantiles=[.1,.1];assert.match(statisticsOutputPreview(value),/Invalid quantiles/);
 value.params.quantiles=[.1];assert.match(statisticsOutputPreview(value),/1 learned statistic outputs/);
 const target={component:'graphs.StatisticTarget',params:{family:'quantile',quantiles:[.25,.5,.75]}};
 assert.match(statisticsOutputPreview({component:'graphs.StatisticsForecast',params:{targets:Array(8).fill(target)}}),/8 target channels · 24 learned statistic outputs/);
});
