// Exercise the production form renderer and actual change handlers.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {configure, componentEditor, makeNode} from '../../src/xdiyo_analytics/ui/static/forms.js';
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
for(const suffix of ['Regressor','Classifier','Ranker']) test('CatBoost'+suffix+' typed parameters and model selection',async()=>{
 let recipe=makeNode('catboost.CatBoost'+suffix);
 const root=componentEditor(recipe,v=>{recipe=v;},'model');
 assert.ok(find(root,'Component','select').children.some(n=>n.value==='catboost.CatBoost'+suffix));
 for(const [title,key,value] of [['Iterations','iterations',12],['Depth','depth',3],['Learning rate','learning_rate',0.15],['Penalties coefficient','penalties_coefficient',0.75]]) {
  const enable=find(root,'Enable '+title,'input');
  enable.checked=true; enable.events.change({target:enable}); await Promise.resolve();
  const input=find(root,title,'input'); assert.equal(input.type,'number');
  input.value=String(value); input.events.change({target:input}); await Promise.resolve();
  assert.equal(recipe.params[key],value);
 }
 const taskEnable=find(root,'Enable Task type','input');
 taskEnable.checked=true; taskEnable.events.change({target:taskEnable}); await Promise.resolve();
 const task=find(root,'Task type','select'); task.value='1'; task.onchange(); await Promise.resolve();
 assert.equal(recipe.params.task_type,'GPU');
 const writingEnable=find(root,'Enable Allow writing files','input');
 writingEnable.checked=true; writingEnable.events.change({target:writingEnable}); await Promise.resolve();
 assert.equal(recipe.params.allow_writing_files,false);
 assert.ok(find(root,'Enable Cat features','input'));
 assert.deepEqual(JSON.parse(JSON.stringify(recipe)),recipe);
});
