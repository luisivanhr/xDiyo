import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {configure, componentEditor, makeNode, valueEditor} from '../../src/xdiyo_analytics/ui/static/forms.js';
class Element {
 constructor(tag){this.tag=tag;this.nodeType=1;this.children=[];this.value='';this.checked=false;this.events={};}
 append(...items){this.children.push(...items);} prepend(...items){this.children.unshift(...items);}
 replaceChildren(...items){this.children=items;} setAttribute(k,v){this[k]=v;} addEventListener(k,v){this.events[k]=v;}
}
globalThis.document={createElement:tag=>new Element(tag),createTextNode:text=>({nodeType:3,text})};
const components=Object.values(JSON.parse(readFileSync(new URL('../../src/xdiyo_analytics/ui/inventory.json',import.meta.url),'utf8')).components);
configure(components,()=>{});
const walk=n=>[n,...(n.children||[]).flatMap(walk)];
const find=(root,label,tag)=>walk(root).find(n=>n['aria-label']===label&&(!tag||n.tag===tag));
test('Composition child estimator has an editable native model selector',()=>{
 let value={component:'composition.Estimator',params:{estimator:{component:'sklearn.linear_model.Ridge',params:{}},preprocessors:[]}};
 const root=componentEditor(value,v=>value=v,null,['composition.Estimator']);
 const selectors=walk(root).filter(n=>n.tag==='select'&&n['aria-label']==='Component');
 assert.ok(selectors.some(s=>s.children.some(n=>n.value==='sklearn.svm.SVC')));
});
test('Chronological plan exposes mode choices and safety declarations',async()=>{
 let value=makeNode('composition.TrainingPlan');
 const root=componentEditor(value,v=>value=v,null,['composition.TrainingPlan']);
 const mode=find(root,'Mode','select');
 assert.ok(mode);
 mode.value='1';mode.onchange();await Promise.resolve();
 assert.equal(value.params.mode,'chronological');
 const entry=components.find(c=>c.id==='composition.TrainingPlan');
 assert.ok(entry.fields.some(f=>f.name==='feature_availability_column'));
 assert.ok(entry.fields.some(f=>f.name==='label_availability_column'));
});
test('Reporter staking and complete-ticket gates are native component fields',()=>{
 for(const id of ['reporting.BetOutcomeReporter','reporting.BetPerformanceReporter']){
  const field=components.find(c=>c.id===id).fields.find(f=>f.name==='stake_policy');
  assert.equal(field.kind,'component');assert.ok(field.components.includes('evaluation.FractionalKelly'));
 }
 const field=components.find(c=>c.id==='evaluation.AllCombinations').fields.find(f=>f.name==='ticket_gate');
 assert.ok(field.components.includes('evaluation.DecisionLayer'));
});

test('Classification schema preserves numeric and text labels distinctly',()=>{
 const field=components.find(c=>c.id==='composition.OutputSchema').fields.find(f=>f.name==='classes');
 assert.equal(field.item.kind,'typed_scalar');
 let value=0;
 const root=valueEditor(value,v=>value=v,field.item);
 let input=find(root,'Label value','input');
 input.events.change({target:{value:'2'}});
 assert.equal(value,2);
 const type=find(root,'Value type','select');
 type.value='1';type.onchange();
 input=find(root,'Label value','input');
 input.events.change({target:{value:'2'}});
 assert.equal(value,'2');
});
