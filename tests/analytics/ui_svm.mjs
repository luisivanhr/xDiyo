// Exercise the real form renderer and event handlers without launching a server.
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

for(const name of ['SVR','SVC']) test(name+' editable gamma, conditional fields and typed parameters',async()=>{
 let recipe=makeNode('sklearn.svm.'+name);
 const root=componentEditor(recipe,v=>{recipe=v;},'model');
 const component=find(root,'Component','select');
 assert.ok(component.children.some(n=>n.value==='sklearn.svm.'+name));
 let gamma=find(root,'Gamma','select');
 assert.equal(gamma.value,'0'); // Scale is the native default.
 gamma.value='2'; gamma.onchange(); await Promise.resolve();
 const numeric=find(root,'Gamma','input');
 assert.equal(numeric.type,'number');
 numeric.value='0.25'; numeric.events.change({target:numeric}); await Promise.resolve();
 assert.equal(recipe.params.gamma,0.25);
 gamma=find(root,'Gamma','select'); gamma.value='1'; gamma.onchange(); await Promise.resolve();
 assert.equal(recipe.params.gamma,'auto');
 const kernel=find(root,'Kernel','select'); kernel.value='0'; kernel.onchange(); await Promise.resolve();
 assert.equal(recipe.params.kernel,'linear');
 assert.equal(find(root,'Gamma','select'),undefined);
 if(name==='SVC') {
  const probability=find(root,'Enable class probabilities','input');
  assert.equal(probability.type,'checkbox'); assert.equal(probability.checked,false);
  probability.checked=true; probability.events.change({target:probability}); await Promise.resolve();
  assert.equal(recipe.params.probability,true);
  assert.ok(find(root,'Enable Random state','input'));
  const weights=find(root,'Enable Class weight','input');
  weights.checked=true; weights.events.change({target:weights}); await Promise.resolve();
  assert.equal(recipe.params.class_weight,'balanced');
 } else assert.equal(find(root,'Epsilon','input').type,'number');
});

test('calibrator exposes margin timing controls and sigmoid without enabling SVC probabilities',async()=>{
 let recipe=makeNode('training.ProbabilityCalibrator');
 const root=componentEditor(recipe,v=>{recipe=v;},null,['training.ProbabilityCalibrator']);
 assert.equal(find(root,'Enable Availability delay','input'),undefined);
 const response=find(root,'Response method','select');
 response.value='1'; response.onchange(); await Promise.resolve();
 assert.equal(recipe.params.response_method,'decision_function');
 const method=find(root,'Method','select');
 assert.ok(method.children.some(n=>n.textContent==='sigmoid'));
 method.value='0'; method.onchange(); await Promise.resolve();
 assert.equal(recipe.params.method,'sigmoid');
 const enable=find(root,'Enable Availability delay','input');
 enable.checked=true; enable.events.change({target:enable}); await Promise.resolve();
 const delay=find(root,'Availability delay','input');
 delay.value='3h'; delay.events.change({target:delay}); await Promise.resolve();
 assert.equal(recipe.params.availability_delay,'3h');
 assert.equal(find(root,'Prediction lead','input').value,'1h');
 assert.equal(find(root,'Min calibration rows','input').value,10);
 assert.equal(find(root,'Min calibration per class','input').value,2);
 assert.deepEqual(makeNode('sklearn.svm.SVC').params,{});
});
