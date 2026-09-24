// Run with node --experimental-vm-modules --test tests/analytics/ui_feature_discovery.mjs.
// Executes app.js unchanged except for a test-only export of lifecycle entry points.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

class Element {
 constructor(tag='div',attrs={}){this.tag=tag;this.children=[];this.options=this.children;this.classList={toggle(){},add(){}};Object.assign(this,attrs);if(attrs.text)this.textContent=attrs.text;}
 append(...items){this.children.push(...items.flat(Infinity).filter(v=>v!=null));}
 replaceChildren(...items){this.children=[];this.options=this.children;this.append(...items);}
 addEventListener(name,fn,options){(this.listeners??={})[name]={fn,options};}
 showModal(){this.open=true;}
 close(){this.open=false;this.listeners?.close?.fn();}
 remove(){this.removed=true;}
 scrollIntoView(options){this.scrollOptions=options;}
}
const el=(tag,attrs={},...children)=>{const item=new Element(tag,attrs);item.append(...children);return item;};
const flush=()=>new Promise(resolve=>setImmediate(resolve));
const source=readFileSync(new URL('../../src/xdiyo_analytics/ui/static/app.js',import.meta.url),'utf8');
const preview={features:{columns:['home::corners_mean','away::corners_mean'],rows:[],count:23},labels:{columns:['corners'],rows:[],count:23},folds:[{fold:0,train:16,test:7,score:7}],classes:[5,6]};

for(const [section,page,field,title] of [['pre_reporters','pre','features','Features'],['post_reporters','post','targets','Labels']]){
 test('empty '+section+' '+field+' names the correction before Run while Prepare still discovers',async()=>{
  const h=await harness(),recipe=h.app.testState().recipe;
  recipe[section]={Distribution:{component:'reporting.FeatureDistributionReporter',params:{[field]:[]}}};
  await h.query('#run').onclick();await flush();
  assert.equal(h.app.testState().stage,page);
  assert.equal(h.calls.filter(c=>c.action==='run').length,0);
  const message=h.query('#notification').textContent;
  assert.ok(message.includes('Distribution → '+title));
  assert.ok(message.includes('Discover feature columns'));
  await h.query('#prepare').onclick();await flush();
  assert.equal(h.calls.filter(c=>c.action==='prepare').length,1);
  assert.deepEqual(h.discovery.features,preview.features.columns);
  assert.deepEqual(structuredClone(recipe[section].Distribution.params[field]),[]);
  recipe[section].Distribution.params[field]=null;
  await h.query('#run').onclick();await flush();
  assert.equal(h.calls.filter(c=>c.action==='run').length,1);
 });
}

async function harness({deferred=false,previewData=preview,foldIds=null}={}){
 const elements=new Map(),discovery={},seenForms=[],calls=[],jobs=new Map();
 const query=id=>id==='#job-result'?null:elements.get(id)||elements.set(id,new Element()).get(id);
 const recipe={version:1,data:{data_root:'synthetic',seasons:['24_25']},features:{corners_mean:{component:'features.RollingMean',params:{window:5}}},labels:{corners:{}},target:'corners',assembly:{layout:'match'},ratings:{},history:{},feature_options:{},split:{component:'splits.TemporalSplit'},split_options:{},pre_reporters:{plot:{component:'reporting.FeatureDistributionReporter',params:{}}},fitted_reporters:{},post_reporters:{},analysis_options:{},stat_selection:null};
 recipe.fold_ids=structuredClone(foldIds);
 const catalog={recipe,components:[],metrics:[],bundles:{},stages:Object.fromEntries(['data','history','stat_selection','assembly','feature_options','rating_options','candidate','run','split_options'].map(k=>[k,{fields:[]}]))};
 let release;
 const jobResponse=new Promise(resolve=>{release=resolve;});
 const fetch=async(url,options)=>{
  const action=url.split('/').at(-1),body=JSON.parse(options.body);calls.push({action,body});
  const responses={catalog,discover:{publications:[]},choices:{stats:[],teams:[],rounds:[],publications:[]}};
  if(['prepare','run'].includes(action)){
   const id='job-'+(jobs.size+1),selected=body.recipe.fold_ids,p=structuredClone(previewData);
   if(action==='run'&&selected!=null)p.folds=selected.map((position,index)=>({...p.folds[position],fold:index}));
   jobs.set(id,{status:'complete',message:'Completed',action,preview:p});responses[action]={id};
  }
  const value=action==='job'?(deferred?await jobResponse:jobs.get(body.id)):responses[action];
  return {ok:true,json:async()=>structuredClone(value)};
 };
 const body=new Element('body');
 const context=vm.createContext({location:{hash:'#token'},document:{querySelector:query,querySelectorAll:()=>[],title:'',body},window:{addEventListener(){}},localStorage:{getItem:()=>null,setItem(){}},fetch,structuredClone,setTimeout,URL,Blob,console});
 const fakeForm=()=>el('div');
 const exports={el,clone:structuredClone,configure(){},discovered:values=>Object.assign(discovery,structuredClone(values)),fieldsEditor:fakeForm,componentEditor:fakeForm,mapEditor(){seenForms.push(structuredClone(discovery.features||[]));return el('div');},listEditor:fakeForm,valueEditor:fakeForm,spec:()=>({fields:[]}),makeNode:id=>({component:id,params:{}}),featureMapEditor:fakeForm};
 const forms=new vm.SyntheticModule(Object.keys(exports),function(){for(const [name,value] of Object.entries(exports))this.setExport(name,value);},{context});
 const app=new vm.SourceTextModule(source+'\nexport {refreshJob,changed,showHtml};export function testState(){return {recipe,stage,preparedColumns,appliedPreviewJob};}',{context});
 await app.link(()=>forms);await app.evaluate();
 const navigate=async title=>{const link=query('#navigation').children.find(item=>item.textContent===title);assert.ok(link);await link.onclick({preventDefault(){}});};
 const findButton=(node,text)=>node?.tag==='button'&&node.text===text?node:(node?.children||[]).map(child=>findButton(child,text)).find(Boolean);
 return {app:app.namespace,discovery,seenForms,calls,navigate,query,body,release:()=>release([...jobs.values()].at(-1)),findButton};
}

test('report embeds retained HTML and expands into a disposable native dialog without API execution',async()=>{
 const h=await harness(),html='<h1>Retained report</h1>',before=h.calls.length;
 const wrapper=h.app.showHtml(html),embedded=wrapper.children.find(c=>c.tag==='iframe');
 assert.equal(embedded.srcdoc,html);
 assert.equal(embedded.sandbox,'allow-scripts allow-downloads');
 embedded.listeners.load.fn();assert.equal(wrapper.scrollOptions.block,'start');
 h.findButton(wrapper,'Expand report').onclick();
 const dialog=h.body.children.at(-1),expanded=dialog.children.find(c=>c.tag==='iframe');
 assert.equal(dialog.tag,'dialog');assert.equal(dialog.open,true);
 assert.equal(dialog['aria-label'],'Expanded experiment report');
 assert.equal(expanded.srcdoc,html);assert.notEqual(expanded,embedded);
 assert.equal(expanded.sandbox,embedded.sandbox);
 h.findButton(dialog,'Close expanded report').onclick();
 assert.equal(dialog.open,false);assert.equal(dialog.removed,true);
 assert.equal(h.calls.length,before);
 const css=readFileSync(new URL('../../src/xdiyo_analytics/ui/static/style.css',import.meta.url),'utf8');
 assert.match(css,/\.report-view \.report-frame\{[^}]*100dvh - 160px/);
 assert.match(css,/\.report-dialog \.report-frame\{[^}]*min-height:0/);
});

test('discovery updates reporter choices on the same page without a job-result host and only once',async()=>{
 const h=await harness();await h.navigate('Pre-training analysis');
 assert.deepEqual(h.discovery.features,[]);
 const button=h.findButton(h.query('#content'),'Discover feature columns');assert.ok(button);
 await button.onclick();await flush();
 assert.equal(h.app.testState().stage,'pre');
 assert.deepEqual(h.discovery.features,preview.features.columns);
 assert.deepEqual(h.seenForms.at(-1),preview.features.columns);
 assert.equal(h.calls.filter(c=>c.action==='run').length,0);
 const renders=h.seenForms.length;await h.app.refreshJob(false);
 assert.equal(h.seenForms.length,renders);
 h.app.testState().recipe.pre_reporters.plot.params.features=[preview.features.columns[0]];
 h.app.changed();
 assert.deepEqual(h.discovery.features,preview.features.columns);
});
test('an older preparation cannot publish choices after feature edits',async()=>{
 const h=await harness({deferred:true});await h.navigate('Pre-training analysis');
 await h.findButton(h.query('#content'),'Discover feature columns').onclick();
 h.app.testState().recipe.features.corners_mean.params.window=8;h.app.changed();
 h.release();await flush();
 assert.equal(h.app.testState().preparedColumns,null);
 assert.deepEqual(h.discovery.features,[]);
});
test('a run finishing after navigation away from results still refreshes reporter choices',async()=>{
 const h=await harness({deferred:true});
 await h.query('#run').onclick();
 assert.equal(h.app.testState().stage,'results');
 await h.navigate('Pre-training analysis');
 assert.deepEqual(h.discovery.features,[]);
 h.release();await flush();
 assert.equal(h.app.testState().stage,'pre');
 assert.deepEqual(h.discovery.features,preview.features.columns);
 assert.deepEqual(h.seenForms.at(-1),preview.features.columns);
});
test('run completion also populates choices and data or layout edits invalidate them',async()=>{
 const h=await harness();
 await h.query('#run').onclick();await flush();
 assert.deepEqual(h.discovery.features,preview.features.columns);
 await h.navigate('Pre-training analysis');
 assert.deepEqual(h.seenForms.at(-1),preview.features.columns);
 h.app.testState().recipe.assembly.layout='team_match';h.app.changed();
 assert.equal(h.app.testState().preparedColumns,null);
 assert.deepEqual(h.discovery.features,[]);
 assert.deepEqual(h.discovery.fold_ids,[]);
 assert.deepEqual(h.discovery.outer_fold_ids,[]);
 assert.deepEqual(h.discovery.classes,[]);
});

const twoFolds={...preview,folds:[{fold:0,train:8,test:8,score:8},{fold:1,train:16,test:7,score:7}]};
test('empty enabled outer selection discovers all folds on Evaluation without changing the live recipe',async()=>{
 const h=await harness({previewData:twoFolds,foldIds:[]});await h.navigate('Evaluation');
 const button=h.findButton(h.query('#content'),'Discover folds');assert.ok(button);
 await button.onclick();await flush();
 assert.equal(h.app.testState().stage,'evaluation');
 assert.deepEqual(structuredClone(h.app.testState().recipe.fold_ids),[]);
 assert.equal(h.calls.find(c=>c.action==='prepare').body.recipe.fold_ids,null);
 assert.equal(h.calls.filter(c=>c.action==='run').length,0);
 assert.deepEqual(h.discovery.outer_fold_ids.map(c=>c.value),[0,1]);
 assert.deepEqual(h.discovery.fold_ids,[]);
});
test('prepare preserves prior fold choice and selected run cannot overwrite the full outer list',async()=>{
 const h=await harness({previewData:twoFolds,foldIds:[1]});
 await h.query('#prepare').onclick();await flush();
 assert.equal(h.calls.find(c=>c.action==='prepare').body.recipe.fold_ids,null);
 assert.deepEqual(structuredClone(h.app.testState().recipe.fold_ids),[1]);
 assert.deepEqual(h.discovery.outer_fold_ids.map(c=>c.value),[0,1]);
 assert.deepEqual(h.discovery.fold_ids.map(c=>c.value),[0]);
 assert.match(h.discovery.fold_ids[0].label,/outer 1/);
 const outer=structuredClone(h.discovery.outer_fold_ids);
 await h.query('#run').onclick();await flush();
 assert.deepEqual(h.calls.find(c=>c.action==='run').body.recipe.fold_ids,[1]);
 assert.deepEqual(h.discovery.outer_fold_ids,outer);
 assert.deepEqual(h.discovery.fold_ids.map(c=>c.value),[0]);
 await h.navigate('Evaluation');
 assert.deepEqual(structuredClone(h.app.testState().recipe.fold_ids),[1]);
 assert.deepEqual(h.discovery.outer_fold_ids,outer);
});

test('prospective post IDs follow outer order and stale explicit subsets are cleared',async()=>{
 const description='2 leagues · train 22/23, 23/24 · test 24/25 · test rounds 11–38';
 const threeFolds={...preview,folds:[...twoFolds.folds,{fold:2,train:20,test:3,score:3,description}]};
 const h=await harness({previewData:threeFolds});await h.navigate('Evaluation');
 await h.findButton(h.query('#content'),'Discover folds').onclick();await flush();
 const recipe=h.app.testState().recipe;
 recipe.analysis_options.post={fold_ids:[0]};h.app.changed();
 recipe.fold_ids=[2,0];h.app.changed();
 assert.equal(recipe.analysis_options.post.fold_ids,undefined);
 assert.deepEqual(h.discovery.fold_ids.map(c=>c.value),[0,1]);
 assert.match(h.discovery.fold_ids[0].label,/outer 2/);
 assert.ok(h.discovery.fold_ids[0].label.includes(description));
 assert.ok(h.discovery.outer_fold_ids[2].label.includes(description));
 assert.match(h.discovery.fold_ids[1].label,/outer 0/);
 assert.deepEqual(h.discovery.outer_fold_ids.map(c=>c.value),[0,1,2]);
 recipe.analysis_options.post.fold_ids=[0];h.app.changed();
 recipe.fold_ids=[0,2];h.app.changed();
 assert.equal(recipe.analysis_options.post.fold_ids,undefined); // Local0 now refers to a different outer fold.
 assert.match(h.discovery.fold_ids[0].label,/outer 0/);
 assert.match(h.discovery.fold_ids[1].label,/outer 2/);
 recipe.analysis_options.post.fold_ids=[1];h.app.changed();
 recipe.analysis_options.post.title='Renamed report';h.app.changed();
 assert.deepEqual(recipe.analysis_options.post.fold_ids,[1]);
 await h.findButton(h.query('#content'),'Discover folds').onclick();await flush();
 assert.deepEqual(recipe.analysis_options.post.fold_ids,[1]);
});

test('a selected run without full discovery supplies only local post choices',async()=>{
 const h=await harness({previewData:twoFolds,foldIds:[1]});
 await h.query('#run').onclick();await flush();
 assert.deepEqual(h.discovery.outer_fold_ids,[]);
 assert.deepEqual(h.discovery.fold_ids.map(c=>c.value),[0]);
 assert.match(h.discovery.fold_ids[0].label,/outer 1/);
 h.app.testState().recipe.fold_ids=[0];h.app.changed();
 assert.deepEqual(h.discovery.fold_ids,[]); // The retained preview was evaluated for a different selection.
});
