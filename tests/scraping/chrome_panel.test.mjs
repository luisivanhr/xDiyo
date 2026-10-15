import assert from 'node:assert/strict';
import test from 'node:test';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {VERSION} from '../../scraping/sofascore_web_v3/devtools_extension/native.mjs';

const base=new URL('../../scraping/sofascore_web_v3/devtools_extension/',import.meta.url);
const html=readFileSync(new URL('panel.html',base),'utf8');
const code=readFileSync(new URL('panel.mjs',base),'utf8').replace(/^import \{ VERSION \} from '\.\/native\.mjs';\r?\n/m,'');
function panel({omit=[]}={}) {
  const elements=new Map([...html.matchAll(/\bid="([^"]+)"/g)].filter(([,id])=>!omit.includes(id)).map(([,id])=>[id,{
    value:'',textContent:'',disabled:false,listeners:new Map(),querySelectorAll:()=>[],
    addEventListener(name,handler){this.listeners.set(name,handler);},replaceChildren(){},append(){}
  }]));
  const document={getElementById:id=>elements.get(id) ?? null,querySelectorAll:()=>[...elements.values()]};
  const window={};
  vm.runInContext(code,vm.createContext({window,document,VERSION,URL,Blob,setTimeout,clearTimeout}));
  return {window,elements};
}
function application() {
  return {version:VERSION,controller:{state:{phase:'ready',message:'Ready'},connected:false,running:false},
    discovery:{state:{phase:'ready',message:'Ready'},connected:false,running:false},
    probeRunning:false,probeResult:null,manual:null,capture:null};
}

test('current panel renders its discovery and collection controls',()=>{
  const {window,elements}=panel(),app=application();
  app.discovery.state.summary={discovery_id:'discover-current',ready_to_collect:true};
  app.discovery.connected=true;
  assert.doesNotThrow(()=>window.attachXDiyo(app));
  assert.equal(elements.get('version').textContent,`Version ${VERSION}`);
  assert.equal(elements.get('discovery-id').value,'discover-current');
  assert.equal(elements.get('discovery-prepare').disabled,false);
  assert.equal(elements.get('connect').disabled,true);
});

test('an older DevTools app without discovery shows reload guidance instead of crashing',async()=>{
  const {window,elements}=panel(),old=application();
  delete old.discovery;
  let actions=0;old.connect=()=>{actions++;};
  assert.doesNotThrow(()=>window.attachXDiyo(old));
  assert.match(elements.get('campaign-status').textContent,/reload.*DevTools/i);
  assert.equal(elements.get('connect').disabled,true);
  await elements.get('connect').listeners.get('click')();
  assert.equal(actions,0);
  assert.match(elements.get('campaign-status').textContent,/reload.*DevTools/i);
});

test('a panel cannot attach to a different DevTools application version',()=>{
  const {window,elements}=panel(),old=application();old.version='0.3.0';
  assert.doesNotThrow(()=>window.attachXDiyo(old));
  assert.match(elements.get('campaign-status').textContent,/reload.*DevTools/i);
  assert.equal(elements.get('discovery-connect').disabled,true);
});

test('older panel markup cannot cause a missing-control exception after an update',()=>{
  const {window,elements}=panel({omit:['discovery-mode','discovery-prepare']});
  assert.doesNotThrow(()=>window.attachXDiyo(application()));
  assert.match(elements.get('campaign-status').textContent,/reload.*DevTools/i);
  assert.equal(elements.get('connect').disabled,true);
});
