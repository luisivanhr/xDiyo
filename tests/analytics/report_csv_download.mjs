// node --test tests/analytics/report_csv_download.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const source=readFileSync(new URL('../../src/xdiyo_analytics/reporting/viewer.py',import.meta.url),'utf8');
const script=source.match(/\nJS = r"""([\s\S]*?)"""/)[1];

test('CSV clicks preserve exact UTF-8 bytes, full rows and filename; URLs are released',async()=>{
 const csv='\ufeffteam,value,note\r\n'+Array.from({length:205},(_,i)=>`東京 Équipe,${i===0?'':i},"quoted, value"\r\n`).join('');
 const expected=Buffer.from(csv,'utf8');
 const href='data:text/csv;base64,'+expected.toString('base64');
 const link={download:'Équipe "<test>".csv',getAttribute:()=>href,addEventListener(type,fn){assert.equal(type,'click');this.click=fn;}};
 const blobs=[],revoked=[],timers=[],anchors=[],attached=[];
 const document={
  querySelectorAll:selector=>selector.startsWith('a.download')?[link]:[],
  getElementById:()=>({addEventListener(){}}),
  body:{appendChild:a=>attached.push(a)},
  createElement(tag){assert.equal(tag,'a');const a={click(){assert.ok(attached.includes(a));a.clicked=true;},remove(){a.removed=true;}};anchors.push(a);return a;}
 };
 vm.runInNewContext(script,{window:{addEventListener(){}},document,Blob,Uint8Array,atob,URL:{createObjectURL(blob){blobs.push(blob);return 'blob:'+blobs.length;},revokeObjectURL:url=>revoked.push(url)},setTimeout(fn,ms){assert.equal(ms,1000);timers.push(fn);},requestAnimationFrame(){}});
 assert.equal(blobs.length,0,'no eager decoded CSV or Blob allocation');
 for(let i=0;i<2;i++){
  let prevented=false;link.click({preventDefault(){prevented=true;}});
  assert.equal(prevented,true);assert.equal(anchors[i].download,link.download);
  assert.equal(anchors[i].href,`blob:${i+1}`);assert.equal(anchors[i].clicked,true);assert.equal(anchors[i].removed,true);
  assert.deepEqual(Buffer.from(await blobs[i].arrayBuffer()),expected);
  assert.equal(blobs[i].type,'text/csv;charset=utf-8');
 }
 assert.equal(link.getAttribute('href'),href,'original offline fallback remains intact');
 assert.deepEqual(revoked,[]);timers.forEach(fn=>fn());assert.deepEqual(revoked,['blob:1','blob:2']);
});


test('builder accepts only exact CSV links from a mounted opaque-origin report',()=>{
 const app=readFileSync(new URL('../../src/xdiyo_analytics/ui/static/app.js',import.meta.url),'utf8');
 const helper=app.slice(app.indexOf('const reportCsvLinks='),app.indexOf('function showHtml(html)'));
 const bytes=Buffer.from('\ufeffvalue\r\n0\r\n','utf8'),href='data:text/csv;base64,'+bytes.toString('base64');
 const source={},frame={contentWindow:source,srcdoc:'known report'},calls=[];let mounted=[frame],parses=0;
 class DOMParser {parseFromString(html,type){assert.equal(html,frame.srcdoc);assert.equal(type,'text/html');parses++;return {querySelectorAll:()=>[{getAttribute:name=>name==='href'?href:'data.csv'}]};}}
 const ctx=vm.createContext({document:{querySelectorAll:()=>mounted},window:{addEventListener(){}},DOMParser,atob,Uint8Array,download:(...args)=>calls.push(args)});
 vm.runInContext(helper,ctx);
 const valid={source,origin:'null',data:{type:'xdiyo-csv-download',href,filename:'data.csv'}};
 const dispatch=e=>{ctx.event=e;vm.runInContext('reportCsvDownload(event)',ctx);};
 for(const invalid of [{...valid,source:{}},{...valid,origin:'https://untrusted.example'}, {...valid,data:{...valid.data,href:'https://untrusted.example/file'}},{...valid,data:{...valid.data,filename:'other.csv'}},{...valid,data:{...valid.data,href:'data:text/csv;base64,AA=='}}])dispatch(invalid);
 assert.equal(calls.length,0);
 dispatch(valid);dispatch(valid);assert.equal(calls.length,2);assert.equal(parses,1);
 assert.deepEqual(Buffer.from(calls[0][0]),bytes);assert.equal(calls[0][1],'data.csv');assert.equal(calls[0][2],'text/csv;charset=utf-8');
 mounted=[];dispatch(valid);assert.equal(calls.length,2,'detached frames lose download access');
});

test('embedded CSV forwards only after a parent handshake without decoding eagerly',()=>{
 const messages=[],listeners={};let click;
 const parent={postMessage:(...args)=>messages.push(args)};
 const href='data:text/csv;base64,AA==';
 const link={download:'data.csv',getAttribute:()=>href,addEventListener:(name,fn)=>{click=fn;}};
 const document={querySelectorAll:s=>s.startsWith('a.download')?[link]:[],getElementById:()=>({addEventListener(){}})};
 vm.runInNewContext(script,{window:{parent,addEventListener:(name,fn)=>{listeners[name]=fn;}},document,requestAnimationFrame(){}});
 listeners.message({source:parent,data:{type:'xdiyo-csv-host'}});
 let prevented=false;click({preventDefault(){prevented=true;}});
 assert.equal(prevented,true);assert.equal(messages.length,1);assert.equal(messages[0][0].href,href);assert.equal(messages[0][0].filename,'data.csv');assert.equal(messages[0][1],'*');
});
