import assert from 'node:assert/strict';
import test from 'node:test';
import vm from 'node:vm';
import {selectSessionHeaders,probeInPage,runSessionProbe} from '../../scraping/sofascore_web_v3/devtools_extension/session-probe.mjs';

const now=1789301700000;
const token=exp=>`header.${Buffer.from(JSON.stringify({exp})).toString('base64url')}.signature`;
const headers={'X-Captcha':token(now/1000+300),'X-Requested-With':'test-site-header'};
function entry({age=1000,url='https://www.sofascore.com/api/v1/event/123/lineups',status=200,...options}={}) {
  return {startedDateTime:new Date(now-age).toISOString(),request:{method:'GET',url,
    headers:Object.entries(headers).map(([name,value])=>({name,value})),...options},response:{status}};
}
test('selects the newest successful first-party session without returning unrelated headers',()=>{
  const old=entry({age:10000}),fresh=entry({age:1000});
  fresh.request.headers.push({name:'Cookie',value:'private-cookie'});
  const selected=selectSessionHeaders([old,fresh],now);
  assert.equal(selected.started,now-1000);
  assert.deepEqual(selected.headers,headers);
});
for (const [label,options] of Object.entries({
  foreign:{url:'https://example.com/api/v1/event/123/lineups'},
  query:{url:'https://www.sofascore.com/api/v1/event/123/lineups?extra=1'},
  unrelated:{url:'https://www.sofascore.com/api/v1/user/123'},
  denial:{status:403}, stale:{age:120001}, future:{age:-1}, post:{method:'POST'},
})) test(`ignores ${label} token observations`,()=>assert.equal(selectSessionHeaders([entry(options)],now),null));

test('rejects expired, malformed, duplicated and too-nearly-expiring tokens',()=>{
  for (const value of [token(now/1000-1),token(now/1000+29),'not-a-jwt']) {
    const request=entry(); request.request.headers[0].value=value;
    assert.equal(selectSessionHeaders([request],now),null);
  }
  const duplicate=entry(); duplicate.request.headers.push({...duplicate.request.headers[0]});
  assert.equal(selectSessionHeaders([duplicate],now),null);
});

test('a fresh successful request without a captcha records observation access, not a token expiry',()=>{
  const observed=entry({headers:[{name:'X-Requested-With',value:'site-client'}]});
  const selected=selectSessionHeaders([observed],now);
  assert.ok(selected);
  assert.deepEqual(selected.headers,{'X-Requested-With':'site-client'});
  assert.equal(selected.expiresAt,null);
  assert.equal(selected.accessMode,'observed_request');
  observed.response.status=304;
  assert.equal(selectSessionHeaders([observed],now).sourceStatus,304);
});

test('missing captcha requires fresh successful access, and malformed or duplicated captcha cannot become token-free access',()=>{
  const base={headers:[{name:'X-Requested-With',value:'site-client'}]};
  for(const options of [{...base,status:404},{...base,status:403},{...base,age:90001},
    {headers:[]},{headers:[...base.headers,{name:'X-Captcha',value:''}]},
    {headers:[...base.headers,{name:'X-Captcha',value:'bad'},{name:'x-captcha',value:'bad'}]}])
    assert.equal(selectSessionHeaders([entry(options)],now),null);
});

function page({origin='https://www.sofascore.com',visible='',status=200,body={event:{id:123}}}={}) {
  const calls=[];
  const context=vm.createContext({location:{origin},document:{body:{innerText:visible}},
    Date:class extends Date {static now(){return now;}},AbortController,TextDecoder,setTimeout,clearTimeout,
    fetch:async(url,options)=>{calls.push({url,options});return new Response(JSON.stringify(body),{
      status,headers:{'content-type':'application/json',server:'nginx','set-cookie':'must-not-export'}});}});
  const execute=vm.runInContext(`(${probeInPage.toString()})`,context);
  return {calls,execute};
}
test('makes exactly one same-origin GET without navigation and returns only redacted identity evidence',async()=>{
  const p=page(),result=await p.execute(123,headers,now+300000,now+10000);
  assert.equal(p.calls.length,1);
  assert.equal(p.calls[0].url,'/api/v1/event/123');
  assert.equal(p.calls[0].options.credentials,'same-origin');
  assert.equal(p.calls[0].options.redirect,'error');
  assert.equal(result.identityVerified,true);
  assert.equal(result.requestCount,1);
  const encoded=JSON.stringify(result);
  for (const secret of [headers['X-Captcha'],headers['X-Requested-With'],'must-not-export']) assert.equal(encoded.includes(secret),false);
});
for (const [label,options,id,expiry,deadline] of [
  ['foreign origin',{origin:'https://example.com'},123,now+300000,now+10000],
  ['visible challenge',{visible:'Verify you are human'},123,now+300000,now+10000],
  ['invalid ID',{},0,now+300000,now+10000],
  ['expired token',{},123,now-1,now+10000],
  ['delayed script',{},123,now+300000,now-1],
]) test(`${label} sends zero requests`,async()=>{
  const p=page(options); await assert.rejects(p.execute(id,headers,expiry,deadline));assert.equal(p.calls.length,0);
});
test('wrong returned match cannot count as verified access',async()=>{
  const p=page({body:{event:{id:456}}}),result=await p.execute(123,headers,now+300000,now+10000);
  assert.equal(result.identityVerified,false);
});
test('records provider challenge evidence and never retries a denial',async()=>{
  const p=page({status:403,body:{error:{code:403,reason:'challenge',token:'must-not-export'}}});
  const result=await p.execute(123,headers,now+300000,now+10000);
  assert.equal(result.apiError.reason,'challenge');assert.equal(p.calls.length,1);
  assert.equal(JSON.stringify(result).includes('must-not-export'),false);
});
test('bounds oversized response evidence',async()=>{
  const p=page({body:{padding:'x'.repeat(1024*1024+1)}});
  const result=await p.execute(123,headers,now+300000,now+10000);
  assert.equal(result.bodyTooLarge,true);assert.equal(result.identityVerified,undefined);
});
test('probe stays in the inspected tab and does not export session headers',async()=>{
  let script;
  const chrome={devtools:{inspectedWindow:{tabId:7},network:{getHAR:cb=>cb({entries:[entry()]})}},
    scripting:{executeScript:async value=>{script={...value,args:structuredClone(value.args)};return [{result:{status:200,identityVerified:true}}];}}};
  const result=await runSessionProbe(chrome,123,{now:()=>now});
  assert.equal(script.target.tabId,7);assert.equal(script.world,'MAIN');assert.equal(script.args[0],123);
  assert.equal(result.sourcePath,'/api/v1/event/123/lineups');
  assert.equal(JSON.stringify(result).includes(headers['X-Captcha']),false);
});
test('missing successful session makes zero page invocations',async()=>{
  let calls=0;
  const chrome={devtools:{network:{getHAR:cb=>cb({entries:[entry({status:403})]})}},
    scripting:{executeScript:()=>{calls++;}}};
  await assert.rejects(runSessionProbe(chrome,123,{now:()=>now}),/no request sent/);
  assert.equal(calls,0);
});

test('probe mirrors observed token-free access with a bounded freshness deadline and no secret export',async()=>{
  const observed=entry({headers:[{name:'X-Requested-With',value:'site-client'}]});
  let script;
  const p=page();
  const chrome={devtools:{inspectedWindow:{tabId:7},network:{getHAR:cb=>cb({entries:[observed]})}},
    scripting:{executeScript:async value=>{script=value;return [{result:await p.execute(...value.args)}];}}};
  const result=await runSessionProbe(chrome,123,{now:()=>now});
  assert.equal(p.calls.length,1);assert.equal(result.identityVerified,true);
  assert.equal(p.calls[0].options.headers['X-Captcha'],undefined);
  assert.equal(script.args[2],now-1000+120000);
  assert.equal(result.accessMode,'observed_request');assert.equal(result.tokenRemainingMs,null);
  assert.equal(JSON.stringify(result).includes('site-client'),false);
});
