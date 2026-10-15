import assert from 'node:assert/strict';
import test from 'node:test';
import vm from 'node:vm';
import {CampaignController} from '../../scraping/sofascore_web_v3/devtools_extension/controller.mjs';
import {ChromeSessionClient,requestInPage} from '../../scraping/sofascore_web_v3/devtools_extension/session-client.mjs';
import {normalizeSessionAnchor} from '../../scraping/sofascore_web_v3/devtools_extension/session-collection.mjs';

const now=1789301700000;
const headers={'X-Captcha':'local-session-sentinel','X-Requested-With':'site-client'};
function page({origin='https://www.sofascore.com',visible='',status=200,body={event:{id:123}}}={}) {
  const calls=[];
  const context=vm.createContext({location:{origin},document:{body:{innerText:visible}},
    Date:class extends Date {static now(){return now;}},AbortController,TextDecoder,setTimeout,clearTimeout,
    fetch:async(url,options)=>{calls.push({url,options});return new Response(JSON.stringify(body),{
      status,headers:{'content-type':'application/json','set-cookie':'not-for-export'}});}});
  return {calls,execute:vm.runInContext(`(${requestInPage.toString()})`,context)};
}

test('session component GET returns importable content without exporting headers',async()=>{
  const p=page(),entry=await p.execute(123,'/event/123',headers,now+300000,now+10000);
  assert.equal(p.calls.length,1);assert.equal(p.calls[0].options.credentials,'same-origin');
  assert.equal(p.calls[0].options.redirect,'error');
  assert.deepEqual(JSON.parse(entry.response.content.text),{event:{id:123}});
  assert.equal(entry._captureStatus,'body_saved');
  assert.equal(entry._observationSource,'chrome_session_request');
  assert.equal(JSON.stringify(entry).includes('sentinel'),false);
  assert.equal(JSON.stringify(entry).includes('not-for-export'),false);
});

test('observed token-free access renews its local freshness only on successful requests',async()=>{
  let time=now,listener,scripts=0;
  const p=page();
  const chrome={devtools:{inspectedWindow:{tabId:7},network:{onRequestFinished:{addListener:f=>{listener=f;}}}},
    scripting:{executeScript:async value=>{scripts++;assert.equal(value.args[2]['X-Captcha'],undefined);
      assert.equal(value.args[3],time+120000);return [{result:await p.execute(...value.args)}];}}};
  const client=new ChromeSessionClient(chrome,{now:()=>time});
  listener({startedDateTime:new Date(time).toISOString(),request:{method:'GET',
    url:'https://www.sofascore.com/api/v1/event/123/lineups',headers:[{name:'X-Requested-With',value:'site-client'}]},response:{status:200}});
  assert.equal(client.ready(110000),true);assert.equal(client.renewalDelayMs(110000),0);
  const job={event_id:123,paths:['/event/123']};
  await client.request(job,'/event/123');assert.equal(scripts,1);
  time+=60000;
  assert.equal(client.ready(110000),true);
  time+=60001;
  await assert.rejects(client.request(job,'/event/123'),/No recently successful/);
  assert.equal(scripts,1);
  client.beginRenewal();assert.equal(client.ready(),false);
});

test('token-free requests retain header validation and stop after a denial',async()=>{
  const p=page({status:403,body:{error:{code:403,reason:'challenge'}}});
  const result=await p.execute(123,'/event/123',{'X-Requested-With':'site-client'},now+120000,now+10000);
  assert.equal(result.response.status,403);assert.equal(p.calls.length,1);
  for(const invalid of [{},{'X-Captcha':'orphan'},{'X-Requested-With':'bad\r\nvalue'},
    {'X-Requested-With':'ok',Cookie:'private'}])
    await assert.rejects(p.execute(123,'/event/123',invalid,now+120000,now+10000),/Invalid session headers/);
  assert.equal(p.calls.length,1);
});

test('session anchor removes an observed section fragment while preserving exact host and match identity',()=>{
  const base='https://www.sofascore.com/football/match/real-madrid-osasuna/vgbsEgb';
  assert.deepEqual(normalizeSessionAnchor({url:base+'#id:10408370,tab:lineups',eventId:10408370}),
    {url:base+'#id:10408370',event_id:10408370});
  for(const [url,eventId] of [[base+'#id:10408370',1],[base+'#id:10408370,tab:unknown',10408370],
    [base+'?extra=1#id:10408370',10408370],[base.replace('www.sofascore.com','example.com')+'#id:10408370',10408370]])
    assert.throws(()=>normalizeSessionAnchor({url,eventId}),/Invalid queued match URL/);
});
for(const path of ['/event/124','/event/123/comments','/event/123/h2h','/event/123/lineups?all=1','/event/123/lineups#x','https://example.com/','/event/123/heatmap/0'])
  test(`rejects unrequested route ${path}`,async()=>{
    const p=page();await assert.rejects(p.execute(123,path,headers,now+300000,now+10000));assert.equal(p.calls.length,0);
  });
test('wrong origin, challenge and expired execution send zero requests',async()=>{
  for(const options of [{origin:'https://example.com'},{visible:'Verify you are human'}]) {
    const p=page(options);await assert.rejects(p.execute(123,'/event/123',headers,now+300000,now+10000));assert.equal(p.calls.length,0);
  }
  const p=page();await assert.rejects(p.execute(123,'/event/123',headers,now+300000,now-1));assert.equal(p.calls.length,0);
});
test('denials retain bounded error evidence and never retry or export their body',async()=>{
  const p=page({status:403,body:{error:{code:403,reason:'challenge',token:'must-not-export'}}});
  const entry=await p.execute(123,'/event/123',headers,now+300000,now+10000);
  assert.equal(entry._apiError.reason,'challenge');assert.equal(p.calls.length,1);
  assert.deepEqual(Object.keys(entry.response.content),[]);assert.equal(JSON.stringify(entry).includes('must-not-export'),false);
});

function setup({cached=false,status=200,pauseDuring=false,invalidIdentity=false,ackError=false,jitter=null,resetEvery=null,
  sessionReady=true,renewSucceeds=true,responseCodes=null,responseReasons=null,recoveryEnabled=false,
  recoveryCooldown=180,expiryInMs=null,pauseDuringRest=false,renewalChallenge=false,maxRequests=120,
  requestStartDelays=null,delayFailure=false,pauseDuringDelay=false,earlyDelay=false,slowBrowserTimers=false,
  restStallMs=0,renewalDelayMs=0,pacingNativeOverheadMs=0,daily=false,upcoming=false}={}) {
  const calls=[],requests=[],actions=[],sleeps=[],nativeWaits=[],stored=new Map();
  let time=now,index=0,current,ready=expiryInMs==null?sessionReady:false,restStalled=false;
  const delayTime=ms=>{
    time+=ms;
    if(ms===1000 && !restStalled && restStallMs) {time+=restStallMs;restStalled=true;}
    if(pauseDuringRest && ms===1000)control.halt('Paused during recovery rest');
  };
  const coverage=()=>{
    const finalPath=upcoming?{path:`/event/${current}/lineups`,component:'lineups'}:{path:`/event/${current}/heatmap/9`,component:'heatmap'};
    const planned=[{path:`/event/${current}`,component:'event'}];
    if(!daily || stored.get(`/event/${current}`)?.response.status===200) planned.push(finalPath);
    const components=planned.map(item=>{
      const entry=stored.get(item.path);
      return {...item,status:entry?.response.status===403?'blocked':entry?.response.status===404?'unavailable':
        entry?(invalidIdentity && item.component==='event'?'invalid_or_missing_body':'success'):'not_observed',
        http_status:entry?.response.status,observed_at:entry?.startedDateTime};
    });
    return {components,complete:components.every(item=>['success','unavailable'].includes(item.status)),blocked:components.some(item=>item.status==='blocked')};
  };
  const native={connect:async()=>({version:'test'}),call:async(op,args={})=>{
    calls.push({op,...args});
    if(op==='open')return {};
    if(op==='next') {
      if(index===2)return {done:true,status:{state:'complete'}};
      current=123+index++;
      if(cached)stored.set(`/event/${current}/heatmap/9`,{response:{status:200}});
      return {event_id:current,lease:`lease${current}`,wait_until:0,label:`Match ${current}`,
        paths:daily?[`/event/${current}`]:[`/event/${current}/heatmap/9`,`/event/${current}`],dynamic_paths:daily,coverage:coverage(),
        policy:{collection_mode:'session',session_request_interval_seconds:.5,random_action_delay_max_seconds:jitter,
          match_timeout_seconds:90,max_api_observations_per_match:maxRequests,memory_reset_every_matches:resetEvery,
          session_challenge_recovery:recoveryEnabled,session_challenge_cooldown_seconds:recoveryCooldown}};
    }
    if(op==='observe') {if(ackError)throw new Error('disk full');stored.set(new URL(args.entry.request.url).pathname.slice(7),args.entry);return {ack:'saved'};}
    if(op==='progress')return coverage();
    if(op==='delay') {
      if(delayFailure)throw new Error('native pacing failed');
      if(pauseDuringDelay)return {waited_ms:0,pause_requested:true,run_limit_reached:false};
      if(earlyDelay)return {waited_ms:0,pause_requested:false};
      nativeWaits.push(args.milliseconds);delayTime(args.milliseconds);time+=pacingNativeOverheadMs;
      return {waited_ms:args.milliseconds,pause_requested:false,run_limit_reached:false};
    }
    if(op==='finish') {assert.equal(coverage().complete,true);assert.equal(args.page_event_id,null);return {state:'complete'};}
    if(op==='pause')return {paused:true};
    throw new Error(`Unexpected ${op}`);
  }};
  const browser={snapshot:async()=>({eventId:999,url:'https://www.sofascore.com/football/match/anchor/a#id:999',challenge:renewalChallenge && actions.length>0}),
    resetMemory:async job=>{actions.push(['reset',job]);time+=renewalDelayMs;ready=renewSucceeds && (expiryInMs==null || time>now+expiryInMs);},
    navigate:()=>{throw new Error('Must not navigate to queued fixtures');},action:()=>{throw new Error('Must not navigate through H2H or views');}};
  const session={ready:()=>ready,refresh:async()=>{},beginRenewal:()=>{ready=false;},
    renewalDelayMs:()=>expiryInMs!=null && time<now+expiryInMs?now+expiryInMs-time+1000:0,
    request:async(job,path)=>{
    assert.ok(job.paths.includes(path),'Each request must be permitted by the current Python plan');
    const code=responseCodes?.[requests.length] ?? status, reason=responseReasons?.[requests.length] ?? 'challenge';
    time+=requestStartDelays?.[requests.length] ?? 0;
    const started=time;requests.push({path,at:time});time+=100;
    if(pauseDuring)control.halt('Paused by user');
    return {startedDateTime:new Date(started).toISOString(),time:100,request:{method:'GET',url:`https://www.sofascore.com/api/v1${path}`},
      response:{status:code,content:{text:'{}'}},_apiError:code===403?{reason}:undefined};
  }};
  const control=new CampaignController({network:{},native,browser,session,now:()=>time,random:()=>.25,
    sleep:async ms=>{sleeps.push(ms);delayTime(slowBrowserTimers?Math.max(1000,ms):ms);}});
  return {control,calls,requests,actions,sleeps,nativeWaits};
}
async function run(options) {const state=setup(options);await state.control.connect('test');await state.control.run();return state;}

test('daily jobs expand metadata-only leases into final endpoints without opening pages',async()=>{
  const s=await run({daily:true});
  assert.equal(s.control.state.phase,'finished');
  assert.deepEqual(s.requests.map(r=>r.path),['/event/123','/event/123/heatmap/9','/event/124','/event/124/heatmap/9']);
  assert.deepEqual(s.actions,[]);
});

test('upcoming daily jobs request lineups and never assume final statistics',async()=>{
  const s=await run({daily:true,upcoming:true,responseCodes:[200,404,200,404]});
  assert.equal(s.control.state.phase,'finished');
  assert.deepEqual(s.requests.map(r=>r.path),['/event/123','/event/123/lineups','/event/124','/event/124/lineups']);
  assert.deepEqual(s.actions,[]);
});

test('collects queued components with metadata first and never opens match pages or H2H',async()=>{
  const s=await run();assert.equal(s.control.state.phase,'finished');
  assert.deepEqual(s.requests.map(r=>r.path),['/event/123','/event/123/heatmap/9','/event/124','/event/124/heatmap/9']);
  assert.deepEqual(s.actions,[]);assert.equal(s.calls.filter(c=>c.op==='finish').length,2);
  assert.ok(s.requests.slice(1).every((r,i)=>r.at-s.requests[i].at>=500));
});
test('acknowledged terminal components are reused on resume',async()=>{
  const s=await run({cached:true});assert.deepEqual(s.requests.map(r=>r.path),['/event/123','/event/124']);
});
test('jitter adds a bounded wait to the minimum request spacing',async()=>{
  const s=await run({jitter:2});assert.ok(s.nativeWaits.includes(500));
  assert.ok(s.requests.slice(1).every((r,i)=>r.at-s.requests[i].at>=1000));
});

test('short request pacing uses the native clock even when browser timers overshoot',async()=>{
  const s=await run({slowBrowserTimers:true});
  assert.equal(s.control.state.phase,'finished');assert.deepEqual(s.sleeps,[]);
  assert.deepEqual(s.requests.slice(1).map((r,i)=>r.at-s.requests[i].at),[500,500,500]);
  for(const report of s.calls.filter(c=>c.op==='finish')) {
    assert.equal(report.page_evidence.session_pacing_backend,'native');
    assert.ok(report.page_evidence.session_pacing_actual_wait_ms>0);
  }
});

test('spacing follows actual API start times after uneven script dispatch delays',async()=>{
  const s=await run({requestStartDelays:[250,0,200,0]});
  assert.equal(s.control.state.phase,'finished');
  assert.ok(s.requests.slice(1).every((r,i)=>r.at-s.requests[i].at>=500));
});

for(const [option,reason] of [['delayFailure',/native pacing failed/],['pauseDuringDelay',/User pause requested/],['earlyDelay',/ended before/]]) {
  test(`${option} stops before another provider request`,async()=>{
    const s=await run({[option]:true});assert.equal(s.requests.length,1);
    assert.equal(s.calls.filter(c=>c.op==='finish').length,0);assert.match(s.control.state.message,reason);
  });
}
test('a new 403 is acknowledged once and stops before another request or match',async()=>{
  const s=await run({status:403});assert.equal(s.requests.length,1);
  assert.equal(s.calls.filter(c=>c.op==='observe').length,1);assert.equal(s.calls.filter(c=>c.op==='finish').length,0);
  assert.match(s.control.state.message,/HTTP 403.*challenge/);
});
test('a pause during GET still acknowledges the response, then stops',async()=>{
  const s=await run({pauseDuring:true});assert.equal(s.requests.length,1);
  assert.equal(s.calls.filter(c=>c.op==='observe').length,1);assert.equal(s.calls.filter(c=>c.op==='finish').length,0);
});
test('invalid returned identity stops before component requests',async()=>{
  const s=await run({invalidIdentity:true});assert.equal(s.requests.length,1);assert.match(s.control.state.message,/invalid_or_missing_body/);
});
test('writer failure stops request production and preserves incomplete queue',async()=>{
  const s=await run({ackError:true});assert.equal(s.requests.length,1);assert.match(s.control.state.message,/disk full/);
});
test('periodic renewal uses the existing anchor between acknowledged matches',async()=>{
  const s=await run({resetEvery:1});assert.equal(s.control.state.phase,'finished');
  assert.deepEqual(s.actions,[['reset',{url:'https://www.sofascore.com/football/match/anchor/a#id:999',event_id:999}]]);
  const finishes=s.calls.filter(c=>c.op==='finish');
  assert.equal(finishes[1].page_evidence.memory_reset_before_match,true);
});
test('unusable session gets one normal renewal and no data request if renewal fails',async()=>{
  const s=await run({sessionReady:false,renewSucceeds:false});
  assert.equal(s.actions.length,1);assert.equal(s.requests.length,0);assert.match(s.control.state.message,/not established/);
});
test('browser client never sends a component outside the saved profile',async()=>{
  let scripts=0;
  const chrome={devtools:{network:{onRequestFinished:{addListener:()=>{}}}},scripting:{executeScript:()=>{scripts++;}}};
  const client=new ChromeSessionClient(chrome);
  await assert.rejects(client.request({event_id:123,paths:['/event/123']},'/event/123/lineups'),/outside/);
  assert.equal(scripts,0);
});

test('one challenge rest renews the anchor and retries only the failed component',async()=>{
  const s=await run({recoveryEnabled:true,responseCodes:[200,403,200,200,200]});
  assert.equal(s.control.state.phase,'finished');
  assert.deepEqual(s.requests.map(r=>r.path),['/event/123','/event/123/heatmap/9','/event/123/heatmap/9','/event/124','/event/124/heatmap/9']);
  assert.ok(s.requests[2].at-s.requests[1].at>=180000);
  assert.equal(s.actions.length,1);
  const report=s.calls.find(c=>c.op==='finish').page_evidence;
  assert.equal(report.session_request_count,3);
  assert.equal(report.session_challenge_recoveries,1);
  assert.equal(report.session_recovery_wait_ms,180000);
  assert.equal(report.session_renewal_count,1);
  assert.equal(report.session_renewed_before_match,false);
});

test('a second challenge stops without a second renewal or a later component',async()=>{
  const s=await run({recoveryEnabled:true,status:403});
  assert.equal(s.requests.length,2);assert.equal(s.actions.length,1);
  assert.equal(s.calls.filter(c=>c.op==='observe').length,2);
  assert.equal(s.calls.filter(c=>c.op==='finish').length,0);
  assert.match(s.control.state.message,/HTTP 403/);
});

for(const options of [{status:401},{status:429},{status:403,responseReasons:['Forbidden']},{status:403,maxRequests:1}])
  test(`nonrecoverable denial is not renewed ${JSON.stringify(options)}`,async()=>{
    const s=await run({recoveryEnabled:true,...options});
    assert.equal(s.requests.length,1);assert.equal(s.actions.length,0);
    assert.equal(s.calls.filter(c=>c.op==='finish').length,0);
  });

test('pausing the challenge rest saves the denial and makes no renewal or retry',async()=>{
  const s=await run({recoveryEnabled:true,status:403,pauseDuringRest:true});
  assert.equal(s.requests.length,1);assert.equal(s.actions.length,0);
  assert.equal(s.calls.filter(c=>c.op==='observe').length,1);
  assert.match(s.control.state.message,/Paused during recovery rest/);
});

test('failed or interactive renewal never retries the blocked component',async()=>{
  for(const options of [{renewSucceeds:false},{renewalChallenge:true}]) {
    const s=await run({recoveryEnabled:true,status:403,...options});
    assert.equal(s.requests.length,1);assert.equal(s.actions.length,1);
    assert.equal(s.calls.filter(c=>c.op==='finish').length,0);
  }
});

test('near-expiry session waits before renewal without spending the collection deadline',async()=>{
  const s=await run({expiryInMs:100000});
  assert.equal(s.control.state.phase,'finished');assert.equal(s.actions.length,1);
  assert.ok(s.requests[0].at>now+100000);
  assert.equal(s.calls.find(c=>c.op==='finish').page_evidence.session_expiry_wait_ms,101000);
});

test('an overslept challenge rest preserves the active match budget and retries only the failed component',async()=>{
  const s=await run({recoveryEnabled:true,responseCodes:[200,403,200,200,200],restStallMs:280000});
  assert.equal(s.control.state.phase,'finished');
  assert.deepEqual(s.requests.map(r=>r.path),['/event/123','/event/123/heatmap/9','/event/123/heatmap/9','/event/124','/event/124/heatmap/9']);
  const report=s.calls.find(c=>c.op==='finish').page_evidence;
  assert.equal(report.session_recovery_wait_ms,180000);
  assert.equal(report.session_recovery_actual_wait_ms,281000);
  assert.equal(report.session_challenge_recoveries,1);
  assert.ok(s.nativeWaits.includes(1000));
  assert.equal(s.actions.length,1);
});

test('an overslept expiry rest preserves the budget without making an early provider request',async()=>{
  const s=await run({expiryInMs:100000,restStallMs:280000});
  assert.equal(s.control.state.phase,'finished');
  assert.ok(s.requests[0].at>=now+281000);
  const report=s.calls.find(c=>c.op==='finish').page_evidence;
  assert.equal(report.session_expiry_wait_ms,101000);
  assert.equal(report.session_expiry_actual_wait_ms,281000);
});

test('slow active renewal still consumes the match budget after an excluded rest',async()=>{
  const s=await run({recoveryEnabled:true,status:403,restStallMs:280000,renewalDelayMs:91000});
  assert.equal(s.control.state.phase,'paused');
  assert.equal(s.actions.length,1);
  assert.equal(s.requests.length,1);
  assert.equal(s.calls.filter(c=>c.op==='finish').length,0);
  assert.match(s.control.state.message,/Match time budget/);
});

test('pacing reports distinguish Python sleep from the native round trip',async()=>{
  const s=await run({pacingNativeOverheadMs:200});
  assert.equal(s.control.state.phase,'finished');
  const report=s.calls.find(c=>c.op==='finish').page_evidence;
  assert.equal(report.session_pacing_native_waited_ms,400);
  assert.equal(report.session_pacing_native_roundtrip_ms,600);
  assert.equal(report.session_pacing_actual_wait_ms,600);
  assert.equal(report.session_pacing_progress_ms,0);
});

test('session renewal rejects old Network candidates and accepts a new successful observation',async()=>{
  let time=now,listener,log;
  const candidate=started=>({startedDateTime:new Date(started).toISOString(),
    request:{method:'GET',url:'https://www.sofascore.com/api/v1/event/999/lineups',headers:[
      {name:'X-Captcha',value:`test.${btoa(JSON.stringify({exp:Math.floor((now+100000)/1000)}))}.signature`},
      {name:'X-Requested-With',value:'test-client'}]},response:{status:200}});
  log={entries:[candidate(now-1000)]};
  const client=new ChromeSessionClient({devtools:{network:{onRequestFinished:{addListener:f=>{listener=f;}},getHAR:f=>f(log)}}},{now:()=>time});
  await client.refresh();assert.equal(client.ready(30000),true);assert.equal(client.renewalDelayMs(110000),101000);
  client.beginRenewal();assert.equal(client.ready(),false);
  await client.refresh();assert.equal(client.ready(),false);
  listener(log.entries[0]);assert.equal(client.ready(),false);
  time+=10;listener(candidate(time));assert.equal(client.ready(),true);
});
