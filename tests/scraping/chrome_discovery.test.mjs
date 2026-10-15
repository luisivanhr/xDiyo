import assert from 'node:assert/strict';
import test from 'node:test';
import vm from 'node:vm';
import {DiscoveryController} from '../../scraping/sofascore_web_v3/devtools_extension/discovery.mjs';
import {requestInPage,ChromeSessionClient} from '../../scraping/sofascore_web_v3/devtools_extension/session-client.mjs';
import {NativeClient,VERSION,PROTOCOL} from '../../scraping/sofascore_web_v3/devtools_extension/native.mjs';

const now=1789646400000,headers={'X-Requested-With':'observed-site-value'};
const spec={competition_id:17,season_id:900};
function page() {
  const calls=[];
  const context=vm.createContext({location:{origin:'https://www.sofascore.com'},document:{body:{innerText:''}},
    Date:class extends Date {static now(){return now;}},AbortController,TextDecoder,setTimeout,clearTimeout,
    fetch:async(url,options)=>{calls.push({url,options});return new Response('{"rounds":[{"round":1}]}',{
      status:200,headers:{'content-type':'application/json'}});}});
  return {calls,execute:vm.runInContext(`(${requestInPage.toString()})`,context)};
}
for(const path of ['/unique-tournament/17/seasons','/unique-tournament/17/season/900/rounds',
  '/unique-tournament/17/season/900/standings/total','/unique-tournament/17/season/900/events/round/1'])
  test(`discovery permits exact first-party route ${path}`,async()=>{
    const p=page();const response=await p.execute(null,path,headers,now+120000,now+10000,spec);
    assert.equal(p.calls.length,1);assert.equal(p.calls[0].options.credentials,'same-origin');
    assert.equal(response.response.status,200);assert.equal(JSON.stringify(response).includes('observed-site-value'),false);
  });
for(const path of ['/unique-tournament/18/seasons','/unique-tournament/17/season/901/rounds',
  '/unique-tournament/17/season/900/events/round/0','/unique-tournament/17/season/900/rounds?x=1',
  '/unique-tournament/17/season/900/events/last/0','/event/123','/event/123/comments',
  'https://other.example/seasons'])
  test(`discovery rejects unleased scope ${path}`,async()=>{
    const p=page();await assert.rejects(p.execute(123,path,headers,now+120000,now+10000,spec));assert.equal(p.calls.length,0);
  });
test('ordinary collection cannot request discovery routes',async()=>{
  const p=page();await assert.rejects(p.execute(17,'/unique-tournament/17/seasons',headers,now+120000,now+10000));
  assert.equal(p.calls.length,0);
});

test('round zero requires an explicitly leased inspection',async()=>{
  const path='/unique-tournament/17/season/900/events/round/0';
  const p=page();
  await p.execute(null,path,headers,now+120000,now+10000,{...spec,inspection_only:true});
  assert.equal(p.calls.length,1);
  await assert.rejects(p.execute(null,path,headers,now+120000,now+10000,{...spec,inspection_only:'true'}));
  await assert.rejects(p.execute(null,path.replace('/900/','/901/'),headers,now+120000,now+10000,{...spec,inspection_only:true}));
  assert.equal(p.calls.length,1);
});

function setup({denial=false,pauseInFlight=false,unready=false,challenge=false,failRenewal=false,jitter=null}={}) {
  let time=now,index=0,ready=!unready;
  const calls=[],requests=[],renewals=[];
  const paths=['/unique-tournament/17/seasons','/unique-tournament/17/season/900/rounds'];
  const summary={discovery_id:'discovery-test',ready_to_collect:true,seasons:[]};
  const native={connect:async()=>({}),close:()=>{},call:async(op,args={})=>{
    calls.push([op,args]);
    if(op==='discovery_create' || op==='discovery_open')return summary;
    if(op==='discovery_next')return index<paths.length?{path:paths[index],lease:'a'.repeat(32),...spec,label:'League',
      policy:{session_request_interval_seconds:0.5,random_action_delay_max_seconds:jitter}}:{done:true,summary};
    if(op==='discovery_delay'){time+=args.milliseconds;return {};}
    if(op==='discovery_observe') {if(denial)throw new Error('HTTP 403: discovery incomplete');index++;return {summary};}
    if(op==='discovery_pause')return {};
    if(op==='discovery_prepare')return {campaign_id:'new-campaign'};
    throw new Error(`Unexpected operation ${op}`);
  }};
  const session={ready:()=>ready,refresh:async()=>{},renewalDelayMs:()=>0,beginRenewal:()=>{},requestDiscovery:async job=>{
    requests.push({path:job.path,time});if(pauseInFlight)await controller.pause();
    return {startedDateTime:new Date(time).toISOString(),response:{status:denial?403:200}};
  }};
  const browser={snapshot:async()=>({url:'https://www.sofascore.com/football/match/a-b/abc#id:123',eventId:123,challenge}),
    resetMemory:async job=>{renewals.push(job);if(failRenewal)throw new Error('Renewal failed');ready=true;}};
  const controller=new DiscoveryController({native,session,browser,now:()=>time,random:()=>0.5});
  return {controller,calls,requests,renewals};
}
test('discovery saves every response, spaces requests and reuses one ready session without navigation',async()=>{
  const {controller,calls,requests,renewals}=setup({jitter:2});
  await controller.connect({season:'25/26'});await controller.run();
  assert.equal(controller.state.phase,'discovered');assert.equal(controller.running,false);
  assert.equal(requests.length,2);assert.equal(renewals.length,0);
  assert.ok(requests[1].time-requests[0].time>=1500);
  assert.equal(calls.filter(([op])=>op==='discovery_observe').length,2);
  assert.deepEqual(await controller.prepare(),{campaign_id:'new-campaign'});
});
test('a discovery denial checkpoints once and stops without retries or campaign creation',async()=>{
  const {controller,calls,requests}=setup({denial:true});
  await controller.connect({season:'25/26'});await controller.run();
  assert.equal(requests.length,1);assert.equal(controller.state.phase,'paused');
  assert.ok(calls.some(([op])=>op==='discovery_observe'));assert.ok(calls.some(([op])=>op==='discovery_pause'));
  assert.ok(!calls.some(([op])=>op==='discovery_prepare'));
});
test('pause during an in-flight discovery saves the response before stopping',async()=>{
  const {controller,calls,requests}=setup({pauseInFlight:true});
  await controller.connect({season:'25/26'});await controller.run();
  assert.equal(requests.length,1);assert.ok(calls.find(([op])=>op==='discovery_observe'));
  assert.equal(controller.state.phase,'paused');
});
test('expired discovery session renews the same anchor once, not once per request',async()=>{
  const {controller,requests,renewals}=setup({unready:true});
  await controller.connect({discovery_id:'existing-id'});await controller.run();
  assert.equal(requests.length,2);assert.equal(renewals.length,1);assert.equal(renewals[0].event_id,123);
});
for(const option of ['challenge','failRenewal']) test(`${option} stops discovery before requests`,async()=>{
  const {controller,requests}=setup({unready:true,[option]:true});
  await controller.connect({season:'25/26'});await controller.run();assert.equal(requests.length,0);
});
test('session client accepts only leased discovery and executes in the existing inspected tab',async()=>{
  let listener,call;
  const p=page();const chrome={devtools:{inspectedWindow:{tabId:7},network:{onRequestFinished:{addListener:f=>{listener=f;}}}},
    scripting:{executeScript:async value=>{call=value;return [{result:await p.execute(...value.args)}];}}};
  const client=new ChromeSessionClient(chrome,{now:()=>now});
  listener({startedDateTime:new Date(now).toISOString(),request:{method:'GET',url:'https://www.sofascore.com/api/v1/event/123',
    headers:[{name:'X-Requested-With',value:'observed-site-value'}]},response:{status:200}});
  const job={...spec,lease:'a'.repeat(32),path:'/unique-tournament/17/seasons'};
  await client.requestDiscovery(job);assert.equal(call.target.tabId,7);assert.equal(p.calls.length,1);
  await assert.rejects(client.requestDiscovery({...job,lease:null}),/lease/);assert.equal(p.calls.length,1);
});

test('local Disconnect permits reconnection even if Chrome only notifies the remote end',async()=>{
  const ports=[];let disconnects=0;
  const runtime={connectNative:()=>{
    const port={onMessage:{addListener:f=>{port.message=f;}},onDisconnect:{addListener:f=>{port.disconnected=f;}},
      postMessage:m=>{if(m.operation==='hello')port.message({id:m.id,ok:true,result:{protocol:PROTOCOL,version:VERSION}});},
      disconnect:()=>{disconnects++;}};
    ports.push(port);return port;
  }};
  const client=new NativeClient(runtime);
  await client.connect();
  const pending=assert.rejects(client.call('discovery_status'),/closed/);
  client.close();await pending;
  assert.equal(client.pending.size,0);assert.equal(client.port,null);
  await client.connect();assert.equal(ports.length,2);
  ports[0].disconnected();assert.equal(client.port,ports[1]);
  client.close();assert.equal(disconnects,2);
});
