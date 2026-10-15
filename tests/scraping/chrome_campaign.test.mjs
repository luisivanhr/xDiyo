import assert from "node:assert/strict";
import test from "node:test";
import {CampaignController} from "../../scraping/sofascore_web_v3/devtools_extension/controller.mjs";
import {NetworkCapture} from "../../scraping/sofascore_web_v3/devtools_extension/capture.mjs";
import {NativeClient} from "../../scraping/sofascore_web_v3/devtools_extension/native.mjs";
import {ChromeMatchBrowser,pageAction} from "../../scraping/sofascore_web_v3/devtools_extension/browser.mjs";
import vm from "node:vm";

function matchPageContext({awarded=false,mainPresent=true}={}) {
  const main={},nested={},aside={},clicks=[];
  const tab=(view,list,{selected=false,hidden=false,disabled=false}={})=>({
    disabled,getClientRects:()=>hidden?[]:[{}],closest:()=>list,
    getAttribute:key=>key==="aria-controls"?`tabpanel-${view}`:key==="aria-selected"?String(selected):null,
    scrollIntoView:()=>{},click:()=>clicks.push({view,list}),
  });
  const all=tab("all",aside,{selected:true}),innerLineups=tab("lineups",nested);
  const buttons=[tab("lineups",main,{disabled:awarded}),tab("statistics",main,{selected:!awarded,hidden:awarded}),
    tab("standings",main,{selected:awarded}),tab("matches",main)];
  main.querySelectorAll=()=>[...buttons,tab("total",nested),tab("home",nested),tab("away",nested)];
  return {location:{origin:"https://www.sofascore.com",hash:"#id:12499586",href:"https://www.sofascore.com/football/match/example#id:12499586"},
    getComputedStyle:()=>({visibility:"visible"}),clicks,main,
    document:{body:{innerText:awarded?"Union Berlin 0 - 2 Bochum Awarded":"Finished"},
      querySelector:selector=>selector.includes('aria-controls="tabpanel-matches"')?(mainPresent?buttons[3]:null):
        selector.includes('aria-controls="tabpanel-lineups"')?innerLineups:all,
      querySelectorAll:()=>[all,tab("key-events",aside),...buttons,innerLineups]}};
}

test("match view evidence excludes nested standings filters",()=>{
  const context=matchPageContext({awarded:true});
  const snapshot=vm.runInNewContext(`(${pageAction.toString()})("snapshot")`,context);
  assert.deepEqual(Array.from(snapshot.views),["standings","matches"]);
  assert.equal(snapshot.eventId,12499586);
  assert.equal(snapshot.activeView,"tabpanel-standings");
});

test("commentary All cannot replace the active main Statistics tab",()=>{
  const snapshot=vm.runInNewContext(`(${pageAction.toString()})("snapshot")`,matchPageContext());
  assert.equal(snapshot.activeView,"tabpanel-statistics");
  assert.deepEqual(Array.from(snapshot.views),["lineups","statistics","standings","matches"]);
});

test("commentary alone cannot make an unrendered match navigation ready",()=>{
  const snapshot=vm.runInNewContext(`(${pageAction.toString()})("snapshot")`,matchPageContext({mainPresent:false}));
  assert.equal(snapshot.activeView,undefined);
  assert.equal(snapshot.views.length,0);
});

test("view actions select the main match navigation rather than a nested duplicate",()=>{
  const context=matchPageContext();
  const result=vm.runInNewContext(`(${pageAction.toString()})("view","lineups")`,context);
  assert.equal(result.acted,true);
  assert.deepEqual(context.clicks,[{view:"lineups",list:context.main}]);
});

test("expired page actions never click, including when scrolling exhausts their deadline",()=>{
  for (const delayDuringScroll of [false,true]) {
    let now=delayDuringScroll?0:10;
    const context=matchPageContext();
    context.Date={now:()=>now};
    context.main.querySelectorAll()[0].scrollIntoView=()=>{now=10;};
    assert.throws(()=>vm.runInNewContext(`(${pageAction.toString()})("view","lineups",5)`,context),/expired before execution/);
    assert.deepEqual(context.clicks,[]);
  }
});

test("a snapshot tolerates a document whose body is not ready yet",()=>{
  const context=matchPageContext({mainPresent:false});
  context.document.body=null;
  const snapshot=vm.runInNewContext(`(${pageAction.toString()})("snapshot")`,context);
  assert.equal(snapshot.challenge,false);
  assert.equal(snapshot.activeView,undefined);
});

function setup({ids=[123,124],code=304,wrong=false,challenge=false,missingHeatmap=false,breakAck=false,jitter=null,savedBlock=null,
                existingPage=false,heatmapCode=code,heatmapResponds=true,historicalAfterPages=null,viewWaitSeconds=.01,epochMs=0}={}) {
  const listeners=new Set(), calls=[], observations=[], actions=[], sleeps=[];
  let time=epochMs,index=0,current,pageId=existingPage?ids[0]:999,requestIndex=0,historyPages=0,activeView="tabpanel-lineups";
  const network={onRequestFinished:{addListener:f=>listeners.add(f),removeListener:f=>listeners.delete(f)},
    getHAR:cb=>cb({entries:[]}),emit:e=>{for(const f of [...listeners]) f(e);}};
  const makeEntry=(id,suffix,status=code)=>({startedDateTime:new Date(Date.UTC(2026,8,12,2,0,0,requestIndex++)).toISOString(),time:1,
    request:{method:"GET",url:`https://www.sofascore.com/api/v1/event/${id}${suffix}`},
    response:{status,bodySize:2,content:{mimeType:"application/json"}},getContent:cb=>cb("{}","")});
  const received=new Map();
  const savedDenial=savedBlock ? makeEntry(ids[0],"/heatmap/9",savedBlock) : null;
  if (savedDenial) received.set(`/api/v1/event/${ids[0]}/heatmap/9`,savedDenial);
  const coverage=()=>{
    const components=[{path:`/event/${current}`,component:"event"},{path:`/event/${current}/heatmap/9`,component:"heatmap"}].map(c=>{
      const e=received.get("/api/v1"+c.path);
      return {...c,http_status:e?.response.status,observed_at:e?.startedDateTime,
        status:[401,403].includes(e?.response.status) ? "blocked" : e?.response.status===429 ? "rate_limited" :
          e?.response.status===404 ? "unavailable" : e?.response.content.text ? "success" : "not_observed"};
    });
    return {components,complete:components.every(c=>["success","unavailable"].includes(c.status)),
      blocked:components.some(c=>["blocked","rate_limited"].includes(c.status))};
  };
  const native={connect:async()=>({version:"0.2.13"}),call:async(op,args={})=>{
    calls.push({op,...args});
    if(op==="open") return {};
    if(op==="next") {
      if(index===ids.length) return {done:true,status:{state:"complete"}};
      current=ids[index++];
      return {event_id:current,lease:`lease${current}`,wait_until:0,label:`Match ${current}`,
        paths:[`/event/${current}`,`/event/${current}/heatmap/9`],
        policy:{view_wait_seconds:viewWaitSeconds,match_timeout_seconds:90,max_api_observations_per_match:120,random_action_delay_max_seconds:jitter},coverage:coverage()};
    }
    if(op==="observe") {
      if(breakAck) throw new Error("disk full");
      observations.push(args.entry);
      const path=new URL(args.entry.request.url).pathname,previous=received.get(path);
      if (args.entry._xdiyoSelected && (!previous || Date.parse(args.entry.startedDateTime)>=Date.parse(previous.startedDateTime))) received.set(path,args.entry);
      return {ack:"saved",pause_requested:false};
    }
    if(op==="progress") return coverage();
    if(op==="finish") {
      assert.equal(coverage().blocked,false);
      assert.equal(args.page_event_id,current);
      assert.equal(received.get(`/api/v1/event/${current}`).response.content.text,"{}");
      return {state:coverage().complete ? "complete" : "needs_attention",pause_requested:!coverage().complete};
    }
    if(op==="pause") return {paused:true};
    throw new Error("Unexpected native operation");
  }};
  const browser={snapshot:async()=>({eventId:pageId,challenge,activeView}),navigate:async(job,{refreshExisting=false}={})=>{
    if(pageId===job.event_id && !refreshExisting) return {existing:true};
    actions.push([pageId===job.event_id&&refreshExisting?"refresh":"navigate",job.event_id]);
    pageId=wrong||historicalAfterPages!==null?999:job.event_id;
    network.emit(makeEntry(job.event_id,""));
    if(!missingHeatmap) network.emit(makeEntry(job.event_id,"/heatmap/9",heatmapCode));
    return {existing:false};
  },action:async(command,value)=>{
    actions.push([command,value]);
    if(command==="view") activeView=`tabpanel-${value}`;
    if(command==="heatmap" && heatmapResponds) network.emit(makeEntry(current,"/heatmap/9",heatmapCode));
    if(command==="matchLink" && historicalAfterPages!==null) {
      if(historyPages<historicalAfterPages) return {acted:false};
      pageId=current; return {acted:true};
    }
    if(command==="h2hNext") historyPages++;
    return {acted:!wrong};
  }};
  const controller=new CampaignController({network,native,browser,now:()=>time,random:()=>.5,
    sleep:async ms=>{sleeps.push(ms);time+=ms;}});
  return {controller,network,native,browser,actions,calls,observations,listeners,sleeps,makeEntry,savedDenial};
}
async function run(env) { await env.controller.connect("pilot"); await env.controller.run(); }

test("periodic resets occur only between saved matches and recording starts before return",async()=>{
  const env=setup({ids:[123,124,125,126,127]});
  const call=env.native.call;
  env.native.call=async(op,args)=>{
    const result=await call(op,args);
    if(op==="next" && !result.done) result.policy.memory_reset_every_matches=2;
    return result;
  };
  const resets=[];
  env.browser.resetMemory=async(job,startRecording)=>{
    resets.push(job.event_id);
    assert.equal(env.listeners.size,0);
    assert.equal(env.calls.filter(c=>c.op==="finish").length,resets.length*2);
    startRecording(); assert.equal(env.listeners.size,1);
    return {...await env.browser.navigate(job),memoryReset:true};
  };
  await run(env);
  assert.equal(env.controller.state.phase,"finished");
  assert.deepEqual(resets,[125,127]);
  assert.equal(env.controller.memoryResets,2);
  assert.equal(env.calls.filter(c=>c.op==="finish" && c.page_evidence.memory_reset_before_match).length,2);
});

test("a failed reset pauses before any next match response is claimed",async()=>{
  const env=setup({ids:[123,124]});
  const call=env.native.call;
  env.native.call=async(op,args)=>{
    const result=await call(op,args);
    if(op==="next" && !result.done) result.policy.memory_reset_every_matches=1;
    return result;
  };
  env.browser.resetMemory=async()=>{ throw new Error("Memory reset timed out"); };
  await run(env);
  assert.equal(env.controller.state.phase,"paused");
  assert.equal(env.calls.filter(c=>c.op==="finish").length,1);
  assert.deepEqual(env.actions,[["navigate",123]]);
  assert.equal(env.listeners.size,0);
  assert.equal(env.controller.completedSinceMemoryReset,1);
});

test("Start / resume preserves the reset counter in the connected DevTools session",async()=>{
  const env=setup({ids:[123,124,125]});
  const call=env.native.call;
  env.native.call=async(op,args)=>{
    const result=await call(op,args);
    if(op==="next" && !result.done) result.policy.memory_reset_every_matches=2;
    if(op==="finish" && args.event_id===124) await env.controller.pause();
    return result;
  };
  let resets=0;
  env.browser.resetMemory=async(job,startRecording)=>{
    resets++; startRecording(); return {...await env.browser.navigate(job),memoryReset:true};
  };
  await run(env);
  assert.equal(env.controller.state.phase,"paused");
  assert.equal(env.controller.completedSinceMemoryReset,2);
  assert.equal(resets,0);
  await env.controller.run();
  assert.equal(env.controller.state.phase,"finished");
  assert.equal(resets,1);
});

function resetBrowser({challenge=false,ready=true,hang=false,pause=false,badSender=false,badNonce=false,badTab=false}={}) {
  const actions=[],listeners=new Set(); let stopped=false;
  const chrome={devtools:{inspectedWindow:{tabId:7}},runtime:{id:"test",getURL:path=>`chrome-extension://test/${path}`,
    onMessage:{addListener:fn=>listeners.add(fn),removeListener:fn=>listeners.delete(fn)}},tabs:{
    update:async(tabId,options)=>{
      assert.equal(tabId,7); actions.push(options.url);
      if(hang) return new Promise(()=>{});
      if(ready && options.url.startsWith("chrome-extension:")) for(const listener of listeners) listener({
        type:"xdiyo_memory_reset_ready",tabId:badTab?8:7,nonce:badNonce?"stale":new URL(options.url).hash.slice(1)
      },{id:badSender?"other":"test",url:options.url},()=>{});
    }
  }};
  const browser=new ChromeMatchBrowser(chrome,{actionTimeoutMs:15,
    sleep:async()=>{if(pause) stopped=true;},check:()=>{if(stopped) throw new Error("Paused by user");}});
  browser.snapshot=async()=>({challenge});
  const job={event_id:123,url:"https://www.sofascore.com/football/match/a-b/x#id:123"};
  return {browser,job,actions,listeners};
}

test("memory renewal uses the same tab and records before loading the next fixture",async()=>{
  const {browser,job,actions,listeners}=resetBrowser();
  const result=await browser.resetMemory(job,()=>actions.push("recording"));
  assert.match(actions[0],/^chrome-extension:\/\/test\/memory-reset.html#[\w-]+$/);
  assert.deepEqual(actions.slice(1),["recording",job.url]);
  assert.equal(listeners.size,0);
  assert.equal(result.memoryReset,true);
});

test("memory renewal does not navigate away from a challenge or accept another origin",async()=>{
  for(const invalid of [false,true]) {
    const {browser,job,actions}=resetBrowser({challenge:!invalid});
    if(invalid) job.url="https://example.org/football/match/a#id:123";
    await assert.rejects(browser.resetMemory(job,()=>assert.fail()),invalid?/Invalid queued/:/Human challenge/);
    assert.deepEqual(actions,[]);
  }
});

test("failed, hung, or canceled renewal never returns to the website",async()=>{
  for(const options of [{ready:false},{hang:true},{pause:true},{badSender:true},{badNonce:true},{badTab:true}]) {
    const {browser,job,actions,listeners}=resetBrowser(options);
    await assert.rejects(browser.resetMemory(job,()=>assert.fail()),/did not load|timed out|Paused by user/);
    assert.equal(actions.length,1);
    assert.match(actions[0],/^chrome-extension:\/\/test\/memory-reset.html#/);
    assert.equal(listeners.size,0);
  }
});

test("one tab is reused; cached bodies are acknowledged before advancing; no redundant view actions",async()=>{
  const env=setup(); await run(env);
  assert.equal(env.controller.state.phase,"finished");
  assert.deepEqual(env.actions,[["navigate",123],["navigate",124]]);
  assert.equal(env.observations.length,4);
  assert.equal(env.calls.filter(c=>c.op==="finish").length,2);
  assert.equal(env.listeners.size,0);
});
test("a missing away heatmap opens statistics then only the missing team",async()=>{
  const env=setup({ids:[123],missingHeatmap:true,jitter:2}); await run(env);
  assert.deepEqual(env.actions,[["navigate",123],["view","statistics"],["heatmap",9]]);
  assert.ok(env.sleeps.includes(1000));
  assert.equal(env.controller.state.phase,"finished");
});

test("data loaded before recording is recovered by one same-tab refresh with capture active",async()=>{
  const env=setup({ids:[123],missingHeatmap:true,heatmapResponds:false});
  const navigate=env.browser.navigate;
  env.browser.navigate=async(job,options)=>{
    const result=await navigate(job,options);
    if(options?.refreshExisting) {
      assert.equal(env.listeners.size,1);
      env.network.emit(env.makeEntry(123,"/heatmap/9"));
    }
    return result;
  };
  await run(env);
  assert.equal(env.controller.state.phase,"finished");
  assert.equal(env.actions.filter(([action])=>action==="refresh").length,1);
  assert.equal(env.calls.filter(call=>call.op==="finish").length,1);
  assert.equal(env.calls.find(call=>call.op==="finish").page_evidence.recovery_refreshed,true);
});

test("persistently missing data gets only one refresh and remains needs_attention",async()=>{
  const env=setup({ids:[123],missingHeatmap:true,heatmapResponds:false});
  await run(env);
  assert.equal(env.controller.state.phase,"paused");
  assert.match(env.controller.state.message,/Match needs attention/);
  assert.equal(env.controller.state.lastResult.state,"needs_attention");
  assert.equal(env.actions.filter(([action])=>action==="refresh").length,1);
  assert.equal(env.calls.filter(call=>call.op==="next").length,1);
});

test("a recorded 404 is unavailable data and does not trigger a recovery refresh",async()=>{
  const env=setup({ids:[123],heatmapCode:404});
  await run(env);
  assert.equal(env.controller.state.phase,"finished");
  assert.deepEqual(env.actions,[["navigate",123]]);
});

test("the recovery refresh shares the original match deadline",async()=>{
  const env=setup({ids:[123],missingHeatmap:true,heatmapResponds:false});
  const navigate=env.browser.navigate;
  env.browser.navigate=async(job,options)=>{
    const result=await navigate(job,options);
    if(options?.refreshExisting) await env.controller.sleep(90001);
    return result;
  };
  await run(env);
  assert.match(env.controller.state.message,/Match time budget/);
  assert.equal(env.calls.filter(call=>call.op==="finish").length,0);
  assert.equal(env.actions.filter(([action])=>action==="refresh").length,1);
});

for(const code of [401,403,429]) test(`a fresh ${code} on the recovery refresh pauses immediately`,async()=>{
  const env=setup({ids:[123],missingHeatmap:true,heatmapResponds:false});
  const navigate=env.browser.navigate;
  env.browser.navigate=async(job,options)=>{
    const result=await navigate(job,options);
    if(options?.refreshExisting) env.network.emit(env.makeEntry(123,"/heatmap/9",code));
    return result;
  };
  await run(env);
  assert.match(env.controller.state.message,new RegExp(`Sofascore HTTP ${code}`));
  assert.equal(env.calls.filter(call=>call.op==="finish").length,0);
  assert.equal(env.actions.filter(([action])=>action==="refresh").length,1);
});
for (const code of [401,403,429]) test(`HTTP ${code} stops actions, persists metadata and pauses`,async()=>{
  const env=setup({code}); await run(env);
  assert.equal(env.controller.state.phase,"paused");
  assert.equal(env.calls.filter(c=>c.op==="finish").length,0);
  assert.ok(env.calls.some(c=>c.op==="pause"));
  assert.equal(env.observations[0].response.status,code);
  assert.equal(env.actions.length,1);
});

for (const code of [401,403,429]) test(`unselected HTTP ${code} remains diagnostic and does not stop the queue`,async()=>{
  const env=setup({ids:[123]});
  const navigate=env.browser.navigate;
  env.browser.navigate=async(...args)=>{
    for(const [id,path] of [[16434040,"/ai-insights/en"],[123,"/ai-insights/en"],[123,"/comments"],
                           [123,"/lineups"],[124,"/heatmap/9"],[123,"/heatmap/9?period=1"]]) {
      const denied=env.makeEntry(id,path,code);
      denied.getContent=()=>assert.fail("Do not read unselected bodies");
      env.network.emit(denied);
    }
    return navigate(...args);
  };
  await run(env);
  assert.equal(env.controller.state.phase,"finished");
  const diagnostics=env.observations.filter(e=>!e._xdiyoSelected);
  assert.equal(diagnostics.length,6);
  assert.ok(diagnostics.every(e=>e.response.status===code && !e.response.content.text));
  assert.equal(env.calls.filter(c=>c.op==="finish").length,1);
});

test("unrelated traffic cannot consume the selected-match observation limit",async()=>{
  const env=setup({ids:[123]});
  const navigate=env.browser.navigate;
  env.browser.navigate=async(...args)=>{
    for(let i=0;i<150;i++) env.network.emit(env.makeEntry(16434040,"/ai-insights/en",403));
    return navigate(...args);
  };
  await run(env);
  assert.equal(env.controller.state.phase,"finished");
  assert.equal(env.observations.filter(e=>!e._xdiyoSelected).length,150);
  assert.equal(env.observations.filter(e=>e._xdiyoSelected).length,2);
});

test("the selected-match observation limit still stops the recorder",async()=>{
  const env=setup({ids:[123]});
  env.browser.navigate=async()=>{
    for(let i=0;i<120;i++) env.network.emit(env.makeEntry(123,""));
    return {existing:false};
  };
  await run(env);
  assert.match(env.controller.state.message,/Selected-match observation limit/);
  assert.equal(env.calls.filter(c=>c.op==="finish").length,0);
});

test("page waits finish early after the matching response is saved",async()=>{
  const env=setup({ids:[123],viewWaitSeconds:8}); await run(env);
  assert.equal(env.controller.state.phase,"finished");
  assert.equal(env.sleeps.reduce((a,b)=>a+b,0),1000);
});

test("a missing team retains the section window and both configured action delays",async()=>{
  const env=setup({ids:[123],missingHeatmap:true,viewWaitSeconds:8,jitter:4}); await run(env);
  assert.equal(env.controller.state.phase,"finished");
  // Navigation and the successful heatmap settle once. The statistics section
  // keeps its eight-second window; both UI actions keep two-second jitter.
  assert.equal(env.sleeps.reduce((a,b)=>a+b,0),14000);
});

test("a still-missing heatmap gets its whole observation window before pausing",async()=>{
  const env=setup({ids:[123],savedBlock:403,missingHeatmap:true,heatmapResponds:false,viewWaitSeconds:8}); await run(env);
  assert.match(env.controller.state.message,/Saved Sofascore HTTP 403/);
  assert.equal(env.sleeps.reduce((a,b)=>a+b,0),17000);
  assert.equal(env.calls.filter(c=>c.op==="finish").length,0);
});

test("event data alone cannot end the page wait before the match tabs render",async()=>{
  const env=setup({ids:[123],viewWaitSeconds:8});
  const snapshot=env.browser.snapshot;
  env.browser.snapshot=async()=>({...await snapshot(),activeView:env.sleeps.reduce((a,b)=>a+b,0)<3000?null:"tabpanel-lineups"});
  await run(env);
  assert.equal(env.controller.state.phase,"finished");
  assert.equal(env.sleeps.reduce((a,b)=>a+b,0),3000);
});

test("statistics can finish once its delayed heatmap arrives without a redundant team click",async()=>{
  const env=setup({ids:[123],viewWaitSeconds:8,missingHeatmap:true});
  const sleep=env.controller.sleep;
  let delivered=false;
  env.controller.sleep=async ms=>{
    await sleep(ms);
    if(!delivered && env.actions.some(([command,value])=>command==="view"&&value==="statistics") && env.sleeps.reduce((a,b)=>a+b,0)>=4000) {
      delivered=true;env.network.emit(env.makeEntry(123,"/heatmap/9"));
    }
  };
  await run(env);
  assert.equal(env.controller.state.phase,"finished");
  assert.deepEqual(env.actions,[["navigate",123],["view","statistics"]]);
  assert.equal(env.sleeps.reduce((a,b)=>a+b,0),4000);
});
test("a wrong displayed fixture never finishes, despite valid matching Network bodies",async()=>{
  const env=setup({wrong:true}); await run(env);
  assert.equal(env.controller.state.phase,"paused");
  assert.match(env.controller.state.message,/Wrong match/);
  assert.equal(env.calls.filter(c=>c.op==="finish").length,0);
});
test("human challenge pauses without attempting any solution",async()=>{
  const env=setup({challenge:true}); await run(env);
  assert.match(env.controller.state.message,/Human challenge/);
  assert.equal(env.calls.filter(c=>c.op==="finish").length,0);
});

test("resume reads a recovered match's Network log before applying the saved 403",async()=>{
  const env=setup({ids:[123],savedBlock:403});
  env.browser.snapshot=async()=>({eventId:123,challenge:false});
  env.browser.navigate=async()=>({existing:true});
  const oldBlock=env.makeEntry(123,"/heatmap/9",403);
  const recovered=env.makeEntry(123,"/heatmap/9",304);
  recovered.startedDateTime="2026-09-12T02:00:01Z";
  env.network.getHAR=cb=>cb({entries:[oldBlock,env.makeEntry(123,""),recovered]});
  await run(env);
  assert.equal(env.controller.state.phase,"finished");
  assert.equal(env.actions.length,0);
  assert.equal(env.observations.length,2);
  assert.ok(env.observations.every(e=>e.response.status===304));
});

for (const code of [403,429]) test(`resume automatically opens the queued match and missing heatmap after a saved ${code}`,async()=>{
  const env=setup({ids:[123],savedBlock:code,missingHeatmap:true}); await run(env);
  assert.equal(env.controller.state.phase,"finished");
  assert.deepEqual(env.actions,[["navigate",123],["view","statistics"],["heatmap",9]]);
  assert.equal(env.calls.filter(c=>c.op==="finish").length,1);
});

test("recovery searches visible history when the queued URL opens a newer match",async()=>{
  const env=setup({ids:[123],savedBlock:403,missingHeatmap:true,historicalAfterPages:2}); await run(env);
  assert.equal(env.controller.state.phase,"finished");
  assert.deepEqual(env.actions,[["navigate",123],["view","matches"],["matchLink",123],["h2hNext",undefined],
    ["matchLink",123],["h2hNext",undefined],["matchLink",123],["view","statistics"],["heatmap",9]]);
});

function samePairResume({found=true,linkCode=304,pageURL="https://www.sofascore.com/football/match/a-b/x#id:999"}={}) {
  const env=setup({ids:[123],savedBlock:403,missingHeatmap:true,historicalAfterPages:found?0:99});
  const snapshot=env.browser.snapshot,call=env.native.call,action=env.browser.action;
  env.browser.snapshot=async()=>({...await snapshot(),url:pageURL,views:["lineups","statistics","matches"]});
  env.native.call=async(op,args)=>{
    const result=await call(op,args);
    if(op==="next" && !result.done) result.url="https://www.sofascore.com/football/match/a-b/x#id:123";
    return result;
  };
  env.browser.action=async(command,value)=>{
    const result=await action(command,value);
    if(command==="matchLink" && result.acted) env.network.emit(env.makeEntry(123,"",linkCode));
    return result;
  };
  return env;
}

test("resume follows an already loaded same-pair H2H link without another document navigation",async()=>{
  const env=samePairResume(); await run(env);
  assert.equal(env.controller.state.phase,"finished");
  assert.deepEqual(env.actions,[["view","matches"],["matchLink",123],["view","statistics"],["heatmap",9]]);
  assert.equal(env.calls.find(c=>c.op==="finish").page_evidence.historical_link_recovery,true);
});

for (const pageURL of ["https://www.sofascore.com/football/match/c-d/y#id:999",
                       "https://example.org/football/match/a-b/x#id:999"]) {
  test(`a resumed history shortcut requires the queued same-pair Sofascore route: ${pageURL}`,async()=>{
    const env=samePairResume({pageURL}); await run(env);
    assert.deepEqual(env.actions[0],["navigate",123]);
    assert.equal(env.calls.find(c=>c.op==="finish").page_evidence.historical_link_recovery,false);
  });
}

test("a fresh denial from the resumed H2H link stops before fallback navigation or publication",async()=>{
  const env=samePairResume({linkCode:403}); await run(env);
  assert.equal(env.controller.state.phase,"paused");
  assert.match(env.controller.state.message,/Sofascore HTTP 403/);
  assert.deepEqual(env.actions,[["view","matches"],["matchLink",123]]);
  assert.equal(env.calls.filter(c=>c.op==="finish").length,0);
});

test("an absent same-pair history link gets one bounded search and one normal navigation",async()=>{
  const env=samePairResume({found:false}); await run(env);
  assert.equal(env.controller.state.phase,"paused");
  assert.match(env.controller.state.message,/automatic history search could not find/);
  assert.equal(env.actions.filter(([action])=>action==="matchLink").length,6);
  assert.equal(env.actions.filter(([action])=>action==="h2hNext").length,5);
  assert.equal(env.actions.filter(([action])=>action==="navigate").length,1);
  assert.equal(env.calls.filter(c=>c.op==="finish").length,0);
});

test("a missing historical fixture stops after the bounded history search",async()=>{
  const env=setup({ids:[123],savedBlock:403,missingHeatmap:true,historicalAfterPages:99}); await run(env);
  assert.match(env.controller.state.message,/automatic history search could not find queued match 123/);
  assert.equal(env.actions.filter(([action])=>action==="matchLink").length,6);
  assert.equal(env.actions.filter(([action])=>action==="navigate").length,1);
  assert.equal(env.calls.filter(c=>c.op==="finish").length,0);
});

test("an unchanged saved denial stops after one view pass without looping or publishing",async()=>{
  const env=setup({ids:[123],savedBlock:403,missingHeatmap:true,heatmapResponds:false}); await run(env);
  assert.equal(env.controller.state.phase,"paused");
  assert.match(env.controller.state.message,/Saved Sofascore HTTP 403/);
  assert.match(env.controller.state.message,/\/event\/123\/heatmap\/9/);
  assert.deepEqual(env.actions,[["navigate",123],["view","statistics"],["heatmap",9]]);
  assert.equal(env.calls.filter(c=>c.op==="finish").length,0);
});

test("a known denial in the existing Network log allows one refresh of the same match",async()=>{
  const env=setup({ids:[123],savedBlock:403,existingPage:true,missingHeatmap:true});
  env.network.getHAR=cb=>cb({entries:[env.savedDenial]});
  await run(env);
  assert.equal(env.controller.state.phase,"finished");
  assert.deepEqual(env.actions,[["refresh",123],["view","statistics"],["heatmap",9]]);
  assert.equal(env.observations[0]._observationSource,"network_log");
  assert.equal(env.observations[0].response.status,403);
});

for (const code of [401,403,429]) test(`a fresh HTTP ${code} during recovery stops immediately`,async()=>{
  const env=setup({ids:[123],savedBlock:403,missingHeatmap:true,heatmapCode:code}); await run(env);
  assert.equal(env.controller.state.phase,"paused");
  assert.match(env.controller.state.message,new RegExp(`Sofascore HTTP ${code}`));
  assert.deepEqual(env.actions,[["navigate",123],["view","statistics"],["heatmap",9]]);
  assert.equal(env.calls.filter(c=>c.op==="finish").length,0);
  assert.equal(env.observations.at(-1)._observationSource,"live_callback");
});

test("an earlier unsaved Network-log denial allows the requested bounded resume",async()=>{
  const env=setup({ids:[123],savedBlock:403,existingPage:true,epochMs:Date.UTC(2026,8,12,3)});
  const oldUnsaved=env.makeEntry(123,"",403);
  env.network.getHAR=cb=>cb({entries:[env.savedDenial,oldUnsaved]});
  await run(env);
  assert.equal(env.controller.state.phase,"finished");
  assert.deepEqual(env.actions,[["refresh",123]]);
  assert.equal(env.calls.filter(c=>c.op==="finish").length,1);
  assert.equal(env.observations.filter(e=>e.response.status===403).length,2);
});

for (const duration of [undefined,NaN,-1,2]) test(`an unsaved denial with uncertain or unfinished timing still pauses (${duration})`,async()=>{
  const cutoff=Date.UTC(2026,8,12,3);
  const env=setup({ids:[123],savedBlock:403,existingPage:true,epochMs:cutoff});
  const denial=env.makeEntry(123,"",403);
  denial.startedDateTime=new Date(cutoff-1).toISOString();
  denial.time=duration;
  env.network.getHAR=cb=>cb({entries:[denial]});
  await run(env);
  assert.match(env.controller.state.message,/Sofascore HTTP 403/);
  assert.deepEqual(env.actions,[]);
  assert.equal(env.calls.filter(c=>c.op==="finish").length,0);
});

test("a live callback remains a stop even with a timestamp before the resume cutoff",async()=>{
  const env=setup({ids:[123],epochMs:Date.UTC(2026,8,12,3),code:403});
  await run(env);
  assert.match(env.controller.state.message,/Sofascore HTTP 403/);
  assert.equal(env.calls.filter(c=>c.op==="finish").length,0);
  assert.equal(env.observations[0]._observationSource,"live_callback");
});

test("the old-log exemption does not carry to the next automatically queued match",async()=>{
  const env=setup({ids:[123,124],epochMs:Date.UTC(2026,8,12,3)});
  const snapshot=env.browser.snapshot;
  env.browser.snapshot=async()=>env.calls.filter(c=>c.op==="next").length===2 ?
    {eventId:124,challenge:false,activeView:"tabpanel-lineups"} : snapshot();
  env.network.getHAR=cb=>cb({entries:[env.makeEntry(124,"",403)]});
  await run(env);
  assert.match(env.controller.state.message,/Sofascore HTTP 403/);
  assert.equal(env.calls.filter(c=>c.op==="finish").length,1);
  assert.deepEqual(env.actions,[["navigate",123]]);
});

test("an older unsaved denial still prevents publication if the refresh does not replace it",async()=>{
  const env=setup({ids:[123],existingPage:true,epochMs:Date.UTC(2026,8,12,3),missingHeatmap:true,heatmapResponds:false});
  env.network.getHAR=cb=>cb({entries:[env.makeEntry(123,"/heatmap/9",403)]});
  await run(env);
  assert.match(env.controller.state.message,/Saved Sofascore HTTP 403/);
  assert.equal(env.calls.filter(c=>c.op==="finish").length,0);
  assert.equal(env.actions.filter(([action])=>action==="refresh").length,1);
});

test("a new denial found in the Network log is not mistaken for the saved failure",async()=>{
  const env=setup({ids:[123],savedBlock:403,existingPage:true});
  env.network.getHAR=cb=>cb({entries:[env.makeEntry(123,"/heatmap/9",403)]});
  await run(env);
  assert.match(env.controller.state.message,/Sofascore HTTP 403/);
  assert.equal(env.actions.length,0);
  assert.equal(env.calls.filter(c=>c.op==="finish").length,0);
});

test("a challenge prevents navigation even while recovering a saved denial",async()=>{
  const env=setup({ids:[123],savedBlock:403,challenge:true}); await run(env);
  assert.match(env.controller.state.message,/Human challenge/);
  assert.equal(env.actions.length,0);
});

for (const [flag,reason] of [["pause_requested",/User pause requested/],["run_limit_reached",/Cumulative observation limit/]]) {
  test(`${flag} has a distinct message and still stops during the pre-capture check`,async()=>{
    const env=setup({ids:[123],savedBlock:403});
    const call=env.native.call;
    env.native.call=async(op,args)=>({...await call(op,args),...(op==="progress" ? {[flag]:true} : {})});
    await run(env);
    assert.match(env.controller.state.message,reason);
    assert.equal(env.actions.length,0);
    assert.equal(env.listeners.size,0);
  });
}
test("unacknowledged writes cannot advance the queue",async()=>{
  const env=setup({breakAck:true}); await run(env);
  assert.match(env.controller.state.message,/disk full/);
  assert.equal(env.calls.filter(c=>c.op==="finish").length,0);
});
test("disabled features do not read their body even when the site loads them",async()=>{
  const env=setup();
  const capture=new NetworkCapture(env.network,{eventId:123,allowedPaths:["/event/123"]});capture.start();
  const denied=env.makeEntry(123,"/lineups"); denied.getContent=()=>assert.fail("Disabled body must not be read");
  env.network.emit(denied); const har=await capture.exportHAR();
  assert.equal(har.log.entries[0]._xdiyoSelected,false);
});
test("browser adapter skips navigation for an already displayed match",async()=>{
  const commands=[],updates=[];
  const chrome={devtools:{inspectedWindow:{tabId:42}},scripting:{executeScript:async options=>{
    commands.push(options);return [{result:{eventId:123,url:"https://www.sofascore.com/football/match/a/b#id:123"}}];
  }},tabs:{update:async(...args)=>updates.push(args)}};
  const browser=new ChromeMatchBrowser(chrome);
  assert.deepEqual(await browser.navigate({event_id:123}),{existing:true});
  assert.equal(updates.length,0);
  assert.ok(commands.every(c=>c.target.tabId===42));
});
test("browser adapter refreshes only the inspected tab when recovery requests it",async()=>{
  const reloaded=[],updated=[];
  const chrome={devtools:{inspectedWindow:{tabId:42}},scripting:{executeScript:async()=>[{result:{eventId:123}}]},
    tabs:{reload:async id=>reloaded.push(id),update:async(...args)=>updated.push(args)}};
  const browser=new ChromeMatchBrowser(chrome);
  assert.deepEqual(await browser.navigate({event_id:123},{refreshExisting:true}),{existing:false,refreshed:true});
  assert.deepEqual(reloaded,[42]);
  assert.equal(updated.length,0);
});

test("queued fallback replaces history in the inspected tab and keeps an explicit rollback mode",async()=>{
  const job={event_id:124,url:"https://www.sofascore.com/football/match/a/b#id:124"};
  for(const navigationMode of ["replace","update"]) {
    const commands=[],updates=[];
    const chrome={devtools:{inspectedWindow:{tabId:42}},scripting:{executeScript:async options=>{
      commands.push(options);
      return [{result:options.args[0]==="snapshot"?{eventId:123}:options.args[0]==="matchLink"?{acted:false}:{acted:true}}];
    }},tabs:{update:async(...args)=>updates.push(args)}};
    const browser=new ChromeMatchBrowser(chrome,{navigationMode});
    assert.deepEqual(await browser.navigate(job),{existing:false,navigationMode});
    assert.ok(commands.every(c=>c.target.tabId===42));
    if(navigationMode==="replace") {
      assert.deepEqual(commands.map(c=>c.args[0]),["snapshot","matchLink","navigate"]);
      assert.deepEqual(commands.at(-1).args[1],{url:job.url,eventId:124});
      assert.equal(updates.length,0);
    } else {
      assert.deepEqual(commands.map(c=>c.args[0]),["snapshot","matchLink"]);
      assert.deepEqual(updates,[[42,{url:job.url}]]);
    }
  }
});

test("a visible match link is preferred over history replacement",async()=>{
  const commands=[];
  const chrome={devtools:{inspectedWindow:{tabId:42}},scripting:{executeScript:async options=>{
    commands.push(options.args[0]);
    return [{result:options.args[0]==="snapshot"?{eventId:123}:{acted:true}}];
  }}};
  assert.deepEqual(await new ChromeMatchBrowser(chrome).navigate({event_id:124}),{existing:false});
  assert.deepEqual(commands,["snapshot","matchLink"]);
});

test("history replacement validates the queued identity, challenge, and late-action deadline",()=>{
  const good={url:"https://www.sofascore.com/football/match/a/b#id:124",eventId:124};
  for(const condition of ["valid","wrong_id","foreign_origin","non_match","challenge","expired"]) {
    const context=matchPageContext(), navigations=[];
    context.URL=URL;
    context.location.replace=url=>navigations.push(url);
    const value={...good};
    if(condition==="wrong_id") value.eventId=125;
    if(condition==="foreign_origin") value.url="https://example.com/football/match/a/b#id:124";
    if(condition==="non_match") value.url="https://www.sofascore.com/api/v1/event/124#id:124";
    if(condition==="challenge") context.document.body.innerText="Verify you are human";
    const code=`(${pageAction.toString()})("navigate",${JSON.stringify(value)},${condition==="expired"?0:"null"})`;
    if(condition==="valid") {
      assert.equal(vm.runInNewContext(code,context).acted,true);
      assert.deepEqual(navigations,[good.url]);
    } else {
      assert.throws(()=>vm.runInNewContext(code,context),/Invalid queued match URL|Human challenge|expired/);
      assert.equal(navigations.length,0);
    }
  }
});

for(const outcome of ["target","wrong","challenge"]) test(`uncertain history replacement is never replayed: ${outcome}`,async()=>{
  let navigated=false;
  const commands=[];
  const chrome={devtools:{inspectedWindow:{tabId:42}},scripting:{executeScript:async options=>{
    const command=options.args[0]; commands.push(command);
    if(command==="matchLink") return [{result:{acted:false}}];
    if(command==="navigate") {navigated=true; throw new Error("Frame with ID 0 was removed.");}
    return [{result:{eventId:navigated?(outcome==="wrong"?125:124):123,challenge:navigated&&outcome==="challenge"}}];
  }},tabs:{update:async()=>assert.fail("Do not replay uncertain navigation")}};
  const browser=new ChromeMatchBrowser(chrome);
  const job={event_id:124,url:"https://www.sofascore.com/football/match/a/b#id:124"};
  if(outcome==="target") assert.deepEqual(await browser.navigate(job),{existing:false,navigationMode:"replace",verifiedAfterFrameChange:true});
  else await assert.rejects(()=>browser.navigate(job),outcome==="challenge"?/Human challenge/:/Chrome navigate: Frame/);
  assert.deepEqual(commands,["snapshot","matchLink","navigate","snapshot"]);
});

test("read-only snapshots wait for transient frame replacement without navigating",async()=>{
  const commands=[],sleeps=[];
  const chrome={devtools:{inspectedWindow:{tabId:42}},scripting:{executeScript:async options=>{
    commands.push(options.args[0]);
    if(commands.length<3) throw new Error("Frame with ID 0 was removed.");
    return [{result:{eventId:123}}];
  }}};
  const browser=new ChromeMatchBrowser(chrome,{sleep:async ms=>sleeps.push(ms)});
  const page=await browser.snapshot();
  assert.equal(page.eventId,123);
  assert.equal(page.navigationFrameWaits,2);
  assert.deepEqual(commands,["snapshot","snapshot","snapshot"]);
  assert.deepEqual(sleeps,[500,500]);
});

test("a persistent missing frame stops after five seconds of bounded DOM reads",async()=>{
  let reads=0; const sleeps=[];
  const chrome={devtools:{inspectedWindow:{tabId:42}},scripting:{executeScript:async()=>{
    reads++; throw new Error("Frame with ID 0 was removed.");
  }}};
  const browser=new ChromeMatchBrowser(chrome,{sleep:async ms=>sleeps.push(ms)});
  await assert.rejects(()=>browser.snapshot(),/Chrome snapshot: Frame with ID 0 was removed/);
  assert.equal(reads,11);
  assert.equal(sleeps.reduce((a,b)=>a+b,0),5000);
});

test("an indefinitely paused page read times out and a late result cannot resume the campaign",async()=>{
  let resolveRead; const env=setup({ids:[123]});
  const chrome={devtools:{inspectedWindow:{tabId:42}},scripting:{executeScript:()=>new Promise(resolve=>{resolveRead=resolve;})}};
  const browser=new ChromeMatchBrowser(chrome,{readTimeoutMs:5,check:()=>env.controller.check()});
  env.browser.snapshot=()=>browser.snapshot();
  await run(env);
  assert.equal(env.controller.state.phase,"paused");
  assert.match(env.controller.state.message,/Chrome snapshot: Page read timed out/);
  assert.equal(env.calls.filter(c=>c.op==="finish").length,0);
  assert.equal(env.calls.filter(c=>c.op==="pause").length,1);
  resolveRead([{result:{eventId:123}}]);
  await Promise.resolve();
  assert.equal(env.actions.length,0);
  assert.equal(env.controller.state.phase,"paused");
});

test("a stalled view action pauses and its delayed script cannot click or advance the queue",async()=>{
  let injected,resolveAction;
  const env=setup({ids:[123],missingHeatmap:true});
  const chrome={devtools:{inspectedWindow:{tabId:42}},scripting:{executeScript:options=>{
    injected=options; return new Promise(resolve=>{resolveAction=resolve;});
  }}};
  const browser=new ChromeMatchBrowser(chrome,{actionTimeoutMs:5,check:()=>env.controller.check()});
  env.browser.action=(command,value)=>browser.action(command,value);
  await run(env);
  assert.equal(env.controller.state.phase,"paused");
  assert.match(env.controller.state.message,/Chrome view: Page action timed out/);
  assert.equal(env.calls.filter(c=>c.op==="finish").length,0);
  assert.equal(env.calls.filter(c=>c.op==="pause").length,1);
  const context=matchPageContext();
  assert.throws(()=>vm.runInNewContext(`(${injected.func.toString()})(...${JSON.stringify(injected.args)})`,context),/expired before execution/);
  assert.deepEqual(context.clicks,[]);
  resolveAction([{result:{acted:true}}]);
  await Promise.resolve();
  assert.equal(env.controller.state.phase,"paused");
  assert.equal(env.calls.filter(c=>c.op==="finish").length,0);
});

test("ordinary browser failures are not retried as frame replacement",async()=>{
  let reads=0;
  const chrome={devtools:{inspectedWindow:{tabId:42}},scripting:{executeScript:async()=>{
    reads++; throw new Error("Cannot access contents of url");
  }}};
  const browser=new ChromeMatchBrowser(chrome,{sleep:async()=>assert.fail("Do not retry access failures")});
  await assert.rejects(()=>browser.snapshot(),/Chrome snapshot: Cannot access/);
  assert.equal(reads,1);
});

test("a stop during frame settling prevents another read or navigation",async()=>{
  let stopped=false,reads=0;
  const chrome={devtools:{inspectedWindow:{tabId:42}},scripting:{executeScript:async()=>{
    reads++; throw new Error("Frame with ID 0 was removed.");
  }},tabs:{update:async()=>assert.fail("Do not navigate after a stop")}};
  const browser=new ChromeMatchBrowser(chrome,{sleep:async()=>{stopped=true;},
    check:()=>{if(stopped) throw new Error("Paused by user");}});
  await assert.rejects(()=>browser.navigate({event_id:124}),/Paused by user/);
  assert.equal(reads,1);
});

for(const outcome of ["target","wrong","challenge"]) test(`a removed link frame requires observed target identity: ${outcome}`,async()=>{
  let clicked=false; const commands=[];
  const chrome={devtools:{inspectedWindow:{tabId:42}},scripting:{executeScript:async options=>{
    const command=options.args[0]; commands.push(command);
    if(command==="matchLink") {clicked=true; throw new Error("Frame with ID 0 was removed.");}
    return [{result:{eventId:clicked?(outcome==="wrong"?125:124):123,challenge:clicked&&outcome==="challenge"}}];
  }},tabs:{update:async()=>assert.fail("Do not replay navigation after an uncertain click")}};
  const browser=new ChromeMatchBrowser(chrome);
  if(outcome==="target") assert.deepEqual(await browser.navigate({event_id:124}),{existing:false,verifiedAfterFrameChange:true});
  else await assert.rejects(()=>browser.navigate({event_id:124}),outcome==="challenge"?/Human challenge/:/Chrome matchLink: Frame/);
  assert.deepEqual(commands,["snapshot","matchLink","snapshot"]);
});

test("view actions retain their operation name and are not replayed after frame removal",async()=>{
  let actions=0;
  const chrome={devtools:{inspectedWindow:{tabId:42}},scripting:{executeScript:async()=>{
    actions++; throw new Error("Frame with ID 0 was removed.");
  }}};
  const browser=new ChromeMatchBrowser(chrome);
  await assert.rejects(()=>browser.action("view","statistics"),/Chrome view: Frame/);
  assert.equal(actions,1);
});
test("native handshake rejects an older writer and disconnects",async()=>{
  let callback,disconnected=false;
  const port={onMessage:{addListener:f=>{callback=f;}},onDisconnect:{addListener:()=>{}},
    postMessage:m=>callback({id:m.id,ok:true,result:{protocol:1,version:"0.1.1"}}),disconnect:()=>{disconnected=true;}};
  const native=new NativeClient({connectNative:()=>port});
  await assert.rejects(()=>native.connect(),/version mismatch/);
  assert.equal(disconnected,true);
});
