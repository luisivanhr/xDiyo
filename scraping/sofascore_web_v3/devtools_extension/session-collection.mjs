const TERMINAL=new Set(["success","empty","unavailable"]);

export function normalizeSessionAnchor(anchor) {
  const url=new URL(anchor.url);
  const id=Number(url.hash.match(/^#id:([1-9]\d*)(?:,tab:(?:lineups|statistics|standings|matches))?$/)?.[1]);
  if (url.origin!=="https://www.sofascore.com" || url.username || url.password || url.search ||
      !url.pathname.startsWith("/football/match/") || !Number.isSafeInteger(anchor.eventId) ||
      anchor.eventId<=0 || id!==anchor.eventId) throw new Error("Invalid queued match URL");
  url.hash=`id:${id}`;
  return {url:url.href,event_id:id};
}

async function localRest(c, ms, message, kind) {
  if (ms<=0) return;
  c.check(); c.update({message});
  // Intentional rests do not consume collection time, including a delayed
  // callback. Local pause/lease/IPC checks remain active; this sends no HTTP.
  const deadline=c.deadline, started=c.now();
  c.deadline=null;
  try { await c.wait(ms,{page:false,checkProvider:false,nativeDelay:true}); }
  finally {
    const elapsed=Math.max(0,c.now()-started);
    c.deadline=deadline==null ? deadline : deadline+elapsed;
    c.sessionRestElapsed[kind]+=elapsed;
  }
  c.check();
}

async function waitForExpiry(c, minimumValidity) {
  const delay=c.session.renewalDelayMs(minimumValidity);
  if (!Number.isFinite(delay) || delay<0 || delay>minimumValidity+1000)
    throw new Error("Invalid session expiry wait; collection paused");
  await localRest(c,delay,"Waiting for the current session to expire before renewing it","expiry");
  return delay;
}

async function renewAnchor(c, minimumValidity, message) {
  c.check();
  const anchor=await c.browser.snapshot();
  if (anchor.challenge) throw new Error("Human challenge visible; complete it in Chrome and resume");
  c.update({message});
  // Require successful access that began after this renewal, not an old log.
  c.session.beginRenewal();
  await c.browser.resetMemory(normalizeSessionAnchor(anchor),()=>{});
  c.completedSinceMemoryReset=0; c.memoryResets++;
  c.update({memoryResets:c.memoryResets});
  for (let attempt=0;attempt<20;attempt++) {
    await c.sleep(1000); c.check();
    const page=await c.browser.snapshot();
    if (page.challenge) throw new Error("Human challenge visible; complete it in Chrome and resume");
    await c.progress({checkProvider:false});
    if (c.session.ready(minimumValidity)) return;
  }
  throw new Error("The renewed page has not established usable session access; collection paused");
}

// A queue-driven alternative to view navigation. Responses are acknowledged by
// Python individually and released; no full-match body buffer or H2H traversal.
export async function collectSessionJob(control) {
  const c=control, job=c.job;
  if (!c.session) throw new Error("Reload the extension for Chrome session collection");
  c.sessionRestElapsed={recovery:0,expiry:0};
  const minimumValidity=job.policy.match_timeout_seconds*1000+20000;
  if (!c.session.ready(minimumValidity)) await c.session.refresh();
  const resetDue=job.policy.memory_reset_every_matches!=null &&
    c.completedSinceMemoryReset>=job.policy.memory_reset_every_matches;
  let renewed=false, renewals=0, expiryWait=0, recoveryCount=0, recoveryWait=0;
  if (!c.session.ready(minimumValidity)) expiryWait+=await waitForExpiry(c,minimumValidity);
  if (resetDue || !c.session.ready(minimumValidity)) {
    await renewAnchor(c,minimumValidity,resetDue?"Renewing the collection page between saved matches":"Renewing the existing Sofascore session");
    renewed=true; renewals++;
  }
  let coverage=await c.progress({checkProvider:false}), requestCount=0;
  let pacingRequested=0, pacingElapsed=0, pacingNative=0, pacingRoundtrip=0, pacingProgress=0;
  for (;;) {
    // Daily jobs start with metadata only. Python then supplies the endpoints
    // appropriate to the newly observed status, including a match just finished.
    if (job.dynamic_paths) job.paths=coverage.components.map(item=>item.path);
    const paths=[...job.paths].sort((a,b)=>Number(b===`/event/${job.event_id}`)-Number(a===`/event/${job.event_id}`));
    const path=paths.find(path=>!TERMINAL.has(coverage.components.find(item=>item.path===path)?.status));
    if (!path) break;
    for (;;) {
      c.check();
      if (requestCount>=job.policy.max_api_observations_per_match) throw new Error("Selected-match request limit reached");
      const interval=(job.policy.session_request_interval_seconds ?? 0.5)*1000;
      const jitter=(job.policy.random_action_delay_max_seconds ?? 0)*c.random()*1000;
      const wait=Math.max(0,(c.lastSessionRequestAt ?? -Infinity)+interval-c.now())+jitter;
      if (wait>0) {
        const started=c.now(); pacingRequested+=wait;
        const timing=await c.wait(wait,{page:false,checkProvider:false,nativeDelay:true});
        pacingElapsed+=c.now()-started;
        pacingNative+=timing.native_waited_ms;
        pacingRoundtrip+=timing.native_roundtrip_ms;
        pacingProgress+=timing.progress_ms;
      }
      c.check(); c.update({message:`Requesting ${path} through the existing Chrome session`});
      c.lastSessionRequestAt=c.now();
      const entry=await c.session.request(job,path); requestCount++;
      // A pause during an in-flight GET still saves that response before stopping.
      const ack=await c.native.call("observe",{event_id:job.event_id,lease:job.lease,entry});
      const observedStart=Date.parse(entry.startedDateTime);
      if (!Number.isFinite(observedStart)) throw new Error("Missing actual request-start timing; collection paused");
      // Scripting can start after dispatch. Space from the observed API start.
      c.lastSessionRequestAt=Math.max(c.lastSessionRequestAt,observedStart);
      c.update({sessionRequests:requestCount});
      if (ack.pause_requested) c.halt(ack.run_limit_reached?"Cumulative observation limit reached":"User pause requested (local pause marker)");
      c.check();
      if (entry.response.status===403 && entry._apiError?.reason==="challenge" &&
          job.policy.session_challenge_recovery===true && recoveryCount===0 && requestCount<job.policy.max_api_observations_per_match) {
        recoveryCount++;
        recoveryWait=Math.ceil(job.policy.session_challenge_cooldown_seconds*1000);
        await localRest(c,recoveryWait,`Sofascore requested a session challenge; resting ${job.policy.session_challenge_cooldown_seconds}s before one renewal`,"recovery");
        expiryWait+=await waitForExpiry(c,minimumValidity);
        await renewAnchor(c,minimumValidity,"Renewing the existing page after the challenge rest");
        renewals++;
        // Retry this failed component once for the whole match, after renewal.
        continue;
      }
      if ([401,403,429].includes(entry.response.status))
        c.halt(`Sofascore HTTP ${entry.response.status} for ${path}${entry._apiError?.reason?`: ${entry._apiError.reason}`:""}; session collection paused`);
      c.check();
      coverage=await c.progress({checkProvider:false});
      const component=coverage.components.find(item=>item.path===path);
      if (!TERMINAL.has(component?.status) || (path===`/event/${job.event_id}` && component.status!=="success"))
        throw new Error(`Component ${path} is ${component?.status || "unverified"}; no automatic retry`);
      break;
    }
  }
  coverage=await c.progress();
  if (!coverage.complete) throw new Error("Session responses do not cover every required component");
  c.check();
  return c.native.call("finish",{event_id:job.event_id,lease:job.lease,page_event_id:null,
    reason:"session_components_complete",page_evidence:{collection_mode:"session",
      session_request_count:requestCount,session_renewed_before_match:renewed,
      session_renewal_count:renewals,session_challenge_recoveries:recoveryCount,
      session_recovery_wait_ms:recoveryWait,session_expiry_wait_ms:expiryWait,
      session_recovery_actual_wait_ms:Math.ceil(c.sessionRestElapsed.recovery),
      session_expiry_actual_wait_ms:Math.ceil(c.sessionRestElapsed.expiry),
      session_pacing_backend:"native",session_pacing_requested_wait_ms:Math.ceil(pacingRequested),
      session_pacing_actual_wait_ms:Math.ceil(pacingElapsed),
      session_pacing_native_waited_ms:Math.ceil(pacingNative),
      session_pacing_native_roundtrip_ms:Math.ceil(pacingRoundtrip),
      session_pacing_progress_ms:Math.ceil(pacingProgress),
      memory_reset_before_match:resetDue,available_views:[]}});
}
