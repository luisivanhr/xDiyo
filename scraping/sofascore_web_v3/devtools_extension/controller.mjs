import { NetworkCapture, captureSummary } from "./capture.mjs";
import {collectSessionJob} from "./session-collection.mjs";
const TERMINAL = new Set(["success","empty","unavailable"]);
const BLOCKED = new Set(["blocked","rate_limited"]);
const blockKey = c => JSON.stringify([c.path,c.http_status,c.observed_at]);

function sameMatchRoute(page,job) {
  try {
    const current=new URL(page.url), target=new URL(job.url);
    return current.origin==="https://www.sofascore.com" && target.origin===current.origin &&
      current.pathname.startsWith("/football/match/") && current.pathname===target.pathname &&
      Number(target.hash.match(/^#id:(\d+)$/)?.[1])===job.event_id &&
      Number.isSafeInteger(page.eventId) && page.eventId>0 && page.eventId!==job.event_id &&
      page.views?.includes("matches");
  } catch { return false; }
}

function providerBlockReason(coverage) {
  const blocked=coverage.components?.find(c=>BLOCKED.has(c.status));
  const code=blocked?.http_status ?? (blocked?.status === "rate_limited" ? 429 : "401/403");
  return `Saved Sofascore HTTP ${code} for ${blocked?.path || "this match"}; this attempt obtained no newer response. Check site access, then Start / resume to retry.`;
}

export class CampaignController {
  constructor({network, native, browser, session=null, onChange=()=>{}, now=()=>Date.now(),
               sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms)), random=Math.random}) {
    Object.assign(this,{network,native,browser,session,onChange,now,sleep,random});
    this.state={phase:"ready",message:"Prepare a campaign in the notebook, then connect to Python."};
    this.running=false; this.uploads=Promise.resolve();
    this.recoverableBlocks=new Set();
    this.resumeLogCutoff=null;
    this.completedSinceMemoryReset=0; this.memoryResets=0;
  }
  update(patch) { Object.assign(this.state,patch); this.onChange(this.state); }
  halt(reason) { this.stopReason ||= reason; this.capture?.stop(reason); this.update({message:reason}); }
  check() {
    if (this.stopReason) throw new Error(this.stopReason);
    if (this.deadline && this.now()>this.deadline) throw new Error("Match time budget reached; saved responses remain resumable");
  }
  async connect(campaignId) {
    if (this.running || this.connected) throw new Error("A campaign is already connected");
    const hello = await this.native.connect();
    await this.native.call("open",{campaign_id:campaignId.trim()});
    this.connected=true;
    this.update({phase:"connected",campaignId,version:hello.version,message:`Python ${hello.version} connected. Ready to start or resume.`});
  }
  async pause() {
    this.halt("Paused by user");
    if (!this.running && this.connected) await this.native.call("pause",{reason:"user_paused"});
  }
  changed(capture) {
    const summary=captureSummary(capture.entries,capture.eventId,capture.pending.size);
    this.update({observations:capture.entries.length,summary});
    for (const entry of capture.entries) {
      if (entry._captureStatus === "reading_body" || this.sent.has(entry)) continue;
      this.sent.add(entry);
      this.uploads=this.uploads.then(async()=>{
        const result=await this.native.call("observe",{event_id:this.job.event_id,lease:this.job.lease,entry});
        if (result.pause_requested) this.halt(result.run_limit_reached ?
          "Cumulative observation limit reached; update the saved policy and reconnect" : "User pause requested (local pause marker)");
      }).catch(error=>{ this.uploadError=error; this.halt(error.message); });
      const path=new URL(entry.request.url).pathname;
      const denialKey=blockKey({path:path.slice("/api/v1".length),
        http_status:entry.response.status,observed_at:entry.startedDateTime});
      // More responses can finish after the first denial stopped recording.
      // A manual resume may backfill those older failures for the first job;
      // it must still obtain newer successful responses before publication.
      if (entry._xdiyoSelected && [401,403,429].includes(entry.response.status) &&
          entry._observationSource === "network_log" && Number.isFinite(this.resumeLogCutoff) &&
          Number.isFinite(entry._responseEndTimeMs) && entry._responseEndTimeMs <= this.resumeLogCutoff)
        this.recoverableBlocks.add(denialKey);
      const knownDenial=entry._observationSource === "network_log" && this.recoverableBlocks.has(denialKey);
      if (entry._xdiyoSelected && [401,403,429].includes(entry.response.status) && !knownDenial && !this.stopReason)
        this.halt(`Sofascore HTTP ${entry.response.status} for ${path}; review site access, then Start / resume`);
    }
    if (summary.selected >= this.job.policy.max_api_observations_per_match && !this.stopReason) this.halt("Selected-match observation limit reached");
    if (capture.reason?.startsWith("recording_") && !this.stopReason) this.halt(capture.reason);
  }
  async flush() {
    if (this.capture) await Promise.all([...this.capture.pending]);
    await this.uploads;
    if (this.uploadError) throw this.uploadError;
  }
  async progress({checkProvider=true}={}) {
    await this.flush(); this.check();
    const coverage=await this.native.call("progress",{event_id:this.job.event_id,lease:this.job.lease});
    this.update({coverage});
    if (coverage.pause_requested) this.halt("User pause requested (local pause marker)");
    else if (coverage.run_limit_reached) this.halt("Cumulative observation limit reached; update the saved policy and reconnect");
    else if (checkProvider && coverage.blocked) {
      const blocking=(coverage.components || []).filter(c=>BLOCKED.has(c.status) && !this.recoverableBlocks.has(blockKey(c)));
      if (blocking.length || !coverage.components?.length) this.halt(providerBlockReason({...coverage,components:blocking}));
    }
    this.check();
    return coverage;
  }
  async wait(ms, {page=true,checkProvider=true,untilReady=null,nativeDelay=false}={}) {
    const started=this.now(), until=started+Math.max(0,ms);
    const timing={elapsed_ms:0,native_waited_ms:0,native_roundtrip_ms:0,progress_ms:0};
    try { do {
      this.check();
      const remaining=until-this.now();
      if (remaining>0) {
        if (nativeDelay) {
          const milliseconds=Math.ceil(Math.min(1000,remaining));
          const dispatched=this.now();
          let result;
          try { result=await this.native.call("delay",{event_id:this.job.event_id,lease:this.job.lease,milliseconds}); }
          finally { timing.native_roundtrip_ms+=Math.max(0,this.now()-dispatched); }
          if (!Number.isFinite(result?.waited_ms) || result.waited_ms<0)
            throw new Error("Python returned invalid local pacing evidence");
          timing.native_waited_ms+=result.waited_ms;
          if (result.pause_requested) this.halt(result.run_limit_reached ?
            "Cumulative observation limit reached" : "User pause requested (local pause marker)");
          this.check();
          if (result.waited_ms+1<milliseconds) throw new Error("Python pacing ended before the requested delay");
        } else await this.sleep(Math.min(1000,remaining));
      }
      let snapshot=null;
      if (page) {
        snapshot=await this.browser.snapshot();
        if (snapshot.challenge) { this.halt("Human challenge visible; complete it in Chrome and resume"); this.check(); }
      }
      const progressStarted=this.now();
      let coverage;
      try { coverage=await this.progress({checkProvider}); } // Local IPC only, never an HTTP poll.
      finally { timing.progress_ms+=Math.max(0,this.now()-progressStarted); }
      if (untilReady?.({snapshot,coverage})) break;
    } while (this.now()<until);
    return timing;
    } catch (error) {
      if (nativeDelay && error.message.startsWith("Match time budget reached"))
        error.message+=`; local wait ${Math.ceil(this.now()-started)}ms, Python ${Math.ceil(timing.native_waited_ms)}ms, round trip ${Math.ceil(timing.native_roundtrip_ms)}ms`;
      throw error;
    } finally { timing.elapsed_ms=Math.max(0,this.now()-started); }
  }
  matchReady({snapshot,coverage}) {
    return snapshot?.eventId === this.job.event_id && snapshot.activeView?.startsWith("tabpanel-") &&
      coverage.components.some(c=>c.component==="event" && c.status==="success");
  }
  actionReady(command,value) {
    if (command==="matchLink") return state=>this.matchReady(state);
    const components={lineups:["lineups"],statistics:["statistics","shotmap","heatmap"],matches:["pregame"]}[value];
    if (command==="view" && components) return ({snapshot,coverage})=>
      snapshot?.eventId === this.job.event_id && snapshot.activeView === `tabpanel-${value}` &&
      coverage.components.filter(c=>components.includes(c.component)).every(c=>TERMINAL.has(c.status));
    if (command==="heatmap") return ({snapshot,coverage})=>snapshot?.eventId === this.job.event_id &&
      snapshot.activeView === "tabpanel-statistics" && coverage.components.some(c=>
        c.path===`/event/${this.job.event_id}/heatmap/${value}` && TERMINAL.has(c.status));
    return null;
  }
  async act(command,value) {
    this.check();
    const max=this.job.policy.random_action_delay_max_seconds;
    if (max != null && max>0) await this.wait(this.random()*max*1000);
    this.check();
    const result=await this.browser.action(command,value);
    if (result.acted) await this.wait(this.job.policy.view_wait_seconds*1000,{untilReady:this.actionReady(command,value)});
    return result;
  }
  async ensureIdentity({required=true,search=true}={}) {
    let page=await this.browser.snapshot();
    if (page.eventId === this.job.event_id) return true;
    if (search) {
      this.update({message:"Finding the historical fixture in the visible H2H list"});
      await this.act("view","matches");
      for (let pages=0; pages<6; pages++) {
        if ((await this.act("matchLink",this.job.event_id)).acted) break;
        if (pages===5 || !(await this.act("h2hNext")).acted) break;
      }
      page=await this.browser.snapshot();
    }
    if (page.eventId === this.job.event_id) return true;
    if (required) throw new Error(`Wrong match displayed (${page.eventId}); automatic history search could not find queued match ${this.job.event_id}`);
    return false;
  }
  async collectViews(coverage) {
    const needed=component=>coverage.components.some(c=>c.component===component && !TERMINAL.has(c.status));
    for (const view of ["lineups","statistics","matches"]) {
      const needs = view==="lineups" ? needed("lineups") : view==="statistics" ?
        needed("statistics") || needed("shotmap") || needed("heatmap") : needed("pregame");
      if (!needs) continue;
      this.check();
      this.update({message:`Reading ${view}`});
      await this.act("view",view); coverage=await this.progress();
      if (view==="statistics") {
        for (const item of coverage.components.filter(c=>c.component==="heatmap" && !TERMINAL.has(c.status))) {
          await this.act("heatmap",Number(item.path.split("/").at(-1)));
          coverage=await this.progress();
        }
      }
      if (coverage.complete) break;
    }
    return coverage;
  }
  async run() {
    if (!this.connected || this.running) throw new Error("Connect to Python first; only one run may be active");
    const requestedAt=this.now();
    this.running=true; this.stopReason=null; this.deadline=null; this.update({phase:"running"});
    try {
      let resume=true;
      while (true) {
        this.check();
        const resuming=resume;
        this.resumeLogCutoff=resuming ? requestedAt : null;
        this.job=await this.native.call("next",{resume}); resume=false;
        if (this.job.done) { this.update({phase:"finished",message:this.job.status.state}); break; }
        if (this.job.paused) throw new Error("Campaign is paused");
        // Page-mode resume grants one attempt for exact saved failures.
        // Session mode has a separate, explicitly configured recovery policy.
        this.recoverableBlocks=new Set((resuming ? this.job.coverage?.components || [] : [])
          .filter(c=>BLOCKED.has(c.status) && c.observed_at && Number.isInteger(c.http_status)).map(blockKey));
        this.capture=null; this.sent=new Set(); this.uploads=Promise.resolve(); this.uploadError=null;
        this.update({eventId:this.job.event_id,label:this.job.label,coverage:this.job.coverage,lastResult:null,message:"Preparing queued match"});
        // During the season rest, check local controls but defer old response
        // failures until the recorder can ingest a recovered page's Network log.
        const seasonRest=this.job.wait_until*1000-this.now();
        if (seasonRest>0) this.update({message:"Resting between seasons; the next match will start automatically"});
        await this.wait(seasonRest,{page:false,checkProvider:false});
        this.deadline=this.now()+this.job.policy.match_timeout_seconds*1000;
        if (this.job.policy.collection_mode === "session") {
          const result=await collectSessionJob(this);
          this.deadline=null;
          this.update({lastResult:result,message:`Match ${this.job.event_id}: ${result.state || "paused"}`});
          if (result.pause_requested || result.paused) throw new Error("Match needs attention; inspect the coverage report before continuing");
          this.completedSinceMemoryReset++;
          continue;
        }
        this.capture=new NetworkCapture(this.network,{eventId:this.job.event_id,allowedPaths:this.job.paths,
          onChange:state=>this.changed(state)});
        const resetEvery=this.job.policy.memory_reset_every_matches;
        const resetDue=resetEvery != null && this.completedSinceMemoryReset>=resetEvery;
        let navigation, existing=false, refreshExisting=false, historicalAttempted=false;
        if (resetDue) {
          this.update({message:`Renewing the page after ${this.completedSinceMemoryReset} saved matches`});
          navigation=await this.browser.resetMemory(this.job,()=>this.capture.start());
          this.completedSinceMemoryReset=0; this.memoryResets++;
          this.update({memoryResets:this.memoryResets,message:"Memory reset finished; opening queued match"});
        } else {
          this.capture.start();
          const beforeNavigation=await this.browser.snapshot();
          this.check();
          if (beforeNavigation.challenge) throw new Error("Human challenge visible; complete it in Chrome and resume");
          existing=beforeNavigation.eventId === this.job.event_id;
          if (existing) await this.capture.includeExisting({selectedOnly:true,latestPerPath:true});
          const before=await this.progress();
          refreshExisting=existing && before.components.some(c=>BLOCKED.has(c.status) && this.recoverableBlocks.has(blockKey(c)));
          if (this.recoverableBlocks.size) this.update({message:"Recovering the queued match and its missing sections"});
          // A failed historical lookup can leave a newer fixture for the same
          // team pair loaded. On explicit resume, try its real H2H links before
          // opening another document. Recording and the existing request/time
          // limits remain active, and any new selected denial still stops.
          historicalAttempted=resuming && this.recoverableBlocks.size>0 && sameMatchRoute(beforeNavigation,this.job);
          if (historicalAttempted && await this.ensureIdentity({required:false}))
            navigation={existing:false,historicalRecovery:true};
          else navigation=await this.browser.navigate(this.job,{refreshExisting});
        }
        let recoveryRefreshed=Boolean(refreshExisting || navigation.refreshed);
        // Backfill only when the tab was already on this fixture. Navigation
        // captures fresh callbacks and does not pull unrelated old Network logs.
        if (navigation.existing && !existing) await this.capture.includeExisting({selectedOnly:true,latestPerPath:true});
        await this.wait(this.job.policy.view_wait_seconds*1000,{untilReady:state=>this.matchReady(state)});
        await this.ensureIdentity({search:!historicalAttempted});
        await this.collectViews(await this.progress());
        // A still-blocked saved response stops here. Ordinary missing data can
        // instead reflect a page populated before DevTools began recording.
        this.recoverableBlocks.clear();
        let coverage=await this.progress();
        if (!coverage.complete && !refreshExisting && !navigation.refreshed) {
          const page=await this.browser.snapshot();
          if (page.eventId !== this.job.event_id) throw new Error("Page switched matches during collection; capture paused");
          this.check();
          this.update({message:"Refreshing incomplete match once with recording active"});
          await this.browser.navigate(this.job,{refreshExisting:true});
          recoveryRefreshed=true;
          await this.wait(this.job.policy.view_wait_seconds*1000,{untilReady:state=>this.matchReady(state)});
          await this.ensureIdentity();
          coverage=await this.collectViews(await this.progress());
        }
        // Both passes share the original deadline and observation limit. A
        // still-incomplete result remains incomplete and pauses in Python.
        await this.progress();
        await this.flush(); this.check();
        const page=await this.browser.snapshot();
        if (page.eventId !== this.job.event_id) throw new Error("Page switched matches during collection; capture paused");
        this.capture.stop("match_finished"); await this.flush(); this.check();
        const result=await this.native.call("finish",{event_id:this.job.event_id,lease:this.job.lease,page_event_id:page.eventId,
          page_evidence:{available_views:page.views || [],recovery_refreshed:recoveryRefreshed,
            historical_link_recovery:Boolean(navigation.historicalRecovery),
            memory_reset_before_match:Boolean(navigation.memoryReset),
            navigation_frame_waits_total:page.navigationFrameWaits ?? 0}});
        this.capture=null;
        this.deadline=null;
        this.update({lastResult:result,message:`Match ${this.job.event_id}: ${result.state || "paused"}`});
        if (result.pause_requested || result.paused) throw new Error("Match needs attention; inspect the coverage report before continuing");
        this.completedSinceMemoryReset++;
      }
    } catch (error) {
      this.halt(error.message);
      try { await this.flush(); } catch { /* unacknowledged entries remain unclaimed */ }
      if (this.connected) {
        try { await this.native.call("pause",{reason:error.message}); } catch { /* Host pauses on disconnect. */ }
      }
      this.update({phase:"paused",message:error.message});
    } finally { this.running=false; this.onChange(this.state); }
  }
}
