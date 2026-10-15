import {selectSessionHeaders,sessionRequestDeadline} from "./session-probe.mjs";

// Runs in the existing first-party document. Only a queued component can be
// requested; the returned HAR entry contains no cookies or session headers.
export async function requestInPage(eventId, path, headers, expiresAt, executeBefore, discovery=null) {
  const allowedDiscovery=discovery && Number.isSafeInteger(discovery.competition_id) && discovery.competition_id>0 &&
    (path===`/unique-tournament/${discovery.competition_id}/seasons` ||
      (Number.isSafeInteger(discovery.season_id) && discovery.season_id>0 &&
       (new RegExp(`^/unique-tournament/${discovery.competition_id}/season/${discovery.season_id}/(?:rounds|standings/total|events/round/[1-9]\\d*)$`).test(path) ||
        (discovery.inspection_only===true && path===`/unique-tournament/${discovery.competition_id}/season/${discovery.season_id}/events/round/0`))));
  const allowedEvent=discovery===null && Number.isSafeInteger(eventId) && eventId>0 &&
    new RegExp(`^/event/${eventId}(?:/(?:lineups|statistics|incidents|shotmap|pregame-form)|/heatmap/[1-9]\\d*)?$`).test(path);
  if (location.origin !== "https://www.sofascore.com" || typeof path !== "string" ||
      !(allowedEvent || allowedDiscovery))
    throw new Error("Invalid queued component or Sofascore origin; no request sent");
  if (/verify you are human|complete the security check|checking your browser|access denied/i.test(document.body?.innerText || ""))
    throw new Error("Human challenge visible; complete it in Chrome and resume");
  const started=Date.now();
  if (!Number.isFinite(executeBefore) || started>=executeBefore || !Number.isFinite(expiresAt) || expiresAt-started<20000)
    throw new Error("Session expired before execution; no request sent");
  if (!["X-Captcha,X-Requested-With","X-Requested-With"].includes(Object.keys(headers || {}).sort().join(",")) ||
      Object.values(headers).some(v=>typeof v!=="string" || !v || v.length>8192 || /[\r\n]/.test(v)))
    throw new Error("Invalid session headers; no request sent");
  const controller=new AbortController(), timer=setTimeout(()=>controller.abort(),15000);
  const url=`https://www.sofascore.com/api/v1${path}`;
  try {
    const response=await fetch(url,{method:"GET",headers:{Accept:"application/json",...headers},
      credentials:"same-origin",redirect:"error",cache:"no-store",signal:controller.signal});
    const entry={startedDateTime:new Date(started).toISOString(),time:0,
      request:{method:"GET",url},response:{status:response.status,content:{}},
      _xdiyoSelected:true,_captureStatus:"metadata_only",_observationSource:"chrome_session_request"};
    const reader=response.body?.getReader(), decoder=new TextDecoder();
    let size=0,body="";
    if (reader) {
      try {
        for (;;) {
          const chunk=await reader.read(); if (chunk.done) break;
          size+=chunk.value.byteLength;
          // Successful component bodies share the existing writer's 8 MiB cap.
          if (size>(response.status===200?8*1024*1024:65536)) {entry._captureStatus="body_size_limit";break;}
          body+=decoder.decode(chunk.value,{stream:true});
        }
        body+=decoder.decode();
      } catch { entry._captureStatus="body_read_failed"; }
      finally { await reader.cancel().catch(()=>{}); }
    }
    if (entry._captureStatus==="metadata_only") {
      if (response.status===200 && body) {
        entry.response.content={text:body,mimeType:response.headers.get("content-type") || ""};
        entry._captureStatus="body_saved";
      } else {
        try {
          const error=JSON.parse(body)?.error;
          if (error && typeof error==="object") entry._apiError=Object.fromEntries(["code","reason"]
            .filter(k=>["number","string"].includes(typeof error[k])).map(k=>[k,String(error[k]).slice(0,120)]));
        } catch { /* Only bounded error metadata is retained. */ }
      }
    }
    entry.time=Date.now()-started;
    return entry;
  } finally {clearTimeout(timer);}
}

export class ChromeSessionClient {
  constructor(chrome,{now=()=>Date.now()}={}) {
    this.chrome=chrome; this.now=now; this.session=null; this.minimumStarted=0;
    // Retain only the observed first-party headers, never the growing Network log.
    this.listener=entry=>{
      const candidate=selectSessionHeaders([entry],this.now());
      if (candidate && candidate.started>=this.minimumStarted && (!this.session || candidate.started>=this.session.started)) this.session=candidate;
    };
    chrome.devtools.network.onRequestFinished.addListener(this.listener);
  }
  ready(minimumValidityMs=30000) {
    const now=this.now(), session=this.session;
    // Without a provider token there is no token expiry to wait for. A recent
    // successful observation still has to cover one bounded request; each 200
    // refreshes that evidence. Token-bearing sessions retain the match margin.
    return Boolean(session && session.started<=now && sessionRequestDeadline(session)-now>=30000 &&
      (session.expiresAt===null || session.expiresAt-now>=minimumValidityMs));
  }
  renewalDelayMs(minimumValidityMs) {
    const remaining=(this.session?.expiresAt ?? 0)-this.now();
    // The site's initializer keeps unexpired tokens. Wait for expiry before
    // reloading when the remaining validity cannot cover another match.
    return remaining>0 && remaining<minimumValidityMs ? Math.ceil(remaining+1000) : 0;
  }
  beginRenewal() {
    this.minimumStarted=this.now();
    this.session=null;
  }
  async refresh() {
    let timer;
    try {
      const log=await Promise.race([
        new Promise(resolve=>this.chrome.devtools.network.getHAR(resolve)),
        new Promise((resolve,reject)=>{timer=setTimeout(()=>reject(new Error("Session Network log read timed out")),10000);}),
      ]);
      const candidate=selectSessionHeaders(log?.entries || [],this.now());
      if (candidate && candidate.started>=this.minimumStarted && (!this.session || candidate.started>=this.session.started)) this.session=candidate;
    } finally {clearTimeout(timer);}
  }
  async request(job,path) {
    if (!job.paths.includes(path)) throw new Error("Component is outside the saved queue profile");
    return this.dispatch(job,path,null);
  }
  async requestDiscovery(job) {
    if (typeof job.lease!=="string" || !/^[a-f0-9]{32}$/.test(job.lease)) throw new Error("Missing discovery lease");
    return this.dispatch({},job.path,{competition_id:job.competition_id,season_id:job.season_id,
      inspection_only:job.inspection_only===true});
  }
  async dispatch(job,path,discovery) {
    if (!this.ready()) throw new Error("No recently successful, unexpired Chrome session; resume after site renewal");
    // Capture a local reference; successful background traffic can replace it.
    const session=this.session;
    let timer;
    try {
      const values=await Promise.race([
        this.chrome.scripting.executeScript({target:{tabId:this.chrome.devtools.inspectedWindow.tabId},world:"MAIN",
          func:requestInPage,args:[job.event_id ?? null,path,session.headers,sessionRequestDeadline(session),this.now()+10000,discovery]}),
        new Promise((resolve,reject)=>{timer=setTimeout(()=>reject(new Error("Session request timed out; collection paused")),30000);}),
      ]);
      const entry=values?.[0]?.result;
      if (!entry || entry.request?.url!==`https://www.sofascore.com/api/v1${path}`)
        throw new Error("No matching session response returned; collection paused");
      if (entry.response?.status===200) session.started=this.now();
      return entry;
    } finally {clearTimeout(timer);}
  }
}
