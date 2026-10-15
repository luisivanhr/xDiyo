// Explicit, bounded metadata experiment, separate from queue collection.
// First-party challenge headers stay in Chrome memory and go only to Sofascore.
const ORIGIN = "https://www.sofascore.com";
const MAX_AGE_MS = 120000;

// This is a local observation deadline, not an inferred provider token expiry.
export function sessionRequestDeadline(session) {
  return Math.min(session.expiresAt ?? Infinity, session.started + MAX_AGE_MS);
}

function header(headers, name) {
  const values = (headers || []).filter(h => h.name?.toLowerCase() === name);
  return values.length === 1 && typeof values[0].value === "string" ? values[0].value : null;
}

export function selectSessionHeaders(entries, now = Date.now()) {
  const candidates = [];
  for (const entry of entries || []) {
    let url;
    try { url = new URL(entry.request.url); } catch { continue; }
    if (url.origin !== ORIGIN || url.search || url.hash || entry.request.method !== "GET" ||
        !/^\/api\/v1\/event\/\d+(?:\/(?:lineups|statistics|incidents|shotmap|pregame-form)|\/heatmap\/\d+)?$/.test(url.pathname) ||
        ![200, 304].includes(entry.response?.status)) continue;
    const started = Date.parse(entry.startedDateTime);
    if (!Number.isFinite(started) || started > now || now - started > MAX_AGE_MS) continue;
    const captcha = header(entry.request.headers, "x-captcha");
    const requestedWith = header(entry.request.headers, "x-requested-with");
    if (!requestedWith || requestedWith.length > 256 || /[\r\n]/.test(requestedWith)) continue;
    const hasCaptcha = (entry.request.headers || []).some(h => h.name?.toLowerCase() === "x-captcha");
    if (!hasCaptcha) {
      // The site's successful requests can omit X-Captcha. Mirror only that
      // observed pattern after a successful 200/304; never discard a bad token.
      // A 304 is a fresh server validation of a conditional page request.
      if (now - started <= MAX_AGE_MS - 30000)
        candidates.push({headers:{"X-Requested-With":requestedWith}, started, expiresAt:null,
          accessMode:"observed_request",sourcePath:url.pathname,sourceStatus:entry.response.status});
      continue;
    }
    if (!captcha || captcha.length > 8192 ||
        /[\r\n]/.test(captcha + requestedWith)) continue;
    let expiresAt;
    try {
      const parts = captcha.split(".");
      if (parts.length !== 3) continue;
      const payload = parts[1].replace(/-/g, "+").replace(/_/g, "/");
      const claims = JSON.parse(atob(payload + "=".repeat((4-payload.length%4)%4)));
      if (!Number.isSafeInteger(claims.exp)) continue;
      expiresAt = claims.exp * 1000;
    } catch { continue; }
    // Leave enough validity for the request timeout. This does not verify the JWT;
    // only Sofascore can validate it. Successful recent use is additional evidence.
    if (expiresAt - now < 30000) continue;
    candidates.push({headers:{"X-Captcha":captcha,"X-Requested-With":requestedWith},
      started, expiresAt, accessMode:"captcha",sourcePath:url.pathname, sourceStatus:entry.response.status});
  }
  candidates.sort((a,b) => b.started-a.started);
  return candidates[0] || null;
}

// Self-contained for chrome.scripting.executeScript. Never returns the token.
export async function probeInPage(eventId, headers, expiresAt, executeBefore) {
  if (location.origin !== "https://www.sofascore.com" || !Number.isSafeInteger(eventId) || eventId <= 0)
    throw new Error("A positive match ID and the existing Sofascore tab are required");
  if (/verify you are human|complete the security check|checking your browser|access denied/i.test(document.body?.innerText || ""))
    throw new Error("Complete the visible site challenge before testing session access");
  const started = Date.now();
  if (!Number.isFinite(executeBefore) || started >= executeBefore || !Number.isFinite(expiresAt) || expiresAt-started < 20000)
    throw new Error("Session test expired before execution; no request was sent");
  if (!["X-Captcha,X-Requested-With","X-Requested-With"].includes(Object.keys(headers || {}).sort().join(",")) ||
      Object.values(headers).some(v => typeof v !== "string" || !v || v.length > 8192 || /[\r\n]/.test(v)))
    throw new Error("Invalid session test headers");
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch(`/api/v1/event/${eventId}`, {
      method:"GET", headers:{Accept:"application/json",...headers}, credentials:"same-origin",
      redirect:"error", cache:"no-store", signal:controller.signal,
    });
    const result = {eventId,status:response.status,elapsedMs:Date.now()-started,
      url:`https://www.sofascore.com/api/v1/event/${eventId}`,requestCount:1,
      responseHeaders:Object.fromEntries(["content-type","server","cf-mitigated","retry-after"]
        .map(name=>[name,response.headers.get(name)]).filter(([,value])=>value !== null))};
    // Bound retained response data; the normal network recorder can save full
    // metadata if manual recording is enabled. This probe never imports a match.
    const reader = response.body?.getReader();
    let bytes = 0, text = "";
    const decoder = new TextDecoder();
    if (reader) {
      try {
        for (;;) {
          const chunk = await reader.read();
          if (chunk.done) break;
          bytes += chunk.value.byteLength;
          if (bytes > 1024*1024) { result.bodyTooLarge=true; break; }
          text += decoder.decode(chunk.value,{stream:true});
        }
        text += decoder.decode();
      } finally { await reader.cancel(); }
    }
    if (!result.bodyTooLarge) {
      try {
        const body = JSON.parse(text);
        if (response.status === 200) {
          result.returnedEventId = body?.event?.id ?? null;
          result.identityVerified = body?.event?.id === eventId;
        } else if (body?.error && typeof body.error === "object") {
          result.apiError = Object.fromEntries(["code","reason"].filter(k=>
            ["number","string"].includes(typeof body.error[k])).map(k=>[k,String(body.error[k]).slice(0,120)]));
        }
      } catch { result.jsonReadable=false; }
    }
    return result;
  } finally { clearTimeout(timer); }
}

export async function runSessionProbe(chrome, eventId, {now=()=>Date.now()}={}) {
  if (!Number.isSafeInteger(eventId) || eventId <= 0) throw new Error("Choose a positive match ID");
  const entries = await new Promise((resolve,reject) => {
    let done=false;
    const timer=setTimeout(()=>{done=true;reject(new Error("Network log read timed out"));},10000);
    chrome.devtools.network.getHAR(log=>{if (!done) {done=true;clearTimeout(timer);resolve(log?.entries || []);} });
  });
  const session = selectSessionHeaders(entries,now());
  if (!session) throw new Error("No recently successful, unexpired Sofascore session header found; no request sent");
  let timer;
  try {
    const result = await Promise.race([
      chrome.scripting.executeScript({target:{tabId:chrome.devtools.inspectedWindow.tabId},
        world:"MAIN",func:probeInPage,args:[eventId,session.headers,sessionRequestDeadline(session),now()+10000]}),
      new Promise((resolve,reject)=>{timer=setTimeout(()=>reject(new Error("Session test timed out; do not repeat until its result is known")),30000);}),
    ]);
    const value=result?.[0]?.result;
    if (!value) throw new Error("The page did not return a session test result");
    return {...value,at:new Date(now()).toISOString(),sourcePath:session.sourcePath,
      sourceStatus:session.sourceStatus,sourceAgeMs:now()-session.started,
      accessMode:session.accessMode,tokenRemainingMs:session.expiresAt===null?null:session.expiresAt-now(),
      requestWindowRemainingMs:sessionRequestDeadline(session)-now(),method:"existing_chrome_session",
      note:"One explicit metadata request. No navigation, H2H, VPN, token export, or campaign queue changes."};
  } finally {
    clearTimeout(timer);
    for (const name of Object.keys(session.headers)) session.headers[name]="";
  }
}
