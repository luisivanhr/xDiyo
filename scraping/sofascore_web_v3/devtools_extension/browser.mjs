// DOM-only site navigation. No fetch, XHR, DevTools request replay, or new tabs.
// Controls verified against the English Sofascore match page on 2026-09-12.
export function pageAction(command, value, expiresAt = null) {
  if (location.origin !== "https://www.sofascore.com") throw new Error("Open a Sofascore match in this tab first");
  const checkExpiry = () => {
    if (expiresAt !== null && Date.now() >= expiresAt) throw new Error("Page action expired before execution");
  };
  checkExpiry();
  const visible = e => e.getClientRects().length > 0 && getComputedStyle(e).visibility !== "hidden";
  const eventId = () => Number(location.hash.match(/(?:^#|,)id:(\d+)(?:,|$)/)?.[1]) || null;
  const matchTabs = () => {
    // H2H anchors the main match navigation. Commentary's All/Key events and
    // nested period/standings filters can each have a selected tab as well.
    const anchor = document.querySelector('button[role="tab"][aria-controls="tabpanel-matches"]');
    const list = anchor?.closest('[role="tablist"]');
    if (!list || !visible(anchor)) return [];
    return [...list.querySelectorAll('button[role="tab"][aria-controls^="tabpanel-"]')]
      .filter(e=>visible(e) && !e.disabled && e.closest('[role="tablist"]') === list);
  };
  const snapshot = () => ({eventId:eventId(), url:location.href,
    // Only rendered challenge wording; never inspect hidden application state.
    challenge:/verify you are human|complete the security check|checking your browser|access denied/i.test(document.body?.innerText || ""),
    activeView:matchTabs().find(e=>e.getAttribute("aria-selected") === "true")?.getAttribute("aria-controls"),
    views:matchTabs().map(e=>e.getAttribute("aria-controls").slice("tabpanel-".length))});
  if (command === "snapshot") return snapshot();
  if (command === "navigate") {
    const url = new URL(value.url);
    if (url.origin !== location.origin || !url.pathname.startsWith("/football/match/") ||
        Number(url.hash.match(/^#id:(\d+)$/)?.[1]) !== value.eventId) throw new Error("Invalid queued match URL");
    if (snapshot().challenge) throw new Error("Human challenge visible; complete it in Chrome and resume");
    checkExpiry();
    // Reuse the inspected tab without preserving each fallback page in its
    // back-button history. This is ordinary navigation, not an API request.
    location.replace(url.href);
    return {acted:true};
  }
  if (command === "matchLink") {
    const link = [...document.querySelectorAll('main a[href]')].find(e => {
      const u = new URL(e.href, location.href);
      return visible(e) && u.origin === location.origin && u.pathname.startsWith("/football/match/") &&
        Number(u.hash.match(/(?:^#|,)id:(\d+)(?:,|$)/)?.[1]) === value;
    });
    if (!link) return {acted:false};
    link.scrollIntoView({block:"center"}); checkExpiry(); link.click(); return {acted:true};
  }
  if (command === "view") {
    if (!["lineups", "statistics", "matches"].includes(value)) throw new Error("Unknown match view");
    const tab = matchTabs().find(e=>e.getAttribute("aria-controls") === `tabpanel-${value}`);
    if (!tab || !visible(tab) || tab.disabled) return {acted:false};
    tab.scrollIntoView({block:"center"});
    checkExpiry();
    if (tab.getAttribute("aria-selected") !== "true") tab.click();
    return {acted:true};
  }
  if (command === "heatmap") {
    const panel = document.querySelector("#tabpanel-statistics");
    const heading = [...(panel?.querySelectorAll("span") || [])].find(e => e.textContent.trim() === "Attack map" && visible(e));
    let group = heading?.parentElement;
    for (let level=0; group && level<5; level++, group=group.parentElement) {
      const tabs = [...group.querySelectorAll('button[role="tab"]')];
      if (!tabs.length) continue;
      const tab = tabs.find(e => [...e.querySelectorAll("img")].some(img =>
        new URL(img.src,location.href).pathname === `/api/v1/team/${value}/image`));
      if (!tab || tabs.length !== 2 || !visible(tab)) return {acted:false};
      tab.scrollIntoView({block:"center"});
      checkExpiry();
      if (tab.getAttribute("aria-selected") !== "true") tab.click();
      return {acted:true};
    }
    return {acted:false};
  }
  if (command === "h2hNext") {
    const panel = document.querySelector("#tabpanel-matches");
    // The first pager outside nested tabpanels belongs to Head-to-head.
    const range = [...(panel?.querySelectorAll("span") || [])].find(e =>
      visible(e) && /^\d+-\d+ of \d+$/.test(e.textContent.trim()) && e.closest('[role="tabpanel"]') === panel);
    const buttons = range ? [...range.parentElement.children].filter(e => e.tagName === "BUTTON") : [];
    if (buttons.length !== 2 || buttons[1].disabled) return {acted:false};
    buttons[1].scrollIntoView({block:"center"}); checkExpiry(); buttons[1].click(); return {acted:true};
  }
  throw new Error("Unknown page operation");
}

const frameRemoved = error => /^Frame with ID 0 was removed\.?$/.test((error.cause || error)?.message || "");

export class ChromeMatchBrowser {
  constructor(chrome,{sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms)),check=()=>{},readTimeoutMs=10000,actionTimeoutMs=10000,
                      navigationMode="replace"}={}) {
    if (!["replace","update"].includes(navigationMode)) throw new Error("Unknown navigation mode");
    this.chrome=chrome; this.tabId=chrome.devtools.inspectedWindow.tabId;
    this.sleep=sleep; this.check=check; this.frameWaits=0; this.readTimeoutMs=readTimeoutMs; this.actionTimeoutMs=actionTimeoutMs;
    this.navigationMode=navigationMode;
  }
  async resetMemory(job, startRecording) {
    // Run only between acknowledged matches. A local page gives Chrome a chance
    // to release the Sofascore renderer without closing DevTools or its writer.
    const target = new URL(job.url);
    if (target.origin !== "https://www.sofascore.com" || !target.pathname.startsWith("/football/match/") ||
        Number(target.hash.match(/^#id:(\d+)$/)?.[1]) !== job.event_id) throw new Error("Invalid queued match URL");
    const page = await this.snapshot();
    this.check();
    if (page.challenge) throw new Error("Human challenge visible; complete it in Chrome and resume");
    const resetPage = this.chrome.runtime.getURL("memory-reset.html");
    const nonce = crypto.randomUUID();
    const resetURL = `${resetPage}#${nonce}`;
    const bounded = async operation => {
      this.check();
      let timer;
      try {
        return await Promise.race([operation(), new Promise((resolve,reject)=>{
          timer=setTimeout(()=>reject(new Error("Memory reset timed out; saved matches remain resumable")),this.actionTimeoutMs);
        })]);
      } finally { clearTimeout(timer); }
    };
    let markReady;
    const ready = new Promise(resolve=>{markReady=resolve;});
    const onReady = (message,sender,reply) => {
      if (sender.id !== this.chrome.runtime.id || sender.url?.split("#")[0] !== resetPage ||
          message?.type !== "xdiyo_memory_reset_ready" || message.nonce !== nonce || message.tabId !== this.tabId) return;
      reply({ok:true}); markReady();
    };
    // Register before navigation: the local page can load before tabs.update
    // resolves. Its message confirms identity without requesting tabs permission.
    this.chrome.runtime.onMessage.addListener(onReady);
    try {
      await bounded(()=>this.chrome.tabs.update(this.tabId,{url:resetURL}));
      await bounded(()=>ready);
      // No website requests on this local page. Never return after a pause.
      for (let second=0; second<2; second++) { await this.sleep(1000); this.check(); }
      startRecording();
      await bounded(()=>this.chrome.tabs.update(this.tabId,{url:target.href}));
      this.check();
      return {existing:false,memoryReset:true};
    } finally { this.chrome.runtime.onMessage.removeListener(onReady); }
  }
  async action(command,value) {
    this.check();
    let timer;
    try {
      const timeout=command === "snapshot" ? this.readTimeoutMs : this.actionTimeoutMs;
      // Chrome can queue this function while its renderer is unresponsive. The
      // in-page expiry prevents a delayed invocation from clicking after timeout.
      // Check it again after scrolling, which can synchronously force layout.
      const request = this.chrome.scripting.executeScript({target:{tabId:this.tabId}, func:pageAction,
        args:[command,value ?? null,Date.now()+timeout]});
      const results = await Promise.race([request,new Promise((resolve,reject)=>{
        const label=command === "snapshot" ? "Page read" : "Page action";
        timer=setTimeout(()=>reject(new Error(`${label} timed out; inspect Chrome for a paused or unresponsive page`)),timeout);
      })]);
      if (!results[0]?.result) throw new Error("Could not read the inspected match page");
      return results[0].result;
    } catch (error) { throw new Error(`Chrome ${command}: ${error.message}`,{cause:error}); }
    finally { clearTimeout(timer); }
  }
  async snapshot() {
    // A new document can replace frame 0 between Chrome accepting navigation
    // and executing this read. Wait at most five seconds for that exact race.
    // Check cancellation/provider stops before every read; never replay clicks.
    for (let attempt=0;;attempt++) {
      try { return {...await this.action("snapshot"),navigationFrameWaits:this.frameWaits}; }
      catch (error) {
        if (!frameRemoved(error) || attempt>=10) throw error;
        this.frameWaits++;
        await this.sleep(500);
      }
    }
  }
  async navigate(job, {refreshExisting=false}={}) {
    const page = await this.snapshot();
    this.check();
    if (page.challenge) throw new Error("Human challenge visible; complete it in Chrome and resume");
    if (page.eventId === job.event_id) {
      if (!refreshExisting) return {existing:true};
      // One ordinary refresh for an explicitly resumed failed match, never
      // a reload for every endpoint or every progress check.
      await this.chrome.tabs.reload(this.tabId);
      return {existing:false,refreshed:true};
    }
    try {
      if ((await this.action("matchLink",job.event_id)).acted) return {existing:false};
    } catch (error) {
      if (!frameRemoved(error)) throw error;
      const after=await this.snapshot();
      this.check();
      if (after.challenge) throw new Error("Human challenge visible; complete it in Chrome and resume");
      // The click may have succeeded before its old document disappeared.
      // Continue only with observed target identity, without a second click.
      if (after.eventId === job.event_id) return {existing:false,verifiedAfterFrameChange:true};
      throw error;
    }
    const url = new URL(job.url);
    if (url.origin !== "https://www.sofascore.com" || !url.pathname.startsWith("/football/match/") ||
        Number(url.hash.match(/^#id:(\d+)$/)?.[1]) !== job.event_id) throw new Error("Invalid queued match URL");
    this.check();
    if (this.navigationMode === "update") {
      await this.chrome.tabs.update(this.tabId,{url:url.href});
      return {existing:false,navigationMode:"update"};
    }
    try {
      await this.action("navigate",{url:url.href,eventId:job.event_id});
      return {existing:false,navigationMode:"replace"};
    } catch (error) {
      if (!frameRemoved(error)) throw error;
      const after=await this.snapshot();
      this.check();
      if (after.challenge) throw new Error("Human challenge visible; complete it in Chrome and resume");
      if (after.eventId === job.event_id) return {existing:false,navigationMode:"replace",verifiedAfterFrameChange:true};
      throw error;
    }
  }
}
