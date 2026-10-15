// Pure selection and a testable adapter for Chrome's existing Network log.
const HOSTS = new Set(["www.sofascore.com", "api.sofascore.com"]);

export function captureSummary(entries, eventId, pending = 0) {
  const selected = entries.filter(entry => entry._xdiyoSelected);
  const saved = selected.filter(entry => entry._captureStatus === "body_saved").length;
  const observedEventIds = [...new Set(entries.map(entry =>
    new URL(entry.request.url).pathname.match(/^\/api\/v1\/event\/(\d+)(?:\/|$)/)?.[1]
  ).filter(Boolean))].sort((a, b) => Number(a) - Number(b));
  const warnings = [];
  if (observedEventIds.length && !observedEventIds.includes(String(eventId))) {
    warnings.push(`No requests for selected match ${eventId}. The Network log contains match ID(s) ${observedEventIds.join(", ")}. Check the match displayed on the page and its #id before recording.`);
  }
  if (entries.length && !saved && !pending) {
    warnings.push("No selected-match response bodies have been saved. Export HAR will contain diagnostics only, not importable match data.");
  }
  return { selected: selected.length, saved, observedEventIds, warnings };
}

export function selectRequest(request, eventId) {
  let url;
  try { url = new URL(request.request.url); } catch { return null; }
  if (url.protocol !== "https:" || !HOSTS.has(url.hostname) || !url.pathname.startsWith("/api/v1/")) return null;
  const path = url.pathname.slice("/api/v1".length);
  const query = Boolean(url.search);
  const suffix = path.slice(`/event/${eventId}`.length);
  const selected = path.startsWith(`/event/${eventId}`) &&
    /^(?:|\/(?:pregame-form|statistics|shotmap|lineups|incidents)|\/heatmap\/\d+)$/.test(suffix);
  return { path, url: url.origin + url.pathname, query,
    selected: selected && !query && request.request.method === "GET" };
}

export class NetworkCapture {
  constructor(network, { eventId, onChange = () => {}, maxEntries = 5000,
                          maxBodyBytes = 8 * 1024 * 1024, maxTotalBytes = 100 * 1024 * 1024,
                          bodyTimeoutMs = 10000, allowedPaths = null } = {}) {
    if (!Number.isSafeInteger(eventId) || eventId <= 0) throw new Error("Choose a positive integer event ID");
    this.network = network;
    this.eventId = eventId;
    this.onChange = onChange;
    this.maxEntries = maxEntries;
    this.maxBodyBytes = maxBodyBytes;
    this.maxTotalBytes = maxTotalBytes;
    this.bodyTimeoutMs = bodyTimeoutMs;
    this.allowedPaths = allowedPaths ? new Set(allowedPaths) : null;
    this.entries = [];
    this.pending = new Set();
    this.seen = new Set();
    this.bytes = 0;
    this.active = false;
    this.reason = null;
    this.duplicates = 0;
    this.listener = request => this.observe(request);
  }

  start() {
    if (this.active) return;
    this.active = true;
    this.network.onRequestFinished.addListener(this.listener);
    this.changed();
  }

  stop(reason = "user_stopped") {
    this.active = false;
    this.reason = reason;
    this.network.onRequestFinished.removeListener(this.listener);
    this.changed();
  }

  includeExisting({ selectedOnly = false, latestPerPath = false } = {}) {
    if (!this.active) throw new Error("Start recording first");
    return new Promise((resolve, reject) => {
      let settled = false;
      const timer = setTimeout(() => { settled = true; reject(new Error("Network log read timed out")); }, this.bodyTimeoutMs);
      this.network.getHAR(log => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        let entries = log.entries || [];
        if (selectedOnly) entries = entries.filter(request => {
          const s=selectRequest(request,this.eventId);
          return s?.selected && (!this.allowedPaths || this.allowedPaths.has(s.path));
        });
        if (latestPerPath) {
          const latest=new Map();
          for (const request of [...entries].sort((a,b)=>
            Date.parse(a.startedDateTime)+Math.max(0,a.time || 0)-Date.parse(b.startedDateTime)-Math.max(0,b.time || 0))) {
            latest.set(selectRequest(request,this.eventId)?.path,request);
          }
          entries=[...latest.values()];
        }
        for (const request of entries) this.observe(request,{fromExistingLog:true});
        resolve();
      });
    });
  }

  observe(request, {fromExistingLog=false}={}) {
    if (!this.active) return;
    const selection = selectRequest(request, this.eventId);
    if (!selection) return;
    if (this.allowedPaths) selection.selected = selection.selected && this.allowedPaths.has(selection.path);
    // getHAR backfill may overlap live callbacks. Keep distinct response timings.
    const key = JSON.stringify([request.startedDateTime, request.time, request.request.method,
      request.request.url, request.response.status, request.response.bodySize]);
    if (this.seen.has(key)) { this.duplicates++; this.changed(); return; }
    if (this.entries.length >= this.maxEntries) { this.stop("recording_entry_limit"); return; }
    this.seen.add(key);
    const status = request.response.status;
    const entry = {
      startedDateTime: request.startedDateTime,
      time: Number.isFinite(request.time) ? request.time : 0,
      request: { method: request.request.method, url: selection.url, httpVersion: "", headers: [],
                 cookies: [], queryString: [], headersSize: -1, bodySize: -1 },
      response: { status, statusText: "", httpVersion: "", headers: [], cookies: [],
                  content: { size: -1, mimeType: request.response.content?.mimeType || "" },
                  redirectURL: "", headersSize: -1, bodySize: -1 },
      cache: {}, timings: { send: -1, wait: -1, receive: -1 },
      _queryPresent: selection.query,
      _xdiyoSelected: selection.selected,
      _observationSource: fromExistingLog ? "network_log" : "live_callback",
      // Preserve unknown timing as unknown for the controller's bounded resume.
      _responseEndTimeMs: Number.isFinite(Date.parse(request.startedDateTime)) &&
        Number.isFinite(request.time) && request.time >= 0 ? Date.parse(request.startedDateTime)+request.time : null,
      _captureStatus: selection.selected && [200, 304].includes(status) ? "reading_body" : "metadata_only",
    };
    this.entries.push(entry);
    // A 304 has no HTTP body, but DevTools may still expose the browser's
    // validated cached content. Never replay the request if that content is absent.
    if (selection.selected && [200, 304].includes(status)) {
      const pending = this.readBody(request, entry);
      this.pending.add(pending);
      pending.finally(() => { this.pending.delete(pending); this.changed(); });
    }
    this.changed();
  }

  readBody(request, entry) {
    return new Promise(resolve => {
      let done = false;
      const finish = (content, encoding, failure) => {
        if (done) return;
        done = true;
        clearTimeout(timer);
        if (failure || typeof content !== "string" || !content) {
          entry._captureStatus = failure || "response_body_missing";
        } else {
          const bytes = new TextEncoder().encode(content).length;
          if (bytes > this.maxBodyBytes || this.bytes + bytes > this.maxTotalBytes) {
            entry._captureStatus = "body_size_limit";
            if (this.bytes + bytes > this.maxTotalBytes) this.stop("recording_memory_limit");
          } else {
            entry.response.content.text = content;
            if (encoding) entry.response.content.encoding = encoding;
            entry.response.content.size = encoding === "base64" ? -1 : bytes;
            entry._captureStatus = "body_saved";
            entry._bodySource = entry.response.status === 304 ? "devtools_cached_body" : "devtools_response_body";
            this.bytes += bytes;
          }
        }
        resolve();
      };
      const timer = setTimeout(() => finish(null, null, "body_read_timeout"), this.bodyTimeoutMs);
      try {
        if (typeof request.getContent !== "function") finish(null, null, "body_reader_unavailable");
        else request.getContent((content, encoding) => finish(content, encoding));
      } catch { finish(null, null, "body_read_failed"); }
    });
  }

  changed() { this.onChange(this); }

  async exportHAR() {
    this.stop("exported");
    await Promise.all([...this.pending]);
    return { log: { version: "1.2", creator: { name: "xDiyo Network Capture", version: "0.2.21" },
      entries: this.entries,
      _xdiyo: { eventId: this.eventId, exportedAt: new Date().toISOString(),
        scope: "Sofascore API metadata and selected match response bodies; other resources excluded",
        duplicateCallbacksSkipped: this.duplicates,
        captureSummary: captureSummary(this.entries, this.eventId),
        note: "Backfill/live overlap is deduplicated by request metadata. Identical timing collisions may be merged. Recording limits do not limit website requests." } } };
  }
}
