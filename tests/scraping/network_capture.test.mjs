import assert from "node:assert/strict";
import test from "node:test";
import { NetworkCapture, selectRequest, captureSummary } from "../../scraping/sofascore_web_v3/devtools_extension/capture.mjs";

function request(path, { status = 200, body = '{"shotmap":[]}', encoding = "", time = 1 } = {}) {
  return { startedDateTime: "2026-09-12T02:00:00Z", time,
    request: { method: "GET", url: "https://www.sofascore.com/api/v1" + path,
      headers: [{ name: "Authorization", value: "secret" }], cookies: [{ name: "session", value: "secret" }] },
    response: { status, bodySize: body.length, content: { mimeType: "application/json" } },
    getContent: callback => callback(body, encoding) };
}
function network(entries = []) {
  const listeners = new Set();
  return { onRequestFinished: { addListener: fn => listeners.add(fn), removeListener: fn => listeners.delete(fn) },
    getHAR: callback => callback({ entries }), emit: request => { for (const fn of listeners) fn(request); }, listeners };
}

test("only full selected match response bodies qualify", () => {
  for (const path of ["/event/123", "/event/123/lineups", "/event/123/heatmap/9", "/event/123/incidents"])
    assert.equal(selectRequest(request(path), 123).selected, true);
  for (const path of ["/event/1234", "/event/123/comments", "/event/124/shotmap", "/event/123/shotmap?period=1"])
    assert.equal(selectRequest(request(path), 123).selected, false);
  const external = request("/event/123"); external.request.url = "https://other.example/api/v1/event/123";
  assert.equal(selectRequest(external, 123), null);
});

test("campaign backfill selects latest current-match responses, excluding old denials and unrelated logs", async () => {
  const old=request("/event/123/shotmap",{status:403,time:1});
  const good=request("/event/123/shotmap",{status:304,time:2});
  const irrelevant=request("/event/999/shotmap",{status:403,time:3});
  const net=network([good,old,irrelevant]);
  const capture=new NetworkCapture(net,{eventId:123}); capture.start();
  await capture.includeExisting({selectedOnly:true,latestPerPath:true});
  const har=await capture.exportHAR();
  assert.equal(har.log.entries.length,1);
  assert.equal(har.log.entries[0].response.status,304);
  assert.equal(har.log.entries[0]._captureStatus,"body_saved");
});

test("live capture and backfill export existing bodies without requests or credentials", async () => {
  const shot = request("/event/123/shotmap");
  const net = network([shot]);
  const capture = new NetworkCapture(net, { eventId: 123 });
  capture.start(); net.emit(shot); await capture.includeExisting();
  const comments = request("/event/123/comments");
  comments.getContent = () => { throw new Error("Must never read commentary body"); };
  net.emit(comments);
  const har = await capture.exportHAR();
  assert.equal(capture.duplicates, 1);
  assert.equal(har.log.entries.length, 2);
  assert.equal(har.log.entries[0].response.content.text, '{"shotmap":[]}');
  assert.equal(har.log.entries[1].response.content.text, undefined);
  assert.equal(JSON.stringify(har).includes("secret"), false);
  assert.equal(net.listeners.size, 0);
});

test("query values are stripped and cannot be mistaken for full match data", async () => {
  const net = network(); const capture = new NetworkCapture(net, { eventId: 123 }); capture.start();
  net.emit(request("/event/123/shotmap?token=private&period=1"));
  const har = await capture.exportHAR();
  assert.equal(har.log.entries[0]._queryPresent, true);
  assert.equal(har.log.entries[0].response.content.text, undefined);
  assert.equal(JSON.stringify(har).includes("private"), false);
});

test("missing, oversized, and timed-out bodies remain explicit", async () => {
  const net = network(); const capture = new NetworkCapture(net, { eventId: 123, maxBodyBytes: 20, bodyTimeoutMs: 10 });
  capture.start();
  net.emit(request("/event/123/shotmap", { body: "" }));
  net.emit(request("/event/123/lineups", { body: "x".repeat(25) }));
  const pending = request("/event/123/incidents"); pending.getContent = () => {};
  net.emit(pending);
  const har = await capture.exportHAR();
  assert.deepEqual(har.log.entries.map(entry => entry._captureStatus), ["response_body_missing", "body_size_limit", "body_read_timeout"]);
});

test("recording cap stops the listener and does not control the browser", async () => {
  const net = network(); const capture = new NetworkCapture(net, { eventId: 123, maxEntries: 1 });
  capture.start(); net.emit(request("/event/123/shotmap")); net.emit(request("/event/123/lineups"));
  assert.equal(capture.reason, "recording_entry_limit");
  assert.equal(net.listeners.size, 0);
  assert.equal((await capture.exportHAR()).log.entries.length, 1);
});

test("HTTP denials are retained without attempting body access", async () => {
  const net = network(); const capture = new NetworkCapture(net, { eventId: 123 }); capture.start();
  const denied = request("/event/123/shotmap", { status: 403 });
  denied.getContent = () => { throw new Error("Do not retrieve denied content"); };
  net.emit(denied);
  const har = await capture.exportHAR();
  assert.equal(har.log.entries[0].response.status, 403);
  assert.equal(har.log.entries[0]._captureStatus, "metadata_only");
});

test("a different match produces explicit diagnostic-only export warnings", async () => {
  const net = network(); const capture = new NetworkCapture(net, { eventId: 12499375 });
  capture.start();
  const other = request("/event/16434120/lineups", { status: 304 });
  other.getContent = () => { throw new Error("Never read another match's body"); };
  net.emit(other);
  const har = await capture.exportHAR();
  const summary = har.log._xdiyo.captureSummary;
  assert.deepEqual(summary.observedEventIds, ["16434120"]);
  assert.equal(summary.selected, 0);
  assert.equal(summary.saved, 0);
  assert.match(summary.warnings.join(" "), /No requests for selected match 12499375/);
  assert.match(summary.warnings.join(" "), /diagnostics only/);
});

test("cached bodies are captured only if DevTools actually returns them", async () => {
  const net = network(); const capture = new NetworkCapture(net, { eventId: 123 });
  capture.start();
  net.emit(request("/event/123/shotmap", { status: 304 }));
  net.emit(request("/event/123/lineups", { status: 304, body: "" }));
  const har = await capture.exportHAR();
  assert.equal(har.log.entries[0].response.status, 304);
  assert.equal(har.log.entries[0]._bodySource, "devtools_cached_body");
  assert.equal(har.log.entries[0]._captureStatus, "body_saved");
  assert.equal(har.log.entries[1]._captureStatus, "response_body_missing");
  assert.equal(har.log._xdiyo.captureSummary.saved, 1);
});

test("empty and pending captures do not get premature export warnings", () => {
  assert.deepEqual(captureSummary([], 123).warnings, []);
  const waiting = { ...request("/event/123/shotmap"), _xdiyoSelected: true, _captureStatus: "reading_body" };
  assert.deepEqual(captureSummary([waiting], 123, 1).warnings, []);
  assert.match(captureSummary([waiting], 123, 0).warnings.join(" "), /diagnostics only/);
});
