import { NetworkCapture, captureSummary } from "./capture.mjs";
import { NativeClient, VERSION } from "./native.mjs";
import { CampaignController } from "./controller.mjs";
import { ChromeMatchBrowser } from "./browser.mjs";
import { runSessionProbe } from "./session-probe.mjs";
import {ChromeSessionClient} from "./session-client.mjs";
import {DiscoveryController} from "./discovery.mjs";

// This document lives as long as DevTools, independently of the selected panel.
let manual, panelWindow;
function refresh() { panelWindow?.renderXDiyo?.(app); }
const native = new NativeClient(chrome.runtime,{onDisconnect:reason=>{
  controller.connected=false;
  if (controller.running) controller.halt(reason);
  controller.update({phase:"disconnected",message:reason});
}});
const controller = new CampaignController({network:chrome.devtools.network,native,
  session:new ChromeSessionClient(chrome),
  browser:new ChromeMatchBrowser(chrome,{check:()=>controller.check()}),onChange:refresh});
const discoveryNative=new NativeClient(chrome.runtime,{onDisconnect:reason=>{
  discovery.connected=false;discovery.stopped=reason;
  discovery.update({phase:'disconnected',message:reason});
}});
const discovery=new DiscoveryController({native:discoveryNative,session:controller.session,
  browser:new ChromeMatchBrowser(chrome,{check:()=>discovery.check()}),onChange:refresh});
const app = {
  version:VERSION, controller, discovery, leagues:null,
  probeRunning:false, probeResult:null,
  get capture() { return controller.capture || manual; },
  summary() { const c=this.capture; return c ? captureSummary(c.entries,c.eventId,c.pending.size) : null; },
  async connect(id) {
    if(discovery.connected || discovery.running) throw new Error("Disconnect discovery first");
    if (manual?.active) throw new Error("Stop the manual recording first");
    try { await controller.connect(id); } catch (error) { native.close(); throw error; }
  },
  start() { if (this.probeRunning || discovery.connected) throw new Error("Finish the session test or disconnect discovery first"); return controller.run(); },
  pause() { return controller.pause(); },
  disconnect() {
    if(controller.running) throw new Error('Pause collection before disconnecting');
    native.close();controller.connected=false;refresh();
  },
  async loadLeagues() {
    if(this.leagues) return this.leagues;
    if(controller.connected || discovery.connected || controller.running || discovery.running || this.probeRunning || manual?.active)
      throw new Error('Disconnect collection or discovery and stop recording before loading leagues');
    const client=new NativeClient(chrome.runtime);
    try {await client.connect();this.leagues=(await client.call('discovery_catalog')).leagues;return this.leagues;}
    finally {client.close();refresh();}
  },
  async connectDiscovery(options) {
    if(controller.connected || controller.running || this.probeRunning || manual?.active)
      throw new Error('Disconnect collection and stop other recording before discovery');
    return discovery.connect(options);
  },
  async prepareDiscoveredCampaign() {
    const campaign=await discovery.prepare();
    discovery.disconnect();
    await this.connect(campaign.campaign_id);
    return campaign;
  },
  async probeSession(eventId) {
    if (controller.running || discovery.connected || this.probeRunning || manual?.active) throw new Error("Pause collection and disconnect discovery before testing");
    this.probeRunning=true; this.probeResult={state:"running",eventId}; refresh();
    try { this.probeResult=await runSessionProbe(chrome,eventId); }
    catch (error) { this.probeResult={state:"failed",message:error.message}; }
    finally { this.probeRunning=false; refresh(); }
  },
  startManual(eventId) {
    if (this.probeRunning) throw new Error("Wait for the session test");
    if (discovery.connected) throw new Error("Disconnect discovery before manual recording");
    if (controller.running) throw new Error("Pause the campaign before manual recording");
    if (manual && manual.reason !== "exported") throw new Error("Export the existing manual recording first");
    manual=new NetworkCapture(chrome.devtools.network,{eventId,onChange:refresh}); manual.start();
  },
  existing() { return manual.includeExisting(); },
  stopManual() { manual.stop(); },
  exportManual() { return manual.exportHAR(); },
  get manual() { return manual; }
};
chrome.devtools.panels.create("xDiyo", "", "panel.html", panel=>{
  panel.onShown.addListener(win=>{
    panelWindow=win;
    const attach=()=>win.attachXDiyo?.(app);
    if (win.attachXDiyo) attach(); else win.addEventListener("load",attach,{once:true});
  });
  panel.onHidden.addListener(()=>{ panelWindow=null; });
});
