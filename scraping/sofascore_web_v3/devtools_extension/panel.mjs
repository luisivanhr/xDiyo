import { VERSION } from './native.mjs';

const ids=['version','campaign-id','connect','run','pause','campaign-status','coverage',
  'disconnect','discovery-season','discovery-mode','load-leagues','discovery-lineups','discovery-jitter','discovery-leagues',
  'new-league-name','new-league-id','new-league-meetings','add-league','discovery-id','discovery-connect',
  'discovery-run','discovery-pause','discovery-disconnect','discovery-prepare','discovery-status','discovery-preview',
  'probe-event-id','probe-session','export-probe','probe-result',
  'event-id','start','existing','stop','export','status','warning','observed-matches','rows'];
const el=Object.fromEntries(ids.map(id=>[id,document.getElementById(id)]));
let app, reloadRequired=false;
const reloadMessage='Reload xDiyo Network Capture at chrome://extensions, then close and reopen DevTools to finish updating the collection panel.';
window.renderXDiyo = current => {
  const control=current?.controller;
  const discovery=current?.discovery;
  // A panel can be reopened from updated files while its DevTools document is
  // still running an older app. Do not render or dispatch into that old session.
  if(current?.version!==VERSION || !control?.state || !discovery?.state || ids.some(id=>!el[id])) {
    reloadRequired=true;app=null;
    if(el.version) el.version.textContent=`Panel version ${VERSION} — reload required`;
    for(const id of ['campaign-status','discovery-status','status']) if(el[id]) el[id].textContent=reloadMessage;
    for(const input of document.querySelectorAll('button,input,select')) input.disabled=true;
    return;
  }
  reloadRequired=false;
  const discovering=Boolean(discovery?.connected || discovery?.running);
  if(discovery.state.summary?.discovery_id) el['discovery-id'].value=discovery.state.summary.discovery_id;
  if(control.state.campaignId) el['campaign-id'].value=control.state.campaignId;
  el.disconnect.disabled=!control.connected || control.running;
  el['discovery-status'].textContent=`${discovery.state.phase}: ${discovery.state.message}`;
  el['discovery-preview'].textContent=discovery.state.summary?JSON.stringify(discovery.state.summary,null,2):'No discovery yet.';
  el['discovery-connect'].disabled=discovering || control.connected || control.running || current.probeRunning || Boolean(current.manual?.active);
  el['discovery-run'].disabled=!discovery.connected || discovery.running;
  el['discovery-pause'].disabled=!discovery.running;
  el['discovery-disconnect'].disabled=!discovery.connected || discovery.running;
  el['discovery-prepare'].disabled=!discovery.connected || discovery.running || !discovery.state.summary?.ready_to_collect;
  for(const id of ['discovery-id','discovery-season','discovery-mode','discovery-lineups','discovery-jitter','load-leagues','add-league']) el[id].disabled=discovering || control.connected;
  for(const input of el['discovery-leagues'].querySelectorAll('input')) input.disabled=discovering || control.connected;
  el.version.textContent=`Version ${current.version}`;
  el.connect.disabled=control.connected || control.running || discovering;
  el['campaign-id'].disabled=control.connected || control.running;
  el.run.disabled=!control.connected || control.running || Boolean(current.manual?.active) || current.probeRunning;
  el['probe-session'].disabled=control.running || discovering || current.probeRunning || Boolean(current.manual?.active);
  el['probe-event-id'].disabled=control.running || current.probeRunning;
  el['export-probe'].disabled=!current.probeResult || current.probeRunning;
  el['probe-result'].textContent=current.probeResult ? JSON.stringify(current.probeResult,null,2) : 'Not tested.';
  el.pause.disabled=!control.connected;
  el['campaign-status'].textContent=`${control.state.phase}: ${control.state.message}`;
  el.coverage.textContent=JSON.stringify(control.state.lastResult || control.state.coverage || {},null,2);
  const manual=current.manual;
  const canStart=!control.running && !discovering && !current.probeRunning && (!manual || (manual.reason==='exported' && !manual.pending.size));
  el.start.disabled=!canStart; el['event-id'].disabled=!canStart;
  el.existing.disabled=!manual?.active || control.running;
  el.stop.disabled=!manual?.active || control.running;
  el.export.disabled=!manual?.entries.length || manual.pending.size>0 || control.running;
  const capture=current.capture;
  if (!capture) return;
  const summary=current.summary();
  el.status.textContent=`${capture.active ? 'Recording' : 'Stopped'}: ${capture.entries.length} API observations, ${summary.saved} saved bodies, ${capture.pending.size} pending. ${capture.reason || ''}`;
  el.warning.hidden=!summary.warnings.length;
  el.warning.textContent=summary.warnings.join(' ');
  el['observed-matches'].textContent=`Selected match: ${capture.eventId}. Observed match IDs: ${summary.observedEventIds.join(', ') || 'none'}.`;
  el.rows.replaceChildren();
  for (const entry of capture.entries.filter(e=>!summary.selected || e._xdiyoSelected).slice(-100)) {
    const row=document.createElement('tr');
    for (const value of [new URL(entry.request.url).pathname,entry.response.status,entry._captureStatus]) {
      const cell=document.createElement('td'); cell.textContent=String(value); row.append(cell);
    }
    el.rows.append(row);
  }
};
window.attachXDiyo = current => { app=current; window.renderXDiyo(current); };
const bind=(id,handler)=>el[id]?.addEventListener('click',async()=>{
  try {
    if(reloadRequired) throw new Error(reloadMessage);
    if (!app) throw new Error('Reopen this panel to connect to DevTools');
    await handler();
  }
  catch (error) { if(el['campaign-status']) el['campaign-status'].textContent=error.message; }
});
bind('connect',()=>app.connect(el['campaign-id'].value));
bind('run',()=>app.start());
bind('pause',()=>app.pause());
bind('disconnect',()=>app.disconnect());
const leagueChoices=new Map();
function addLeague(league) {
  if(leagueChoices.has(league.competition_id)) throw new Error('This league is already listed');
  if(!/^[A-Za-z][A-Za-z0-9_]{0,60}$/.test(league.name) || !Number.isSafeInteger(league.competition_id) || league.competition_id<=0 ||
    !Number.isInteger(league.meetings_per_pair) || league.meetings_per_pair<1 || league.meetings_per_pair>4) throw new Error('Enter a valid name, competition ID and format');
  const label=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.checked=true;
  label.append(input,document.createTextNode(league.name.replaceAll('_',' ')));el['discovery-leagues'].append(label);
  leagueChoices.set(league.competition_id,{league,input});
}
bind('load-leagues',async()=>{for(const league of await app.loadLeagues()) if(!leagueChoices.has(league.competition_id)) addLeague(league);});
bind('add-league',()=>addLeague({name:el['new-league-name'].value.trim(),competition_id:Number(el['new-league-id'].value),meetings_per_pair:Number(el['new-league-meetings'].value)}));
bind('discovery-connect',async()=>{
  const summary=await app.connectDiscovery({discovery_id:el['discovery-id'].value.trim(),season:el['discovery-season'].value.trim(),
    mode:el['discovery-mode'].value,
    leagues:[...leagueChoices.values()].filter(v=>v.input.checked).map(v=>v.league),include_lineups:el['discovery-lineups'].checked,
    random_delay_max_seconds:el['discovery-jitter'].value===''?null:Number(el['discovery-jitter'].value)});
  el['discovery-id'].value=summary.discovery_id;
});
bind('discovery-run',()=>app.discovery.run());
bind('discovery-pause',()=>app.discovery.pause());
bind('discovery-disconnect',()=>app.discovery.disconnect());
bind('discovery-prepare',async()=>{const campaign=await app.prepareDiscoveredCampaign();el['campaign-id'].value=campaign.campaign_id;});
bind('probe-session',()=>app.probeSession(Number(el['probe-event-id'].value)));
bind('export-probe',()=>{
  const url=URL.createObjectURL(new Blob([JSON.stringify(app.probeResult,null,2)],{type:'application/json'}));
  const link=document.createElement('a'); link.href=url;
  link.download=`sofascore-session-test-${Date.now()}.json`; link.click();
  setTimeout(()=>URL.revokeObjectURL(url),60000);
});
bind('start',()=>app.startManual(Number(el['event-id'].value)));
bind('existing',()=>app.existing());
bind('stop',()=>app.stopManual());
bind('export',async()=>{
  const har=await app.exportManual();
  const url=URL.createObjectURL(new Blob([JSON.stringify(har)],{type:'application/json'}));
  const link=document.createElement('a'); link.href=url;
  link.download=`sofascore-event-${app.manual.eventId}-${new Date().toISOString().replace(/[:.]/g,'-')}.har`;
  link.click(); setTimeout(()=>URL.revokeObjectURL(url),60000);
});
