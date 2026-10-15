import {normalizeSessionAnchor} from './session-collection.mjs';

// The same inspected tab and session are reused for the whole inventory.
// Python leases one exact discovery route at a time and acknowledges it on disk.
export class DiscoveryController {
  constructor({native,session,browser,onChange=()=>{},now=()=>Date.now(),random=Math.random}) {
    Object.assign(this,{native,session,browser,onChange,now,random});
    this.connected=false; this.running=false; this.state={phase:'ready',message:'Select leagues and season to discover.'};
  }
  update(patch) {Object.assign(this.state,patch);this.onChange(this.state);}
  check() {if(this.stopped) throw new Error(this.stopped);}
  async connect(options) {
    if(this.connected || this.running) throw new Error('Disconnect the current discovery first');
    await this.native.connect();
    try {
      const summary=await this.native.call(options.discovery_id?'discovery_open':'discovery_create',options);
      this.connected=true;
      this.update({phase:'connected',summary,message:`Discovery ${summary.discovery_id} saved. Ready to start or resume.`});
      return summary;
    } catch(error) {this.native.close();throw error;}
  }
  async wait(ms) {
    const deadline=this.now()+ms;
    while(this.now()<deadline) {
      this.check();
      await this.native.call('discovery_delay',{milliseconds:Math.max(1,Math.min(1000,Math.ceil(deadline-this.now())))});
    }
    this.check();
  }
  async ensureSession() {
    if(!this.session.ready()) await this.session.refresh();
    if(this.session.ready()) return;
    const anchor=await this.browser.snapshot();
    if(anchor.challenge) throw new Error('Complete the visible Sofascore challenge, then resume discovery');
    const route=normalizeSessionAnchor(anchor);
    this.update({message:'Renewing the existing Sofascore match page for discovery'});
    const delay=this.session.renewalDelayMs(30000);
    if(!Number.isFinite(delay) || delay<0 || delay>31000) throw new Error('Invalid session renewal delay');
    await this.wait(delay);
    this.session.beginRenewal();
    await this.browser.resetMemory(route,()=>{});
    for(let attempt=0;attempt<20;attempt++) {
      await this.wait(1000);
      if((await this.browser.snapshot()).challenge) throw new Error('Complete the visible Sofascore challenge, then resume discovery');
      if(this.session.ready()) return;
    }
    throw new Error('The existing page has not established session access; discovery is saved and can resume');
  }
  async run() {
    if(!this.connected || this.running) throw new Error('Connect one discovery before starting');
    this.running=true; this.stopped=null; this.update({phase:'running'});
    try {
      let resume=true;
      for(;;) {
        this.check();
        const job=await this.native.call('discovery_next',{resume}); resume=false;
        if(job.paused) throw new Error(job.reason || 'Discovery paused');
        if(job.done) {
          this.update({phase:'discovered',summary:job.summary,message:job.summary.ready_to_collect?
            'Inventory complete. Review the counts, then create the collection campaign.':
            'Enumeration finished; the preview shows unresolved schedules or match statuses. Collection is not ready.'});
          return job.summary;
        }
        await this.ensureSession(); this.check();
        const interval=(job.policy.session_request_interval_seconds ?? 0.5)*1000;
        const jitter=(job.policy.random_action_delay_max_seconds ?? 0)*this.random()*1000;
        await this.wait(Math.max(0,(this.lastRequestAt ?? -Infinity)+interval-this.now())+jitter);
        this.update({message:`Discovering ${job.label}: ${job.path}`});
        this.lastRequestAt=this.now();
        const entry=await this.session.requestDiscovery(job);
        // Save an in-flight response even when the user has just pressed Pause.
        const ack=await this.native.call('discovery_observe',{path:job.path,lease:job.lease,entry});
        const started=Date.parse(entry.startedDateTime);
        if(!Number.isFinite(started)) throw new Error('Invalid discovery request timing');
        this.lastRequestAt=Math.max(this.lastRequestAt,started);
        this.update({summary:ack.summary});
      }
    } catch(error) {
      try {await this.native.call('discovery_pause',{reason:error.message});} catch { /* The acknowledged data is already durable. */ }
      this.update({phase:'paused',message:error.message});
    } finally {this.running=false;this.update({});}
  }
  async pause() {
    this.stopped='Paused by user';
    if(!this.running && this.connected) await this.native.call('discovery_pause',{reason:this.stopped});
    this.update({message:this.stopped});
  }
  async prepare() {
    if(this.running || !this.connected || !this.state.summary?.ready_to_collect) throw new Error('Finish and review discovery first');
    const campaign=await this.native.call('discovery_prepare');
    this.update({campaign,message:`Collection campaign ${campaign.campaign_id} is prepared.`});
    return campaign;
  }
  disconnect() {
    if(this.running) throw new Error('Pause discovery before disconnecting');
    this.connected=false;this.native.close();this.update({});
  }
}
