// Extracted from Arena-1/frontend/src/media.ts: prerecorded announcer voices,
// original gains, multi-kill upgrades, serial queue, cache and eight-second expiry.
import {KILL_CUES,type KillCue,type KillAnnouncement} from './killAnnouncements';
type Request={announcement:KillAnnouncement;started:(seconds:number)=>void;queuedAt:number};
type Voice={node:AudioBufferSourceNode;gain:GainNode;request:Request};
const DURATION:Record<KillCue,number>={'first-blood':3.142,'double-kill':3.671,'triple-kill':2.134,'quadra-kill':2.646,'penta-kill':3.650,aced:2.793};
export class AnnouncerAudio {
 private context:AudioContext|null=null;private master:GainNode|null=null;private bus:GainNode|null=null;
 private buffers=new Map<KillCue,AudioBuffer>();private loads=new Map<KillCue,Promise<boolean>>();
 private queue:Request[]=[];private voice:Voice|null=null;private epoch=0;private volume=0;private active=true;private disposed=false;
 private starts=0;private error:string|null=null;
 get diagnostics(){return {state:this.context?.state??'none',starts:this.starts,queued:this.queue.map(r=>r.announcement.cue),loaded:this.buffers.size,error:this.error};}
 setVolume(value:number){this.volume=Math.max(0,Math.min(1,value));if(this.master&&this.context)this.master.gain.setTargetAtTime(.8*this.volume,this.context.currentTime,.025);if(!this.volume)this.cancel();}
 setActive(active:boolean){this.active=active;if(!active)this.cancel();}
 async unlock(){
  if(this.disposed||!this.active||!this.volume)return;
  try{
   if(!this.context){
    const c=this.context=new AudioContext({latencyHint:'interactive'});
    this.master=c.createGain();this.master.gain.value=.8*this.volume;this.master.connect(c.destination);
    const limiter=c.createDynamicsCompressor();limiter.threshold.value=-9;limiter.knee.value=8;limiter.ratio.value=6;limiter.attack.value=.002;limiter.release.value=.2;limiter.connect(this.master);
    this.bus=c.createGain();this.bus.gain.value=.65;this.bus.connect(limiter);
   }
   await this.context.resume();
   if(this.disposed)return;
   for(const cue of Object.keys(KILL_CUES) as KillCue[])void this.load(cue);
   this.sync();
  }catch(error){this.fail(error);}
 }
 announce(announcement:KillAnnouncement,started:(seconds:number)=>void){
  if(this.disposed||!this.active)return;
  const c=this.context;
  if(!this.volume||!c||c.state!=='running'){started(Math.max(3.4,DURATION[announcement.cue]+.2));return;}
  // Keep the current voice intact; upgrade only waiting multi-kills. ACE follows PENTA.
  if(KILL_CUES[announcement.cue].rank>=2&&KILL_CUES[announcement.cue].rank<=5)
   this.queue=this.queue.filter(r=>r.announcement.killer!==announcement.killer||KILL_CUES[r.announcement.cue].rank<2||KILL_CUES[r.announcement.cue].rank>5);
  this.queue.push({announcement,started,queuedAt:c.currentTime});
  if(this.queue.length>4){const rank=Math.min(...this.queue.map(r=>KILL_CUES[r.announcement.cue].rank));this.queue.splice(this.queue.findIndex(r=>KILL_CUES[r.announcement.cue].rank===rank),1);}
  this.sync();
 }
 cancel(){
  this.epoch++;this.queue=[];const voice=this.voice;this.voice=null;
  if(voice&&this.context){voice.gain.gain.cancelScheduledValues(this.context.currentTime);voice.gain.gain.setValueAtTime(.0001,this.context.currentTime);voice.node.stop(this.context.currentTime);}
 }
 dispose(){this.disposed=true;this.cancel();void this.context?.close();}
 private fail(error:unknown){this.error=String(error instanceof Error?error.message:error);console.warn('ARENA_ANNOUNCER',this.error);}
 private load(cue:KillCue):Promise<boolean>{
  if(this.buffers.has(cue))return Promise.resolve(true);
  const pending=this.loads.get(cue);if(pending)return pending;if(!this.context)return Promise.resolve(false);
  const load=fetch(KILL_CUES[cue].url).then(r=>{if(!r.ok)throw new Error(`Announcer unavailable: ${cue}`);return r.arrayBuffer();})
   .then(data=>this.context!.decodeAudioData(data)).then(buffer=>{this.buffers.set(cue,buffer);return true;})
   .catch(error=>{this.fail(error);this.loads.delete(cue);return false;});
  this.loads.set(cue,load);return load;
 }
 private sync(){
  const c=this.context;if(this.disposed||!this.active||!this.volume||!c||c.state!=='running'||this.voice)return;
  while(this.queue.length&&c.currentTime-this.queue[0].queuedAt>8)this.queue.shift();
  const request=this.queue[0];if(!request)return;
  const buffer=this.buffers.get(request.announcement.cue);
  if(!buffer){const epoch=this.epoch;void this.load(request.announcement.cue).then(ok=>{
   if(epoch!==this.epoch||this.queue[0]!==request)return;
   if(!ok){this.queue.shift();request.started(Math.max(3.4,DURATION[request.announcement.cue]+.2));}
   this.sync();
  });return;}
  this.queue.shift();const node=c.createBufferSource(),gain=c.createGain();node.buffer=buffer;gain.gain.value=KILL_CUES[request.announcement.cue].gain;
  node.connect(gain).connect(this.bus!);const voice={node,gain,request};this.voice=voice;this.starts++;
  node.onended=()=>{node.disconnect();gain.disconnect();if(this.voice!==voice)return;this.voice=null;this.sync();};
  try{node.start(c.currentTime);request.started(Math.max(3.4,buffer.duration+.2));}catch(error){this.voice=null;node.disconnect();gain.disconnect();this.fail(error);request.started(Math.max(3.4,DURATION[request.announcement.cue]+.2));this.sync();}
 }
}
