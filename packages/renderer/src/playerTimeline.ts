import type {Frame,MatchEvent,Robot} from './types';
import type {PlayerCamera} from './lanTypes';
import {cameraFromPose,interpolateRobotPose,type RobotPose} from './playerPose';

export const PLAYER_BUFFER_MS=50,PLAYER_SAMPLE_LIMIT=12,PLAYER_RESET_GAP_MS=250;
interface Sample {frame:Frame;camera:PlayerCamera;received:number}
export interface PlayerRenderSample {frame:Frame;previous:Frame;alpha:number;camera:PlayerCamera;pose:RobotPose|null;owner:string;events:MatchEvent[];time:number}

/** The network clock never rewinds the render clock. Nothing is extrapolated. */
export class PlayerTimeline {
 private samples:Sample[]=[];private generation='';private owner='';private time=0;private renderedAt=0;
 private pending=new Map<string,{event:MatchEvent;time:number}>();
 private seen=new Set<string>();
 private resets=0;private duplicates=0;private outdated=0;private underruns=0;
 clear(){this.samples=[];this.pending.clear();this.seen.clear();this.generation='';this.owner='';this.renderedAt=0;}
 get diagnostics(){return {buffer_ms:PLAYER_BUFFER_MS,samples:this.samples.length,render_time:this.time,resets:this.resets,duplicates:this.duplicates,outdated:this.outdated,underruns:this.underruns};}
 push(frame:Frame,camera:PlayerCamera,owner:string,generation:string,received=performance.now()){
  let last:Sample|undefined=this.samples[this.samples.length-1];
  if(last&&generation===this.generation&&owner===this.owner){
   if(frame.tick<=last.frame.tick){if(frame.tick===last.frame.tick)this.duplicates++;else this.outdated++;return false;}
   if(frame.time<=last.frame.time){this.outdated++;return false;}
  }
  const current=frame.robots.find(r=>r.id===owner),before=last?.frame.robots.find(r=>r.id===owner);
  const reset=!last||generation!==this.generation||owner!==this.owner||received-last.received>PLAYER_RESET_GAP_MS
   ||frame.time-last.frame.time>PLAYER_RESET_GAP_MS/1000
   ||(before&&current&&(!before.alive&&current.alive||before.model_asset!==current.model_asset||Math.hypot(current.x-before.x,current.y-before.y,current.z-before.z)>2));
  if(reset){this.clear();this.resets++;this.generation=generation;this.owner=owner;this.time=frame.time;this.renderedAt=received;last=undefined;}
  this.samples.push({frame,camera,received});if(this.samples.length>PLAYER_SAMPLE_LIMIT)this.samples.shift();
  for(const event of frame.events||[]){const key=event.id||`${event.tick}:${event.type}:${event.robot}`;if(this.seen.has(key))continue;this.seen.add(key);this.pending.set(key,{event,time:event.time??frame.time});}
  return true;
 }
 /** Geometry membership is a bounded union; visibility still uses the sampled optical frame. */
 sceneFrame():Frame|null{
  const latest=this.samples[this.samples.length-1]?.frame;if(!latest)return null;
  const robots=new Map<string,Robot>(),drones=new Map<string,NonNullable<Frame['visible_drones']>[number]>();
  for(const s of this.samples){for(const r of s.frame.robots)robots.set(r.id,r);for(const d of s.frame.visible_drones||[])drones.set(d.id,d);}
  return {...latest,robots:[...robots.values()],visible_drones:[...drones.values()]};
 }
 sample(now=performance.now()):PlayerRenderSample|null{
  const latest=this.samples[this.samples.length-1];if(!latest)return null;
  const dt=Math.max(0,now-this.renderedAt)/1000;this.renderedAt=now;
  const desired=Math.min(latest.frame.time,latest.frame.time+(now-latest.received-PLAYER_BUFFER_MS)/1000);
  // At most 10% catch-up, so packet arrival cannot jump an unfinished interpolation.
  this.time=Math.min(latest.frame.time,Math.max(this.time,Math.min(desired,this.time+dt*1.1)));
  if(this.time<this.samples[0].frame.time)this.time=latest.frame.time;
  let a=this.samples[0],b=a;
  for(let i=1;i<this.samples.length;i++){b=this.samples[i];if(b.frame.time>=this.time)break;a=b;}
  const span=b.frame.time-a.frame.time,alpha=span>0?Math.max(0,Math.min(1,(this.time-a.frame.time)/span)):1;
  const r=b.frame.robots.find(r=>r.id===this.owner),old=a.frame.robots.find(r=>r.id===this.owner)||r;
  const pose=r&&old?interpolateRobotPose(old,r,alpha):null;
  const events:MatchEvent[]=[];
  for(const [key,entry] of this.pending)if(entry.time<=this.time+1e-8){events.push(entry.event);this.pending.delete(key);}
  if(this.time>=latest.frame.time&&now-latest.received>PLAYER_BUFFER_MS)this.underruns++;
  return {frame:b.frame,previous:a.frame,alpha,camera:pose?cameraFromPose(pose,b.camera):b.camera,pose,owner:this.owner,events,time:this.time};
 }
}
