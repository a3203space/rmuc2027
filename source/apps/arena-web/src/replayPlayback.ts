import type {Frame,MatchEvent,Metadata,RenderFeed,Replay} from './types';

// A slow rendered frame must not discard recorded time. Hidden/paused periods
// are excluded by the caller before advancing this presentation clock.
export function replayFrameAdvance(elapsedMs:number,frameSeconds:number,speed:number){
  if(!Number.isFinite(elapsedMs)||elapsedMs<=0||!Number.isFinite(speed)||speed<=0)return 0;
  const dt=Number.isFinite(frameSeconds)&&frameSeconds>0?frameSeconds:.05;
  return elapsedMs/1000*speed/dt;
}

export function consumeReplayIntro(elapsedMs:number,remainingSeconds:number){
  const elapsed=Number.isFinite(elapsedMs)?Math.max(0,elapsedMs):0;
  const remaining=Number.isFinite(remainingSeconds)?Math.max(0,remainingSeconds):0;
  const consumed=Math.min(elapsed,remaining*1000);
  return {remainingSeconds:Math.max(0,remaining-consumed/1000),playbackElapsedMs:elapsed-consumed};
}

export interface RenderedReplaySample {generation:string;tick:number}
export function renderedReplaySample(feed:RenderFeed):RenderedReplaySample|null{
  return feed.frame?{generation:feed.generation,tick:feed.frame.tick}:null;
}
export function renderedTickFor(feed:RenderFeed,rendered:RenderedReplaySample|null){
  return rendered?.generation===feed.generation?rendered.tick:null;
}

export function replayEndTime(replay:Pick<Replay,'frames'>){
  return replay.frames[replay.frames.length-1]?.time??0;
}

// React's timeline and the renderer must consume the same sample immediately
// after seek/load, even when the next animation callback is delayed.
export function replaySample(data:Replay,position:number,speed:number,paused:boolean,generation:number,received:number,events:MatchEvent[]=[]):RenderFeed{
  const index=Math.max(0,Math.min(data.frames.length-1,Math.floor(position)));
  return {frame:data.frames[index],previous:data.frames[Math.max(0,index-1)],received,speed,paused,events,
    generation:'replay:'+data.metadata.match_id+':'+data.metadata.seed+':'+generation,replay:true,alpha:position%1};
}

export function backendLabel(metadata:Metadata|null,frame:Frame|null|undefined,replay:boolean){
  const backend=metadata?.physics_backend||metadata?.research?.backend;
  const label=backend==='rapier'?'Rapier':backend==='warp_kinematic'?'Warp':backend==='mujoco_warp'?'MuJoCo Warp':backend||'后端未标注';
  const dt=metadata?.dt??frame?.dt;
  return replay&&dt&&dt>0?`${label} · ${Number((1/dt).toFixed(2))} Hz 全帧`:label;
}
