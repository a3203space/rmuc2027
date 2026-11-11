const {test}=require('node:test');
const assert=require('node:assert/strict');
const {consumeReplayIntro,renderedReplaySample,renderedTickFor,replayFrameAdvance,replayEndTime,replaySample,backendLabel}=require('../../apps/arena-web/src/replayPlayback.ts');

test('recorded time is conserved across slow and uneven visible frames',()=>{
 const intervals=[40,350,1250,60,2300];
 const frames=intervals.reduce((sum,ms)=>sum+replayFrameAdvance(ms,.05,1),0);
 assert.equal(frames*.05,4);
 assert.equal(replayFrameAdvance(2000,.05,8),320);
 assert.equal(replayFrameAdvance(3000,.05,.5),30);
});

test('paused/hidden zero elapsed and invalid elapsed cannot advance a replay',()=>{
 for(const elapsed of [0,-1,NaN,Infinity])assert.equal(replayFrameAdvance(elapsed,.05,1),0);
 assert.equal(replayFrameAdvance(50,0,1),1);
});

test('a 60 second clip of a 420 second match ends at its actual final frame',()=>{
 assert.equal(replayEndTime({frames:[{time:0,duration:420},{time:60,duration:420,status:'running'}]}),60);
 assert.equal(replayEndTime({frames:[{time:36},{time:173}]}),173);
});

test('backend and recorded frame rate come from the selected recording',()=>{
 assert.equal(backendLabel({physics_backend:'warp_kinematic',dt:.05},{dt:.05},true),'Warp · 20 Hz 全帧');
 assert.equal(backendLabel({physics_backend:'rapier',dt:.05},{dt:.05},true),'Rapier · 20 Hz 全帧');
 assert.equal(backendLabel(null,null,false),'后端未标注');
 assert.equal(backendLabel({physics_backend:'warp_kinematic'},{dt:.05},false),'Warp');
});

test('seek and loading another replay replace the renderer sample without waiting for RAF',()=>{
 const first={metadata:{match_id:'old',seed:1},frames:[{tick:0,time:0},{tick:1,time:.05},{tick:2,time:.1}]};
 let feed=replaySample(first,2,1,true,1,10);
 assert.equal(feed.frame.tick,2);
 feed=replaySample(first,1,1,true,2,20);
 assert.equal(feed.frame,first.frames[1]);assert.equal(feed.previous,first.frames[0]);
 assert.equal(feed.paused,true);assert.deepEqual(feed.events,[]);
 const other={metadata:{match_id:'new',seed:2},frames:[{tick:100,time:5},{tick:101,time:5.05}]};
 feed=replaySample(other,0,1,false,3,30);
 assert.equal(feed.frame,other.frames[0]);assert.equal(feed.previous,other.frames[0]);
 assert.equal(feed.generation,'replay:new:2:3');
});

test('an animation interval crossing the intro boundary advances only its leftover time',()=>{
 const split=consumeReplayIntro(2000,1);
 assert.deepEqual(split,{remainingSeconds:0,playbackElapsedMs:1000});
 assert.equal(replayFrameAdvance(split.playbackElapsedMs,.05,1),20);
 assert.deepEqual(consumeReplayIntro(1000,1),{remainingSeconds:0,playbackElapsedMs:0});
 assert.deepEqual(consumeReplayIntro(250,1),{remainingSeconds:.75,playbackElapsedMs:0});
 assert.deepEqual(consumeReplayIntro(250,0),{remainingSeconds:0,playbackElapsedMs:250});
 // Equal visible time must produce equal replay progress across frame cadences.
 let remaining=1,playback=0;
 for(const interval of [350,500,400,750]){
  const next=consumeReplayIntro(interval,remaining);remaining=next.remainingSeconds;playback+=next.playbackElapsedMs;
 }
 assert.ok(Math.abs(playback-split.playbackElapsedMs)<1e-8);
});

test('a drawn tick from another replay generation cannot satisfy current-frame readiness',()=>{
 const data={metadata:{match_id:'recording',seed:1},frames:[{tick:1,time:.05}]};
 const first=replaySample(data,0,1,true,1,10),second=replaySample(data,0,1,true,2,20);
 const oldDraw=renderedReplaySample(first);
 assert.equal(renderedTickFor(first,oldDraw),1);
 assert.equal(renderedTickFor(second,oldDraw),null);
 assert.equal(renderedTickFor(second,null),null); // Canvas/reset has no drawn sample.
 const currentDraw=renderedReplaySample(second);
 assert.equal(renderedTickFor(second,currentDraw),1);
 assert.equal(currentDraw.generation,second.generation);
});
