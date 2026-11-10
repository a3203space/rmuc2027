const test=require('node:test');
const assert=require('node:assert/strict');
const {replaySummary,sceneReady}=require('../../scripts/v6_web_acceptance.cjs');
function recording(){return {format:'rmuc-replay-v4',metadata:{test_match:true},static:{teams:{}},
 frames:[{tick:10,time:.5,robots:[],projectiles:[],events:[]},{tick:11,time:.55,robots:[],projectiles:[],events:[]}],actions:[{}]};}
test('archives canonical frames at their original 20 Hz cadence',()=>{
 assert.deepEqual(replaySummary(Buffer.from(JSON.stringify(recording()))),{frame_count:2,action_count:1,frame_hz:20,first_tick:10,last_tick:11});
});
test('rejects traces, omitted physics frames and Elo matches',()=>{
 for(const mutate of [r=>delete r.frames[0].robots,r=>r.frames[1].tick=12,r=>r.frames[1].time=.6,r=>r.actions=[],r=>r.metadata.test_match=false]){
  const replay=recording();mutate(replay);assert.throws(()=>replaySummary(Buffer.from(JSON.stringify(replay))));
 }
});
test('visual readiness requires the canvas and completed Suspense assets',()=>{
 const doc=nodes=>({querySelector:selector=>nodes[selector]||null});
 const canvas={'#viewport canvas':{width:1440,height:1000}};
 assert.equal(sceneReady(doc(canvas)),true);
 for(const nodes of [{},{...canvas,'.scene-loading':{}},{...canvas,'.scene-recovery':{}},{'#viewport canvas':{width:0,height:0}}])assert.equal(sceneReady(doc(nodes)),false);
});
