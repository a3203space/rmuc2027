import {describe,test,expect} from 'bun:test';
import * as THREE from 'three';
import {PlayerTimeline,PLAYER_BUFFER_MS,PLAYER_SAMPLE_LIMIT} from '../../packages/renderer/src/playerTimeline';
import {cameraFromPose,interpolateRobotPose} from '../../packages/renderer/src/playerPose';
import {feedAlpha} from '../../packages/renderer/src/world';
import type {Frame,RenderFeed,Robot} from '../../packages/renderer/src/types';
import type {PlayerCamera} from '../../packages/renderer/src/lanTypes';
const aperture:PlayerCamera={origin:[0,0,.48],direction:[1,0,0],up:[0,0,1],horizontal_fov_deg:100,vertical_fov_deg:70};
function frame(tick:number,changes:Partial<Robot>={}):Frame{
 return {tick,time:tick*.05,dt:.05,arena:{width:28,height:15},robots:[{id:'red-3',team:'red',role:'infantry',slot:3,x:tick*.1,y:2,z:0,heading:0,aim:0,pitch:0,quaternion:[0,0,0,1],muzzle_height_m:.4,transform_progress:0,alive:true,model_asset:'/assets/robot-2732-spec2027.glb',...changes}],events:[]} as Frame;
}
const push=(timeline:PlayerTimeline,tick:number,at:number,changes:Partial<Robot>={})=>{const f=frame(tick,changes),r=f.robots[0];return timeline.push(f,cameraFromPose(interpolateRobotPose(r,r,1),aperture),'red-3','match',at);};
describe('one-tick player render timeline',()=>{
 test('constant forward motion is continuous at render rate with exactly 50 ms buffer',()=>{
  const t=new PlayerTimeline();push(t,0,0);let previous=0;
  for(let ms=0;ms<=500;ms+=5){if(ms>0&&ms%50===0)push(t,ms/50,ms);const s=t.sample(ms)!;
   expect(s.time).toBeCloseTo(Math.max(0,(ms-PLAYER_BUFFER_MS)/1000),8);
   expect(s.pose!.x).toBeCloseTo(s.time*2,8);expect(s.pose!.x).toBeGreaterThanOrEqual(previous-1e-10);previous=s.pose!.x;
   expect(s.camera.origin[0]).toBeCloseTo(s.pose!.x,8);
  }
 });
 test('duplicates do not reset receipt time, pose, alpha or the 50 ms delay',()=>{
  const t=new PlayerTimeline();push(t,0,0);push(t,1,50);t.sample(75);
  expect(push(t,1,77)).toBe(false);const s=t.sample(80)!;expect(s.alpha).toBeCloseTo(.6,8);expect(s.pose!.x).toBeCloseTo(.06,8);expect(t.diagnostics.duplicates).toBe(1);
 });
 test('old packets are rejected even after a long silence',()=>{
  const t=new PlayerTimeline();push(t,2,100);t.sample(125);
  expect(push(t,1,500)).toBe(false);expect(t.sample(500)!.frame.tick).toBe(2);expect(t.diagnostics.outdated).toBe(1);
 });
 test('jitter, skipped snapshots and late arrivals never rewind or jump an unfinished pose',()=>{
  const t=new PlayerTimeline(),arrivals=new Map([[0,0],[55,1],[155,3],[170,4],[250,5],[315,6],[355,7]]);let lastTime=0,lastX=0;
  for(let ms=0;ms<=450;ms+=5){if(arrivals.has(ms))push(t,arrivals.get(ms)!,ms);const s=t.sample(ms)!;
   expect(s.time).toBeGreaterThanOrEqual(lastTime-1e-10);expect(s.pose!.x).toBeGreaterThanOrEqual(lastX-1e-10);
   expect(s.pose!.x-lastX).toBeLessThanOrEqual(.01100001);expect(s.time).toBeLessThanOrEqual(s.frame.time+1e-10);lastTime=s.time;lastX=s.pose!.x;
  }
 });
 test('buffer exhaustion holds the latest camera without extrapolation',()=>{
  const t=new PlayerTimeline();push(t,0,0);push(t,1,50);const a=t.sample(250)!,b=t.sample(450)!;
  expect(a.time).toBe(.05);expect(b.time).toBe(.05);expect(b.camera).toEqual(a.camera);
 });
 test('new match, revival, teleport, model change and gaps reset without flying across the arena',()=>{
  for(const change of [{alive:true},{x:10},{model_asset:'/assets/robot-2731-spec2027.glb'}]){
   const t=new PlayerTimeline();push(t,0,0,{alive:change.alive===true?false:true});push(t,1,50,change);
   expect(t.sample(50)!.pose!.x).toBe(frame(1,change).robots[0].x);expect(t.diagnostics.samples).toBe(1);
  }
  const t=new PlayerTimeline();push(t,0,0);push(t,7,350);expect(t.sample(350)!.time).toBeCloseTo(.35,8);
  const f=frame(0);t.push(f,aperture,'red-3','new',355);expect(t.sample(355)!.time).toBe(0);
  t.clear();expect(t.sample(400)).toBeNull();
 });
 test('sampled feed alpha is identical for camera, model and world effects',()=>{
  const t=new PlayerTimeline();push(t,0,0);push(t,1,50);const s=t.sample(75)!;
  const f={frame:s.frame,previous:s.previous,playerSample:s,paused:false,received:0,speed:1,replay:false,events:[],generation:'match'} as RenderFeed;
  expect(feedAlpha(f)).toBe(s.alpha);
 });
 test('bounded geometry membership does not change the optical frame used for visibility',()=>{
  const t=new PlayerTimeline();for(let tick=0;tick<20;tick++){const f=frame(tick);f.robots.push({...f.robots[0],id:'blue-'+tick});t.push(f,aperture,'red-3','match',tick*50);t.sample(tick*50);}
  expect(t.diagnostics.samples).toBe(PLAYER_SAMPLE_LIMIT);const visible=t.sample(975)!.frame.robots.map(r=>r.id);
  expect(visible).not.toContain('blue-0');expect(t.sceneFrame()!.robots.length).toBeLessThanOrEqual(PLAYER_SAMPLE_LIMIT+1);
 });
 test('confirmed shot events are emitted once at their sampled simulation time',()=>{
  const t=new PlayerTimeline();push(t,0,0);const f=frame(1);f.events=[{id:'shot-1',tick:1,time:.05,type:'shot',robot:'red-3'}];t.push(f,aperture,'red-3','match',50);
  expect(t.sample(75)!.events).toHaveLength(0);expect(t.sample(100)!.events.map(e=>e.id)).toEqual(['shot-1']);expect(t.sample(110)!.events).toHaveLength(0);
 });
 test('body rotation crosses ±pi on the short arc and preserves real roll and pitch',()=>{
  const a=frame(0,{heading:3.1,aim:3.1,quaternion:new THREE.Quaternion().setFromEuler(new THREE.Euler(.12,-.08,3.1,'ZYX')).toArray()}).robots[0];
  const b={...a,heading:-3.1,aim:-3.1,quaternion:new THREE.Quaternion().setFromEuler(new THREE.Euler(.12,-.08,-3.1,'ZYX')).toArray()};
  const p=interpolateRobotPose(a,b,.5),c=cameraFromPose(p,aperture);
  expect(p.heading).toBeCloseTo(Math.PI,8);expect(new THREE.Vector3(...c.up as [number,number,number]).distanceTo(new THREE.Vector3(0,0,1))).toBeGreaterThan(.1);
  expect(new THREE.Vector3(...c.direction as [number,number,number]).length()).toBeCloseTo(1,10);
 });
 test('folding camera, gun elevator and mast progress share continuous height samples',()=>{
  const a=frame(0,{muzzle_height_m:.4}).robots[0],b=frame(1,{muzzle_height_m:.145,transform_progress:1}).robots[0];
  let last=.5;
  for(let i=0;i<=20;i++){const p=interpolateRobotPose(a,b,i/20),c=cameraFromPose(p,aperture);expect(c.origin[2]).toBeCloseTo(.4+(.145-.4)*i/20+.08,9);expect(p.transform_progress).toBeCloseTo(i/20,9);expect(c.origin[2]).toBeLessThanOrEqual(last+1e-8);last=c.origin[2];}
 });
});
