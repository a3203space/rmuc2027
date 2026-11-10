import {beforeAll,test,expect} from 'bun:test';
import {spawnSync} from 'node:child_process';
import {readFileSync} from 'node:fs';
import * as THREE from 'three';
import {GLTFLoader} from 'three/examples/jsm/loaders/GLTFLoader.js';
import {interpolateRobotPose,cameraFromPose} from '../../packages/renderer/src/playerPose';
import {firstPersonRig} from '../../packages/renderer/src/firstPersonRig';
import {batchRigidMeshes} from '../../packages/renderer/src/rigidMeshes';
let samples:any[];
beforeAll(()=>{
 const code=`import json,math
from rmuc2027_core.engine import Match
with Match(duration=6) as match:
 rows=[]
 for tick in range(100):
  yaw=tick*.015
  f=match.step({'red':{str(slot):{'move':[math.cos(yaw),math.sin(yaw)],'aim':yaw,'pitch':.12*math.sin(tick*.1),'heading':yaw,'transform':30<=tick<75} for slot in (1,3,4,7)}})
  obs=match.compact_observation('red',[1,3,4,7])
  rows.append({'tick':f['tick'],'time':f['time'],'robots':[r for r in f['robots'] if r['team']=='red' and r['slot'] in (1,3,4,7)],'vision':obs['vision']})
 print(json.dumps(rows))`;
 const r=spawnSync('.venv/bin/python',['-c',code],{encoding:'utf8',env:{...process.env,PYTHONPATH:'packages/core:apps/server'},maxBuffer:8*1024*1024});
 if(r.status!==0)throw Error(r.stderr);samples=JSON.parse(r.stdout);
});
for(const slot of [1,3,4,7])test(`real engine seat ${slot} keeps the optical mount and real gun aligned during movement`,async()=>{
 const model=({1:2711,3:2731,4:2732,7:2771})[slot]!,bytes=readFileSync(`assets/web/assets/robot-${model}-spec2027.glb`);
 const {scene:source}=await new GLTFLoader().parseAsync(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),'');
 const scene=batchRigidMeshes(source).clone(true),restore=firstPersonRig(scene,'infantry'),turret=scene.getObjectByName('Turret')!,gun=scene.getObjectByName('GunPitch')!,lift=scene.getObjectByName('GunElevator');
 const rest=gun.position.y;
 let folded=false;
 for(let k=0;k<samples.length-1;k++){
  const a=samples[k].robots.find((r:any)=>r.slot===slot),b=samples[k+1].robots.find((r:any)=>r.slot===slot),meta=samples[k+1].vision[String(slot)].camera;
  // Interpolation endpoints exactly match the engine, including the tilted body's up vector.
  const endpoint=cameraFromPose(interpolateRobotPose(b,b,1),meta);
  for(const key of ['origin','direction','up'] as const)for(let i=0;i<3;i++)expect(endpoint[key]![i]).toBeCloseTo(meta[key][i],9);
  for(const alpha of [0,.25,.5,.75,1]){
   const pose=interpolateRobotPose(a,b,alpha),camera=cameraFromPose(pose,meta),q=pose.quaternion!;
   scene.position.set(pose.x,pose.z,pose.y);scene.quaternion.set(-q[0],-q[2],-q[1],q[3]);
   turret.rotation.y=-(pose.aim-pose.heading);gun.rotation.z=pose.pitch;
   if(lift)lift.position.y=pose.muzzle_height_m!-rest;
   scene.updateMatrixWorld(true);
   const orientation=gun.getWorldQuaternion(new THREE.Quaternion()),eye=gun.getWorldPosition(new THREE.Vector3()).add(new THREE.Vector3(0,.08,0).applyQuaternion(orientation));
   const forward=new THREE.Vector3(1,0,0).applyQuaternion(orientation).normalize();
   expect(eye.distanceTo(new THREE.Vector3(camera.origin[0],camera.origin[2],camera.origin[1]))).toBeLessThan(1e-6);
   expect(forward.distanceTo(new THREE.Vector3(camera.direction[0],camera.direction[2],camera.direction[1]))).toBeLessThan(1e-6);
   folded||=(pose.transform_progress||0)>.9;
  }
 }
 if(slot===4)expect(folded).toBe(true);
 restore();
});
