import {describe,expect,test} from 'bun:test';
import {readFileSync} from 'node:fs';
import * as THREE from 'three';
import {GLTFLoader} from 'three/examples/jsm/loaders/GLTFLoader.js';
import {firstPersonRig,interpolateAngle} from '../../packages/renderer/src/firstPersonRig';
import {batchRigidMeshes} from '../../packages/renderer/src/rigidMeshes';
import {fitOperatorViewport} from '../../apps/client/web/operatorViewport';

describe('operator camera and actual weapon geometry',()=>{
 test('camera and HUD fit the same 16:9 region at every screen size',()=>{
  for(const [w,h] of [[1920,1080],[1280,720],[1536,864],[3440,1440],[1280,1024]]){
   const rect=fitOperatorViewport(w,h);
   expect(rect.width/rect.height).toBeCloseTo(16/9,10);
   expect(rect.left).toBeGreaterThanOrEqual(0);expect(rect.top).toBeGreaterThanOrEqual(0);
   expect(rect.left*2+rect.width).toBeCloseTo(w,8);expect(rect.top*2+rect.height).toBeCloseTo(h,8);
   expect(960*rect.scale+rect.left).toBeCloseTo(w/2,8);
   expect(540*rect.scale+rect.top).toBeCloseTo(h/2,8);
  }
 });
 test('yaw interpolation takes the short path across ±pi',()=>{
  expect(interpolateAngle(3.1,-3.1,.5)).toBeCloseTo(Math.PI,8);
  expect(interpolateAngle(-3.1,3.1,.5)).toBeCloseTo(-Math.PI,8);
 });
 for(const model of [2711,2712,2713,2731,2732,2771,2772])test(`model ${model} keeps real hardware below the sight without covering its center`,async()=>{
  const bytes=readFileSync(`assets/web/assets/robot-${model}-spec2027.glb`);
  const {scene:source}=await new GLTFLoader().parseAsync(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),'');
  // Exercise the same rigid batching used in the live viewport.
  const scene=batchRigidMeshes(source).clone(true),role=model<2730?'heavy':model>=2770?'sentry':'infantry';
  const restore=firstPersonRig(scene,role),gun=scene.getObjectByName('GunPitch')!,turret=scene.getObjectByName('Turret')!;
  const module=scene.getObjectByName('LauncherModule');
  const lift=scene.getObjectByName('GunElevator');
  const rest=gun.position.y;
  for(const height of lift?[rest,model===2732?.145:.185]:[rest])for(const installed of module?[false,true]:[false])for(const yaw of [0,1.8,-2.7])for(const pitch of [-.26,0,.78]){
   if(module)module.visible=installed;
   if(lift)lift.position.y=height-rest;
   turret.rotation.y=-yaw;gun.rotation.z=pitch;
   scene.quaternion.setFromEuler(new THREE.Euler(.12,0,-.08));scene.updateMatrixWorld(true);
   const orientation=gun.getWorldQuaternion(new THREE.Quaternion());
   const origin=gun.getWorldPosition(new THREE.Vector3()).add(new THREE.Vector3(0,.08,0).applyQuaternion(orientation));
   const forward=new THREE.Vector3(1,0,0).applyQuaternion(orientation);
   const camera=new THREE.PerspectiveCamera(67.67,16/9,.025,220);
   camera.position.copy(origin);camera.up.set(0,1,0).applyQuaternion(orientation);camera.lookAt(origin.clone().add(forward));camera.updateMatrixWorld();
   const meshes:THREE.Mesh[]=[];
   scene.traverseVisible(node=>{if(node instanceof THREE.Mesh)meshes.push(node);});
   if(module&&!installed){expect(meshes.length).toBe(0);continue;}
   expect(meshes.length).toBeGreaterThan(0);
   const rays=new THREE.Raycaster();rays.near=.025;
   rays.setFromCamera(new THREE.Vector2(0,0),camera);
   expect(rays.intersectObjects(meshes,false).length).toBe(0);
   let visibleBarrel=false;
   for(const y of [-.5,-.75,-.95]){
    rays.setFromCamera(new THREE.Vector2(0,y),camera);
    visibleBarrel ||= rays.intersectObjects(meshes,false).length>0;
   }
   expect(visibleBarrel).toBe(true);
   for(const mesh of meshes){
    expect(mesh.material instanceof THREE.Material?mesh.material.depthTest:true).toBe(true);
    let joint:THREE.Object3D|null=mesh;
    while(joint&&joint!==gun&&joint!==module)joint=joint.parent;
    expect(joint!==null).toBe(true);
   }
  }
  restore();
  let chassisVisible=false;scene.traverseVisible(node=>{if(node instanceof THREE.Mesh&&node.name.includes('Chassis'))chassisVisible=true;});
  // Restoring the mask must bring back spectator geometry.
  expect(chassisVisible||scene.getObjectByName('Chassis')?.visible).toBe(true);
 });
});
