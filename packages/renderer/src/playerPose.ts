import * as THREE from 'three';
import type {Robot} from './types';
import type {PlayerCamera} from './lanTypes';
import {interpolateAngle} from './firstPersonRig';

export type RobotPose=Pick<Robot,'x'|'y'|'z'|'heading'|'aim'|'pitch'|'quaternion'|'muzzle_height_m'|'transform_progress'>;
const mix=(a:number,b:number,t:number)=>a+(b-a)*t;

/** Both the visible gun and the camera consume this exact physical mount pose. */
export function interpolateRobotPose(a:RobotPose,b:RobotPose,t:number):RobotPose{
 const heading=interpolateAngle(a.heading,b.heading,t);
 const localYaw=interpolateAngle(a.aim-a.heading,b.aim-b.heading,t);
 const qa=a.quaternion?new THREE.Quaternion(...a.quaternion as [number,number,number,number]):new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,0,1),a.heading);
 const qb=b.quaternion?new THREE.Quaternion(...b.quaternion as [number,number,number,number]):new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,0,1),b.heading);
 return {x:mix(a.x,b.x,t),y:mix(a.y,b.y,t),z:mix(a.z||0,b.z||0,t),heading,aim:heading+localYaw,pitch:mix(a.pitch||0,b.pitch||0,t),quaternion:qa.normalize().slerp(qb.normalize(),t).toArray(),
  muzzle_height_m:mix(a.muzzle_height_m??b.muzzle_height_m??.4,b.muzzle_height_m??a.muzzle_height_m??.4,t),transform_progress:mix(a.transform_progress||0,b.transform_progress||0,t)};
}

/** Same SI/Z-up formula as Engine::eye and physics::muzzle; aperture stays host-owned. */
export function cameraFromPose(pose:RobotPose,aperture:PlayerCamera):PlayerCamera{
 const q=pose.quaternion?new THREE.Quaternion(...pose.quaternion as [number,number,number,number]):new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0,0,1),pose.heading);
 const yaw=pose.aim-pose.heading,pitch=pose.pitch;
 const mount=new THREE.Vector3(-Math.cos(yaw)*Math.sin(pitch)*.08,-Math.sin(yaw)*Math.sin(pitch)*.08,(pose.muzzle_height_m??.4)+Math.cos(pitch)*.08).applyQuaternion(q);
 const direction=new THREE.Vector3(Math.cos(yaw)*Math.cos(pitch),Math.sin(yaw)*Math.cos(pitch),Math.sin(pitch)).applyQuaternion(q).normalize();
 return {...aperture,origin:[pose.x+mount.x,pose.y+mount.y,pose.z+mount.z],direction:direction.toArray(),up:new THREE.Vector3(0,0,1).applyQuaternion(q).normalize().toArray()};
}
