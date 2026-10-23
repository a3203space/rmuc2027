import * as THREE from 'three';

/** Mask only this robot's cloned meshes. Keep joints and module visibility intact. */
export function firstPersonRig(asset:THREE.Object3D,role?:string){
 const previous=new Map<THREE.Mesh,boolean>();
 asset.traverse(node=>{
  if(!(node instanceof THREE.Mesh))return;
  let part='';
  for(let parent:THREE.Object3D|null=node;parent&&parent!==asset;parent=parent.parent){
   if(parent.name==='GunPitch'||parent.name==='LauncherModule'){part=parent.name;break;}
  }
  previous.set(node,node.visible);
  // Heavy models merge the camera's surrounding shell by material. That shell
  // encloses the optical mount; retain the launcher and gun hardware below it.
  const materials=Array.isArray(node.material)?node.material:[node.material];
  const cameraHousing=role==='heavy'&&part==='GunPitch'&&materials.every(m=>['CarbonPlate','TeamPrimaryShell'].includes(m.name));
  if(!part||cameraHousing)node.visible=false;
 });
 return()=>{for(const [node,visible] of previous)node.visible=visible;};
}

export function interpolateAngle(a:number,b:number,t:number){
 return a+Math.atan2(Math.sin(b-a),Math.cos(b-a))*t;
}
