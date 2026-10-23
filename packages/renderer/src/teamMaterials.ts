import * as THREE from 'three';
import type {TeamId} from './types';

// Broadcast palette: saturated referee-light colours for LEDs and HUD fills; TEAM_COLORS stays the softer UI tint.
export const LED_COLORS:Record<TeamId,string>={red:'#ff3b30',blue:'#2f8cff'};
export const SHELL:Record<TeamId,string>={red:'#1c1517',blue:'#131922'};
export const ROBOT_LED_COLORS:Record<TeamId,string>={red:'#ff1010',blue:'#164dff'};
export const ROBOT_LED_INTENSITY=3;
export const ROBOT_HDR_ALPHA=.25;

// Real robots, bases and outposts are black machines whose team identity is carried by light strips and armour lights.
export function teamMaterial(source:THREE.Material,team?:TeamId,robotLights=false){
  const material=source.clone() as THREE.MeshStandardMaterial;
  if(!team||!material.name.startsWith('Team'))return material;
  if(material.name.startsWith('TeamPrimary')){material.color.set(SHELL[team]);material.metalness=Math.max(material.metalness,.35);material.roughness=Math.min(material.roughness,.55);material.emissive?.set('#000000');}
  else{material.color.set(LED_COLORS[team]);material.emissive?.set(LED_COLORS[team]);}
  if(robotLights&&material.name.startsWith('TeamGlow')){
    material.color.set(ROBOT_LED_COLORS[team]);material.emissive?.set(ROBOT_LED_COLORS[team]);
    material.emissiveIntensity=ROBOT_LED_INTENSITY;material.metalness=0;material.roughness=1;material.toneMapped=false;
    material.userData.robotTeamGlow=true;
    // Mark only opaque lamp pixels in the existing HDR target. The FPV tone pass
    // preserves their hue instead of desaturating them; direct/PNG output keeps normal alpha.
    const hdr={value:0};
    material.onBeforeRender=renderer=>{hdr.value=renderer.getRenderTarget()?.texture.userData.playerHdr&&material.opacity>=.99&&!material.transparent?1:0;};
    material.onBeforeCompile=shader=>{
      shader.uniforms.arenaRobotHdr=hdr;
      shader.fragmentShader='uniform float arenaRobotHdr;\n'+shader.fragmentShader.replace('#include <opaque_fragment>',`#include <opaque_fragment>\nif(arenaRobotHdr>.5)gl_FragColor.a=${ROBOT_HDR_ALPHA};`);
    };
    material.customProgramCacheKey=()=>'arena-robot-team-lamp-v1';
  }
  return material;
}

export function cloneForTeam(scene:THREE.Object3D,team?:TeamId,robotLights=false){
  const clone=scene.clone(true);
  clone.traverse(o=>{if(o instanceof THREE.Mesh){o.castShadow=true;o.receiveShadow=true;const list=Array.isArray(o.material)?o.material as THREE.Material[]:[o.material as THREE.Material];const tinted=list.map(m=>teamMaterial(m,team,robotLights));o.material=Array.isArray(o.material)?tinted:tinted[0];}});
  return clone;
}

export function updateRobotGlow(material:THREE.MeshStandardMaterial,color:THREE.Color,intensity:number,alive:boolean,hit:number){
  material.emissive.copy(color);
  material.emissiveIntensity=alive?intensity*(1+Math.max(0,Math.min(1,hit))*.35):intensity*.08;
}
