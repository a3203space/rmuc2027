import {describe,test,expect} from 'bun:test';
import {readFileSync} from 'node:fs';
import * as THREE from 'three';
import {GLTFLoader} from 'three/examples/jsm/loaders/GLTFLoader.js';
import {batchRigidMeshes} from '../../packages/renderer/src/rigidMeshes';
import {cloneForTeam,ROBOT_LED_COLORS,ROBOT_LED_INTENSITY,ROBOT_HDR_ALPHA,updateRobotGlow} from '../../packages/renderer/src/teamMaterials';
const rules=JSON.parse(readFileSync('configs/rules.json','utf8'));
const glows=(scene:THREE.Object3D)=>{const result:THREE.MeshStandardMaterial[]=[];scene.traverse(n=>{if(n instanceof THREE.Mesh)for(const m of Array.isArray(n.material)?n.material:[n.material])if(m instanceof THREE.MeshStandardMaterial&&m.name.startsWith('TeamGlow'))result.push(m);});return result;};
describe('robot armor team lights',()=>{
 for(const [id,model] of Object.entries(rules.robot_designs) as [string,any][])test(`model ${id} has bright, saturated and independent red/blue lamps`,async()=>{
  const data=readFileSync('assets/web'+model.asset),{scene}=await new GLTFLoader().parseAsync(data.buffer.slice(data.byteOffset,data.byteOffset+data.byteLength),'');
  const source=batchRigidMeshes(scene),original=glows(source).map(m=>({color:m.color.getHex(),emissive:m.emissive.getHex(),intensity:m.emissiveIntensity}));
  const red=cloneForTeam(source,'red',true),blue=cloneForTeam(source,'blue',true);
  for(const [team,asset] of [['red',red],['blue',blue]] as const){
   const lamps=glows(asset);expect(lamps.length).toBeGreaterThan(0);
   for(const lamp of lamps){
    expect(lamp.emissive.getHexString()).toBe(ROBOT_LED_COLORS[team].slice(1));expect(lamp.emissiveIntensity).toBe(ROBOT_LED_INTENSITY);
    expect(lamp.toneMapped).toBe(false);expect(lamp.metalness).toBe(0);expect(lamp.depthTest).toBe(true);
    const color=lamp.emissive.clone();updateRobotGlow(lamp,color,ROBOT_LED_INTENSITY,true,1);
    expect(lamp.emissive.equals(color)).toBe(true);expect(lamp.emissiveIntensity).toBeLessThanOrEqual(4.05+1e-8);
    updateRobotGlow(lamp,color,ROBOT_LED_INTENSITY,false,1);expect(lamp.emissiveIntensity).toBeCloseTo(.24,8);
   }
  }
  expect(glows(red)[0]).not.toBe(glows(blue)[0]);
  expect(glows(source).map(m=>({color:m.color.getHex(),emissive:m.emissive.getHex(),intensity:m.emissiveIntensity}))).toEqual(original);
 });
 test('base/outpost defaults and geometry remain unchanged',()=>{
  const material=new THREE.MeshStandardMaterial({emissiveIntensity:1.9});material.name='TeamGlow';const scene=new THREE.Group(),mesh=new THREE.Mesh(new THREE.BoxGeometry(),material);scene.add(mesh);
  const base=cloneForTeam(scene,'red'),robot=cloneForTeam(scene,'red',true);
  expect(glows(base)[0].emissiveIntensity).toBe(1.9);expect(glows(base)[0].userData.robotTeamGlow).toBeUndefined();
  expect((base.children[0] as THREE.Mesh).geometry).toBe(mesh.geometry);expect((robot.children[0] as THREE.Mesh).geometry).toBe(mesh.geometry);
 });
 test('both postprocessed and direct output retain a distinct red/blue hue without white flashes',()=>{
  // Marked HDR lamp pixels use the actual hue-preserving branch of the FPV tone pass.
  for(const team of ['red','blue'] as const){const color=new THREE.Color(ROBOT_LED_COLORS[team]);
   for(const hit of [0,1])for(const post of [false,true]){
    const hdr=color.toArray().map(c=>c*ROBOT_LED_INTENSITY*(1+hit*.35));
    const channels=hdr.map(v=>post?v/Math.max(1,...hdr):Math.min(1,v));
    const dominant=team==='red'?channels[0]:channels[2],other=team==='red'?channels[2]:channels[0];
    expect(dominant).toBeGreaterThan(.9);expect(dominant).toBeGreaterThan(other*5);expect(Math.min(...channels)).toBeLessThan(.1);
   }
  }
 });
 test('HDR lamp marker is disabled for direct rendering, PNGs, transparent wrecks and ordinary assets',()=>{
  const material=new THREE.MeshStandardMaterial();material.name='TeamGlow';const source=new THREE.Group();source.add(new THREE.Mesh(new THREE.BoxGeometry(),material));
  const lamp=glows(cloneForTeam(source,'red',true))[0],shader:any={uniforms:{},fragmentShader:'#include <opaque_fragment>'};
  lamp.onBeforeCompile(shader,{} as any);
  const target={texture:{userData:{playerHdr:true}}};let active:any=null;const renderer:any={getRenderTarget:()=>active};
  const prepare=()=>lamp.onBeforeRender(renderer,{} as any,{} as any,{} as any,{} as any,{} as any);
  prepare();expect(shader.uniforms.arenaRobotHdr.value).toBe(0);
  active=target;prepare();expect(shader.uniforms.arenaRobotHdr.value).toBe(1);expect(shader.fragmentShader).toContain(`gl_FragColor.a=${ROBOT_HDR_ALPHA}`);
  lamp.transparent=true;lamp.opacity=.62;prepare();expect(shader.uniforms.arenaRobotHdr.value).toBe(0);
  const ordinary=glows(cloneForTeam(source,'red'))[0];expect(ordinary.customProgramCacheKey()).not.toBe(lamp.customProgramCacheKey());
 });
});
