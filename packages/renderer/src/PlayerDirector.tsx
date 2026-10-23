import {useEffect,useMemo,useRef,type MutableRefObject} from 'react';
import {useFrame,useThree} from '@react-three/fiber';
import * as THREE from 'three';
import type {PlayerCamera} from './lanTypes';
import type {RenderFeed} from './types';
import {feedAlpha} from './world';

/** FPV uses the host camera, including body roll and pitch. No spectator controls. */
export function PlayerDirector({camera:meta,feed,scale,width,height,onRender}:{camera:PlayerCamera;feed:MutableRefObject<RenderFeed>;scale:number;width:number;height:number;onRender:()=>void}){
  const {camera,gl}=useThree();
  const previous=useRef(meta),current=useRef(meta);
  const shadowSlot=useRef(-1);
  const postFailed=useRef(false),targetChecked=useRef(false);
  const pixelsChecked=useRef(false),sample=useMemo(()=>new Uint8Array(4),[]);
  const origin=useMemo(()=>new THREE.Vector3(),[]),direction=useMemo(()=>new THREE.Vector3(),[]);
  const outputSize=useMemo(()=>new THREE.Vector2(),[]);
  useEffect(()=>{previous.current=current.current;current.current=meta;},[meta]);
  // Render at 1080p, box downsample according to lambda, then linearly upscale.
  const post=useMemo(()=>{
    const high=new THREE.WebGLRenderTarget(1920,1080,{type:THREE.HalfFloatType,minFilter:THREE.LinearFilter,magFilter:THREE.LinearFilter});
    const color=new THREE.WebGLRenderTarget(1920,1080,{depthBuffer:false,minFilter:THREE.LinearFilter,magFilter:THREE.LinearFilter});
    const low=new THREE.WebGLRenderTarget(1920,1080,{minFilter:THREE.LinearFilter,magFilter:THREE.LinearFilter});
    const quad=new THREE.PlaneGeometry(2,2),screen=new THREE.Scene(),eye=new THREE.OrthographicCamera(-1,1,1,-1,0,1);
    const vertex='varying vec2 uv0;void main(){uv0=uv;gl_Position=vec4(position.xy,0.,1.);}';
    const tone=new THREE.ShaderMaterial({uniforms:{source:{value:high.texture},toneMappingExposure:{value:1.18}},vertexShader:vertex,
      fragmentShader:`uniform sampler2D source;varying vec2 uv0;
      #include <tonemapping_pars_fragment>
      void main(){vec3 c=ACESFilmicToneMapping(texture2D(source,uv0).rgb);
      c=mix(12.92*c,1.055*pow(c,vec3(1./2.4))-.055,step(vec3(.0031308),c));gl_FragColor=vec4(c,1.);}`});
    const box=new THREE.ShaderMaterial({uniforms:{source:{value:color.texture},size:{value:new THREE.Vector2(1920,1080)},outputSize:{value:new THREE.Vector2(1920,1080)}},
      vertexShader:'varying vec2 uv0;void main(){uv0=uv;gl_Position=vec4(position.xy,0.,1.);}',
      fragmentShader:`uniform sampler2D source;uniform vec2 size;uniform vec2 outputSize;varying vec2 uv0;
      void main(){vec2 lo=floor(uv0*outputSize)*size/outputSize;vec2 hi=lo+size/outputSize;vec2 base=floor(lo);vec4 color=vec4(0.);float weight=0.;
      for(int y=0;y<3;y++){for(int x=0;x<3;x++){vec2 p=base+vec2(float(x),float(y));vec2 overlap=max(vec2(0.),min(p+1.,hi)-max(p,lo));float w=overlap.x*overlap.y;color+=texture2D(source,(p+.5)/size)*w;weight+=w;}}
      gl_FragColor=color/max(weight,.00001);}`});
    const blit=new THREE.ShaderMaterial({uniforms:{source:{value:low.texture}},vertexShader:vertex,
      fragmentShader:'uniform sampler2D source;varying vec2 uv0;void main(){gl_FragColor=texture2D(source,uv0);}',depthTest:false,depthWrite:false,toneMapped:false});
    const mesh=new THREE.Mesh<THREE.PlaneGeometry,THREE.Material>(quad,box);screen.add(mesh);
    return {high,color,low,tone,box,blit,mesh,screen,eye,quad};
  },[]);
  useEffect(()=>()=>{post.high.dispose();post.color.dispose();post.low.dispose();post.tone.dispose();post.box.dispose();post.blit.dispose();post.quad.dispose();},[post]);
  useEffect(()=>{
    const previous=gl.debug.onShaderError;
    const failed:NonNullable<typeof gl.debug.onShaderError>=(context,program,vertex,fragment)=>{
      postFailed.current=true;
      console.error('ARENA_FPV_SHADER',context.getProgramInfoLog(program),context.getShaderInfoLog(fragment));
      previous?.(context,program,vertex,fragment);
    };
    gl.debug.onShaderError=failed;
    return()=>{if(gl.debug.onShaderError===failed)gl.debug.onShaderError=previous;};
  },[gl]);
  useEffect(()=>{
    const perspective=camera as THREE.PerspectiveCamera;
    perspective.near=.025;
    perspective.fov=meta.vertical_fov_deg;
    perspective.updateProjectionMatrix();
    const lambda=Math.max(.5,Math.min(1,scale));
    const w=Math.max(1,Math.round(1920*lambda)),h=Math.max(1,Math.round(1080*lambda));
    post.low.setSize(w,h);post.box.uniforms.outputSize.value.set(w,h);
  },[camera,meta.vertical_fov_deg,scale,post]);
  useFrame(({scene})=>{
    if(gl.getContext().isContextLost())return;
    const a=previous.current,b=current.current,t=feedAlpha(feed.current);
    origin.set(a.origin[0]+(b.origin[0]-a.origin[0])*t-width/2,a.origin[2]+(b.origin[2]-a.origin[2])*t,a.origin[1]+(b.origin[1]-a.origin[1])*t-height/2);
    direction.set(a.direction[0]+(b.direction[0]-a.direction[0])*t,a.direction[2]+(b.direction[2]-a.direction[2])*t,a.direction[1]+(b.direction[1]-a.direction[1])*t);
    if(b.up){const up=a.up||b.up;camera.up.set(up[0]+(b.up[0]-up[0])*t,up[2]+(b.up[2]-up[2])*t,up[1]+(b.up[1]-up[1])*t).normalize();}
    camera.position.copy(origin);camera.lookAt(origin.add(direction));
    const slot=Math.floor(performance.now()/1000*30);
    if(shadowSlot.current!==slot){shadowSlot.current=slot;gl.shadowMap.needsUpdate=true;}
    const aspect=(camera as THREE.PerspectiveCamera).aspect;
    // Fit both host FOV limits rather than exposing extra pixels on wide displays.
    (camera as THREE.PerspectiveCamera).fov=Math.min(b.vertical_fov_deg,THREE.MathUtils.radToDeg(2*Math.atan(Math.tan(THREE.MathUtils.degToRad(b.horizontal_fov_deg)/2)/aspect)));
    camera.updateProjectionMatrix();
    // Some Qt/driver combinations expose WebGL 2 without a usable RGBA16F
    // framebuffer. Keep the host camera and render directly if postprocessing
    // cannot run, instead of reporting frames from an empty render target.
    if(!postFailed.current){
      gl.setRenderTarget(post.high);
      if(!targetChecked.current){
        targetChecked.current=true;
        const context=gl.getContext();
        if(context.checkFramebufferStatus(context.FRAMEBUFFER)!==context.FRAMEBUFFER_COMPLETE){
          postFailed.current=true;console.warn('ARENA_FPV_FALLBACK: floating-point framebuffer unavailable');
        }
      }
    }
    if(postFailed.current){
      gl.getSize(outputSize);
      const pixelRatio=Math.min(1920/Math.max(1,outputSize.x),1080/Math.max(1,outputSize.y))*Math.max(.5,Math.min(1,scale));
      if(Math.abs(gl.getPixelRatio()-pixelRatio)>.001)gl.setPixelRatio(pixelRatio);
      gl.setRenderTarget(null);gl.render(scene,camera);onRender();return;
    }
    gl.render(scene,camera);
    post.tone.uniforms.toneMappingExposure.value=gl.toneMappingExposure;
    post.mesh.material=post.tone;gl.setRenderTarget(post.color);gl.render(post.screen,post.eye);
    post.mesh.material=post.box;gl.setRenderTarget(post.low);gl.render(post.screen,post.eye);
    post.mesh.material=post.blit;gl.setRenderTarget(null);gl.render(post.screen,post.eye);
    if(!pixelsChecked.current){
      pixelsChecked.current=true;const context=gl.getContext();let visible=false;
      for(const x of [.25,.5,.75]){context.readPixels(Math.floor(context.drawingBufferWidth*x),Math.floor(context.drawingBufferHeight*.5),1,1,context.RGBA,context.UNSIGNED_BYTE,sample);visible ||= sample[0]+sample[1]+sample[2]>0;}
      if(!visible){postFailed.current=true;console.warn('ARENA_FPV_FALLBACK: postprocessing produced an empty image');}
    }
    onRender();
  },1);
  return null;
}
