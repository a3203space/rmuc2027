import {useCallback,useEffect,useRef,useState} from 'react';
import {createRoot} from 'react-dom/client';
import {ArenaScene} from '../../../packages/renderer/src/ArenaScene';
import type {ArenaBridge,LanPlayer} from '../../../packages/renderer/src/lanTypes';
import type {Frame,RenderFeed} from '../../../packages/renderer/src/types';
import './lan.css';

function PlayerViewport(){
  const [player,setPlayer]=useState<LanPlayer|null>(null),[frame,setFrame]=useState<Frame|null>(null);
  const feed=useRef<RenderFeed>({frame:null,previous:null,received:0,speed:1,paused:false,events:[],generation:'',replay:false});
  const bridge=useRef<ArenaBridge|null>(null),staticFrame=useRef<Partial<Frame>>({});
  const state=useRef({yaw:0,pitch:0,forward:0,strafe:0,fire:false,boost:false,jump:false,crouch:false,active:false});
  const keys=useRef(new Set<string>()),synced=useRef(''),view=useRef<HTMLDivElement>(null);
  const rendering=useRef({at:performance.now(),frames:0,tick:0});
  const rendered=useCallback(()=>{
    const stats=rendering.current,now=performance.now();
    if(!stats.frames&&now-stats.at>1500)stats.at=now;
    stats.frames++;
    if(now-stats.at>=1000){bridge.current?.renderStats(JSON.stringify({fps:stats.frames*1000/(now-stats.at),tick:stats.tick}));stats.at=now;stats.frames=0;}
  },[]);
  useEffect(()=>{
    if(!window.qt||!window.QWebChannel)return;
    new window.QWebChannel(window.qt.webChannelTransport,channel=>{
      bridge.current=channel.objects.arena;
      bridge.current.snapshot.connect(payload=>{
        const snapshot=JSON.parse(payload) as {player:LanPlayer|null;static?:Partial<Frame>;room:{match_id:string;status:string}};
        if(snapshot.static)staticFrame.current=snapshot.static;
        const p=snapshot.player;
        if(p)rendering.current.tick=p.tick;
        setPlayer(p);
        if(!p){document.exitPointerLock();keys.current.clear();Object.assign(state.current,{forward:0,strafe:0,fire:false,active:false});setFrame(null);return;}
        if(p.scene?.status==='finished'){document.exitPointerLock();keys.current.clear();state.current.active=false;}
        if(p.display.ui_blocked||p.display.camera_off||!p.scene){setFrame(null);return;}
        if(synced.current!==p.match_id){synced.current=p.match_id;state.current.yaw=p.self.aim;state.current.pitch=p.self.pitch;feed.current.frame=null;feed.current.previous=null;}
        const next={...staticFrame.current,...p.scene} as Frame;
        feed.current={...feed.current,previous:feed.current.frame,frame:next,received:performance.now(),generation:p.match_id};
        setFrame(next);
      });
      bridge.current.viewportReady();
    });
    const neutral=()=>{keys.current.clear();Object.assign(state.current,{forward:0,strafe:0,fire:false,boost:false,jump:false,crouch:false,active:false});bridge.current?.input(JSON.stringify(state.current));};
    const key=(event:KeyboardEvent)=>{
      if(event.code==='Escape'){neutral();document.exitPointerLock();bridge.current?.pause();return;}
      if(!document.pointerLockElement)return;
      if(['KeyW','KeyS','KeyA','KeyD','ShiftLeft','ShiftRight','Space','ControlLeft','ControlRight'].includes(event.code)){
        event.preventDefault();if(event.type==='keydown')keys.current.add(event.code);else keys.current.delete(event.code);
      }
    };
    const mouse=(event:MouseEvent)=>{
      if(!document.pointerLockElement)return;
      if(event.type==='mousemove'){
        const sensitivity=Number(localStorage.getItem('arena-sensitivity')||.002);
        state.current.yaw=(state.current.yaw+event.movementX*sensitivity)%(Math.PI*2);
        state.current.pitch=Math.max(-1.5,Math.min(1.5,state.current.pitch-event.movementY*sensitivity));
      }else if(event.button===0)state.current.fire=event.type==='mousedown';
    };
    const lock=()=>{state.current.active=!!document.pointerLockElement;document.body.classList.toggle('captured',state.current.active);if(!state.current.active)neutral();};
    window.addEventListener('keydown',key);window.addEventListener('keyup',key);
    window.addEventListener('mousemove',mouse);window.addEventListener('mousedown',mouse);window.addEventListener('mouseup',mouse);
    window.addEventListener('blur',neutral);document.addEventListener('pointerlockchange',lock);
    const timer=setInterval(()=>{
      const k=keys.current,s=state.current;
      s.forward=Number(k.has('KeyW'))-Number(k.has('KeyS'));s.strafe=Number(k.has('KeyD'))-Number(k.has('KeyA'));
      s.boost=k.has('ShiftLeft')||k.has('ShiftRight');s.jump=k.has('Space');s.crouch=k.has('ControlLeft')||k.has('ControlRight');
      bridge.current?.input(JSON.stringify(s));
    },20);
    return()=>{clearInterval(timer);neutral();window.removeEventListener('keydown',key);window.removeEventListener('keyup',key);window.removeEventListener('mousemove',mouse);window.removeEventListener('mousedown',mouse);window.removeEventListener('mouseup',mouse);window.removeEventListener('blur',neutral);document.removeEventListener('pointerlockchange',lock);};
  },[]);
  const reason=player?.display.reason;
  return <div className="player-viewport" ref={view} onClick={()=>{if(player?.scene?.status==='running'&&!player.display.ui_blocked&&!player.control?.needs_resume)view.current?.requestPointerLock();}}>
    {frame&&player?.display.camera?<ArenaScene frame={frame} feed={feed} mode="follow" selected={player.self.id} playerCamera={player.display.camera} resolutionScale={player.display.resolution_scale} onSelect={()=>{}} onCanvas={()=>{}} onRender={rendered}/>:
      <div className="camera-status">{reason==='overheat'?'发射机构过热 · 图传已断开':reason==='destroyed'?'机器人战亡 · 等待复活':reason==='hangar'?'机库指挥席 · 使用战术地图与空中系统面板':reason==='red_card'?'机器人已被罚下':reason==='penalty'?'判罚遮挡':'等待仿真画面'}</div>}
    {frame&&<div className="capture-hint">点击画面接管鼠标 · Esc 释放操作</div>}
  </div>;
}
createRoot(document.getElementById('root')!).render(<PlayerViewport/>);
