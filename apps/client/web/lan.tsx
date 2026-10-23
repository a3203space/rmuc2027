import {Component,useCallback,useEffect,useRef,useState,type ReactNode} from 'react';
import {createRoot} from 'react-dom/client';
import {ArenaScene} from '../../../packages/renderer/src/ArenaScene';
import type {AppSettings,ArenaBridge,LanPlayer,LanSnapshot,Receipt,RoomState,RuntimeCatalog,PerformanceProfile,ProfileResult} from '../../../packages/renderer/src/lanTypes';
import type {Arena,Frame,RenderFeed} from '../../../packages/renderer/src/types';
import {PlayerTimeline} from '../../../packages/renderer/src/playerTimeline';
import {Results} from './Results';
import {combatButton,createCombatInput,releaseCombat} from './combatInput';
import {OperatorHud} from './OperatorHud';
import {StartFlash} from './arena/ceremony';
import {useOperatorViewport} from './operatorViewport';
import {cameraLayerClass,operatorControlAllowed} from './operatorDisplay';
import {useProfileSave} from './useProfileSave';
import {newerRoom} from './performance';
import {ArenaShell,DEFAULT_SETTINGS} from './ArenaShell';
import {Ceremony,introductionShot,useStageClock,useArenaMusic} from './Ceremony';
import './lan.css';
class ViewportBoundary extends Component<{children:ReactNode;failed:(message:string)=>void},{error:string}>{
 state={error:''};static getDerivedStateFromError(e:Error){return {error:e.message};}
 componentDidCatch(e:Error){this.props.failed(e.message);}
 render(){return this.state.error?<div className="camera-status">三维画面加载失败：{this.state.error}</div>:this.props.children;}
}
const noOp=()=>{},preMatch=['starting','loading','intro','countdown'];
function PlayerViewport(){
 const [graphicsError]=useState(()=>{try{const gl=document.createElement('canvas').getContext('webgl2');if(!gl)return '无法创建 WebGL 2 上下文 · 请启用 OpenGL 渲染';gl.getExtension('WEBGL_lose_context')?.loseContext();return '';}catch(e){return String(e);}});
 const [player,setPlayer]=useState<LanPlayer|null>(null),[frame,setFrame]=useState<Frame|null>(null),[arena,setArena]=useState<Arena>();
 const [room,setRoom]=useState<RoomState|null>(null),[playerId,setPlayerId]=useState(''),[roomOpen,setRoomOpen]=useState(false);
 const [settings,setSettings]=useState<AppSettings>(()=>{try{return {...DEFAULT_SETTINGS,...JSON.parse(localStorage.getItem('arena-settings')||'{}')};}catch{return DEFAULT_SETTINGS;}});
 const [surface,setSurface]=useState('home'),[resultSurface,setResultSurface]=useState('victory'),[startFlash,setStartFlash]=useState(false),[inspected,setInspected]=useState('');
 const surfaceRef=useRef(surface);surfaceRef.current=surface;
 const [catalog,setCatalog]=useState<RuntimeCatalog>({models:{},strategies:[]});
 const [ammoMenu,setAmmoMenu]=useState(false),[aimHeld,setAimHeld]=useState(false);
 const [menu,setMenu]=useState(false),[expanded,setExpanded]=useState(false),[stale,setStale]=useState(false),[notice,setNotice]=useState(''),[bridgeError,setBridgeError]=useState('');
 const [renderProblem,setRenderProblem]=useState(''),[viewportVersion,setViewportVersion]=useState(0);
 const profileOwner=room?.players?.find(p=>p.id===playerId);
 const {state:profileSave,begin:beginProfile,complete:completeProfile,isPending:profilePending}=useProfileSave(`${playerId}:${profileOwner?.team}:${profileOwner?.slot}`,!stale&&!!room&&['lobby','finished'].includes(room.status));
 const roomCache=useRef<RoomState|null>(null);
 const updateRoom=useCallback((next:RoomState|null)=>{const value=newerRoom(roomCache.current,next);roomCache.current=value;setRoom(value);},[]);
 const feed=useRef<RenderFeed>({frame:null,previous:null,received:0,speed:1,paused:false,events:[],generation:'',replay:false});
 const timeline=useRef(new PlayerTimeline()).current;
 const resetFeed=useCallback((freeze=false)=>{timeline.clear();feed.current.timeline=undefined;feed.current.events=[];if(!freeze)feed.current.playerSample=undefined;},[timeline]);
 const catalogRevision=useRef(-1);
 const bridge=useRef<ArenaBridge|null>(null),staticFrame=useRef<Partial<Frame>>({}),staticMatch=useRef('');
 const state=useRef({yaw:0,pitch:0,forward:0,strafe:0,...createCombatInput(),boost:false,jump:false,crouch:false,active:false,transform:null as boolean|null});
 const keys=useRef(new Set<string>()),synced=useRef(''),view=useRef<HTMLDivElement>(null),menuRef=useRef(false),allowed=useRef(false),received=useRef(0),online=useRef(false),self=useRef<LanPlayer['self']|null>(null),phaseRef=useRef('');
 const operatorViewport=useOperatorViewport(view);
 const receipts=useRef(new Set<number>()),events=useRef(new Set<string>()),menuPolicy=useRef(''),loadedMatch=useRef('');
 const rendering=useRef({at:performance.now(),frames:0,tick:0,last:0});
 const isPresentation=preMatch.includes(room?.status||''),elapsed=useStageClock(room),shot=isPresentation?introductionShot(frame,room,elapsed):null;
 const soundPhase=room?.status==='finished'?resultSurface:surface==='hangar'?'hangar':surface==='home'?'home':!room||roomOpen||room.status==='lobby'?(room?'lobby':'home'):room.status;
 useArenaMusic(soundPhase,settings.volume??35,elapsed);
 useEffect(()=>{if(room?.status!=='running'){setStartFlash(false);return;}setStartFlash(true);const timer=setTimeout(()=>setStartFlash(false),1200);return()=>clearTimeout(timer);},[room?.status,room?.match_id]);
 const invoke=useCallback((name:string,body:Record<string,unknown>={})=>{
  if(!bridge.current){setNotice('客户端数据通道未连接，请稍后重试');return;}
  if(profilePending()&&['ready','seat','start','seat-config','team-config'].includes(name)){setNotice('请等待性能保存完成后再操作');return;}
  if(name==='profile'){
   const id=beginProfile(body as unknown as PerformanceProfile);if(!id)return;
   body={...body,request_id:id};
  }
  bridge.current.command(JSON.stringify({name,body}));
 },[beginProfile,profilePending]);
 const updateSettings=useCallback((value:AppSettings)=>{setSettings(value);invoke('settings',{...value});},[invoke]);
 const neutral=useCallback(()=>{keys.current.clear();releaseCombat(state.current);setAimHeld(false);Object.assign(state.current,{forward:0,strafe:0,fire:false,boost:false,jump:false,crouch:false,active:false,transform:null});bridge.current?.input(JSON.stringify(state.current));},[]);
 const inspect=useCallback((id:string)=>{neutral();document.exitPointerLock();setInspected(id);},[neutral]);
 useEffect(()=>{if(surface!=='room'){neutral();document.exitPointerLock();}},[surface,neutral]);
 const showMenu=useCallback((open:boolean,ammoOnly=false)=>{setAmmoMenu(open&&ammoOnly);menuRef.current=open;setMenu(open);if(open){neutral();document.exitPointerLock();bridge.current?.pause();}},[neutral]);
 const command=useCallback((name:string,value='')=>{
  if(profilePending()&&['ready','start'].includes(name)){setNotice('请等待性能保存完成后再操作');return;}
  neutral();
  if(name==='buy-ammo'){showMenu(true,true);return;}
  if(name==='room'){setRoomOpen(true);showMenu(true);return;}
  if(name==='transform'){state.current.transform=value==='fold';return;}
  if(name==='ready'){const me=room?.players?.find(p=>p.id===playerId);invoke('ready',{ready:!me?.ready});return;}
  bridge.current?.command(JSON.stringify({name,value}));
  if(name==='resume')showMenu(false);
  else if(name==='leave'){resetFeed();setPlayer(null);setFrame(null);updateRoom(null);setArena(undefined);staticFrame.current={};staticMatch.current='';synced.current='';setRoomOpen(false);}
  else setNotice('请求已发送，等待主机处理');
 },[neutral,showMenu,invoke,room,playerId,updateRoom,profilePending,resetFeed]);
 const rendered=useCallback(()=>{const s=rendering.current,now=performance.now();s.frames++;s.last=now;if(now-s.at>=1000){let visibleSamples=0;const gl=view.current?.querySelector('canvas')?.getContext('webgl2');if(gl&&!gl.isContextLost()){const pixel=new Uint8Array(4);for(const x of [.25,.5,.75]){gl.readPixels(Math.floor(gl.drawingBufferWidth*x),Math.floor(gl.drawingBufferHeight*.5),1,1,gl.RGBA,gl.UNSIGNED_BYTE,pixel);if(pixel[0]+pixel[1]+pixel[2]>0)visibleSamples++;}}bridge.current?.renderStats(JSON.stringify({fps:s.frames*1000/(now-s.at),tick:s.tick,phase:phaseRef.current,visible_samples:visibleSamples,player_timeline:feed.current.timeline?.diagnostics}));s.at=now;s.frames=0;}},[]);
 const sceneFailed=useCallback((error:string)=>{setRenderProblem(error);console.error('ARENA_VIEWPORT',error);if(room?.status==='loading')invoke('stage-ready',{match_id:room.match_id,error:error.slice(0,300)});},[room?.status,room?.match_id,invoke]);
 const assetsReady=useCallback(()=>{if(room?.status==='loading'&&loadedMatch.current!==room.match_id&&rendering.current.last>0){loadedMatch.current=room.match_id||'';invoke('stage-ready',{match_id:room.match_id});}},[room?.status,room?.match_id,invoke]);
 useEffect(()=>{if(graphicsError&&room?.status==='loading')invoke('stage-ready',{match_id:room.match_id,error:graphicsError.slice(0,300)});},[graphicsError,room?.status,room?.match_id,invoke]);
 const wantsCanvas=!!frame&&!graphicsError&&(isPresentation||!!player?.display.camera);
 const cameraState=JSON.stringify({phase:room?.status,player:!!player,arena:!!arena,scene:!!frame,camera:!!player?.display.camera,reason:player?.display.reason||'',graphicsError});
 useEffect(()=>{console.info('ARENA_CAMERA_STATE',cameraState);},[cameraState]);
 useEffect(()=>{localStorage.setItem('arena-settings',JSON.stringify(settings));localStorage.setItem('arena-sensitivity',String((settings.sensitivity||20)*.0001));},[settings]);
 useEffect(()=>{setRenderProblem('');if(!wantsCanvas)return;const started=performance.now();let previous='';const timer=setInterval(()=>{const canvas=view.current?.querySelector('canvas'),b=canvas?.getBoundingClientRect();const fresh=rendering.current.last>=started&&performance.now()-rendering.current.last<3000;const problem=fresh?'':!b?.width||!b?.height?'三维画布尚未获得有效尺寸':'三维渲染尚未输出画面';if(problem!==previous){previous=problem;console.warn('ARENA_VIEWPORT_STATE',JSON.stringify({problem,width:b?.width||0,height:b?.height||0,tick:rendering.current.tick}));}setRenderProblem(problem);},3000);return()=>clearInterval(timer);},[wantsCanvas,viewportVersion,isPresentation]);
 useEffect(()=>{
  let disposed=false,connected=false;
  const receive=(payload:string)=>{
   const snapshot=JSON.parse(payload) as LanSnapshot;
   if(snapshot.transport===false){online.current=false;setStale(true);resetFeed(true);neutral();document.exitPointerLock();return;}
   if(!('player' in snapshot))return;
   if(snapshot.room?.status==='closed'){resetFeed();updateRoom(null);setPlayer(null);setFrame(null);allowed.current=false;neutral();setNotice('房间已关闭');return;}
   received.current=performance.now();online.current=true;setStale(false);
   if('room' in snapshot)updateRoom(snapshot.room||null);if(snapshot.player_id)setPlayerId(snapshot.player_id);
   if(snapshot.room?.catalog_revision!==undefined&&catalogRevision.current!==snapshot.room.catalog_revision){catalogRevision.current=snapshot.room.catalog_revision;invoke('catalog');}
   const p=snapshot.player;self.current=p?.self||null;phaseRef.current=roomCache.current?.status||'';
   const matchId=snapshot.room?.match_id||p?.match_id;
   if(matchId&&staticMatch.current!==matchId){resetFeed();staticMatch.current=matchId;staticFrame.current={};setArena(undefined);loadedMatch.current='';}
   if(p&&synced.current!==p.match_id){synced.current=p.match_id;receipts.current.clear();events.current.clear();setNotice('');feed.current.frame=null;feed.current.previous=null;state.current.yaw=p.self.aim||0;state.current.pitch=p.self.pitch||0;state.current.fire_press_seq=0;}
   if(snapshot.static){staticFrame.current=snapshot.static;setArena(snapshot.static.arena);}
   setPlayer(p);
   const phase=roomCache.current?.status||'';
   if(phase!==menuPolicy.current){menuPolicy.current=phase;showMenu(phase==='finished');setRoomOpen(false);if(phase!=='running'){neutral();document.exitPointerLock();}}
   if(preMatch.includes(phase)&&snapshot.presentation){
    resetFeed();
    allowed.current=false;const next={...staticFrame.current,...snapshot.presentation} as Frame;
    if(next.arena){feed.current={...feed.current,frame:next,previous:null,received:performance.now(),generation:matchId||'',paused:true};setFrame(next);}return;
   }
   if(!p){resetFeed();allowed.current=false;neutral();document.exitPointerLock();setFrame(null);return;}
   rendering.current.tick=p.tick;
   allowed.current=operatorControlAllowed(phase,p);
   if(!allowed.current){neutral();document.exitPointerLock();}
   const next={...staticFrame.current,...p.scene} as Frame;
   if(p.display.ui_blocked||p.display.camera_off||!next.arena||!p.scene||!p.display.camera){resetFeed();setFrame(null);}
   else if(timeline.push(next,p.display.camera,p.self.id,p.match_id)){
    // Only the render sampler advances scene/camera poses. HUD and receipts remain live.
    feed.current={...feed.current,timeline,generation:p.match_id,paused:false};setFrame(timeline.sceneFrame());
   }
   for(const r of snapshot.operations||[]){if(receipts.current.has(r.seq))continue;receipts.current.add(r.seq);setNotice(receiptText(r));}
   for(const e of p.events||[]){const id=e.id||`${e.tick}:${e.type}:${e.amount}`;if(events.current.has(id))continue;events.current.add(id);setNotice(e.type==='ammo_purchase'?`已购买 ${e.amount} 发，花费 ${e.cost} 金币${e.remote?'，6 秒后送达':'，已到账'}`:e.type==='ammo_delivered'?`弹丸已送达：${e.amount} 发`:e.type==='remote_heal'?'远程回血已生效':e.type==='respawn'?'已复活':e.message||'操作未执行');}
   if(receipts.current.size>512)receipts.current=new Set([...receipts.current].slice(-256));if(events.current.size>512)events.current=new Set([...events.current].slice(-256));
  };
  const connect=()=>{if(connected||disposed||!window.qt||!window.QWebChannel)return;connected=true;new window.QWebChannel(window.qt.webChannelTransport,channel=>{if(disposed)return;const b=bridge.current=channel.objects.arena;b.snapshot.connect(receive);b.notice.connect(setNotice);b.room.connect(payload=>{const r=JSON.parse(payload);updateRoom(r);phaseRef.current=roomCache.current?.status||'';});b.response?.connect((name,payload)=>{const value=JSON.parse(payload);if(name==='catalog'){setCatalog(value);catalogRevision.current=value.revision??catalogRevision.current;}if(name==='settings')setSettings(s=>({...s,...value}));if(name==='profile-result')completeProfile(value as ProfileResult);if(name==='room'){setRoomOpen(true);showMenu(true);}});b.viewportReady();setBridgeError('');});};
  connect();const connectTimer=setInterval(connect,100),deadline=setTimeout(()=>{if(!bridge.current)setBridgeError('客户端数据通道尚未建立，请重新启动客户端');},5000);
  const key=(event:KeyboardEvent)=>{
   if((event.target instanceof HTMLInputElement)||(event.target instanceof HTMLSelectElement)||(event.target instanceof HTMLTextAreaElement))return;
   if(surfaceRef.current!=='room')return;
   if(event.code==='Escape'&&event.type==='keydown'&&phaseRef.current==='running'){event.preventDefault();showMenu(true);return;}
   if(event.code==='KeyB'&&event.type==='keydown'&&!event.repeat&&phaseRef.current==='running'&&self.current?.role!=='aerial'){event.preventDefault();showMenu(true,true);return;}
   if(event.code==='Tab'&&phaseRef.current==='running'){event.preventDefault();setExpanded(event.type==='keydown');return;}
   if(surfaceRef.current!=='room'||!document.pointerLockElement||menuRef.current||!allowed.current)return;
   if(event.code==='KeyF'&&event.type==='keydown'&&!event.repeat&&self.current?.transform_allowed){event.preventDefault();state.current.transform=state.current.transform===null?(self.current.transform_progress||0)<.5:!state.current.transform;}
   if(['KeyW','KeyS','KeyA','KeyD','ShiftLeft','ShiftRight','Space','ControlLeft','ControlRight'].includes(event.code)){event.preventDefault();if(event.type==='keydown')keys.current.add(event.code);else keys.current.delete(event.code);}
  };
  const sendCombat=()=>{bridge.current?.input(JSON.stringify(state.current));};
  const mouse=(event:MouseEvent)=>{
   // Releases must be handled even after capture or eligibility is lost.
   if(event.type==='mouseup'&&(event.button===0||event.button===2)){
    combatButton(state.current,event.button,false);if(event.button===2)setAimHeld(false);sendCombat();return;
   }
   if(surfaceRef.current!=='room'||!document.pointerLockElement||menuRef.current||!allowed.current)return;
   if(event.type==='mousemove'){
    // Recover from a missed mouse-up using the browser's current button mask.
    if(state.current.fire&&!(event.buttons&1))combatButton(state.current,0,false);
    if(state.current.aim_assist&&!(event.buttons&2)){combatButton(state.current,2,false);setAimHeld(false);sendCombat();}
    const sensitivity=Number(localStorage.getItem('arena-sensitivity')||.002);
    state.current.yaw=(state.current.yaw+event.movementX*sensitivity)%(Math.PI*2);
    state.current.pitch=Math.max(-1.5,Math.min(1.5,state.current.pitch-event.movementY*sensitivity));
   }else if(event.button===0||event.button===2){
    event.preventDefault();combatButton(state.current,event.button,true);
    if(event.button===2)setAimHeld(true);sendCombat();
   }
  };
  const context=(event:MouseEvent)=>{if(document.pointerLockElement||view.current?.querySelector('.camera-layer')?.contains(event.target as Node))event.preventDefault();};
  const lock=()=>{state.current.active=!!document.pointerLockElement;document.body.classList.toggle('captured',state.current.active);if(!state.current.active)neutral();};const blur=()=>{neutral();setExpanded(false);};
  window.addEventListener('keydown',key);window.addEventListener('keyup',key);window.addEventListener('mousemove',mouse);window.addEventListener('mousedown',mouse);window.addEventListener('mouseup',mouse);window.addEventListener('blur',blur);window.addEventListener('contextmenu',context);document.addEventListener('pointerlockchange',lock);
  const timer=setInterval(()=>{if(received.current&&performance.now()-received.current>2500){if(online.current){online.current=false;resetFeed(true);neutral();document.exitPointerLock();}setStale(true);}const k=keys.current,s=state.current;if(surfaceRef.current!=='room'||!online.current||menuRef.current||!allowed.current){s.active=false;releaseCombat(s);setAimHeld(false);k.clear();}s.forward=Number(k.has('KeyW'))-Number(k.has('KeyS'));s.strafe=Number(k.has('KeyD'))-Number(k.has('KeyA'));s.boost=k.has('ShiftLeft')||k.has('ShiftRight');s.jump=k.has('Space');s.crouch=k.has('ControlLeft')||k.has('ControlRight');if(phaseRef.current==='running')bridge.current?.input(JSON.stringify(s));},20);
  return()=>{disposed=true;clearInterval(timer);clearInterval(connectTimer);clearTimeout(deadline);neutral();window.removeEventListener('keydown',key);window.removeEventListener('keyup',key);window.removeEventListener('mousemove',mouse);window.removeEventListener('mousedown',mouse);window.removeEventListener('mouseup',mouse);window.removeEventListener('blur',blur);window.removeEventListener('contextmenu',context);document.removeEventListener('pointerlockchange',lock);};
 },[neutral,showMenu,invoke,updateRoom,completeProfile,resetFeed,timeline]);
 const reason=player?.display.reason;
 const cameraText=bridgeError||graphicsError||({overheat:'发射机构过热 · 图传已断开',destroyed:'机器人战亡 · 等待复活',hangar:'机库指挥席 · 通过战术地图与操作菜单指挥无人机',red_card:'机器人已被罚下',penalty:'判罚遮挡'} as Record<string,string>)[reason||'']||(player&&!arena?'等待主机场地数据':player&&!player.display.camera?'等待主机相机数据':'等待仿真画面');
 const me=room?.players?.find(p=>p.id===playerId),isHost=!!me?.host;
 const policyAlert=Object.entries(room?.policy_status||{}).filter(([,status])=>status!=='就绪').slice(0,2).map(([seat,status])=>`${seat.startsWith('red')?'红方':'蓝方'} ${seat.split(':')[1]} 号 AI：${status}`).join(' · ');
 return <ArenaShell onSurface={setSurface} room={room} catalog={catalog} playerId={playerId} profileSave={profileSave} offline={stale} settings={settings} onSettings={updateSettings} invoke={invoke} notice={room?.error||policyAlert||notice||bridgeError} visible={!room||room.status==='lobby'||roomOpen} onClose={()=>{setRoomOpen(false);showMenu(false);}} player={player} arena={arena} presentation={isPresentation?frame:null} inspected={inspected} onInspect={inspect} viewport={
 <div className="player-viewport" ref={view}>
  <div className="operator-surface" style={{left:operatorViewport.left,top:operatorViewport.top,width:operatorViewport.width,height:operatorViewport.height}}>
   <div className={cameraLayerClass(player?.display,isPresentation)} onClick={()=>{if(surfaceRef.current==='room'&&allowed.current&&!stale&&!menu&&!graphicsError)view.current?.requestPointerLock();}}>{wantsCanvas&&frame?<ViewportBoundary key={`${room?.match_id}:${isPresentation?'intro':'fpv'}:${viewportVersion}`} failed={sceneFailed}><ArenaScene frame={frame} feed={feed} mode="orbit" selected={isPresentation?'':player?.self.id||''} playerCamera={isPresentation?undefined:player?.display.camera||undefined} shot={shot} resolutionScale={isPresentation?1:player?.display.resolution_scale} onSelect={noOp} onCanvas={noOp} onRender={rendered} onAssetsReady={assetsReady} onSceneError={sceneFailed}/></ViewportBoundary>:<div className="camera-status">{room?.status==='lobby'?'选择席位并准备，房主开始比赛后加载战场。':cameraText}</div>}</div>
   {!isPresentation&&room?.status!=='finished'&&player&&<OperatorHud player={player} arena={arena} room={room} stale={stale} expanded={expanded} menu={menu&&!roomOpen} onMenu={showMenu} command={command} notice={notice||policyAlert} isHost={isHost} ammoOnly={ammoMenu} aimHeld={aimHeld} scale={operatorViewport.scale} catalog={catalog} onInspect={inspect} volume={settings.volume??35} audible={surface==='room'&&!stale}/>}
   {isPresentation&&room&&frame&&<Ceremony frame={frame} room={room} catalog={catalog} elapsed={elapsed} isHost={isHost} skip={()=>invoke('skip-intro')}/>}
   {room?.status==='finished'&&<Results room={room} isHost={isHost} ready={!!me?.ready} configBusy={!!profileSave.pending||stale} command={command} catalog={catalog} arena={arena} onSurface={setResultSurface}/>}
   {startFlash&&<StartFlash/>}
  </div>
  {wantsCanvas&&renderProblem&&!player?.display.ui_blocked&&<div className="viewport-recovery">{renderProblem}<button onClick={()=>{neutral();document.exitPointerLock();setViewportVersion(v=>v+1);}}>重新加载画面</button></div>}
  {frame&&!isPresentation&&!graphicsError&&!menu&&!player?.display.ui_blocked&&player?.self.alive&&<div className="capture-hint">点击画面捕获鼠标 · 左键射击 · 右键按住自瞄 · B 购弹 · Esc 菜单</div>}
  {isPresentation&&!frame&&<div className="prepare-wait">{cameraText}</div>}
 </div>}/>;

}
function receiptText(r:Receipt){const name=({ammo:'补弹',heal:'回血',revive:'复活',pick:'拾取',assemble:'装配',install:'安装',posture:'姿态',resume:'恢复控制',cancel:'取消辅助',air_target:'空中目标',air_launch:'起飞',air_return:'召回'} as Record<string,string>)[r.name||'']||'操作';return r.status==='rejected'?`${name}未执行：${r.message}`:r.status==='applied'?`${name}已交给控制器，实际效果以主机状态为准`:name+'请求已接收';}
createRoot(document.getElementById('root')!).render(<PlayerViewport/>);
