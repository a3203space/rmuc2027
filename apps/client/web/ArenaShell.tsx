// Page frame and arena columns ported from Arena-1 App.tsx; callbacks use the LAN bridge.
import {useCallback,useEffect,useMemo,useRef,useState,type ReactNode} from 'react';
import {Activity,Bot,ChevronRight,Crosshair,Home as HomeIcon,LibraryBig,Maximize2,Minimize2,Play,Settings2,Shield,Swords,Wifi,X} from 'lucide-react';
import type {AppSettings,RoomState,RuntimeCatalog,ProfileSaveState,LanPlayer} from '../../../packages/renderer/src/lanTypes';
import type {Arena,Frame as SimulationFrame,TeamId} from '../../../packages/renderer/src/types';
import {EMPTY_PROFILE_SAVE} from './performance';
import {clock,number,roles,side} from './hudModel';
import {Hangar} from './arena/Hangar';
import {Home} from './arena/Home';
import {AudioControls} from './arena/AudioControls';
import {TelemetryPanel} from './arena/TelemetryPanel';
import {StrategyGallery} from './arena/StrategyGallery';
import {compatible} from './arena/configuration';
import {matchMetadata,operatorView,presentationView,resultView} from './arena/viewModel';
import {teamName} from './arena/broadcast';
import './arena/style.css';
import './arena/interaction.css';
import './arena/lan-integration.css';
export {compatible} from './arena/configuration';
export type Invoke=(name:string,body?:Record<string,unknown>)=>void;
export const DEFAULT_SETTINGS:AppSettings={nickname:'玩家',address:'127.0.0.1',port:8766,runtime:'',sensitivity:20,volume:35};
const slots=[1,3,4,6,7],teams:TeamId[]=['red','blue'];
export function ArenaShell({room,catalog,playerId,settings,onSettings,invoke,notice,visible,onClose,onSurface,profileSave=EMPTY_PROFILE_SAVE,offline=false,viewport,player=null,arena,presentation,inspected='',onInspect=()=>{}}:{room:RoomState|null;catalog:RuntimeCatalog;playerId:string;settings:AppSettings;onSettings:(s:AppSettings)=>void;invoke:Invoke;notice:string;visible:boolean;onClose:()=>void;onSurface?:(page:string)=>void;profileSave?:ProfileSaveState;offline?:boolean;viewport?:ReactNode;player?:LanPlayer|null;arena?:Arena;presentation?:SimulationFrame|null;inspected?:string;onInspect?:(id:string)=>void}){
 const [page,setPage]=useState('home'),[active,setActive]=useState<{team:TeamId;slot:number}>({team:'red',slot:3}),[ownerKey,setOwnerKey]=useState(''),[connectOpen,setConnectOpen]=useState(false),[fullscreen,setFullscreen]=useState(false);
 const viewportHost=useRef<HTMLDivElement>(null),me=room?.players?.find(p=>p.id===playerId),host=!!me?.host,editable=!!room&&['lobby','finished'].includes(room.status),saving=!!profileSave.pending;
 const metadata=useMemo(()=>matchMetadata(room,catalog),[room,catalog]);
 const view=room?.status==='finished'?resultView(room,catalog,arena):room&&['starting','loading','intro','countdown'].includes(room.status)&&presentation?presentationView(presentation):operatorView(player,arena,room,catalog);
 const focus=view?.robots.find(r=>r.id===(inspected||player?.self.id)),model=catalog.models[String(room?.seat_configs?.[active.team]?.[String(active.slot)]?.model)];
 useEffect(()=>{setPage(p=>room?(p==='home'?'room':p):'home');if(room)setConnectOpen(false);},[!!room]);
 useEffect(()=>{if(room&&!['lobby','closed'].includes(room.status))setPage('room');},[room?.status,room?.match_id]);
 useEffect(()=>{if(me?.team&&me.slot)setActive({team:me.team,slot:me.slot});},[me?.team,me?.slot]);
 useEffect(()=>{onSurface?.(page);},[page,onSurface]);
 useEffect(()=>{if(visible&&room&&room.status!=='lobby'){setPage('room');}},[visible]);
 const leaveCapture=useCallback(()=>{document.exitPointerLock?.();},[]);
 const navigate=(target:string)=>{leaveCapture();setPage(target);setFullscreen(false);};
 const toggleFullscreen=()=>{leaveCapture();onClose();setFullscreen(value=>!value);};
 useEffect(()=>{if(!fullscreen)return;const escape=(event:KeyboardEvent)=>{if(event.key==='Escape'&&!document.pointerLockElement)setFullscreen(false);};window.addEventListener('keydown',escape);return()=>window.removeEventListener('keydown',escape);},[fullscreen]);
 const update=(key:string,value:string|number)=>onSettings({...settings,[key]:value});
 const inspect=(id:string)=>{leaveCapture();onInspect(id);};
 const catalogEntries=Object.fromEntries(catalog.strategies.map(s=>[s.id,s]));
 const phase=room?.status||'',busy=!!room&&!['lobby','finished'].includes(phase);
 const eventText=(event:{type:string;message?:string;amount?:number})=>event.message||({ammo_purchase:'弹药已购买',ammo_delivered:'弹药已送达',remote_heal:'远程回血',respawn:'机器人复活',rpc_error:'操作未执行'} as Record<string,string>)[event.type]||event.type;
 return <div className="app-shell arena-source" data-page={page}>
 <header className="app-header"><a className="brand" onClick={()=>navigate(room?'room':'home')}><img src="/brand/arena-1.png" alt="Arena-1"/> ARENA-1 <small>STRATEGY LEAGUE</small></a><nav>
 <button className={page==='home'?'active':''} onClick={()=>navigate('home')}><HomeIcon size={14}/>主页</button>
 <button className={page==='room'?'active':''} aria-label="对战大厅" onClick={()=>room?navigate('room'):setConnectOpen(true)}><Swords size={14}/>RM 对战</button>
 {room&&<button className={page==='hangar'?'active':''} onClick={()=>navigate('hangar')}><Bot size={14}/>机器人机库</button>}
 <button className={page==='strategies'?'active':''} onClick={()=>navigate('strategies')}><LibraryBig size={14}/>策略库</button>
 <button className={page==='settings'?'active':''} onClick={()=>navigate('settings')}><Settings2 size={14}/>设置</button></nav>
 <AudioControls sound={(settings.volume??35)>0} volume={(settings.volume??35)/100} onToggle={()=>update('volume',(settings.volume??35)>0?0:35)} onVolume={value=>update('volume',Math.round(value*100))}/>
 <div className={`connection ${room&&!offline?'online':''}`}><span/>{offline?'连接已中断':room?'WebSocket 已连接':'本地客户端'}<b>RMUC 2027</b></div></header>
 {profileSave.status==='error'&&<div className="global-error" role="status"><Activity size={14}/><span>{profileSave.message}</span></div>}
 {page==='home'&&<Home models={catalog.models} catalog={catalogEntries} cards={{}} connected={!!room&&!offline} busy={busy} live={room?{red:'红方',blue:'蓝方'}:null} onTraining={()=>{}} onBattle={()=>room?navigate('room'):setConnectOpen(true)} onHistory={()=>{}} onRankings={()=>{}} onGallery={()=>navigate('strategies')} onWatch={()=>navigate('room')}/>}
 <div className="arena-battle-page" style={{display:page==='room'?'block':'none'}}>
 <div className="page-heading"><div><span className="eyebrow">LIVE SIMULATION</span><h1>RM 对战 · RMUC 2027 · 5V5</h1></div><div className="architecture-tags"><span><Wifi size={12}/>{offline?'已断开':'局域网联机'}</span><span><Crosshair size={12}/>操作手第一视角</span></div></div>
 <main className="arena-layout"><aside className="setup-panel"><div className="panel-heading"><span className="eyebrow">MATCH SETUP</span><span className="tiny-badge">5 V 5</span></div><h2 className="matchup-title">准备上场阵容</h2>
 <button className="submission-button" onClick={()=>navigate('hangar')} disabled={!room}><Bot size={14}/>调整机器人与性能</button><button className="submission-button" onClick={()=>navigate('strategies')}><LibraryBig size={14}/>配置接管策略</button>
 <div className="setup-divider"/><div className="lan-seat-grid">{teams.map(team=><section key={team}><label><i className={`team-dot ${team}`}/>{side(team)}</label>{slots.map(slot=>{const occupant=room?.players?.find(p=>p.team===team&&p.slot===slot),seat=room?.seat_configs?.[team]?.[String(slot)],hardware=catalog.models[String(seat?.model)];return <button key={slot} title={hardware?.name} aria-label={`${side(team)} ${slot} 号席位`} className={occupant?.id===playerId?'active':''} disabled={!editable||saving||offline||!!occupant&&occupant.id!==playerId} onClick={()=>invoke('seat',{team,slot})}><b>{slot}</b><span>{hardware?roles[hardware.role]:'机器人'}<small>{occupant?`${occupant.nickname} · ${occupant.ready?'已准备':'未准备'}`:'AI 补位'}</small></span></button>;})}</section>)}</div>
 <div className="setup-divider"/>{teams.map(team=><label key={team}><i className={`team-dot ${team}`}/>{side(team)}策略<strong className="lan-strategy-label">{room?.team_configs?.[team]?.mode==='team'?catalog.strategies.find(s=>s.id===room.team_configs![team].strategy)?.name:'逐机器人独立策略'}</strong></label>)}
 <label>当前操控席位<strong className="lan-strategy-label">{me?.team&&me.slot?`${side(me.team)} ${me.slot} 号 · ${me.nickname}`:'尚未选择席位'}</strong></label>
 <button className="primary start-button" disabled={!editable||!me?.slot||saving||offline} onClick={()=>invoke('ready',{ready:!me?.ready})}><Shield size={15}/>{me?.ready?'✓ 已准备 · 取消':'准备出战'}</button>
 {host&&<button className="primary start-button" disabled={!editable||saving||offline||!room?.players?.every(p=>p.ready&&p.connected&&p.slot)} onClick={()=>invoke('start')}><Play size={15} fill="currentColor"/>开始比赛 ↗</button>}
 {busy&&<button className="submission-button" onClick={onClose}>返回战斗画面</button>}<button className="submission-button" onClick={()=>invoke('leave')} disabled={saving}>离开房间</button>
 <div className="sdk-note"><Shield size={15}/><div>点击画面捕获鼠标后操控。<span>左键射击 · 右键按住自瞄<br/>B 购弹 · Esc 菜单 · Tab 总览<br/>F 折叠 / 展开</span></div></div></aside>
 <section className="stage-column"><div className="scoreboard">{teams.map(team=><div key={team} className={`team-score ${team}-score`}><div><span className="eyebrow">{team.toUpperCase()} / 攻击伤害</span><h2>{teamName(metadata,team)}</h2><span>{view?`基地 ${number(view.bases[team].hp)} / 前哨 ${number(view.outposts[team].hp)}`:'等待开赛'}</span></div><strong>{number(view?.teams[team].attack_damage)}</strong></div>)}<div className="match-clock"><span>{offline?'连接已中断':({lobby:'等待开赛',loading:'载入场地',intro:'阵容入场',countdown:'即将开始',running:'比赛进行中',finished:'比赛结束'} as Record<string,string>)[phase]||'等待房间'}</span><strong>{clock(view?.remaining??(player?.display.ui_blocked?undefined:420))}</strong><span>RMUC 2027</span></div></div>
 <div className={`viewport ${fullscreen?'page-fullscreen':''}`} ref={viewportHost} id="viewport">{viewport}<button className="fullscreen-toggle" title={fullscreen?'退出全屏':'全屏操控'} aria-label={fullscreen?'退出全屏':'全屏操控'} onClick={toggleFullscreen}>{fullscreen?<Minimize2 size={16}/>:<Maximize2 size={16}/>}<span>{fullscreen?'退出全屏':'全屏'}</span></button></div>
 <div className="transport"><span className="live-badge">LIVE ARENA</span><div className="timeline"><input aria-label="比赛进度" type="range" disabled min={0} max={view?.duration||420} value={view?.time||0}/></div><span>{clock(view?.time||0)} / {clock(view?.duration||420)}</span></div>
 {focus&&<div className="robot-inspector"><div className="inspector-title"><Crosshair size={16}/><span>{focus.team==='red'?'R':'B'}{focus.slot}<strong>{roles[focus.role]}</strong></span><small>{focus.level===undefined?'等级 —':`LV ${focus.level}`}</small></div><div className="inspector-stat"><span>生命</span><strong>{number(focus.hp)}<small> / {number(focus.max_hp)}</small></strong></div><div className="inspector-stat"><span>热量</span><strong>{number(focus.heat)}<small> / {number(focus.max_heat)}</small></strong></div><div className="inspector-stat"><span>弹量</span><strong>{number(focus.ammo)}</strong></div></div>}</section>
 {view?<TelemetryPanel frame={view} selected={inspected||player?.self.id||''} onSelect={inspect} replay={false} events={[]} eventText={eventText}/>:<aside className="telemetry-panel match-data-panel"><div className="data-panel-heading"><div><Activity size={14}/><h3>比赛数据</h3></div></div><p className="loading-app">{player?.display.ui_blocked?'判罚遮挡中':'开始比赛后显示实时数据'}</p></aside>}
 </main></div>
 {page==='hangar'&&room&&<Hangar room={room} catalog={catalog} playerId={playerId} active={active} onActive={setActive} profileSave={profileSave} offline={offline} invoke={invoke} onStrategies={()=>navigate('strategies')}/>}
 {page==='strategies'&&<StrategyGallery room={room} catalog={catalog} playerId={playerId} active={active} saving={saving} offline={offline} invoke={invoke} onConfigure={()=>navigate(room?'hangar':'home')}/>}
 {page==='settings'&&<section className="archive-page lan-settings"><span className="eyebrow">INTERFACE & CONTROL</span><h1>设置</h1><label>鼠标灵敏度 <b>{settings.sensitivity||20}</b><input type="range" min={1} max={50} value={settings.sensitivity||20} onChange={e=>update('sensitivity',Number(e.target.value))}/></label><label>音乐与入场音效 <b>{settings.volume??35}%</b><input type="range" min={0} max={100} value={settings.volume??35} onChange={e=>update('volume',Number(e.target.value))}/></label><p>WASD 移动 · 鼠标云台 · 左键射击 · 右键按住自瞄<br/>Shift 加速 · Space 跳跃 · Ctrl 下蹲 · F 折叠/展开<br/>轮足底盘：W/S 前后、A/D 转向。</p>{room&&<button onClick={()=>{navigate('room');onClose();}}>返回战斗画面</button>}</section>}
 <footer><span role="status">{notice||'RMUC 2027 · 5V5 · 局域网真人联机'}</span><span>ROBOT STRATEGY LEAGUE</span></footer>
 {connectOpen&&<ConnectionDialog settings={settings} onSettings={onSettings} ownerKey={ownerKey} onOwnerKey={setOwnerKey} invoke={invoke} onClose={()=>setConnectOpen(false)}/>}
 </div>;
}

export function ConnectionDialog({settings,onSettings,ownerKey,onOwnerKey,invoke,onClose}:{settings:AppSettings;onSettings:(value:AppSettings)=>void;ownerKey:string;onOwnerKey:(value:string)=>void;invoke:Invoke;onClose:()=>void}){
 const update=(key:string,value:string|number)=>onSettings({...settings,[key]:value});
 return <div className="modal-backdrop lan-connect-backdrop" onMouseDown={e=>{if(e.target===e.currentTarget)onClose();}}><section className="setup-panel lan-connect-panel" role="dialog" aria-modal="true" aria-label="创建或加入房间"><div className="panel-heading"><span className="eyebrow">JOIN THE ARENA</span><button className="icon-button" aria-label="关闭房间连接" onClick={()=>onClose()}><X size={16}/></button></div><h2 className="matchup-title">组建你的队伍</h2><label>昵称<input value={settings.nickname} maxLength={24} onChange={e=>update('nickname',e.target.value)}/></label><div className="form-row"><label>主机 IP<input value={settings.address} onChange={e=>update('address',e.target.value)}/></label><label>端口<input type="number" value={settings.port} min={1024} max={65535} onChange={e=>update('port',Number(e.target.value))}/></label></div><label>仿真工程或运行包目录<div className="lan-runtime-field"><input value={settings.runtime} onChange={e=>update('runtime',e.target.value)}/><button onClick={()=>invoke('browse-runtime')}>浏览</button></div></label><label>独立主机房主口令（普通玩家留空）<input type="password" value={ownerKey} onChange={e=>onOwnerKey(e.target.value)}/></label><button className="primary start-button" onClick={()=>invoke('host',{...settings})}>创建房间 ↗</button><button className="submission-button" onClick={()=>invoke('join',{...settings,owner_key:ownerKey})}>加入房间</button></section></div>;
}
