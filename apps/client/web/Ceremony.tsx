import {useEffect,useMemo,useRef,useState,type CSSProperties,type ReactNode} from 'react';
import type {Frame,Robot,RobotHardware,TeamId} from '../../../packages/renderer/src/types';
import type {RoomState,RuntimeCatalog} from '../../../packages/renderer/src/lanTypes';
import {introAt,introSchedule} from '../../../packages/renderer/src/intro';
import type {IntroShot} from '../../../packages/renderer/src/ArenaScene';
import {roles,side} from './hudModel';
import './arena-broadcast.css';
import './arena-intro.css';
import './ceremony.css';
export function useStageClock(room:RoomState|null){
 const anchor=useRef({at:performance.now(),elapsed:0}),[elapsed,setElapsed]=useState(0);
 useEffect(()=>{anchor.current={at:performance.now(),elapsed:room?.stage_elapsed||0};setElapsed(room?.stage_elapsed||0);},[room?.status,room?.stage_elapsed]);
 useEffect(()=>{const timer=setInterval(()=>setElapsed(anchor.current.elapsed+(performance.now()-anchor.current.at)/1000),33);return()=>clearInterval(timer);},[]);
 return elapsed;
}
export function introductionShot(frame:Frame|null,room:RoomState|null,elapsed:number):IntroShot|null{
 if(!frame||!room)return null;
 if(room.status==='intro'){const at=introAt(introSchedule(frame.robots,room.stage_duration||38.3),elapsed);return at?{key:at.cut.key,segment:at.cut.segment,length:at.length,progress:at.progress}:null;}
 return {key:'prepare',segment:{kind:'faceoff'},length:1,progress:1};
}
export function Stage({children,className=''}:{children:ReactNode;className?:string}){
 const root=useRef<HTMLDivElement>(null),[scale,setScale]=useState(1);
 useEffect(()=>{const element=root.current!;const resize=()=>setScale(Math.min(element.clientWidth/1920,element.clientHeight/1080));const observer=new ResizeObserver(resize);observer.observe(element);resize();return()=>observer.disconnect();},[]);
 return <div className="ceremony-root" ref={root}><div className={`bc-stage ${className}`} style={{width:1920,height:1080,left:'50%',top:'50%',transform:`translate(-50%,-50%) scale(${scale})`,transformOrigin:'center'}}>{children}</div></div>;
}
function Crest({team}:{team:TeamId}){return <div className={`lan-crest ${team}`}><svg viewBox="-100 -100 200 200"><path d="M0 -90 78 -45 78 45 0 90 -78 45 -78 -45Z"/><path className="inner" d="M0 -70 60 -35 60 35 0 70 -60 35 -60 -35Z"/></svg><b>{team==='red'?'R':'B'}</b></div>;}
function NumberStat({label,value,max,unit='',index}:{label:string;value:number;max:number;unit?:string;index:number}){
 const [shown,setShown]=useState(0);useEffect(()=>{let id=0;const start=performance.now()+index*70;const step=()=>{const t=Math.max(0,Math.min(1,(performance.now()-start)/700));setShown(value*(1-(1-t)**3));if(t<1)id=requestAnimationFrame(step);};id=requestAnimationFrame(step);return()=>cancelAnimationFrame(id);},[value,index]);
 return <div className="li-stat" style={{'--i':index,'--fill':`${Math.max(4,Math.min(100,value/Math.max(1,max)*100))}%`} as CSSProperties}><span>{label}</span><div><i/></div><b>{shown.toFixed(unit===' m/s'||unit===' kg'?1:0)}<small>{unit}</small></b></div>;
}
function RobotCard({r,model,room,catalog,length,index,count}:{r:Robot;model?:RobotHardware;room:RoomState;catalog:RuntimeCatalog;length:number;index:number;count:number}){
 const player=room.players?.find(p=>p.team===r.team&&p.slot===r.slot),key=room.team_configs?.[r.team].mode==='team'?room.team_configs[r.team].strategy:room.seat_configs?.[r.team][String(r.slot)].strategy,strategy=catalog.strategies.find(s=>s.id===key);
 const stats=model?.initial_stats,labels=[model?.leg_type==='serial'?'串联腿':model?.chassis_type==='wheel_legged'?'轮足':model?.chassis_type==='mecanum'?'麦克纳姆轮':model?.chassis_type==='omni'?'全向轮':model?.chassis_type==='swerve'?'舵轮':'机库',model?.transform_allowed?'折叠 / 过洞':null,model?.chassis_type==='wheel_legged'?'跳跃 / 登阶':null,model?.arm_count===2?'双机械臂':model?.arm_count===1?'机械臂':null,model?.drone_count?`${model.drone_count} 架无人机`:null].filter(Boolean);
 return <div className={`li-shot ${r.team}`} style={{'--len':`${length}s`} as CSSProperties}><div className="li-flash"/><div className="li-sweep"/><div className="li-reticle"><i/><i/><i/><i/><b/></div><div className="li-spec"><div className="li-number">{String(r.slot).padStart(2,'0')}</div><div className="li-id"><span>{side(r.team)} · {roles[r.role]}</span><h3>{model?.name||r.design_name}</h3><small>{r.model_slot} 型 · {model?.focus}</small></div><div className="li-stats">{[{label:'生命值',value:r.max_hp,max:650},{label:'底盘功率',value:r.power_limit||stats?.power_w||0,max:200,unit:' W'},{label:'热量上限',value:r.max_heat||0,max:400},{label:'单发伤害',value:stats?.damage||0,max:400},{label:'机动速度',value:model?.speed_reference_mps||0,max:7,unit:' m/s'},{label:'整机质量',value:model?.mass_kg||0,max:50,unit:' kg'}].map((s,i)=><NumberStat key={s.label} {...s} index={i}/>)}</div><div className="li-chips">{labels.map((label,i)=><span key={label} style={{'--i':i} as CSSProperties}>{label}</span>)}</div></div><div className={`li-agent ${r.team}`}><div className="li-agent-ring"><Crest team={r.team}/><i/><i/></div><span className="li-agent-label">{player?'PLAYER · 真人操作':'STRATEGY · AI 控制'}</span><strong>{player?.nickname||strategy?.author||'ARENA AI'}</strong><small>{player?'人工驾驶 · AI 断线接管':strategy?.mode==='team'?'全队统筹':'独立决策'}</small><em>{strategy?.name||'通用基线'}</em></div><div className="li-pips"><span>{side(r.team)}阵容</span>{Array.from({length:count},(_,i)=><i key={i} className={i<index?'done':i===index?'now':''}/>)}<b>{index+1}<small>/{count}</small></b></div></div>;
}
export function Ceremony({frame,room,catalog,elapsed,isHost,skip}:{frame:Frame;room:RoomState;catalog:RuntimeCatalog;elapsed:number;isHost:boolean;skip:()=>void}){
 const cuts=useMemo(()=>introSchedule(frame.robots,room.stage_duration||38.3),[frame.robots.map(r=>r.id).join(),room.stage_duration]);
 const at=introAt(cuts,elapsed),s=at?.cut.segment,robot=s?.kind==='robot'?frame.robots.find(r=>r.id===s.id):null,remaining=Math.max(0,(room.stage_duration||0)-elapsed);
 if(room.status==='loading'||room.status==='starting')return <Stage className="cer-loading"><div className="loading-emblem">A</div><h1>准备入场</h1><p>加载战场、机器人与策略进程</p><div className="loading-players">{room.players?.map(p=><span key={p.id} className={p.stage_loaded?'ready':''}>{p.stage_loaded?'✓':'◌'} {p.nickname}</span>)}</div><small>所有客户端完成首帧渲染后同步入场 · {Math.ceil(remaining)}s</small></Stage>;
 if(room.status==='countdown')return <Stage className="cer-countdown"><div className="cd-shade"/><div className="cd-number" key={Math.ceil(remaining)}>{Math.max(1,Math.ceil(remaining))}</div><div className="cd-caption">比赛即将开始</div></Stage>;
 return <Stage className={`cer-lineup ${s?.kind||'open'}`}><div key={at?.cut.key} className="li-cut" style={{'--len':`${at?.length||1}s`} as CSSProperties}>
 {s?.kind==='open'&&<div className="li-open"><div className="li-open-bg red"/><div className="li-open-bg blue"/><div className="li-open-grid"/><div className="li-open-title"><span>RMUC 2027 · MULTIPLAYER ARENA</span><h1>机甲对决</h1><small>5 V 5 · 局域网真人与 AI</small></div>{(['red','blue'] as const).map(t=><div className={`li-open-team ${t}`} key={t}><Crest team={t}/><h2>{side(t)}</h2><p>{frame.robots.filter(r=>r.team===t).length} 台机器人 · 独立出战阵容</p><div className="li-open-rate"><b>READY</b><span>真人协作 · 策略补位</span></div></div>)}<div className="li-open-vs">VS</div></div>}
 {s?.kind==='team'&&<div className={`li-team ${s.team}`}><div className="li-slash"/><div className="li-slash two"/><div className="li-team-word">{s.team.toUpperCase()}</div><div className="li-team-body"><div className="li-team-crest"><Crest team={s.team}/></div><div><span className="li-kicker">{side(s.team)} · 五台机甲协同</span><h2>{side(s.team)}出战</h2><div className="li-team-agents"><b>{room.team_configs?.[s.team].mode==='team'?'全队策略统筹':'独立机器人策略'} · 真人优先控制</b></div></div></div><div className="li-team-strip">{frame.robots.filter(r=>r.team===s.team).map(r=><span key={r.id}><b>{r.slot}</b><small>{roles[r.role]} · {r.design_name}</small></span>)}</div></div>}
 {s?.kind==='robot'&&robot&&<RobotCard r={robot} model={catalog.models[String(robot.model_slot)]} room={room} catalog={catalog} length={at!.length} index={s.index} count={s.count}/>}
 {s?.kind==='faceoff'&&<div className="li-face"><div className="li-face-shade"/>{(['red','blue'] as const).map(team=><div className={`li-face-team ${team}`} key={team}><div className="li-face-head"><Crest team={team}/><h3>{side(team)}</h3></div><div className="li-face-grid n2">{frame.robots.filter(r=>r.team===team).map(r=><div className="li-face-bot" key={r.id}><span className="li-face-art"><b>{r.model_slot}</b></span><b>{r.slot}</b><small>{r.design_name}</small></div>)}</div></div>)}<div className="li-face-vs">VS</div><div className="li-face-check"><strong className="ready">自检完成 · READY</strong></div></div>}
 </div><div className="li-bug"><i/>LIVE · 赛前介绍<b>{Math.ceil(remaining)}s</b></div><div className="li-progress">{cuts.map(c=><i key={c.key} className={`${c.segment.kind} ${'team' in c.segment?c.segment.team:''} ${c.end<=elapsed?'done':c.key===at?.cut.key?'now':''}`} style={{flexGrow:c.end-c.start}}/>)}</div>{isHost&&<button className="cer-skip" onClick={skip}>跳过介绍 → 倒计时</button>}</Stage>;
}
export function useArenaMusic(phase:string,volume:number,elapsed=0){
 const player=useRef<HTMLAudioElement|null>(null),latest=useRef({phase,elapsed});latest.current={phase,elapsed};
 const file=({home:'home.mp3',lobby:'lobby.m4a',hangar:'hangar.m4a',starting:'lobby.m4a',loading:'lobby.m4a',intro:'self-check.m4a',countdown:'countdown.m4a',finished:'victory.m4a',victory:'victory.m4a',results:'results.m4a',mvp:'mvp.m4a',settling:'results.m4a'} as Record<string,string>)[phase];
 useEffect(()=>{if(!file){player.current?.pause();player.current=null;return;}const audio=new Audio('/assets/audio/'+file);player.current=audio;audio.volume=volume/100;audio.loop=['home','lobby','hangar','loading','starting'].includes(phase);const play=()=>{if(audio===player.current&&volume>0)void audio.play().catch(()=>{});};audio.addEventListener('loadedmetadata',()=>{if(['intro','countdown'].includes(latest.current.phase))audio.currentTime=Math.min(latest.current.elapsed,Math.max(0,audio.duration-.1));play();});window.addEventListener('pointerdown',play);return()=>{audio.pause();audio.src='';window.removeEventListener('pointerdown',play);};},[file]);
 useEffect(()=>{if(player.current){player.current.volume=volume/100;if(!volume)player.current.pause();else void player.current.play().catch(()=>{});}},[volume]);
}
