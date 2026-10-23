import {useEffect,useRef,useState} from 'react';
import type {Arena,Structure,TeamId} from '../../../packages/renderer/src/types';
import type {LanPlayer,MapZone,RoomState} from '../../../packages/renderer/src/lanTypes';
import {assemblyOperation,assemblyReason,buffs,clock,number,positionDisplay,ratio,roles,side,zoneLabel} from './hudModel';

type Command=(name:string,value?:string)=>void;
const teams:TeamId[]=['red','blue'];
export function Meter({value,max,label,color}:{value?:number;max?:number;label:string;color?:string}){
 return <div className="meter" role="meter" aria-label={label} aria-valuemin={0} aria-valuemax={max} aria-valuenow={value}><i style={{width:`${ratio(value,max)*100}%`,background:color}}/></div>;
}
function StructureBar({value,team,kind,protectedBase=false}:{value?:Structure;team:TeamId;kind:string;protectedBase?:boolean}){
 return <div className={`structure ${team}`} data-testid={`${team}-${kind}`}><div className="label-line"><span>{kind==='base'?'基地':'前哨站'}</span><strong>{number(value?.hp)} <small>/ {number(value?.max_hp)}</small></strong></div><Meter value={value?.hp} max={value?.max_hp} label={`${side(team)}${kind}血量`}/><div className="structure-meta">{protectedBase?'基地保护生效':kind==='base'?'基地可受损':value?.hp===0?'已摧毁':value?.owner?`${side(value.owner)}持有`:'运行中'}{value?.armor_open?' · 装甲展开':''}{(value?.shield_hp||0)>0?` · 护盾 ${number(value?.shield_hp)}`:''}{value?.rebuild?` · 重建 ${number(value.rebuild.progress*100)}%`:''}</div></div>;
}
function TeamHeader({p,team}:{p:LanPlayer;team:TeamId}){
 const t=p.tactical,s=t?.teams[team],position=t?.positions[`${team}_outpost`];
 return <section className={`team-summary ${team}`}><div className="team-title"><b>{side(team)}</b><span>控制度 <strong>{number(s?.control,1)}</strong></span><span>经济 <strong>{number(s?.resources)}</strong></span></div><div className="structures"><StructureBar team={team} kind="base" value={t?.bases[team]} protectedBase={position?.controller===team&&position?.control==='full'}/><StructureBar team={team} kind="outpost" value={t?.outposts[team]}/></div></section>;
}
function Positions({p}:{p:LanPlayer}){
 const tactical=p.tactical;
 return <section className="control-strip" aria-label="控制位置">{teams.flatMap(team=>['outpost','base'].map(kind=>{
 const id=`${team}_${kind}`,v=tactical?.positions[id];
 const d=v?positionDisplay(v,tactical?.control_rules||{capture_s:0,release_s:0}):null;
 return <div key={id} className={`control-card ${v?.controller||team}`} data-testid={id}><div className="label-line"><b>{side(team)}{kind==='base'?'基地':'前哨'}控制位</b><span>{v?`${side(v.controller)} · ${v.control==='full'?'完全':'临时'}`:'—'}</span></div><Meter value={v?.progress} max={d?.duration} label={`${id}控制进度`}/><div className="control-detail">{d?.text||'等待数据'} <span>{number(v?.progress,1)}/{number(d?.duration,1)}s</span></div><div className="zone-pair">{v?.zones.map(z=><span key={z} className={tactical?.zones[z]?.active?'lit':''}>{zoneLabel({id:z,x:0,y:0,radius:0})} <b>{tactical?.zones[z]?.active_teams?.map(side).join('/')||'未激活'}</b></span>)}</div></div>;
 }))}</section>;
}
export function TacticalMap({p,arena,expanded=false}:{p:LanPlayer;arena?:Arena;expanded?:boolean}){
 if(!arena)return <section className="tactical-map panel">等待场地数据</section>;
 const w=arena.width,h=arena.height,t=p.tactical;
 return <section className={`tactical-map panel ${expanded?'expanded':''}`}><div className="panel-title">战术地图 <span>红 / 蓝 / 灰：阵营与中立 · 亮框：激活 · 黄框：争夺</span></div><svg viewBox={`-.7 -.7 ${w+1.4} ${h+1.4}`} aria-label="RMUC 2027 战术地图">
 <rect width={w} height={h} fill="#142535" stroke="#577080" strokeWidth=".05"/>
 {arena.terrain?.filter(x=>x.elevation>0).map((x,i)=>x.polygon?<polygon key={i} points={x.polygon.map(p=>p.join(',')).join(' ')} fill="#284055"/>:<rect key={i} x={x.x} y={x.y} width={x.w} height={x.h} fill="#284055"/>)}
 {arena.obstacles?.map((o,i)=><rect key={i} x={o.x} y={o.y} width={o.w} height={o.h} fill="#496071" opacity=".6" transform={`rotate(${(o.rotation||0)*180/Math.PI} ${o.x+o.w/2} ${o.y+o.h/2})`}/>)}
 {(arena.zones as MapZone[]).map(z=>{const s=t?.zones[z.id],color=s?.contested?'#ffd774':s?.owner==='red'?'#ee7e79':s?.owner==='blue'?'#75baff':'#a3b6c9';return <g key={z.id} className="map-zone"><title>{`${zoneLabel(z)} · ${s?.active_teams?.map(side).join('/')||'未激活'}`}</title>{z.polygon?<polygon points={z.polygon.map(p=>p.join(',')).join(' ')} fill={color} fillOpacity={s?.active?.65:.16} stroke={color} strokeWidth={s?.active?.12:.05}/>:<circle cx={z.x} cy={z.y} r={z.radius} fill={color} fillOpacity=".2" stroke={color} strokeWidth=".06"/>}{(!['road','tunnel','highland'].includes(z.kind||'')||s?.active)&&<text x={z.x} y={z.y-.2} textAnchor="middle" fontSize={expanded?.39:.47} className="map-label">{zoneLabel(z).replace('红方','红·').replace('蓝方','蓝·')}</text>}</g>;})}
 {teams.flatMap(team=>['base','outpost'].map(kind=>{const s=(kind==='base'?t?.bases:t?.outposts)?.[team];return s?<g key={team+kind}><rect x={s.x-.22} y={s.y-.22} width=".44" height=".44" fill={team==='red'?'#ff8c83':'#78c4ff'}/><title>{`${side(team)}${kind==='base'?'基地':'前哨'} ${number(s.hp)}`}</title></g>:null;}))}
 {t?.robots.map(r=><g key={r.id} transform={`translate(${r.x} ${r.y})`} opacity={r.alive?1:.4}><circle r={r.id===p.self.id?.42:.28} fill={r.team==='red'?'#ff8c83':'#78c4ff'} stroke={r.id===p.self.id?'white':'#09131e'} strokeWidth=".09"/>{r.heading!==undefined&&<path d="M .32 0 L .72 0" transform={`rotate(${r.heading*180/Math.PI})`} stroke="white" strokeWidth=".1"/>}<text y=".16" textAnchor="middle" fill="#07111b" fontSize=".46" fontWeight="900">{r.slot}</text></g>)}
 </svg>{expanded&&<div className="zone-directory">{(arena.zones as MapZone[]).map(z=><span key={z.id}>{zoneLabel(z)}<b>{t?.zones[z.id]?.active_teams?.map(side).join('/')||'未激活'}</b></span>)}</div>}<div className="map-legend">白圈：自己 · 暗色：战亡 · Tab 展开战术总览</div></section>;
}
function RobotGlyph({role}:{role:string}){
 return <svg viewBox="0 0 64 48" className="robot-glyph" aria-hidden="true"><path d={role==='aerial'?'M8 12h48v26H8z M16 12V6h32v6 M18 20h28M18 28h28':role==='heavy'?'M12 28h40v12H12z M24 28V8h12v20 M36 12h18v6H36 M4 32h8m40 0h8':role==='sentry'?'M8 28h48v12H8z M20 28V16h24v12 M32 16V8h24v5H32':'M12 28h40v10H12z M24 28V16h16v12 M32 16V9h24v5H32'}/><circle cx="16" cy="40" r="5"/><circle cx="48" cy="40" r="5"/></svg>;
}
function Roster({p,team,expanded}:{p:LanPlayer;team:TeamId;expanded:boolean}){
 const robots=p.tactical?.robots.filter(r=>r.team===team).sort((a,b)=>team==='red'?b.slot-a.slot:a.slot-b.slot);
 return <section className={`roster ${team}`} aria-label={`${side(team)}机器人`}>{robots?.map(r=><div className={`robot-card ${r.alive?'':'dead'} ${r.id===p.self.id?'self':''}`} key={r.id}><div className="card-identity"><strong>{r.slot}</strong><RobotGlyph role={r.role}/><span>{roles[r.role]}{r.id===p.self.id&&<small>自己</small>}</span></div><Meter value={r.hp} max={r.max_hp} label={`${r.id}血量`}/><div className="card-health">{number(r.hp)}<small> / {number(r.max_hp)}</small>{!r.alive&&<em>战亡</em>}</div>{expanded&&<small className="card-detail">弹 {number(r.ammo)} · {r.alive?'存活':'战亡'} {Object.entries(r.buffs).filter(([,v])=>v!==false&&v!==0&&v!==1).map(([k,v])=>`${({invulnerable:'无敌',attack_multiplier:'攻击倍率',defense_buff:'防御',vulnerability:'易伤',cooling_multiplier:'冷却倍率',cooling_bonus:'冷却加成',posture:'姿态',posture_enhanced:'强化'} as Record<string,string>)[k]||k} ${v===true?'':v}`).join(' · ')}</small>}</div>)}</section>;
}
function OwnDetails({p}:{p:LanPlayer}){
 const r=p.self,labels:Record<string,string>={locked:'未解锁',unlocked:'待安装',installing:'安装中',fitted:'待首次发射',installed:'已安装'},a=p.assembly;
 return <div className="own-details">
 <div className="own-grid"><span>热量 <b>{number(r.heat)}/{number(r.max_heat)}</b></span><span>功率 <b>{number(r.power_w)}/{number(r.power_limit)} W</b></span><span>缓冲 <b>{number(r.buffer_energy_j,1)} J</b></span><span>后勤 <b>{number(p.display.logistics*100)}%</b></span></div>
 {r.pending_heal!=null&&<p>远程回血配送：{number(Math.max(0,r.pending_heal-(p.scene?.time||0)),1)} 秒</p>}
 {r.pending_ammo?.map((v,i)=><p key={i}>弹药配送：{v[1]} 发 / {number(Math.max(0,v[0]-(p.scene?.time||0)),1)} 秒</p>)}
 {r.role==='heavy'&&<div className="role-state"><b>携带 {number(r.carried_units)} 单元 · 场上剩余 {number(p.economy?.energy_units_left)}</b><p>发射机构：{labels[r.launcher||'']||'—'} {r.launcher==='installing'?`${number((r.launcher_progress||0)*100)}%`:''}</p><p>装配等级 {number(a?.difficulty)} · {({idle:'空闲',moving:'就位',inserting:'插入',buffered:'确认',complete:'完成',failed:'失败'} as Record<string,string>)[a?.stage||'']||a?.stage||'待命'}</p><Meter value={a?.progress} max={1} label="装配进度"/><p>{assemblyOperation[a?.operation||'']||a?.operation||'待命'} · 本步 {number((a?.operation_progress||0)*100)}%</p>{a?.blocked_reason&&<p className="buffs">{assemblyReason[a.blocked_reason]||a.blocked_reason}</p>}<p>{p.control?.status}</p><p>已完成等级：{p.economy?.completed_difficulties?.join(' / ')||'无'}</p></div>}
 {r.role==='infantry'&&<p>性能 {r.chassis_profile==='hp'?'血量':'功率'} / {r.weapon_profile==='cooling'?'冷却':'爆发'} · 冷却 {number(r.cooling_rate,1)}/s<br/>{r.chassis_type==='wheel_legged'?'轮腿：W/S 前后，A/D 转向':'WASD 移动'} · 鼠标云台</p>}
 {r.role==='sentry'&&<p>姿态：{({attack:'进攻',defense:'防御',move:'移动'} as Record<string,string>)[r.posture||'']||'—'}{r.posture_enhanced?' · 强化生效':''}<br/>切换冷却 {number(Math.max(0,(r.posture_ready_at||0)-(p.scene?.time||0)),1)} 秒</p>}
 {r.role==='aerial'&&<div className="fleet">{p.air?.map((d,i)=><p key={d.id}>无人机 {i+1} · {({docked:'停靠',flying:'飞行',down:'待回收',disabled:'不可用'} as Record<string,string>)[d.state]||d.state}<br/>电量 {number(d.battery_wh,1)} Wh · 完整度 {number(d.integrity)}</p>)}<p>{p.control?.status}</p></div>}
 </div>;
}
function OwnState({p,expanded}:{p:LanPlayer;expanded:boolean}){
 const r=p.self;
 return <section className={`own-state ${p.team}`} aria-label="本机状态">
 <div className="buffs">{buffs(p)}</div>
 <div className="own-identity"><div className="own-number"><RobotGlyph role={r.role}/><strong>{p.slot}</strong></div><div className="own-vitals"><div className="panel-title">{side(p.team)} {p.slot} 号 · {roles[r.role]}<span>{p.control?.manual?'人工控制':'AI 接管'}</span></div><div className="own-hp"><strong>{number(r.hp)}</strong><span>/ {number(r.max_hp)} HP</span></div><Meter value={r.hp} max={r.max_hp} label="自身血量"/></div></div>
 <div className="drive-meters"><div><span>底盘功率 <b>{number(r.power_w)} / {number(r.power_limit)} W</b></span><Meter value={r.power_w} max={r.power_limit} label="底盘功率"/></div><div><span>缓冲能量 <b>{number(r.buffer_energy_j,1)} J</b></span><Meter value={r.buffer_energy_j} max={60} label="缓冲能量"/></div></div>
 <div className="own-quick">{r.role==='infantry'?`性能 ${r.chassis_profile==='hp'?'血量':'功率'} / ${r.weapon_profile==='cooling'?'冷却':'爆发'} · 冷却 ${number(r.cooling_rate,1)}/s`:r.role==='heavy'?`携带 ${number(r.carried_units)} 单元 · 装配 ${number((p.assembly?.progress||0)*100)}%`:r.role==='sentry'?`姿态 ${({attack:'进攻',defense:'防御',move:'移动'} as Record<string,string>)[r.posture||'']||'—'}${r.posture_enhanced?' · 强化':''}`:`无人机 ${p.air?.filter(d=>d.state==='flying').length||0} 架飞行中`}{r.transform_allowed&&` · 变形 ${number((r.transform_progress||0)*100)}% · F 折叠`}</div>
 {!r.alive&&r.role!=='aerial'&&<div className="respawn-state"><Meter value={r.respawn_progress} max={r.respawn_required} label="复活进度"/><p>复活 {number(r.respawn_progress,1)} / {number(r.respawn_required,1)}</p></div>}
 {r.pending_heal!=null&&<p className="delivery-status">回血配送 {number(Math.max(0,r.pending_heal-(p.scene?.time||0)),1)} 秒</p>}
 {r.pending_ammo?.map((v,i)=><p className="delivery-status" key={i}>弹药配送 {v[1]} 发 · {number(Math.max(0,v[0]-(p.scene?.time||0)),1)} 秒</p>)}
 {(expanded||r.role==='aerial')&&<OwnDetails p={p}/>}
 </section>;
}
function CombatStatus({p,command,stale,finished,aimHeld}:{p:LanPlayer;command:Command;stale:boolean;finished:boolean;aimHeld?:boolean}){
 const aim=p.control?.aim_assist,tracking=(aimHeld??(!!aim&&aim.state!=='idle'))&&!stale&&!finished&&!p.control?.needs_resume;
 const label=!tracking?'待机':({idle:'搜索目标',searching:'搜索目标',tracking:'跟踪目标',lost:'目标丢失 · 搜索中'} as Record<string,string>)[aim?.state||'searching'];
 return <section className="combat-status" aria-label="武器与自瞄状态"><div className="weapon-ammo"><strong>{number(p.self.ammo)}</strong><span>发<br/>{p.self.role==='heavy'?'42':'17'} mm</span></div><div className="weapon-heat">弹速 {number(p.weapon?.muzzle_speed_mps,1)} m/s<br/>额定射频 {number(p.weapon?.fire_rate_hz,1)} 发/s<br/>热量 {number(p.self.heat)} / {number(p.self.max_heat)}</div>
 <div className="ammo-entry"><span>{p.self.ammo===0?'无弹丸 · 请手动购弹':'左键点按单发 / 按住连发'}</span><button disabled={stale||finished||!!p.control?.needs_resume} onClick={()=>command('buy-ammo')}>B · 购弹</button></div>
 <div className={`aim-status ${tracking&&aim?.state==='tracking'?'tracking':''}`}>右键按住：自瞄 · {label}{tracking&&aim?.target_id?` · ${aim.target_id}`:''}</div>
 {p.weapon?.detail&&<p className="weapon-reason">{p.weapon.detail}</p>}
 {!!p.control?.last_press_seq&&<small>上次点按：{p.control.last_press_result||'等待主机结算'}</small>}
 </section>;
}
function Reticle({p,aimHeld}:{p:LanPlayer;aimHeld?:boolean}){const tracking=aimHeld&&p.control?.aim_assist?.state==='tracking';return <div className={`reticle ${p.self.overheated?'hot':''} ${tracking?'tracking':''}`} aria-label="第一视角瞄准框"><div className="sight-frame"><i/><i/><i/><i/></div><svg className="sight-center" viewBox="0 0 64 64" aria-hidden="true"><circle cx="32" cy="32" r="15"/><path d="M32 7v14m0 22v14M7 32h14m22 0h14"/><circle cx="32" cy="32" r="1"/></svg></div>;}
function OperatorFeedback({p}:{p:LanPlayer}){
 const previous=useRef({match:p.match_id,id:p.self.id,tick:p.tick,hp:p.self.hp});
 const [hit,setHit]=useState(0),[entries,setEntries]=useState<{key:string;text:string;time:number;at:number}[]>([]);
 const seen=useRef(new Set<string>());
 useEffect(()=>{
  const old=previous.current,reset=old.match!==p.match_id||old.id!==p.self.id||p.tick<old.tick;
  if(reset){setHit(0);seen.current.clear();setEntries([]);}
  else if(p.self.hp<old.hp&&p.self.alive)setHit(value=>value+1);
  previous.current={match:p.match_id,id:p.self.id,tick:p.tick,hp:p.self.hp};
  const labels:Record<string,string>={ammo_purchase:'弹药已购买',ammo_delivered:'弹药已送达',remote_heal:'远程回血',respawn:'机器人已复活',rpc_error:'操作未执行'};
  const fresh=(p.events||[]).flatMap(event=>{
   const key=event.id||`${event.tick??p.tick}:${event.type}:${event.amount}:${event.cost}:${event.message}`;
   if(seen.current.has(key))return [];
   seen.current.add(key);
   return [{key,text:event.message||`${labels[event.type]||event.type}${event.amount!=null?' · '+number(event.amount):''}`,time:p.scene?.time||0,at:Date.now()}];
  });
  if(fresh.length)setEntries(value=>[...(reset?[]:value),...fresh].slice(-3));
  if(seen.current.size>100)seen.current=new Set([...seen.current].slice(-50));
 },[p]);
 useEffect(()=>{const timer=setInterval(()=>setEntries(value=>value.filter(event=>Date.now()-event.at<6500)),1000);return()=>clearInterval(timer);},[]);
 return <>{p.self.alive&&p.self.role!=='aerial'&&<>{ratio(p.self.hp,p.self.max_hp)<.25&&<div className="feedback-vignette low-health" aria-hidden="true"/>}{hit>0&&<div key={hit} className="feedback-vignette hit" aria-hidden="true"/>}</>}<div className="event-feed" aria-label="本机事件" aria-live="polite">{entries.map(event=><div key={event.key}><time>{clock(event.time)}</time>{event.text}</div>)}</div></>;
}
export function OperatorHud({player:p,arena,room,stale,expanded,menu,onMenu,command,notice,isHost,ammoOnly=false,aimHeld,scale=1}:{player:LanPlayer;arena?:Arena;room:RoomState|null;stale:boolean;expanded:boolean;menu:boolean;onMenu:(open:boolean)=>void;command:Command;notice:string;isHost:boolean;ammoOnly?:boolean;aimHeld?:boolean;scale?:number}){
 if(p.display.ui_blocked)return <div className="penalty-cover">判罚遮挡中<button onClick={()=>command('room')}>房间设置</button></div>;
 const finished=room?.status==='finished'||p.scene?.status==='finished';
 return <div className={`operator-hud ${expanded?'tactical-expanded':''} ${stale?'stale':''} ${p.team} role-${p.self.role} ${p.display.camera_off?'camera-off':''}`} style={{transform:`scale(${scale})`}}>
 <OperatorFeedback p={p}/>
 <header className="scoreboard"><TeamHeader p={p} team="red"/><div className="match-clock"><small>RMUC 2027 · 5 V 5</small><strong>{clock(p.scene?.remaining)}</strong><span>{stale?'数据已过期':finished?'比赛结束':'比赛进行中'}</span></div><TeamHeader p={p} team="blue"/></header>
 <Positions p={p}/>
 <div className="rosters"><Roster p={p} team="red" expanded={expanded}/><Roster p={p} team="blue" expanded={expanded}/></div>
 <TacticalMap p={p} arena={arena} expanded={expanded}/><OwnState p={p} expanded={expanded}/>
 {!expanded&&!menu&&!finished&&!p.display.camera_off&&p.self.role!=='aerial'&&<Reticle p={p} aimHeld={aimHeld&&!stale&&!p.control?.needs_resume}/>}
 {p.self.role!=='aerial'&&<><CombatStatus p={p} command={command} stale={stale} finished={finished} aimHeld={aimHeld}/><div className={`heat-strip ${p.self.overheated?'hot':''}`}><span>发射热量 <b>{number(p.self.heat)} / {number(p.self.max_heat)}</b></span><Meter value={p.self.heat} max={p.self.max_heat} label="发射热量"/><small>冷却 {number(p.self.cooling_rate,1)} /s · {p.weapon?.detail||'等待发射状态'}</small></div></>}
 <div className="control-status">{notice||p.control?.status||'等待主机状态'}</div>
 <button className="menu-toggle" onClick={()=>onMenu(!menu)}>Esc · 操作菜单</button>
 {p.control?.needs_resume&&!stale&&<div className="resume-banner">已重连 · AI 暂管<button onClick={()=>command('resume')}>恢复人工控制</button></div>}
 {stale&&<div className="stale-banner">连接中断或数据超时 · 已停止输入，显示最后状态</div>}
 {(menu||finished)&&<div className="operator-menu" onClick={e=>e.stopPropagation()}><div className="label-line"><h2>{finished?`${p.scene?.winner?side(p.scene.winner)+'胜利':'比赛结束'}`:ammoOnly?'购买弹丸':'操作菜单'}</h2><button onClick={()=>onMenu(false)}>关闭</button></div><p>{finished?'保存本局回放，或准备下一局。':'比赛继续运行。关闭菜单后点击画面恢复鼠标。'}</p>
 {!finished&&ammoOnly&&<div className="purchase-summary"><strong>队伍余额：{number(p.tactical?.teams[p.team]?.resources)} 金币</strong><p>当前弹丸 {number(p.self.ammo)} 发 · {p.self.role==='heavy'?'42 mm':'17 mm'}<br/>步兵、重装按规则零弹开局，购买后才能射击。区域补给立即到账，远程补给需脱战且配送 6 秒。</p>{p.self.pending_ammo?.map((v,i)=><p key={i}>配送中：{v[1]} 发 · 剩余 {number(Math.max(0,v[0]-(p.scene?.time||0)),1)} 秒</p>)}</div>}
 {!finished&&<div className="action-grid">{p.actions?.filter(a=>!ammoOnly||a.name==='ammo').map(a=><button key={a.name+a.value} disabled={stale||!a.enabled||!!p.control?.needs_resume} onClick={()=>command(a.name,a.value)}><b>{a.label}{a.cost!=null?` · ${a.cost} 金币`:''}</b><small>{a.reason||a.detail||'执行指令'}</small></button>)}</div>}
 {!finished&&!ammoOnly&&<OwnDetails p={p}/>}
 {!finished&&!ammoOnly&&p.self.transform_allowed&&<div className="action-grid"><button disabled={stale||!p.self.alive} onClick={()=>command('transform','fold')}>折叠 · 低姿态过洞</button><button disabled={stale||!p.self.alive} onClick={()=>command('transform','unfold')}>展开 · 洞内受屋顶限制</button></div>}
 <div className="menu-footer"><button onClick={()=>command('room')}>房间与设置</button>{finished&&<><button onClick={()=>command('save')}>保存回放</button><button disabled={stale} onClick={()=>command('ready')}>{room?.players?.find(x=>x.team===p.team&&x.slot===p.slot)?.ready?'已准备 · 取消准备':'准备下一局'}</button>{isHost&&<button disabled={stale||!!room?.players?.some(x=>!x.ready)} onClick={()=>command('start')}>开始下一局</button>}</>}<button onClick={()=>command('leave')}>离开房间</button></div><output>{notice}</output></div>}
 </div>;
}
