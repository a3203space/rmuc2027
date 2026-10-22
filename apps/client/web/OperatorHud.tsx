import type {Arena,ControlPosition,Structure,TeamId} from '../../../packages/renderer/src/types';
import type {LanPlayer,MapZone,RoomState} from '../../../packages/renderer/src/lanTypes';
import {assemblyOperation,assemblyReason,buffs,clock,number,positionDisplay,ratio,roles,side,zoneLabel} from './hudModel';
import ring from '../../../assets/client/crosshair/crosshair_bg_single.png';
import center from '../../../assets/client/crosshair/crosshair_middle.png';
import heatRing from '../../../assets/client/crosshair/crosshair_slid_circle_bg.png';

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
function Roster({p,team,expanded}:{p:LanPlayer;team:TeamId;expanded:boolean}){
 return <section className={`roster ${team}`}><b>{side(team)}机器人</b>{p.tactical?.robots.filter(r=>r.team===team).map(r=><div className={`robot-card ${r.alive?'':'dead'}`} key={r.id}><div className="label-line"><span>{r.slot} {roles[r.role]}{r.id===p.self.id?' · 自己':''}</span><b>{number(r.hp)}/{number(r.max_hp)}</b></div><Meter value={r.hp} max={r.max_hp} label={`${r.id}血量`}/>{expanded&&<small>弹 {number(r.ammo)} · {r.alive?'存活':'战亡'} {Object.entries(r.buffs).filter(([,v])=>v!==false&&v!==0&&v!==1).map(([k,v])=>`${({invulnerable:'无敌',attack_multiplier:'攻击倍率',defense_buff:'防御',vulnerability:'易伤',cooling_multiplier:'冷却倍率',cooling_bonus:'冷却加成',posture:'姿态',posture_enhanced:'强化'} as Record<string,string>)[k]||k} ${v===true?'':v}`).join(' · ')}</small>}</div>)}</section>;
}
function OwnState({p}:{p:LanPlayer}){
 const r=p.self,labels:Record<string,string>={locked:'未解锁',unlocked:'待安装',installing:'安装中',fitted:'待首次发射',installed:'已安装'},a=p.assembly;
 return <section className="own-state panel"><div className="panel-title">{side(p.team)} {p.slot} 号 · {roles[r.role]}<span>{p.control?.manual?'人工控制':'AI 接管'}</span></div><div className="own-hp"><strong>{number(r.hp)}</strong><span>/ {number(r.max_hp)} HP</span><b>弹 {number(r.ammo)}</b></div><Meter value={r.hp} max={r.max_hp} label="自身血量"/>
 <div className="own-grid"><span>热量 <b>{number(r.heat)}/{number(r.max_heat)}</b></span><span>功率 <b>{number(r.power_w)}/{number(r.power_limit)} W</b></span><span>缓冲 <b>{number(r.buffer_energy_j,1)} J</b></span><span>后勤 <b>{number(p.display.logistics*100)}%</b></span></div><p className="buffs">{buffs(p)}</p>
 {r.transform_allowed&&<p>变形 {number((r.transform_progress||0)*100)}% · F 折叠 / 展开</p>}
 {!r.alive&&r.role!=='aerial'&&<><Meter value={r.respawn_progress} max={r.respawn_required} label="复活进度"/><p>复活 {number(r.respawn_progress,1)} / {number(r.respawn_required,1)}</p></>}
 {r.pending_heal!=null&&<p>远程回血配送：{number(Math.max(0,r.pending_heal-(p.scene?.time||0)),1)} 秒</p>}
 {r.pending_ammo?.map((v,i)=><p key={i}>弹药配送：{v[1]} 发 / {number(Math.max(0,v[0]-(p.scene?.time||0)),1)} 秒</p>)}
 {r.role==='heavy'&&<div className="role-state"><b>携带 {number(r.carried_units)} 单元 · 场上剩余 {number(p.economy?.energy_units_left)}</b><p>发射机构：{labels[r.launcher||'']||'—'} {r.launcher==='installing'?`${number((r.launcher_progress||0)*100)}%`:''}</p><p>装配等级 {number(a?.difficulty)} · {({idle:'空闲',moving:'就位',inserting:'插入',buffered:'确认',complete:'完成',failed:'失败'} as Record<string,string>)[a?.stage||'']||a?.stage||'待命'}</p><Meter value={a?.progress} max={1} label="装配进度"/><p>{assemblyOperation[a?.operation||'']||a?.operation||'待命'} · 本步 {number((a?.operation_progress||0)*100)}%</p>{a?.blocked_reason&&<p className="buffs">{assemblyReason[a.blocked_reason]||a.blocked_reason}</p>}<p>{p.control?.status}</p><p>已完成等级：{p.economy?.completed_difficulties?.join(' / ')||'无'}</p></div>}
 {r.role==='infantry'&&<p>性能 {r.chassis_profile==='hp'?'血量':'功率'} / {r.weapon_profile==='cooling'?'冷却':'爆发'}<br/>{r.chassis_type==='wheel_legged'?'轮腿：W/S 前后，A/D 转向':'WASD 移动'} · 鼠标云台</p>}
 {r.role==='sentry'&&<p>姿态：{({attack:'进攻',defense:'防御',move:'移动'} as Record<string,string>)[r.posture||'']||'—'}{r.posture_enhanced?' · 强化生效':''}<br/>切换冷却 {number(Math.max(0,(r.posture_ready_at||0)-(p.scene?.time||0)),1)} 秒</p>}
 {r.role==='aerial'&&<div className="fleet">{p.air?.map((d,i)=><p key={d.id}>无人机 {i+1} · {({docked:'停靠',flying:'飞行',down:'待回收',disabled:'不可用'} as Record<string,string>)[d.state]||d.state}<br/>电量 {number(d.battery_wh,1)} Wh · 完整度 {number(d.integrity)}</p>)}<p>{p.control?.status}</p></div>}
 </section>;
}
function Reticle({p}:{p:LanPlayer}){return <div className={`reticle ${p.self.overheated?'hot':''}`} aria-hidden><img className="outer-ring" src={ring}/><img className="heat-ring" src={heatRing}/><svg viewBox="0 0 120 120"><circle cx="60" cy="60" r="53" fill="none" stroke="currentColor" strokeWidth="3" strokeDasharray={`${ratio(p.self.heat,p.self.max_heat)*333} 333`} transform="rotate(-90 60 60)"/></svg><img className="crosshair-center" src={center}/><span>{number(p.self.ammo)} 发</span></div>;}
export function OperatorHud({player:p,arena,room,stale,expanded,menu,onMenu,command,notice,isHost}:{player:LanPlayer;arena?:Arena;room:RoomState|null;stale:boolean;expanded:boolean;menu:boolean;onMenu:(open:boolean)=>void;command:Command;notice:string;isHost:boolean}){
 if(p.display.ui_blocked)return <div className="penalty-cover">判罚遮挡中<button onClick={()=>command('room')}>房间设置</button></div>;
 const finished=room?.status==='finished'||p.scene?.status==='finished';
 return <div className={`operator-hud ${expanded?'tactical-expanded':''} ${stale?'stale':''}`}>
 <header className="scoreboard"><TeamHeader p={p} team="red"/><div className="match-clock"><small>RMUC 2027 · 5 V 5</small><strong>{clock(p.scene?.remaining)}</strong><span>{stale?'数据已过期':finished?'比赛结束':'比赛进行中'}</span></div><TeamHeader p={p} team="blue"/></header>
 <Positions p={p}/>
 <div className="rosters"><Roster p={p} team="red" expanded={expanded}/><Roster p={p} team="blue" expanded={expanded}/></div>
 <TacticalMap p={p} arena={arena} expanded={expanded}/><OwnState p={p}/>
 {!expanded&&!menu&&!finished&&!p.display.camera_off&&<Reticle p={p}/>}
 <div className="control-status">{notice||p.control?.status||'等待主机状态'}</div>
 <button className="menu-toggle" onClick={()=>onMenu(!menu)}>Esc · 操作菜单</button>
 {p.control?.needs_resume&&!stale&&<div className="resume-banner">已重连 · AI 暂管<button onClick={()=>command('resume')}>恢复人工控制</button></div>}
 {stale&&<div className="stale-banner">连接中断或数据超时 · 已停止输入，显示最后状态</div>}
 {(menu||finished)&&<div className="operator-menu" onClick={e=>e.stopPropagation()}><div className="label-line"><h2>{finished?`${p.scene?.winner?side(p.scene.winner)+'胜利':'比赛结束'}`:'操作菜单'}</h2><button onClick={()=>onMenu(false)}>关闭</button></div><p>{finished?'保存本局回放，或准备下一局。':'比赛继续运行。关闭菜单后点击画面恢复鼠标。'}</p>
 {!finished&&<div className="action-grid">{p.actions?.map(a=><button key={a.name+a.value} disabled={stale||!a.enabled||!!p.control?.needs_resume} onClick={()=>command(a.name,a.value)}><b>{a.label}{a.cost!=null?` · ${a.cost} 金币`:''}</b><small>{a.reason||a.detail||'执行指令'}</small></button>)}</div>}
 {!finished&&p.self.transform_allowed&&<div className="action-grid"><button disabled={stale||!p.self.alive} onClick={()=>command('transform','fold')}>折叠 · 低姿态过洞</button><button disabled={stale||!p.self.alive} onClick={()=>command('transform','unfold')}>展开 · 洞内受屋顶限制</button></div>}
 <div className="menu-footer"><button onClick={()=>command('room')}>房间与设置</button>{finished&&<><button onClick={()=>command('save')}>保存回放</button><button disabled={stale} onClick={()=>command('ready')}>{room?.players?.find(x=>x.team===p.team&&x.slot===p.slot)?.ready?'已准备 · 取消准备':'准备下一局'}</button>{isHost&&<button disabled={stale||!!room?.players?.some(x=>!x.ready)} onClick={()=>command('start')}>开始下一局</button>}</>}<button onClick={()=>command('leave')}>离开房间</button></div><output>{notice}</output></div>}
 </div>;
}
