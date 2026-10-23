// Selection markup and model details ported from Arena-1 TeamLineup.tsx.
import {ArrowLeft,Ban,Battery,Bot,Check,ChevronRight,Copy,Crosshair,Download,Gauge,Hammer,LockKeyhole,Rocket,Shield,ShieldCheck,Timer,Wrench,X,Zap} from 'lucide-react';
import type {RobotHardware,TeamId,Season} from '../../../../packages/renderer/src/types';
import {ROLE_NAMES,SEASON_LABELS} from '../../../../packages/renderer/src/types';
import type {RoomState,RuntimeCatalog,ProfileSaveState} from '../../../../packages/renderer/src/lanTypes';
import {RobotPreview} from '../RobotPreview';
import {PerformancePicker} from '../PerformancePicker';
import {EMPTY_PROFILE_SAVE,profileDisabledReason,seatProfile} from '../performance';
import {compatible} from './configuration';
import './teamLineup.css';
const legalIn=(model:RobotHardware|undefined,_season:Season)=>!!model&&!!model.seasons?.['2027']?.legal;
const legalityIssues=(model:RobotHardware,_season:Season)=>(model.seasons?.['2027']?.issues||[]).map(i=>i.zh).join('；');
const drives:Record<string,string>={mecanum:'麦克纳姆轮',swerve:'舵轮',omni:'全向轮',wheel_legged:'双轮足',quadrotor:'旋翼'};
const roleIcons={infantry:Shield,aerial:Rocket,heavy:Hammer,sentry:ShieldCheck};
const code=(model?:RobotHardware)=>model?.name.split(' ')[0]||'—';
const title=(model:RobotHardware)=>model.name.replace(/^\S+\s+/, '');
export function modelAbilities(model:RobotHardware){return [model.transform_allowed?'折叠穿隧道':null,model.arm_count===2?'双机械臂':null,model.chassis_type==='wheel_legged'?(model.step_climb_m?'可跳跃 / 登阶':'可跳跃 / 飞坡'):null,model.leg_type==='serial'?'翻倒自救':null,model.role==='heavy'?'模块化42mm':null,model.role==='sentry'?'全自动':null,model.role==='aerial'?(model.drone_count?`${model.drone_count}架空中机器人`:`${model.rotor_count}旋翼`):null].filter(Boolean).join(' · ')||model.focus;}

function features(model:RobotHardware){
  const result:{name:string;detail:string}[]=[];
  if(model.transform_allowed)result.push({name:'低姿态穿越',detail:`炮塔与导轨折叠后高 ${Math.round((model.folded_height_m??.235)*1000)}mm，可通过两侧低矮隧道；洞内展开受屋顶限制。`});
  if(model.arm_count)result.push({name:model.arm_count===2?'双臂独立协作':'升降伸缩机械臂',detail:model.arm_count===2?'两只手可分别控制，支持一手扶正队友、另一手夹取能量单元。':'支持夹取、抬升、装配与扶正。取物需要真实的闭爪和入仓过程。'});
  if(model.chassis_type==='wheel_legged')result.push({name:'主动平衡与跳跃',detail:`腿长 ${Math.round((model.leg_min_m??.12)*1000)}–${Math.round((model.leg_max_m??.34)*1000)}mm。支持飞坡与越障，驱动沿车身前后方向。`});
  if(model.leg_type==='serial')result.push({name:'串联腿 · 髋关节整周旋转',detail:'髋关节没有限位。翻倒、侧躺或车身倒扣后，腿先转到地面，再把车身绕轮轴翻回来，约 1–2 秒，无需工程救援。'});
  if(model.chassis_type==='swerve')result.push({name:'四轮独立转向',detail:`舵轮以 ${Math.round((model.steer_rate_rad_s??0)*180/Math.PI)}°/s 转向，适合高抓地机动；换向需要转动轮组。`});
  if(model.chassis_type==='omni'||model.chassis_type==='mecanum')result.push({name:'全向移动',detail: `${drives[model.chassis_type]}支持前后与横向移动。不同构型的抓地、效率与重量分别配置。`});
  if(model.role==='aerial'&&model.drone)result.push({name:`机库 · ${model.drone_count} 架全自动空中机器人`,detail:`每架 ${Math.round(model.drone.size_m*1000)}mm / ${Math.round(model.drone.mass_kg*1000)}g，护罩可承受 ${model.drone.impact_integrity} 次有效撞击；只能撞击基地与前哨站中层装甲，机库内不充电。`});
  else if(model.role==='aerial')result.push({name:`${model.rotor_count}旋翼空中支援`,detail:'完整桨叶护罩与绳长约束；使用17mm弹丸，飞行会消耗电池和支援时间。'});
  if(model.role==='heavy')result.push({name:'模块化 42mm 发射机构',detail:`完成 3 级装配或按时间规则解锁后，在工作台停稳约 ${model.launcher_module_kg?.toFixed(1)??'5'} kg 模块安装，首发弹丸完成装配确认；之前只负责能量单元与科技核心。`});
  if(model.role==='sentry')result.push({name:'全自动哨兵 · 三种姿态',detail:'进攻、防御、移动姿态各有增益与代价，切换间隔 5 秒；每分钟可在补给区领取 100 发。'});
  return result;
}

function ModelDetails({model,peers,season}:{model:RobotHardware;peers:RobotHardware[];season:Season}){
  const stats=model.initial_stats_by_season?.[season]??model.initial_stats;
  const meters=[{name:'参考速度',value:model.speed_reference_mps,max:Math.max(...peers.map(m=>m.speed_reference_mps)),unit:'m/s'},
    {name:'电池容量',value:model.battery_wh,max:Math.max(...peers.map(m=>m.battery_wh)),unit:'Wh'}];
  return <aside className="hangar-details">
    <div className="hangar-model-heading"><span>MODEL {String(model.slot).padStart(3,'0')}</span><b>{code(model)}</b><h3>{title(model)}</h3><p>{model.focus}</p></div>
    <div className="hangar-tags"><span>{model.chassis_type==='hangar'?'固定机库':drives[model.chassis_type]}</span><span>{model.role==='heavy'?'模块化42mm':model.role==='aerial'&&model.drone?'撞击':'17mm'}</span>{model.seasons&&(['2027'] as Season[]).map(s=><span key={s} className={legalIn(model,s)?'legal':'illegal'} title={legalIn(model,s)?'':legalityIssues(model,s)}>{SEASON_LABELS[s]} {legalIn(model,s)?'✓':'✗'}</span>)}</div>
    <div className="hangar-stat-grid">
      <div><Shield size={15}/><span>型号默认 Lv.1 血量</span><strong>{model.role==='aerial'?'—':stats?.hp??'—'}<small>{model.role==='aerial'?'无血量机制':'HP'}</small></strong></div>
      {model.drone?<div><Zap size={15}/><span>空中机器人</span><strong>{model.drone_count}<small>架 · {Math.round(model.drone.mass_kg*1000)} g</small></strong></div>:<div><Zap size={15}/><span>{model.role==='aerial'?'飞行峰值功率':'型号默认底盘功率'}</span><strong>{model.role==='aerial'?model.flight_power_max_w:stats?.power_w??'—'}<small>W</small></strong></div>}
      <div><Gauge size={15}/><span>整机基础质量</span><strong>{model.mass_kg.toFixed(2)}<small>kg</small></strong></div>
      {model.drone?<div><Crosshair size={15}/><span>护罩耐撞</span><strong>{model.drone.impact_integrity}<small>次 / 架</small></strong></div>:<div><Crosshair size={15}/><span>{model.role==='heavy'?'独立机械臂':'热量 / 冷却'}</span><strong>{model.role==='heavy'?model.arm_count:`${stats?.max_heat??'—'} / ${stats?.cooling??'—'}`}<small>{model.role==='heavy'?'只':'/s'}</small></strong></div>}
    </div>
    <div className="hangar-meters">{meters.map(m=><div key={m.name}><div><span>{m.name}</span><b>{m.value.toFixed(m.unit==='m/s'?1:0)} <small>{m.unit}</small></b></div><span className="hangar-meter"><i style={{width:`${m.value/m.max*100}%`}}/></span></div>)}</div>
    <div className="hangar-feature-list">{features(model).map(f=><div key={f.name}><Check size={13}/><div><strong>{f.name}</strong><p>{f.detail}</p></div></div>)}</div>
    <div className="hangar-footprint"><span>收纳尺寸</span><b>{model.stowed_m.map(v=>Math.round(v*1000)).join(' × ')} <small>mm</small></b></div>
    <p className="hangar-stats-note">条形按同兵种型号比较。速度为控制参考值；初始性能可在规则允许的准备阶段改选。</p>
  </aside>;
}

export function Hangar({room,catalog,playerId,active,onActive,profileSave,offline,invoke,onStrategies}:{room:RoomState;catalog:RuntimeCatalog;playerId:string;active:{team:TeamId;slot:number};onActive:(seat:{team:TeamId;slot:number})=>void;profileSave:ProfileSaveState;offline:boolean;invoke:(name:string,value?:Record<string,unknown>)=>void;onStrategies:()=>void}){
 const me=room.players?.find(p=>p.id===playerId),occupied=room.players?.find(p=>p.team===active.team&&p.slot===active.slot),ownSeat=occupied?.id===playerId,saving=!!profileSave.pending;
 const editable=['lobby','finished'].includes(room.status)&&!saving&&!offline;
 const canEdit=editable&&(ownSeat?!!me?.connected:!!me?.host&&!occupied);
 const roles:Record<number,RobotHardware['role']>={1:'heavy',3:'infantry',4:'infantry',6:'aerial',7:'sentry'};
 const config=room.seat_configs?.[active.team]?.[String(active.slot)],model=catalog.models[String(config?.model)],strategy=catalog.strategies.find(s=>s.id===config?.strategy);
 const peers=Object.values(catalog.models).filter(m=>m.role===roles[active.slot as keyof typeof roles]);
 const roster=[1,3,4,6,7].map(slot=>({slot,model:catalog.models[String(room.seat_configs?.[active.team]?.[String(slot)]?.model)]}));
 const code=(model?:RobotHardware)=>model?.name.split(' ')[0]||'—';
 const activeSlot=active.slot,setActiveSlot=(slot:number)=>onActive({...active,slot}),season='2027' as const,locked=!canEdit,busy=saving;
 const fleetMass=roster.reduce((sum,entry)=>sum+(entry.model?.mass_kg||0),0),fleetBattery=roster.reduce((sum,entry)=>sum+(entry.model?.battery_wh||0),0);
 const value=Object.fromEntries(Object.entries(room.seat_configs?.[active.team]||{}).map(([slot,entry])=>[slot,entry.model]));
 const onChange=(next:Record<string,number>)=>{const nextModel=next[String(activeSlot)];invoke('seat-config',{...active,model:nextModel,strategy:compatible(strategy,activeSlot,nextModel)?config!.strategy:'baseline'});};
 return <div className="hangar-page"><section className="team-lineup-dialog hangar" aria-label="出战机库">
 <header className="hangar-header"><div className="hangar-brand"><span><Bot size={22}/></span><div><small>RMUC ARENA / ROBOT ARMORY</small><h2>出战机库</h2></div></div><div className="hangar-symmetry"><i/><span>{active.team==='red'?'红方':'蓝方'} {active.slot} 号 · {occupied?.nickname||'AI 席位'} · {canEdit?'可配置':'查看模式'}</span></div><div className="hangar-team-switch">{(['red','blue'] as const).map(team=><button key={team} disabled={saving} className={active.team===team?'active':''} onClick={()=>onActive({...active,team})}>{team==='red'?'红方':'蓝方'}</button>)}</div></header>
      <div className="hangar-body"><nav className="hangar-roster" aria-label="出战位置"><span className="hangar-section-label">SLOTS / 出战位置</span>{roster.map(({slot,model:m})=>{const Icon=roleIcons[roles[slot] as keyof typeof roleIcons];return <button key={slot} aria-label={`${slot}号${ROLE_NAMES[roles[slot] as keyof typeof ROLE_NAMES]}位置`} aria-pressed={activeSlot===slot} className={activeSlot===slot?'active':''} onClick={()=>setActiveSlot(slot)}><span className="hangar-slot-number">{String(slot).padStart(2,'0')}</span><div><b><Icon size={13}/>{ROLE_NAMES[roles[slot] as keyof typeof ROLE_NAMES]}</b><span>{m?title(m):'选择型号'}</span><small>{code(m)}</small></div><ChevronRight size={14}/></button>;})}<div className="hangar-fleet"><span><Gauge size={13}/> 整队基础质量 <b>{fleetMass.toFixed(1)} kg</b></span><span><Battery size={13}/> 电池总容量 <b>{fleetBattery} Wh</b></span></div></nav>
        <div className="hangar-center">{model?<RobotPreview model={model} slot={activeSlot}/>:<div className="hangar-empty">正在加载机器人型号…</div>}
          <div className="hangar-candidates"><div><span className="hangar-section-label">CHASSIS SELECTION / 可选构型</span><small>{peers.length} 款{locked?' · 阵容已确认':''}</small></div><div className="hangar-model-grid">{peers.map(m=>{const legal=legalIn(m,season);return <button key={m.slot} disabled={locked||busy||!legal} aria-label={`选择${m.name}`} aria-pressed={model?.slot===m.slot} title={legal?undefined:`不符合 ${SEASON_LABELS[season]} 制作规范：${legalityIssues(m,season)}`} className={`${model?.slot===m.slot?'selected':''}${legal?'':' illegal'}`} onClick={()=>onChange({...value,[activeSlot]:m.slot})}><span>{code(m)}{model?.slot===m.slot&&<Check size={14}/>}{!legal&&<Ban size={13}/>}</span><b>{title(m)}</b><small>{legal?`${m.chassis_type==='hangar'?'固定机库':drives[m.chassis_type]} · ${m.bare_mass_kg.toFixed(1)}kg`:`不符合 ${season} 规范`}</small></button>;})}</div></div>
        </div>{model&&<ModelDetails model={model} peers={peers.filter(m=>legalIn(m,season))} season={season}/>}</div>
 <footer className="hangar-footer"><div><span><Shield size={15}/>{canEdit?'型号调整后需要重新准备':'只能配置自己的席位，房主可配置 AI 空位'}</span><small>{saving?'性能保存中，请稍候':'真人优先操控，断线后由所选 AI 接管'}</small></div><button onClick={onStrategies}>配置接管 AI <ChevronRight size={15}/></button></footer>
 {(active.slot===3||active.slot===4)&&<div className="hangar-performance"><PerformancePicker profile={seatProfile(room,active)} reason={profileDisabledReason(room,playerId,active,!offline)} save={ownSeat?profileSave:EMPTY_PROFILE_SAVE} onChange={profile=>invoke('profile',{...profile})} onReturnToOwn={!ownSeat&&me?.team&&me.slot?()=>onActive({team:me.team!,slot:me.slot!}):undefined}/></div>}
 </section></div>;
}
