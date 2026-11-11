import type {ControlPosition} from '../../../packages/renderer/src/types';
import type {LanPlayer,MapZone} from '../../../packages/renderer/src/lanTypes';
export const side=(s:string)=>s==='red'?'红方':s==='blue'?'蓝方':'中立';
export const roles:Record<string,string>={heavy:'重装',infantry:'步兵',aerial:'机库',sentry:'哨兵'};
export const zoneNames:Record<string,string>={base:'基地',outpost:'前哨',central:'中央高地',fortress:'堡垒',high_point:'高地',supply:'补给区',assembly:'装配区',resource:'能量单元',workbench:'工作台',rune:'能量机关',road:'公路',tunnel:'隧道',highland:'高地通道'};
export const number=(n:unknown,d=0)=>typeof n==='number'&&Number.isFinite(n)?n.toFixed(d):'—';
export const ratio=(v?:number,max?:number)=>typeof v==='number'&&typeof max==='number'&&max>0?Math.max(0,Math.min(1,v/max)):0;
export const clock=(n?:number)=>typeof n==='number'?`${Math.floor(Math.max(0,n)/60)}`.padStart(2,'0')+':'+`${Math.floor(Math.max(0,n)%60)}`.padStart(2,'0'):'—';
export function zoneLabel(z:MapZone){
 const name=z.id.replace(/^(red|blue)_/,''),owner=z.owner?side(z.owner):z.id.startsWith('red_')?'红方':z.id.startsWith('blue_')?'蓝方':'';
 const numbered=name.match(/^(road_tunnel|near_tunnel|far_tunnel|road)_(\d+)$/);
 const suffix=numbered?({road_tunnel:'公路隧道',near_tunnel:'近侧隧道',far_tunnel:'远侧隧道',road:'公路'} as Record<string,string>)[numbered[1]]+(Number(numbered[2])+1):zoneNames[z.kind||name]||z.kind||name;
 return owner+suffix;
}
export const assemblyOperation:Record<string,string>={positioning:'核心就位',inserting:'插入',lifting:'抬升',flipping:'翻转',translating:'平移',rotating:'旋转',verifying:'确认',buffered:'缓冲确认'};
export const assemblyReason:Record<string,string>={heavy_absent:'重装未就位',chassis_unstable:'底盘未停稳',arm_out_of_reach:'机械臂未到位',arm_misaligned:'机构姿态未对齐',heavy_destroyed:'重装战亡',pair_window:'双核心同步超时',timeout:'装配超时',left_assembly_area:'已离开装配区'};
export function positionDisplay(p:ControlPosition,rules:{capture_s:number;release_s:number}){
 const duration=p.control==='full'?rules.capture_s:rules.release_s;
 const both=p.active.length>1, holder=p.active.includes(p.controller);
 const text=both?'双方激活 · 进度暂停':p.control==='full'?(p.active.length&&!holder?'敌方夺取中':p.progress>0?'夺取进度回退':'保持控制'):holder?(p.progress>0?'释放进度回退':'保持临时控制'):'释放计时';
 return {fraction:ratio(p.progress,duration),duration,text};
}
export function buffs(p:LanPlayer){const r=p.self;return [r.invulnerable?'无敌':null,r.weak?'虚弱':null,r.overheated?'过热':null,r.powered===false?'底盘断电':null,r.overspeed_locked?'超速锁定':null,r.heat_permanent_lock?'热量永久锁定':null,r.launcher_fault?'发射机构故障':null,r.red_card?'红牌':null,r.yellow_cards?`黄牌 ${r.yellow_cards}`:null,(r.attack_multiplier||1)>1?`攻击 ×${number(r.attack_multiplier,2)}`:null,(r.defense_buff||0)>0?`防御 +${number((r.defense_buff||0)*100)}%`:null].filter(Boolean).join(' · ')||'状态正常';}
