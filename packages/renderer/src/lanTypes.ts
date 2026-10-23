import type {Arena,ControlPosition,Drone,Frame,Robot,Structure,TeamId} from './types';
export interface PerformanceProfile {chassis:'hp'|'power';weapon:'cooling'|'burst'}
export interface ProfileResult {request_id:string;ok:boolean;message?:string;room?:RoomState}
export interface ProfileSaveState {pending:{id:string;profile:PerformanceProfile}|null;status:'idle'|'saving'|'saved'|'error';message:string}
export interface PlayerCamera {origin:number[];direction:number[];up?:number[];horizontal_fov_deg:number;vertical_fov_deg:number}
export interface ZoneState {active:boolean;owner:TeamId|null;progress:number;contested:boolean;active_teams:TeamId[]}
export interface RadarRobot {id:string;slot:number;team:TeamId;role:string;x:number;y:number;heading?:number;hp:number;max_hp:number;alive:boolean;ammo:number;armor_hit_times:number[];buffs:Record<string,number|string|boolean>}
export interface OperatorAction {name:string;value:string;label:string;enabled:boolean;reason:string;cost:number|null;detail:string}
export interface AssemblyState {stage?:string;operation?:string;difficulty?:number;progress?:number;operation_progress?:number;blocked_reason?:string;inserted_units?:number;buffer_until?:number;permanent_lock?:boolean}
export interface LanPlayer {
  protocol:number;match_id:string;tick:number;team:TeamId;slot:number;
  self:Robot & {chassis_profile?:string;weapon_profile?:string;posture_ready_at?:number;pending_heal?:number|null;pending_ammo?:number[][];last_raw_zones?:string[];attack_multiplier?:number;defense_buff?:number;vulnerability?:number};
  display:{ui_blocked:boolean;blocked_until?:number;camera_off:boolean;reason:string;logistics:number;resolution_scale:number;camera:PlayerCamera|null};
  tactical?:{robots:RadarRobot[];teams:Record<TeamId,{resources:number;total_resources?:number;control:number;score:number}>;zones:Record<string,ZoneState>;positions:Record<string,ControlPosition>;bases:Record<TeamId,Structure>;outposts:Record<TeamId,Structure>;sample_time:number;decoded:boolean;control_rules:{capture_s:number;release_s:number}};
  scene?:Partial<Frame>;control?:{needs_resume:boolean;status:string;ack_seq:number;manual:boolean;aim_assist?:{state:'idle'|'searching'|'tracking'|'lost';target_id:string|null};last_press_seq?:number;last_press_result?:string};
  weapon?:{ready:boolean;reason:string;detail:string;muzzle_speed_mps?:number|null;fire_rate_hz?:number|null};
  actions?:OperatorAction[];assembly?:AssemblyState;air?:Drone[];
  economy?:{completed_difficulties?:number[];energy_units_left?:number;launcher_unlock_s?:number;level_cap?:number};
  events?:{type:string;id?:string;tick?:number;amount?:number;cost?:number;message?:string;remote?:boolean}[];
}
export interface RoomState {revision?:number;catalog_revision?:number;seat_configs?:Record<TeamId,Record<string,SeatConfiguration>>;team_configs?:Record<TeamId,{mode:'team'|'independent';strategy:string}>;stage_elapsed?:number;stage_duration?:number;policy_status?:Record<string,string>;lineups?:Record<TeamId,Record<string,number>>;status:string;match_id?:string;error?:string;players?:{id:string;nickname:string;team:TeamId|null;slot:number|null;ready:boolean;host:boolean;connected:boolean;stage_loaded?:boolean;profile?:{chassis:string;weapon:string}}[];result?:MatchResult}
export interface Receipt {seq:number;status:string;name?:string;value?:string;message?:string;tick?:number}
export interface LanSnapshot {presentation?:Partial<Frame>;player:LanPlayer|null;static?:Partial<Frame>;room?:RoomState;player_id?:string;operations?:Receipt[];transport?:boolean}
export interface ArenaBridge {
  response:{connect:(fn:(name:string,payload:string)=>void)=>void};
  snapshot:{connect:(fn:(payload:string)=>void)=>void};
  notice:{connect:(fn:(payload:string)=>void)=>void};
  room:{connect:(fn:(payload:string)=>void)=>void};
  input:(payload:string)=>void;command:(payload:string)=>void;
  viewportReady:()=>void;pause:()=>void;renderStats:(payload:string)=>void;
}
export type MapZone=Arena['zones'][number] & {kind?:string;owner?:TeamId};
declare global {interface Window {qt?:{webChannelTransport:unknown};QWebChannel?:new(transport:unknown,callback:(channel:{objects:{arena:ArenaBridge}})=>void)=>unknown}}
export interface StrategyEntry {id:string;version:string;mode:string;robot:number|null;lineup:Record<string,number>|null;name:string;author:string;description:string;flexible:boolean}
export interface RuntimeCatalog {revision?:number;models:Record<string,import('./types').RobotHardware>;strategies:StrategyEntry[]}
export interface SeatConfiguration {model:number;strategy:string}
export interface AppSettings {nickname:string;address:string;port:number;runtime:string;sensitivity?:number;volume?:number}

export interface FinalRobotSummary {id:string;team:TeamId;slot:number;model_slot:number;design_name:string;role:string;kills:number;damage:number;hp:number;max_hp:number}
export interface MatchResult {winner?:TeamId|null;reason?:string;robots?:FinalRobotSummary[];teams?:Record<TeamId,{control?:number;resources?:number;kills?:number;attack_damage?:number}>;replay?:string;digest?:string;policy_errors?:number}
