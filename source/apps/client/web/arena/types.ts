import type {Robot as SimulationRobot,Arena,ControlPosition,Structure,TeamId,RobotHardware,Frame as SimulationFrame,MatchEvent} from '../../../../packages/renderer/src/types';
import type {AssemblyState,ZoneState,RadarRobot} from '../../../../packages/renderer/src/lanTypes';
export type {TeamId,RobotHardware};
export type {MatchEvent} from '../../../../packages/renderer/src/types';
export type CatalogEntry={name:string;display_name?:string;version:string};
export {ROLE_NAMES,TEAM_COLORS,baseProtected,outpostHolder} from '../../../../packages/renderer/src/types';
export type Robot=Pick<SimulationRobot,'id'|'slot'|'team'|'role'|'x'|'y'|'hp'|'max_hp'|'alive'>&Partial<Omit<SimulationRobot,'id'|'slot'|'team'|'role'|'x'|'y'|'hp'|'max_hp'|'alive'>>&{buffs?:RadarRobot['buffs']};
export interface Frame {
 arena:Arena;robots:Robot[];robot_models?:Record<string,RobotHardware>;bases:Record<TeamId,Structure>;outposts:Record<TeamId,Structure>;
 teams:Record<TeamId,{resources:number;total_resources?:number;score:number;control?:number|null;kills?:number;attack_damage?:number;completed_difficulties?:number[]}>;
 positions?:Record<string,ControlPosition>;zones?:Record<string,ZoneState>;assemblies?:Partial<Record<TeamId,AssemblyState>>;runes?:SimulationFrame['runes'];
 time:number;remaining:number;duration:number;tick:number;dt?:number;events?:MatchEvent[];status:string;winner?:TeamId|null;reason?:string|null;season:'2027';
}
export interface Metadata {teams:Record<TeamId,{name:string;display_name?:string;version:string;mode:string;agent?:import('./agents').RobotAgent}>;agents?:Record<TeamId,Record<string,import('./agents').RobotAgent>>}
export interface Standing{name:string;version:string;games:number;wins:number;losses:number;draws:number;rating:number}
export const MODE_LABELS:Record<string,string>={lan:'局域网对战',human:'真人操控',team:'全队策略',independent:'独立机器人',adaptive:'型号适配',squad:'单机组队',distributed:'多机协同',robot:'单机策略'};
export const frameSeason=(_frame:unknown)=>'2027' as const;
