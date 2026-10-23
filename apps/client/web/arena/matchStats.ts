import type {Frame,Robot,TeamId} from './types';
export const TEAMS=['red','blue'] as const;
export const TEAM_LABEL={red:'红方',blue:'蓝方'};
export const other=(team:TeamId):TeamId=>team==='red'?'blue':'red';
export const robotName=(r:Robot)=>({heavy:'重装',infantry:'步兵',sentry:'哨兵',aerial:'机库'})[r.role];
export function nickname(r:Robot){const parts=(r.design_name||'').split(' ');return {code:parts.length>1?parts[0]:'',name:parts.slice(parts.length>1?1:0).join(' ')||robotName(r)};}
export interface TeamTally {economy?:number;assemblies?:number[];shots17?:number;shots42?:number;runes?:{large:number;small:number};levelMax?:number;outpostsLost?:number}
export interface MatchStats {teams:Record<TeamId,TeamTally>}
export function displayStats(frame:Frame):MatchStats{
 const teams={} as MatchStats['teams'];
 for(const team of TEAMS){const robots=frame.robots.filter(r=>r.team===team);teams[team]={economy:frame.teams[team].total_resources,assemblies:frame.teams[team].completed_difficulties?.reduce<number[]>((list,d)=>{list[d-1]=1;return list;},[0,0,0,0]),levelMax:robots.length&&robots.every(r=>r.level!==undefined)?Math.max(...robots.map(r=>r.level!)):undefined,outpostsLost:typeof frame.outposts[team].ever_destroyed==='boolean'?Number(frame.outposts[team].ever_destroyed):undefined};}
 return {teams};
}
export function verdict(frame:Frame){return {headline:frame.winner?`${TEAM_LABEL[frame.winner]}胜利`:'双方平局',detail:({base_destroyed:'基地被摧毁',control:'控制度判定',timeout:'比赛时间耗尽，按主机裁判结果结算',time:'比赛时间耗尽',stopped:'比赛已结束'} as Record<string,string>)[frame.reason||'']||frame.reason||'以主机裁判结果为准'};}
export interface Mvp {robot:Robot;lines:{label:string;value:string;unit?:string}[];note?:string}
