import type {Arena,Frame as SimulationFrame} from '../../../../packages/renderer/src/types';
import type {LanPlayer,RoomState,RuntimeCatalog} from '../../../../packages/renderer/src/lanTypes';
import type {Frame,Metadata,Robot,TeamId} from './types';
import {detectProvider} from './agents';

export function matchMetadata(room:RoomState|null,catalog:RuntimeCatalog):Metadata{
 const teams={} as Metadata['teams'],agents={} as NonNullable<Metadata['agents']>;
 for(const team of ['red','blue'] as const){
  const config=room?.team_configs?.[team],strategy=catalog.strategies.find(s=>s.id===config?.strategy);
  teams[team]={name:team==='red'?'红方':'蓝方',display_name:config?.mode==='team'?strategy?.name:undefined,version:room?.match_id||'',mode:config?.mode==='team'?'team':'lan'};
  agents[team]={};
  for(const [slot,seat] of Object.entries(room?.seat_configs?.[team]||{})){
   const driver=room?.players?.find(p=>p.team===team&&p.slot===Number(slot)),entry=catalog.strategies.find(s=>s.id===(config?.mode==='team'?config.strategy:seat.strategy));
   agents[team][slot]={provider:driver?'human':detectProvider(entry?.author||'')||'unknown',model:driver?.nickname||entry?.author||entry?.name||'未标注',strategy:entry?.name,driver:!!driver};
  }
 }
 return {teams,agents};
}

/** Public radar and the owner's private telemetry are UI data, never a render frame. */
export function operatorView(player:LanPlayer|null,arena:Arena|undefined,room:RoomState|null,catalog:RuntimeCatalog):Frame|null{
 if(!player?.tactical||player.display.ui_blocked||!arena)return null;
 const tactical=player.tactical;
 const robots:Robot[]=tactical.robots.map(r=>{
  const modelId=room?.seat_configs?.[r.team]?.[String(r.slot)]?.model,model=modelId===undefined?undefined:catalog.models[String(modelId)];
  const own=r.id===player.self.id?player.self:undefined,exterior=player.scene?.robots?.find(robot=>robot.id===r.id);
  // Only public identity/hardware is enriched. Unobserved private values stay absent.
  return {...r,...own,model_slot:modelId??own?.model_slot??exterior?.model_slot,model_asset:model?.asset||own?.model_asset||exterior?.model_asset,design_name:model?.name||own?.design_name||exterior?.design_name};
 });
 return {arena,robots,robot_models:catalog.models,bases:tactical.bases,outposts:tactical.outposts,positions:tactical.positions,teams:{...tactical.teams,[player.team]:{...tactical.teams[player.team],completed_difficulties:player.economy?.completed_difficulties}},
  zones:tactical.zones,assemblies:{[player.team]:player.assembly},runes:player.scene?.runes,dt:player.scene?.dt,time:player.scene?.time??tactical.sample_time,remaining:player.scene?.remaining??0,duration:player.scene?.duration??420,tick:player.tick,status:player.scene?.status||room?.status||'running',winner:player.scene?.winner,reason:player.scene?.reason,season:'2027'};
}

export function presentationView(frame:SimulationFrame):Frame{return {...frame,zones:undefined,season:'2027'};}
export function resultView(room:RoomState,catalog:RuntimeCatalog,arena?:Arena):Frame|null{
 const result=room.result,map=result?.arena||arena;if(!result?.bases||!result.outposts||!map||!result.teams)return null;
 return {arena:map,bases:result.bases,outposts:result.outposts,positions:result.positions,teams:result.teams,robots:(result.robots||[]).map(r=>{
  const model=catalog.models[String(r.model_slot)];return {...r,x:0,y:0,alive:r.hp>0,model_asset:model?.asset||r.model_asset,design_name:r.design_name||model?.name};
 }),time:result.time??0,remaining:0,duration:result.time??420,tick:0,status:'finished',winner:result.winner,reason:result.reason,season:'2027'};
}
export const side=(team:TeamId)=>team==='red'?'红方':'蓝方';
