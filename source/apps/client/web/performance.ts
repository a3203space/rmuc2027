import type {PerformanceProfile,ProfileResult,ProfileSaveState,RoomState} from '../../../packages/renderer/src/lanTypes';
import type {TeamId} from '../../../packages/renderer/src/types';
export const EMPTY_PROFILE_SAVE:ProfileSaveState={pending:null,status:'idle',message:''};
export function normalizeProfile(value?:{chassis?:string;weapon?:string}):PerformanceProfile{
 return {chassis:value?.chassis==='power'?'power':'hp',weapon:value?.weapon==='burst'?'burst':'cooling'};
}
export function seatProfile(room:RoomState|null,seat:{team:TeamId;slot:number}):PerformanceProfile{
 const player=room?.players?.find(p=>p.team===seat.team&&p.slot===seat.slot);
 return normalizeProfile(player?.profile);
}
export function profileDisabledReason(room:RoomState|null,playerId:string,seat:{team:TeamId;slot:number},online=true):string{
 if(!online)return '客户端连接已中断，请重连后选择性能';
 const me=room?.players?.find(p=>p.id===playerId);
 if(!me?.team||!me.slot)return '请先在大厅加入自己的 3 或 4 号步兵席位';
 if(!room||!['lobby','finished'].includes(room.status))return '比赛开始后性能已锁定，下一局准备阶段可修改';
 if(![3,4].includes(seat.slot))return '仅 3 或 4 号步兵支持性能选择';
 if(me.team!==seat.team||me.slot!==seat.slot)return '正在查看其他席位；只能修改自己的步兵性能';
 if(!me.connected)return '连接已断开，请重连后选择性能';
 return '';
}
export function beginProfileSave(state:ProfileSaveState,id:string,profile:PerformanceProfile):ProfileSaveState{
 return state.pending?state:{pending:{id,profile:{...profile}},status:'saving',message:'保存中，等待主机确认…'};
}
export function finishProfileSave(state:ProfileSaveState,result:ProfileResult):ProfileSaveState{
 if(state.pending?.id!==result.request_id)return state;
 return {pending:null,status:result.ok?'saved':'error',message:result.ok?'已保存 · 性能修改后需重新准备':result.message||'保存失败，已恢复主机确认的配置'};
}
/** HTTP replies and WebSocket snapshots can arrive out of revision order. */
export function newerRoom(current:RoomState|null,next:RoomState|null):RoomState|null{
 if(current&&next&&typeof current.revision==='number'&&typeof next.revision==='number'&&next.revision<current.revision)return current;
 return next;
}
