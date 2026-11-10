import {useCallback,useEffect,useRef,useState} from 'react';
import type {PerformanceProfile,ProfileResult,ProfileSaveState} from '../../../packages/renderer/src/lanTypes';
import {beginProfileSave,EMPTY_PROFILE_SAVE,finishProfileSave} from './performance';
export function useProfileSave(owner:string,enabled:boolean){
 const [state,setState]=useState(EMPTY_PROFILE_SAVE),current=useRef(state),sequence=useRef(0),available=useRef(enabled);
 available.current=enabled;
 const commit=useCallback((next:ProfileSaveState)=>{current.current=next;setState(next);},[]);
 const begin=useCallback((profile:PerformanceProfile)=>{
  if(current.current.pending||!available.current)return null;
  const id=`profile-${Date.now()}-${++sequence.current}`;
  commit(beginProfileSave(current.current,id,profile));return id;
 },[commit]);
 const complete=useCallback((result:ProfileResult)=>commit(finishProfileSave(current.current,result)),[commit]);
 const isPending=useCallback(()=>!!current.current.pending,[]);
 useEffect(()=>{commit(EMPTY_PROFILE_SAVE);},[owner,commit]);
 useEffect(()=>{
  if(!enabled&&current.current.pending)complete({request_id:current.current.pending.id,ok:false,message:'当前无法保存，已恢复主机配置；请回到准备阶段重试'});
 },[enabled,complete]);
 useEffect(()=>{
  if(!state.pending)return;
  const id=state.pending.id;
  const timer=setTimeout(()=>complete({request_id:id,ok:false,message:'保存超时，当前显示主机配置；连接恢复后可重试'}),7000);
  return()=>clearTimeout(timer);
 },[state.pending,complete]);
 return {state,begin,complete,isPending};
}
