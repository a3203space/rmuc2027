import {useCallback,useEffect,useRef,useState} from 'react';
import type {RoomState,RuntimeCatalog} from '../../../packages/renderer/src/lanTypes';

export interface CatalogStatus {loading:boolean;error:string;retry:()=>void}
const EMPTY:RuntimeCatalog={models:{},strategies:[]};
function validCatalog(value:RuntimeCatalog):boolean{
 return !!value?.models&&Object.keys(value.models).length>0&&Array.isArray(value.strategies)&&
  Object.values(value.models).every(model=>!!model&&typeof model.name==='string'&&typeof model.role==='string'&&
   typeof model.asset==='string'&&Number.isFinite(model.slot)&&Number.isFinite(model.mass_kg));
}

/** Local hardware is available before any LAN session; host strategies stay session-owned. */
export function useCatalog(room:RoomState|null,sessionId:string,invoke:(name:string)=>void){
 const connected=!!room&&room.status!=='closed',revision=room?.catalog_revision??0;
 const [local,setLocal]=useState<RuntimeCatalog|null>(null),[localError,setLocalError]=useState(''),[localAttempt,setLocalAttempt]=useState(0);
 const [remote,setRemote]=useState<{sessionId:string;catalog:RuntimeCatalog}|null>(null),[remoteError,setRemoteError]=useState(''),[remoteAttempt,setRemoteAttempt]=useState(0);
 const context=useRef({connected,sessionId});context.current={connected,sessionId};
 const command=useRef(invoke);command.current=invoke;
 useEffect(()=>{
  let disposed=false;const abort=new AbortController(),deadline=setTimeout(()=>abort.abort(),5000);
  setLocalError('');
  fetch('/catalog.json',{signal:abort.signal,cache:'no-store'}).then(response=>{
   if(!response.ok)throw Error(`HTTP ${response.status}`);return response.json();
  }).then((value:RuntimeCatalog)=>{
   if(!validCatalog(value))throw Error('构型数据不完整');
   if(!disposed)setLocal({models:value.models,strategies:[]});
  }).catch(error=>{if(!disposed){setLocalError('本地机器人构型库加载失败，请检查资源并重试');console.error('ARENA_CATALOG_LOCAL',String(error));}})
   .finally(()=>clearTimeout(deadline));
  return()=>{disposed=true;clearTimeout(deadline);abort.abort();};
 },[localAttempt]);
 useEffect(()=>{setRemoteError('');if(!connected)setRemote(null);},[connected,sessionId]);
 const current=remote?.sessionId===sessionId?remote.catalog:null;
 const synced=!!current&&current.revision===revision;
 useEffect(()=>{
  if(!connected||synced)return;
  let attempts=0;setRemoteError('');
  const request=()=>{attempts++;command.current('catalog');};request();
  const timer=setInterval(()=>{
   if(attempts<3)request();
   else {clearInterval(timer);setRemoteError('主机机器人构型库同步失败，请重试');console.error('ARENA_CATALOG_REMOTE','catalog request timed out',revision);}
  },5000);
  return()=>clearInterval(timer);
 },[connected,sessionId,revision,synced,remoteAttempt]);
 const acceptRemote=useCallback((value:RuntimeCatalog)=>{
  if(!context.current.connected)return;
  if(!validCatalog(value)){setRemoteError('主机返回的机器人构型库不完整，请重试');return;}
  setRemoteError('');setRemote({sessionId:context.current.sessionId,catalog:value});
 },[]);
 const retry=useCallback(()=>{
  if(context.current.connected){setRemote(null);setRemoteError('');setRemoteAttempt(n=>n+1);}
  if(!local||localError)setLocalAttempt(n=>n+1);
 },[local,localError]);
 const catalog=connected&&current?current:local||EMPTY;
 const status:CatalogStatus={loading:connected?!synced&&!remoteError:!local&&!localError,error:connected?remoteError:localError,retry};
 return {catalog,status,acceptRemote};
}
