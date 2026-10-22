import {useEffect,useRef,useState} from 'react';
import {createRoot} from 'react-dom/client';
import {ArenaScene} from '../../packages/renderer/src/ArenaScene';
import type {Frame,RenderFeed} from '../../packages/renderer/src/types';
import './style.css';
type Header={metadata:{map:string};static?:Partial<Frame>;count:number};
type Replay={metadata:{map:string};static?:Partial<Frame>;frames:Frame[]};
function App(){
 const [header,setHeader]=useState<Header|null>(null),[index,setIndex]=useState(0),[playing,setPlaying]=useState(false),[error,setError]=useState(''),[frame,setFrame]=useState<Frame|null>(null);
 const local=useRef<Replay|null>(null),generation=useRef(0),cache=useRef(new Map<number,Frame>());
 const feed=useRef<RenderFeed>({frame:null,previous:null,received:0,speed:1,paused:true,events:[],generation:'',replay:true});
 useEffect(()=>{fetch('/api/replay/header').then(r=>{if(!r.ok)throw Error();return r.json();}).then((h:Header)=>{if(h.metadata.map!=='rmuc2027')throw Error('仅支持 RMUC 2027');setHeader(h);}).catch(()=>{});},[]);
 useEffect(()=>{
  if(!header)return;const epoch=generation.current;let cancelled=false;
  const show=(next:Frame)=>{if(cancelled||epoch!==generation.current)return;const merged={...header.static,...next} as Frame;feed.current={...feed.current,previous:feed.current.frame,frame:merged,received:performance.now(),paused:!playing,generation:String(epoch)};setFrame(merged);};
  const hit=local.current?.frames[index]||cache.current.get(index);
  if(hit){show(hit);return;}
  fetch(`/api/replay/frames?start=${index}&count=60`).then(r=>{if(!r.ok)throw Error('读取回放失败');return r.json();}).then((result:{start:number;frames:Frame[]})=>{
   if(cancelled||epoch!==generation.current)return;
   cache.current.clear();result.frames.forEach((f,i)=>cache.current.set(result.start+i,f));if(result.frames[0])show(result.frames[0]);
  }).catch(err=>{if(!cancelled)setError(String(err));});
  return()=>{cancelled=true;};
 },[header,index,playing]);
 useEffect(()=>{if(!playing||!header)return;const timer=setInterval(()=>setIndex(i=>{if(i>=header.count-1){setPlaying(false);return i;}if(!local.current&&!cache.current.has(i))return i;return i+1;}),50);return()=>clearInterval(timer);},[playing,header]);
 const open=async(file:File)=>{try{
  if(file.size>64*1024*1024)throw Error('大回放请使用 rmuc2027 replay 文件路径，按帧读取，避免一次加载整局');
  const value=JSON.parse(await file.text()) as Replay;if(value.metadata?.map!=='rmuc2027'||!value.frames?.length)throw Error('请选择有效的 RMUC 2027 回放');
  generation.current++;setPlaying(false);local.current=value;cache.current.clear();feed.current.previous=null;setIndex(0);setHeader({metadata:value.metadata,static:value.static,count:value.frames.length});setError('');
 }catch(err){setError(String(err));}};
 return <main><header><strong>RMUC 2027 回放</strong><input type="file" accept=".json" onChange={e=>{const file=e.target.files?.[0];if(file)void open(file);}}/><button disabled={!header} onClick={()=>setPlaying(!playing)}>{playing?'暂停':'播放'}</button><input type="range" min="0" max={header?header.count-1:0} value={index} onChange={e=>{setPlaying(false);setIndex(Number(e.target.value));feed.current.previous=null;}}/><span>{frame?.time.toFixed(1)||'0.0'} 秒</span></header>{error&&<p>{error}</p>}<section>{frame?<ArenaScene frame={frame} feed={feed} mode="orbit" selected="" onSelect={()=>{}} onCanvas={()=>{}} onRender={()=>{}}/>:<p>打开本地回放，或使用 rmuc2027 replay 命令。</p>}</section></main>;
}
createRoot(document.getElementById('root')!).render(<App/>);
