import {useEffect,useMemo,useRef,useState} from 'react';
import type {Frame} from '../../../packages/renderer/src/types';
import type {RoomState,RuntimeCatalog} from '../../../packages/renderer/src/lanTypes';
import {introAt,introSchedule} from '../../../packages/renderer/src/intro';
import type {IntroShot} from '../../../packages/renderer/src/ArenaScene';
import {Prematch} from './arena/ceremony';
import {Stage} from './arena/broadcast';
import {Faceoff} from './arena/LineupIntro';
import {matchMetadata,presentationView} from './arena/viewModel';
import {useRobotImages} from './arena/useRobotImages';
import './arena-broadcast.css';
import './arena-intro.css';
export {Stage} from './arena/broadcast';
export function useStageClock(room:RoomState|null){
 const anchor=useRef({at:performance.now(),elapsed:0}),[elapsed,setElapsed]=useState(0);
 useEffect(()=>{anchor.current={at:performance.now(),elapsed:room?.stage_elapsed||0};setElapsed(room?.stage_elapsed||0);},[room?.status,room?.stage_elapsed]);
 useEffect(()=>{const timer=setInterval(()=>setElapsed(anchor.current.elapsed+(performance.now()-anchor.current.at)/1000),33);return()=>clearInterval(timer);},[]);
 return elapsed;
}
export function introductionShot(frame:Frame|null,room:RoomState|null,elapsed:number):IntroShot|null{
 if(!frame||!room)return null;
 if(room.status==='intro'){const at=introAt(introSchedule(frame.robots,room.stage_duration||38.3),elapsed);return at?{key:at.cut.key,segment:at.cut.segment,length:at.length,progress:at.progress}:null;}
 return {key:'prepare',segment:{kind:'faceoff'},length:1,progress:1};
}
export function Ceremony({frame,room,catalog,elapsed,isHost,skip}:{frame:Frame;room:RoomState;catalog:RuntimeCatalog;elapsed:number;isHost:boolean;skip:()=>void}){
 const view=useMemo(()=>presentationView(frame),[frame]),metadata=useMemo(()=>matchMetadata(room,catalog),[room,catalog]),images=useRobotImages(view.robots);
 const remaining=Math.max(0,(room.stage_duration||0)-elapsed);
 if(room.status==='loading'||room.status==='starting')return <Stage className="cer-lineup cer-loading"><Faceoff frame={view} metadata={metadata} images={images} progress={0}/><div className="li-bug"><i/>加载战场与策略<b>{Math.ceil(remaining)}s</b></div><div className="lan-loading-players">{room.players?.map(player=><span key={player.id}>{player.stage_loaded?'✓':'◌'} {player.nickname}</span>)}</div></Stage>;
 return <Prematch frame={view} metadata={metadata} standings={[]} images={images} models={catalog.models} duration={room.stage_duration||38.3} remaining={remaining} stage={room.status==='countdown'?'countdown':'self_check'} free={false} paused={false} locked={!isHost} onCut={()=>{}} onSkip={skip}/>;
}
export function useArenaMusic(phase:string,volume:number,elapsed=0){
 const player=useRef<HTMLAudioElement|null>(null),latest=useRef({phase,elapsed});latest.current={phase,elapsed};
 const file=({home:'home.mp3',lobby:'lobby.m4a',hangar:'hangar.m4a',starting:'lobby.m4a',loading:'lobby.m4a',intro:'self-check.m4a',countdown:'countdown.m4a',finished:'victory.m4a',victory:'victory.m4a',results:'results.m4a',mvp:'mvp.m4a',settling:'results.m4a'} as Record<string,string>)[phase];
 useEffect(()=>{if(!file){player.current?.pause();player.current=null;return;}const audio=new Audio('/assets/audio/'+file);player.current=audio;audio.volume=volume/100;audio.loop=['home','lobby','hangar','loading','starting'].includes(phase);const play=()=>{if(audio===player.current&&volume>0)void audio.play().catch(()=>{});};audio.addEventListener('loadedmetadata',()=>{if(['intro','countdown'].includes(latest.current.phase))audio.currentTime=Math.min(latest.current.elapsed,Math.max(0,audio.duration-.1));play();});window.addEventListener('pointerdown',play);return()=>{audio.pause();audio.src='';window.removeEventListener('pointerdown',play);};},[file]);
 useEffect(()=>{if(player.current){player.current.volume=volume/100;if(!volume)player.current.pause();else void player.current.play().catch(()=>{});}},[volume]);
}
