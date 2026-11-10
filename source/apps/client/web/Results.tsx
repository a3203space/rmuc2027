import {useEffect,useMemo,useState} from 'react';
import type {RoomState,RuntimeCatalog} from '../../../packages/renderer/src/lanTypes';
import type {Arena} from '../../../packages/renderer/src/types';
import {Victory,Results as ArenaResults,MvpCard} from './arena/ceremony';
import {resultView,matchMetadata} from './arena/viewModel';
import {displayStats,type Mvp} from './arena/matchStats';
import {useRobotImages} from './arena/useRobotImages';
const EMPTY_CATALOG:RuntimeCatalog={models:{},strategies:[]};
export function Results({room,isHost,ready,command,onSurface,configBusy=false,catalog=EMPTY_CATALOG,arena}:{room:RoomState;isHost:boolean;ready:boolean;command:(name:string)=>void;onSurface?:(page:string)=>void;configBusy?:boolean;catalog?:RuntimeCatalog;arena?:Arena}){
 const [page,setPage]=useState('victory');
 useEffect(()=>{setPage('victory');const timer=setTimeout(()=>setPage('results'),3500);return()=>clearTimeout(timer);},[room.match_id]);
 useEffect(()=>{onSurface?.(page);},[page,onSurface]);
 const frame=useMemo(()=>resultView(room,catalog,arena),[room,catalog,arena]),metadata=useMemo(()=>matchMetadata(room,catalog),[room,catalog]),images=useRobotImages(frame?.robots||[],true);
 const mvp=useMemo<Mvp|null>(()=>{const robot=frame&&[...frame.robots].sort((a,b)=>(b.damage||0)-(a.damage||0)||(b.kills||0)-(a.kills||0))[0];return robot?{robot,lines:[{label:'造成伤害',value:String(Math.round(robot.damage||0))},{label:'击毁数',value:String(robot.kills||0)},{label:'剩余血量',value:`${robot.hp}/${robot.max_hp}`}],note:'按造成伤害排序，伤害相同时比较击毁数'}:null;},[frame]);
 return <div className={`lan-results ${page}`}>
 {!frame?<div className="scene-loading">正在汇总比赛结果…</div>:page==='victory'?<Victory frame={frame}/>:page==='mvp'&&mvp?<MvpCard mvp={mvp} metadata={metadata} image={images[`${mvp.robot.model_asset}|${mvp.robot.team}|showcase`]} onBack={()=>setPage('results')} onClose={()=>setPage('field')}/>:page==='results'?<ArenaResults frame={frame} metadata={metadata} stats={displayStats(frame)} images={images} auto={false} onMvp={()=>setPage('mvp')} onClose={()=>setPage('field')}/>:null}
 <footer className="lan-results-actions"><button onClick={()=>setPage('results')}>对战数据</button><button disabled={!mvp} onClick={()=>setPage('mvp')}>MVP</button><button onClick={()=>command('save')}>保存回放</button><button disabled={configBusy} onClick={()=>command('ready')}>{ready?'已准备 · 取消':'准备下一局'}</button>{isHost&&<button disabled={configBusy||!room.players?.every(p=>p.ready&&p.connected)} onClick={()=>command('start')}>开始下一局</button>}<button onClick={()=>command('room')}>返回大厅 / 配置</button><button onClick={()=>command('leave')}>离开房间</button></footer>
 </div>;
}
