import {useEffect,useState} from 'react';
import type {RoomState} from '../../../packages/renderer/src/lanTypes';
import {Stage} from './Ceremony';
import {roles,side,number} from './hudModel';
export function Results({room,isHost,ready,command,onSurface}:{room:RoomState;isHost:boolean;ready:boolean;command:(name:string)=>void;onSurface?:(page:string)=>void}){
 const [page,setPage]=useState('victory');useEffect(()=>{setPage('victory');const t=setTimeout(()=>setPage('results'),3500);return()=>clearTimeout(t);},[room.match_id]);
 useEffect(()=>{onSurface?.(page);},[page,onSurface]);
 const result=room.result;
 const robots=result?.robots||[],mvp=[...robots].sort((a,b)=>b.damage-a.damage||b.kills-a.kills)[0];
 return <Stage className={`lan-results ${page} ${result?.winner||'draw'}`}><div className="results-tint"/>
 <div className="results-heading"><span>RMUC 2027 · MATCH COMPLETE</span><h1>{result?.winner?side(result.winner)+'胜利':'比赛结束'}</h1><p>{({base_destroyed:'基地摧毁',control:'控制度判定',time:'比赛时间结束',timeout:'比赛时间结束'} as Record<string,string>)[result?.reason||'']||result?.reason||'本局对战已结束'}</p></div>
 {page==='victory'?<div className="victory-mark"><svg viewBox="-150 -150 300 300"><path d="M0 -130 112 -65 112 65 0 130 -112 65 -112 -65Z"/></svg><strong>{result?.winner==='red'?'R':result?.winner==='blue'?'B':'A'}</strong></div>:page==='mvp'&&mvp?<div className="mvp-panel"><span>MOST VALUABLE PLAYER</span><strong>{mvp.design_name}</strong><p>{side(mvp.team)} {mvp.slot} 号 · {roles[mvp.role]}</p><div><b>{number(mvp.damage)}<small>造成伤害</small></b><b>{mvp.kills}<small>击毁数</small></b></div><small>按造成伤害排序，伤害相同时比较击毁数</small></div>:<><div className="results-metrics">{(['red','blue'] as const).map(team=><div key={team} className={team}><h2>{side(team)}</h2><strong>{number(result?.teams?.[team].control,1)}<small>控制度</small></strong><p>击毁 {number(result?.teams?.[team].kills)} · 伤害 {number(result?.teams?.[team].attack_damage)}</p></div>)}</div><table className="results-table"><thead><tr><th>机器人</th><th>型号</th><th>兵种</th><th>击毁</th><th>伤害</th><th>剩余血量</th></tr></thead><tbody>{robots.map(r=><tr key={r.id} className={r.team}><td>{side(r.team)} {r.slot}</td><td>{r.design_name}</td><td>{roles[r.role]}</td><td>{r.kills}</td><td>{number(r.damage)}</td><td>{number(r.hp)}/{number(r.max_hp)}</td></tr>)}</tbody></table></>}
 <footer className="results-actions"><button onClick={()=>setPage('results')}>对战数据</button><button disabled={!mvp} onClick={()=>setPage('mvp')}>MVP</button><button onClick={()=>command('save')}>保存回放</button><button onClick={()=>command('ready')}>{ready?'已准备 · 取消':'准备下一局'}</button>{isHost&&<button disabled={!room.players?.every(p=>p.ready&&p.connected)} onClick={()=>command('start')}>开始下一局</button>}<button onClick={()=>command('room')}>返回大厅 / 配置</button><button onClick={()=>command('leave')}>离开房间</button></footer>
 </Stage>;
}
