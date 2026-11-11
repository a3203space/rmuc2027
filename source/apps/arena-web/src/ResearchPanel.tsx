import {useEffect,useState} from 'react';
import {Activity,Play,RotateCw,Shield} from 'lucide-react';
import type {Metadata,ResearchReplay,ResearchRun,ResearchStart} from './types';
import './research.css';

async function read<T>(path:string):Promise<T>{const response=await fetch(path);const data=await response.json();if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:'研究目录读取失败');return data;}
export function researchReplayUrl(run:string,replay:string){return `/api/research/runs/${encodeURIComponent(run)}/replays/${encodeURIComponent(replay)}`;}
export function researchStart(run:ResearchRun,checkpoint:string,scenario:string,profile:string,backend:'rapier'|'warp_kinematic',seed:number,duration:number):ResearchStart{
  const selected=run.scenarios.find(row=>row.id===scenario),capability=run.capability_profiles.find(row=>row.id===profile);
  if(!run.checkpoints.some(row=>row.id===checkpoint)||!selected||!capability)throw new Error('请选择已登记的权重、阵容和能力档');
  if(capability.learner_side&&capability.learner_side!==selected.learner_side)throw new Error('能力档与主哨兵所在方不匹配');
  if(!Number.isInteger(seed)||seed<-(2**31)||seed>2**31-1||!Number.isInteger(duration)||duration<1||duration>420)throw new Error('随机种子或时长无效');
  return {mode:'research',research:{run_id:run.run_id,checkpoint_id:checkpoint,scenario_id:scenario,capability_profile_id:profile,backend},seed,duration,save_mp4:false};
}
export function ResearchBanner({metadata,onResearch}:{metadata:Metadata|null;onResearch:()=>void}){
  const provenance=metadata?.research,recording=metadata?.recording;
  const training=metadata?.kind==='training'||recording?.kind==='training';
  const evaluation=metadata?.kind==='evaluation';
  if(!provenance&&!recording&&!training&&!evaluation)return null;
  const partial=recording?.truncated??metadata?.truncated;
  return <div className="research-banner"><Shield size={16}/><div><strong>{training?'已保存训练回放':evaluation?'已保存验收回放':'冻结研究对局'}{metadata?.test_match?' · 不计 Elo':''}</strong><span>{provenance?.run_id} {provenance?.checkpoint_label||provenance?.checkpoint_id} · {provenance?.backend||metadata?.physics_backend||'后端未标注'}{provenance?.checkpoint_sha256&&` · SHA ${provenance.checkpoint_sha256.slice(0,12)}`}{partial?' · 截断片段':recording?.horizon_completed?' · 已完成设定时长':''}{training&&metadata?.training_recording?.policy_changes_during_episode?' · 含策略更新':''}{recording?.frame_hz?` · ${recording.frame_hz} Hz 全帧`:''}</span></div><button onClick={onResearch}>研究目录</button></div>;
}
export function ResearchPanel({busy,onStart,onReplay}:{busy:boolean;onStart:(request:ResearchStart)=>Promise<void>;onReplay:(run:string,replay:string)=>Promise<void>}){
  const [runs,setRuns]=useState<ResearchRun[]>([]),[runId,setRunId]=useState(''),[checkpoint,setCheckpoint]=useState(''),[scenario,setScenario]=useState(''),[profile,setProfile]=useState('');
  const [backend,setBackend]=useState<'rapier'|'warp_kinematic'>('rapier'),[seed,setSeed]=useState(81001),[duration,setDuration]=useState(60),[replays,setReplays]=useState<ResearchReplay[]>([]);
  const [loading,setLoading]=useState(false),[working,setWorking]=useState(''),[error,setError]=useState(''),[catalogErrors,setCatalogErrors]=useState<string[]>([]);
  const run=runs.find(row=>row.run_id===runId),selected=run?.scenarios.find(row=>row.id===scenario);
  async function refresh(){setLoading(true);setError('');try{const data=await read<{runs:ResearchRun[];errors:{run_id:string;error:string}[]}>('/api/research/runs');setRuns(data.runs);setCatalogErrors(data.errors.map(row=>`${row.run_id}: ${row.error}`));setRunId(old=>data.runs.some(row=>row.run_id===old)?old:data.runs[0]?.run_id||'');}catch(e){setError((e as Error).message);}finally{setLoading(false);}}
  useEffect(()=>{void refresh();},[]);
  useEffect(()=>{setCheckpoint(run?.checkpoints[0]?.id||'');setScenario(run?.scenarios[0]?.id||'');setProfile(run?.capability_profiles[0]?.id||'');setReplays([]);if(!run)return;let active=true;read<ResearchReplay[]>(`/api/research/runs/${encodeURIComponent(run.run_id)}/replays`).then(rows=>{if(active)setReplays(rows);}).catch(e=>{if(active)setError(e.message);});return()=>{active=false;};},[run]);
  const profiles=run?.capability_profiles.filter(row=>!row.learner_side||row.learner_side===selected?.learner_side)||[];
  useEffect(()=>{if(!profiles.some(row=>row.id===profile))setProfile(profiles[0]?.id||'');},[scenario,run,profile]);
  async function start(){if(!run)return;setWorking('start');setError('');try{await onStart(researchStart(run,checkpoint,scenario,profile,backend,seed,duration));}catch(e){setError((e as Error).message);}finally{setWorking('');}}
  async function open(row:ResearchReplay){setWorking(row.id);setError('');try{await onReplay(runId,row.id);}catch(e){setError((e as Error).message);}finally{setWorking('');}}
  return <section className="archive-page research-page"><div className="archive-heading"><div><span className="eyebrow">FROZEN RESEARCH</span><h1>研究权重与真实回放</h1><p>从已登记的权重启动冻结对局，或观看保存的训练与验收全帧。这里的对局不计 Elo。</p></div><button disabled={loading||!!working} onClick={refresh}><RotateCw size={14}/>刷新目录</button></div>
    {error&&<div role="alert" className="research-error">{error}</div>}{catalogErrors.length>0&&<details className="research-error"><summary>有 {catalogErrors.length} 个目录暂不可用</summary>{catalogErrors.map(row=><p key={row}>{row}</p>)}</details>}
    {!runs.length?<div className="empty-archive"><Activity size={28}/><p>{loading?'正在读取研究目录…':'尚无已登记研究运行'}</p><span>研究运行登记后会出现在这里。</span></div>:<><div className="research-form"><label className="research-wide">研究运行<select value={runId} onChange={e=>setRunId(e.target.value)}>{runs.map(row=><option key={row.run_id} value={row.run_id}>{row.title} · {row.run_id}</option>)}</select></label>
      <label>冻结权重<select value={checkpoint} onChange={e=>setCheckpoint(e.target.value)}>{run?.checkpoints.map(row=><option key={row.id} value={row.id}>{row.label} · {row.sha256.slice(0,12)} · v{row.policy_version}</option>)}</select></label>
      <label>阵容与主哨兵所在方<select value={scenario} onChange={e=>setScenario(e.target.value)}>{run?.scenarios.map(row=><option key={row.id} value={row.id}>{row.learner_side==='red'?'红方':'蓝方'} · {row.id}</option>)}</select></label>
      <label>整局能力配置<select value={profile} onChange={e=>setProfile(e.target.value)}>{profiles.map(row=><option key={row.id} value={row.id}>{row.label} · {row.sha256?.slice(0,12)}</option>)}</select></label>
      <label>物理后端<select value={backend} onChange={e=>setBackend(e.target.value as typeof backend)}><option value="rapier">Rapier（默认验收）</option><option value="warp_kinematic">Warp（训练后端）</option></select></label>
      <label>随机种子<input type="number" value={seed} step={1} min={-(2**31)} max={2**31-1} onChange={e=>setSeed(Number(e.target.value))}/></label><label>观测时长（秒）<input type="number" value={duration} step={1} min={1} max={420} onChange={e=>setDuration(Number(e.target.value))}/></label>
      {selected&&<div className="research-wide research-lineups">{(['red','blue'] as const).map(side=><p key={side}><strong>{side==='red'?'红方':'蓝方'}</strong>{Object.entries(selected.lineups[side]).map(([slot,model])=><span key={slot}>{slot} 号 · {model}</span>)}</p>)}<small>固定辅助策略：{Object.entries(run?.helpers||{}).map(([role,row])=>`${role} ${row.sha256.slice(0,12)}`).join(' · ')}</small></div>}
      <div className="research-wide research-actions"><button className="primary" disabled={busy||!!working||!checkpoint||!scenario||!profile} onClick={start}><Play size={15}/>{working==='start'?'核验权重与配置…':'启动冻结对局'}</button><span>{busy?'当前网页对局结束后可启动。':'主策略与辅助策略均冻结；播放不会更新权重。'}</span></div>
    </div><div className="archive-heading research-replays-heading"><div><h2>已保存全帧回放</h2><p>仅加载选中的记录；截断记录单独标记，不代表完整比赛。</p></div></div><div className="table-shell"><table><thead><tr><th>记录</th><th>后端 / 权重</th><th>时长 / 文件</th><th>完整性</th><th/></tr></thead><tbody>{replays.map(row=><tr key={row.id}><td><strong>{row.label}</strong><small>{row.kind==='training'?'训练采样回放':'冻结验收回放'} · {row.scenario_id||'场景未标注'}</small></td><td>{row.backend||'未标注'}<small>{row.checkpoint_id||'未标注'}</small></td><td>{row.seconds===undefined?'—':`${row.seconds.toFixed(1)} s`}<small>{(row.bytes/1024/1024).toFixed(1)} MiB · SHA {row.sha256.slice(0,12)}</small></td><td>{row.truncated===true?'截断片段':row.truncated===false?'记录时段完整':'未标注'}<small>{row.recording_end_reason}</small></td><td><button disabled={!!working} onClick={()=>open(row)}><Play size={14}/>{working===row.id?'读取与校验…':'观看'}</button></td></tr>)}</tbody></table>{!replays.length&&<div className="empty-archive"><p>此运行尚无已登记的全帧回放。</p></div>}</div></>}
  </section>;
}
