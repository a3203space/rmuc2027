import {describe,expect,test} from 'bun:test';
import {renderToStaticMarkup} from 'react-dom/server';
import {ResearchBanner,ResearchPanel,researchStart,researchReplayUrl} from '../../apps/arena-web/src/ResearchPanel';
import type {Metadata,ResearchRun} from '../../apps/arena-web/src/types';
const sha='a'.repeat(64),id='baseline-'+sha.slice(0,12);
const run:ResearchRun={run_id:'run-one',title:'冻结研究',manifest_sha256:sha,checkpoints:[{id,label:'baseline',sha256:sha,policy_version:6}],scenarios:[{id:'case_blue',learner_side:'blue',lineups:{red:{'7':2772},blue:{'7':2773}}}],capability_profiles:[{id:'blue-fixed',label:'蓝方',sha256:sha,learner_side:'blue'},{id:'red-fixed',label:'红方',sha256:sha,learner_side:'red'}],helpers:{infantry:{sha256:sha},heavy:{sha256:sha}},replay_count:0,backends:['rapier','warp_kinematic']};
describe('frozen research UI',()=>{
 test('launch contains registered IDs, separate side/capability choices, and no paths or ordinary strategies',()=>{
  const request=researchStart(run,id,'case_blue','blue-fixed','rapier',81,60);
  expect(request).toEqual({mode:'research',research:{run_id:'run-one',checkpoint_id:id,scenario_id:'case_blue',capability_profile_id:'blue-fixed',backend:'rapier'},seed:81,duration:60,save_mp4:false});
  expect(request).not.toHaveProperty('red');expect(JSON.stringify(request)).not.toContain('path');
 });
 test('mutable aliases, missing scenario, wrong-side abilities and invalid duration fail before launch',()=>{
  expect(()=>researchStart(run,'selected','case_blue','blue-fixed','rapier',81,60)).toThrow();
  expect(()=>researchStart(run,id,'other','blue-fixed','rapier',81,60)).toThrow();
  expect(()=>researchStart(run,id,'case_blue','red-fixed','rapier',81,60)).toThrow();
  expect(()=>researchStart(run,id,'case_blue','blue-fixed','rapier',81,421)).toThrow();
 });
 test('catalog absence is explicit and does not invent a checkpoint',()=>{
  const html=renderToStaticMarkup(<ResearchPanel busy={false} onStart={async()=>{}} onReplay={async()=>{}}/>);
  expect(html).toContain('尚无已登记研究运行');expect(html).toContain('不计 Elo');expect(html).not.toContain('research-actions');
 });
 test('viewer identifies immutable checkpoint/backend and truncated full-frame data',()=>{
  const metadata={research:{run_id:'run-one',checkpoint_id:id,checkpoint_sha256:sha,backend:'rapier'},recording:{kind:'evaluation',truncated:true,frame_hz:20}} as Metadata;
  const html=renderToStaticMarkup(<ResearchBanner metadata={metadata} onResearch={()=>{}}/>);
  for(const text of ['冻结研究对局','不计 Elo',id,'rapier','截断片段','20 Hz 全帧'])expect(html).toContain(text);
 });
 test('training replay keeps policy updates visible without claiming a frozen checkpoint',()=>{
  const metadata={kind:'training',truncated:true,physics_backend:'warp_kinematic',training_recording:{policy_changes_during_episode:true}} as Metadata;
  const html=renderToStaticMarkup(<ResearchBanner metadata={metadata} onResearch={()=>{}}/>);
  for(const text of ['已保存训练回放','warp_kinematic','截断片段','含策略更新'])expect(html).toContain(text);
  expect(html).not.toContain('冻结研究对局');
 });
 test('registered replay links encode each identifier',()=>{
  expect(researchReplayUrl('run-one','episode-000')).toBe('/api/research/runs/run-one/replays/episode-000');
  expect(researchReplayUrl('bad/path','a b')).toBe('/api/research/runs/bad%2Fpath/replays/a%20b');
 });
});
