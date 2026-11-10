import {beforeAll,describe,expect,test} from 'bun:test';
import {spawnSync} from 'node:child_process';
import {readFileSync} from 'node:fs';
import {renderToStaticMarkup} from 'react-dom/server';
import type {Frame} from '../../packages/renderer/src/types';
import type {RuntimeCatalog,RoomState} from '../../packages/renderer/src/lanTypes';
import {introAt,introSchedule,introSeconds} from '../../packages/renderer/src/intro';
import {ArenaShell,ConnectionDialog,compatible,DEFAULT_SETTINGS} from '../../apps/client/web/ArenaShell';
import {Ceremony,introductionShot} from '../../apps/client/web/Ceremony';
import {Results} from '../../apps/client/web/Results';
let fixtures:any,catalog:RuntimeCatalog,frame:Frame,room:RoomState;
beforeAll(()=>{const r=spawnSync('.venv/bin/python',['scripts/operator_fixtures.py'],{encoding:'utf8'});if(r.status!==0)throw Error(r.stderr);fixtures=JSON.parse(readFileSync('build/operator-ui/fixtures.json','utf8'));catalog=fixtures._catalog;frame={...fixtures.intro.static,...fixtures.intro.presentation};room=fixtures.intro.room;});
describe('Arena match presentation',()=>{
 test('original Arena home links to a dialog with both LAN connection flows',()=>{const html=renderToStaticMarkup(<ArenaShell room={null} catalog={catalog} playerId="" settings={DEFAULT_SETTINGS} onSettings={()=>{}} invoke={()=>{}} notice="" visible onClose={()=>{}}/>);expect(html).toContain('home-title');expect(html).toContain('准备正式对战');expect(html).toContain('/brand/arena-1.png');const dialog=renderToStaticMarkup(<ConnectionDialog settings={DEFAULT_SETTINGS} onSettings={()=>{}} ownerKey="" onOwnerKey={()=>{}} invoke={()=>{}} onClose={()=>{}}/>);expect(dialog).toContain('创建房间');expect(dialog).toContain('加入房间');expect(dialog).toContain('仿真工程或运行包目录');});
 test('hardware strategy compatibility follows exact model instead of role alone',()=>{const adaptive=catalog.strategies.find(s=>s.mode==='adaptive'&&s.robot===301)!;expect(compatible(adaptive,3,301)).toBe(true);expect(compatible(adaptive,3,305)).toBe(false);expect(compatible(catalog.strategies.find(s=>s.flexible)!,3,305)).toBe(true);});
 test('all ten robots have a host-timed shot before faceoff',()=>{const cuts=introSchedule(frame.robots,38.3);expect(introSeconds(10)).toBe(38.3);expect(cuts.filter(c=>c.segment.kind==='robot')).toHaveLength(10);expect(cuts.at(-1)?.segment.kind).toBe('faceoff');expect(cuts.at(-1)?.end).toBeCloseTo(38.3,5);const t=cuts[3].start+.8;expect(introductionShot(frame,room,t)?.progress).toBeCloseTo(introAt(cuts,t)!.progress,5);});
 test('every robot introduction includes actual selected hardware and controller',()=>{for(const c of introSchedule(frame.robots,38.3).filter(c=>c.segment.kind==='robot')){const html=renderToStaticMarkup(<Ceremony frame={frame} room={room} catalog={catalog} elapsed={c.start+.1} isHost skip={()=>{}}/>);expect(html).toContain('li-shot');expect(html).toContain('策略');expect(html).toContain('型 ·');}});
 test('infantry introduction labels and stats follow selected referee performance',()=>{
  const source=frame.robots.find(r=>r.team==='red'&&r.slot===3)!;
  const cut=introSchedule(frame.robots,38.3).find(c=>c.segment.kind==='robot'&&c.segment.id===source.id)!;
  for(const chassis of ['hp','power'])for(const weapon of ['cooling','burst']){
   const robot={...source,chassis_profile:chassis,weapon_profile:weapon,max_hp:chassis==='hp'?200:150,power_limit:chassis==='hp'?45:60,max_heat:weapon==='cooling'?50:100,cooling_rate:weapon==='cooling'?40:20};
   const selected={...frame,robots:frame.robots.map(r=>r.id===robot.id?robot:r)};
   const html=renderToStaticMarkup(<Ceremony frame={selected} room={room} catalog={catalog} elapsed={cut.start+.1} isHost skip={()=>{}}/>);
   expect(html).toContain(chassis==='hp'?'血量优先':'功率优先');expect(html).toContain(weapon==='cooling'?'冷却优先':'爆发优先');
   expect(html).toContain(`aria-label="生命值 ${robot.max_hp}"`);expect(html).toContain(`aria-label="冷却速率 ${robot.cooling_rate} /s"`);
  }
 });
 test('only host sees the intro skip control and countdown has no skip',()=>{const render=(r:RoomState,host:boolean)=>renderToStaticMarkup(<Ceremony frame={frame} room={r} catalog={catalog} elapsed={0} isHost={host} skip={()=>{}}/>);expect(render(room,true)).toContain('跳过介绍');expect(render(room,false)).not.toContain('跳过介绍');expect(render({...room,status:'countdown',stage_duration:5},true)).not.toContain('跳过');});
 test('results use the original victory component and retain LAN next-match actions',()=>{const html=renderToStaticMarkup(<Results room={fixtures.finished.room} catalog={catalog} arena={fixtures.finished.static.arena} isHost ready={false} command={()=>{}}/>);for(const label of ['vic-emblem','红方胜利','对战数据','MVP','保存回放','准备下一局','开始下一局'])expect(html).toContain(label);});
});
