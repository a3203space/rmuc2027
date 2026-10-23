import {beforeAll,describe,expect,test} from 'bun:test';
import {readFileSync} from 'node:fs';
import {spawnSync} from 'node:child_process';
import {renderToStaticMarkup} from 'react-dom/server';
import {OperatorHud} from '../../apps/client/web/OperatorHud';
import {positionDisplay} from '../../apps/client/web/hudModel';
import type {LanSnapshot} from '../../packages/renderer/src/lanTypes';
let fixtures:Record<string,LanSnapshot>;
beforeAll(()=>{const r=spawnSync('.venv/bin/python',['scripts/operator_fixtures.py'],{encoding:'utf8'});if(r.status!==0)throw Error(r.stderr);fixtures=JSON.parse(readFileSync('build/operator-ui/fixtures.json','utf8'));});
function render(name:string,extras={}){const s=fixtures[name];return renderToStaticMarkup(<OperatorHud player={s.player!} arena={s.static!.arena} room={s.room!} stale={false} expanded={false} menu={false} onMenu={()=>{}} command={()=>{}} notice="" isHost={true} {...extras}/>);}
describe('operator HUD',()=>{
 test('displays both bases with authoritative maxima and rebuild state',()=>{const html=render('damaged');expect(html).toContain('data-testid="red-base"');expect(html).toContain('/ 1500');expect(html).toContain('875');expect(html).toContain('重建 40%');});
 test('all four positions and map polygons are present',()=>{const html=render('contested');for(const id of ['red_base','red_outpost','blue_base','blue_outpost'])expect(html).toContain(`data-testid="${id}"`);expect(html).toContain('双方激活');expect(html).toContain('<polygon');expect(html).toContain('红·堡垒');expect(html).toContain('蓝·中央高地');});
 test('seconds are normalized and contested progress is preserved',()=>{const p=fixtures.contested.player!.tactical!.positions.red_outpost;const d=positionDisplay(p,{capture_s:5,release_s:5});expect(d.fraction).toBe(.5);expect(d.text).toContain('暂停');});
 test('temporary release and rollback labels differ',()=>{const p={...fixtures.contested.player!.tactical!.positions.red_outpost,control:'temporary' as const,active:[]};expect(positionDisplay(p,{capture_s:5,release_s:5}).text).toBe('释放计时');p.active=[p.controller];expect(positionDisplay(p,{capture_s:5,release_s:5}).text).toContain('回退');});
 test('all five seats expose their role UI',()=>{expect(render('heavy',{menu:true})).toContain('装配辅助');expect(render('infantry')).toContain('性能');expect(render('infantry4')).toContain('步兵');expect(render('sentry',{menu:true})).toContain('进攻姿态');expect(render('hangar')).toContain('无人机');expect(render('hangar',{menu:true})).not.toContain('远程回血');});
 test('penalty contains no tactical state, health or map',()=>{const html=render('penalty');expect(html).toContain('判罚遮挡中');expect(html).not.toContain('scoreboard');expect(html).not.toContain('战术地图');expect(html).not.toContain('data-testid');});
 test('dead and heat-cut cameras retain team HUD',()=>{expect(render('dead')).toContain('复活进度');expect(render('overheat')).toContain('data-testid="red-base"');expect(render('overheat')).not.toContain('class="reticle');});
 test('operator layout exposes thin sight, compact vitals and horizontal robot identities',()=>{
  const html=render('infantry',{scale:2/3});
  for(const name of ['sight-frame','sight-center','own-identity','drive-meters','heat-strip','card-identity'])expect(html).toMatch(new RegExp(`class="${name}(?: |")`));
  expect(html.match(/class="robot-card/g)).toHaveLength(10);
  expect(html).toContain('scale(0.6666666666666666)');
  expect(html).not.toContain('outer-ring');expect(html).not.toContain('<video');
  expect(render('infantry',{expanded:true})).toContain('own-details');
  expect(render('infantry',{menu:true})).toContain('own-details');
 });
 test('aim indication follows the held button and is absent for menus, stale state and hangar',()=>{
  const s=fixtures.infantry,p={...s.player!,control:{...s.player!.control!,aim_assist:{state:'tracking' as const,target_id:'blue-3'}}};
  expect(render('infantry',{player:p,aimHeld:true})).toContain('reticle  tracking');
  expect(render('infantry',{player:p,aimHeld:false})).not.toContain('reticle  tracking');
  expect(render('infantry',{player:p,aimHeld:true,stale:true})).not.toContain('reticle  tracking');
  expect(render('infantry',{menu:true})).not.toContain('sight-frame');
  expect(render('hangar')).not.toContain('sight-frame');
 });
 test('reconnect, stale data, settlement and role commands are explicit',()=>{expect(render('resume')).toContain('恢复人工控制');expect(render('infantry',{stale:true})).toContain('数据已过期');expect(render('finished')).toContain('保存回放');expect(render('finished')).toContain('准备下一局');expect(render('infantry',{menu:true})).toContain('150 金币');});
});

describe('weapon controls',()=>{
 test('zero ammo offers a visible purchase entry and hold-to-aim hint',()=>{const html=render('infantry');expect(html).toContain('B · 购弹');expect(html).toContain('无弹丸');expect(html).toContain('右键按住：自瞄');expect(html).toContain('待机');});
 test('purchase panel contains only ammunition offers and host balance',()=>{const html=render('heavy',{menu:true,ammoOnly:true});expect(html).toContain('购买弹丸');expect(html).toContain('队伍余额');expect(html).toContain('42 mm');expect(html).toContain('150 金币');expect(html).not.toContain('装配辅助');expect(html).not.toContain('远程回血');});
 test('right release hides a previously confirmed target immediately',()=>{const s=fixtures.infantry,p={...s.player!,control:{...s.player!.control!,aim_assist:{state:'tracking' as const,target_id:'blue-3'}}};expect(render('infantry',{player:p,aimHeld:true})).toContain('跟踪目标');const html=render('infantry',{player:p,aimHeld:false});expect(html).toContain('自瞄 · 待机');expect(html).not.toContain('跟踪目标');});
 test('host weapon rejection and delivery timing remain visible',()=>{const s=fixtures.infantry,p={...s.player!,self:{...s.player!.self,pending_ammo:[[70,100]]},weapon:{ready:false,reason:'overheat',detail:'发射机构过热或热量锁定'}};const html=render('infantry',{player:p,menu:true,ammoOnly:true});expect(html).toContain('发射机构过热或热量锁定');expect(html).toContain('配送中：100');expect(html).toContain('剩余 5.0 秒');});
 test('hangar has no gun or aim controls',()=>{const html=render('hangar');expect(html).not.toContain('B · 购弹');expect(html).not.toContain('右键按住');});
});
