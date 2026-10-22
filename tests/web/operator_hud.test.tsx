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
 test('reconnect, stale data, settlement and role commands are explicit',()=>{expect(render('resume')).toContain('恢复人工控制');expect(render('infantry',{stale:true})).toContain('数据已过期');expect(render('finished')).toContain('保存回放');expect(render('finished')).toContain('准备下一局');expect(render('infantry',{menu:true})).toContain('150 金币');});
});
