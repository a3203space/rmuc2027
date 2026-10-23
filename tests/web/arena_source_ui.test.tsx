import {beforeAll,describe,expect,test} from 'bun:test';
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {spawnSync} from 'node:child_process';
import {renderToStaticMarkup} from 'react-dom/server';
import {ArenaShell,DEFAULT_SETTINGS} from '../../apps/client/web/ArenaShell';
import {BroadcastHud,BroadcastDirector} from '../../apps/client/web/arena/broadcast';
import {TelemetryPanel} from '../../apps/client/web/arena/TelemetryPanel';
import {Results} from '../../apps/client/web/arena/ceremony';
import {displayStats} from '../../apps/client/web/arena/matchStats';
import {matchMetadata,operatorView,resultView} from '../../apps/client/web/arena/viewModel';
let fixtures:any;
beforeAll(()=>{const generated=spawnSync('.venv/bin/python',['scripts/operator_fixtures.py'],{encoding:'utf8'});if(generated.status!==0)throw Error(generated.stderr);fixtures=JSON.parse(readFileSync('build/operator-ui/fixtures.json','utf8'));});
describe('Arena-1 source UI port',()=>{
 test('copied styles, icons and audio controls match the recorded upstream bytes',()=>{
  const records=JSON.parse(readFileSync('docs/legal/arena-ui-port.json','utf8'));
  const copied=records.filter((entry:any)=>entry.mode==='copied');
  expect(copied.length).toBeGreaterThanOrEqual(7);
  for(const entry of copied)expect(createHash('sha256').update(readFileSync(entry.path)).digest('hex')).toBe(entry.source_sha256);
 });
 test('radar presentation enriches only known identity, preserving the owner telemetry',()=>{
  const snapshot=fixtures.infantry,p=snapshot.player,view=operatorView(p,snapshot.static.arena,snapshot.room,fixtures._catalog)!;
  expect(view.robots).toHaveLength(10);
  expect(view.robots.find(r=>r.id===p.self.id)!.heat).toBe(p.self.heat);
  for(const robot of view.robots.filter(r=>r.id!==p.self.id)){
   expect(robot.model_asset).toContain('.glb');
   expect(robot).not.toHaveProperty('heat');expect(robot).not.toHaveProperty('drones');expect(robot).not.toHaveProperty('level');
  }
  expect(view).not.toHaveProperty('projectiles');
  expect(operatorView({...p,display:{...p.display,ui_blocked:true}},snapshot.static.arena,snapshot.room,fixtures._catalog)).toBeNull();
 });
 test('the original broadcast uses model art, honeycomb bars and map glyphs',()=>{
  const s=fixtures.infantry,frame=operatorView(s.player,s.static.arena,s.room,fixtures._catalog)!;
  const images=Object.fromEntries(frame.robots.map(robot=>[`${robot.model_asset}|${robot.team}`,'data:image/png;base64,robot']));
  const html=renderToStaticMarkup(<BroadcastHud frame={frame} metadata={matchMetadata(s.room,fixtures._catalog)} stats={displayStats(frame)} director={new BroadcastDirector()} images={images} selected={s.player.self.id} onSelect={()=>{}} round="Round 1/1"/>);
  for(const marker of ['bc-basebar-hex','bc-emblem','bc-card-art','bc-minimap','mm-base','bc-coins','bc-plate'])expect(html).toContain(marker);
  expect(html.match(/class="bc-card /g)).toHaveLength(10);
  expect(html.match(/data:image\/png;base64,robot/g)).toHaveLength(10);
  expect(html.match(/title="Lv\./g)||[]).toHaveLength(frame.robots.filter(robot=>robot.level!==undefined).length);
 });
 test('the source telemetry panel keeps public HP and unknown private values distinct',()=>{
  const s=fixtures.damaged,frame=operatorView(s.player,s.static.arena,s.room,fixtures._catalog)!;
  const html=renderToStaticMarkup(<TelemetryPanel frame={frame} selected={s.player.self.id} onSelect={()=>{}} replay={false} events={[]} eventText={()=>''}/>);
  expect(html).toContain('match-data-panel');expect(html).toContain('data-structure-hp="red-base"');expect(html).toContain('875');expect(html).toContain('1500');expect(html).toContain('热 —/—');
 });
 test('three-column page keeps the viewport mounted and adds no observer camera switch',()=>{
  const html=renderToStaticMarkup(<ArenaShell room={fixtures.infantry.room} player={fixtures.infantry.player} arena={fixtures.infantry.static.arena} catalog={fixtures._catalog} playerId="operator" settings={DEFAULT_SETTINGS} onSettings={()=>{}} invoke={()=>{}} notice="" visible onClose={()=>{}} viewport={<div data-testid="persistent-view"/>}/>);
  for(const marker of ['app-header','arena-layout','setup-panel','stage-column','persistent-view','fullscreen-toggle','match-data-panel'])expect(html).toContain(marker);
  expect(html).toContain('全屏操控');expect(html).not.toContain('自由视角');expect(html).not.toContain('训练');
 });
 test('result layout uses final structures and host verdict without synthesizing unavailable shot counts',()=>{
  const s=fixtures.finished,frame=resultView(s.room,fixtures._catalog,s.static.arena)!;
  expect(frame).not.toBeNull();expect(frame.bases.red.hp).toBe(s.room.result.bases.red.hp);
  const html=renderToStaticMarkup(<Results frame={frame} metadata={matchMetadata(s.room,fixtures._catalog)} stats={displayStats(frame)} images={{}} auto={false} onMvp={()=>{}} onClose={()=>{}}/>);
  for(const marker of ['cer-results','res-base','res-verdict','res-row','控制度','红方胜利'])expect(html).toContain(marker);
  expect(html).not.toContain('NaN');expect(html).not.toContain('大/小弹丸数');
 });
});
