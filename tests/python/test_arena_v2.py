"""Behavior checks for independently configured hardware, policies and synchronized starts."""
import copy
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from rmuc2027_server.room import Room
from rmuc2027_server.protocol import Join, Seat, SeatConfig, TeamConfig, StageReady, Input
from rmuc2027_core.engine import Match
from rmuc2027_server.control import HumanController, baseline_helpers


class ArenaV2Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, RMUC_USER_DATA=self.temp.name)
        self.env.start()
        self.room = Room(owner_key='test', directory=self.temp.name, duration=1, ai=False)
        self.addCleanup(self.room.close); self.addCleanup(self.env.stop); self.addCleanup(self.temp.cleanup)
        self.host = self.player('房主', 'test', 'red', 3)

    def player(self, name, key=None, team='blue', slot=4):
        token = self.room.join(Join(nickname=name, host_key=key))['token']
        self.room.attach(token); self.room.select(token, Seat(team=team, slot=slot))
        return token

    def wait(self, predicate, seconds=8):
        deadline = time.monotonic()+seconds
        while not predicate() and time.monotonic()<deadline: time.sleep(.02)
        self.assertTrue(predicate(), (self.room.status,self.room.error))

    def test_model_permissions_and_readiness(self):
        other = self.player('蓝方')
        for p in self.room.players.values(): p.ready=True
        self.room.configure_seat(self.host,SeatConfig(team='red',slot=3,model=301))
        self.assertEqual(self.room.lineup('red')['3'],301)
        self.assertEqual(self.room.lineup('blue')['3'],2731)
        self.assertTrue(all(not p.ready for p in self.room.players.values()))
        with self.assertRaisesRegex(ValueError,'自己'):self.room.configure_seat(self.host,SeatConfig(team='blue',slot=4,model=305))
        with self.assertRaises(ValueError):self.room.configure_seat(other,SeatConfig(team='blue',slot=4,model=2711))
        self.room.configure_seat(self.host,SeatConfig(team='blue',slot=7,model=2772))

    def test_strategy_constraints_and_local_immutable_import(self):
        adaptive=next(s for s in self.room.catalog.entries.values() if s['mode']=='adaptive' and s['robot']==301)
        self.room.configure_seat(self.host,SeatConfig(team='red',slot=3,model=301,strategy=adaptive['id']))
        with self.assertRaisesRegex(ValueError,'兼容'):self.room.configure_seat(self.host,SeatConfig(team='red',slot=3,model=305,strategy=adaptive['id']))
        source=self.room.catalog.sources[adaptive['id']]
        files={p.name:p.read_text() for p in source.iterdir() if p.suffix in ('.json','.py')}
        catalog=self.room.import_strategy(self.host,files)
        self.assertTrue(any(s['id'].startswith('local:') for s in catalog['strategies']))
        ordinary=self.player('普通玩家')
        with self.assertRaisesRegex(ValueError,'房主'):self.room.import_strategy(ordinary,files)
        bad=copy.deepcopy(files);bad['strategy.py']+='\n# changed\n'
        with self.assertRaises(ValueError):self.room.import_strategy(self.host,bad)
        with self.assertRaises(ValueError):self.room.import_strategy(self.host,{'../x':'x'})

    def test_team_preset_requires_registered_hardware(self):
        entry=next(s for s in self.room.catalog.entries.values() if s['mode']=='team' and s['lineup'] and s['lineup']!=self.room.lineup('red'))
        with self.assertRaisesRegex(ValueError,'兼容'):self.room.configure_team(self.host,TeamConfig(team='red',mode='team',strategy=entry['id']))
        for slot,model in entry['lineup'].items():self.room.configure_seat(self.host,SeatConfig(team='red',slot=int(slot),model=model))
        self.room.configure_team(self.host,TeamConfig(team='red',mode='team',strategy=entry['id']))
        self.assertEqual(self.room.team_configs['red']['mode'],'team')

    def test_loading_barrier_freezes_tick_and_disconnect_aborts(self):
        other=self.player('蓝方')
        for p in self.room.players.values():p.ready=True
        self.room.start(self.host)
        self.wait(lambda:self.room.status=='loading')
        self.assertEqual(self.room.snapshot(self.host)['presentation']['tick'],0)
        self.room.stage_ready(self.host,StageReady(match_id=self.room.match_id))
        time.sleep(.12)
        self.assertEqual(self.room.status,'loading')
        self.assertEqual(self.room.snapshot(self.host)['presentation']['time'],0)
        with self.assertRaises(ValueError):self.room.submit(self.host,Input(match_id=self.room.match_id,seq=0,yaw=0,pitch=0,fire=True))
        self.room.detach(other,self.room.player(other).generation)
        self.wait(lambda:self.room.status=='lobby')
        self.assertIn('断开',self.room.error)
        self.assertIsNone(self.room.result)

    def test_skip_is_host_only_and_countdown_precedes_physics(self):
        other=self.player('蓝方')
        for p in self.room.players.values():p.ready=True
        self.room.start(self.host);self.wait(lambda:self.room.status=='loading')
        for t in (self.host,other):self.room.stage_ready(t,StageReady(match_id=self.room.match_id))
        self.wait(lambda:self.room.status=='intro')
        with self.assertRaisesRegex(ValueError,'房主'):self.room.skip_intro(other)
        self.room.skip_intro(self.host);self.wait(lambda:self.room.status=='countdown')
        self.assertEqual(self.room.snapshot(self.host)['presentation']['tick'],0)
        self.wait(lambda:self.room.status=='finished',seconds=8)
        self.assertIsNone(self.room.error)
        self.assertEqual(len(self.room.result['robots']),10)

    def test_first_frame_uses_actual_asymmetric_models(self):
        self.room.ceremony=False
        self.room.configure_seat(self.host,SeatConfig(team='red',slot=3,model=301))
        self.room.ready(self.host,True);self.room.start(self.host)
        self.wait(lambda:self.room.status=='finished')
        robots=self.room.result['robots']
        self.assertEqual(next(r for r in robots if r['id']=='red-3')['model_slot'],301)
        self.assertEqual(next(r for r in robots if r['id']=='blue-3')['model_slot'],2731)

    def test_selected_ai_strategy_executes_and_records_version(self):
        self.room.ceremony=False;self.room.ai=True
        entry=next(e for e in self.room.catalog.entries.values() if e['mode']=='adaptive' and e['robot']==305)
        self.room.configure_seat(self.host,SeatConfig(team='blue',slot=4,model=305,strategy=entry['id']))
        self.room.ready(self.host,True);self.room.start(self.host)
        self.wait(lambda:self.room.status=='finished',seconds=25)
        self.assertIsNone(self.room.error)
        self.assertEqual(self.room.result['policy_errors'],0)
        import json
        replay=json.loads((Path(self.temp.name)/self.room.result['replay']).read_text())
        self.assertEqual(replay['metadata']['seat_configs']['blue']['4']['strategy'],entry['id'])
        self.assertEqual(next(r for r in replay['frames'][0]['robots'] if r['id']=='blue-4')['model_slot'],305)

    def test_fold_input_reaches_engine(self):
        m=Match(duration=1,lineups={'red':[{'slot':1,'model':2711},{'slot':3,'model':301},{'slot':4,'model':2732},{'slot':6,'model':2761},{'slot':7,'model':2771}], 'blue':[{'slot':1,'model':2711},{'slot':3,'model':2731},{'slot':4,'model':2732},{'slot':6,'model':2761},{'slot':7,'model':2771}]})
        try:
            controller=HumanController(baseline_helpers())
            for tick in range(8):
                robot=next(r for r in m._frame['robots'] if r['id']=='red-3')
                action=controller.action(m.compact_observation('red',[3]),robot,Input(match_id='test',seq=tick,yaw=0,pitch=0,transform=True))
                frame=m.step({'red':{'3':action}})
            robot=next(r for r in frame['robots'] if r['id']=='red-3')
            self.assertGreater(robot['transform_progress'],0)
            neutral=controller.action(m.compact_observation('red',[3]),robot,Input(match_id='test',seq=9,yaw=0,pitch=0,active=False))
            self.assertTrue(neutral['transform'])
        finally:m.close()

    def test_team_strategy_executes_with_selected_lineup(self):
        self.room.ceremony=False;self.room.ai=True
        entry=self.room.catalog.entries['builtin:gpt61_2027_trident']
        for slot,model in entry['lineup'].items():self.room.configure_seat(self.host,SeatConfig(team='red',slot=int(slot),model=model))
        self.room.configure_team(self.host,TeamConfig(team='red',mode='team',strategy=entry['id']))
        self.room.ready(self.host,True);self.room.start(self.host)
        self.wait(lambda:self.room.status=='finished',seconds=25)
        self.assertIsNone(self.room.error)
        self.assertEqual(self.room.result['policy_errors'],0)
        import json
        replay=json.loads((Path(self.temp.name)/self.room.result['replay']).read_text())
        self.assertEqual(replay['metadata']['strategies'][entry['id']]['version'],entry['version'])
        self.assertEqual(self.room.team_configs['red']['mode'],'team')

if __name__=='__main__':unittest.main()
