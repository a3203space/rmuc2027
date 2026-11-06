"""Transport integration with a deterministic stationary fixture, not a training backend."""
import unittest
import numpy as np
from rmuc2027_core.engine import Match
from rmuc2027_core.sdk import terrain_height,aim_at,aim_pitch
from rmuc2027_research.rl.warp_kinematic.protocol import MotionFrame

class StationaryFixture:
    protocol = 'kinematic_frame'
    def __init__(self):
        self.sequences=[]
    def step(self, request):
        self.sequences.append(request['sequence'])
        robots=request['robots'];n=len(robots)
        return MotionFrame(tuple(r['id'] for r in robots),
            np.tile([r['pos'] for r in robots],(6,1,1)),np.zeros((6,n,3)),
            np.tile([r['heading'] for r in robots],(6,1)),
            np.tile([r['folded'] for r in robots],(6,1)),
            np.ones((6,n),dtype=bool)).response(request['sequence'])
    def close(self): pass

class ShootingFixture(StationaryFixture):
    def step(self,request):
        if request['sequence']==0:self.arena=request['arena']
        for robot in request['robots']:
            if robot['id'] in ('red-3','blue-4'):
                x=5. if robot['id']=='red-3' else 7.
                robot['pos']=[x,3.,terrain_height(self.arena,x,3.)]
        return super().step(request)

class KinematicEngineTests(unittest.TestCase):
    def test_query_mirror_preserves_visibility_and_armor_hits(self):
        with Match(physics_adapter=ShootingFixture(),battle_settings={'drones_enabled':False}) as match:
            for _ in range(120):frame=match.step({})
            match.step({'red':{'3':{'buy_ammo':{'remote':True,'amount':100}}}})
            for _ in range(130):frame=match.step({})
            self.assertTrue(any(r['id']=='blue-4' for r in match.observation('red',[3])['enemies']))
            target=next(r for r in frame['robots'] if r['id']=='blue-4');initial_hp=target['hp']
            events=[]
            for _ in range(60):
                shooter=next(r for r in frame['robots'] if r['id']=='red-3')
                target=next(r for r in frame['robots'] if r['id']=='blue-4')
                frame=match.step({'red':{'3':{'aim':aim_at(shooter,target),'pitch':aim_pitch(shooter,target),'fire':True}}})
                events.extend(frame['events'])
            target=next(r for r in frame['robots'] if r['id']=='blue-4')
            self.assertLess(target['hp'],initial_hp)
            self.assertTrue(any(e['type']=='hit' for e in events))

    def test_loaded_weapon_matches_real_physics_firing_clock(self):
        results=[]
        for adapter in (None,StationaryFixture()):
            events=[]
            with Match(physics_adapter=adapter,battle_settings={'drones_enabled':False}) as match:
                for _ in range(120):match.step({})
                match.step({'red':{'3':{'buy_ammo':{'remote':True,'amount':100}}}})
                for _ in range(130):frame=match.step({})
                loaded=next(r for r in frame['robots'] if r['id']=='red-3')
                self.assertEqual(loaded['ammo'],100,'The fixture must really load ammunition')
                for tick in range(80):
                    frame=match.step({'red':{'3':{'aim':-1.5707963267948966,'pitch':0.,'fire':tick<20}}})
                    events.extend(frame['events'])
                robot=next(r for r in frame['robots'] if r['id']=='red-3')
                shots=[e for e in events if e['type']=='shot']
                self.assertGreater(len(shots),0)
                self.assertTrue(any(e['type']=='impact' for e in events))
                results.append((robot,shots))
        self.assertEqual(results[0][0]['ammo'],results[1][0]['ammo'])
        self.assertEqual(len(results[0][1]),len(results[1][1]))
        for key in ('heat','cooldown'):
            self.assertAlmostEqual(results[0][0][key],results[1][0][key],places=7)

    def test_one_exchange_per_frame_and_weapon_clock_advances(self):
        adapter=StationaryFixture()
        with Match(physics_adapter=adapter,battle_settings={'drones_enabled':False,'timed_assembly':True}) as match:
            for _ in range(20):
                frame=match.step({'red':{'3':{'aim':1.,'fire':True}}})
            self.assertEqual(adapter.sequences,list(range(20)))
            self.assertAlmostEqual(frame['time'],1.)
            robot=next(r for r in frame['robots'] if r['id']=='red-3')
            self.assertGreater(robot['aim'],.1)
            self.assertFalse(robot['tipped'])
            self.assertTrue(all(np.isfinite(robot[k]) for k in ('x','y','z','cooldown')))
