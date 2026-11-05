"""Transport integration with a deterministic stationary fixture, not a training backend."""
import unittest
import numpy as np
from rmuc2027_core.engine import Match
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

class KinematicEngineTests(unittest.TestCase):
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
