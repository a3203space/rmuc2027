import unittest
import numpy as np
import warp as wp
from rmuc2027_core.sdk import _REFERENCE_ARENA, terrain_height
from rmuc2027_research.rl.warp_kinematic.terrain import Terrain, limit_terrain
from rmuc2027_research.rl.warp_kinematic.motion import MotionBatch

class TerrainTests(unittest.TestCase):
    def test_public_map_height_agrees_with_reference(self):
        scene=Terrain(_REFERENCE_ARENA)
        points=np.random.default_rng(42).uniform([0,0],[28,15],size=(2000,2)).astype(np.float32)
        actual=scene.heights(points)
        expected=[terrain_height(_REFERENCE_ARENA,float(x),float(y)) for x,y in points]
        np.testing.assert_allclose(actual,expected,atol=3e-5)

    def test_step_cannot_be_crossed_without_capability(self):
        scene=Terrain(dict(width=10,height=10,terrain=[dict(x=2,y=0,w=8,h=10,kind='platform',elevation=.3)]))
        sim=MotionBatch([[1.8,5]],[[10,100,.1]],[])
        sim.step([[1,0]])
        heights=wp.zeros(1,dtype=float,device=sim.device)
        limits=wp.array([.04],dtype=float,device=sim.device)
        wp.launch(limit_terrain,1,inputs=[scene.data,sim.previous,sim.pos,sim.vel,sim.limits,limits,heights,sim.flight],device=sim.device)
        p,v=sim.state()
        self.assertLess(p[0,0],2.)
        self.assertEqual(v[0,0],0.)

    def test_motion_batch_updates_height_and_roof_clearance_on_platform(self):
        scene=Terrain(dict(width=10,height=10,terrain=[dict(x=0,y=0,w=10,h=10,kind='platform',elevation=.3)]))
        # The roof is .25 m above this platform, not .55 m above the robot's feet.
        for robot_height, blocked in ((.2,False),(.4,True)):
            sim=MotionBatch([[1,5]],[[2,100,.1]],[[2,4,2.1,6]])
            sim.terrain=scene
            sim.wall_z.assign(np.array([[.55,.7]],dtype=np.float32))
            sim.height_offsets.assign(np.array([[0.,robot_height]],dtype=np.float32))
            for _ in range(20):sim.step([[1,0]])
            p,_=sim.state()
            self.assertEqual(p[0,0]<2.,blocked)
            self.assertAlmostEqual(sim.heights.numpy()[0],.3,places=5)
