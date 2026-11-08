"""Explicit CUDA tests; run only on a CUDA-capable development host."""
import unittest
import numpy as np
from rmuc2027_research.rl.warp_kinematic.motion import MotionBatch
from rmuc2027_research.rl.warp_kinematic.terrain import Terrain


class CudaMotionTests(unittest.TestCase):
    def test_three_way_coincident_overlap_recovers_symmetrically(self):
        sim=MotionBatch([[0,0]]*3,[[3,10,.2]]*3,[])
        for _ in range(40):sim.step([[0,0]]*3,dt=1/120)
        p,_=sim.state()
        self.assertAlmostEqual(p[0,0],-p[2,0],places=5)
        for a,b in ((0,1),(0,2),(1,2)):
            self.assertGreaterEqual(np.linalg.norm(p[a]-p[b]),.3999)

    def test_recovery_cannot_cross_cliff_or_map_boundary(self):
        terrain=Terrain(dict(width=4.,height=4.,terrain=[dict(kind='platform',x=2.,y=0.,w=2.,h=4.,
            elevation=.5,polygon=[[2.,0.],[4.,0.],[4.,4.],[2.,4.]])]))
        for x in (1.99,.11):
            sim=MotionBatch([[x,2.]]*2,[[3,10,.1]]*2,[]);sim.terrain=terrain
            for _ in range(30):sim.step([[0,0]]*2,dt=1/120)
            p,_=sim.state()
            self.assertGreaterEqual(p[:,0].min(),.1-1e-6)
            self.assertLess(p[:,0].max(),2.)
            self.assertGreaterEqual(np.linalg.norm(p[0]-p[1]),.1999)

    def test_coincident_spawn_recovers_without_crossing_wall(self):
        sim=MotionBatch([[.21,0.],[.21,0.]],[[3,10,.2]]*2,[[-1,-2,0,2]])
        for _ in range(30):sim.step([[0,0],[0,0]],dt=1/120)
        p,_=sim.state()
        self.assertGreaterEqual(p[:,0].min(),.1999)
        self.assertGreaterEqual(np.linalg.norm(p[0]-p[1]),.3999)

    def test_overlap_recovery_respects_disabled_robot_and_world(self):
        sim=MotionBatch([[0,0],[0,0],[0,0]],[[3,10,.2]]*3,[],worlds=[0,0,1])
        sim.enabled.assign(np.array([1,0,1],dtype=np.int32))
        for _ in range(30):sim.step([[0,0]]*3,dt=1/120)
        p,_=sim.state()
        np.testing.assert_array_equal(p[1:],[[0,0],[0,0]])
        self.assertGreaterEqual(np.linalg.norm(p[0]),.3999)

    def test_float32_contact_keeps_tangential_motion(self):
        radius=.38890877
        # Just inside the expanded boundary after a float32 contact readback.
        sim=MotionBatch([[2.79939866,6.65-radius+2.e-7]],[[3,10,radius]],
                        [[1.461,6.65,3.361,8.35]])
        for _ in range(20):sim.step([[1,1]])
        p,v=sim.state()
        self.assertGreater(p[0,0],3.4)
        self.assertGreater(v[0,0],.1)

    def test_swept_collision_cannot_tunnel_through_thin_wall(self):
        sim = MotionBatch([[0,0]], [[30,1000,.2]], [[1,-2,1.01,2]])
        for _ in range(20): sim.step([[1,0]])
        p,v = sim.state()
        self.assertGreater(p[0,0], .79)
        self.assertLessEqual(p[0,0], .8)
        self.assertAlmostEqual(v[0,0], 0., places=5)

    def test_slides_and_agents_keep_independent_state(self):
        sim = MotionBatch([[0,0],[-10,0]], [[3,10,.2]]*2, [[1,-10,2,10]])
        for _ in range(20): sim.step([[1,1],[0,0]])
        p,_ = sim.state()
        self.assertLessEqual(p[0,0], .801)
        self.assertGreater(p[0,1], 1.)
        np.testing.assert_array_equal(p[1],[-10,0])

    def test_acceleration_and_cpu_rejection(self):
        sim = MotionBatch([[0,0]], [[3,2,.2]], [])
        sim.step([[1,0]])
        _,v = sim.state()
        self.assertAlmostEqual(v[0,0], .1, places=5)
        with self.assertRaises(ValueError): MotionBatch([[0,0]],[[3,2,.2]],[],device='cpu')
        with self.assertRaises(ValueError): sim.step([[float('nan'),0]])

    def test_head_on_contact_and_world_isolation(self):
        for worlds in ([0,0], [0,1]):
            sim = MotionBatch([[-1,0],[1,0]],[[30,1000,.2]]*2,[],worlds=worlds)
            sim.step([[1,0],[-1,0]])
            p,_ = sim.state()
            if worlds[0] == worlds[1]:
                self.assertLessEqual(p[0,0],-.1999)
                self.assertGreaterEqual(p[1,0],.1999)
                self.assertAlmostEqual(p[0,0],-p[1,0],places=5)
            else:
                self.assertGreater(p[0,0],p[1,0])

    def test_frame_samples_match_individual_steps(self):
        params = ([[0,0]],[[3,2,.2]],[])
        frame = MotionBatch(*params)
        reference = MotionBatch(*params)
        positions, velocities = frame.frame([[1,0]])
        for index in range(6):
            reference.step([[1,0]],dt=1/120)
            p,v = reference.state()
            np.testing.assert_allclose(positions[index],p,atol=1e-7)
            np.testing.assert_allclose(velocities[index],v,atol=1e-7)

    def test_roof_clearance_and_rotated_wall(self):
        for top, blocked in ((.2,False),(.5,True)):
            sim=MotionBatch([[0,0]],[[3,100,.1]],[[1,-2,1.1,2]])
            sim.wall_z.assign(np.array([[.3,.8]],dtype=np.float32))
            sim.body_z.assign(np.array([[0.,top]],dtype=np.float32))
            for _ in range(20):sim.step([[1,0]])
            p,_=sim.state()
            self.assertEqual(p[0,0]<1.,blocked)
        sim=MotionBatch([[0,0]],[[3,100,.1]],[[1,-2,1.1,2]])
        sim.angles.assign(np.array([np.pi/4],dtype=np.float32))
        for _ in range(6):sim.step([[1,0]])
        p,_=sim.state()
        self.assertLess(p[0,0],1.)
        self.assertGreater(abs(p[0,1]),.001)

    def test_power_budget_bounds_kinetic_energy(self):
        sim=MotionBatch([[0,0]],[[30,100,.1]],[])
        sim.drive.assign(np.array([[20.,.75,23.]],dtype=np.float32))
        for _ in range(20):sim.step([[1,0]])
        _,v=sim.state()
        self.assertLessEqual(.5*20*float(np.sum(v*v)),20*.75+1e-4)
        zero=MotionBatch([[0,0]],[[30,100,.1]],[])
        zero.drive.assign(np.array([[20.,.75,0.]],dtype=np.float32))
        zero.step([[1,0]])
        np.testing.assert_array_equal(zero.state()[0],[[0.,0.]])

    def test_per_agent_forbidden_mask(self):
        sim=MotionBatch([[0,0],[0,0]],[[3,100,.1]]*2,[[1,-2,1.1,2]],worlds=[0,1])
        sim.wall_masks.assign(np.array([[1],[0]],dtype=np.int32))
        for _ in range(20):sim.step([[1,0],[1,0]])
        p,_=sim.state()
        self.assertLess(p[0,0],1.)
        self.assertGreater(p[1,0],2.)
