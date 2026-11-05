"""Explicit CUDA tests; run only on a CUDA-capable development host."""
import unittest
import numpy as np
from rmuc2027_research.rl.warp_kinematic.motion import MotionBatch


class CudaMotionTests(unittest.TestCase):
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
