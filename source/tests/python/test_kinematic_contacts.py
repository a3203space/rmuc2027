import unittest
import numpy as np
from rmuc2027_research.rl.warp_kinematic.motion import MotionBatch


class ContactTests(unittest.TestCase):
    def sim(self,positions,worlds=None,boxes=(),masses=None):
        sim=MotionBatch(positions,[[40,10000,.2]]*len(positions),boxes,worlds=worlds)
        sim.dynamic_contacts=True
        if masses is not None:sim.drive.assign(np.asarray([[m,1,1.e9] for m in masses],np.float32))
        return sim

    def test_equal_mass_push_conserves_momentum_and_slides(self):
        sim=self.sim([[0,0],[.41,0]])
        sim.step([[.1,.03],[0,0]],dt=.05)
        pos,vel=sim.state()
        self.assertGreater(pos[1,0],.05)
        self.assertGreater(pos[0,1],0.)
        np.testing.assert_allclose(vel.sum(axis=0),[4,1.2],atol=1.e-5)
        self.assertGreaterEqual(np.linalg.norm(pos[0]-pos[1]),.4-1.e-5)
        self.assertFalse(sim.contact_exhausted.numpy().any())

    def test_unequal_mass_head_on_no_tunnelling(self):
        sim=self.sim([[-1,0],[1,0]],masses=[1,3])
        sim.step([[1,0],[-1,0]],dt=.05)
        pos,vel=sim.state()
        self.assertLess(pos[0,0],pos[1,0])
        self.assertGreaterEqual(pos[1,0]-pos[0,0],.4-1.e-5)
        np.testing.assert_allclose(vel[:,0],[-20,-20],atol=1.e-5)
        self.assertFalse(sim.contact_exhausted.numpy().any())

    def test_push_against_wall_stops_without_penetration(self):
        sim=self.sim([[0,0],[.41,0]],boxes=[[.8,-2,.9,2]])
        for _ in range(30):sim.step([[.1,0],[0,0]],dt=1/120)
        pos,vel=sim.state()
        self.assertLessEqual(pos[1,0],.6+1.e-5)
        self.assertGreaterEqual(pos[1,0]-pos[0,0],.4-1.e-5)
        self.assertFalse(sim.contact_exhausted.numpy().any())

    def test_three_body_chain_and_independent_world(self):
        sim=self.sim([[0,0],[.41,0],[.82,0],[0,0]],worlds=[0,0,0,1])
        sim.step([[.1,0],[0,0],[0,0],[.1,0]],dt=.05)
        pos,vel=sim.state()
        self.assertGreater(pos[2,0],.82)
        self.assertGreater(pos[3,0],pos[0,0])
        self.assertTrue(np.all(np.diff(pos[:3,0])>=.4-1.e-5))
        np.testing.assert_allclose(vel[:3,0].sum(),4,atol=1.e-5)
        self.assertFalse(sim.contact_exhausted.numpy().any())

    def test_disabled_agent_and_vertical_noncontact(self):
        sim=self.sim([[0,0],[.41,0],[0,0]],worlds=[0,0,0])
        sim.enabled.assign(np.asarray([1,0,1],np.int32))
        sim.body_z.assign(np.asarray([[0,1],[0,1],[2,3]],np.float32))
        sim.step([[.1,0],[0,0],[.1,0]],dt=.05)
        pos,_=sim.state()
        self.assertAlmostEqual(pos[1,0],.41,places=6)
        self.assertLessEqual(pos[0,0],.01001)
        self.assertGreater(pos[2,0],pos[0,0])

    def test_interleaved_world_members_keep_global_agent_identity(self):
        sim=self.sim([[0,0],[0,0],[.41,0],[.41,0]],worlds=[7,2,7,2])
        sim.enabled.assign(np.asarray([1,1,1,0],np.int32))
        sim.step([[.1,0],[.1,0],[0,0],[0,0]],dt=.05)
        pos,_=sim.state()
        self.assertGreater(pos[2,0],.45)
        self.assertAlmostEqual(pos[3,0],.41,places=6)
        self.assertGreater(pos[0,0],pos[1,0]+.05)
        self.assertLessEqual(pos[1,0],.0101)
        self.assertFalse(sim.contact_exhausted.numpy().any())

    def test_ten_body_free_chain_does_not_exhaust_or_interpenetrate(self):
        sim=self.sim([[i*.40001,0] for i in range(10)])
        for _ in range(30):sim.step([[.1,0]]+[[0,0]]*9,dt=1/120)
        pos,vel=sim.state()
        self.assertTrue(np.all(np.diff(pos[:,0])>=.4-1.e-4))
        self.assertGreater(pos[-1,0],3.60009+.05)
        self.assertFalse(sim.contact_exhausted.numpy().any())
        self.assertAlmostEqual(float(vel[:,0].sum()),4.,places=4)

    def test_wall_overlap_allows_escape_but_not_deeper_motion(self):
        sim=self.sim([[.40002,0]],boxes=[[.6,-1,.7,1]])
        sim.step([[.1,0]])
        self.assertLessEqual(sim.state()[0][0,0],.400021)
        sim.step([[-.1,0]])
        self.assertLess(sim.state()[0][0,0],.3)

    def test_ten_body_chain_against_wall_converges_without_budget_exhaustion(self):
        wall=9*.40001+.21
        sim=self.sim([[i*.40001,0] for i in range(10)],boxes=[[wall,-2,wall+.1,2]])
        for _ in range(30):
            sim.step([[.1,0]]+[[0,0]]*9,dt=1/120)
            pos,_=sim.state()
            self.assertTrue(np.all(np.diff(pos[:,0])>=.4-.0001))
            self.assertLessEqual(pos[-1,0],wall-.2+.000002)
        self.assertFalse(sim.contact_exhausted.numpy().any())

    def test_real_roster_contact_pushes_idle_teammate(self):
        from rmuc2027_core.engine import Match
        from rmuc2027_research.rl.warp_kinematic.arena import ArenaMotion
        for backend in ('rapier','warp_kinematic'):
            adapter=ArenaMotion() if backend=='warp_kinematic' else None
            with Match(physics_adapter=adapter,battle_settings={'drones_enabled':False,'timed_assembly':True}) as match:
                for _ in range(80):
                    frame=match.step({'red':{'7':{'move':[.7,0.],'heading':0.}}})
                    if adapter:
                        bodies={r['id']:r for r in frame['robots']}
                        a,b=bodies['red-3'],bodies['red-7']
                        self.assertGreaterEqual(np.hypot(a['x']-b['x'],a['y']-b['y']),a['radius']+b['radius']-.0002)
                bodies={r['id']:r for r in frame['robots']}
                self.assertGreater(bodies['red-3']['x'],4.)
                self.assertGreater(bodies['red-7']['x'],4.5)
                if adapter:self.assertFalse(adapter.motion.contact_exhausted.numpy().any())


if __name__=='__main__':unittest.main()
