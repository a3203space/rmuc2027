"""CPU kernel/state tests only: no CUDA adapter, native Match or scene probe."""
import unittest
import numpy as np
import warp as wp
from rmuc2027_core.sdk import DESIGNS
from rmuc2027_research.rl.warp_kinematic.dynamics import launch, record_latch
from rmuc2027_research.rl.warp_kinematic.arena import reset_traversal
from rmuc2027_research.rl.warp_kinematic.batching import restore_state
from rmuc2027_research.rl.warp_kinematic.terrain import TerrainData


class JumpState:
    def __init__(self,values):
        self.n=len(values)
        def array(v,t=float):return wp.array(v,dtype=t,device='cpu')
        def zero(t=float):return wp.zeros(self.n,dtype=t,device='cpu')
        self.scene=TerrainData();self.scene.width=20.;self.scene.height=20.;self.scene.floor_slope=0.
        for name,t in (('bounds',wp.vec4),('ranges',wp.vec2i),('vertices',wp.vec2),('profiles',wp.vec4),('ramps',wp.vec4),('kinds',wp.int32),('edge_ramps',float)):
            setattr(self.scene,name,wp.empty(0,dtype=t,device='cpu'))
        d=DESIGNS['2772']
        self.pos=array([[4.,4.]]*self.n,wp.vec2);self.vel=zero(wp.vec2);self.base=zero()
        self.capable=array([1]*self.n,wp.int32);self.enabled=array([1]*self.n,wp.int32)
        self.command=array(values);self.latched=zero();self.strength=zero()
        self.spec=array([[d['jump_duration_s'],d['jump_tuck_s'],d['jump_cooldown_s'],d['jump_thrust_gain']]]*self.n,wp.vec4)
        self.stance=array([[d['leg_min_m'],d['leg_max_m'],d['leg_nominal_m'],d['leg_speed_mps']]]*self.n,wp.vec4)
        self.leg=zero();self.elapsed=zero();self.cooldown=zero();self.status=zero(wp.int32);self.flight=zero(wp.int32)
        self.z=zero();self.vz=zero();self.goal=zero(wp.vec2);self.tilt=zero(wp.vec4)
        self.limits=array([[2.,2.,.25]]*self.n,wp.vec3);self.worlds=array(list(range(self.n)),wp.int32)
        self.starts=zero(wp.vec3);self.ends=zero(wp.vec3);self.durations=zero();self.offsets=zero(wp.vec2)

    def step(self):
        wp.launch(launch,self.n,inputs=[self.scene,self.pos,self.vel,self.base,self.capable,self.enabled,
            self.command,self.latched,self.strength,self.spec,self.stance,self.leg,self.elapsed,self.cooldown,
            self.status,self.flight,self.z,self.vz,self.goal,self.tilt,self.limits,self.worlds,1/120],device='cpu')

    def state(self):
        packed=wp.zeros((6,self.n,37),dtype=float,device='cpu')
        wp.launch(record_latch,self.n,inputs=[self.latched,self.strength,packed],device='cpu')
        state=np.zeros((self.n,19),np.float32)
        for column,value in ((6,self.elapsed),(8,self.cooldown),(9,self.status),(10,self.vz),(11,self.z),
                             (14,self.base),(15,self.flight),(16,self.leg)):
            state[:,column]=value.numpy()
        state[:,17:19]=packed.numpy()[5,:,35:37]
        return state

    def restore(self,state):
        wp.launch(restore_state,self.n,inputs=[wp.array(state,dtype=float,device='cpu'),self.starts,self.ends,
            self.elapsed,self.durations,self.cooldown,self.status,self.vz,self.z,self.offsets,self.base,
            self.flight,self.leg,self.latched,self.strength],device='cpu')


class JumpStrengthTests(unittest.TestCase):
    def test_native_continuous_strength_scales_vertical_push(self):
        state=JumpState([.1,.5,1.]);state.step()
        speed=state.vz.numpy()
        np.testing.assert_allclose(speed/speed[0],[1.,5.,10.],rtol=1e-6)
        np.testing.assert_allclose(state.strength.numpy(),[.1,.5,1.],rtol=1e-6)

    def test_release_during_push_keeps_accepted_strength(self):
        state=JumpState([.2,.8]);state.step();state.command.assign(np.zeros(2,np.float32));state.step()
        self.assertAlmostEqual(float(state.vz.numpy()[1]/state.vz.numpy()[0]),4.,places=5)
        np.testing.assert_allclose(state.strength.numpy(),[.2,.8]);np.testing.assert_array_equal(state.latched.numpy(),[0.,0.])

    def test_held_request_cannot_change_strength_midflight(self):
        state=JumpState([.2]);state.step();state.command.assign(np.ones(1,np.float32));state.step()
        self.assertAlmostEqual(float(state.strength.numpy()[0]),.2,places=6)

    def test_strength_state_roundtrips_with_world_reordering(self):
        original=JumpState([.2,.8]);original.step();saved=original.state()
        reordered=JumpState([0.,0.]);reordered.restore(saved[::-1].copy())
        original.command.assign(np.zeros(2,np.float32));original.step();reordered.step()
        np.testing.assert_array_equal(reordered.state()[::-1],original.state())

    def test_teleport_reset_clears_only_selected_world_strength(self):
        state=JumpState([.2,.8]);state.step()
        mask=wp.array([[1,1],[0,0]],dtype=wp.vec2i,device='cpu')
        wp.launch(reset_traversal,state.n,inputs=[mask,state.status,state.elapsed,state.cooldown,state.vz,
            state.scene,state.pos,state.z,state.base,state.offsets,state.leg,state.stance,state.flight,
            state.latched,state.strength],device='cpu')
        self.assertEqual(float(state.strength.numpy()[0]),0.);self.assertAlmostEqual(float(state.strength.numpy()[1]),.8,places=6)
        state.command.assign(np.array([.6,0.],np.float32));state.step()
        self.assertAlmostEqual(float(state.strength.numpy()[0]),.6,places=6)

    def test_disabled_request_does_not_poison_next_powered_rising_edge(self):
        state=JumpState([.2]);state.enabled.assign(np.array([0],np.int32));state.step()
        self.assertEqual(float(state.latched.numpy()[0]),0.)
        state.enabled.assign(np.array([1],np.int32));state.step()
        self.assertAlmostEqual(float(state.strength.numpy()[0]),.2,places=6)

if __name__=='__main__':unittest.main()
