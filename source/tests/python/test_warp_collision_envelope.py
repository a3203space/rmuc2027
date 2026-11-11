"""Body-local geometry and real swept-kernel tests; CPU only, no Match."""
import unittest
import warp as wp
from rmuc2027_core.sdk import DESIGNS
from rmuc2027_research.rl.warp_kinematic.pose import collision_heights
from rmuc2027_research.rl.warp_kinematic.motion import integrate


def swept_x(top, roof, body_z=0.):
    def a(value, dtype=float):return wp.array(value,dtype=dtype,device='cpu')
    pos=a([[0.,0.]],wp.vec2);velocity=a([[2.,0.]],wp.vec2)
    command=a([[1.,0.]],wp.vec2);zero=a([0.]);zero_i=a([0],wp.int32)
    output=a([[0.,0.]],wp.vec2);previous=a([[0.,0.]],wp.vec2)
    wp.launch(integrate,1,inputs=[pos,velocity,command,command,a([[2.,100.,.1]],wp.vec3),
        a([1],wp.int32),a([[.3,-1.,.5,1.]],wp.vec4),zero,
        a([[body_z+roof,body_z+roof+.2]],wp.vec2),a([[body_z,body_z+top]],wp.vec2),
        a([[1]],wp.int32),a([[25.,1.,1000.]],wp.vec3),a([[0.,0.]],wp.vec2),zero_i,
        a([[0.,0.,0.,0.]],wp.vec4),zero,zero_i,zero_i,output,.2,output,previous,1],device='cpu')
    return float(pos.numpy()[0,0])


class CollisionEnvelopeTests(unittest.TestCase):
    def test_folding_sentry_matches_native_solid_hull_aabb_at_each_progress(self):
        stand,folded=collision_heights(DESIGNS['2773'])
        # Independent native compound AABB fixtures, verified by Rust test
        # collision_envelope_regression (solid gimbal, not visual barrel).
        for progress,expected in ((0.,.465),(.25,.40125),(.5,.3375),(.75,.27375),(1.,.210)):
            self.assertAlmostEqual(stand+(folded-stand)*progress,expected,places=10)

    def test_legged_sentry_matches_body_local_aabb_without_adding_leg_height(self):
        d=DESIGNS['2772']
        for value in collision_heights(d):self.assertAlmostEqual(value,.46,places=10)
        for body_z in (-.15,-.025,.08):
            self.assertAlmostEqual(body_z+collision_heights(d)[0],body_z+.460)

    def test_folded_solid_gimbal_blocks_gap_that_old_envelope_missed(self):
        old=DESIGNS['2773']['folded_muzzle_z_m']+.06
        new=collision_heights(DESIGNS['2773'])[1]
        self.assertGreater(swept_x(old,.207),.39)
        for body_z in (-.018,0.,.37):
            self.assertLess(swept_x(new,.207,body_z),.201)
            self.assertGreater(swept_x(new,.213,body_z),.39)

    def test_legged_hull_passes_real_clearance_but_still_blocks_low_roof(self):
        old=DESIGNS['2772']['muzzle_z_m']+.06
        new=collision_heights(DESIGNS['2772'])[0]
        self.assertLess(swept_x(old,.49),.201)
        for body_z in (-.15,-.025,.08):
            self.assertGreater(swept_x(new,.49,body_z),.39)
            self.assertLess(swept_x(new,.455,body_z),.201)

    def test_unverified_roles_and_other_chassis_keep_existing_envelope(self):
        for key,d in DESIGNS.items():
            if d.get('role')=='sentry' and (d.get('transform_allowed') or d.get('chassis_type')=='wheel_legged'):
                continue
            with self.subTest(model=key):
                muzzle=d.get('muzzle_z_m',.4)
                self.assertEqual(collision_heights(d),(muzzle+.06,d.get('folded_muzzle_z_m',muzzle)+.06))


if __name__=='__main__':unittest.main()
