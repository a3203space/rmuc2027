import unittest
import numpy as np
from rmuc2027_research.rl.warp_kinematic.arena import ArenaMotion
from test_kinematic_adapter import request
from test_kinematic_traversal import launch_request
from rmuc2027_core.sdk import _REFERENCE_ARENA,terrain_height


def feedback(command,frame):
    command['sequence']+=1
    robot=command['robots'][0]
    robot.update(pos=frame['positions'][-1][0],velocity=frame['velocities'][-1][0],
                 heading=frame['headings'][-1][0],folded=frame['folded'][-1][0],grounded=frame['grounded'][-1][0])


class LoadedPoseTests(unittest.TestCase):
    def test_midframe_airborne_death_does_not_become_a_floating_ride_height(self):
        adapter=ArenaMotion();command=launch_request(8.6,0.)
        try:
            first=adapter.step(command);robot=command['robots'][0]
            # Referee death freezes the first microstep, before the last
            # predicted airborne pose. This must cancel flight, not reset the
            # chassis' ground datum to its airborne altitude.
            robot.update(pos=first['positions'][0][0],velocity=[0.,0.,0.],grounded=False,alive=False,enabled=False)
            command['sequence']=1
            for _ in range(25):frame=adapter.step(command);feedback(command,frame)
            self.assertTrue(robot['grounded'])
            self.assertAlmostEqual(robot['pos'][2],terrain_height(_REFERENCE_ARENA,*robot['pos'][:2])-.025,places=5)
        finally:adapter.close()

    def test_negative_authoritative_height_is_not_clamped_for_a_wreck(self):
        adapter=ArenaMotion();command=request()
        command['robots'][0].update(pos=[5.,5.,-.02],enabled=False,alive=False)
        try:
            frame=adapter.step(command)
            np.testing.assert_allclose(np.asarray(frame['positions'])[:,:,2],-.02,atol=1e-7)
        finally:adapter.close()

    def test_leg_target_is_rate_limited_and_changes_loaded_camera_height(self):
        adapter=ArenaMotion();command=request(model=2731);robot=command['robots'][0]
        design=robot['design'];extensions=[];heights=[]
        try:
            for target in (design['leg_min_m'],design['leg_max_m'],None):
                robot['leg_height']=target
                for _ in range(12):
                    frame=adapter.step(command);extensions.append(float(adapter.leg_command.numpy()[0]));feedback(command,frame)
                heights.append(robot['pos'][2]+design['muzzle_z_m'])
            self.assertGreater(heights[1]-heights[0],.20)
            self.assertAlmostEqual(heights[2],design['muzzle_z_m']-.025,places=5)
            self.assertLessEqual(np.abs(np.diff(extensions)).max(),design['leg_speed_mps']*.05+1e-6)
        finally:adapter.close()

    def test_leg_extension_cannot_raise_hull_through_a_roof(self):
        adapter=ArenaMotion();command=request(model=2731);robot=command['robots'][0]
        # A folded transport stance fits; the raised stance cannot fit.
        roof=.415
        command['arena']['obstacles']=[dict(x=4.,y=4.,w=2.,h=2.,bottom=roof,height=.1)]
        robot['leg_height']=robot['design']['leg_min_m']
        try:
            for _ in range(10):frame=adapter.step(command);feedback(command,frame)
            robot['leg_height']=robot['design']['leg_max_m']
            for _ in range(12):
                frame=adapter.step(command);feedback(command,frame)
                self.assertLessEqual(np.asarray(frame['positions'])[:,:,2].max()+robot['design']['muzzle_z_m']+.06,roof+1e-6)
        finally:adapter.close()

    def test_teleport_does_not_reuse_old_extended_leg_state(self):
        adapter=ArenaMotion();command=request(model=2731);robot=command['robots'][0]
        robot['leg_height']=robot['design']['leg_max_m']
        try:
            for _ in range(12):frame=adapter.step(command);feedback(command,frame)
            robot.update(pos=[3.,3.,0.],velocity=[0.,0.,0.],leg_height=None)
            frame=adapter.step(command)
            self.assertAlmostEqual(frame['positions'][-1][0][2],-.025,places=5)
        finally:adapter.close()
