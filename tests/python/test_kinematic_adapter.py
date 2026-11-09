import unittest
import numpy as np
from rmuc2027_core.engine import Match
from rmuc2027_core.sdk import DESIGNS
from rmuc2027_research.rl.warp_kinematic.arena import ArenaMotion
from rmuc2027_research.rl.warp_kinematic.protocol import DT


def request(model=2732,fold=0.,roof=False):
    robot=dict(id='red-4',model=model,team='red',role='infantry',pos=[5.,5.,0.],velocity=[0.,0.,0.],
               heading=0.,folded=fold,grounded=True,enabled=True,alive=True,radius=.25,power_limit=100.,
               move=[0.,0.],heading_goal=0.,yaw_rate=None,transform=False,jump=0.,coast=False,design=DESIGNS[str(model)])
    arena=dict(width=10.,height=10.,terrain=[],obstacles=[dict(x=4.,y=4.,w=2.,h=2.,height=.1,bottom=.25)] if roof else [])
    return dict(op='kinematic_frame',version=1,sequence=0,dt=DT,substeps=6,time=0.,robots=[robot],arena=arena)


class AdapterTests(unittest.TestCase):
    def test_coast_flag_reaches_motion_and_releasing_it_brakes(self):
        adapter=ArenaMotion();command=request()
        robot=command['robots'][0];robot.update(coast=True,velocity=[2.,0.,0.])
        try:
            first=adapter.step(command)
            self.assertGreater(first['velocities'][-1][0][0],1.9)
            robot.update(pos=first['positions'][-1][0],velocity=first['velocities'][-1][0],coast=False)
            command['sequence']=1
            second=adapter.step(command)
            self.assertLess(second['velocities'][-1][0][0],first['velocities'][-1][0][0]-.1)
        finally:adapter.close()

    def test_fully_folded_drive_uses_original_half_speed_rule(self):
        speeds=[]
        for folded in (False,True):
            adapter=ArenaMotion();command=request(fold=float(folded))
            robot=command['robots'][0];robot.update(move=[.2,0.],transform=folded)
            try:
                for tick in range(20):
                    command['sequence']=tick;frame=adapter.step(command)
                    robot.update(pos=frame['positions'][-1][0],velocity=frame['velocities'][-1][0],
                        heading=frame['headings'][-1][0],folded=frame['folded'][-1][0])
                speeds.append(robot['velocity'][0])
            finally:adapter.close()
        self.assertGreater(speeds[0],.7)
        self.assertAlmostEqual(speeds[1],speeds[0]*.5,places=6)

    def test_real_engine_gpu_frame_updates_position_and_fold(self):
        adapter=ArenaMotion()
        with Match(physics_adapter=adapter,battle_settings={'drones_enabled':False,'timed_assembly':True}) as match:
            initial=next(r for r in match.snapshot()['robots'] if r['id']=='red-4')
            for _ in range(20):frame=match.step({'red':{'4':{'move':[.3,0.],'transform':True}}})
            robot=next(r for r in frame['robots'] if r['id']=='red-4')
            self.assertGreater(robot['x'],initial['x']+.5)
            self.assertLess(robot['battery_remaining_wh'],initial['battery_remaining_wh'])
            self.assertGreater(robot['motor_power_w'],0.)
            self.assertAlmostEqual(robot['transform_progress'],1.,places=5)
            self.assertEqual(adapter.next_sequence,20)
            self.assertTrue(adapter.motion.device.is_cuda)

    def test_cannot_unfold_beneath_roof(self):
        adapter=ArenaMotion()
        frame=adapter.step(request(fold=1.,roof=True))
        np.testing.assert_array_equal(frame['folded'],np.ones((6,1)))
        adapter.close()

    def test_wheel_leg_does_not_strafe(self):
        adapter=ArenaMotion()
        command=request(model=2731)
        command['robots'][0]['move']=[0.,1.]
        frame=adapter.step(command)
        np.testing.assert_allclose(np.asarray(frame['positions'])[:,:,:2],np.full((6,1,2),5.),atol=1e-6)
        adapter.close()

    def test_jump_outside_supported_map_does_not_start(self):
        adapter=ArenaMotion();command=request(model=2731);command['robots'][0]['jump']=1.
        frame=adapter.step(command)
        self.assertTrue(np.asarray(frame['grounded']).all())
        self.assertNotEqual(adapter.jump_status.numpy()[0],1)

    def test_owner_restricted_zone_blocks_only_opponent(self):
        for team,blocked in (('red',False),('blue',True)):
            adapter=ArenaMotion();command=request();robot=command['robots'][0]
            robot['team']=team;robot['move']=[1.,0.]
            command['arena']['forbidden']=[dict(kind='restricted',owner='red',x=5.8,y=3.,w=.2,h=4.)]
            for index in range(20):
                command['sequence']=index
                frame=adapter.step(command)
                robot['pos']=frame['positions'][-1][0]
                robot['velocity']=frame['velocities'][-1][0]
                robot['heading']=frame['headings'][-1][0]
                robot['folded']=frame['folded'][-1][0]
            self.assertEqual(robot['pos'][0]<5.8,blocked)
            adapter.close()
