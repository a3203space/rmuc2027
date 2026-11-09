import math
import unittest
from unittest.mock import patch
from rmuc2027_core import sdk

class JumpTests(unittest.TestCase):
    def test_two_hz_launch_precedes_crossing_the_takeoff_line(self):
        for team in ('red', 'blue'):
            flip=team=='blue'
            robot=dict(model_slot=2714,team=team,alive=True,
                       x=28-8.442 if flip else 8.442,y=7.5,z=.1,
                       vx=-1.725 if flip else 1.725,vy=0.,vz=0.,
                       upright=1.,grounded=True,heading=math.pi if flip else 0.)
            state={'heavy_stage_ready':True,'goal':[14,7.5],
                   'entry':{'phase':'align','since':4.5,'jumps':0,'last_jump':-10.}}
            obs={'time':5.,'policy_dt':.5,'arena':{'layout_revision':'chapter4-2027-v1'}}
            command=sdk.navigate_action(robot,(14,7.5),obs,state)
            self.assertEqual(command['jump'],1.)
            self.assertFalse(command['brake'])
            self.assertEqual(command['move'],(0.,0.))
            self.assertEqual(command['leg_height'],sdk.DESIGNS['2714']['leg_min_m'])

    def test_exact_two_second_boundary_cannot_issue_fourth_jump(self):
        robot=dict(model_slot=2731,team='red',alive=True,x=9.,y=7.5,z=.1,
                   vx=0.,vy=0.,vz=0.,upright=1.,grounded=True,heading=0.)
        state={'goal':[14,7.5],'entry':{'phase':'align','since':0.,
               'jumps':3,'last_jump':2.,'stable_since':2.}}
        obs={'time':4.,'policy_dt':.5,'arena':{'layout_revision':'chapter4-2027-v1'}}
        self.assertFalse(sdk.navigate_action(robot,(14,7.5),obs,state).get('jump'))
        self.assertEqual(state['entry']['jumps'],3)

    def test_heavy_keeps_transport_stance_after_arrival(self):
        robot=dict(model_slot=2714,role='heavy',team='red',alive=True,x=12.,y=7.5,z=.4,
                   vx=0.,vy=0.,vz=0.,upright=1.,grounded=True,heading=0.)
        state={'heavy_low_stance':True,'heavy_stage_ready':True,'goal':[12.,7.5]}
        obs={'time':15.,'policy_dt':.5,'arena':sdk._REFERENCE_ARENA}
        command=sdk.navigate_action(robot,(12.,7.5),obs,state)
        self.assertEqual(command['leg_height'],sdk.DESIGNS['2714']['leg_min_m'])
        self.assertNotIn('entry',state)

    def test_launch_keeps_momentum_without_driving_or_braking(self):
        for team in ('red', 'blue'):
            flip = team == 'blue'
            robot = dict(model_slot=2731, team=team, alive=True,
                         x=19.4 if flip else 8.6, y=7.5, z=.12, vz=0.,
                         upright=1., grounded=True, heading=math.pi if flip else 0.)
            state = {'goal':[14,7.5], 'entry':{'phase':'align', 'since':0.,
                     'jumps':0, 'last_jump':-10.}}
            obs = {'time':1., 'arena':{'layout_revision':'chapter4-2027-v1'}}
            command = sdk.navigate_action(robot, (14,7.5), obs, state)
            self.assertEqual(command['jump'], 1.)
            self.assertEqual(command['move'], (0.,0.))
            self.assertFalse(command['brake'])
            obs['time'] = 1.1
            command = sdk.navigate_action(robot, (14,7.5), obs, state)
            self.assertEqual(command['jump'], 0.)
            self.assertEqual(command['move'], (0.,0.))
            self.assertFalse(command['brake'])

    def test_rejump_requires_stable_ground_not_rim_contact(self):
        for team in ('red','blue'):
            flip=team=='blue'
            robot=dict(model_slot=2731,team=team,alive=True,x=18 if flip else 10,y=7.5,
                       vx=0.,vy=0.,vz=1.5,heading=3.141592653589793 if flip else 0.,upright=1.,grounded=True)
            state={'goal':[14,7.5], 'entry':{'phase':'align','since':0.,'jumps':1,'last_jump':0.}}
            obs={'time':1.2,'arena':{'layout_revision':'chapter4-2027-v1'}}
            with patch.object(sdk,'_physical_toward',return_value=(1,0)):
                action=sdk.navigate_action(robot,(14,7.5),obs,state)
                self.assertFalse(action.get('jump'))
                robot['vz']=0.
                obs['time']=1.3
                self.assertFalse(sdk.navigate_action(robot,(14,7.5),obs,state).get('jump'))
                obs['time']=1.8
                self.assertEqual(sdk.navigate_action(robot,(14,7.5),obs,state)['jump'],1.)

    def test_failed_jump_backs_out_before_retry(self):
        for team in ('red','blue'):
            flip=team=='blue'
            robot=dict(model_slot=2731,team=team,alive=True,x=17.85 if flip else 10.15,
                       y=7.5,vz=0.,upright=.79,grounded=True)
            state={'goal':[14,7.5],'entry':{'phase':'align','since':0.,'jumps':1,'last_jump':1.}}
            obs={'time':4.,'arena':{'layout_revision':'chapter4-2027-v1'}}
            action=sdk.navigate_action(robot,(14,7.5),obs,state)
            self.assertEqual(action['move'],(.35 if flip else -.35,0.))
            self.assertEqual(action['jump'],0.)
            robot['x']=19.7 if flip else 8.3
            state['entry']['jumps']=3
            obs['time']=5.
            sdk.navigate_action(robot,(14,7.5),obs,state)
            self.assertNotIn('entry',state)
            self.assertTrue(state['failed'])

    def test_robot_on_platform_is_not_pulled_back_to_rim(self):
        for team in ('red','blue'):
            flip=team=='blue'
            r=dict(model_slot=2731,team=team,alive=True,x=17.5 if flip else 10.5,
                   y=7.5,z=.34,vz=0.,upright=.99,grounded=True)
            state={'goal':[14,7.5],'entry':{'phase':'align','since':1.,'jumps':1,'last_jump':5.8,'takeoff_z':.1}}
            obs={'time':8.,'arena':{'layout_revision':'chapter4-2027-v1'}}
            a=sdk.navigate_action(r,(14,7.5),obs,state)
            self.assertEqual(a['move'],(0.,0.))
            obs['time']=8.4
            a=sdk.navigate_action(r,(14,7.5),obs,state)
            self.assertEqual(a['move'],(-.35 if flip else .35,0.))
            self.assertNotIn('entry',state)

    def test_tunnel_capability_does_not_depend_on_model_id(self):
        design = dict(sdk.DESIGNS['2732'])
        with patch.dict(sdk.DESIGNS, {'9999': design}):
            robot = dict(model_slot=9999, team='red', alive=True, x=9.32, y=11.7,
                         heading=1.57079632679, transform_progress=0., grounded=True)
            state = {}
            obs = {'time':1., 'arena':{'layout_revision':'chapter4-2027-v1'}}
            command = sdk.navigate_action(robot,(14.,7.5),obs,state)
            self.assertTrue(command['transform'])
            self.assertEqual(command['move'],(0.,0.))
