import unittest
from unittest.mock import patch
from rmuc2027_core import sdk

class JumpTests(unittest.TestCase):
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
