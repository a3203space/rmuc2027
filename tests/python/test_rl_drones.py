import unittest
from rmuc2027_core.engine import Match
from rmuc2027_core.sdk import DESIGNS, navigate_action

class SceneTests(unittest.TestCase):
    def test_no_drones_but_hangars_remain(self):
        with Match(battle_settings={'drones_enabled':False}) as match:
            frame = match.snapshot()
            self.assertEqual(len(frame['robots']), 10)
            self.assertTrue(all(not r.get('drones', []) for r in frame['robots']))
            for _ in range(20):
                frame = match.step({})
            self.assertTrue(all(not r.get('drones', []) for r in frame['robots']))
        with Match() as match:
            self.assertTrue(any(r.get('drones', []) for r in match.snapshot()['robots']))

    def test_serial_airborne_then_settle(self):
        # 305 no longer advertises step climbing. The serial stair state
        # machine is exercised by the current 2772 sentry hardware contract.
        for model in (2772,):
            design = DESIGNS[str(model)]
            self.assertEqual(design['leg_type'], 'serial')
            self.assertTrue(design['capabilities']['jump'])
            self.assertGreater(design['capabilities']['step_climb_m'], 0.)
            robot = dict(model_slot=model, role=design['role'], team='red', x=9., y=7.5,
                         grounded=False, alive=True, upright=1.)
            state = {'goal':[14.,7.5], 'entry':{'phase':'align','since':0.,'jumps':1,'last_jump':0.}}
            obs = {'time':.5,'arena':{'layout_revision':'chapter4-2027-v1'}}
            action = navigate_action(robot, (14.,7.5), obs, state)
            self.assertEqual(action['jump'], 0.)
            self.assertEqual(action['move'], (0.,0.))
            robot['grounded'] = True
            obs['time'] = 1.
            action = navigate_action(robot, (14.,7.5), obs, state)
            self.assertTrue(action['brake'])
            self.assertEqual(state['entry']['phase'], 'settle')
