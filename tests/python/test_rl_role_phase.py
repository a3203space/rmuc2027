import unittest
from rmuc2027_research.rl.portable import learning_ready, fixed_opening
from rmuc2027_research.rl.session import load_config


class RolePhaseTests(unittest.TestCase):
    def test_roles_and_default_controller(self):
        for slot in (1, 3, 4, 7):
            cfg = load_config({'slot': slot, 'team': 'blue'})
            self.assertEqual(cfg['controllers'], {f'blue:{slot}': {'kind': 'learning'}})
        with self.assertRaises(ValueError): load_config({'slot': 6})

    def test_installation_gate_and_latch(self):
        state = {'navigation': {'old': 1}, 'held_actions': {'1': {}}, 'goal': [1, 2]}
        for phase in ('locked', 'unlocked', 'installing', 'fitted'):
            self.assertFalse(learning_ready({'role': 'heavy', 'launcher': phase}, state))
        self.assertTrue(learning_ready({'role': 'heavy', 'launcher': 'installed'}, state))
        self.assertNotIn('navigation', state)
        self.assertNotIn('goal', state)
        self.assertTrue(learning_ready({'role': 'heavy', 'launcher': 'locked', 'alive': False}, state))

    def test_opening_has_no_actor_action(self):
        class Baseline:
            @staticmethod
            def act(obs, memory): return {'1': {'brake': True}}, {'ticks': 1}
        obs = {'controlled_ids': ['1'], 'robots': [{'slot': 1, 'role': 'heavy', 'launcher': 'fitted'}]}
        state = {}
        self.assertEqual(fixed_opening(obs, state, {'slot': 1}, Baseline), {'1': {'brake': True}})
        self.assertEqual(state['opening']['ticks'], 1)

    def test_timed_opening_replaces_pickup(self):
        from rmuc2027_core.sdk import timed_assembly_action, assembly_work_point
        from rmuc2027_core.engine import Match
        with Match() as match:
            obs = match.observation('red', [1])
            obs['simulation_assumptions']['timed_assembly'] = True
            robot = next(r for r in obs['robots'] if r['slot'] == 1)
            robot['x'], robot['y'] = assembly_work_point(robot, obs['arena'])
            command = timed_assembly_action(robot, {'pick_unit': True}, obs, {})
            self.assertEqual(command['assembly'], 1)
            self.assertNotIn('pick_unit', command)
            zone = next(z for z in obs['arena']['zones'] if z['id'] == 'red_assembly')
            # A valid card position away from the old arm docking point must stop.
            robot['x'], robot['y'] = zone['x']-.3, zone['y']
            command = timed_assembly_action(robot, {'pick_unit': True}, obs, {})
            self.assertEqual(command['move'], [0, 0])
            self.assertEqual(command['assembly'], 1)
