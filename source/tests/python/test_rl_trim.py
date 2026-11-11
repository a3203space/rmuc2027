import unittest
from rmuc2027_core.engine import Match
from rmuc2027_research.rl.session import ArenaSession


class TrimTests(unittest.TestCase):
    def test_summary_preserves_state_and_vision(self):
        with Match(seed=42) as full, Match(seed=42) as compact:
            compact.training_summary = True
            for _ in range(6):
                commands = {'red': {'7': {'move': [1, 0]}}}
                a, b = full.step(commands), compact.step(commands)
                for key in ('tick', 'time', 'status', 'winner', 'reason', 'bases', 'teams', 'zones', 'positions'):
                    self.assertEqual(a[key], b[key], key)
                for ra, rb in zip(a['robots'], b['robots']):
                    self.assertEqual({key:ra[key] for key in rb}, rb)
                self.assertEqual(full.observation('red', [7]), compact.observation('red', [7]))
            self.assertEqual(full.digest(), compact.digest())
            self.assertEqual(full.snapshot(), compact.snapshot())

    def test_fixed_policies_at_five_hz_and_goal_control_at_ten(self):
        with ArenaSession({'fixed_policy_hz':5, 'rl_seconds':.4, 'episode_seconds':.4}) as session:
            session.reset(seed=42)
            session.step([0., 0.])
            self.assertEqual(session.controller.calls, 4)
            for _, _, policy in session.battle.policies:
                if policy is not session.controller:
                    self.assertEqual(policy.calls, 2)
            self.assertFalse(session.battle.diagnostics)

    def test_learning_control_can_run_at_two_hz(self):
        with ArenaSession({'fixed_policy_hz':5, 'learning_control_hz':2, 'rl_seconds':1., 'episode_seconds':1.}) as session:
            session.reset(seed=42)
            session.step([0., 0.])
            self.assertEqual(session.controller.calls, 2)
            for _, _, policy in session.battle.policies:
                if policy is not session.controller:
                    self.assertEqual(policy.calls, 5)
            self.assertFalse(session.battle.diagnostics)

    def test_all_learning_decisions_at_two_hz(self):
        with ArenaSession({'fixed_policy_hz':5, 'learning_control_hz':2,
                           'rl_seconds':.5, 'episode_seconds':1.}) as session:
            session.reset(seed=42)
            for count in (1, 2):
                session.step([0., 0.])
                self.assertEqual(session.controller.calls, count)
                self.assertAlmostEqual(session.battle.match.time, count*.5)
            for _, _, policy in session.battle.policies:
                if policy is not session.controller:
                    self.assertEqual(policy.calls, 5)

    def test_half_second_heat_budget(self):
        from rmuc2027_research.rl.portable import held_fire_safe
        robot = dict(ammo=100, heat=10, max_heat=100, trigger_interval_s=.05)
        self.assertFalse(held_fire_safe(robot, .5))
        self.assertTrue(held_fire_safe(robot, .1))
        robot['heat'] = 0
        self.assertTrue(held_fire_safe(robot, .5))
        robot.update(heat=80, ammo=2)
        self.assertTrue(held_fire_safe(robot, .5))
        robot.update(ammo=100, trigger_interval_s=.1, heat=50)
        self.assertTrue(held_fire_safe(robot, .5))
        robot['heat'] = 51
        self.assertFalse(held_fire_safe(robot, .5))

    def test_export_holds_entire_command_for_half_second(self):
        import sys
        import tempfile
        from pathlib import Path
        from unittest.mock import Mock, patch
        from rmuc2027_core import sdk
        from rmuc2027_core.submissions import load_submission
        from rmuc2027_research.rl.session import load_config, provenance
        from rmuc2027_research.rl.exporting import export_layers
        config = load_config({'rl_seconds':.5, 'learning_control_hz':2})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'policy'
            export_layers([], provenance(config), output)
            _, entries = load_submission(output)
            ns = {}
            with patch.dict(sys.modules, {'sdk':sdk}):
                exec(entries['7'].read_text(), ns)
            ns['encode_observation'] = Mock(return_value=[])
            ns['mlp_forward'] = Mock(return_value=[0, 0])
            def select(obs, state, *args):
                state['goal_time'] = obs['time']
            ns['select_goal'] = Mock(side_effect=select)
            ns['execute_goal'] = Mock(side_effect=[{'7':{'fire':True}}, {'7':{'fire':False}}])
            state = {}
            for tick in range(20):
                actions, state = ns['act']({'tick':tick, 'time':tick*.05}, state)
                self.assertEqual(actions['7']['fire'], tick < 10)
            self.assertEqual(ns['select_goal'].call_count, 2)
            self.assertEqual(ns['execute_goal'].call_count, 2)
            self.assertEqual(ns['execute_goal'].call_args.args[0]['policy_dt'], .5)
