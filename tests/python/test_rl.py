"""RL boundary regression tests; engine/runtime tests need no neural libraries."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rmuc2027_core.engine import Match
from rmuc2027_research.strategies.battle import Battle
from rmuc2027_research.rl.portable import OBS_SIZE, SdkNavigator, encode_observation, execute_goal, mlp_forward, select_goal
from rmuc2027_research.rl.session import GoalController, load_config, provenance, resource
from rmuc2027_research.rl.exporting import export_layers
from rmuc2027_core.replay import verify_replay


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with Match(seed=42, map_name='rmuc2027') as match:
            cls.observation = match.observation('red', [7])
        cls.config = load_config()

    def test_fixed_features_are_bounded_and_ignore_referee_state(self):
        original = encode_observation(self.observation, {})
        obs = copy.deepcopy(self.observation)
        obs['referee'] = {'robots': [{'x': float('nan'), 'hp': 12345}], 'reward': 1000}
        self.assertEqual(original, encode_observation(obs, {}))
        self.assertEqual(len(original), OBS_SIZE)
        self.assertTrue(all(math.isfinite(v) and -1 <= v <= 1 for v in original))
        obs['controlled_ids'] = ['1', '7']
        with self.assertRaises(ValueError):
            encode_observation(obs, {})

    def test_interrupted_transport_cannot_consume_a_late_frame_as_digest(self):
        with Match(seed=42, map_name='rmuc2027') as match:
            with patch.object(match, '_read_line', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    match.step({})
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                match.digest()

    @unittest.skipUnless(importlib.util.find_spec('gymnasium'), 'Optional Gymnasium dependency unavailable')
    def test_non_numeric_gym_action_closes_every_owned_child(self):
        from rmuc2027_research.rl.env import SentryEnv
        env = SentryEnv(self.config)
        try:
            env.reset(seed=42)
            pids = env.process_ids()
            with self.assertRaises(ValueError):
                env.step(['invalid', 'coordinates'])
            self.assertIsNone(env.session.battle)
            self.assertFalse([pid for pid in pids if Path(f'/proc/{pid}').exists()])
        finally:
            env.close()

    def test_coordinate_projection_rejects_nonfinite_and_avoids_obstacles(self):
        navigator = SdkNavigator(.25)
        state = {}
        for bad in ([float('nan'), 0], [float('inf'), 0], [0]):
            with self.assertRaises(ValueError):
                select_goal(self.observation, state, bad, navigator, self.config)
        select_goal(self.observation, state, [0, 0], navigator, self.config)
        self.assertEqual(state['proposed_goal'], [14.0, 7.5])
        self.assertTrue(all(math.isclose(v/.25, round(v/.25)) for v in state['goal']))
        self.assertNotEqual(state['goal'], state['proposed_goal'])

    def test_controller_owns_only_sentry_and_recovery_has_hysteresis(self):
        controller = GoalController(self.config)
        obs = copy.deepcopy(self.observation)
        robot = next(r for r in obs['robots'] if r['slot'] == 7)
        controller.select(obs, [0, 0])
        for hp, alive, weak, expected in ((.2, True, False, 'recover'), (.5, True, False, 'recover'),
                (.9, True, False, 'policy'), (0, False, False, 'dead'), (.9, True, True, 'recover'), (.9, True, False, 'policy')):
            robot.update(hp=robot['max_hp']*hp, alive=alive, weak=weak)
            actions, error = controller.decide(obs)
            self.assertIsNone(error)
            self.assertEqual(set(actions), {'7'})
            self.assertEqual(controller.state['mode'], expected)
            if not alive:
                self.assertEqual(actions['7']['move'], [0, 0])
        controller.close()

    def test_truncation_does_not_finish_game_or_award_win(self):
        from rmuc2027_research.rl.session import ArenaSession
        config = {**self.config, 'episode_seconds': 1}
        with ArenaSession(config) as session:
            session.reset(42)
            _, _, terminated, truncated, info = session.step([0, 0])
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(session.battle.match.status, 'running')
            self.assertEqual(info['reward_terms']['outcome'], 0)
            with self.assertRaises(RuntimeError):
                session.step([0, 0])

    def test_natural_end_and_no_recording_keep_constant_frame_count(self):
        from rmuc2027_research.rl.session import ArenaSession
        with ArenaSession(self.config) as session:
            session.reset(42)
            session.battle.queue_referee({'kind': 'forfeit', 'team': 'blue'})
            _, reward, terminated, truncated, info = session.step([0, 0])
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info['reward_terms']['outcome'], 10)
            self.assertEqual(len(session.battle.frames), 1)
            self.assertEqual(session.battle.actions, [])
            with self.assertRaises(RuntimeError):
                session.battle.replay()

    def test_exported_stdlib_policy_runs_in_existing_worker_and_replays(self):
        # Test fixture only: actual trained-weight parity is checked by export_model.
        layers = [{'weight': [[0.0]*OBS_SIZE for _ in range(2)], 'bias': [0.0, 0.0]}]
        self.assertEqual(mlp_forward([0]*OBS_SIZE, layers), [0, 0])
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'squad'
            metadata = export_layers(layers, provenance(self.config), target)
            self.assertLess(metadata['sentry_source_bytes'], 512*1024)
            with Battle(target, resource(self.config['opponent']), 42, 1, 'rmuc2027', 'trusted-local') as battle:
                result = battle.run()
                self.assertEqual(result['policy_errors'], 0, battle.diagnostics)
                replay = Path(directory)/'match.json'
                battle.save(replay)
            self.assertEqual(verify_replay(replay)['digest'], result['digest'])


if __name__ == '__main__':
    unittest.main()
