"""Parallel training gets its own fixed-controller budget, with strict errors."""
from pathlib import Path
import tempfile
import unittest

from rmuc2027_core.submissions import save_submission
from rmuc2027_research.rl.session import ArenaSession, load_config


class PolicyBudgetTests(unittest.TestCase):
    def test_invalid_training_timeouts_are_rejected(self):
        for value in (0, -1, float('inf'), float('nan'), True, '5'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                load_config({'policy_timeout_s': value})

    def test_slow_fixed_override_uses_training_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root/'slow.py'
            script.write_text('import time\ndef act(obs,state):\n time.sleep(.3)\n return {"1":{"move":[0,0]}},state\n')
            target = root/'agent'
            # A single-slot controller's valid submission; injected into red:1.
            save_submission(target, 'robot', {'1': script}, robot=1, season='2027')
            config = load_config({'policy_timeout_s': 2., 'rl_seconds': .1,
                'controllers': {'red:7': {'kind': 'learning'},
                                'red:1': {'kind': 'script', 'submission': str(target)}}})
            with ArenaSession(config) as session:
                session.reset(42)
                self.assertEqual(session.battle.metadata['decision_timeout_s'], 2.)
                for _, _, policy in session.battle.policies:
                    if hasattr(policy, 'timeout'):
                        self.assertEqual(policy.timeout, 2.)
                _, _, _, _, info = session.step([0, 0])
                self.assertEqual(info['metrics']['policy_errors'], 0)
                self.assertEqual(session.battle.diagnostics, [])
