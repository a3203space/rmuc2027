"""Explicit training reward configuration; no simulation or policy changes."""
import math
import unittest

from rmuc2027_research.rl.session import load_config


class V4CaptureRewardTests(unittest.TestCase):
    def config(self, **extra):
        return load_config(dict(policy_version=4, slot=7,
            navigation_mode='adaptive', rl_seconds=.1, **extra))

    def test_default_preserves_original_zero_occupancy_reward(self):
        cfg = self.config(rewards={'capture_seconds': .02})
        self.assertEqual(cfg['rewards']['capture_seconds'], 0.)
        self.assertEqual(cfg['rewards']['control'], .01)
        self.assertEqual(cfg['rewards']['outcome'], 10.)
        self.assertEqual(cfg['rewards']['death'], -.1)

    def test_explicit_finite_nonnegative_reward(self):
        for value in (0, 0., .02, 1):
            with self.subTest(value=value):
                cfg = self.config(v4_capture_reward=value)
                self.assertEqual(cfg['rewards']['capture_seconds'], float(value))
                self.assertEqual(cfg['rewards']['control'], .01)
                self.assertEqual(cfg['rewards']['outcome'], 10.)

    def test_invalid_reward_rejected(self):
        for value in (-.01, math.inf, -math.inf, math.nan, True, False, '.02', None):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'finite and nonnegative'):
                    self.config(v4_capture_reward=value)

    def test_evaluation_can_explicitly_zero_training_reward(self):
        training = self.config(v4_capture_reward=.02)
        evaluation = load_config(dict(training, evaluation=True, v4_capture_reward=0.))
        self.assertEqual(evaluation['rewards']['capture_seconds'], 0.)
        self.assertEqual(evaluation['rewards']['control'], training['rewards']['control'])
        self.assertEqual(evaluation['rewards']['outcome'], training['rewards']['outcome'])


if __name__ == '__main__':
    unittest.main()
