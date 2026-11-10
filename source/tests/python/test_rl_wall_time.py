"""Wall-clock stopping must save a readable model, even before an update."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

AVAILABLE = all(importlib.util.find_spec(n) for n in ('torch', 'gymnasium', 'stable_baselines3'))


@unittest.skipUnless(AVAILABLE, 'Optional neural training dependencies unavailable')
class WallTimeTests(unittest.TestCase):
    def test_deadline_saves_model_and_reports_untrained_partial_rollout(self):
        from rmuc2027_research.rl.learning import train
        from rmuc2027_research.rl.session import load_config
        config = load_config({'rl_seconds': .1, 'ppo': {'n_steps': 2, 'batch_size': 2, 'n_epochs': 1}})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'run'
            report = train(config, output, None, wall_seconds=.000001)
            self.assertEqual(report['stop_reason'], 'wall_time_limit')
            self.assertTrue((output/'model.zip').is_file())
            self.assertTrue(report['reload_prediction_equal'])
            self.assertEqual(report['gradient_epochs'], 0)
            self.assertFalse(report['actor_updated'])

    def test_heavy_deadline_does_not_create_opening_samples(self):
        from rmuc2027_research.rl.learning import train
        from rmuc2027_research.rl.session import load_config
        from stable_baselines3 import PPO
        config = load_config({'slot':1,'timed_assembly':True})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'run'
            report = train(config, output, None, wall_seconds=.000001)
            self.assertEqual(report['stop_reason'], 'wall_time_limit_during_opening')
            self.assertEqual(report['actual_steps'], 0)
            restored = PPO.load(output/'model.zip', device='cpu')
            self.assertEqual(restored.num_timesteps, 0)
