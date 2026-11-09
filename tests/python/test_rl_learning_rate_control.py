import json
from pathlib import Path
import tempfile
import unittest

import gymnasium as gym
import torch

from rmuc2027_research.rl.learning import TimedPPO


class LearningRateControlTests(unittest.TestCase):
    def test_request_applies_to_optimizer_and_survives_checkpoint_reload(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            env=gym.make('CartPole-v1')
            try:
                model=TimedPPO('MlpPolicy',env,n_steps=8,batch_size=8,n_epochs=1,learning_rate=3e-4,device='cpu')
                path=Path(temp)/'learning-rate.json'
                model.learning_rate_control_path=str(path)
                model.update_log_path=str(Path(temp)/'updates.jsonl')
                path.write_text(json.dumps({'learning_rate':1e-4,'reason':'test intervention'}))
                model.learn(total_timesteps=16)
                self.assertEqual(model.policy.optimizer.param_groups[0]['lr'],1e-4)
                events=(Path(temp)/'learning-rate-events.jsonl').read_text().splitlines()
                self.assertEqual(len(events),1)
                self.assertEqual(len((Path(temp)/'updates.jsonl').read_text().splitlines()),2)
                model.save(Path(temp)/'model.zip')
                restored=TimedPPO.load(Path(temp)/'model.zip',device='cpu')
                self.assertEqual(restored.lr_schedule(.5),1e-4)
            finally:env.close()

    def test_invalid_request_keeps_existing_schedule(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            env=gym.make('CartPole-v1')
            try:
                model=TimedPPO('MlpPolicy',env,n_steps=8,batch_size=8,n_epochs=1,learning_rate=3e-4,device='cpu')
                path=Path(temp)/'learning-rate.json';model.learning_rate_control_path=str(path)
                for value in ('{', '{"learning_rate": -1}', '{"learning_rate": true}'):
                    path.write_text(value);model.apply_learning_rate_control()
                    self.assertEqual(model.lr_schedule(.5),3e-4)
            finally:env.close()
