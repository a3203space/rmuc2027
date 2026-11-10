import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from rmuc2027_research.rl.learning import TraceCallback
from rmuc2027_research.rl import campaign


class TraceSerializationTests(unittest.TestCase):
    def test_completed_episode_survives_next_opening_failure(self):
        from stable_baselines3.common.vec_env import DummyVecEnv
        from rmuc2027_research.rl.env import SentryEnv
        from rmuc2027_research.rl.session import OpeningIncomplete
        with tempfile.TemporaryDirectory() as directory:
            env=SentryEnv(campaign.configuration(4,500))
            obs=[0.]*152
            env.session=SimpleNamespace(config={'_episode_log_dir':directory},close=lambda:None,
                reset=lambda seed:(obs,{}),
                step=lambda action:(obs,1.,True,False,{'stage':'match','time':420.,'winner':'red'}))
            vec=DummyVecEnv([lambda:env]);vec.reset()
            def failed_reset(seed):raise OpeningIncomplete('next opening failed')
            env.session.reset=failed_reset
            try:
                with self.assertRaises(OpeningIncomplete):vec.step(np.zeros((1,5)))
                files=list(Path(directory).glob('*.json'))
                self.assertEqual(len(files),1)
                saved=json.loads(files[0].read_text())
                self.assertEqual(saved['info']['time'],420.)
                self.assertTrue(saved['terminated'])
            finally:vec.close()

    def test_terminal_observation_is_serializable_in_summary_and_final(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'actions.jsonl'
            callback = TraceCallback(path, detailed=False)
            callback.model = SimpleNamespace(n_steps=64, _n_updates=4,
                get_env=lambda: SimpleNamespace(num_envs=1))
            callback.locals = dict(infos=[dict(terminal_observation=np.array([1., 2.], dtype=np.float32))], dones=[True])
            callback.num_timesteps = 64
            callback._on_training_start()
            callback._on_rollout_start()
            callback.last_progress = time.perf_counter()-16
            self.assertTrue(callback._on_step())
            callback._on_training_end()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row['kind'] for row in rows], ['summary', 'final'])
            for row in rows:
                self.assertEqual(row['metrics'][0]['terminal_observation'], [1., 2.])

    def test_resume_keeps_original_deadline_and_consumed_curriculum_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'configs').mkdir()
            (root/'configs/4.json').write_text('{}')
            checkpoint = root/'old.zip'
            checkpoint.write_bytes(b'saved')
            previous = dict(model=4, status='incomplete', wall_seconds=50., checkpoint=str(checkpoint), training={})
            (root/'campaign.json').write_text(json.dumps(dict(status='interrupted', started_unix=900.,
                elapsed_seconds=50., wall_budget_seconds=140., reserve_seconds=20., num_envs=1,
                policy_device='cpu', models=[4], allocations={'4':120.}, results=[previous])))
            commands = []

            def launch(command, **kwargs):
                commands.append(command)
                output = Path(command[command.index('--run-dir')+1])
                output.mkdir()
                (output/'training.json').write_text('{"actor_updated": true}')
                (output/'model.zip').write_bytes(b'updated')
                return SimpleNamespace(returncode=0, wait=lambda **kw: 0)

            with patch.object(campaign.time, 'time', return_value=1000.), \
                 patch.object(campaign.time, 'monotonic', return_value=100.), \
                 patch.object(campaign.subprocess, 'Popen', side_effect=launch), \
                 patch.object(campaign, 'stop_worker'):
                report = campaign.run(root, [4], resume=True)
            command = commands[0]
            self.assertEqual(float(command[command.index('--wall-seconds')+1]), 5.)
            self.assertEqual(float(command[command.index('--elapsed-wall-seconds')+1]), 50.)
            self.assertEqual(command[command.index('--resume')+1], str(checkpoint))
            self.assertEqual(report['started_unix'], 900.)
            self.assertEqual(len(report['attempts']), 2)


if __name__ == '__main__':
    unittest.main()
