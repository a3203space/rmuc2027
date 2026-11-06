import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from rmuc2027_research.rl.session import load_config
from rmuc2027_research.rl.learning import TraceCallback


class BackupIntervalTests(unittest.TestCase):
    def test_config_requires_positive_finite_interval(self):
        self.assertEqual(load_config()['github_backup_seconds'],300.)
        for value in (True,0,-1,float('nan'),float('inf'),'300'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                load_config({'github_backup_seconds':value})

    def test_upload_waits_for_interval_and_published_checkpoint(self):
        for interval,expected in ((300.,2),(600.,1)):
            callback=TraceCallback(Path('/tmp/checkpoint-interval-test/actions.jsonl'),detailed=False,backup_seconds=interval)
            callback.model=SimpleNamespace(n_steps=64,_n_updates=2,get_env=lambda:SimpleNamespace(num_envs=1))
            callback.locals={'infos':[]};callback.num_timesteps=256;callback.stream=io.StringIO();callback.backup=Mock()
            callback.last_checkpoint=0.;callback.last_backup=0.;callback.progress_started=0.
            with patch('rmuc2027_research.rl.learning.save_checkpoint') as save:
                for now in (299.,301.,500.,601.):
                    callback.last_progress=now
                    with patch('rmuc2027_research.rl.learning.time.perf_counter',return_value=now):
                        self.assertTrue(callback._on_step())
                self.assertEqual(save.call_count,2)
            self.assertEqual(callback.backup.submit.call_count,expected)
