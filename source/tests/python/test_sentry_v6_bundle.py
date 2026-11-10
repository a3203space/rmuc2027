"""Toy v6 package relocation/resume; no native simulation or PPO campaign."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rmuc2027_research.rl import privileged_aux
from rmuc2027_research.rl.aux_policy import AuxLayerNormPolicy, export_actor as export_aux
from rmuc2027_research.rl.sentry_v6_policy import export_actor
from rmuc2027_research.rl.v6_ppo import HybridPPO
from rmuc2027_research.rl.v6_training import save_state
from rmuc2027_research.rl.v4_training import save_state as save_aux_state
from rmuc2027_research.rl.v6_bundle import create_bundle, extract_bundle, load_runtime_config, preflight, prepare_resume_run
from rmuc2027_research.rl.v6_sampling import TrainingCoordinator
import test_sentry_v4_bundle as v4_fixture
from test_sentry_v6_training import ToyEnv, make_model, fill_update, assert_optimizer_equal


class AuxiliaryToy(ToyEnv):
    def __init__(self):
        super().__init__()
        self.observation_space = privileged_aux.observation_space()
        self.action_space = privileged_aux.action_space()

    def _obs(self):
        return dict(actor=np.zeros(self.observation_space.spaces['actor'].shape, np.float32),
                    action_mask=np.ones(self.observation_space.spaces['action_mask'].shape, np.float32))


class V6BundleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_real_toy_checkpoint_bundle_relocates_three_optimizers_and_all_artifacts(self):
        main, vec = make_model()
        aux_vec = DummyVecEnv([AuxiliaryToy])
        try:
            with tempfile.TemporaryDirectory() as directory:
                source, run, _ = v4_fixture.BundleTests().fixture(directory)
                for name in ('run-results.json', 'run-manifest.json', 'sampler-coordinator.json', 'main/migration.json'):
                    (run / name).write_text('{}')
                (run / 'sampler-coordinator.json').unlink()
                coordinator = TrainingCoordinator(run / 'sampler-coordinator.json')
                lease = coordinator.acquire(0)
                coordinator.progress(0, lease['id'], valid_steps=5, seconds=.5)
                (run / 'main/coordinator-state.json').write_text(json.dumps(coordinator.snapshot()))
                coordinator.progress(0, lease['id'], valid_steps=8, seconds=.8)
                (run / 'capability-profiles').mkdir()
                for side in ('red', 'blue'):
                    (run / f'capability-profiles/evaluation-{side}.json').write_text('{}')
                for name in ('training', 'evaluation'):
                    target = run / 'replays' / name
                    target.mkdir(parents=True)
                    (target / 'replay.json').write_text('{"recorded":true}')
                backup = run / 'github-backups/.git'
                backup.mkdir(parents=True)
                (backup / 'objects').write_bytes(b'never embed git backups')
                (run / 'github-backups/part.zip').write_bytes(b'never recursively package backups')
                (run / 'nested').mkdir()
                (run / 'nested/private.sqlite').write_bytes(b'no database in reports')
                (run / 'release-index.json').write_text('{"outside_zip_only":true}')
                metadata_path = run / 'main/experiment.json'
                metadata = json.loads(metadata_path.read_text())
                metadata['config'].update(policy_version=6, _v6_sampler_path='/old/host/coordinator.json',
                    _v6_replay_dir='/old/host/replays', _worker_id=9)
                metadata_path.write_text(json.dumps(metadata))
                save_state(main, vec, run / 'main', 'initialized.zip')
                fill_update(main)
                save_state(main, vec, run / 'main')
                (run / 'main/actor.json').write_text(json.dumps(export_actor(main.policy)))
                generation = run / 'main/checkpoints/update-0001'
                generation.mkdir(parents=True)
                for name in ('model.zip', 'resume-state.pt', 'samplers.json'):
                    shutil.copy2(run / 'main' / name, generation / name)
                (run / 'main/checkpoint-ready.json').write_text('{"generation":"checkpoints/update-0001"}')
                for role in ('infantry', 'heavy'):
                    auxiliary = PPO(AuxLayerNormPolicy, aux_vec, n_steps=4, batch_size=4, n_epochs=1, device='cpu')
                    save_aux_state(auxiliary, aux_vec, run / role)
                    (run / role / 'actor.json').write_text(json.dumps(export_aux(auxiliary.policy, family=role)))
                archive = Path(directory) / 'release.zip'
                result = create_bundle(run, archive, source_root=source)
                self.assertEqual(result['manifest']['format'], 'rmuc2027-sentry-v6-resume')
                with zipfile.ZipFile(archive) as zipped:
                    names = zipped.namelist()
                    self.assertFalse(any('github-backups' in name or name.endswith('.sqlite') for name in names))
                    self.assertFalse(any(name.endswith('release-index.json') for name in names))
                    for required in ('replays/training/replay.json', 'replays/evaluation/replay.json',
                                     'models/main/checkpoints/update-0001/model.zip',
                                     'models/main/checkpoints/update-0001/resume-state.pt',
                                     'models/main/migration.json', 'run-state/run-results.json',
                                     'run-state/sampler-coordinator.json'):
                        self.assertIn(required, names)
                moved = Path(directory) / 'another-machine'
                extract_bundle(archive, moved)
                cfg = load_runtime_config(moved, 'main')
                self.assertEqual(cfg['_v6_sampler_path'], str(moved / 'run-state/sampler-coordinator.json'))
                self.assertFalse(any(key in cfg for key in ('_worker_id', '_v6_replay_dir', '_training_deadline', '_episode_log_dir')))
                restored = HybridPPO.load(moved / 'models/main/model.zip', env=vec, device='cpu')
                for name in ('tactical_optimizer', 'revive_optimizer', 'critic_optimizer'):
                    assert_optimizer_equal(self, getattr(main, name), getattr(restored, name))
                with patch('warp.init'), patch('warp.get_cuda_devices', return_value=[]), patch('torch.cuda.is_available', return_value=False):
                    report = preflight(moved)
                self.assertTrue(report['inference_ready'], report['inference'])
                self.assertFalse(report['training_ready'])  # No native simulator is built in the fixture.
                row = report['inference']['main']
                self.assertEqual(row['algorithm'], 'HybridPPO')
                self.assertTrue(row['frozen_skill_matches_initial'])
                self.assertTrue(row['hybrid_contract_valid'])
                self.assertFalse(report['target_hardware_claimed'])
                self.assertEqual(report['gradient_updates'], 0)
                resumed = Path(directory) / 'next-round'
                resumed.mkdir()
                (resumed / 'budget.json').write_text('{"preexisting_campaign_budget":true}')
                # The fixture's fake baseline is not a production submission;
                # metadata resource hashing is tested by the standalone export.
                with patch('rmuc2027_research.rl.learning.experiment', side_effect=lambda config: {'config': config}):
                    next_config, migration = prepare_resume_run(moved, resumed)
                self.assertEqual((resumed / 'main/model.zip').read_bytes(), (moved / 'models/main/model.zip').read_bytes())
                self.assertEqual((resumed / 'main/initialized.zip').read_bytes(), (resumed / 'main/model.zip').read_bytes())
                self.assertEqual((resumed / 'main/resume-state.pt').read_bytes(), (moved / 'models/main/resume-state.pt').read_bytes())
                self.assertTrue(next_config['friendly'].startswith(str(resumed)))
                self.assertEqual(next_config['_v6_sampler_path'], str(resumed / 'sampler-coordinator.json'))
                self.assertFalse(migration['optimizer_reset'])
                self.assertEqual(migration['released_interrupted_leases'], 1)
                self.assertEqual(migration['coordinator_restore'], 'checkpoint_coordinator')
                next_coordinator = json.loads((resumed / 'sampler-coordinator.json').read_text())
                self.assertEqual(next_coordinator['leases'], {})
                self.assertEqual(next_coordinator['counts'][lease['scenario_id']]['valid_steps'], 5)
                self.assertEqual(json.loads((moved / 'run-state/sampler-coordinator.json').read_text())['counts'][lease['scenario_id']]['valid_steps'], 8)
                self.assertTrue((resumed / 'main/checkpoints/baseline/complete.json').is_file())
                self.assertTrue((resumed / 'budget.json').is_file())
        finally:
            vec.close()
            aux_vec.close()

    def test_incomplete_release_is_rejected_before_archiving(self):
        with tempfile.TemporaryDirectory() as directory:
            source, run, _ = v4_fixture.BundleTests().fixture(directory)
            with self.assertRaisesRegex(ValueError, 'run-results.json'):
                create_bundle(run, Path(directory) / 'bad.zip', source_root=source)
            self.assertFalse((Path(directory) / 'bad.zip').exists())


if __name__ == '__main__':
    unittest.main()
