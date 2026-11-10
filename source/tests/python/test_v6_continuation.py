"""CPU toy continuation checks: no native simulation or research training."""
import hashlib
import json
from pathlib import Path
import random
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rmuc2027_research.rl.sentry_v6_policy import export_actor
from rmuc2027_research.rl.v6_continuation import MEMBERS, _skill_hash, materialize_generation
from rmuc2027_research.rl.v6_ppo import HybridPPO
from rmuc2027_research.rl.v6_sampling import TrainingCoordinator
from rmuc2027_research.rl.v6_training import save_state, restore_state
from test_sentry_v6_training import make_model, fill_update, assert_optimizer_equal


def write(path, data):
    Path(path).write_text(json.dumps(data))


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def fixture(self, directory):
        root = Path(directory) / 'ancestor'
        main = root / 'main'
        main.mkdir(parents=True)
        for role in ('infantry', 'heavy'):
            (root / role).mkdir()
            for name in ('model.zip', 'resume-state.pt', 'samplers.json', 'actor.json'):
                (root / role / name).write_bytes(('frozen-' + role + '-' + name).encode())
            write(root / role / 'experiment.json', {'config': {'policy_version': 5}})
        (root / 'capability-profiles').mkdir()
        for side in ('red', 'blue'):
            write(root / f'capability-profiles/evaluation-{side}.json', {'side': side})
        (root / 'radar-calibration').mkdir()
        write(root / 'radar-calibration/classifier.json', {'validated': False})
        config = dict(policy_version=6, auxiliary_pool={role: str(root / role / 'actor.json')
            for role in ('infantry', 'heavy')}, radar_classifier_path=str(root / 'radar-calibration/classifier.json'),
            _v6_sampler_path=str(root / 'sampler-coordinator.json'), _training_deadline=42, _worker_id=7)
        write(main / 'experiment.json', {'config': config})
        coordinator = TrainingCoordinator(root / 'sampler-coordinator.json')
        lease = coordinator.acquire(0)
        coordinator.progress(0, lease['id'], valid_steps=5, seconds=.5)
        write(main / 'coordinator-state.json', coordinator.snapshot())
        model, vec = make_model()
        fill_update(model)
        save_state(model, vec, main)
        write(main / 'actor.json', export_actor(model.policy))
        shutil.copy2(main / 'model.zip', main / 'initialized.zip')
        generation = main / 'checkpoints/baseline'
        generation.mkdir(parents=True)
        for name in MEMBERS:
            shutil.copy2(main / name, generation / name)
        ready = dict(directory='main/checkpoints/baseline', samples=model.num_timesteps,
            update_id=model.completed_updates, optimizer_steps=model.optimizer_steps,
            skill_sha256=_skill_hash(model.policy), sha256={name: hashlib.sha256(
                (generation / name).read_bytes()).hexdigest() for name in MEMBERS})
        write(generation / 'complete.json', ready)
        coordinator.progress(0, lease['id'], valid_steps=8, seconds=.8)
        fill_update(model, target=4.)
        save_state(model, vec, main)
        return root, generation, model, vec, lease

    def test_ancestor_generation_restores_three_adams_rng_and_its_coverage_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root, generation, candidate, vec, lease = self.fixture(directory)
            try:
                baseline = HybridPPO.load(generation / 'model.zip', device='cpu')
                out = Path(directory) / 'continuation'
                out.mkdir()
                (out / 'authorization.json').write_text('{"preserved":true}')
                with patch('rmuc2027_research.rl.learning.experiment', side_effect=lambda cfg: {'config': cfg}):
                    cfg, report = materialize_generation(root, 'main/checkpoints/baseline', out,
                        rejected_candidate='main/model.zip', reason='Complete paired capture regression')
                restored = HybridPPO.load(out / 'main/model.zip', device='cpu')
                self.assertLess(restored.num_timesteps, candidate.num_timesteps)
                self.assertEqual(restored.num_timesteps, baseline.num_timesteps)
                for name in ('tactical_optimizer', 'revive_optimizer', 'critic_optimizer'):
                    assert_optimizer_equal(self, getattr(baseline, name), getattr(restored, name))
                for name in ('model.zip', 'resume-state.pt', 'samplers.json', 'actor.json'):
                    self.assertEqual((out / 'main' / name).read_bytes(), (generation / name).read_bytes())
                self.assertEqual((out / 'main/initialized.zip').read_bytes(), (generation / 'model.zip').read_bytes())
                restore_state(generation)
                expected = (random.random(), np.random.random(), torch.rand(4))
                restore_state(out / 'main')
                actual = (random.random(), np.random.random(), torch.rand(4))
                self.assertEqual(expected[:2], actual[:2])
                torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)
                coverage = json.loads((out / 'sampler-coordinator.json').read_text())
                self.assertFalse(coverage['leases'])
                self.assertEqual(coverage['counts'][lease['scenario_id']]['valid_steps'], 5)
                self.assertEqual(json.loads((root / 'sampler-coordinator.json').read_text())['counts'][lease['scenario_id']]['valid_steps'], 8)
                self.assertEqual(cfg['_v6_sampler_path'], str(out / 'sampler-coordinator.json'))
                self.assertFalse('_training_deadline' in cfg or '_worker_id' in cfg)
                self.assertEqual(cfg['auxiliary_pool']['infantry'], str(out / 'infantry/actor.json'))
                self.assertTrue(report['rng_file_copied_exact'])
                self.assertFalse(report['optimizer_reset'])
                self.assertFalse(report['descendant_coverage_imported'])
                self.assertEqual(report['rejected_candidate']['sha256'], hashlib.sha256((root / 'main/model.zip').read_bytes()).hexdigest())
                ready = json.loads((out / 'main/checkpoint-ready.json').read_text())
                self.assertEqual(ready, json.loads((out / ready['directory'] / 'complete.json').read_text()))
                self.assertEqual(ready['optimizer_steps'], baseline.optimizer_steps)
                self.assertEqual(ready['skill_sha256'], _skill_hash(baseline.policy))
                for name, digest in ready['sha256'].items():
                    self.assertEqual(digest, hashlib.sha256((out / ready['directory'] / name).read_bytes()).hexdigest())
                self.assertTrue((out / 'authorization.json').exists())
                with self.assertRaises(FileExistsError):
                    materialize_generation(root, generation, out)
            finally:
                vec.close()

    def test_hash_or_cross_generation_resume_state_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root, generation, _, vec, _ = self.fixture(directory)
            try:
                out = Path(directory) / 'bad'
                shutil.copy2(root / 'main/resume-state.pt', generation / 'resume-state.pt')
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    materialize_generation(root, generation, out)
                self.assertFalse((out / 'main').exists())
                complete = json.loads((generation / 'complete.json').read_text())
                complete['sha256']['resume-state.pt'] = hashlib.sha256((generation / 'resume-state.pt').read_bytes()).hexdigest()
                write(generation / 'complete.json', complete)
                with self.assertRaisesRegex(ValueError, 'counters disagree'):
                    materialize_generation(root, generation, out)
                self.assertFalse((out / 'main').exists())
            finally:
                vec.close()


if __name__ == '__main__':
    unittest.main()
