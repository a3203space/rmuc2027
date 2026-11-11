"""Toy v6 package relocation/resume; no native simulation or PPO campaign."""
import json
import copy
import hashlib
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
from rmuc2027_research.rl.v6_bundle import create_bundle, extract_bundle, load_runtime_config, preflight, prepare_resume_run, GENERATION_MEMBERS
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

    def test_baseline_restores_complete_real_three_optimizer_generation_without_current_state(self):
        from rmuc2027_research.rl.v6_continuation import _skill_hash
        sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
        main,vec=make_model();aux_vec=DummyVecEnv([AuxiliaryToy])
        try:
            with tempfile.TemporaryDirectory() as directory:
                source,run,_=v4_fixture.BundleTests().fixture(directory)
                for name in ('run-results.json','run-manifest.json','main/migration.json'):
                    (run/name).write_text('{}')
                coordinator=TrainingCoordinator(run/'sampler-coordinator.json')
                lease=coordinator.acquire(0);coordinator.progress(0,lease['id'],valid_steps=5,seconds=.5)
                (run/'capability-profiles').mkdir()
                for side in ('red','blue'):(run/f'capability-profiles/evaluation-{side}.json').write_text('{}')
                for role in ('infantry','heavy'):
                    auxiliary=PPO(AuxLayerNormPolicy,aux_vec,n_steps=4,batch_size=4,n_epochs=1,device='cpu')
                    save_aux_state(auxiliary,aux_vec,run/role)
                    (run/role/'actor.json').write_text(json.dumps(export_aux(auxiliary.policy,family=role)))
                # Frozen resource provenance is independent of machine-local paths.
                submission=source/'research/strategies/examples/base'
                submission_manifest={'entries':{'team':'strategy.py'}}
                (submission/'manifest.json').write_text(json.dumps(submission_manifest))
                metadata=json.loads((run/'main/experiment.json').read_text())
                metadata['config'].update(policy_version=6,resume_contract_probe='baseline',
                    _v6_sampler_path=str(run/'sampler-coordinator.json'))
                metadata['auxiliary_pool']={role:dict(path=str(run/role/'actor.json'),sha256=sha(run/role/'actor.json')) for role in ('infantry','heavy')}
                metadata['radar_classifier_sha256']=sha(run/'radar-calibration/classifier.json')
                metadata['sources']={key:dict(manifest=submission_manifest,hashes={'team':sha(submission/'strategy.py')}) for key in ('friendly','opponent')}
                (run/'main/experiment.json').write_text(json.dumps(metadata))
                vec.seed(31);vec.reset();fill_update(main)
                save_state(main,vec,run/'main')
                (run/'main/actor.json').write_text(json.dumps(export_actor(main.policy)))
                (run/'main/coordinator-state.json').write_text(json.dumps(coordinator.snapshot()))
                ancestor=run/'main/checkpoints/baseline';ancestor.mkdir(parents=True)
                for name in GENERATION_MEMBERS:shutil.copy2(run/'main'/name,ancestor/name)
                complete=dict(directory='main/checkpoints/baseline',samples=main.num_timesteps,
                    update_id=main.completed_updates,optimizer_steps=main.optimizer_steps,
                    skill_sha256=_skill_hash(main.policy),sha256={name:sha(ancestor/name) for name in GENERATION_MEMBERS})
                (ancestor/'complete.json').write_text(json.dumps(complete))
                shutil.copy2(ancestor/'model.zip',run/'main/initialized.zip')
                baseline=HybridPPO.load(ancestor/'model.zip',device='cpu')
                # A genuine subsequent CPU optimizer update, not a weights-only
                # file swap, creates the deliberately different descendant.
                vec.seed(53);vec.reset();fill_update(main,target=.7);save_state(main,vec,run/'main')
                (run/'main/actor.json').write_text(json.dumps(export_actor(main.policy)))
                coordinator.progress(0,lease['id'],valid_steps=9,seconds=.9)
                (run/'main/coordinator-state.json').write_text(json.dumps(coordinator.snapshot()))
                metadata['config']['resume_contract_probe']='current'
                (run/'main/experiment.json').write_text(json.dumps(metadata))
                archive=Path(directory)/'bundle.zip';create_bundle(run,archive,source_root=source)
                moved=Path(directory)/'target';extract_bundle(archive,moved)
                restored_dir=Path(directory)/'restored-baseline'
                with patch('rmuc2027_research.rl.learning.experiment',side_effect=lambda config:{'config':config}):
                    config,report=prepare_resume_run(moved,restored_dir,generation='baseline')
                restored=HybridPPO.load(restored_dir/'main/model.zip',device='cpu')
                self.assertEqual(restored.num_timesteps,4);self.assertEqual(main.num_timesteps,8)
                self.assertEqual(config['resume_contract_probe'],'baseline')
                self.assertEqual(report['source_generation'],'baseline')
                self.assertEqual(report['new_round_baseline'],'restored_baseline_model')
                self.assertEqual(report['coordinator_restore'],'baseline_generation_coordinator')
                for name in ('tactical_optimizer','revive_optimizer','critic_optimizer'):
                    self.assertGreater(len(getattr(baseline,name).state),0)
                    assert_optimizer_equal(self,getattr(baseline,name),getattr(restored,name))
                for name in ('model.zip','resume-state.pt','samplers.json','actor.json'):
                    self.assertEqual((restored_dir/'main'/name).read_bytes(),(ancestor/name).read_bytes())
                    self.assertNotEqual((restored_dir/'main'/name).read_bytes(),(run/'main'/name).read_bytes())
                restored_coordinator=json.loads((restored_dir/'sampler-coordinator.json').read_text())
                self.assertEqual(restored_coordinator['leases'],{})
                self.assertEqual(restored_coordinator['counts'][lease['scenario_id']]['valid_steps'],5)
                self.assertEqual(report['source_generation_members_sha256'],complete['sha256'])
                self.assertEqual(set(report['baseline_resource_validation']),{'friendly','opponent','radar_classifier_path','auxiliary_pool:infantry','auxiliary_pool:heavy'})
                ready=json.loads((restored_dir/'main/checkpoints/baseline/complete.json').read_text())
                self.assertEqual(ready['samples'],4);self.assertEqual(ready['optimizer_steps'],baseline.optimizer_steps)
                self.assertEqual(ready['skill_sha256'],_skill_hash(baseline.policy))

                # Re-sign only outer test manifests to exercise generation-level
                # rejection after normal bundle integrity validation succeeds.
                def rewrite_bundle_member(relative,value):
                    path=moved/relative;path.write_text(json.dumps(value))
                    manifest=json.loads((moved/'bundle-manifest.json').read_text())
                    manifest['files'][relative]=dict(bytes=path.stat().st_size,sha256=sha(path))
                    (moved/'bundle-manifest.json').write_text(json.dumps(manifest))
                publication='models/main/checkpoints/baseline/complete.json'
                wrong=copy.deepcopy(complete);wrong['sha256']['samplers.json']='0'*64
                rewrite_bundle_member(publication,wrong)
                with self.assertRaisesRegex(ValueError,'publication hash mismatch: samplers'):
                    prepare_resume_run(moved,Path(directory)/'bad-hash',generation='baseline')
                self.assertFalse((Path(directory)/'bad-hash').exists())
                rewrite_bundle_member(publication,complete)
                bad_metadata=json.loads((ancestor/'experiment.json').read_text())
                bad_metadata['auxiliary_pool']['infantry']['sha256']='1'*64
                relative='models/main/checkpoints/baseline/experiment.json'
                rewrite_bundle_member(relative,bad_metadata)
                wrong=copy.deepcopy(complete);wrong['sha256']['experiment.json']=sha(moved/relative)
                rewrite_bundle_member(publication,wrong)
                with self.assertRaisesRegex(ValueError,'frozen auxiliary binding mismatch: infantry'):
                    prepare_resume_run(moved,Path(directory)/'mixed-helper',generation='baseline')
                self.assertFalse((Path(directory)/'mixed-helper').exists())
                with self.assertRaisesRegex(ValueError,'must be current or baseline'):
                    prepare_resume_run(moved,Path(directory)/'alias',generation='selected')
        finally:
            vec.close();aux_vec.close()


if __name__ == '__main__':
    unittest.main()
