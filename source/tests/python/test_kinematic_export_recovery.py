"""Untrained PPO fixture: backup/export integration, never a training result."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile
import numpy as np
from stable_baselines3 import PPO
from rmuc2027_core.replay import verify_replay
from rmuc2027_research.rl.backup import bundle,_upload_git_archive
from rmuc2027_research.rl.env import SentryEnv
from rmuc2027_research.rl.exporting import export_model
from rmuc2027_research.rl.learning import experiment
from rmuc2027_research.rl.session import load_config,resource
from rmuc2027_research.strategies.battle import Battle


class ExportRecoveryTests(unittest.TestCase):
    def test_untrained_checkpoint_restores_exports_and_replays_in_rapier(self):
        config=load_config('configs/rl/sentry-mjwarp-hour.json')
        config['sim_backend']='warp_kinematic'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'run';run.mkdir()
            env=SentryEnv(config)
            try:
                model=PPO('MlpPolicy',env,n_steps=8,batch_size=8,device='cpu',seed=42)
                self.assertEqual(model._n_updates,0)
                observation=np.zeros(env.observation_space.shape,dtype=np.float32)
                expected=model.predict(observation,deterministic=True)[0]
                model.save(run/'model.zip')
            finally:env.close()
            (run/'experiment.json').write_text(json.dumps(experiment(config)))
            archive=bundle(run,run/'model.zip',{'kind':'untrained-test-fixture','gradient_updates':0})
            remote=root/'remote.git';subprocess.run(['git','init','--bare','--quiet',str(remote)],check=True)
            receipt=_upload_git_archive(archive,str(remote),'untrained-fixture')
            restored_bytes=subprocess.check_output(['git','--git-dir',str(remote),'show',receipt['ref']+':latest.zip'])
            self.assertEqual(hashlib.sha256(restored_bytes).hexdigest(),receipt['sha256'])
            restored=root/'restored';restored.mkdir();restored_archive=root/'restored.zip';restored_archive.write_bytes(restored_bytes)
            with zipfile.ZipFile(restored_archive) as source:
                checks=json.loads(source.read('checksums.json'))
                for name in ('model.zip','experiment.json','progress.json'):
                    payload=source.read(name);self.assertEqual(hashlib.sha256(payload).hexdigest(),checks[name])
                    (restored/name).write_bytes(payload)
            recovered=PPO.load(restored/'model.zip',device='cpu')
            np.testing.assert_array_equal(recovered.predict(observation,deterministic=True)[0],expected)
            exported=root/'exported';metadata=export_model(restored/'model.zip',exported)
            self.assertLess(metadata['actor_parity_max_error'],1e-5)
            with Battle(exported,resource(config['opponent']),duration=1,sandbox='trusted-local',timeout=5) as battle:
                result=battle.run();self.assertEqual(result['policy_errors'],0)
                self.assertEqual(battle.metadata['physics_backend'],'rapier')
                replay=root/'match.json';battle.save(replay)
            self.assertEqual(verify_replay(replay)['digest'],result['digest'])
