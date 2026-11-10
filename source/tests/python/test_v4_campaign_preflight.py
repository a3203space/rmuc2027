"""CPU policy Torch and CUDA Warp are distinct preflight resources; no training."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import hashlib

import numpy as np

from rmuc2027_research.rl.v4_campaign import preflight_worker


class FakeCuda:
    name = 'fixture CUDA physics device'
    arch = 120
    is_cuda = True
    total_memory = 8*1024**3
    free_memory = 5*1024**3

    def __str__(self):
        return 'cuda:0'


class FakeVector:
    def __init__(self, count):
        self.count = count; self.closed = False; self.actions=[]

    def reset(self):
        return {'actor': np.zeros((self.count, 3), dtype=np.float32)}

    def step(self, actions):
        self.actions.append(np.asarray(actions).copy())
        return self.reset(), np.zeros(self.count), np.zeros(self.count, dtype=bool), [{} for _ in range(self.count)]

    def close(self):
        self.closed = True


class CampaignPreflightTests(unittest.TestCase):
    def run_fixture(self, root, device, *, unavailable=False,resume_from=None):
        instances = []
        def vector_env(cfg, count):
            item = FakeVector(count); item.config=cfg; instances.append(item); return item
        with ExitStack() as stack:
            stack.enter_context(patch('torch.cuda.is_available', return_value=False))
            stack.enter_context(patch('torch.cuda.mem_get_info', side_effect=AssertionError('CPU Torch must not measure Warp allocation')))
            stack.enter_context(patch('warp.init'))
            stack.enter_context(patch('warp.get_cuda_devices', return_value=[] if unavailable else [device]))
            stack.enter_context(patch('warp.get_device', side_effect=RuntimeError('no CUDA') if unavailable else None, return_value=device))
            stack.enter_context(patch('rmuc2027_research.rl.v4_training.configuration', return_value={'sim_device':'cuda:0'}))
            stack.enter_context(patch('rmuc2027_research.rl.learning.vector_env', side_effect=vector_env))
            stack.enter_context(patch('rmuc2027_research.rl.portable.policy_schema', return_value={'action_size':3}))
            stack.enter_context(patch('importlib.metadata.version', return_value='fixture-version'))
            stack.enter_context(redirect_stdout(io.StringIO()))
            preflight_worker(root, time.time()+120,resume_from=resume_from)
        return instances

    def test_cpu_torch_accepts_warp_cuda_and_reports_real_physics_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            device = FakeCuda()
            instances = self.run_fixture(Path(directory), device)
            result = json.loads((Path(directory)/'preflight.json').read_text())
            self.assertTrue(result['training_ready'])
            self.assertFalse(result['torch_cuda_available'])
            self.assertEqual(result['torch_cuda_devices'], [])
            self.assertTrue(result['warp_cuda_available'])
            self.assertEqual(result['policy_device'], 'cpu')
            self.assertEqual(result['physics_device'], 'cuda:0')
            self.assertEqual(result['gpu_name'], device.name)
            self.assertEqual(result['warp_cuda_devices'][0]['architecture'], 120)
            self.assertEqual(result['gradient_updates'], 0)
            self.assertEqual(result['benchmark_kind'],'scripted_environment_only')
            self.assertFalse(result['benchmark_includes_ppo_updates'])
            self.assertEqual([r['num_envs'] for r in result['results']], [2,4,8])
            self.assertTrue(all(r['gpu_total_bytes']==device.total_memory and r['gpu_free_bytes']==device.free_memory
                                and r['gpu_memory_source']=='warp' for r in result['results']))
            self.assertTrue(all(v.closed for v in instances))

    def test_resume_benchmarks_loaded_actor_and_real_helper_paths_without_gradients(self):
        import torch
        calls=[]
        class FrozenModel:
            def predict(self,obs,deterministic):
                self_outer.assertFalse(torch.is_grad_enabled())
                self_outer.assertFalse(deterministic)
                calls.append(len(obs['actor']))
                return np.full((len(obs['actor']),3),.25,np.float32),None
            def learn(self,*args,**kwargs):raise AssertionError('preflight cannot train')
        self_outer=self
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'output';root.mkdir();bundle=Path(directory)/'bundle'
            checkpoint=bundle/'models/main/model.zip';checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'checkpoint')
            helpers={}
            for role in ('infantry','heavy'):
                path=bundle/'models'/role/'actor.json';path.parent.mkdir();path.write_bytes(role.encode());helpers[role]=str(path)
            cfg=dict(sim_device='cuda:0',auxiliary_pool=helpers,episode_seconds=120,capability_sampling=True)
            with patch('rmuc2027_research.rl.v4_bundle.verify_bundle',return_value={}) as verify, \
                 patch('rmuc2027_research.rl.v4_bundle.load_runtime_config',return_value=cfg) as load_config, \
                 patch('rmuc2027_research.rl.asymmetric_ppo.AsymmetricPPO.load',return_value=FrozenModel()) as load:
                instances=self.run_fixture(root,FakeCuda(),resume_from=bundle)
            verify.assert_called_once_with(bundle.resolve());load_config.assert_called_once_with(bundle.resolve(),'main')
            load.assert_called_once_with(checkpoint.resolve(),device='cpu')
            self.assertEqual(calls,[2]*9+[4]*9+[8]*9)
            for env in instances:
                self.assertEqual(env.config['auxiliary_pool'],helpers)
                self.assertEqual(env.config['episode_seconds'],120)
                self.assertTrue(all(np.all(a==.25) for a in env.actions))
            report=json.loads((root/'preflight.json').read_text())
            self.assertEqual(report['benchmark_kind'],'frozen_main_and_auxiliary_rollout')
            self.assertEqual(report['benchmark_measured_steps'],8)
            self.assertFalse(report['benchmark_includes_ppo_updates']);self.assertEqual(report['gradient_updates'],0)
            self.assertEqual(report['benchmark_sources']['main_checkpoint']['sha256'],hashlib.sha256(b'checkpoint').hexdigest())
            self.assertEqual(checkpoint.read_bytes(),b'checkpoint')
            for role,path in helpers.items():
                self.assertEqual(Path(path).read_bytes(),role.encode())
                self.assertEqual(report['benchmark_sources']['helpers'][role]['sha256'],hashlib.sha256(role.encode()).hexdigest())

    def test_resume_run_passes_bundle_to_preflight_before_copying_models(self):
        from rmuc2027_research.rl.v4_campaign import run
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);bundle=root/'bundle';bundle.mkdir()
            (bundle/'bundle-manifest.json').write_text('{}')
            (bundle/'preparation').mkdir();(bundle/'preparation/manifest.json').write_text('{}')
            with patch('rmuc2027_research.rl.v4_bundle.verify_bundle',return_value={}), \
                 patch('rmuc2027_research.rl.v4_campaign.child',side_effect=RuntimeError('stop at preflight')) as child:
                with self.assertRaisesRegex(RuntimeError,'stop at preflight'):
                    run(root/'output',seconds=120,num_envs=0,resume_from=bundle)
            args=child.call_args.args[1]
            self.assertEqual(args[0],'preflight')
            self.assertEqual(args[args.index('--resume-from')+1],str(bundle.resolve()))
            self.assertFalse((root/'output/main').exists())

    def test_no_warp_cuda_fails_explicitly_even_with_cpu_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, 'physics is unavailable'):
                self.run_fixture(Path(directory), FakeCuda(), unavailable=True)
            result = json.loads((Path(directory)/'preflight.json').read_text())
            self.assertFalse(result['training_ready'])
            self.assertEqual(result['results'], [])
            self.assertEqual(result['gradient_updates'], 0)

    def test_cpu_torch_does_not_bypass_warp_memory_headroom_check(self):
        with tempfile.TemporaryDirectory() as directory:
            device = FakeCuda(); device.free_memory = 512*1024**2
            with self.assertRaisesRegex(RuntimeError, 'No valid vector configuration'):
                self.run_fixture(Path(directory), device)
            result = json.loads((Path(directory)/'preflight.json').read_text())
            self.assertFalse(result['training_ready'])
            self.assertTrue(all(r['gpu_total_bytes']>0 for r in result['results']))


if __name__ == '__main__':
    unittest.main()
