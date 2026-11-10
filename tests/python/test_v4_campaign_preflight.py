"""CPU policy Torch and CUDA Warp are distinct preflight resources; no training."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

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
        self.count = count; self.closed = False

    def reset(self):
        return {'actor': np.zeros((self.count, 3), dtype=np.float32)}

    def step(self, actions):
        return self.reset(), np.zeros(self.count), np.zeros(self.count, dtype=bool), [{} for _ in range(self.count)]

    def close(self):
        self.closed = True


class CampaignPreflightTests(unittest.TestCase):
    def run_fixture(self, root, device, *, unavailable=False):
        instances = []
        def vector_env(cfg, count):
            item = FakeVector(count); instances.append(item); return item
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
            preflight_worker(root, time.time()+120)
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
            self.assertEqual([r['num_envs'] for r in result['results']], [2,4,8])
            self.assertTrue(all(r['gpu_total_bytes']==device.total_memory and r['gpu_free_bytes']==device.free_memory
                                and r['gpu_memory_source']=='warp' for r in result['results']))
            self.assertTrue(all(v.closed for v in instances))

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
