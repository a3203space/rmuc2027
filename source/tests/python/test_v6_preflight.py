"""Preflight scheduling/measurement tests using a fake clock and vector env."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from rmuc2027_research.rl import v6_preflight as probe


class Clock:
    now = 0.


class FakeVec:
    def __init__(self, count, clock):
        self.count, self.clock, self.steps = count, clock, 0

    def reset(self):
        self.clock.now += .1
        return dict(actor=np.zeros((self.count, 775)), critic=np.zeros((self.count, 454)),
                    revive_mask=np.tile([1., 0., 0.], (self.count, 1)))

    def step(self, actions):
        self.clock.now += .08
        self.steps += 1
        obs = dict(actor=np.zeros((self.count, 775)), critic=np.zeros((self.count, 454)),
                   revive_mask=np.tile([1., 0., 0.], (self.count, 1)))
        return obs, np.zeros(self.count), np.zeros(self.count, bool), [
            dict(sampled_simulation_seconds=.1, metrics=dict(policy_errors=0)) for _ in range(self.count)]


class FakeModel:
    def __init__(self, count, clock):
        self.count, self.clock = count, clock

    def predict(self, obs, deterministic):
        if deterministic:
            raise AssertionError('Benchmark must include real sampling overhead')
        self.clock.now += .02
        return np.zeros((self.count, 7)), None


class PreflightTests(unittest.TestCase):
    def test_target_probe_is_explicit_and_does_not_expand_campaign_default(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(probe, '_run_preflight', return_value={'toy': True}) as run:
            root = Path(directory)
            probe.preflight(root, 123.)
            self.assertEqual(run.call_args.args, (root, 123., (1, 2), root))
            probe.target_preflight(root, 456.)
            self.assertEqual(run.call_args.args, (root, 456., (1, 2, 4, 8), root / 'target-preflight'))
            with self.assertRaises(ValueError):
                probe.target_preflight(root, 456., counts=(1, 3))
            (root / 'target-preflight').mkdir()
            with self.assertRaises(FileExistsError):
                probe.target_preflight(root, 456.)

    def test_outer_interrupt_stops_only_owned_worker_group(self):
        with tempfile.TemporaryDirectory() as directory:
            process = SimpleNamespace(pid=7654321, returncode=None)
            calls = []
            def wait(timeout):
                calls.append(timeout)
                if len(calls) == 1:
                    raise KeyboardInterrupt()
                process.returncode = -9
            process.wait = wait
            process.poll = lambda: process.returncode
            with patch.object(probe, 'freeze_environment'), \
                 patch.object(probe.subprocess, 'Popen', return_value=process), \
                 patch('rmuc2027_core.platform_support.kill_process_tree') as stop:
                with self.assertRaises(KeyboardInterrupt):
                    probe.preflight(directory, probe.time.time() + 100)
                stop.assert_called_once_with(process)
            self.assertEqual(process.returncode, -9)

    def test_two_windows_measure_actual_sum_seconds_and_single_world_rate(self):
        clock = Clock()
        vec = FakeVec(2, clock)
        with patch.object(probe.time, 'monotonic', side_effect=lambda: clock.now):
            result = probe.measure_windows(vec, FakeModel(2, clock), 2, 100.)
        self.assertTrue(result['completed'])
        self.assertEqual(vec.steps, 61)
        self.assertEqual(result['measured_rl_steps'], 60)
        self.assertEqual(result['warmup_steps'], 1)
        self.assertEqual(len(result['windows']), 2)
        for window in result['windows']:
            self.assertAlmostEqual(window['sum_simulation_seconds'], 6.)
            self.assertAlmostEqual(window['r'], 2.)
            self.assertAlmostEqual(window['single_world_rate'], 1.)
        self.assertEqual(result['gradient_updates'], 0)

    def test_partial_deadline_cannot_be_reported_complete(self):
        clock = Clock()
        with patch.object(probe.time, 'monotonic', side_effect=lambda: clock.now):
            result = probe.measure_windows(FakeVec(1, clock), FakeModel(1, clock), 1, 2.5)
        self.assertFalse(result['completed'])
        self.assertTrue(result['censored'])
        self.assertLess(result['windows'][0]['rl_steps'], 30)
        self.assertFalse(result['windows'][0]['completed'])

    def test_readiness_requires_both_complete_single_world_backends(self):
        def row(backend, count, rate):
            return dict(backend=backend, num_envs=count, completed=True,
                conservative_r=rate, single_world_rate=rate / count,
                devices=dict(warp_cuda_available=backend == 'warp_kinematic'),
                windows=[dict(rl_steps=30, completed=True, r=rate) for _ in range(2)])
        warp = row('warp_kinematic', 1, 2.)
        rapier = row('rapier', 1, 1.)
        two = row('warp_kinematic', 2, 2.05)
        self.assertFalse(probe.summarize([warp, two])['training_ready'])
        result = probe.summarize([warp, rapier, two])
        self.assertTrue(result['training_ready'])
        self.assertEqual(result['selected_num_envs'], 1)
        self.assertEqual(result['evaluation_rates'], {'warp_kinematic': 2., 'rapier': 1.})
        two.update(conservative_r=3., single_world_rate=1.5)
        self.assertEqual(probe.summarize([warp, rapier, two])['selected_num_envs'], 2)
        rapier['windows'][1]['completed'] = False
        self.assertFalse(probe.summarize([warp, rapier, two])['training_ready'])

    def test_workload_disables_training_lease_and_prefix_without_touching_live_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'main').mkdir()
            (root / 'capability-profiles').mkdir()
            original = dict(policy_version=6, auxiliary_pool=dict(infantry='one.json', heavy='two.json'),
                _v6_sampler_path='do-not-lease.json', _v6_replay_dir='do-not-auto-record',
                _episode_log_dir='do-not-log', recovery_kind='paid_early', curriculum_seconds=10)
            (root / 'main/experiment.json').write_text(json.dumps(dict(config=original)))
            for side in ('red', 'blue'):
                (root / f'capability-profiles/evaluation-{side}.json').write_text(json.dumps({'side_fixture': side}))
            configs = probe.benchmark_configurations(root, 'warp_kinematic', 2, 50.)
            self.assertEqual([cfg['team'] for cfg in configs], ['red', 'blue'])
            for cfg in configs:
                for forbidden in ('_v6_sampler_path', '_v6_replay_dir', '_episode_log_dir', 'recovery_kind', 'curriculum_seconds'):
                    self.assertNotIn(forbidden, cfg)
                self.assertFalse(cfg['scenario_sampling'])
                self.assertFalse(cfg['capability_sampling'])
                self.assertTrue(cfg['evaluation'])
                self.assertEqual(cfg['_training_deadline'], 50.)
            self.assertEqual(json.loads((root / 'main/experiment.json').read_text())['config'], original)
            eight = probe.benchmark_configurations(root, 'warp_kinematic', 8, 50.)
            self.assertEqual(len(eight), 8)
            self.assertEqual([cfg['team'] for cfg in eight], ['red', 'blue'] * 4)
            self.assertTrue(all('_v6_sampler_path' not in cfg for cfg in eight))
            with self.assertRaises(ValueError):
                probe.benchmark_configurations(root, 'rapier', 4, 50.)

    def test_package_does_not_require_external_database_for_resume_only(self):
        from rmuc2027_research.rl.v6_campaign import package
        class StopAfterBundleArguments(Exception):
            pass
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'run'
            (root / 'main').mkdir(parents=True)
            (root / 'selection.json').write_text('{"selected":"model.zip"}')
            (root / 'main/model.zip').write_bytes(b'fixture model, never loaded')
            resource_root = Path(directory) / 'source'
            resource_root.mkdir()
            for exists in (False, True):
                database = resource_root / 'rmuc2026shuj/rmuc_2026_region_dataset.sqlite'
                if exists:
                    database.parent.mkdir()
                    database.write_bytes(b'fixture external data')
                with patch('rmuc2027_core.paths.ROOT', resource_root), \
                     patch('rmuc2027_research.rl.exporting.export_model'), \
                     patch('rmuc2027_research.rl.v6_bundle.create_bundle', side_effect=StopAfterBundleArguments) as create:
                    with self.assertRaises(StopAfterBundleArguments):
                        package(root, 9999999999.)
                    self.assertEqual(create.call_args.kwargs['external_database'], database if exists else None)


if __name__ == '__main__':
    unittest.main()
