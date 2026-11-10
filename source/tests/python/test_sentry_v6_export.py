"""CPU standalone hybrid export; loading the sandbox worker runs no physics."""
import hashlib
import copy
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rmuc2027_core import sdk
from rmuc2027_core.policy import Policy
from rmuc2027_core.submissions import load_submission
from rmuc2027_core.worker import SOURCE_LIMIT
from rmuc2027_research.rl.exporting import export_model, export_layers, _deployment_source_changes
from rmuc2027_research.rl.learning import experiment
from rmuc2027_research.rl.session import provenance
from rmuc2027_research.rl.sentry_v6_policy import export_actor
from rmuc2027_research.rl.v6_training import configuration
from test_sentry_v6_training import make_model, observations


class V6ExportTests(unittest.TestCase):
    def test_float32_exact_roundtrip_portable_parity_and_isolated_worker_import(self):
        torch.set_num_threads(1)
        model, vec = make_model()
        try:
            cfg = configuration()
            cfg.update(sim_backend='rapier', scenario_sampling=False, capability_sampling=False)
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model.save(root / 'model.zip')
                training = experiment(cfg)
                # The deployment bundle must retain the historical metadata,
                # even when only the reviewed episode/training modules differ.
                for name in ('v6_recovery.py', 'v6_sampling.py', 'v6_training.py'):
                    training['sentry_sources']['research/rmuc2027_research/rl/'+name] = '0'*64
                original = copy.deepcopy(training)
                (root / 'experiment.json').write_text(json.dumps(training))
                metadata = export_model(root / 'model.zip', root / 'export')
                self.assertEqual(metadata['sentry_sources'], original['sentry_sources'])
                self.assertEqual(json.loads((root / 'experiment.json').read_text()), original)
                changes = metadata['deployment_training_only_source_changes']
                self.assertEqual(len(changes), 3)
                for name, change in changes.items():
                    self.assertEqual(change['training_sha256'], '0'*64)
                    self.assertEqual(change['deployment_source_tree_sha256'], provenance(cfg)['sentry_sources'][name])
                    self.assertFalse(change['embedded_in_actor'])
                _, entries = load_submission(root / 'export')
                self.assertLessEqual(entries['7'].stat().st_size, SOURCE_LIMIT)
                namespace = dict(__name__='standalone_v6_cpu_test', __package__='')
                with patch.dict(sys.modules, {'sdk': sdk}):
                    exec(compile(entries['7'].read_text(), str(entries['7']), 'exec'), namespace)
                expected_actor = export_actor(model.policy)
                self.assertEqual(namespace['_LAYERS'], expected_actor)
                restored = [value for branch in ('tactical', 'skill', 'revive')
                    for layer in namespace['_LAYERS'][branch]
                    for value in [*[x for row in layer['weight'] for x in row], *layer['bias']]]
                self.assertEqual(hashlib.sha256(struct.pack('<%df' % len(restored), *restored)).hexdigest(),
                                 metadata['actor_float32_sha256'])
                obs = observations((False, True, False, True))
                with torch.no_grad():
                    expected = model.policy.get_distribution(obs).mode().numpy()
                actual = np.asarray([namespace['mlp_forward'](row.tolist(), namespace['_LAYERS']) for row in obs['actor']])
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
                policy = Policy(entries['7'], 19, 'trusted-local', 5.)
                try:
                    self.assertEqual(policy.errors, 0)
                    self.assertIsNone(policy.process.poll())
                finally:
                    policy.close()
                self.assertEqual(model.num_timesteps, 0)
                self.assertFalse(model.revive_optimizer.state)
        finally:
            vec.close()

    def test_exception_is_exact_paths_complete_hashes_and_v6_only(self):
        path = 'research/rmuc2027_research/rl/v6_recovery.py'
        old = {'sentry_sources': {path: '0'*64}}
        new = {'sentry_sources': {path: '1'*64}}
        original = copy.deepcopy(old)
        self.assertEqual(set(_deployment_source_changes(old, new, 6)), {path})
        self.assertEqual(old, original)
        for version in (3, 4):
            with self.assertRaisesRegex(ValueError, 'sentry_sources'):
                _deployment_source_changes(old, new, version)
        for modified in ({}, {path: None}, {path: 'bad'}, {'other/v6_recovery.py': '1'*64}):
            with self.subTest(modified=modified), self.assertRaisesRegex(ValueError, 'sentry_sources'):
                _deployment_source_changes(old, {'sentry_sources': modified}, 6)

    def test_actor_belief_policy_ppo_and_unknown_source_changes_still_refused(self):
        cfg = configuration()
        cfg.update(sim_backend='rapier', scenario_sampling=False, capability_sampling=False)
        metadata = experiment(cfg)
        current = provenance(cfg)
        for name in ('sentry_v6.py', 'recovery_belief.py', 'sentry_v6_policy.py', 'v6_ppo.py'):
            changed = copy.deepcopy(current)
            changed['sentry_sources']['research/rmuc2027_research/rl/'+name] = '0'*64
            with self.subTest(name=name), patch('rmuc2027_research.rl.exporting.provenance', return_value=changed):
                with self.assertRaisesRegex(ValueError, 'sentry_sources'):
                    export_layers({}, metadata, 'must-not-be-created')
        changed = copy.deepcopy(current)
        changed['sentry_sources']['unexpected.py'] = '0'*64
        with patch('rmuc2027_research.rl.exporting.provenance', return_value=changed):
            with self.assertRaisesRegex(ValueError, 'source set'):
                export_layers({}, metadata, 'must-not-be-created')

    def test_schema_sdk_runtime_rules_navigation_and_engine_remain_strict(self):
        cfg = configuration()
        cfg.update(sim_backend='rapier', scenario_sampling=False, capability_sampling=False)
        metadata = experiment(cfg)
        current = provenance(cfg)
        for field in ('sentry_schema', 'sdk_hash', 'runtime_hash', 'engine_binary_sha256',
                      'rules_sha256', 'navigation', 'sources'):
            changed = copy.deepcopy(current)
            changed[field] = '0'*64
            with self.subTest(field=field), patch('rmuc2027_research.rl.exporting.provenance', return_value=changed):
                with self.assertRaises(ValueError):
                    export_layers({}, metadata, 'must-not-be-created', allow_sdk_transfer=True)


if __name__ == '__main__':
    unittest.main()
