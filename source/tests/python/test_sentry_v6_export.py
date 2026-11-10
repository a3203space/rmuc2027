"""CPU standalone hybrid export; loading the sandbox worker runs no physics."""
import hashlib
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
from rmuc2027_research.rl.exporting import export_model
from rmuc2027_research.rl.learning import experiment
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
                (root / 'experiment.json').write_text(json.dumps(experiment(cfg)))
                metadata = export_model(root / 'model.zip', root / 'export')
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


if __name__ == '__main__':
    unittest.main()
