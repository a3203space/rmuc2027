import unittest
from rmuc2027_research.rl.session import load_config,provenance
from rmuc2027_research.rl.warp_kinematic.protocol import FRAME_ENCODER
from rmuc2027_core import engine


class KinematicProvenanceTests(unittest.TestCase):
    def test_experimental_backend_has_distinct_sources_and_dependencies(self):
        config=load_config({'sim_backend':'warp_kinematic','rl_seconds':.5,'learning_control_hz':2})
        record=provenance(config)['simulation']
        self.assertEqual(record['backend'],'warp_kinematic')
        self.assertTrue(record['experimental'])
        self.assertIn('arena.py',record['files'])
        self.assertIn('traversal.py',record['files'])
        self.assertIn('pose.py',record['files'])
        self.assertIn('batching.py',record['files'])
        self.assertIn('vector.py',record['files'])
        self.assertNotIn('benchmark.py',record['files'])
        self.assertEqual(record['frame_encoder'],FRAME_ENCODER)
        self.assertEqual(len(record['transport_sha256']),64)
        self.assertEqual(record['policy_observation_cache'],engine._CACHE_POLICY_OBSERVATIONS)
        self.assertEqual(record['pipe_reader'],'posix-select' if engine._DIRECT_PIPE_READER else 'threaded')
        self.assertEqual(set(record['dependencies']),{'warp-lang','orjson'} if FRAME_ENCODER=='orjson-numpy' else {'warp-lang'})
        self.assertNotEqual(record,provenance(load_config({'sim_backend':'rapier'}))['simulation'])
