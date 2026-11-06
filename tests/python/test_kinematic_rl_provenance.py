import unittest
from rmuc2027_research.rl.session import load_config,provenance


class KinematicProvenanceTests(unittest.TestCase):
    def test_experimental_backend_has_distinct_sources_and_dependencies(self):
        config=load_config({'sim_backend':'warp_kinematic','rl_seconds':.5,'learning_control_hz':2})
        record=provenance(config)['simulation']
        self.assertEqual(record['backend'],'warp_kinematic')
        self.assertTrue(record['experimental'])
        self.assertIn('arena.py',record['files'])
        self.assertIn('traversal.py',record['files'])
        self.assertNotIn('benchmark.py',record['files'])
        self.assertEqual(set(record['dependencies']),{'warp-lang'})
        self.assertNotEqual(record,provenance(load_config({'sim_backend':'rapier'}))['simulation'])
