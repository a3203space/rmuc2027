import tempfile
import unittest
from pathlib import Path
from rmuc2027_research.rl.checkpoints import save_checkpoint
from rmuc2027_research.rl.session import load_config
from rmuc2027_core.models import DEFAULT_TEAM_LINEUP


class CheckpointTests(unittest.TestCase):
    def test_failed_save_preserves_complete_generation(self):
        class Model:
            def __init__(self, value, fail=False): self.value, self.fail = value, fail
            def save(self, path):
                Path(path).write_bytes(self.value)
                if self.fail: raise OSError('simulated disk error')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'latest.zip'
            save_checkpoint(Model(b'one'), path)
            with self.assertRaises(OSError): save_checkpoint(Model(b'partial', True), path)
            self.assertEqual(path.read_bytes(), b'one')
            save_checkpoint(Model(b'two'), path)
            self.assertEqual(path.read_bytes(), b'two')
            self.assertEqual(path.with_suffix('.previous.zip').read_bytes(), b'one')

    def test_independent_lineups_validated(self):
        red = dict(DEFAULT_TEAM_LINEUP)
        blue = dict(red, **{'3': 2732, '7': 2772})
        cfg = load_config({'lineups': {'red': red, 'blue': blue}})
        self.assertEqual(cfg['lineups']['blue']['7'], 2772)
        with self.assertRaises(ValueError): load_config({'lineups': {'red': red}})
        with self.assertRaises(ValueError):
            load_config({'lineups': {'red': red, 'blue': dict(blue, **{'7': 2732})}})
