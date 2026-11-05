import tempfile
import unittest
from pathlib import Path
from rmuc2027_research.rl.exporting import export_layers, compose_exports
from rmuc2027_research.rl.session import load_config, provenance
from rmuc2027_core.submissions import load_submission


class ComposeTests(unittest.TestCase):
    def test_roles_preserved_and_conflicts_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for slot in (3, 7):
                cfg = load_config({'slot':slot,'timed_assembly':True})
                export_layers([], provenance(cfg), root/str(slot))
            result = compose_exports([root/'3',root/'7'],root/'combined')
            self.assertEqual(result['learned_slots'],[3,7])
            manifest, entries = load_submission(root/'combined')
            self.assertIn('team_lineup',manifest)
            for slot in ('3','7'):
                _, original = load_submission(root/slot)
                self.assertEqual(entries[slot].read_bytes(),original[slot].read_bytes())
            with self.assertRaisesRegex(ValueError,'same slot'):
                compose_exports([root/'3',root/'3'],root/'duplicate')
            cfg=load_config({'slot':4,'timed_assembly':False})
            export_layers([],provenance(cfg),root/'4')
            with self.assertRaisesRegex(ValueError,'different lineups or battle settings'):
                compose_exports([root/'3',root/'4'],root/'conflict')
