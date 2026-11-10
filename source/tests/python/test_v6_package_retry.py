"""Packaging retries preserve an existing export and verify its contents."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from rmuc2027_research.rl.v6_campaign import package


class PackageRetryTests(unittest.TestCase):
    def test_retry_rechecks_export_and_rejects_different_existing_files(self):
        def export(model,destination):
            destination=Path(destination);destination.mkdir()
            (destination/'robot_7.py').write_text('verified actor')
        with TemporaryDirectory() as directory:
            root=Path(directory);(root/'main').mkdir()
            (root/'main/model.zip').write_bytes(b'checkpoint')
            (root/'selection.json').write_text(json.dumps({'selected':'model.zip'}))
            with patch('rmuc2027_research.rl.exporting.export_model',export),patch(
                    'rmuc2027_research.rl.v6_bundle.create_bundle',
                    side_effect=RuntimeError('after verified export')) as create:
                for _ in range(2):
                    with self.assertRaisesRegex(RuntimeError,'after verified export'):
                        package(root,time.time()+60)
                self.assertEqual(create.call_count,2)
                existing=root/'selected-export/robot_7.py'
                existing.write_text('different existing contents')
                with self.assertRaisesRegex(ValueError,'differs'):
                    package(root,time.time()+60)
                self.assertEqual(create.call_count,2)
                self.assertEqual(existing.read_text(),'different existing contents')
                self.assertFalse(list(root.glob('.selected-export-*')))


if __name__=='__main__':unittest.main()
