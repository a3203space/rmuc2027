import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from rmuc2027_research.rl.backup import bundle, ReleaseBackup

class BackupTests(unittest.TestCase):
    def test_archive_failure_does_not_abort_training(self):
        with tempfile.TemporaryDirectory() as d, patch(
                'rmuc2027_research.rl.backup.bundle', side_effect=OSError('disk full')):
            worker = ReleaseBackup(Path(d), 'owner/repo')
            try:
                self.assertIsNone(worker.submit(Path(d)/'model.zip', {}))
                self.assertEqual(worker.jobs.unfinished_tasks, 0)
            finally:
                worker.close()

    def test_bundle_is_immutable_and_hashes_match(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'experiment.json').write_text('{}');(p/'latest.zip').write_bytes(b'original')
            archive=bundle(p,p/'latest.zip',{'steps':12})
            (p/'latest.zip').write_bytes(b'next')
            with zipfile.ZipFile(archive) as z:
                checks=json.loads(z.read('checksums.json'))
                for name,digest in checks.items():self.assertEqual(hashlib.sha256(z.read(name)).hexdigest(),digest)
                self.assertEqual(z.read('model.zip'),b'original')

    def test_success_is_recorded_only_after_upload(self):
        with tempfile.TemporaryDirectory() as d, patch('rmuc2027_research.rl.backup.upload') as send:
            p=Path(d);(p/'experiment.json').write_text('{}');(p/'model.zip').write_bytes(b'model')
            worker=ReleaseBackup(p,'owner/repo');archive=worker.submit(p/'model.zip',{})
            worker.close();send.assert_called_once()
            self.assertEqual(json.loads(archive.with_suffix('.status.json').read_text())['status'],'uploaded')
