"""Generated target instructions only; fake weights, no model/physics/network."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
import zipfile

from rmuc2027_research.rl.v4_bundle import create_bundle


ROOT = Path(__file__).resolve().parents[2]


class ResumeInstructionTests(unittest.TestCase):
    def instructions(self, temporary):
        root = Path(temporary) / 'source'
        run = Path(temporary) / 'run'
        root.mkdir()
        (root / 'Cargo.toml').write_text('[workspace]\nmembers=[]\n')
        for stage in ('main', 'infantry', 'heavy'):
            folder = run / stage
            folder.mkdir(parents=True)
            (folder / 'model.zip').write_bytes(b'deliberately not a neural checkpoint')
            (folder / 'experiment.json').write_text(json.dumps({
                'config': {'policy_version': 6 if stage == 'main' else 5}}))
        (run / 'sampler-coordinator.json').write_text('{}')
        archive = Path(temporary) / 'bundle.zip'
        create_bundle(run, archive, source_root=root)
        with zipfile.ZipFile(archive) as zipped:
            self.assertEqual(zipped.read('models/main/model.zip'), b'deliberately not a neural checkpoint')
            return zipped.read('RESUME.txt').decode()

    def test_generated_filter_preserves_lock_and_uses_bundled_local_packages(self):
        with tempfile.TemporaryDirectory() as temporary:
            instructions = self.instructions(temporary)
            body = instructions.split(".venv/bin/python - <<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
            target = Path(temporary) / 'target'
            (target / 'run-state').mkdir(parents=True)
            (target / '.venv').mkdir()
            lock = target / 'run-state/requirements-runtime.txt'
            original = ('# original freeze\nrmuc2027-core==0.1.0\nRMUC2027_Research==0.1.0\n'
                        'rmuc2027.server==0.1.0\nrmuc2027==0.1.0\n'
                        'torch==2.15.0+cpu\nsetuptools==84.2.1\n')
            lock.write_text(original)
            before = hashlib.sha256(lock.read_bytes()).hexdigest()
            result = subprocess.run([sys.executable, '-c', body], cwd=target, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(hashlib.sha256(lock.read_bytes()).hexdigest(), before)
            self.assertEqual((target / '.venv/requirements-third-party.txt').read_text(),
                             '# original freeze\ntorch==2.15.0+cpu\nsetuptools==84.2.1\n')
            self.assertIn('--no-index --no-deps --no-build-isolation', instructions)
            self.assertIn('-e ./source/packages/core -e ./source/research -e ./source/apps/server', instructions)
            self.assertNotIn('-r run-state/requirements-runtime.txt', instructions)
            docs = (ROOT / 'docs/research/sentry-v6-web-and-backup.md').read_text()
            self.assertEqual(body, docs.split(".venv/bin/python - <<'PY'\n", 1)[1].split('\nPY\n', 1)[0])

    def test_documented_shell_is_valid_and_frozen_web_uses_registered_run_path(self):
        docs = (ROOT / 'docs/research/sentry-v6-web-and-backup.md').read_text()
        for block in re.findall(r'```bash\n(.*?)\n```', docs, re.DOTALL):
            result = subprocess.run(['bash', '-n'], input=block, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)
        with tempfile.TemporaryDirectory() as temporary:
            instructions = self.instructions(temporary)
            self.assertIn('new_run="$PWD/source/data/research/new-v6-round"', instructions)
            self.assertIn('source/scripts/v6_web_watch.py --run "$new_run"', instructions)
            self.assertIn('--start-service --port 8768', instructions)
            self.assertIn('After budget.json exists', instructions)
            self.assertIn('NEW bounded two-hour experiment', instructions)
            self.assertIn('not automatically the selected deployment actor', instructions)
            self.assertIn('source/build/arena-web', instructions)
            self.assertIn('external SQLite is unnecessary for v6 continuation', instructions)
            self.assertNotIn('-m rmuc2027_research.rl.v6_longrun', instructions)


if __name__ == '__main__':
    unittest.main()
