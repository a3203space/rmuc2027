import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location('v6_status', Path(__file__).resolve().parents[2] / 'scripts/v6_status.py')
status = importlib.util.module_from_spec(spec)
spec.loader.exec_module(status)


class StatusTests(unittest.TestCase):
    def test_coverage_ignores_stale_cache_and_separates_new_activity(self):
        baseline = {'counts': {'old': {'started': 1, 'valid_steps': 4200, 'full_completed': 1}}}
        current = {'counts': {'old': baseline['counts']['old'], 'new': {'started': 1, 'valid_steps': 10}},
                   'coverage': {'started': {'scenarios': 486}}}
        report = status.coverage(current, baseline)
        self.assertEqual(report['cumulative_unique']['started'], 2)
        self.assertEqual(report['run_unique_with_new_activity']['started'], 1)
        self.assertEqual(report['run_unique_with_new_activity']['full_completed'], 0)

    def test_missing_or_reset_coverage_is_not_zero_new_work(self):
        baseline = {'counts': {'old': {'started': 1, 'valid_steps': 100}}}
        self.assertIsNone(status.coverage({}, baseline)['cumulative_unique'])
        self.assertIsNone(status.coverage({'counts': {}}, baseline)['run_unique_with_new_activity'])

    def test_running_branch_is_not_accepted_and_reader_changes_no_files(self):
        with tempfile.TemporaryDirectory() as directory:
            master = Path(directory) / 'master'
            active = Path(directory) / 'master-r02'
            def write(path, row):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(row))
            write(master / 'campaign.json', {'status': 'training', 'active_run': str(active), 'rounds': [
                {'accepted_for_continuation': False, 'training': {'actual_samples': 128, 'published_samples': 128}}]})
            write(master / 'budget.json', {'deadline_unix': 2000})
            write(active / 'campaign.json', {'status': 'training'})
            base = {'samples': 100, 'update_id': 1, 'optimizer_steps': 3, 'skill_sha256': 'frozen'}
            ready = dict(base, directory='main/checkpoints/update-2', samples=164, update_id=2, optimizer_steps=6)
            write(active / 'main/checkpoints/baseline/complete.json', base)
            write(active / 'main/checkpoint-ready.json', ready)
            write(active / ready['directory'] / 'complete.json', ready)
            write(active / 'main/training.json', {'status': 'not_started', 'actual_samples': 0})
            before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in Path(directory).rglob('*') if p.is_file()}
            row = status.snapshot(master, now=1200)
            self.assertEqual(row['remaining_wall_seconds'], 800)
            self.assertEqual(row['checkpoint']['run_delta']['samples'], 64)
            self.assertTrue(row['checkpoint']['pointer_matches_complete'])
            self.assertEqual(row['completed_rounds']['accepted_published_samples'], 0)
            self.assertEqual(row['completed_rounds']['rejected_observed_samples'], 128)
            self.assertEqual(row['status'], 'training')
            self.assertEqual(row['issues'], [])
            after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in Path(directory).rglob('*') if p.is_file()}
            self.assertEqual(before, after)

    def test_incomplete_or_malformed_campaign_cannot_claim_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'campaign.json').write_text('{')
            row = status.snapshot(root, now=1200)
            self.assertEqual(row['status'], 'unknown')
            self.assertFalse(row['checkpoint']['pointer_matches_complete'])
            self.assertTrue(row['issues'])
            self.assertIsNone(row['checkpoint']['run_delta']['samples'])
            self.assertFalse(row['worker']['matching_live_process'])


if __name__ == '__main__':
    unittest.main()
