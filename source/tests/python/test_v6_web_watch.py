import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('v6_web_watch', Path(__file__).resolve().parents[2]/'scripts/v6_web_watch.py')
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


class WebWatchTests(unittest.TestCase):
    def test_authorization_never_extends_actual_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'authorization.json').write_text(json.dumps({'deadline_unix':2000}))
            (root/'budget.json').write_text(json.dumps({'deadline_unix':1800}))
            self.assertEqual(watch.budget_deadline(root, 9999), 1800)
            self.assertEqual(watch.budget_deadline(root, 1700), 1700)
            with self.assertRaises(ValueError):watch.budget_deadline(root, float('nan'))

    def test_requires_both_phase_and_matching_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIsNone(watch.acceptance_deadline(root, 'web_acceptance', 2000, 1000))
            marker=root/'web-acceptance-request.json'
            marker.write_text(json.dumps({'run_id':root.name,'deadline_unix':1900}))
            self.assertIsNone(watch.acceptance_deadline(root, 'training', 2000, 1000))
            self.assertEqual(watch.acceptance_deadline(root, 'web_acceptance', 2000, 1000),1480)
            self.assertEqual(watch.acceptance_deadline(root, 'web_acceptance', 1800, 1700),1800)
            marker.write_text(json.dumps({'run_id':'other','deadline_unix':1900}))
            with self.assertRaises(ValueError):watch.acceptance_deadline(root, 'web_acceptance', 2000, 1000)

    def test_atomic_report_and_missing_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with self.assertRaises(ValueError):watch.budget_deadline(root)
            watch.atomic_json(root/'report.json', {'status':'等待'})
            self.assertEqual(watch.read_json(root/'report.json'), {'status':'等待'})
            self.assertFalse((root/'report.json.pending').exists())


if __name__ == '__main__':unittest.main()
