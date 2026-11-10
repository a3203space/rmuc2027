"""Admission reports must distinguish success, subprocess failure and timeout."""
import subprocess
import unittest
from unittest.mock import patch
from rmuc2027_research.rl.mjwarp.doctor import run_check


class DoctorTests(unittest.TestCase):
    @patch('rmuc2027_research.rl.mjwarp.doctor.subprocess.run')
    def test_success_requires_marker_and_zero_exit(self, execute):
        execute.return_value = subprocess.CompletedProcess([], 0, 'Warp log\nDOCTOR_RESULT {"ok": true}\n', '')
        report = run_check('mjwarp', 10)
        self.assertTrue(report['passed'])
        self.assertTrue(report['result']['ok'])
        self.assertIn('rmuc2027_research.rl.mjwarp.doctor', execute.call_args.args[0])
        execute.return_value = subprocess.CompletedProcess([], 0, '', '')
        self.assertFalse(run_check('mjwarp', 10)['passed'])

    @patch('rmuc2027_research.rl.mjwarp.doctor.subprocess.run')
    def test_cuda_failure_is_not_success(self, execute):
        execute.return_value = subprocess.CompletedProcess([], 1, '', 'CUDA unavailable')
        report = run_check('torch', 10)
        self.assertFalse(report['passed'])
        self.assertEqual(report['stderr'], 'CUDA unavailable')

    @patch('rmuc2027_research.rl.mjwarp.doctor.subprocess.run')
    def test_timeout_preserves_diagnostics(self, execute):
        execute.side_effect = subprocess.TimeoutExpired([], 10, output=b'compiling', stderr=b'last log')
        report = run_check('mjwarp', 10)
        self.assertFalse(report['passed'])
        self.assertEqual(report['stdout'], 'compiling')
        self.assertEqual(report['stderr'], 'last log')
