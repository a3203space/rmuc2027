"""Prevent a short or selectively successful ability probe passing transfer."""
import copy
import unittest
from rmuc2027_research.rl.warp_kinematic.calibration import ability_summary


def pair():
    return dict(model=2731,reference_success=True,simplified_success=True,pass_gate=True)


class CalibrationGateTests(unittest.TestCase):
    def test_two_directions_of_one_probe_do_not_establish_twenty_attempts(self):
        result=ability_summary([pair(),pair()])['2731']
        self.assertEqual(result['paired_attempts'],2)
        self.assertFalse(result['ability_gate_pass'])

    def test_failed_attempt_is_counted_in_success_rate(self):
        rows=[pair() for _ in range(20)]
        rows[0].update(simplified_success=False,pass_gate=False)
        self.assertTrue(ability_summary(rows)['2731']['ability_gate_pass'])
        rows[1].update(simplified_success=False,pass_gate=False)
        result=ability_summary(rows)['2731']
        self.assertEqual(result['simplified_success_rate'],.9)
        self.assertFalse(result['ability_gate_pass'])

    def test_time_mismatch_and_other_models_cannot_be_hidden_by_successes(self):
        rows=[pair() for _ in range(20)]
        rows[0]['pass_gate']=False
        other=copy.deepcopy(pair());other['model']=2773
        result=ability_summary(rows+[other])
        self.assertFalse(result['2731']['ability_gate_pass'])
        self.assertFalse(result['2773']['ability_gate_pass'])
