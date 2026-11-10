import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from rmuc2027_research.rl.sentry_selection import select


class SentrySelectionTests(unittest.TestCase):
    def test_cached_trial_cannot_hide_execution_runtime_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);checkpoint=root/'model.zip';checkpoint.write_bytes(b'checkpoint')
            (root/'experiment.json').write_text(json.dumps({'sdk_hash':'current'}))
            folder=root/'selected/trials/candidate/red';folder.mkdir(parents=True)
            (folder/'result.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),evaluation_sdk_hash='previous')))
            with self.assertRaisesRegex(ValueError,'runtime or checkpoint bytes'):
                select(root/'selected',2772,{'candidate':str(checkpoint)},time.time()+60)

    def test_paired_safety_gate_rejects_high_capture_with_tip(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);candidates={}
            for name in ('unsafe','one_side','balanced'):
                folder=root/name;folder.mkdir();(folder/'model.zip').write_bytes(name.encode())
                (folder/'experiment.json').write_text('{}');candidates[name]=str(folder/'model.zip')
            def evaluate(checkpoint,folder,backend,side,seed,deadline,horizon):
                name=checkpoint.parent.name;folder.mkdir(parents=True)
                capture=40 if name=='unsafe' else (30 if side=='red' else 0) if name=='one_side' else 8
                value=dict(checkpoint=str(checkpoint),side=side,horizon_completed=True,policy_errors=0,tipped_seconds=int(name=='unsafe'),
                    tunnel_spin_violations=0,path_m=20,simulation_seconds=60,final={'metrics':{'capture_seconds':capture}})
                (folder/'result.json').write_text(json.dumps(value))
            with patch('rmuc2027_research.rl.sentry_selection.evaluate_worker',side_effect=evaluate),contextlib.redirect_stdout(io.StringIO()):
                result=select(root/'selected',2771,candidates,time.time()+600)
            self.assertEqual(result['selected'],'balanced')
            self.assertEqual((root/'selected/model.zip').read_bytes(),b'balanced')
            self.assertFalse(result['full_match_strength_qualified'])
            self.assertEqual((root/'unsafe/model.zip').read_bytes(),b'unsafe')

    def test_expired_selection_cannot_run_or_accept_partial_trial(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);checkpoint=root/'model.zip';checkpoint.write_bytes(b'checkpoint')
            with patch('rmuc2027_research.rl.sentry_selection.evaluate_worker',side_effect=AssertionError('expired trial started')):
                result=select(root/'selected',2771,{'candidate':str(checkpoint)},time.time()-1)
            self.assertIsNone(result);self.assertFalse((root/'selected/model.zip').exists())

    def test_zero_capture_prefers_public_card_progress_over_standing_still(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);candidates={}
            for name in ('stationary','near_card'):
                folder=root/name;folder.mkdir();(folder/'model.zip').write_bytes(name.encode())
                (folder/'experiment.json').write_text('{}');candidates[name]=str(folder/'model.zip')
            def evaluate(checkpoint,folder,backend,side,seed,deadline,horizon):
                moving=checkpoint.parent.name=='near_card';folder.mkdir(parents=True)
                value=dict(checkpoint=str(checkpoint),side=side,horizon_completed=True,policy_errors=0,tipped_seconds=0,
                    tunnel_spin_violations=0,path_m=20 if moving else 0,simulation_seconds=60,
                    final={'metrics':{'capture_seconds':0,'objective_gap_m':.5 if moving else 8}})
                (folder/'result.json').write_text(json.dumps(value))
            with patch('rmuc2027_research.rl.sentry_selection.evaluate_worker',side_effect=evaluate),contextlib.redirect_stdout(io.StringIO()):
                result=select(root/'selected',2772,candidates,time.time()+600)
            self.assertEqual(result['selected'],'near_card')


if __name__=='__main__':unittest.main()
