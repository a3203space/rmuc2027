import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rmuc2027_research.rl import v4_anchor_reuse as reuse


class AnchorReuseTests(unittest.TestCase):
    def fixture(self, directory):
        source=Path(directory)/'old';target=Path(directory)/'new'
        for root in (source,target):
            (root/'main').mkdir(parents=True);(root/'evaluation').mkdir()
            (root/'main/initialized.zip').write_bytes(b'identical frozen anchor')
        rows=[dict(backend='rapier',side='red',checkpoint=checkpoint,steps=2,simulation_seconds=60.,
            horizon_completed=True,censored=False,final_position=[3.,4.,.3]) for checkpoint in ('initialized.zip','model.zip')]
        (source/'evaluation/rapier-red.json').write_text(json.dumps(rows))
        trace=[dict(time=t,position=[3.,4.,.3],action=[0.]*6,navigation={}) for t in (30.,60.)]
        (source/'evaluation/rapier-red-initialized.zip.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in trace))
        return source,target,rows

    def test_reuse_returns_exact_trace_with_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            source,target,_=self.fixture(directory)
            with patch.object(reuse,'_signature',return_value={'effective':'same'}), \
                 patch.object(reuse,'_runtime_evidence',return_value={'manifest_sha256':'source-proof'}):
                row,trace=reuse.validate_anchor_reuse(target,source,'rapier','red')
            self.assertTrue(row['reused'])
            self.assertEqual(trace,(source/'evaluation/rapier-red-initialized.zip.jsonl').read_bytes())
            self.assertEqual(row['reuse']['trace_sha256'],hashlib.sha256(trace).hexdigest())
            self.assertFalse((target/'evaluation/rapier-red-initialized.zip.jsonl').exists())

    def test_rejects_checkpoint_config_source_or_trace_mismatch(self):
        for corruption in ('checkpoint','configuration','source','trace','order','incomplete'):
            with self.subTest(corruption=corruption),tempfile.TemporaryDirectory() as directory:
                source,target,rows=self.fixture(directory)
                if corruption=='checkpoint':(target/'main/initialized.zip').write_bytes(b'changed')
                if corruption=='trace':(source/'evaluation/rapier-red-initialized.zip.jsonl').write_text('{}\n')
                if corruption=='order':rows.reverse()
                if corruption=='incomplete':rows[1]['horizon_completed']=False
                (source/'evaluation/rapier-red.json').write_text(json.dumps(rows))
                with patch.object(reuse,'_signature',side_effect=[{'same':1},{'same':2 if corruption=='configuration' else 1}]), \
                     patch.object(reuse,'_runtime_evidence',side_effect=ValueError('source changed') if corruption=='source' else None,
                                  return_value={'manifest_sha256':'source-proof'}):
                    with self.assertRaises(ValueError):reuse.validate_anchor_reuse(target,source,'rapier','red')

    def test_session_exception_accepts_only_exact_reward_insertion(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'session.py';original=b'before\nafter\n'
            expected=hashlib.sha256(original).hexdigest()
            path.write_bytes(b'before\n'+reuse.CAPTURE_REWARD_INSERTION+b'after\n')
            self.assertIn(expected,reuse._session_equivalent_hashes(path))
            path.write_bytes(path.read_bytes()+b'other behavior change\n')
            self.assertNotIn(expected,reuse._session_equivalent_hashes(path))


if __name__=='__main__':unittest.main()
