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
            (root/'main/experiment.json').write_text('{}')
            for role in ('infantry','heavy'):
                (root/role).mkdir();(root/role/'actor.json').write_bytes(b'frozen-helper')
        rows=[dict(backend='rapier',side='red',checkpoint=checkpoint,steps=2,simulation_seconds=60.,
            horizon_completed=True,censored=False,final_position=[3.,4.,.3]) for checkpoint in ('initialized.zip','model.zip')]
        (source/'evaluation/rapier-red.json').write_text(json.dumps(rows))
        trace=[dict(time=t,position=[3.,4.,.3],action=[0.]*6,navigation={}) for t in (30.,60.)]
        (source/'evaluation/rapier-red-initialized.zip.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in trace))
        return source,target,rows

    def evidence(self,source):
        records={}
        for path in source.rglob('*'):
            if path.is_file():
                relative=path.relative_to(source).as_posix()
                if relative.split('/')[0] in ('main','infantry','heavy'):relative='models/'+relative
                records[relative]={'sha256':reuse.digest(path)}
        return {'manifest_sha256':'source-proof','sealed_files':records}

    def test_reuse_returns_exact_trace_with_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            source,target,_=self.fixture(directory)
            with patch.object(reuse,'_signature',return_value={'effective':'same'}), \
                 patch.object(reuse,'_runtime_evidence',return_value=self.evidence(source)):
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
                                  return_value=self.evidence(source)):
                    with self.assertRaises(ValueError):reuse.validate_anchor_reuse(target,source,'rapier','red')

    def test_session_exception_accepts_only_exact_reward_insertion(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'session.py';original=b'before\nafter\n'
            expected=hashlib.sha256(original).hexdigest()
            path.write_bytes(b'before\n'+reuse.CAPTURE_REWARD_INSERTION+b'after\n')
            self.assertIn(expected,reuse._session_equivalent_hashes(path))
            path.write_bytes(path.read_bytes()+b'other behavior change\n')
            self.assertNotIn(expected,reuse._session_equivalent_hashes(path))

    def test_signature_normalizes_paths_and_training_options_but_rejects_helper_radar_changes(self):
        from rmuc2027_research.rl import session
        with tempfile.TemporaryDirectory() as directory:
            source,target,_=self.fixture(directory)
            runtime={key:'unchanged' for key in ('sdk_hash','runtime_hash','sentry_sources','sentry_schema',
                'scenario_registry_hash','capability_schema_hash','engine_binary_sha256','rules_sha256')}
            runtime.update(sources={'friendly':{'hash':'f'},'opponent':{'hash':'o'}},
                navigation={'preparation_sha256':'n'},simulation={'backend':'rapier'},
                training_sources={name:'same' for name in ('session.py','env.py','aux_policy.py',
                    'privileged_aux.py','capabilities.py','scenarios.py')})
            for root in (source,target):
                radar=root/'radar.json';radar.write_bytes(b'frozen-radar')
                cfg=dict(friendly=str(root/'friendly'),opponent=str(root/'opponent'),
                    radar_classifier_path=str(radar),ppo={'learning_rate':.1 if root==source else .2},
                    episode_seconds=420 if root==source else 120,v4_capture_reward=.02 if root==target else 0.)
                metadata=dict(runtime,config=cfg,dependencies={},
                    auxiliary_pool={role:{'sha256':reuse.digest(root/role/'actor.json')} for role in ('infantry','heavy')},
                    radar_classifier_sha256=reuse.digest(radar))
                (root/'main/experiment.json').write_text(json.dumps(metadata))
            with patch.object(session,'provenance',return_value=runtime),patch.object(session,'load_config',side_effect=lambda cfg:cfg):
                self.assertEqual(reuse._signature(source,'rapier','red'),reuse._signature(target,'rapier','red'))
                (target/'infantry/actor.json').write_bytes(b'new helper')
                with self.assertRaisesRegex(ValueError,'helper'):reuse._signature(target,'rapier','red')
                (target/'infantry/actor.json').write_bytes(b'frozen-helper')
                (target/'radar.json').write_bytes(b'new radar')
                with self.assertRaisesRegex(ValueError,'radar'):reuse._signature(target,'rapier','red')


if __name__=='__main__':unittest.main()
