import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rmuc2027_research.rl.campaign import configuration
from rmuc2027_research.rl.selection import evaluate_candidate, finish_gpu
from rmuc2027_research.rl.session import provenance


class FrozenCacheTests(unittest.TestCase):
    def fixture(self,root):
        model=root/'model';model.mkdir()
        checkpoint=model/'model.zip';checkpoint.write_bytes(b'frozen checkpoint')
        config=configuration(4,500)
        (model/'experiment.json').write_text(json.dumps({'config':config}))
        directory=root/'evaluation';directory.mkdir()
        report=dict(weights_unchanged=True,gradient_updates=0,backend='rapier',stage='course',guided_courses=True,
            model_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),evaluation_sdk_hash=provenance(config)['sdk_hash'],
            results=[dict(seed=seed,side=side) for seed in (42,43,44) for side in ('red','blue')])
        (directory/'validation.json').write_text(json.dumps(report))
        return checkpoint,directory,report

    def test_reuses_complete_frozen_report_without_running_again(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint,output,report=self.fixture(Path(directory))
            with patch('rmuc2027_research.rl.selection.validate') as validate:
                self.assertEqual(evaluate_candidate(checkpoint,output,'rapier'),report)
                validate.assert_not_called()

    def test_changed_checkpoint_and_incomplete_report_are_preserved_and_retested(self):
        for change in ('weights','incomplete','backend'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);checkpoint,output,report=self.fixture(root)
                if change=='weights':checkpoint.write_bytes(b'other checkpoint')
                if change=='incomplete':report['results'].pop()
                if change=='backend':report['backend']='mjwarp'
                (output/'validation.json').write_text(json.dumps(report))
                with patch('rmuc2027_research.rl.selection.validate',return_value={'fresh':True}) as validate:
                    self.assertEqual(evaluate_candidate(checkpoint,output,'rapier'),{'fresh':True})
                    validate.assert_called_once()
                self.assertEqual(len(list(root.glob('evaluation-incomplete-*/validation.json'))),1)

    def test_prunes_candidate_that_cannot_match_verified_stable_success(self):
        def report(backend,successes):
            return dict(backend=backend,results=[dict(success=i<successes,
                final=dict(metrics=dict(tipped_seconds=0.,alive_seconds=10.))) for i in range(6)])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);prepared=root/'prepared';prepared.mkdir()
            output=root/'output';(output/'2772').mkdir(parents=True)
            (prepared/'preparation.json').write_text(json.dumps(dict(results=[dict(model=2772,checkpoint='reference.zip')])))
            result=dict(model=2772,candidates=[
                dict(kind='ppo',checkpoint='ppo.zip',evaluations=[report('rapier',4)]),
                dict(kind='reference',checkpoint='reference.zip',evaluations=[report('rapier',5)]),
                dict(kind='restored',checkpoint='restored.zip',evaluations=[report('rapier',5)])])
            def evaluate(checkpoint,path,backend):
                self.assertNotEqual(checkpoint,'ppo.zip')
                return report(backend,6 if checkpoint=='reference.zip' or backend=='warp_kinematic' else 5)
            with patch('rmuc2027_research.rl.selection.evaluate_candidate',side_effect=evaluate) as validate, \
                    patch('rmuc2027_research.rl.selection.export_model') as export:
                finish_gpu(result,prepared,output)
            self.assertEqual(validate.call_count,4)
            self.assertEqual(result['combined_score'][0],17)
            self.assertEqual(result['checkpoint'],'reference.zip')
            self.assertEqual(result['pruned_candidates'][0]['maximum_stable_success'],16)
            export.assert_called_once_with('reference.zip',output/'2772/export',allow_sdk_transfer=True)


if __name__=='__main__':unittest.main()
