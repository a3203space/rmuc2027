"""CPU-only long-run scheduling, accounting and finite canary checks."""
import copy
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import zipfile

import torch

from rmuc2027_research.rl import v6_longrun as run
from rmuc2027_research.rl import v6_evaluation as evaluation
from test_v6_evaluation import fixture, write
from test_sentry_v6_training import make_model, fill_update


class BudgetAndControlTests(unittest.TestCase):
    def test_duplicate_launch_preserves_existing_state_bytes_and_mtime(self):
        for suffix in ('r01','final'):
            for entry in ('direct','main'):
                with self.subTest(suffix=suffix,entry=entry),TemporaryDirectory() as folder:
                    master=Path(folder)/'master';master.mkdir()
                    run.run_directory(master,suffix).mkdir()
                    for name in ('campaign.json','budget.json','orchestrator-error.json'):
                        write(master/name,dict(original=name,status='training',rounds=[{'accepted':True}]))
                    # No authorization exists: rejection must precede even its read.
                    before={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in master.iterdir()}
                    args=SimpleNamespace(output=master,previous=Path(folder)/'old')
                    with patch.object(run,'wait_previous') as wait,patch.object(run,'child') as child:
                        with self.assertRaisesRegex(run.InPlaceResumeUnsupported,'不支持原地resume'):
                            if entry=='direct':run.run(args)
                            else:
                                argv=['v6_longrun','--output',str(master),'--previous',str(args.previous),
                                    '--playwright-module',str(Path(folder)/'unused-playwright')]
                                with patch.object(run.sys,'argv',argv),patch.object(run.signal,'signal'):
                                    run.main()
                        wait.assert_not_called();child.assert_not_called()
                    after={p.name:(p.read_bytes(),p.stat().st_mtime_ns) for p in master.iterdir()}
                    self.assertEqual(before,after)

    def test_first_launch_still_reaches_normal_initialization(self):
        class StopBeforeWork(Exception):pass
        with TemporaryDirectory() as folder:
            master=Path(folder)/'master';master.mkdir()
            write(master/'authorization.json',dict(authorization_start_unix=1000,
                deadline_unix=29800,wall_budget_seconds=28800))
            args=SimpleNamespace(output=master,previous=Path(folder)/'old')
            with patch.object(run.time,'time',return_value=2000),\
                    patch.object(run,'wait_previous',side_effect=StopBeforeWork) as wait,\
                    patch.object(run,'child') as child:
                with self.assertRaises(StopBeforeWork):run.run(args)
                wait.assert_called_once_with(args.previous,29800)
                child.assert_not_called()
            self.assertEqual(json.loads((master/'budget.json').read_text())['deadline_unix'],29800)
            self.assertEqual(json.loads((master/'campaign.json').read_text())['status'],'waiting_for_prior_run')
            self.assertFalse(run.run_directory(master,'r01').exists())
            self.assertFalse((master/'orchestrator-error.json').exists())

    def test_playable_rounds_and_final_are_top_level_catalog_siblings(self):
        with TemporaryDirectory() as folder:
            research=Path(folder);master=research/'authorized-master';master.mkdir()
            paths=[run.run_directory(master,suffix) for suffix in ('r01','r02','final')]
            for path in paths:
                path.mkdir();write(path/'run-manifest.json',{'run_id':path.name})
                self.assertEqual(path.parent,research)
                self.assertNotIn(master,path.parents)
            self.assertEqual(set(research.glob('*/run-manifest.json')),{p/'run-manifest.json' for p in paths})
            with self.assertRaises(ValueError):run.run_directory(master,'../bad')

    def test_absolute_budget_does_not_restart_when_driver_starts_late(self):
        auth=dict(authorization_start_unix=1000,deadline_unix=29800,wall_budget_seconds=28800)
        with patch('time.time',return_value=6000):a=run.budget_from_authorization(auth)
        with patch('time.time',return_value=20000):b=run.budget_from_authorization(auth)
        self.assertEqual(a,b)
        self.assertEqual(a['training_deadline_unix'],22600)
        self.assertEqual(a['evaluation_deadline_unix'],28600)
        self.assertEqual(a['web_deadline_unix'],29080)
        for key,value in [('wall_budget_seconds',float('nan')),('deadline_unix',float('inf')),
                          ('authorization_start_unix',True),('wall_budget_seconds',28801)]:
            bad={**auth,key:value}
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):run.budget_from_authorization(bad)

    def test_tactical_lr_change_preserves_all_weights_adam_moments_and_other_heads(self):
        torch.set_num_threads(1);model,vec=make_model()
        try:
            fill_update(model)
            weights={key:value.clone() for key,value in model.policy.state_dict().items()}
            names=('tactical_optimizer','revive_optimizer','critic_optimizer')
            before={name:copy.deepcopy(getattr(model,name).state_dict()) for name in names}
            other_rates=(model.revive_lr,model.critic_lr)
            with TemporaryDirectory() as folder:
                path=Path(folder)/'control.json';write(path,dict(tactical_lr=5e-6,reason='Declared failed-branch rollback'))
                report=run.apply_training_control(model,dict(ppo=dict(tactical_lr=5e-6)),path)
                self.assertFalse(report['adam_moments_reset'])
                self.assertEqual(model.tactical_lr,5e-6)
                self.assertEqual((model.revive_lr,model.critic_lr),other_rates)
                for name in names:
                    after=getattr(model,name).state_dict()
                    expected=before[name]
                    if name=='tactical_optimizer':
                        for group in expected['param_groups']:group['lr']=5e-6
                    self.assertEqual(after['param_groups'],expected['param_groups'])
                    self.assertEqual(set(after['state']),set(expected['state']))
                    for index,state in expected['state'].items():
                        for key,value in state.items():
                            if torch.is_tensor(value):torch.testing.assert_close(value,after['state'][index][key],rtol=0,atol=0)
                            else:self.assertEqual(value,after['state'][index][key])
                for key,value in weights.items():torch.testing.assert_close(value,model.policy.state_dict()[key],rtol=0,atol=0)
                for value in (0,-1,True,float('nan'),1e-4):
                    write(path,dict(tactical_lr=value,reason='Invalid'))
                    with self.subTest(value=value),self.assertRaises(ValueError):
                        run.apply_training_control(model,dict(ppo=dict(tactical_lr=value)),path)
        finally:vec.close()

    def test_wait_owned_cleans_up_timeout_and_keyboard_interrupt(self):
        for failure in (subprocess.TimeoutExpired('owned',1),KeyboardInterrupt()):
            child=Mock(returncode=-15);child.wait.side_effect=failure
            with patch.object(run,'stop_owned') as stop,patch.object(run.time,'time',return_value=10):
                if isinstance(failure,KeyboardInterrupt):
                    with self.assertRaises(KeyboardInterrupt):run.wait_owned(child,12,120)
                else:self.assertEqual(run.wait_owned(child,12,120),-15)
                child.wait.assert_called_once_with(timeout=2)
                stop.assert_called_once_with(child)

    def test_final_summary_counts_only_accepted_lineage_and_rejects_mixed_counters(self):
        from test_v6_finalization_accounting import lineage_fixture
        with TemporaryDirectory() as folder:
            _,final,anchor,rows=lineage_fixture(Path(folder))
            report=run.final_training_summary(final,rows,anchor)
            self.assertEqual(report['actual_samples'],18)
            self.assertEqual(report['observed_samples_all_branches'],118)
            self.assertEqual(report['rejected_branch_observed_samples'],100)
            self.assertEqual(report['published_samples'],16)
            self.assertEqual(report['accepted_actor_steps'],4)
            self.assertEqual(report['rejected_actor_steps'],7)
            self.assertEqual(report['status'],'accepted_lineage_frozen_for_evaluation')
            rows[0]['training']['published_samples']+=1
            with self.assertRaisesRegex(ValueError,'lineage counters'):
                run.final_training_summary(final,rows,anchor)


class CanaryTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);fixture(self.root)
        self.case=next(c for c in evaluation.required_cases() if c['id']=='main-rapier-blue')

    def mutate(self,member,fn):
        path=self.root/'evaluation'/(self.case['id']+'-'+member+'.json')
        row=json.loads(path.read_text());fn(row);write(path,row)

    def test_valid_full_pair_passes_continuation_only(self):
        report=run.canary_gate(self.root,self.case)
        self.assertTrue(report['passed'],report)
        self.assertEqual(report['reference']['capture_seconds'],6)
        self.assertIn('not deployment',report['purpose'])

    def test_truncation_or_missing_passage_rejects_even_if_capture_is_large(self):
        self.mutate('candidate',lambda row:row.update(censored=True,horizon_completed=False))
        report=run.canary_gate(self.root,self.case)
        self.assertFalse(report['passed']);self.assertIn('incomplete_trial',str(report['failures']))

    def test_execution_and_prefix_difference_rejects(self):
        def change(row):
            row['identity']['sources']['sdk_hash']=evaluation._digest('different executor')
            row['identity']['execution_hash']=evaluation._digest(row['identity']['sources'])
        self.mutate('candidate',change)
        report=run.canary_gate(self.root,self.case)
        self.assertFalse(report['passed']);self.assertIn('paired_execution',str(report['failures']))
        self.assertIsNone(report['reference'])

    def test_capture_control_regression_and_zero_anchor_reject(self):
        def change(row):
            row['metrics']['capture_seconds']=0;row['capture_after_recovery']=0;row['control']['blue']=0
        self.mutate('candidate',change)
        report=run.canary_gate(self.root,self.case)
        self.assertFalse(report['passed']);self.assertIn('capture_regression',str(report['failures']))
        self.assertIn('control_regression',str(report['failures']))
        self.mutate('baseline',change)
        report=run.canary_gate(self.root,self.case)
        self.assertFalse(report['passed']);self.assertIn('baseline_did_not_exercise',str(report['failures']))
        self.assertIsNone(report['reference'])

    def test_initial_anchor_floor_prevents_repeated_one_second_drift(self):
        reference=run.canary_gate(self.root,self.case)['reference']
        for member,capture in [('baseline',5.2),('candidate',4.5)]:
            self.mutate(member,lambda row,value=capture:(row['metrics'].update(capture_seconds=value),row.update(capture_after_recovery=0)))
        report=run.canary_gate(self.root,self.case,reference)
        self.assertFalse(report['passed']);self.assertIn('original_guard_anchor_regression',str(report['failures']))
        reference['capabilities_sha256']=evaluation._digest('different capacities')
        report=run.canary_gate(self.root,self.case,reference)
        self.assertIn('guard_reference_identity_mismatch',str(report['failures']))

    def test_malformed_record_is_a_rejection_not_driver_crash(self):
        write(self.root/'evaluation'/(self.case['id']+'-candidate.json'),['wrong type'])
        report=run.canary_gate(self.root,self.case)
        self.assertFalse(report['passed']);self.assertIsNone(report['candidate_capture'])


class EvidenceArchiveTests(unittest.TestCase):
    def fixture(self,folder):
        parent=Path(folder);rows=[]
        for index in (1,2):
            root=parent/f'round-{index}';root.mkdir()
            row=dict(round=index,path=str(root),accepted_for_continuation=index==1);rows.append(row)
            write(root/'main/completed-episodes/prefix-failed.json',dict(prepared=False,learning_samples=0))
            write(root/'main/training.json',dict(actual_samples=index*4))
            (root/'main/updates.jsonl').write_text('{"tactical_steps":1}\n')
            write(root/'evaluation/main-rapier-blue-candidate.json',dict(case_id='main-rapier-blue'))
            (root/'evaluation/main-rapier-blue-candidate.jsonl').write_text('{"time":0.1}\n')
            write(root/'github-backups/receipt.json',dict(remote_verified=True,commit=str(index),
                url='https://github.com/example/research/commit/'+str(index),checkpoint=dict(samples=index*4)))
            write(root/'github-backups/.git/objects/private',{'never':'copy'})
            write(root/'main/checkpoints/update-1/model.zip',{'never':'copy'})
            (root/'do-not-copy.zip').write_bytes(b'never recursively copy')
            self.replay(root/'replays/training/worker-0-episode-00001.json',index)
            self.replay(root/'replays/evaluation/main-rapier-blue-candidate.json',index,kind='evaluation')
        final=parent/'final';final.mkdir()
        return final,rows

    def replay(self,path,index,kind='training'):
        write(path,dict(format='rmuc-replay-v4',metadata=dict(map='rmuc2027',kind=kind,
            training_recording=dict(learner_start_time=0,policy_changes_during_episode=True),
            recording_end_time=.1,recording_complete=True,truncated=True,policy_update_range=[index,index]),
            static={},frames=[dict(time=0,tick=0),dict(time=.1,tick=1)],actions=[{}],diagnostics=[],result={}))

    def test_round_names_preserve_same_basename_and_all_copied_hashes(self):
        with TemporaryDirectory() as folder:
            final,rows=self.fixture(folder)
            report=run.archive_round_evidence(final,rows,replay_byte_limit=100000)
            self.assertEqual(report['included_training_replays'],2)
            files=list((final/'replays/training').glob('*.json'))
            self.assertEqual(len(files),2)
            self.assertEqual(len({p.name for p in files}),2)
            for record in report['records']:
                if record['included']:
                    self.assertEqual(run.sha(final/record['destination']),record['sha256'])
                    self.assertEqual(run.sha(record['source']),record['sha256'])
                    self.assertEqual((final/record['destination']).stat().st_size,record['bytes'])
            self.assertEqual(len(report['backup_receipts']),2)
            self.assertTrue(all(item['url'] for item in report['backup_receipts']))
            self.assertFalse(any(p.suffix=='.zip' or '.git' in p.parts or 'checkpoints' in p.parts for p in final.rglob('*')))
            self.assertEqual(len(list((final/'review/round-evidence').glob('*/evaluation/*.jsonl'))),2)
            # A preexisting conflicting destination must never be overwritten.
            first=files[0];first.write_text('conflicting evidence')
            with self.assertRaisesRegex(ValueError,'hash collision'):run.archive_round_evidence(final,rows,replay_byte_limit=100000)
            self.assertEqual(first.read_text(),'conflicting evidence')

    def test_quota_prioritizes_true_training_and_records_every_omission(self):
        with TemporaryDirectory() as folder:
            final,rows=self.fixture(folder)
            source=Path(rows[0]['path'])/'replays/training/worker-0-episode-00001.json'
            limit=source.stat().st_size
            report=run.archive_round_evidence(final,rows,replay_byte_limit=limit)
            self.assertLessEqual(report['included_native_replay_bytes'],limit)
            self.assertEqual(report['included_training_replays'],1)
            self.assertTrue(report['training_playback_available'])
            omitted=[x for x in report['records'] if not x['included']]
            self.assertEqual(len(omitted),3)
            self.assertTrue(all(x['omission_reason']=='native_replay_archive_byte_limit' and x['sha256'] and x['bytes']>0 for x in omitted))
            self.assertTrue(all(json.loads(p.read_text())['metadata']['kind']=='training' for p in (final/'replays/training').glob('*.json')))

    def test_evaluation_masquerading_as_training_is_never_relabelled(self):
        with TemporaryDirectory() as folder:
            final,rows=self.fixture(folder)
            for row in rows:
                self.replay(Path(row['path'])/'replays/training/worker-0-episode-00001.json',1,kind='evaluation')
            report=run.archive_round_evidence(final,rows,replay_byte_limit=100000)
            self.assertFalse(report['training_playback_available'])
            self.assertEqual(report['included_training_replays'],0)
            self.assertTrue(any('kind disagree' in x.get('omission_reason','') for x in report['records']))

    def test_backup_requires_exact_final_zip_generation_and_counters(self):
        with TemporaryDirectory() as folder:
            root=Path(folder);ready=dict(samples=123,update_id=9,sha256={'model.zip':'a'*64})
            write(root/'main/checkpoint-ready.json',ready);write(root/'package.json',dict(sha256='b'*64))
            path=root/'github-backups/receipt.json'
            receipt=dict(remote_verified=True,final=dict(sha256='b'*64),checkpoint=dict(files=ready['sha256'],samples=123,update_id=9))
            write(path,receipt);self.assertTrue(run.audit_final_backup(root)['verified'])
            receipt['checkpoint']['samples']=124;write(path,receipt)
            self.assertFalse(run.audit_final_backup(root)['verified'])
            receipt['checkpoint']['samples']=123;receipt['final']=None;write(path,receipt)
            self.assertFalse(run.audit_final_backup(root)['verified'])


if __name__=='__main__':unittest.main()
