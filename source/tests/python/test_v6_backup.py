"""Backup transport regression tests use disposable local Git repos, no network."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

from rmuc2027_research.rl.v6_backup import BackupWorker, REQUIRED, sha, write_json


class BackupTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name);self.repo=root/'repo';self.repo.mkdir();self.remote=root/'remote.git'
        self.git('init','-q');self.git('config','user.name','Tests');self.git('config','user.email','test@example.invalid')
        (self.repo/'source.py').write_text('value = 1\n')
        self.git('add','source.py');self.git('commit','-qm','initial')
        subprocess.run(['git','init','--bare','--quiet',str(self.remote)],check=True)
        self.git('remote','add','origin',str(self.remote))
        self.run=self.repo/'data/research/run-one';self.run.mkdir(parents=True)
        self.worker=BackupWorker(self.run,time.time()+120,repo_root=self.repo)
        self.head=self.git('rev-parse','HEAD');self.index=sha(self.repo/'.git/index')

    def git(self,*args):
        return subprocess.check_output(['git',*args],cwd=self.repo,stderr=subprocess.PIPE).decode().strip()

    def generation(self,samples=64):
        folder=self.run/'main/checkpoints/update-1';folder.mkdir(parents=True)
        with zipfile.ZipFile(folder/'model.zip','w') as archive:archive.writestr('data',json.dumps(dict(num_timesteps=samples)))
        (folder/'resume-state.pt').write_bytes(b'rng checkpoint fixture')
        for name in ('samplers.json','actor.json','experiment.json','coordinator-state.json'):(folder/name).write_text('{}')
        ready=dict(directory=str(folder.relative_to(self.run)),samples=samples,update_id=1,
                   sha256={name:sha(folder/name) for name in REQUIRED})
        write_json(self.run/'main/checkpoint-ready.json',ready)
        return folder

    def assert_shared_unchanged(self):
        self.assertEqual(self.git('rev-parse','HEAD'),self.head)
        self.assertEqual(sha(self.repo/'.git/index'),self.index)

    def test_periodic_worker_yields_to_web_and_final_packaging(self):
        for stage in ('web_acceptance','results_frozen'):
            write_json(self.run/'campaign.json',dict(status=stage))
            self.worker.deadline=time.time()+.1
            with patch.object(self.worker,'snapshot') as snapshot:
                self.worker.loop()
                snapshot.assert_not_called()
        self.assert_shared_unchanged()

    def test_snapshot_keeps_worktree_without_db_and_complete_generation(self):
        self.generation()
        (self.repo/'source.py').write_text('value = 2\n')
        (self.repo/'new.py').write_text('extra = True\n')
        (self.repo/'rmuc2026shuj').mkdir();(self.repo/'rmuc2026shuj/raw.sqlite').write_text('PRIVATE RAW DB')
        (self.repo/'node_modules').mkdir();(self.repo/'node_modules/large.js').write_text('cache')
        result=self.worker.snapshot()
        self.assertTrue(result['remote_verified'])
        self.assertEqual(result['checkpoint']['samples'],64)
        self.assertIn('coordinator-state.json',result['checkpoint']['files'])
        files=self.git('ls-tree','-r','--name-only',result['commit'])
        self.assertIn('source/new.py',files);self.assertNotIn('raw.sqlite',files)
        self.assertNotIn('node_modules',files);self.assertNotIn('source/data/',files)
        self.assertEqual(self.git('show',result['commit']+':source/source.py'),'value = 2')
        self.assert_shared_unchanged()
        second=self.worker.snapshot()
        self.assertEqual(second['parent'],result['commit'])
        self.assert_shared_unchanged()

    def test_source_deletion_and_new_source_do_not_retain_stale_paths(self):
        first=self.worker.snapshot()
        (self.repo/'source.py').unlink();(self.repo/'replacement.py').write_text('replacement')
        second=self.worker.snapshot()
        self.assertEqual(second['parent'],first['commit'])
        files=self.git('ls-tree','-r','--name-only',second['commit'])
        self.assertIn('source/replacement.py',files);self.assertNotIn('source/source.py',files)
        self.assert_shared_unchanged()

    def test_unowned_remote_tip_is_rejected_without_force_or_shared_changes(self):
        self.worker.snapshot()
        self.worker.receipt_path.unlink()
        with self.assertRaisesRegex(RuntimeError,'outside this worker'):self.worker.snapshot()
        self.assert_shared_unchanged()

    def test_pending_push_is_confirmed_only_when_remote_matches_that_commit(self):
        first=self.worker.snapshot()
        write_json(self.worker.pending_path,first)
        self.worker.receipt_path.unlink()
        second=self.worker.snapshot()
        self.assertEqual(second['parent'],first['commit'])
        self.assertTrue(second['remote_verified'])
        self.assertFalse(self.worker.pending_path.exists())
        self.assert_shared_unchanged()

    def test_two_backup_operations_cannot_overlap(self):
        other=BackupWorker(self.run,time.time()+120,repo_root=self.repo)
        with self.worker.locked():
            with self.assertRaisesRegex(RuntimeError,'Another backup'):other.snapshot()
        self.assert_shared_unchanged()

    def test_ready_hash_and_count_must_agree(self):
        folder=self.generation()
        (folder/'actor.json').write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError,'SHA mismatch'):self.worker.generation()
        ready=json.loads((self.run/'main/checkpoint-ready.json').read_text())
        ready['sha256']['actor.json']=sha(folder/'actor.json');ready['samples']=65
        write_json(self.run/'main/checkpoint-ready.json',ready)
        with self.assertRaisesRegex(ValueError,'sample counts'):self.worker.generation()

    def test_final_parts_reassemble_exact_bytes_and_follow_same_branch(self):
        first=self.worker.snapshot()
        bundle=self.run.parent/'final.zip'
        with zipfile.ZipFile(bundle,'w') as archive:archive.writestr('report.json','{"partial":true}')
        write_json(self.run/'package.json',dict(sha256=sha(bundle)))
        with patch('rmuc2027_research.rl.v6_backup.CHUNK_BYTES',128):
            # The tiny chunk test uses an empty source tree/report helper override,
            # leaving the source snapshot's ordinary size rules independent.
            with patch.object(self.worker,'_source_tree',return_value=(first['source_tree'],1)),patch.object(self.worker,'_reports_helpers',return_value={}):
                # JSON manifests exceed 128 bytes, so only stream slicing is reduced.
                original=self.worker._include
                def include(env,name,value):
                    with patch('rmuc2027_research.rl.v6_backup.CHUNK_BYTES',48*1024*1024):return original(env,name,value)
                with patch.object(self.worker,'_include',side_effect=include):result=self.worker.upload_final(bundle)
        info=result['final'];self.assertEqual(result['parent'],first['commit'])
        data=b''.join(subprocess.check_output(['git','show',result['commit']+':'+info['path']+'/'+part['name']],cwd=self.repo) for part in info['parts'])
        self.assertEqual(data,bundle.read_bytes());self.assertEqual(info['sha256'],sha(bundle))
        self.assert_shared_unchanged()

    def test_final_database_and_package_hash_rejected(self):
        bundle=self.run.parent/'final.zip'
        with zipfile.ZipFile(bundle,'w') as archive:archive.writestr('preparation/raw.sqlite','data')
        with self.assertRaisesRegex(ValueError,'Raw database'):self.worker.upload_final(bundle)
        with zipfile.ZipFile(bundle,'w') as archive:archive.writestr('ok.json','{}')
        write_json(self.run/'package.json',dict(sha256='0'*64))
        with self.assertRaisesRegex(ValueError,'package.json'):self.worker.upload_final(bundle)

    def test_deadline_prevents_spawn_and_timeouts_never_exceed_remaining(self):
        worker=BackupWorker(self.run,100,repo_root=self.repo,clock=lambda:95.)
        with patch('subprocess.run',return_value=type('Result',(),dict(returncode=0,stdout=b'',stderr=b''))()) as call:
            worker.git('ls-remote','origin',worker.ref,limit=7200)
        self.assertEqual(call.call_args.kwargs['timeout'],5.)
        self.assertNotIn('start_new_session',call.call_args.kwargs)
        worker.clock=lambda:101.
        with patch('subprocess.run') as call:
            with self.assertRaises(TimeoutError):worker.git('ls-remote','origin',worker.ref)
        call.assert_not_called()

    def test_unknown_generation_path_is_rejected(self):
        self.generation();marker=self.run/'main/checkpoint-ready.json'
        ready=json.loads(marker.read_text());ready['directory']='../../elsewhere';write_json(marker,ready)
        with self.assertRaisesRegex(ValueError,'within main/checkpoints'):self.worker.generation()


if __name__=='__main__':unittest.main()
