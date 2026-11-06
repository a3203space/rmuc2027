"""Immutable recovery bundles with asynchronous GitHub/SSH uploads."""
import hashlib
import json
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path


def bundle(run, checkpoint, progress):
    run = Path(run)
    directory = run/'backups'
    directory.mkdir(exist_ok=True)
    target = directory/f'checkpoint-{time.time_ns()}-{uuid.uuid4().hex[:8]}.zip'
    # Read each atomically published generation once; hashing and archive agree.
    payload = {'model.zip': Path(checkpoint).read_bytes(),
               'experiment.json': (run/'experiment.json').read_bytes(),
               'progress.json': json.dumps(progress, indent=2).encode()}
    manifest = {name:hashlib.sha256(data).hexdigest() for name,data in payload.items()}
    with zipfile.ZipFile(target.with_suffix('.tmp'), 'w', zipfile.ZIP_DEFLATED) as archive:
        for name,data in payload.items(): archive.writestr(name,data)
        archive.writestr('checksums.json',json.dumps(manifest))
    target.with_suffix('.tmp').replace(target)
    return target


def _upload_git_archive(archive, remote, tag):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',tag):
        raise ValueError('Invalid checkpoint backup tag')
    archive=Path(archive)
    if archive.stat().st_size>95*1024*1024:
        raise ValueError('Checkpoint bundle exceeds the SSH Git transport size limit')
    ref='refs/heads/backup/rl-checkpoints/'+tag
    with tempfile.TemporaryDirectory(prefix='rmuc-checkpoint-git-') as directory:
        subprocess.run(['git','init','--bare','--quiet',directory],capture_output=True,check=True,timeout=60)
        def git(*args,payload=None):
            return subprocess.run(['git','--git-dir',directory,*args],input=payload,
                                  capture_output=True,check=True,timeout=60).stdout.strip()
        old=git('ls-remote',remote,ref).split()
        parent=[]
        if old:
            git('fetch','--quiet','--depth=1',remote,ref)
            parent=['-p',old[0].decode()]
        data=archive.read_bytes();digest=hashlib.sha256(data).hexdigest()
        blob=git('hash-object','-w','--stdin',payload=data).decode()
        metadata=json.dumps({'archive':archive.name,'sha256':digest,'transport':'ssh-git'}).encode()
        meta=git('hash-object','-w','--stdin',payload=metadata).decode()
        tree=git('mktree',payload=f'100644 blob {meta}\tlatest.json\n100644 blob {blob}\tlatest.zip\n'.encode()).decode()
        commit=git('-c','user.name=RMUC RL Backup','-c','user.email=rmuc2027-backup@users.noreply.github.com',
                   'commit-tree',tree,*parent,'-m','RL recovery checkpoint '+digest[:12]).decode()
        # A concurrent update is rejected rather than overwriting remote work.
        git('push','--quiet',remote,commit+':'+ref)
        verified=git('ls-remote',remote,ref).split()
        if not verified or verified[0].decode()!=commit:
            raise RuntimeError('Checkpoint ref could not be verified after push')
    return {'transport':'ssh-git','ref':ref,'commit':commit,'sha256':digest}


def upload(archive, repository, tag):
    if not shutil.which('gh'):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',repository):
            raise ValueError('Invalid backup repository')
        return _upload_git_archive(archive,'git@github.com:'+repository+'.git',tag)
    def gh(*args, check=True):
        return subprocess.run(['gh',*args,'--repo',repository],capture_output=True,text=True,timeout=60,check=check)
    if gh('release','view',tag,check=False).returncode:
        gh('release','create',tag,'--target','backup/mjwarp-progress','--prerelease',
           '--title',tag,'--notes','RL recovery checkpoints; development run, not a validated release.')
    gh('release','upload',tag,str(archive),'--clobber')
    return {'transport':'github-release'}


class ReleaseBackup:
    def __init__(self, run, repository):
        self.run, self.repository = Path(run), repository
        self.tag = 'rl-run-'+hashlib.sha256(str(self.run.resolve()).encode()).hexdigest()[:16]
        self.jobs = queue.Queue()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.worker,daemon=True)
        self.thread.start()

    def submit(self, checkpoint, progress):
        # Backup storage is best effort; a full disk or missing metadata must
        # not abort an otherwise healthy training rollout.
        try:
            archive = bundle(self.run,checkpoint,progress)
        except Exception as error:
            print(json.dumps({'kind':'backup-error','error':type(error).__name__}),flush=True)
            return None
        self.jobs.put(archive)
        return archive

    def status(self, archive, state, detail='', receipt=None):
        value={'archive':str(archive),'repository':self.repository,'tag':self.tag,'status':state,'detail':detail}
        if isinstance(receipt,dict):value.update(receipt)
        path=archive.with_suffix('.status.json')
        path.with_suffix('.tmp').write_text(json.dumps(value,indent=2))
        path.with_suffix('.tmp').replace(path)
        print(json.dumps({'kind':'checkpoint-backup',**value}),flush=True)

    def worker(self):
        while True:
            try: archive=self.jobs.get(timeout=.2)
            except queue.Empty:
                if self.stop.is_set(): return
                continue
            self.status(archive,'pending')
            while True:
                try:
                    receipt=upload(archive,self.repository,self.tag)
                    self.status(archive,'uploaded',receipt=receipt)
                    break
                except Exception as error:
                    # Never print subprocess output, which may contain credentials.
                    self.status(archive,'retry_pending',type(error).__name__)
                    if self.stop.wait(30): break
            self.jobs.task_done()
            if self.stop.is_set(): return

    def close(self):
        deadline = time.monotonic()+30
        while self.jobs.unfinished_tasks and time.monotonic()<deadline:
            time.sleep(.05)
        self.stop.set()
        self.thread.join(timeout=.5)
