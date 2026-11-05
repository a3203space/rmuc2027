"""Immutable recovery bundles and asynchronous GitHub Release uploads."""
import hashlib
import json
import queue
import shutil
import subprocess
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


def upload(archive, repository, tag):
    if not shutil.which('gh'):
        raise RuntimeError('GitHub CLI missing; install gh and run gh auth login locally')
    def gh(*args, check=True):
        return subprocess.run(['gh',*args,'--repo',repository],capture_output=True,text=True,timeout=60,check=check)
    if gh('release','view',tag,check=False).returncode:
        gh('release','create',tag,'--target','backup/mjwarp-progress','--prerelease',
           '--title',tag,'--notes','RL recovery checkpoints; development run, not a validated release.')
    gh('release','upload',tag,str(archive),'--clobber')


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

    def status(self, archive, state, detail=''):
        value={'archive':str(archive),'repository':self.repository,'tag':self.tag,'status':state,'detail':detail}
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
                    upload(archive,self.repository,self.tag)
                    self.status(archive,'uploaded')
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
