"""Deadline-bounded Git/SSH backups; no training imports and no shared index edits.

The caller authorizes/starts this worker. Importing it performs no network access.
Every published generation is immutable and announced by checkpoint-ready.json.
"""
import argparse
from contextlib import contextmanager
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import zipfile

from rmuc2027_core.paths import ROOT

CHUNK_BYTES = 48*1024*1024
REQUIRED = ('model.zip', 'resume-state.pt', 'samplers.json', 'actor.json', 'experiment.json')
EXCLUDED_ROOTS = {'data','rmuc2026shuj','node_modules','target','build','dist','.git','.venv','.repo-tools'}
EXCLUDED_PARTS = {'node_modules','target','__pycache__','.venv','venv','.pytest_cache','.mypy_cache','.ruff_cache','.cache','.git'}
REPORTS = ('run-manifest.json','run-results.json','release-index.json','campaign.json','package.json',
           'bundle-verification.json','budget.json','initialization.json','preflight.json',
           'runtime-environment.json','requirements-runtime.txt','main/training.json')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    pending=path.with_suffix(path.suffix+'.pending')
    pending.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    pending.replace(path)


def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def excluded(name):
    path=Path(name)
    return (not path.parts or path.is_absolute() or '..' in path.parts or path.parts[0] in EXCLUDED_ROOTS
            or any(part in EXCLUDED_PARTS for part in path.parts)
            or path.suffix.lower() in ('.sqlite','.sqlite3','.db','.pyc','.pyo'))


class BackupWorker:
    def __init__(self, run, deadline_unix, *, remote='origin', repo_root=ROOT, interval_s=900, clock=time.time):
        self.run=Path(run).resolve();self.repo=Path(repo_root).resolve()
        self.deadline=float(deadline_unix);self.clock=clock
        if not math.isfinite(self.deadline):raise ValueError('Finite absolute backup deadline required')
        self.interval=float(interval_s)
        if not math.isfinite(self.interval) or self.interval<=0:raise ValueError('Positive backup interval required')
        run_id=self.run.name
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,159}',run_id):raise ValueError('Invalid run identifier')
        self.ref='refs/heads/backup/sentry-v6-'+run_id
        self.remote=remote
        self.output=self.run/'github-backups';self.output.mkdir(parents=True,exist_ok=True)
        self.receipt_path=self.output/'receipt.json'
        self.pending_path=self.output/'pending-push.json'
        self.status_path=self.output/'status.json'

    def remaining(self):
        left=self.deadline-self.clock()
        if left<=0:raise TimeoutError('Backup wall-clock deadline expired')
        return left

    def git(self,*args,data=None,env=None,limit=120):
        # Child commands stay in the caller's process group, so the campaign can
        # terminate this worker, Git and SSH together at its absolute cutoff.
        timeout=min(float(limit),self.remaining())
        environment={**os.environ,'GIT_TERMINAL_PROMPT':'0','GIT_SSH_COMMAND':'ssh -o BatchMode=yes -o ConnectTimeout=10',**(env or {})}
        result=subprocess.run(['git',*args],input=data,capture_output=True,cwd=self.repo,
                              env=environment,timeout=timeout,check=False)
        if result.returncode:
            raise RuntimeError('Git '+args[0]+' failed: '+result.stderr.decode(errors='replace')[-1000:])
        return result.stdout

    def status(self,phase,**fields):
        write_json(self.status_path,dict(phase=phase,pid=os.getpid(),utc=utc(),deadline_unix=self.deadline,**fields))

    @contextmanager
    def locked(self):
        with (self.output/'worker.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('Another backup operation owns this run') from None
            try:yield
            finally:fcntl.flock(lock,fcntl.LOCK_UN)

    def remote_tip(self):
        rows=self.git('ls-remote',self.remote,self.ref).split()
        return rows[0].decode() if rows else None

    def parent(self):
        previous=json.loads(self.receipt_path.read_text()) if self.receipt_path.exists() else None
        tip=self.remote_tip()
        pending=json.loads(self.pending_path.read_text()) if self.pending_path.exists() else None
        if pending and tip==pending['commit']:
            # A push may have succeeded just before a prior verification timed out.
            previous={**pending,'verified_utc':utc(),'remote_verified':True}
            write_json(self.receipt_path,previous);self.pending_path.unlink()
        if tip != (previous or {}).get('commit'):
            raise RuntimeError('Remote backup ref changed outside this worker; refusing to overwrite it')
        parent=tip or self.git('rev-parse','HEAD').decode().strip()
        if tip:
            self.git('fetch','--quiet',self.remote,self.ref)
        return parent,tip,previous

    def _source_tree(self, environment):
        self.git('read-tree','--empty',env=environment)
        names=self.git('ls-files','--cached','--others','--exclude-standard','-z').decode().split('\0')
        files=sorted(set(name for name in names if name and not excluded(name)
                         and ((self.repo/name).is_file() or (self.repo/name).is_symlink())))
        for name in files:
            self.remaining()
            if not (self.repo/name).is_symlink() and (self.repo/name).stat().st_size>CHUNK_BYTES:
                raise ValueError('Source file exceeds the 48 MiB backup blob limit: '+name)
        for offset in range(0,len(files),128):
            self.git('add','--force','--',*files[offset:offset+128],env=environment)
        return self.git('write-tree',env=environment).decode().strip(),len(files)

    def generation(self):
        marker=self.run/'main/checkpoint-ready.json'
        if not marker.is_file():return None
        ready=json.loads(marker.read_text())
        relative=Path(ready['directory'])
        directory=(self.run/relative).resolve()
        if relative.is_absolute() or not directory.is_relative_to(self.run/'main/checkpoints') or not directory.is_dir():
            raise ValueError('Checkpoint generation must be within main/checkpoints')
        expected=ready['sha256']
        if not isinstance(expected,dict) or not set(REQUIRED)<=expected.keys():
            raise ValueError('Checkpoint-ready must hash all five published checkpoint files')
        names=sorted(set(expected)|({'coordinator-state.json'} if (directory/'coordinator-state.json').exists() else set()))
        hashes={}
        for name in names:
            self.remaining();path=(directory/name).resolve()
            if Path(name).is_absolute() or not path.is_relative_to(directory) or not path.is_file():
                raise ValueError('Invalid checkpoint generation member')
            if excluded(name):raise ValueError('Database/cache inside checkpoint generation')
            digest=sha(path)
            if name in expected and digest!=expected[name]:raise ValueError('Checkpoint generation SHA mismatch: '+name)
            hashes[name]=digest
        with zipfile.ZipFile(directory/'model.zip') as model:
            if model.testzip() is not None:raise ValueError('Invalid model ZIP')
            data=json.loads(model.read('data'))
            if int(data['num_timesteps'])!=int(ready['samples']):raise ValueError('Checkpoint sample counts do not match')
        for name in ('samplers.json','actor.json','experiment.json'):
            json.loads((directory/name).read_text())
        content=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
        target=self.output/('checkpoint-'+content+'.zip')
        if not target.exists():
            temporary=target.with_suffix('.pending')
            with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_STORED) as archive:
                for name in names:
                    self.remaining();archive.write(directory/name,name)
                archive.writestr('backup-generation.json',json.dumps(dict(**ready,files=hashes,
                    reset_episode=True,discard_partial_rollout=True,bitwise_resume=False),ensure_ascii=False,indent=2))
            if any(sha(directory/name)!=digest for name,digest in hashes.items()):
                temporary.unlink(missing_ok=True);raise ValueError('Published checkpoint generation changed during backup')
            temporary.replace(target)
        with zipfile.ZipFile(target) as archive:
            if archive.testzip() is not None or json.loads(archive.read('backup-generation.json'))['files']!=hashes:
                raise ValueError('Cached checkpoint archive failed integrity verification')
        return dict(archive=target,sha256=sha(target),files=hashes,samples=int(ready['samples']),update_id=ready['update_id'],directory=ready['directory'])

    def _include(self,index_env,name,value):
        if isinstance(value,Path):
            if value.stat().st_size>CHUNK_BYTES:raise ValueError('Unsplit backup blob exceeds 48 MiB')
            blob=self.git('hash-object','-w',str(value),env=index_env).decode().strip()
        else:
            if len(value)>CHUNK_BYTES:raise ValueError('Unsplit backup blob exceeds 48 MiB')
            blob=self.git('hash-object','-w','--stdin',data=value,env=index_env).decode().strip()
        self.git('update-index','--add','--cacheinfo',f'100644,{blob},{name}',env=index_env)

    def _chunks(self,index_env,path,prefix):
        before=path.stat();digest=hashlib.sha256();parts=[]
        with path.open('rb') as stream:
            for number in range((before.st_size+CHUNK_BYTES-1)//CHUNK_BYTES):
                self.remaining();value=stream.read(CHUNK_BYTES);digest.update(value)
                name=f'bundle.part-{number:04d}'
                self._include(index_env,prefix+'/'+name,value)
                parts.append(dict(name=name,bytes=len(value),sha256=hashlib.sha256(value).hexdigest()))
                self.status('hashing_and_chunking',bytes_processed=stream.tell(),total_bytes=before.st_size)
        after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Archive changed while snapshotting')
        info=dict(archive='bundle.zip',bytes=before.st_size,sha256=digest.hexdigest(),parts=parts)
        self._include(index_env,prefix+'/parts.json',json.dumps(info,ensure_ascii=False,indent=2).encode())
        self._include(index_env,prefix+'/SHA256SUMS',(info['sha256']+'  bundle.zip\n').encode())
        return info

    def _reports_helpers(self,environment):
        for name in REPORTS:
            path=self.run/name
            if path.is_file():self._include(environment,'reports/'+name,path)
        manifest_path=self.run/'run-manifest.json'
        manifest=json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        helpers={}
        for role in ('infantry','heavy'):
            row=manifest.get('helpers',{}).get(role)
            path=(self.run/row['path']).resolve() if row else self.run/role/'actor.json'
            if row and (Path(row['path']).is_absolute() or not path.is_relative_to(self.run)):
                raise ValueError('Frozen helper must be a run-relative resource')
            if not path.is_file():
                if row:raise ValueError('Registered frozen helper is missing')
                continue
            digest=sha(path)
            if row and row['sha256']!=digest:raise ValueError('Frozen helper SHA mismatch')
            name=f'helpers/{role}-{digest[:12]}.json'
            self._include(environment,name,path);helpers[role]=dict(path=name,sha256=digest)
        return helpers

    def _publish(self,environment,parent,expected_tip,details,message):
        tree=self.git('write-tree',env=environment).decode().strip()
        commit=self.git('-c','user.name=RMUC RL Backup','-c','user.email=rmuc2027-backup@users.noreply.github.com',
            'commit-tree',tree,'-p',parent,data=message.encode(),env=environment).decode().strip()
        address=self.remote
        if re.fullmatch(r'[A-Za-z0-9_.-]+',address):
            address=self.git('remote','get-url',address).decode().strip()
        github=re.search(r'github\.com[:/]([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?$',address)
        receipt=dict(ref=self.ref,commit=commit,parent=parent,tree=tree,created_utc=utc(),**details)
        if github:receipt['url']='https://github.com/'+github.group(1)+'/commit/'+commit
        write_json(self.pending_path,receipt)
        if self.remote_tip()!=expected_tip:raise RuntimeError('Remote parent changed before push')
        self.status('uploading',commit=commit,**details)
        self.git('-c','pack.threads=1','-c','pack.window=0','push',self.remote,commit+':'+self.ref,env=environment,limit=self.remaining())
        if self.remote_tip()!=commit:raise RuntimeError('Remote verification did not confirm the uploaded commit')
        receipt.update(remote_verified=True,verified_utc=utc())
        write_json(self.receipt_path,receipt);self.pending_path.unlink(missing_ok=True)
        self.status('uploaded_and_verified',**receipt)
        return receipt

    def _backup(self,final=None):
        self.remaining()
        with self.locked(),tempfile.TemporaryDirectory(prefix='rmuc-v6-backup-index-') as folder:
            original_head=self.git('rev-parse','HEAD').decode().strip()
            index_name=self.git('rev-parse','--git-path','index').decode().strip()
            index=Path(index_name) if Path(index_name).is_absolute() else self.repo/index_name
            original_index=sha(index) if index.exists() else None
            try:
                parent,tip,previous=self.parent()
                source_env={'GIT_INDEX_FILE':str(Path(folder)/'source-index')}
                source_tree,count=self._source_tree(source_env)
                environment={'GIT_INDEX_FILE':str(Path(folder)/'artifact-index')}
                self.git('read-tree',parent if tip else '--empty',env=environment)
                # Replace only source/; immutable checkpoint/final directories survive.
                old=self.git('ls-files','-z','source',env=environment)
                if old:self.git('update-index','--force-remove','-z','--stdin',data=old,env=environment)
                self.git('read-tree','--prefix=source/',source_tree,env=environment)
                helpers=self._reports_helpers(environment)
                generation=self.generation()
                generation_public=None
                if generation:
                    prefix='checkpoints/'+generation['sha256']
                    parts=self._chunks(environment,generation['archive'],prefix)
                    generation_public={k:v for k,v in generation.items() if k!='archive'}
                    generation_public.update(path=prefix,parts=parts)
                    self._include(environment,'checkpoints/latest.json',json.dumps(generation_public,ensure_ascii=False,indent=2).encode())
                final_info=None
                if final is not None:
                    final=Path(final).resolve()
                    with zipfile.ZipFile(final) as archive:
                        if any(excluded(name) or any(part=='rmuc2026shuj' for part in Path(name).parts)
                               for name in archive.namelist() if Path(name).suffix.lower() in ('.sqlite','.sqlite3','.db')):
                            raise ValueError('Raw database found in final bundle')
                    checksum=sha(final)
                    report_path=self.run/'package.json'
                    report=json.loads(report_path.read_text()) if report_path.is_file() else {}
                    if report.get('sha256') and checksum!=report['sha256']:raise ValueError('Final ZIP differs from package.json SHA')
                    prefix='final/'+checksum
                    final_info={**self._chunks(environment,final,prefix), 'path':prefix,'original_name':final.name}
                    if final_info['sha256']!=checksum:raise ValueError('Final ZIP changed during upload preparation')
                    self._include(environment,'final/latest.json',json.dumps(final_info,ensure_ascii=False,indent=2).encode())
                campaign_path=self.run/'campaign.json'
                campaign=json.loads(campaign_path.read_text()) if campaign_path.is_file() else {}
                details=dict(run_id=self.run.name,source_tree=source_tree,source_files=count,source_base=original_head,
                    checkpoint=generation_public,helpers=helpers,final=final_info,campaign_status=campaign.get('status','尚未发布阶段报告'),
                    raw_database_included=False,shared_head_unchanged=True,shared_index_unchanged=True)
                readme=('# 哨兵 v6 研究备份\n\n源码位于 source/，保存包含未提交修改的工作树快照；主分支和共享索引保持不变。\n'
                    'checkpoints/latest.json 指向已原子发布的完整代：模型/优化器、RNG、采样状态、actor 和配置。helpers/ 是冻结辅助 actor。\n'
                    '若存在 final/latest.json，它指向最终恢复包的 48 MiB 分片。进入该 path 后重组：\n\n'
                    '```bash\ncat bundle.part-* > bundle.zip\nsha256sum -c SHA256SUMS\n```\n\n'
                    '恢复时开启新 episode，丢弃半个 rollout；不承诺跨机器逐位一致。原始 2026 SQLite 未上传。\n'
                    '训练/验收范围与选用权重以 reports/ 中冻结报告为准；备份成功不等于策略改善、收敛或全矩阵通过。\n'
                    f'本次阶段：{details["campaign_status"]}；完整检查点样本：{generation["samples"] if generation else "尚未发布"}。\n')
                self._include(environment,'README.md',readme.encode())
                self._include(environment,'backup-manifest.json',json.dumps(details,ensure_ascii=False,indent=2).encode())
                return self._publish(environment,parent,tip,details,
                    ('最终备份' if final else '定时备份')+'：哨兵v6源码、完整检查点与阶段证据\n\n'+readme)
            finally:
                if self.git('rev-parse','HEAD').decode().strip()!=original_head or (sha(index) if index.exists() else None)!=original_index:
                    raise RuntimeError('Shared HEAD/index changed during isolated backup; no local reset was performed')

    def snapshot(self):return self._operation()
    def upload_final(self,path):return self._operation(Path(path))

    def _operation(self,final=None):
        try:return self._backup(final)
        except Exception as error:
            self.status('failed_local_artifacts_retained',error=type(error).__name__+': '+str(error))
            raise

    def loop(self):
        while self.clock()<self.deadline:
            started=self.clock()
            try:self.snapshot()
            except Exception as error:print(json.dumps(dict(kind='backup-failed',error=str(error)),ensure_ascii=False),flush=True)
            next_time=started+self.interval
            while self.clock()<min(next_time,self.deadline):time.sleep(min(5.,min(next_time,self.deadline)-self.clock()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--deadline-unix',type=float,required=True)
    parser.add_argument('--remote',default='origin');parser.add_argument('--repo-root',type=Path,default=ROOT)
    choice=parser.add_mutually_exclusive_group();choice.add_argument('--loop',action='store_true');choice.add_argument('--final',type=Path)
    parser.add_argument('--interval-seconds',type=float,default=900)
    args=parser.parse_args()
    try:os.nice(10)
    except OSError:pass
    worker=BackupWorker(args.run,args.deadline_unix,remote=args.remote,repo_root=args.repo_root,interval_s=args.interval_seconds)
    if args.loop:worker.loop()
    else:print(json.dumps(worker.upload_final(args.final) if args.final else worker.snapshot(),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
