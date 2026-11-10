"""Two-hour v6 experiment: bounded training, declared pairs, and sealed artifacts.

Every physical operation, including initialization and recovery prefixes, runs
inside the one absolute experiment deadline. Other projects are never signaled.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .campaign import write_json,stop_worker


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def skill_hash(policy):
    value=hashlib.sha256()
    for name,tensor in sorted(policy.state_dict().items()):
        if '.skill.' in name or name=='skill_log_std':
            value.update(name.encode());value.update(tensor.detach().cpu().numpy().tobytes())
    return value.hexdigest()


def checkpoint_counters(path):
    import zipfile
    with zipfile.ZipFile(path) as archive:data=json.loads(archive.read('data'))
    return {key:int(data.get(key,0)) for key in ('num_timesteps','optimizer_steps','completed_updates')}


def publish_checkpoint(root,model,vec,*,baseline=False):
    """Publish a complete immutable generation before changing the ready pointer."""
    from .v6_training import save_state
    from .sentry_v6_policy import export_actor
    from .v6_sampling import TrainingCoordinator
    root=Path(root);folder=root/'main'
    identifier='baseline' if baseline else f'update-{getattr(model,"completed_updates",0):06d}'
    destination=folder/'checkpoints'/identifier
    if destination.exists():
        raise FileExistsError('A completed generation is immutable: '+str(destination))
    destination.mkdir(parents=True)
    save_state(model,vec,destination)
    write_json(destination/'actor.json',export_actor(model.policy))
    shutil.copy2(folder/'experiment.json',destination/'experiment.json')
    coordinator=TrainingCoordinator(root/'sampler-coordinator.json').snapshot()
    write_json(destination/'coordinator-state.json',coordinator)
    names=('model.zip','resume-state.pt','samplers.json','actor.json','experiment.json','coordinator-state.json')
    ready=dict(directory=destination.relative_to(root).as_posix(),samples=model.num_timesteps,
        update_id=getattr(model,'completed_updates',0),optimizer_steps=getattr(model,'optimizer_steps',0),
        skill_sha256=skill_hash(model.policy),sha256={name:sha(destination/name) for name in names})
    write_json(destination/'complete.json',ready)
    for name in ('model.zip','resume-state.pt','samplers.json','actor.json','coordinator-state.json'):
        pending=folder/(name+'.publish.tmp');shutil.copy2(destination/name,pending);os.replace(pending,folder/name)
    if baseline:
        shutil.copy2(destination/'model.zip',folder/'initialized.zip')
        shutil.copy2(destination/'actor.json',folder/'initialized-actor.json')
    write_json(folder/'checkpoint-ready.json',ready)
    return ready


def reconcile_checkpoint(root):
    """The ready generation, not individually replaced aliases, is authoritative."""
    root=Path(root).resolve();folder=root/'main'
    ready=json.loads((folder/'checkpoint-ready.json').read_text())
    generation=(root/ready['directory']).resolve()
    if root not in generation.parents:raise ValueError('Unsafe checkpoint generation path')
    complete=json.loads((generation/'complete.json').read_text())
    if complete!=ready:raise ValueError('Checkpoint generation was not completely published')
    for name,digest in ready['sha256'].items():
        path=(generation/name).resolve()
        if generation not in path.parents or sha(path)!=digest:raise ValueError('Checkpoint generation hash mismatch')
    for name in ('model.zip','resume-state.pt','samplers.json','actor.json','coordinator-state.json'):
        if name not in ready['sha256']:raise ValueError('Missing checkpoint generation member: '+name)
        pending=folder/(name+'.reconcile.tmp');shutil.copy2(generation/name,pending);os.replace(pending,folder/name)
    return ready


def seal_training_report(root):
    """Recover truthful published counters even if the worker died in finally."""
    from .v6_ppo import HybridPPO
    from .v6_sampling import TrainingCoordinator
    import torch
    torch.set_num_threads(1)
    root=Path(root);folder=root/'main'
    ready=reconcile_checkpoint(root)
    baseline=checkpoint_counters(folder/'initialized.zip');candidate=checkpoint_counters(folder/'model.zip')
    before=HybridPPO.load(folder/'initialized.zip',device='cpu')
    after=HybridPPO.load(folder/'model.zip',device='cpu')
    report=json.loads((folder/'training.json').read_text()) if (folder/'training.json').exists() else {}
    report.update(published_accepted_actor_steps=candidate['optimizer_steps']-baseline['optimizer_steps'],
        published_completed_updates=candidate['completed_updates']-baseline['completed_updates'],
        published_samples=candidate['num_timesteps']-baseline['num_timesteps'],
        published_checkpoint_sha256=ready['sha256']['model.zip'],
        published_skill_unchanged=skill_hash(before.policy)==skill_hash(after.policy),
        checkpoint_samples=candidate['num_timesteps'],publication_verified=True)
    if report.get('status')=='not_started' and report['published_samples']>0:
        report['status']='worker_interrupted_report_reconstructed_from_checkpoint'
    write_json(folder/'training.json',report)
    coordinator=TrainingCoordinator(root/'sampler-coordinator.json')
    coordinator.release_interrupted_leases()
    write_json(root/'coverage.json',coordinator.snapshot())
    return ready


def initialize(root,parent):
    import torch
    from stable_baselines3.common.vec_env import DummyVecEnv
    from .env import SentryEnv
    from .session import load_config
    from .learning import experiment
    from .v6_training import configuration,migrate_from_v4
    from .v6_sampling import TrainingCoordinator
    from .capabilities import CapabilitySampler
    from .scenarios import ScenarioSpec
    root,parent=Path(root),Path(parent);torch.set_num_threads(1)
    for role in ('infantry','heavy'):
        (root/role).mkdir()
        for name in ('model.zip','initialized.zip','actor.json','experiment.json','resume-state.pt','samplers.json'):
            shutil.copy2(parent/role/name,root/role/name)
        if (parent/role/'training.json').exists():shutil.copy2(parent/role/'training.json',root/role/'training.json')
    for directory in ('preparation','radar-calibration'):
        shutil.copytree(parent/directory,root/directory)
    folder=root/'main';folder.mkdir()
    cfg=configuration()
    cfg.update(auxiliary_pool={role:str((root/role/'actor.json').resolve()) for role in ('infantry','heavy')},
        radar_classifier_path=str((root/'radar-calibration/classifier.json').resolve()),
        _v6_sampler_path=str((root/'sampler-coordinator.json').resolve()),
        _v6_replay_dir=str((root/'replays/training').resolve()),
        _episode_log_dir=str((folder/'completed-episodes').resolve()),
        sim_backend='warp_kinematic',scenario_sampling=True,capability_sampling=True,
        navigation_precompute=True,action_trace=True)
    cfg=load_config(cfg)
    coordinator=TrainingCoordinator(root/'sampler-coordinator.json',cfg['seed'])
    coordinator.initialize_history(json.loads((parent/'main/samplers.json').read_text()))
    # Spaces and sampler construction only: no physics or environment reset.
    vec=DummyVecEnv([lambda i=i:SentryEnv({**cfg,'_worker_id':i}) for i in range(2)])
    try:
        model,migration=migrate_from_v4(parent/'main/model.zip',vec,cfg,resume_folder=parent/'main')
        write_json(folder/'experiment.json',experiment(cfg))
        migration.update(parent_run=str(parent),helpers_frozen=True,
            helper_hashes={role:sha(root/role/'actor.json') for role in ('infantry','heavy')},
            historical_selected=dict(path=str(parent/'main/selected.zip'),sha256=sha(parent/'main/selected.zip')))
        write_json(folder/'migration.json',migration)
        write_json(root/'initialization.json',migration)
        publish_checkpoint(root,model,vec,baseline=True)
    finally:vec.close()
    (root/'capability-profiles').mkdir()
    for side in ('red','blue'):
        spec=ScenarioSpec(learner_side=side)
        profiles=CapabilitySampler(20261111).sample(spec.lineups(),side)
        write_json(root/'capability-profiles'/f'evaluation-{side}.json',profiles)
    write_json(root/'main/training.json',dict(actual_samples=0,total_samples=model.num_timesteps,
        accepted_actor_steps=0,skill_unchanged=True,status='not_started',demonstration_updates_during_ppo=0))
    refresh_manifest(root)


def refresh_manifest(root):
    from .scenarios import ScenarioSpec
    root=Path(root)
    def resource(path,**kwargs):return dict(path=path.relative_to(root).as_posix(),sha256=sha(path),**kwargs)
    manifest=dict(format='rmuc-research-run-v1',run_id=root.name,title='折叠哨兵 v6：恢复与复活选择',
        runtime_config=resource(root/'main/experiment.json'),checkpoints={},
        helpers={role:resource(root/role/'actor.json') for role in ('infantry','heavy')},
        scenarios={},capability_profiles={},replays={})
    for member,filename in (('baseline','initialized.zip'),('candidate','model.zip')):
        path=root/'main'/filename
        if not path.exists():continue
        digest=sha(path);identifier=member+'-'+digest[:12]
        # The registered checkpoint itself must remain immutable while training.
        frozen=root/'main/catalog'/f'{digest}.zip';frozen.parent.mkdir(exist_ok=True)
        if not frozen.exists():shutil.copy2(path,frozen)
        manifest['checkpoints'][identifier]=resource(frozen,label=member,policy_version=6)
    for side in ('red','blue'):
        spec=ScenarioSpec(learner_side=side)
        manifest['scenarios'][spec.scenario_id]=spec.to_dict()
        manifest['capability_profiles']['evaluation-'+side]=resource(
            root/'capability-profiles'/f'evaluation-{side}.json',label=f'{side} 冻结能力',learner_side=side)
    classifier=root/'radar-calibration/classifier.json'
    if classifier.exists():manifest['radar_classifier']=resource(classifier)
    # Metadata is read incrementally from the canonical header, never loading
    # every full-frame replay simultaneously into memory.
    from rmuc2027_core.replay import ReplayArchive
    for kind in ('training','evaluation'):
        for path in sorted((root/'replays'/kind).glob('*.json')):
            try:
                archive=ReplayArchive(path);metadata=archive.metadata
                identifier=kind+'-'+path.stem
                row=resource(path,kind=kind,label=path.stem,
                    backend=metadata.get('physics_backend',metadata.get('rl',{}).get('config',{}).get('sim_backend','rapier')),
                    truncated=bool(metadata.get('truncated')),seconds=metadata.get('recording_end_time'),
                    recording_end_reason=metadata.get('recording_stop_reason'))
                scenario=metadata.get('rl',{}).get('config',{}).get('scenario')
                if scenario:row['scenario_id']=scenario.get('scenario_id')
                if kind=='evaluation' and metadata.get('checkpoint_member'):
                    member=metadata['checkpoint_member']
                    matching=[key for key,value in manifest['checkpoints'].items() if value['label']==member]
                    if matching:row['checkpoint_id']=matching[0]
                case_id=metadata.get('case_id')
                if case_id:row['case_id']=case_id
                manifest['replays'][identifier]=row
            except Exception as error:
                print(json.dumps(dict(kind='replay_catalog_error',path=str(path),error=str(error))),flush=True)
    write_json(root/'run-manifest.json',manifest)
    return manifest


def train(root,deadline_unix,num_envs):
    import numpy as np
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from .learning import vector_env,trace_json
    from .v6_ppo import HybridPPO
    from .v6_training import restore_state
    from .v6_sampling import TrainingCoordinator
    root=Path(root);folder=root/'main';torch.set_num_threads(1)
    deadline=time.monotonic()+max(0,deadline_unix-time.time())
    cfg=json.loads((folder/'experiment.json').read_text())['config']
    cfg['_training_deadline']=deadline-12
    TrainingCoordinator(root/'sampler-coordinator.json').release_interrupted_leases()
    vec=vector_env(cfg,num_envs)
    model=HybridPPO.load(folder/'model.zip',env=vec,device='cpu')
    restore_state(folder,vec)
    initial_skill={name:value.detach().cpu().clone() for name,value in model.policy.state_dict().items()
                   if '.skill.' in name or name=='skill_log_std'}
    start_steps=model.num_timesteps;start_optimizer=getattr(model,'optimizer_steps',0)
    initial_skill_hash=skill_hash(model.policy)
    started=time.monotonic();start_updates=getattr(model,'completed_updates',0)
    model.training_deadline=deadline-10;model.update_log_path=str(folder/'updates.jsonl')
    last_saved=0.;last_saved_update=start_updates
    vec.env_method('mark_policy_update',start_updates,start_steps)
    def completed(updated,row):
        nonlocal last_saved,last_saved_update
        vec.env_method('mark_policy_update',row['completed_updates'],row['total_samples'])
        if time.monotonic()<deadline-8:
            publish_checkpoint(root,updated,vec)
            last_saved=time.monotonic();last_saved_update=row['completed_updates']
    model.completed_update_hook=completed
    class Callback(BaseCallback):
        def __init__(self):super().__init__();self.last=0.
        def _on_step(self):
            now=time.monotonic()
            for done,info in zip(self.locals['dones'],self.locals['infos']):
                if done:
                    with (folder/'episodes.jsonl').open('a') as stream:stream.write(trace_json(info)+'\n')
            if now-self.last>=20:
                infos=self.locals['infos']
                progress=dict(kind='v6-training-progress',samples=self.model.num_timesteps,
                    actual_samples=self.model.num_timesteps-start_steps,
                    completed_updates=getattr(self.model,'completed_updates',0),
                    accepted_actor_steps=getattr(self.model,'optimizer_steps',0)-start_optimizer,
                    remaining_seconds=max(0,deadline-now),simulation_seconds=[x.get('time') for x in infos],
                    reward_mean=float(np.mean(self.locals['rewards'])))
                write_json(folder/'progress.json',progress);print(json.dumps(progress),flush=True);self.last=now
            return now<deadline-15
    error=None
    try:
        model.learn(total_timesteps=10**12,callback=Callback(),reset_num_timesteps=False)
    except (Exception,KeyboardInterrupt) as exc:
        error=type(exc).__name__+': '+str(exc)
    finally:
        unchanged=all(torch.equal(value,model.policy.state_dict()[name].detach().cpu()) for name,value in initial_skill.items())
        published=json.loads((folder/'checkpoint-ready.json').read_text())
        counters=checkpoint_counters(root/published['directory']/'model.zip')
        if published.get('skill_sha256'):
            published_skill_unchanged=published['skill_sha256']==initial_skill_hash
        else:
            saved=HybridPPO.load(root/published['directory']/'model.zip',device='cpu')
            published_skill_unchanged=skill_hash(saved.policy)==initial_skill_hash
        report=dict(actual_samples=model.num_timesteps-start_steps,total_samples=model.num_timesteps,
            completed_updates=getattr(model,'completed_updates',0)-start_updates,
            accepted_actor_steps=getattr(model,'optimizer_steps',0)-start_optimizer,
            published_accepted_actor_steps=counters['optimizer_steps']-start_optimizer,
            unpublished_accepted_actor_steps=getattr(model,'optimizer_steps',0)-counters['optimizer_steps'],
            unpublished_completed_updates=getattr(model,'completed_updates',0)-counters['completed_updates'],
            unpublished_observed_samples=model.num_timesteps-counters['num_timesteps'],
            published_skill_unchanged=published_skill_unchanged,
            published_checkpoint_sha256=published['sha256']['model.zip'],
            rejected_actor_steps=getattr(model,'rejected_actor_steps',0),
            optimized_samples=getattr(model,'optimized_samples',0),skill_unchanged=unchanged,
            wall_seconds=time.monotonic()-started,demonstration_updates_during_ppo=0,
            partial_rollout_discarded=max(0,model.num_timesteps-getattr(model,'last_completed_samples',model.num_timesteps)),
            status='finished' if error is None else 'interrupted',error=error,
            checkpoint_samples=json.loads((folder/'checkpoint-ready.json').read_text())['samples'],
            checkpoint='main/checkpoint-ready.json',deadline_unix=deadline_unix)
        write_json(folder/'training.json',report);print(json.dumps(dict(kind='v6-training-finished',**report)),flush=True)
        vec.close()


def child(root,command,deadline_unix,label):
    """Only this process group is stopped, never other workloads."""
    root=Path(root);remaining=deadline_unix-time.time()
    if remaining<=2:return dict(exit_code=None,reason='deadline_before_start')
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    args=[sys.executable,'-u','-m',__package__+'.v6_campaign',command,'--output',str(root),*label[1]]
    with (root/(label[0]+'.log')).open('a') as stream:
        process=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        write_json(root/'active-worker.json',dict(pid=process.pid,stage=label[0],deadline_unix=deadline_unix))
        timed_out=False
        try:
            while process.poll() is None:
                if time.time()>=deadline_unix-4:
                    timed_out=True;break
                time.sleep(.25)
        finally:
            stop_worker(process,min(2.,max(0.,deadline_unix-time.time()-1)))
    return dict(exit_code=process.returncode,timed_out=timed_out,log=label[0]+'.log')


def package(root,deadline_unix):
    from rmuc2027_core.paths import ROOT
    from rmuc2027_core.engine import binary_path
    from .v6_bundle import create_bundle,extract_bundle
    from .exporting import export_model
    import tempfile
    root=Path(root)
    selection=json.loads((root/'selection.json').read_text())
    shutil.copy2(root/'main'/selection['selected'],root/'main/selected.zip')
    export_model(root/'main/selected.zip',root/'selected-export')
    database=ROOT/'rmuc2026shuj/rmuc_2026_region_dataset.sqlite'
    report=create_bundle(root,root.with_suffix('.zip'),preparation=root/'preparation',
        external_database=database if database.is_file() else None)
    public={key:value for key,value in report.items() if key!='manifest'}
    write_json(root/'package.json',public)
    with tempfile.TemporaryDirectory(prefix='rmuc-v6-relocation-') as temporary:
        extracted=Path(temporary)/'bundle';extract_bundle(root.with_suffix('.zip'),extracted)
        native=extracted/'source/target/release'/binary_path().name
        native.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(binary_path(),native)
        env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
            'RMUC_RESOURCE_ROOT':str(extracted/'source'),'RMUC_USER_DATA':str(Path(temporary)/'data'),
            'PYTHONPATH':os.pathsep.join(str(extracted/'source'/name) for name in ('packages/core','research','apps/server'))}
        code=('import json,sys; from pathlib import Path; from rmuc2027_research.rl.v6_bundle import preflight; '
              'r=preflight(sys.argv[1]);Path(sys.argv[2]).write_text(json.dumps(r,indent=2));'
              'sys.exit(0 if r["training_ready"] else 1)')
        result=subprocess.run([sys.executable,'-c',code,str(extracted),str(root/'bundle-verification.json')],
            cwd=extracted/'source',env=env,capture_output=True,text=True,timeout=max(1,min(150,deadline_unix-time.time()-5)))
        (root/'bundle-verification.log').write_text(result.stdout+'\n'+result.stderr)
    write_json(root/'release-index.json',dict(bundle=public,relocation_exit_code=result.returncode,
        verified_after_relocation=result.returncode==0,target_5600x_5070ti_verified=False,
        results_sha256=sha(root/'run-results.json'),source='sealed bundle source, not mutable working tree'))
    if result.returncode:raise RuntimeError('Relocated package preflight failed')


def run(root,parent=None,seconds=7200,resume_bundle=None):
    from .v6_evaluation import required_cases,select_candidate
    root=Path(root).resolve();parent=Path(parent).resolve() if parent is not None else None
    resume_bundle=Path(resume_bundle).resolve() if resume_bundle is not None else None
    if (parent is None)==(resume_bundle is None):raise ValueError('Choose exactly one v4 parent or v6 resume bundle')
    if seconds!=7200:raise ValueError('This authorized campaign is a fixed two-hour experiment')
    root.mkdir(parents=True,exist_ok=False)
    started=time.time();deadline=started+seconds
    budget=dict(started_unix=started,deadline_unix=deadline,wall_budget_seconds=seconds,
        initialization_deadline_unix=started+1200,evaluation_deadline_unix=started+6000,
        web_deadline_unix=started+6480,other_workloads_may_continue=True,
        scope='initialization, physics warmup, prefixes, PPO, evaluation, web acceptance, packaging and backup')
    write_json(root/'budget.json',budget)
    cases=required_cases();write_json(root/'required-cases.json',cases)
    report=dict(status='initializing',budget=budget,parent_run=str(parent) if parent else None,
        resume_bundle=str(resume_bundle) if resume_bundle else None,stages=[],evaluations=[])
    write_json(root/'campaign.json',report)
    initialization=child(root,'initialize-resume' if resume_bundle else 'initialize',started+420,
        ('initialize',['--resume-bundle',str(resume_bundle)] if resume_bundle else ['--parent',str(parent)]))
    report['stages'].append(dict(stage='initialize',**initialization))
    if initialization['exit_code']!=0:
        report.update(status='failed_initialization',finished_unix=time.time())
        write_json(root/'campaign.json',report);return report
    backup=None
    backup_log=(root/'backup.log').open('a')
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    backup=subprocess.Popen([sys.executable,'-u','-m',__package__+'.v6_backup','--run',str(root),
        '--deadline-unix',str(deadline-10),'--loop'],stdout=backup_log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    try:
        report['status']='preflight';write_json(root/'campaign.json',report)
        checked=child(root,'preflight',started+1200,('preflight',['--deadline-unix',str(started+1195)]))
        report['stages'].append(dict(stage='preflight',**checked))
        preflight=json.loads((root/'preflight.json').read_text()) if (root/'preflight.json').exists() else {}
        rates=preflight.get('evaluation_rates',{})
        ready=bool(preflight.get('training_ready'))
        if ready:
            # Prefixes are replayed physically for both members, never restored
            # by teleporting internal state. Reserve their declared worst case.
            estimate=1.25*(240/rates['warp_kinematic']+(500+420)/rates['rapier'])+120
        else:estimate=4800.
        reserve=max(2100.,estimate)
        train_seconds=max(0.,min(2700.,started+6000-time.time()-reserve)) if ready else 0.
        train_deadline=time.time()+train_seconds
        report['allocation']=dict(estimated_evaluation_seconds=estimate,training_seconds=train_seconds,
            evaluation_reserve_seconds=reserve,num_envs=preflight.get('selected_num_envs',1))
        if train_seconds>=30:
            report['status']='training';write_json(root/'campaign.json',report)
            result=child(root,'train',train_deadline,('train',['--deadline-unix',str(train_deadline-2),
                '--num-envs',str(preflight.get('selected_num_envs',1))]))
            report['stages'].append(dict(stage='train',**result))
        else:
            report['stages'].append(dict(stage='train',skipped=True,reason='insufficient_time_after_required_evaluation_reserve'))
        report['published_checkpoint']=seal_training_report(root)
        report['status']='evaluating';write_json(root/'campaign.json',report)
        # Equal fractions of estimated work preserve whole cases when possible;
        # a timeout remains a missing/censored required trial, never a pass.
        jobs=[(case,member) for case in cases for member in ('baseline','candidate')]
        for index,(case,member) in enumerate(jobs):
            available=started+6000-time.time()
            if available<=10:break
            def weight(c):return (c['seconds']+(60 if c['recovery_kind'] in ('free_ready','alive_weak') else 45 if c['recovery_kind'] else 0))/max(.001,rates.get(c['backend'],.1))
            remaining=sum(weight(c) for c,_ in jobs[index:])
            until=min(started+6000,time.time()+available*weight(case)/remaining)
            label=case['id']+'-'+member
            result=child(root,'evaluate',until,(label,['--case',case['id'],'--member',member,'--deadline-unix',str(until-2)]))
            report['evaluations'].append(dict(case_id=case['id'],member=member,**result))
            write_json(root/'campaign.json',report)
        report['selection']=select_candidate(root)
        refresh_manifest(root)
        report['status']='web_acceptance';write_json(root/'campaign.json',report)
        # An external browser verifier may consume this bounded acceptance slot.
        # The marker lets the supervising agent inspect the exact recorded run.
        write_json(root/'web-acceptance-request.json',dict(run_id=root.name,
            deadline_unix=started+6480,required=['frozen_live_match','saved_training_or_evaluation_replay']))
        while time.time()<started+6480-5 and not (root/'web-acceptance.json').exists():
            time.sleep(.5)
        web=json.loads((root/'web-acceptance.json').read_text()) if (root/'web-acceptance.json').exists() else dict(status='not_completed')
        report['web_acceptance']=web
        report['status']='results_frozen';report['results_frozen_unix']=time.time()
        report['training']=json.loads((root/'main/training.json').read_text())
        report['limitations']=['finite paired cases only','no MJWarp this round','target computer not connected']
        write_json(root/'run-results.json',report);write_json(root/'campaign.json',report)
        packaged=child(root,'package',deadline-100,('package',['--deadline-unix',str(deadline-105)]))
        report['package']=packaged
        if backup and backup.poll() is None:
            os.killpg(backup.pid,signal.SIGTERM)
            try:backup.wait(timeout=2)
            except subprocess.TimeoutExpired:os.killpg(backup.pid,signal.SIGKILL);backup.wait()
        if packaged['exit_code']==0 and time.time()<deadline-15:
            with (root/'final-backup.log').open('a') as log:
                upload=subprocess.Popen([sys.executable,'-u','-m',__package__+'.v6_backup','--run',str(root),
                    '--deadline-unix',str(deadline-3),'--final',str(root.with_suffix('.zip'))],
                    stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                try:upload.wait(timeout=max(1,deadline-time.time()-2))
                except subprocess.TimeoutExpired:os.killpg(upload.pid,signal.SIGKILL);upload.wait()
                report['final_backup_exit_code']=upload.returncode
        report.update(status='completed' if packaged['exit_code']==0 and len(report['evaluations'])==16
            and not report['selection']['missing'] and web.get('status')=='passed' else 'completed_partial',
            finished_unix=time.time(),elapsed_seconds=time.time()-started,within_wall_budget=time.time()<=deadline)
        # Post-package release bookkeeping stays outside the sealed result file.
        write_json(root/'campaign-final.json',report)
        print(json.dumps(report),flush=True)
        return report
    finally:
        if backup and backup.poll() is None:
            os.killpg(backup.pid,signal.SIGTERM)
            try:backup.wait(timeout=2)
            except subprocess.TimeoutExpired:os.killpg(backup.pid,signal.SIGKILL);backup.wait()
        backup_log.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('run','initialize','initialize-resume','preflight','train','evaluate','package'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--parent',type=Path)
    parser.add_argument('--resume-bundle',type=Path)
    parser.add_argument('--deadline-unix',type=float)
    parser.add_argument('--num-envs',type=int,default=2)
    parser.add_argument('--case');parser.add_argument('--member',choices=('baseline','candidate'))
    args=parser.parse_args()
    if args.command=='run':run(args.output,args.parent,resume_bundle=args.resume_bundle)
    elif args.command=='initialize':initialize(args.output,args.parent)
    elif args.command=='initialize-resume':
        from .v6_bundle import prepare_resume_run
        prepare_resume_run(args.resume_bundle,args.output)
        refresh_manifest(args.output)
    elif args.command=='preflight':
        from .v6_preflight import preflight
        preflight(args.output,args.deadline_unix)
    elif args.command=='train':train(args.output,args.deadline_unix,args.num_envs)
    elif args.command=='evaluate':
        from .v6_evaluation import required_cases,trial
        trial(args.output,next(c for c in required_cases() if c['id']==args.case),args.member,args.deadline_unix)
    elif args.command=='package':package(args.output,args.deadline_unix)


if __name__=='__main__':main()
