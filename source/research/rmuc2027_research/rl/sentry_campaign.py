"""Three new sentry policies under one immutable real-time deadline.

Initialization, public demonstrations, PPO, interventions, frozen trials and
exports all spend the same budget. The last 25 minutes are reserved for trials.
"""
import argparse
import copy
import gzip
import hashlib
import json
import math
import shutil
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from .campaign import write_json,stop_worker


def remaining_deadline(unix):
    return time.monotonic()+max(0.,unix-time.time())


def train_worker(output,model_id,deadline_unix,num_envs=2,resume=None):
    import numpy as np
    import torch
    from .sentry_training import configuration,collect,initialize_actor
    from .learning import TimedPPO,TraceCallback,experiment,vector_env,_actor_parameters
    from .checkpoints import save_checkpoint
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    deadline=remaining_deadline(deadline_unix);started=time.monotonic()
    cfg=configuration(model_id)
    cfg['goal_exploration_std']=.05
    torch.set_num_threads(1);torch.manual_seed(cfg['seed']);torch.use_deterministic_algorithms(True)
    metadata=experiment(cfg)
    if resume:
        prior=json.loads((Path(resume).parent/'experiment.json').read_text())
        if prior.get('sentry_schema')!=metadata['sentry_schema']:
            raise ValueError('Cannot continue a different sentry observation schema')
        metadata['runtime_revisions']=[*prior.get('runtime_revisions',[]),dict(
            parent=str(resume),parent_sha256=hashlib.sha256(Path(resume).read_bytes()).hexdigest(),
            previous_sources=prior.get('sentry_sources'),new_sources=metadata['sentry_sources'],
            previous_sdk_hash=prior.get('sdk_hash'),new_sdk_hash=metadata['sdk_hash'],
            previous_training_sources=prior.get('training_sources'),new_training_sources=metadata['training_sources'],
            rollout_discarded=True,reason='authorized intervention; fresh on-policy rollout')]
    write_json(output/'experiment.json',metadata)
    guidance_parent=Path(resume).parent/'guidance' if resume else None
    reuse=(guidance_parent is not None and (guidance_parent/'demonstrations.npz').exists()
        and prior.get('sdk_hash')==metadata['sdk_hash']
        and prior['training_sources'].get('sentry_training.py')==metadata['training_sources']['sentry_training.py']
        and (guidance_parent/'probes.json').exists()
        and all(not p['metrics']['tipped_seconds'] and not p['metrics']['policy_errors'] for p in json.loads((guidance_parent/'probes.json').read_text())))
    if reuse:
        shutil.copytree(guidance_parent,output/'guidance');data=output/'guidance/demonstrations.npz'
    else:data=collect(cfg,output/'guidance',min(deadline-30,time.monotonic()+min(240.,(deadline-time.monotonic())*.15)),seconds=45)
    runtime=dict(cfg,_episode_log_dir=str(output/'completed-episodes'))
    vec=vector_env(runtime,num_envs);callback=None;model=None
    try:
        if resume:model=TimedPPO.load(resume,env=vec,device='cpu')
        else:model=TimedPPO('MlpPolicy',vec,seed=cfg['seed'],device='cpu',verbose=0,
            policy_kwargs={'net_arch':{'pi':[64,64],'vf':[64,64]},'log_std_init':-1.5},**cfg['ppo'])
        before=_actor_parameters(model)
        warmstart=initialize_actor(model,data,min(deadline-20,time.monotonic()+20)) if data.exists() else dict(samples=0)
        warmstart['demonstrations_reused_from']=str(guidance_parent) if reuse else None
        if data.exists():warmstart['demonstrations_sha256']=hashlib.sha256(data.read_bytes()).hexdigest()
        with torch.no_grad():model.policy.log_std[:2].fill_(math.log(cfg['goal_exploration_std']))
        warmstart['exploration_std']=model.policy.log_std.exp().detach().cpu().tolist()
        write_json(output/'initialization.json',warmstart)
        save_checkpoint(model,output/'initialized.zip')
        callback=TraceCallback(output/'actions.jsonl',time.perf_counter()+max(0.,deadline-time.monotonic()-15),detailed=True)
        callback.checkpoint_seconds=60
        model.learning_rate_control_path=str(output/'learning-rate.json')
        write_json(output/'learning-rate.json',dict(learning_rate=cfg['ppo']['learning_rate'],target_kl=cfg['ppo']['target_kl'],reason='limit actor drift after public-observation initialization'))
        if data.exists():model.sentry_guidance_path=str(data.resolve())
        if hasattr(model,'_sentry_guidance_cache'):del model._sentry_guidance_cache
        model.update_log_path=str(output/'updates.jsonl')
        steps_before=model.num_timesteps;updates_before=model._n_updates
        model.learn(total_timesteps=64*num_envs*100000000,callback=callback,reset_num_timesteps=not bool(resume))
        save_checkpoint(model,output/'model.zip')
        after=_actor_parameters(model)
        change=math.sqrt(sum(float(torch.sum((after[k]-v)**2)) for k,v in before.items()))
        restored=TimedPPO.load(output/'model.zip',device='cpu')
        check=np.zeros(model.observation_space.shape,dtype=np.float32)
        equal=np.array_equal(model.predict(check,deterministic=True)[0],restored.predict(check,deterministic=True)[0])
        report=dict(model=model_id,wall_seconds=time.monotonic()-started,deadline_unix=deadline_unix,
            actual_steps=model.num_timesteps-steps_before,gradient_epochs=model._n_updates-updates_before,
            actor_parameter_l2_change=change,actor_updated=change>0 and math.isfinite(change),
            reload_prediction_equal=bool(equal),optimizer_reloaded=bool(restored.policy.optimizer.state_dict()['state']),
            initialization=warmstart,checkpoint=str(output/'model.zip'))
        write_json(output/'training.json',report);print(json.dumps(report),flush=True)
    except BaseException:
        if model is not None:save_checkpoint(model,output/'interrupted.zip')
        raise
    finally:
        if callback and hasattr(callback,'stream') and not callback.stream.closed:callback.stream.close()
        vec.close()


def evaluate_worker(checkpoint,output,backend,side,seed,deadline_unix,horizon=30):
    import numpy as np
    from .validation import frozen_model
    from .env import SentryEnv
    from .portable import own_robot
    checkpoint=Path(checkpoint);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    model,metadata=frozen_model(checkpoint)
    cfg=copy.deepcopy(metadata['config']);cfg.update(team=side,sim_backend=backend,
        episode_seconds=horizon,alternate_sides=False,controllers={side+':7':{'kind':'learning'}},navigation_precompute=False)
    deadline=remaining_deadline(deadline_unix);started=time.monotonic()
    env=SentryEnv(cfg);steps=0;info=None;tip=0.;guarded=0;violations=0;spun=0;fired=0;path=0.;last=None
    try:
        obs,info=env.reset(seed=seed)
        with gzip.open(output/'trace.jsonl.gz','wt') as stream:
            while time.monotonic()<deadline-3:
                action,_=model.predict(obs,deterministic=True)
                if not np.isfinite(action).all():raise ValueError('Non-finite actor output')
                obs,reward,done,truncated,info=env.step(action);steps+=1
                robot=own_robot(env.session.obs,7);point=(robot['x'],robot['y'])
                if last:path+=math.dist(last,point)
                last=point;tip+=.1*robot.get('tipped',False)
                trace=info['action_trace'].get('execution_trace') or {}
                guarded+=int(trace.get('spin_guard',False))
                accepted=trace.get('accepted',{})
                violations+=int(trace.get('spin_guard') and accepted.get('spin'))
                spun+=int(accepted.get('spin',False));fired+=int(accepted.get('fire',False))
                stream.write(json.dumps(dict(info,robot={k:robot.get(k) for k in ('x','y','z','angular_velocity','transform_progress','grounded')}))+'\n')
                if done or truncated:break
        report=dict(backend=backend,side=side,seed=seed,checkpoint=str(checkpoint),steps=steps,
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),evaluation_sdk_hash=metadata['sdk_hash'],
            wall_seconds=time.monotonic()-started,simulation_seconds=info['time'],requested_horizon=horizon,
            horizon_completed=info['time']>=horizon-1e-8,full_match_completed=info['time']>=420,
            policy_errors=info['metrics']['policy_errors'],tipped_seconds=tip,path_m=path,
            tunnel_guard_samples=guarded,tunnel_spin_violations=violations,spin_samples=spun,fire_samples=fired,
            final_position=last,final=info)
        write_json(output/'result.json',report);print(json.dumps(report),flush=True)
    finally:env.close()


def child(command,log,deadline_unix,report,report_path):
    with Path(log).open('a') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while process.poll() is None:
                request_path=report_path.parent/'intervention.json'
                if report.get('phase')=='training' and request_path.exists():
                    request=json.loads(request_path.read_text())
                    if request.get('model')==report.get('active_model') and request.get('restart') is True:
                        report.setdefault('interventions',[]).append({**request,'applied_unix':time.time()})
                        request_path.rename(report_path.parent/f'intervention-applied-{len(report["interventions"])}.json')
                        stop_worker(process,3.)
                        return -77
                if time.time()>=deadline_unix-5:
                    stop_worker(process,3.);break
                report.update(updated_unix=time.time(),worker_pid=process.pid)
                write_json(report_path,report)
                time.sleep(2.)
        finally:stop_worker(process,0.)
    return process.returncode


def checkpoint_in(folder):
    return next((folder/name for name in ('model.zip','interrupted.zip','latest.zip','initialized.zip') if (folder/name).exists()),None)


def run(output,seconds=7200.,num_envs=2,resume_run=False):
    if not 120<seconds<=7200:raise ValueError('Budget must be at most two real hours')
    output=Path(output).resolve()
    if resume_run:
        report=json.loads((output/'campaign.json').read_text())
        budget=json.loads((output/'budget.json').read_text())
        if any(report[k]!=v for k,v in budget.items()):raise ValueError('Immutable budget changed')
        started=budget['started_unix'];deadline=budget['deadline_unix'];seconds=budget['wall_budget_seconds']
        report.setdefault('recoveries',[]).append(dict(time_unix=time.time(),original_deadline_unix=deadline))
        request_path=output/'intervention.json'
        if request_path.exists():
            request=json.loads(request_path.read_text())
            report.setdefault('interventions',[]).append({**request,'applied_unix':time.time(),'supervisor_recovered':True})
            request_path.rename(output/f'intervention-applied-{len(report["interventions"])}.json')
    else:
        output.mkdir(parents=True,exist_ok=False)
        started=time.time();deadline=started+seconds
    # Small budgets support preflight of the runner itself; real run reserves 25 minutes.
    reserve=min(1500.,seconds*.35)
    train_end=deadline-reserve
    if not resume_run:
        report=dict(started_unix=started,deadline_unix=deadline,gradient_deadline_unix=train_end,
            wall_budget_seconds=seconds,reserve_seconds=reserve,status='training',models=[2771,2772,2773],training=[],evaluations=[])
        write_json(output/'budget.json',{k:report[k] for k in ('started_unix','deadline_unix','gradient_deadline_unix','wall_budget_seconds','reserve_seconds')})
    report_path=output/'campaign.json';write_json(report_path,report)
    from .session import ROOT
    from .sentry_v3 import schema,FEATURES,ACTION_NAMES
    write_json(output/'observation-schema.json',dict(**schema(),actions=ACTION_NAMES,
        features=[dict(index=i,name=n,divisor=s,source=src) for i,(n,s,src) in enumerate(FEATURES)]))
    # Preserve the exact runtime and build inputs before the first initialization.
    snapshot_name=f'recovery-{len(report.get("recoveries",[]))}-source.zip' if resume_run else 'source.zip'
    with zipfile.ZipFile(output/snapshot_name,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for folder in ('packages/core/rmuc2027_core','packages/simulation/src','research/rmuc2027_research','configs'):
            for path in sorted((ROOT/folder).rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py','.rs','.json','.toml'):
                    archive.write(path,path.relative_to(ROOT))
        for name in ('Cargo.toml','Cargo.lock','packages/simulation/Cargo.toml'):
            archive.write(ROOT/name,name)
    checkpoints={row['model']:Path(row['checkpoint']) for row in report['training'] if row.get('checkpoint')}
    completed={row['model'] for row in report['training']}
    recovering=report.get('active_model') if resume_run and report.get('phase')=='training' else None
    recovery_folder=Path(report['active_directory']) if recovering else None
    for index,model in enumerate(report['models']):
        if model in completed:continue
        if time.time()>=train_end:break
        until=(report.get('active_deadline_unix',started+(index+1)*(train_end-started)/3)
            if model==recovering else time.time()+(train_end-time.time())/(3-index))
        report.update(active_model=model,phase='training',active_deadline_unix=until)
        folder=output/str(model)
        recovery_checkpoint=checkpoint_in(recovery_folder) if model==recovering else None
        revision=0
        if recovery_checkpoint:
            revision=1
            while (output/f'{model}-revision-{revision}').exists():revision+=1
            folder=output/f'{model}-revision-{revision}'
        args=[sys.executable,'-u','-m',__package__+'.sentry_campaign','train-worker','--output',str(folder),
            '--model',str(model),'--deadline-unix',str(until),'--num-envs',str(num_envs)]
        if recovery_checkpoint:args+=['--resume',str(recovery_checkpoint)]
        report['active_directory']=str(folder)
        code=child(args,output/f'{folder.name}.log',until,report,report_path)
        while code==-77 and time.time()<until-90:
            previous=checkpoint_in(folder);revision+=1
            if previous is None:break
            folder=output/f'{model}-revision-{revision}'
            args=[sys.executable,'-u','-m',__package__+'.sentry_campaign','train-worker','--output',str(folder),
                '--model',str(model),'--deadline-unix',str(until),'--num-envs',str(num_envs),'--resume',str(previous)]
            report['active_directory']=str(folder)
            code=child(args,output/f'{folder.name}.log',until,report,report_path)
        checkpoint=checkpoint_in(folder)
        if checkpoint:checkpoints[model]=checkpoint
        report['training'].append(dict(model=model,exit_code=code,checkpoint=str(checkpoint) if checkpoint else None))
        write_json(report_path,report)
    report.update(status='evaluating',phase='frozen_trials')
    jobs=[(m,backend,side,seed) for backend in ('warp_kinematic','rapier','mjwarp')
          for m in checkpoints for side,seed in (('red',1001),('blue',1002))]
    for index,(m,backend,side,seed) in enumerate(jobs):
        if time.time()>=deadline-75:break
        until=time.time()+max(1.,(deadline-75-time.time())/(len(jobs)-index))
        folder=output/'evaluation'/f'{m}-{backend}-{side}'
        report.update(active_model=m,active_trial=f'{backend}-{side}-{seed}')
        args=[sys.executable,'-u','-m',__package__+'.sentry_campaign','eval-worker','--output',str(folder),
            '--checkpoint',str(checkpoints[m]),'--backend',backend,'--side',side,'--seed',str(seed),'--deadline-unix',str(until)]
        code=child(args,output/f'eval-{m}-{backend}-{side}.log',until,report,report_path)
        row=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else dict(backend=backend,side=side,seed=seed,error='worker_deadline_or_failure')
        report['evaluations'].append(dict(model=m,exit_code=code,**row));write_json(report_path,report)
    report.update(phase='export',exports=[])
    for m,checkpoint in checkpoints.items():
        if time.time()>=deadline-10:break
        from .exporting import export_model
        folder=output/'exports'/str(m)
        try:
            meta=export_model(checkpoint,folder)
            report['exports'].append(dict(model=m,path=str(folder),version=meta['deployment_version'],parity_error=meta.get('actor_parity_max_error')))
        except Exception as error:report['exports'].append(dict(model=m,error=str(error)))
    report.update(status='complete',phase='finished',finished_unix=time.time(),wall_seconds=time.time()-started)
    report['deadline_respected']=time.time()<=deadline
    report['all_models_have_checkpoints']=len(checkpoints)==3
    report['performance_qualified']=False  # Short frozen trials cannot establish full-match strength.
    write_json(report_path,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('evaluations',)}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('run','train-worker','eval-worker'))
    p.add_argument('--output',required=True);p.add_argument('--wall-seconds',type=float,default=7200)
    p.add_argument('--num-envs',type=int,default=2);p.add_argument('--model',type=int,choices=(2771,2772,2773))
    p.add_argument('--deadline-unix',type=float);p.add_argument('--checkpoint');p.add_argument('--resume')
    p.add_argument('--resume-run',action='store_true',help='Recover supervisor using the existing immutable deadline')
    p.add_argument('--backend',choices=('rapier','warp_kinematic','mjwarp'),default='warp_kinematic')
    p.add_argument('--side',choices=('red','blue'),default='red');p.add_argument('--seed',type=int,default=1001)
    args=p.parse_args()
    if args.mode=='run':run(args.output,args.wall_seconds,args.num_envs,args.resume_run)
    elif args.mode=='train-worker':train_worker(args.output,args.model,args.deadline_unix,args.num_envs,args.resume)
    else:evaluate_worker(args.checkpoint,args.output,args.backend,args.side,args.seed,args.deadline_unix)


if __name__=='__main__':main()
