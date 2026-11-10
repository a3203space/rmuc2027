"""Bounded v4 experiment: three resumable policies and paired backend trials.

Usage: python -m rmuc2027_research.rl.v4_campaign run --output PATH
The budget is wall time, shared by initialization, PPO, evaluation and packaging.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .campaign import write_json,stop_worker


def monotonic_deadline(unix):return time.monotonic()+max(0.,unix-time.time())


def pool_config(root):
    return {role:str((Path(root)/role/'actor.json').resolve()) for role in ('infantry','heavy')
            if (Path(root)/role/'actor.json').exists()}


def actor_export(model,role,folder):
    if role=='main':
        from .sentry_v4_policy import export_actor
        actor=export_actor(model.policy)
    else:
        from .aux_policy import export_actor
        actor=export_actor(model.policy,family=role)
    write_json(Path(folder)/'actor.json',actor)


def fit_radar(root,destination='radar-calibration'):
    from .radar_calibration import save_classifier
    root=Path(root);rows=[]
    for role in ('main','infantry','heavy'):
        for name in ('guidance/radar-calibration-rows.json','training-radar-calibration-rows.json'):
            path=root/role/name
            if path.exists():rows.extend(json.loads(path.read_text()))
    return save_classifier(rows,root/destination)


def model_class(role):
    if role=='main':
        from .asymmetric_ppo import AsymmetricPPO
        from .sentry_v4_policy import SentryV4Policy
        return AsymmetricPPO,SentryV4Policy
    from stable_baselines3 import PPO
    from .aux_policy import AuxLayerNormPolicy
    return PPO,AuxLayerNormPolicy


def preflight_worker(root,deadline_unix,resume_from=None):
    import numpy as np
    import torch
    import warp as wp
    from .v4_training import configuration
    from .learning import vector_env
    from .portable import policy_schema
    from importlib.metadata import version,distributions
    import platform
    root=Path(root);deadline=monotonic_deadline(deadline_unix);rows=[]
    installed=[];editable=[]
    for distribution in distributions():
        direct=json.loads(distribution.read_text('direct_url.json') or '{}')
        name=distribution.metadata.get('Name')
        if not name:continue
        if direct.get('dir_info',{}).get('editable'):
            editable.append(name);continue
        installed.append(f'{name}=={distribution.version}')
    (root/'requirements-runtime.txt').write_text('\n'.join(sorted(set(installed),key=str.lower))+'\n')
    write_json(root/'runtime-environment.json',dict(python=platform.python_version(),platform=platform.platform(),
        editable_source_packages=sorted(editable),policy_device='cpu',freeze_file='requirements-runtime.txt',
        wheelhouse_embedded=False))
    torch.set_num_threads(1)
    benchmark_model=None
    benchmark=dict(benchmark_kind='scripted_environment_only',benchmark_measured_steps=8,
        benchmark_warmup_steps=1,benchmark_includes_ppo_updates=False,
        benchmark_scope='eight opening steps per environment; not full-match or optimization throughput',
        benchmark_sources={})
    if resume_from is not None:
        from .v4_bundle import verify_bundle,load_runtime_config
        from .asymmetric_ppo import AsymmetricPPO
        bundle=Path(resume_from).resolve();verify_bundle(bundle)
        base_cfg=load_runtime_config(bundle,'main')
        if set(base_cfg.get('auxiliary_pool',{}))!={'infantry','heavy'}:
            raise ValueError('Resume preflight requires both frozen auxiliary actors')
        checkpoint=bundle/'models/main/model.zip'
        benchmark_model=AsymmetricPPO.load(checkpoint,device='cpu')
        benchmark.update(benchmark_kind='frozen_main_and_auxiliary_rollout',
            benchmark_sources=dict(resume_bundle=str(bundle),main_checkpoint=dict(path=str(checkpoint),
                sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()),
                helpers={role:dict(path=str(path),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
                         for role,path in base_cfg['auxiliary_pool'].items()}))
    else:base_cfg=configuration()

    def actions_for(observation,count):
        if benchmark_model is None:return np.zeros((count,policy_schema(base_cfg)['action_size']),dtype=np.float32)
        with torch.no_grad():actions,_=benchmark_model.predict(observation,deterministic=False)
        if not np.isfinite(actions).all():raise ValueError('Nonfinite resumed actor action')
        return actions

    devices=dict(policy_device='cpu',torch_cuda_available=torch.cuda.is_available(),torch_cuda_devices=[],
                 warp_cuda_available=False,warp_cuda_devices=[],physics_device=None,gpu_name=None)
    devices.update(benchmark)
    if devices['torch_cuda_available']:
        for index in range(torch.cuda.device_count()):
            free,total=torch.cuda.mem_get_info(index)
            devices['torch_cuda_devices'].append(dict(index=index,name=torch.cuda.get_device_name(index),
                free_bytes=int(free),memory_bytes=int(total),capability=list(torch.cuda.get_device_capability(index))))
    try:
        wp.init()
        for device in wp.get_cuda_devices():
            devices['warp_cuda_devices'].append(dict(device=str(device),name=device.name,architecture=device.arch,
                is_cuda=bool(device.is_cuda),free_bytes=int(device.free_memory),memory_bytes=int(device.total_memory)))
        selected=wp.get_device(base_cfg.get('sim_device','cuda:0'))
        devices.update(warp_cuda_available=bool(selected.is_cuda),physics_device=str(selected),gpu_name=selected.name)
        if not selected.is_cuda:raise RuntimeError('The configured Warp physics device is not CUDA')
    except Exception as exc:
        report=dict(**devices,results=rows,gradient_updates=0,training_ready=False,error=str(exc))
        write_json(root/'preflight.json',report)
        raise RuntimeError('Warp CUDA physics is unavailable; CPU Torch is supported but physics cannot fall back') from exc
    for count in (2,4,8):
        if time.monotonic()>deadline-20:break
        cfg=copy.deepcopy(base_cfg);cfg.update(action_trace=False)
        if resume_from is None:cfg['episode_seconds']=420
        started=time.monotonic();vec=None
        try:
            vec=vector_env(cfg,count);obs=vec.reset();init=time.monotonic()-started
            obs,_,_,_=vec.step(actions_for(obs,count));begin=time.monotonic();steps=0
            for _ in range(8):
                if time.monotonic()>deadline-5:break
                obs,reward,done,infos=vec.step(actions_for(obs,count));steps+=count
                if not all(np.isfinite(v).all() for v in obs.values()) or not np.isfinite(reward).all():raise ValueError('Nonfinite preflight')
            # The actor/critic use CPU Torch. GPU allocation belongs to Warp's
            # physics context and must be measured through that device.
            free,total=int(selected.free_memory),int(selected.total_memory)
            rows.append(dict(num_envs=count,initialization_seconds=init,samples_per_second=steps/max(.001,time.monotonic()-begin),
                             samples=steps,gpu_free_bytes=free,gpu_total_bytes=total,
                             physics_device=str(selected),gpu_memory_source='warp',success=bool(steps)))
        except Exception as exc:rows.append(dict(num_envs=count,success=False,error=str(exc)))
        finally:
            if vec is not None:vec.close()
        write_json(root/'preflight.json',dict(**devices,results=rows,gradient_updates=0,training_ready=False))
    eligible=[r for r in rows if r.get('success') and r.get('gpu_total_bytes',0)>0 and r['gpu_free_bytes']>1024**3]
    if not eligible:
        write_json(root/'preflight.json',dict(**devices,results=rows,gradient_updates=0,training_ready=False,
            error='No valid vector configuration has the required Warp GPU memory headroom'))
        raise RuntimeError('No valid vector configuration passed preflight')
    best=max(eligible,key=lambda r:r['samples_per_second'])
    report=dict(**devices,results=rows,recommended_num_envs=best['num_envs'],gradient_updates=0,training_ready=True,
        dependencies={name:version(name) for name in ('torch','numpy','stable-baselines3','warp-lang','mujoco','mujoco-warp')})
    write_json(root/'preflight.json',report);print(json.dumps(report),flush=True)


def initialization(root,deadline_unix):
    import torch
    from .v4_training import configuration,collect_main,initialize_main,save_state
    from .learning import experiment,vector_env
    from .aux_training import collect_demonstrations,behavior_clone,pretrain_human_representation
    from .env import SentryEnv
    root=Path(root);deadline=monotonic_deadline(deadline_unix)
    reports=[];torch.set_num_threads(1)
    for role in ('infantry','heavy','main'):
        folder=root/role;folder.mkdir(parents=True,exist_ok=True)
        cfg=configuration(role);cfg['auxiliary_pool']=pool_config(root)
        # Initialization is shared fairly, leaving at least one stage for every role.
        available=max(1.,deadline-time.monotonic())
        stage_end=time.monotonic()+available/(3-len(reports))
        torch.manual_seed(cfg['seed']+len(reports));vec=vector_env(cfg,1)
        algorithm,policy=model_class(role);model=None
        try:
            model=algorithm(policy,vec,device='cpu',seed=cfg['seed'],verbose=0,**cfg['ppo'])
            write_json(folder/'experiment.json',experiment(cfg))
            if role=='main':
                data=collect_main(cfg,folder/'guidance',stage_end-25)
                report=initialize_main(model,data,stage_end-5)
            else:
                offline=pretrain_human_representation(model,root/'preparation',family=role,deadline=min(stage_end-10,time.monotonic()+45))
                demo_env=SentryEnv(cfg)
                try:demos=collect_demonstrations(demo_env,deadline=stage_end-25,episode_seconds=45.)
                finally:demo_env.close()
                import numpy as np
                (folder/'guidance').mkdir(exist_ok=True)
                np.savez_compressed(folder/'guidance/demonstrations.npz',**demos['observations'],actions=demos['actions'],head_mask=demos['head_mask'])
                write_json(folder/'guidance/provenance.json',demos['provenance'])
                write_json(folder/'guidance/radar-calibration-rows.json',demos.get('radar_calibration_rows',[]))
                report=behavior_clone(model,demos,deadline=stage_end-5,epochs=20)
                report['offline_representation']=offline
            actor_export(model,role,folder)
            save_state(model,vec,folder,'initialized.zip')
            shutil.copyfile(folder/'initialized.zip',folder/'model.zip')
            shutil.copyfile(folder/'actor.json',folder/'initialized-actor.json')
            write_json(folder/'initialization.json',report)
            reports.append(dict(role=role,**report))
            print(json.dumps({'kind':'initialization-complete','role':role,'report':report}),flush=True)
        finally:vec.close()
    write_json(root/'initialization.json',reports)
    fit_radar(root)


def train_stage(root,role,deadline_unix,num_envs):
    import numpy as np
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from .v4_training import save_state,restore_state
    from .learning import experiment,vector_env,trace_json
    root=Path(root);folder=root/role;deadline=monotonic_deadline(deadline_unix)
    torch.set_num_threads(1)
    metadata=json.loads((folder/'experiment.json').read_text());cfg=metadata['config']
    cfg['auxiliary_pool']=pool_config(root)
    if (root/'radar-calibration/classifier.json').exists():cfg['radar_classifier_path']=str(root/'radar-calibration/classifier.json')
    cfg['_episode_log_dir']=str(folder/'completed-episodes')
    cfg['_training_deadline']=deadline-10
    vec=vector_env(cfg,num_envs);algorithm,_=model_class(role)
    model=algorithm.load(folder/'model.zip',env=vec,device='cpu')
    resume_bundle=metadata.get('resume_bundle')
    if resume_bundle:
        from .v4_bundle import relocate_model_runtime
        relocate_model_runtime(model,resume_bundle,role,output_dir=folder)
    restore_state(folder,vec)
    metadata=experiment({k:v for k,v in cfg.items() if not k.startswith('_')})
    write_json(folder/'experiment.json',metadata)
    start_steps=model.num_timesteps;start_epochs=model._n_updates;started=time.monotonic()
    model.last_completed_samples=model.num_timesteps
    initial={name:value.detach().cpu().clone() for name,value in model.policy.named_parameters()}
    actor_ids={id(p) for p in model.policy.actor_parameters()} if role=='main' else None

    class Callback(BaseCallback):
        def __init__(self):super().__init__();self.last=0.;self.last_saved=0.
        def _on_step(self):
            now=time.monotonic()
            for done,info in zip(self.locals['dones'],self.locals['infos']):
                if done:
                    with (folder/'episodes.jsonl').open('a') as stream:stream.write(trace_json(info)+'\n')
            if now-self.last>20:
                infos=self.locals['infos']
                print(json.dumps(dict(kind='v4-training-progress',role=role,samples=self.model.num_timesteps,
                    gradient_epochs=self.model._n_updates,elapsed=now-started,remaining=deadline-now,
                    simulation_seconds=[info.get('time') for info in infos],
                    reward_mean=float(np.mean(self.locals['rewards'])),
                    policy_errors=sum(info.get('metrics',{}).get('policy_errors',0) for info in infos),
                    tipped_seconds_max=max((info.get('metrics',{}).get('tipped_seconds',0) for info in infos),default=0),
                    action_means=np.asarray(self.locals['clipped_actions']).mean(axis=0).tolist() if role=='main' else None)),flush=True)
                self.last=now
            return now<deadline-15
        def _on_rollout_start(self):
            # Called after the previous PPO update. Save only completed updates.
            if self.model._n_updates>start_epochs:self.model.last_completed_samples=self.model.num_timesteps
            if self.model._n_updates>start_epochs and time.monotonic()-self.last_saved>60:
                save_state(self.model,vec,folder)
                self.last_saved=time.monotonic()

    model.update_log_path=str(folder/'updates.jsonl')
    try:
        model.learn(total_timesteps=10**12,callback=Callback(),reset_num_timesteps=False)
    finally:
        save_state(model,vec,folder)
        actor_export(model,role,folder)
        change=sum(float(torch.sum((value.detach().cpu()-initial[name])**2)) for name,value in model.policy.named_parameters())**.5
        actor_change=(sum(float(torch.sum((value.detach().cpu()-initial[name])**2))
            for name,value in model.policy.named_parameters() if id(value) in actor_ids)**.5) if actor_ids is not None else None
        write_json(folder/'training.json',dict(role=role,actual_samples=model.num_timesteps-start_steps,
            total_samples=model.num_timesteps,gradient_epochs=model._n_updates-start_epochs,
            optimized_samples=getattr(model,'optimized_samples',None),parameter_l2_change=change,
            actor_parameter_l2_change=actor_change,optimizer_steps=getattr(model,'optimizer_steps',None),
            wall_seconds=time.monotonic()-started,deadline_unix=deadline_unix,
            demonstration_updates_during_ppo=0,checkpoint='model.zip'))
        vec.close()


def _evaluation_diagnostics(legal_obs,info,private_frame,robot):
    """Copy post-step diagnostics into a trace only, never into observations.

    RFID can retain an earlier contact; private occupants/raw contacts remain
    explicitly separate so offline analysis can identify the actual cause.
    """
    def fields(value,names):return {name:copy.deepcopy(value[name]) for name in names if name in value}
    referee=legal_obs.get('referee',{});sources=referee.get('sources',{})
    raw=next((r for r in private_frame.get('robots',[]) if r.get('team')==robot.get('team')
              and r.get('slot')==robot.get('slot')), {})
    return dict(diagnostics_version=1,
        legal_self=fields(robot,('id','team','slot','vx','vy','vz','quaternion','angular_velocity',
            'grounded','upright','tipped','alive','powered','transform_progress','rfid')),
        legal_referee=dict(sources={key:fields(sources.get(key,{}),('valid','age','source_time','receipt_time'))
            for key in ('0x0209','0x0201','0x0101')},control=copy.deepcopy(referee.get('control'))),
        evaluation_metrics=dict(capture_seconds=info.get('metrics',{}).get('capture_seconds'),
            control=copy.deepcopy(info.get('control')),reward_terms=copy.deepcopy(info.get('reward_terms'))),
        execution=fields(info.get('action_trace',{}),('execution_trace','executed_action')),
        evaluation_ground_truth=dict(scope='post_step_trace_only_not_actor_input',time=private_frame.get('time'),
            robot=fields(raw,('weak','invulnerable','zone_contacts','last_raw_zones')),
            zones={key:fields(value,('owner','progress','occupants','contested'))
                   for key,value in private_frame.get('zones',{}).items()},
            positions={key:fields(value,('owner','kind','zones','control','controller','progress','active'))
                       for key,value in private_frame.get('positions',{}).items()}))


def evaluate_pair(root,backend,side,deadline_unix,reuse_anchor_from=None):
    import numpy as np
    from .env import SentryEnv
    from .scenarios import ScenarioSpec
    from .portable import own_robot
    from .asymmetric_ppo import AsymmetricPPO
    from .v4_evaluation import detect_tunnel_passages
    root=Path(root);folder=root/'main';deadline=monotonic_deadline(deadline_unix)
    cfg=json.loads((folder/'experiment.json').read_text())['config']
    cfg.update(sim_backend=backend,episode_seconds=60,scenario_sampling=False,evaluation=True,
        scenario=ScenarioSpec(learner_side=side).to_dict(),navigation_precompute=False,
        auxiliary_pool=pool_config(root),action_trace=True)
    reports=[]
    if reuse_anchor_from is not None:
        from .v4_anchor_reuse import validate_anchor_reuse
        cfg['v4_capture_reward']=0.
        anchor,anchor_trace=validate_anchor_reuse(root,reuse_anchor_from,backend,side)
        (root/'evaluation'/f'{backend}-{side}-initialized.zip.jsonl').write_bytes(anchor_trace)
        reports.append(anchor)
    for member,checkpoint in enumerate(('initialized.zip','model.zip')):
        if member==0 and reuse_anchor_from is not None:continue
        member_deadline=time.monotonic()+max(0.,deadline-time.monotonic())/(2-member)
        model=AsymmetricPPO.load(folder/checkpoint,device='cpu')
        env=SentryEnv(cfg);started=time.monotonic();steps=folds=0;max_fold=0.;info=None;trace=[]
        try:
            obs,info=env.reset(seed=81001 if side=='red' else 81002)
            robot=own_robot(env.session.obs,7)
            with (root/'evaluation'/f'{backend}-{side}-{checkpoint}.jsonl').open('w') as stream:
                while time.monotonic()<member_deadline-5:
                    action,_=model.predict(obs,deterministic=True)
                    if not np.isfinite(action).all():raise ValueError('Nonfinite frozen action')
                    obs,reward,done,truncated,info=env.step(action);steps+=1
                    robot=own_robot(env.session.obs,7)
                    max_fold=max(max_fold,robot.get('transform_progress',0.))
                    folds+=int(action[2]>.3)
                    frame=dict(time=info['time'],position=[robot[k] for k in ('x','y','z')],
                        fold=robot.get('transform_progress'),action=action.tolist(),navigation=info['action_trace']['navigation'])
                    frame.update(_evaluation_diagnostics(env.session.obs,info,env.session.battle.match._frame,robot))
                    trace.append(frame);stream.write(json.dumps(frame)+'\n')
                    if done or truncated:break
            passage=detect_tunnel_passages(trace)
            row=dict(backend=backend,side=side,checkpoint=checkpoint,steps=steps,simulation_seconds=info['time'],
                horizon_completed=info['time']>=60-1e-8,censored=info['time']<60-1e-8,full_match=False,metrics=info['metrics'],
                control=info['control'],fold_requests=folds,max_fold=max_fold,
                passed_tunnel=passage['passed_tunnel'],passed_tunnel_evidence=passage,
                final_position=[robot[k] for k in ('x','y','z')],wall_seconds=time.monotonic()-started)
            if reuse_anchor_from is not None:row['reused']=False
            reports.append(row);print(json.dumps({'kind':'paired-trial',**row}),flush=True)
        finally:env.close()
    write_json(root/'evaluation'/f'{backend}-{side}.json',reports)


def select_candidate(root):
    root=Path(root);pairs=[]
    for path in sorted((root/'evaluation').glob('*.json')):
        rows=json.loads(path.read_text())
        if len(rows)==2 and all(r.get('horizon_completed') for r in rows):pairs.append(rows)
    improved=False
    if len(pairs)==6:
        def score(index):
            rows=[p[index] for p in pairs]
            return (sum(r['passed_tunnel'] for r in rows),sum(r['control'][r['side']]-r['control']['blue' if r['side']=='red' else 'red'] for r in rows))
        anchor,candidate=score(0),score(1)
        clean=all(p[1]['metrics']['policy_errors']==0 and p[1]['metrics'].get('tipped_seconds',0)<=p[0]['metrics'].get('tipped_seconds',0)+1e-6 for p in pairs)
        improved=clean and candidate[0]>=anchor[0] and candidate[1]>=anchor[1]-1e-6 and (candidate[0]>anchor[0] or candidate[1]>anchor[1]+1e-6)
    selected='model.zip' if improved else 'initialized.zip'
    if (root/'main'/selected).exists():shutil.copyfile(root/'main'/selected,root/'main/selected.zip')
    report=dict(selected=selected,ppo_improvement_verified=improved,complete_pairs=len(pairs),
                resume_default='model.zip',reason='paired_improvement' if improved else 'keep_demonstration_anchor_pending_evidence')
    write_json(root/'selection.json',report)
    return report


def package(root):
    from rmuc2027_core.paths import ROOT
    from rmuc2027_core.engine import binary_path
    from .v4_bundle import create_bundle,extract_bundle
    from .exporting import export_model
    import tempfile
    root=Path(root).resolve()
    renderer=root/'report-tools/render_review.py'
    if renderer.exists():
        import importlib.util
        renderer_python=sys.executable if importlib.util.find_spec('matplotlib') else str(Path(sys.base_prefix)/'bin/python3')
        with (root/'review-render.log').open('w') as stream:
            try:
                rendered=subprocess.run([renderer_python,str(renderer),'--allow-partial'],stdout=stream,stderr=subprocess.STDOUT,timeout=60)
                render_report=dict(exit_code=rendered.returncode)
            except (OSError,subprocess.TimeoutExpired) as exc:render_report=dict(error=str(exc))
        write_json(root/'review-render.json',dict(**render_report,log='review-render.log',python=renderer_python))
    if (root/'main/selected.zip').exists():export_model(root/'main/selected.zip',root/'selected-export')
    database=ROOT/'rmuc2026shuj/rmuc_2026_region_dataset.sqlite'
    create_bundle(root,root.with_suffix('.zip'),preparation=root/'preparation',
                  external_database=database if database.is_file() else None)
    # Exercise the actual archived Python/source/resources from another path.
    # Reuse only this host's matching Rust executable; the portable archive
    # itself excludes binaries and requires a build on the destination.
    with tempfile.TemporaryDirectory(prefix='rmuc-v4-bundle-check-') as temporary:
        extracted=Path(temporary)/'bundle';extract_bundle(root.with_suffix('.zip'),extracted)
        source=extracted/'source';native=source/'target/release'/binary_path().name
        native.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(binary_path(),native)
        environment={**os.environ,'RMUC_RESOURCE_ROOT':str(source),'RMUC_USER_DATA':str(Path(temporary)/'data'),
            'PYTHONPATH':os.pathsep.join(str(source/name) for name in ('packages/core','research','apps/server'))}
        check_code=('import json,sys; from pathlib import Path; '
            'from rmuc2027_research.rl.v4_bundle import preflight; result=preflight(sys.argv[1]); '
            'Path(sys.argv[2]).write_text(json.dumps(result,indent=2)); '
            'sys.exit(0 if result["training_ready"] else 1)')
        checked=subprocess.run([sys.executable,'-c',check_code,str(extracted),str((root/'bundle-verification.json').resolve())],
            cwd=source,env=environment,capture_output=True,text=True,timeout=120)
        (root/'bundle-verification.log').write_text(checked.stdout+'\n'+checked.stderr)
        if checked.returncode:
            raise RuntimeError('Extracted bundle readiness failed; see bundle-verification.log')
        verification=json.loads((root/'bundle-verification.json').read_text())
    write_json(root/'package.json',dict(path=str(root.with_suffix('.zip')),
        verified_after_relocation=True,training_ready=verification['training_ready'],
        sha256=hashlib.sha256(root.with_suffix('.zip').read_bytes()).hexdigest()))


def child(root,args,deadline_unix,label):
    root=Path(root)
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    command=[sys.executable,'-u','-m',__package__+'.v4_campaign',*args]
    with (root/f'{label}.log').open('a') as stream:
        p=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env=env)
        try:
            while p.poll() is None:
                if time.time()>=deadline_unix-2:stop_worker(p,1.);break
                time.sleep(1)
        finally:stop_worker(p,0.)
    return p.returncode


def run(output,seconds=7200,num_envs=2,resume_from=None):
    if not 120<=seconds<=7200:raise ValueError('Each local campaign is bounded to at most 7200 wall seconds')
    from rmuc2027_core.paths import ROOT
    if resume_from:
        from .v4_bundle import verify_bundle
        resume_from=Path(resume_from).resolve()
        if not (resume_from/'bundle-manifest.json').is_file():
            raise ValueError('--resume-from requires an extracted, verified v4 bundle')
        verify_bundle(resume_from)
    preparation=(Path(resume_from)/'preparation') if resume_from and (Path(resume_from)/'preparation').exists() else ROOT/'data/research/sentry-v4-preparation'
    if not (preparation/'manifest.json').exists():raise ValueError('Prepare the readonly 2026 dataset before starting the training clock')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    started=time.time();end=started+seconds;scale=seconds/7200.
    report=dict(status='running',started_unix=started,deadline_unix=end,wall_budget_seconds=seconds,
                stages=[],evaluations=[],gradient_deadline_unix=started+4800*scale)
    write_json(output/'budget.json',{key:report[key] for key in ('started_unix','deadline_unix','wall_budget_seconds','gradient_deadline_unix')})
    if preparation.exists():shutil.copytree(preparation,output/'preparation')
    preflight_end=started+600*scale
    preflight_args=['preflight','--output',str(output),'--deadline-unix',str(preflight_end)]
    if resume_from:preflight_args.extend(['--resume-from',str(resume_from)])
    code=child(output,preflight_args,preflight_end,'preflight')
    report['stages'].append(dict(stage='preflight',exit_code=code))
    if code!=0:
        report.update(status='failed_preflight',finished_unix=time.time())
        write_json(output/'campaign.json',report)
        raise RuntimeError('Preflight failed; no weights have been fitted')
    if num_envs<=0:num_envs=json.loads((output/'preflight.json').read_text())['recommended_num_envs']
    report['num_envs']=num_envs
    if resume_from:
        source=Path(resume_from)
        from .v4_bundle import load_runtime_config
        for role in ('main','infantry','heavy'):
            source_role=source/'models'/role if (source/'models'/role).exists() else source/role
            shutil.copytree(source_role,output/role)
            metadata=json.loads((output/role/'experiment.json').read_text())
            metadata['config']=load_runtime_config(source,role)
            metadata['resume_bundle']=str(source.resolve())
            write_json(output/role/'experiment.json',metadata)
    else:
        until=started+1500*scale
        code=child(output,['initialize','--output',str(output),'--deadline-unix',str(until)],until,'initialize')
        report['stages'].append(dict(stage='initialize',exit_code=code))
    for role,offset in (('infantry',1950),('heavy',2400),('main',4800)):
        until=started+offset*scale
        if time.time()>=until-20 or not (output/role/'model.zip').exists():
            report['stages'].append(dict(stage=role,status='missing_checkpoint_or_budget'));continue
        if role=='main':fit_radar(output)
        code=child(output,['train-stage','--output',str(output),'--role',role,
            '--deadline-unix',str(until),'--num-envs',str(num_envs)],until,role)
        report['stages'].append(dict(stage=role,exit_code=code))
        write_json(output/'campaign.json',report)
    (output/'evaluation').mkdir(exist_ok=True)
    if (output/'main/model.zip').exists():
        jobs=[(backend,side) for backend in ('warp_kinematic','rapier','mjwarp') for side in ('red','blue')]
        for index,(backend,side) in enumerate(jobs):
            if time.time()>=end-300*scale:break
            until=time.time()+(end-300*scale-time.time())/(len(jobs)-index)
            code=child(output,['evaluate-pair','--output',str(output),'--backend',backend,'--side',side,
                '--deadline-unix',str(until)],until,f'evaluate-{backend}-{side}')
            report['evaluations'].append(dict(backend=backend,side=side,exit_code=code))
            write_json(output/'campaign.json',report)
    report['status']='packaging';write_json(output/'campaign.json',report)
    fit_radar(output,'radar-calibration-next')
    report['selection']=select_candidate(output)
    code=child(output,['package','--output',str(output)],end-1,'package')
    package_ok=code==0 and (output/'package.json').exists()
    if package_ok:report['bundle']=json.loads((output/'package.json').read_text())
    else:report['package_error']=dict(exit_code=code,log='package.log')
    complete=all(x.get('exit_code')==0 for x in report['stages']) and report['selection']['complete_pairs']==6
    report.update(status=('completed' if complete else 'completed_partial') if package_ok else 'failed_packaging',
                  finished_unix=time.time(),elapsed_seconds=time.time()-started,
                  within_wall_budget=time.time()<=end,all_stages_succeeded=complete)
    write_json(output/'campaign.json',report)
    print(json.dumps(report),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    runp=sub.add_parser('run');runp.add_argument('--output',type=Path,required=True);runp.add_argument('--wall-seconds',type=float,default=7200)
    runp.add_argument('--num-envs',type=int,default=2);runp.add_argument('--resume-from',type=Path)
    packp=sub.add_parser('package');packp.add_argument('--output',type=Path,required=True)
    for name in ('preflight','initialize','train-stage','evaluate-pair'):
        p=sub.add_parser(name);p.add_argument('--output',type=Path,required=True);p.add_argument('--deadline-unix',type=float,required=True)
        if name=='preflight':p.add_argument('--resume-from',type=Path)
        if name=='train-stage':p.add_argument('--role',choices=['main','infantry','heavy'],required=True);p.add_argument('--num-envs',type=int,default=2)
        if name=='evaluate-pair':
            p.add_argument('--backend',choices=['warp_kinematic','rapier','mjwarp'],required=True)
            p.add_argument('--side',choices=['red','blue'],required=True)
            p.add_argument('--reuse-anchor-from',type=Path)
    args=parser.parse_args()
    if args.command=='run':run(args.output,args.wall_seconds,args.num_envs,args.resume_from)
    elif args.command=='package':package(args.output)
    elif args.command=='preflight':preflight_worker(args.output,args.deadline_unix,args.resume_from)
    elif args.command=='initialize':initialization(args.output,args.deadline_unix)
    elif args.command=='train-stage':train_stage(args.output,args.role,args.deadline_unix,args.num_envs)
    else:evaluate_pair(args.output,args.backend,args.side,args.deadline_unix,args.reuse_anchor_from)


if __name__=='__main__':main()
