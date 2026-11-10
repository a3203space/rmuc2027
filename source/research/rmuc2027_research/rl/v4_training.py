"""Explicit v4 demonstrations, actor initialization and resumable state."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .campaign import write_json
from .checkpoints import save_checkpoint
from .env import SentryEnv
from .scenarios import ScenarioSpec
from .sentry_training import teacher_action,configuration as old_configuration


def configuration(role='main',seed=20261110):
    cfg=old_configuration(2773)
    slot={'main':7,'infantry':3,'heavy':1}[role]
    cfg.update(policy_version=4 if role=='main' else 5,slot=slot,seed=seed,
               episode_seconds=420,scenario_sampling=True,capability_sampling=True,
               drones_enabled=True,alternate_sides=False,action_trace=False,
               controllers={f'red:{slot}':{'kind':'learning'}},training_summary=False)
    cfg['ppo'].update(n_steps=64,batch_size=64,n_epochs=4,learning_rate=5e-5,
                      gamma=.999,gae_lambda=.99,ent_coef=.003,target_kl=.012)
    return cfg


def collect_main(config,output,deadline):
    from .radar_calibration import RadarCalibrationRecorder
    recorder=RadarCalibrationRecorder()
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    observations=[];targets=[];reports=[]
    jobs=[('warp_kinematic','red',0),('warp_kinematic','blue',3),
          ('warp_kinematic','red',7),('warp_kinematic','blue',0),
          ('rapier','red',3),('rapier','blue',7),('mjwarp','red',0)]
    for index,(backend,side,delay) in enumerate(jobs):
        if time.monotonic()>deadline-10:break
        cfg=copy.deepcopy(config)
        pairs=((2731,2732),(303,2731),(2732,303),(303,303),(2731,2731),(2732,2732))
        scenario=ScenarioSpec(friendly_infantry=pairs[index%len(pairs)],enemy_infantry=pairs[(index+2)%len(pairs)],
                              enemy_sentry=(2771,2772,2773)[index%3],learner_side=side)
        cfg.update(sim_backend=backend,episode_seconds=45,scenario=scenario.to_dict(),
                   scenario_sampling=False,navigation_precompute=backend=='warp_kinematic')
        env=SentryEnv(cfg);info=None;count=0;started=time.monotonic()
        try:
            features,info=env.reset(seed=20501+index)
            while time.monotonic()<deadline-2:
                obs=env.session.obs
                if obs['time']<delay:
                    r=next(r for r in obs['robots'] if r['slot']==7)
                    x,y=(28-r['x'],15-r['y']) if side=='blue' else (r['x'],r['y'])
                    action=np.array([x/14-1,y/7.5-1,-1,-1,-1,-1],dtype=np.float32)
                else:
                    action=teacher_action(obs,env.session.controller.state)
                    observations.append(features['actor']);targets.append(action);count+=1
                features,_,done,truncated,info=env.step(action)
                recorder.observe(env.session.battle.match,f'main-seed-{20501+index}')
                if done or truncated:break
            reports.append(dict(backend=backend,side=side,legal_prefix_seconds=delay,samples=count,
                simulation_seconds=info['time'],wall_seconds=time.monotonic()-started,metrics=info['metrics']))
            print(json.dumps({'kind':'v4-demonstration',**reports[-1]}),flush=True)
        finally:env.close()
    if not observations:raise RuntimeError('No legal v4 demonstrations collected')
    path=output/'demonstrations.npz'
    np.savez_compressed(path,observations=np.asarray(observations,dtype=np.float32),targets=np.asarray(targets,dtype=np.float32))
    write_json(output/'probes.json',reports)
    write_json(output/'radar-calibration-rows.json',recorder.rows())
    return path


def initialize_main(model,path,deadline,epochs=500):
    from .sentry_v4 import schema,FEATURE_INDEX
    with np.load(path) as data:x=torch.as_tensor(data['observations'],device=model.device);y=torch.as_tensor(data['targets'],device=model.device)
    if x.shape[1]!=schema()['obs_size']:raise ValueError('v4 demonstration schema changed')
    params=list(model.policy.actor_parameters())
    optimizer=torch.optim.Adam(params,lr=5e-4)
    weights=torch.ones_like(y)
    for axis in (2,3,4,5):
        positive=y[:,axis]>0
        if positive.any():weights[positive,axis]=min(50.,.5*len(y)/positive.sum().item())
    # Sustained folding is much more common than the first fold command.
    # Balance the two real transitions separately; execution never supplies an
    # automatic fold request and demonstrations remain physical observations.
    latched=x[:,FEATURE_INDEX['navigation.fold_latched']]>.5
    onset=(y[:,2]>0)&~latched&(x[:,FEATURE_INDEX['self.fold_progress']]<.05)
    release=(y[:,2]<0)&latched&(x[:,FEATURE_INDEX['roof.can_unfold']]>.5)
    for transition in (onset,release):
        if transition.any():weights[transition,2]=max(1.,.5*len(y)/transition.sum().item())
    first=None;last=None;steps=0
    for epoch in range(epochs):
        if time.monotonic()>=deadline:break
        ids=torch.randperm(len(x),device=model.device)[:512]
        predicted=model.policy.mlp_extractor.policy_net(x[ids])
        loss=((predicted-y[ids]).square()*weights[ids]).mean()
        first=loss.item() if first is None else first
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(params,.5);optimizer.step()
        last=loss.item();steps+=1
    with torch.no_grad():
        model.policy.log_std[:2].fill_(math.log(.04))
        model.policy.log_std[2:].fill_(math.log(.15))
        prediction=model.policy.mlp_extractor.policy_net(x)
    def recall(mask,positive):
        return float(((prediction[mask,2]>.3) if positive else (prediction[mask,2]<-.3)).float().mean()) if mask.any() else None
    return dict(kind='explicit_initialization',samples=len(x),epochs=steps,loss_before=first,loss_after=last,
                dataset_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),ppo_samples=0,
                fold_onset_samples=int(onset.sum()),fold_onset_recall=recall(onset,True),
                unfold_transition_samples=int(release.sum()),unfold_transition_recall=recall(release,False),
                fold_false_positive_rate=float((prediction[y[:,2]<0,2]>.3).float().mean()),
                transition_weighting='independently_balance_real_fold_onset_and_release')


def save_state(model,vec,folder,name='model.zip'):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    save_checkpoint(model,folder/name)
    samplers=vec.env_method('sampler_state') if vec is not None else []
    write_json(folder/'samplers.json',samplers)
    if vec is not None:
        rows=[row for group in vec.env_method('calibration_rows') for row in group]
        write_json(folder/'training-radar-calibration-rows.json',rows)
    state=dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),
               torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
               num_timesteps=model.num_timesteps,completed_updates=getattr(model,'completed_updates',0),
               optimized_samples=getattr(model,'optimized_samples',0),
               partial_rollout_discarded=max(0,model.num_timesteps-getattr(model,'last_completed_samples',model.num_timesteps)),
               resume_boundary='new_episode_after_completed_update',bitwise_gpu_replay=False,normalization=None,
               execution_state=vec.env_method('execution_state') if vec else [])
    pending=folder/'resume-state.tmp';torch.save(state,pending);os.replace(pending,folder/'resume-state.pt')


def plan_sampler_restore(samplers, num_envs, *, num_timesteps=0, source_sha256=None):
    """Pure future-resume migration, preserving recorded history exactly once.

    A changed topology or cloned legacy streams needs independent new streams.
    Started coverage is not completed coverage. Raw ancestors are retained for
    audit: historical duplication cannot be inferred away from these records.
    """
    from .scenarios import ScenarioSampler,canonical_hash
    from .capabilities import CapabilitySampler
    if type(num_envs) is not int or num_envs < 1:raise ValueError('Positive sampler worker count required')
    source_sha256=source_sha256 or canonical_hash(samplers)
    def duplicates(kind):
        groups={}
        for index,item in enumerate(samplers):
            state=item.get('session',{}).get(kind)
            if state is None:continue
            identity={'rng_state':state['rng_state']}
            if kind=='scenario':identity['queue']=state['queue']
            groups.setdefault(canonical_hash(identity),[]).append(index)
        return [ids for ids in groups.values() if len(ids)>1]
    cloned_scenario=duplicates('scenario');cloned_capabilities=duplicates('capabilities')
    reasons=[]
    if samplers and len(samplers)!=num_envs:reasons.append('worker_topology_changed')
    if cloned_scenario:reasons.append('legacy_cloned_scenario_streams')
    if cloned_capabilities:reasons.append('legacy_cloned_capability_streams')
    report=dict(version=1,source_sampler_sha256=source_sha256,source_workers=len(samplers),target_workers=num_envs,
        checkpoint_num_timesteps=int(num_timesteps),reasons=reasons,
        duplicate_scenario_workers=cloned_scenario,duplicate_capability_workers=cloned_capabilities,
        bitwise_sampler_restore=bool(samplers) and not reasons,bitwise_hardware_resume=False,
        mode='migrated' if reasons else ('exact_sampler_state' if samplers else 'fresh_no_saved_sampler'))
    if not reasons:return copy.deepcopy(samplers),report
    scenarios=[ScenarioSampler.from_state_dict(item['session']['scenario']) for item in samplers]
    capabilities=[CapabilitySampler.from_state_dict(item['session']['capabilities']) for item in samplers]
    if any(type(item['session']['episode_index']) is not int or item['session']['episode_index']<0 for item in samplers):
        raise ValueError('Invalid recorded session episode count')
    started=set().union(*(set(item.queue[:item.cursor]) for item in scenarios))
    counts={};failures={}
    for sampler in scenarios:
        for case,row in sampler.counts.items():
            merged=counts.setdefault(case,{})
            for field,value in row.items():
                if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
                    raise ValueError('Invalid recorded scenario counter')
                merged[field]=merged.get(field,0)+value
        for case,weight in sampler.failures.items():failures[case]=max(failures.get(case,0.),weight)
    def seed(domain,index):
        value=f'sampler-restore-v1/{source_sha256}/{num_timesteps}/{num_envs}/{domain}/{index}'
        return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8],'big')
    def share(total,index):return total//num_envs+int(index<total%num_envs)
    assigned=[{} for _ in range(num_envs)]
    for case,row in counts.items():
        owner=int(hashlib.sha256(('coverage-owner/'+case).encode()).hexdigest(),16)%num_envs
        assigned[owner][case]=copy.deepcopy(row)
    total_episodes=sum(int(item['session']['episode_index']) for item in samplers)
    total_capability_episodes=sum(item.episodes for item in capabilities)
    result=[];streams=[]
    for index in range(num_envs):
        seeds={name:seed(name,index) for name in ('scenario','capabilities','session','gym')}
        scenario=ScenarioSampler(seeds['scenario'],scenarios[index%len(scenarios)].failure_fraction)
        # Retain globally started cases as a prefix, then independently order
        # the remaining cases. Normal post-coverage/failure sampling can revisit
        # any case; migration does not claim a completed episode or extra data.
        scenario.queue=[key for key in scenario.queue if key in started]+[key for key in scenario.queue if key not in started]
        scenario.cursor=len(started);scenario.counts=assigned[index];scenario.failures=copy.deepcopy(failures)
        capability=CapabilitySampler(seeds['capabilities']);capability.episodes=share(total_capability_episodes,index)
        migrated=copy.deepcopy(samplers[index%len(samplers)])
        migrated['session'].update(episode_index=share(total_episodes,index),scenario=scenario.state_dict(),
            capabilities=capability.state_dict(),rng_state=random.Random(seeds['session']).getstate())
        migrated['gym_rng']=np.random.default_rng(seeds['gym']).bit_generator.state
        result.append(migrated);streams.append(dict(worker=index,seeds=seeds,started_cursor=scenario.cursor,
            assigned_counter_cases=sorted(assigned[index]),failure_fraction=scenario.failure_fraction))
    report.update(ancestor_states=copy.deepcopy(samplers),streams=streams,
        original_cursors=[item.cursor for item in scenarios],started_prefix_union=sorted(started),
        coverage_semantics='union of started prefixes, including incomplete episodes; not completed coverage; later sampling may revisit',
        recorded_counts_before=counts,recorded_counts_after={case:row for group in assigned for case,row in group.items()},
        failure_weights_before=[copy.deepcopy(item.failures) for item in scenarios],failure_weights_after=failures,
        failure_merge='per-case maximum priority replicated for independent replay; not summed failure events',
        historical_clone_records_preserved_without_deduplication=True,migration_adds_observed_coverage=False,
        session_episode_total_before=total_episodes,session_episode_total_after=sum(x['session']['episode_index'] for x in result),
        capability_episode_total_before=total_capability_episodes,
        capability_episode_total_after=sum(x['session']['capabilities']['episodes'] for x in result))
    return result,report


def restore_state(folder,vec):
    folder=Path(folder)
    if (folder/'resume-state.pt').exists():
        state=torch.load(folder/'resume-state.pt',map_location='cpu',weights_only=False)
        random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch_cpu'])
        cuda=state.get('torch_cuda',[])
        if cuda and torch.cuda.is_available() and len(cuda)==torch.cuda.device_count():torch.cuda.set_rng_state_all(cuda)
    else:state={}
    raw=(folder/'samplers.json').read_bytes() if (folder/'samplers.json').exists() else b'[]'
    samplers,report=plan_sampler_restore(json.loads(raw),vec.num_envs,num_timesteps=state.get('num_timesteps',0),
        source_sha256=hashlib.sha256(raw).hexdigest())
    for index,item in enumerate(samplers):vec.env_method('restore_sampler_state',item,indices=index)
    if samplers:vec._seeds=[None]*vec.num_envs
    # Never rewrite the source sampler/RNG checkpoint during migration. A later
    # completed-update save persists the new live states; this report retains
    # every original ancestor and explicitly marks the non-bitwise transition.
    report_path=folder/'sampler-restore.json'
    if report_path.exists():
        previous=report_path.read_bytes();previous_hash=hashlib.sha256(previous).hexdigest()
        relative=f'sampler-restore-history/{previous_hash}.json';history=folder/relative
        history.parent.mkdir(exist_ok=True)
        if not history.exists():history.write_bytes(previous)
        report['previous_restore_report']=dict(path=relative,sha256=previous_hash)
    write_json(report_path,report)
    state['sampler_restore']=report
    return state
