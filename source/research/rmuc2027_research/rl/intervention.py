"""Audited skill demonstrations and corrective actor fitting.

The teacher consumes scoped observations and controller history only. It is
used during training, never substituted for the deployed actor's decisions.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO

from rmuc2027_core import sdk
from .campaign import write_json
from .checkpoints import save_checkpoint
from .env import SentryEnv
from .learning import experiment, _actor_parameters
from .portable import own_robot
from .validation import frozen_model

_SOURCE_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def teacher_action(observation, state, config):
    robot=own_robot(observation,config['slot'])
    design=sdk._design(robot)
    goal=state['course_goal']
    nav=copy.deepcopy(state.get('navigation',{}))
    public=dict(observation,navigation_mode='adaptive',policy_dt=config['rl_seconds'])
    fields=sdk.navigate_action(robot,goal,public,nav)
    capable=design.get('capabilities',{})
    legged=design.get('chassis_type')=='wheel_legged'
    mask=np.asarray([bool(design.get('transform_allowed')),legged,bool(capable.get('jump'))],dtype=np.float32)
    leg=fields.get('leg_height',design.get('leg_nominal_m',0.))
    leg_value=(2*(leg-design.get('leg_min_m',0.))/max(.01,design.get('leg_max_m',1.)-design.get('leg_min_m',0.))-1) if legged else 0.
    jump=bool(fields.get('jump'))
    if state.get('course_kind')=='flat_jump' and capable.get('jump'):
        actual=state.get('navigation',{})
        ctx=sdk.locomotion_context(robot,public,actual)
        # A held positive intent preserves the observed latch after the first
        # pulse; the existing rising-edge guard prevents repeated jumping.
        jump=(actual.get('last_jump',-1.)>=0 or
              (ctx['stable'] and ctx['cooldown']<=0 and ctx['can_unfold']
               and observation['time']>=config.get('teacher_settle_seconds',.5) and math.hypot(robot.get('vx',0.),robot.get('vy',0.))<.15
               and math.dist((robot['x'],robot['y']),goal)<.3))
    target=np.asarray([.95 if fields.get('transform',False) else -.95,
                       max(-1.,min(1.,leg_value)),.95 if jump else -.95],dtype=np.float32)
    action=np.concatenate([np.zeros(2,dtype=np.float32),target])
    return action,target,mask


def collect(config, output, seeds=(42,43,44), sides=('red','blue'), backend='rapier', actor=None, deadline=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    observations=[];targets=[];masks=[];groups=[];episodes=[]
    for seed in seeds:
        for side in sides:
            if deadline is not None and time.monotonic()>=deadline:break
            cfg=copy.deepcopy(config)
            cfg.update(team=side,controllers={f"{side}:{cfg['slot']}":{'kind':'learning'}},
                       curriculum_seconds=1.,alternate_sides=False,sim_backend=backend,navigation_precompute=True,
                       guided_locomotion=True)
            cfg.pop('_training_deadline',None)
            env=SentryEnv(cfg)
            started=time.monotonic()
            try:
                obs,info=env.reset(seed=seed)
                steps=0;ended=False
                while deadline is None or time.monotonic()<deadline:
                    action,target,mask=teacher_action(env.session.obs,env.session.controller.state,cfg)
                    observations.append(obs.copy());targets.append(target);masks.append(mask)
                    groups.append({'travel':0,'ability':1,'flat_jump':2}[env.session.controller.state['course_kind']])
                    if actor is not None:action=actor.predict(obs,deterministic=True)[0]
                    obs,_,terminated,truncated,info=env.step(action)
                    steps+=1
                    if terminated or truncated:
                        ended=True;break
                episodes.append(dict(seed=seed,side=side,backend=backend,steps=steps,finished=ended,
                    course=info['course'],metrics=info['metrics'],wall_seconds=time.monotonic()-started,
                    final_navigation=copy.deepcopy(env.session.controller.state.get('navigation',{})),
                    final_robot={k:own_robot(env.session.obs,cfg['slot']).get(k) for k in ('x','y','z','legs','grounded','upright','tipped')},
                    controller='actor' if actor is not None else 'public_sdk_teacher'))
                write_json(output/'episodes.json',episodes)
                print(json.dumps(episodes[-1]),flush=True)
            finally:env.close()
    if not observations:raise RuntimeError('No demonstration samples collected within the deadline')
    path=output/'demonstrations.npz'
    np.savez_compressed(path,observations=np.asarray(observations),targets=np.asarray(targets),
                        masks=np.asarray(masks),groups=np.asarray(groups,dtype=np.int64))
    write_json(output/'dataset.json',dict(samples=len(observations),backend=backend,
        public_observations_only=True,actor_executed=actor is not None,config=config,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),episodes=episodes,
        teacher_source_sha256=_SOURCE_SHA256,
        sdk_sha256=hashlib.sha256(Path(sdk.__file__).read_bytes()).hexdigest()))
    return path


def data_tensors(paths):
    arrays=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as data:
            arrays.append({key:data[key] for key in ('observations','targets','masks','groups')})
    merged={key:np.concatenate([a[key] for a in arrays]) for key in arrays[0]}
    if merged['observations'].shape[1]!=152 or merged['targets'].shape[1]!=3:
        raise ValueError('Guidance requires the v2 observation and skill schema')
    # PPO's Gaussian mean is unbounded, while executed actions are clipped.
    # A small mean margin makes an exact minimum/maximum leg command survive
    # fitting error; otherwise sub-millimetre offsets can change contact timing.
    limits=np.abs(merged['targets'][:,1])>=.999
    merged['targets'][limits,1]=np.sign(merged['targets'][limits,1])*1.1
    weights=merged['masks'].copy()
    # Balance rare takeoff pulses separately from the persistent flat-jump
    # latch, so easy flat-ground positives cannot drown out ledge launches.
    for group in np.unique(merged['groups']):
        for skill in (0,2):
            selected=(merged['groups']==group)&(merged['masks'][:,skill]>0)
            positive=selected&(merged['targets'][:,skill]>0)
            negative=selected&~positive
            categories=[positive,negative]
            if skill==2:
                held=merged['observations'][:,116+21]>.5
                categories=[positive&~held,positive&held,negative]
            categories=[category for category in categories if category.any()]
            for category in categories:
                weights[category,skill]*=min(256.,selected.sum()/(len(categories)*category.sum()))
    weights[:,1]*=.5
    return tuple(torch.as_tensor(a,dtype=torch.float32) for a in
                 (merged['observations'],merged['targets'],weights))


def actor_mean(model, observations):
    return model.policy.action_net(model.policy.mlp_extractor.policy_net(observations))


def anchor_update(model, paths):
    """Keep successful demonstrated skills during subsequent PPO updates."""
    if not hasattr(model,'_skill_guidance_cache'):
        values=data_tensors(paths)
        model._skill_guidance_cache=tuple(value.to(model.device) for value in values)
    inputs,targets,weights=model._skill_guidance_cache
    if weights.sum()==0:return 0.
    indices=torch.randint(len(inputs),(min(256,len(inputs)),),device=model.device)
    mean=actor_mean(model,inputs[indices])
    loss=((mean[:,2:]-targets[indices]).square()*weights[indices]).sum()/weights[indices].sum().clamp_min(1.)
    model.policy.optimizer.zero_grad()
    (loss*.5).backward()
    torch.nn.utils.clip_grad_norm_(model.policy.parameters(),model.max_grad_norm)
    model.policy.optimizer.step()
    model.skill_guidance_updates=getattr(model,'skill_guidance_updates',0)+1
    return float(loss.detach())


def fit(model, paths, *, steps=1200, learning_rate=.001, max_seconds=40., deadline=None):
    torch.set_num_threads(1)
    inputs,targets,weights=data_tensors(paths)
    if weights.sum()==0:return dict(steps=0,reason='no_special_skills')
    parameters=[p for name,p in model.policy.named_parameters() if 'policy_net' in name or 'action_net' in name]
    optimizer=torch.optim.Adam(parameters,lr=learning_rate)
    generator=torch.Generator().manual_seed(20261110)
    with torch.no_grad():goals=actor_mean(model,inputs)[:,:2].clone()
    def losses(indices):
        prediction=actor_mean(model,inputs[indices])
        skill=((prediction[:,2:]-targets[indices]).square()*weights[indices]).sum()/weights[indices].sum().clamp_min(1.)
        goal=(prediction[:,:2]-goals[indices]).square().mean()
        return skill,goal
    all_indices=torch.arange(len(inputs))
    with torch.no_grad():initial=float(losses(all_indices)[0])
    started=time.monotonic();updates=0
    for _ in range(steps):
        if time.monotonic()-started>=max_seconds or deadline is not None and time.monotonic()>=deadline:break
        indices=torch.randint(len(inputs),(min(256,len(inputs)),),generator=generator)
        skill,goal=losses(indices)
        optimizer.zero_grad();(skill+.1*goal).backward()
        torch.nn.utils.clip_grad_norm_(parameters,1.)
        optimizer.step();updates+=1
    with torch.no_grad():
        final,goal=losses(all_indices)
        if not torch.isfinite(final):raise RuntimeError('Non-finite corrective loss')
        # Reduce skill exploration after fitting. Goal exploration is retained.
        model.policy.log_std[2:].clamp_(max=-1.5)
    return dict(steps=updates,initial_skill_mse=initial,final_skill_mse=float(final),
                goal_output_mse=float(goal),wall_seconds=time.monotonic()-started,
                skill_log_std_max=-1.5,class_balanced_by_course=True)


def correct(checkpoint, output, dataset_paths, max_seconds=40., *, fit_steps=1200, fit_learning_rate=.001):
    checkpoint,output=Path(checkpoint),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    model,parent=frozen_model(checkpoint,allow_sdk_transfer=True)
    before=_actor_parameters(model)
    result=fit(model,dataset_paths,max_seconds=max_seconds,steps=fit_steps,learning_rate=fit_learning_rate) if dataset_paths else dict(steps=0,reason='no_special_skills')
    config=copy.deepcopy(parent['config'])
    config.update(guided_locomotion=True,curriculum_seconds=0.,skill_guidance_datasets=[str(p) for p in dataset_paths])
    config['ppo'].update(n_epochs=8)
    model.n_epochs=8
    if dataset_paths:
        config['ppo']['ent_coef']=.001;model.ent_coef=.001
    # PPO resumes with a fresh optimizer after the supervised actor correction.
    if dataset_paths:
        model.policy.optimizer=model.policy.optimizer_class(model.policy.parameters(),lr=float(model.lr_schedule(1.)),**model.policy.optimizer_kwargs)
    metadata=experiment(config)
    metadata['initialization']=dict(kind='public_skill_demonstrations',
        parent_checkpoint=str(checkpoint),parent_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        parent_experiment=parent,optimizer_reset=bool(dataset_paths),
        corrective_source_sha256=_SOURCE_SHA256,
        datasets=[dict(path=str(p),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest()) for p in dataset_paths])
    write_json(output/'experiment.json',metadata)
    save_checkpoint(model,output/'model.zip')
    after=_actor_parameters(model)
    change=math.sqrt(sum(float(torch.sum((after[k]-v).square())) for k,v in before.items()))
    result.update(actor_parameter_l2_change=change,actor_updated=math.isfinite(change) and change>0,
                  model_sha256=hashlib.sha256((output/'model.zip').read_bytes()).hexdigest(),
                  ppo_updates=0,optimizer_reset=bool(dataset_paths),fit_learning_rate=fit_learning_rate)
    write_json(output/'correction.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--backend',choices=['rapier','warp_kinematic','mjwarp'],default='rapier')
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    parser.add_argument('--sides',nargs='+',choices=['red','blue'],default=['red','blue'])
    args=parser.parse_args()
    _,metadata=frozen_model(args.model,allow_sdk_transfer=True)
    path=collect(metadata['config'],args.output/'teacher',args.seeds,args.sides,args.backend)
    print(json.dumps(correct(args.model,args.output/'corrected',[path])),flush=True)


if __name__=='__main__':main()
