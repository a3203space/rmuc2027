"""Version-3 sentry probes and supervised initialization from public observations.

The teacher emits the same six actions as the learner. It cannot change a
latched goal, access a referee snapshot, or act through the navigator directly.
"""
import copy
import math
import time
from pathlib import Path

import numpy as np
from rmuc2027_core import sdk
from rmuc2027_core.models import default_team_lineup
from .session import load_config
from .portable import own_robot,SdkNavigator


def configuration(model):
    lineup=default_team_lineup();lineup['7']=model
    return load_config(dict(policy_version=3,navigation_mode='adaptive',rl_seconds=.1,
        sim_backend='warp_kinematic',sim_device='cuda:0',navigation_precompute=True,
        drones_enabled=False,timed_assembly=False,episode_seconds=60,alternate_sides=True,
        policy_timeout_s=10.,lineups={t:dict(lineup) for t in ('red','blue')},
        action_trace=True,github_backup_repo=None,checkpoint_seconds=90,
        ppo=dict(n_steps=64,batch_size=64,n_epochs=4,learning_rate=5e-5,target_kl=.012,
            gamma=.999,gae_lambda=.99,ent_coef=.003)))


def teacher_action(obs,state,goal=None):
    r=own_robot(obs,7);d=sdk._design(r);flip=obs['team']=='blue'
    if goal is None:
        # These are policy destinations, never injected execution overrides.
        capable=d.get('transform_allowed') or d.get('capabilities',{}).get('jump')
        canonical=(12.,7.5) if capable else (7.,10.)
        goal=[28-canonical[0],15-canonical[1]] if flip else list(canonical)
        local=(28-r['x'],15-r['y']) if flip else (r['x'],r['y'])
        # A mixed learner trajectory can pass the staging point without ever
        # entering a tiny arrival disk. Public crossing progress is sufficient;
        # do not label such states with a destination behind the robot.
        crossed=(capable and local[0]>=canonical[0]-.35 and abs(local[1]-canonical[1])<.6)
        if crossed or math.dist((r['x'],r['y']),goal)<.35:state['teacher_staging_reached']=True
        if state.get('teacher_staging_reached'):
            other='red' if flip else 'blue'
            destination=next(z for z in obs['arena']['zones'] if z['id']==other+('_central' if capable else '_outpost'))
            if 'teacher_capture_goal' not in state:
                # A building may occupy the nominal card reference point. Pick
                # a public legal grid point that stays within the actual card.
                polygon=destination['polygon'];nav=SdkNavigator()
                candidates=[(i*.25,j*.25) for i in range(math.ceil(min(x for x,y in polygon)*4),math.floor(max(x for x,y in polygon)*4)+1)
                    for j in range(math.ceil(min(y for x,y in polygon)*4),math.floor(max(y for x,y in polygon)*4)+1)
                    if sdk.point_in_polygon(i*.25,j*.25,polygon)]
                candidates.sort(key=lambda p:math.dist(p,(destination['x'],destination['y'])))
                state['teacher_capture_goal']=next((list(p) for p in candidates if nav.resolve_goal(r,p,obs['arena'])==list(p)),[destination['x'],destination['y']])
            goal=state['teacher_capture_goal']
    public=dict(obs,navigation_mode='adaptive',policy_dt=.05)
    fields=sdk.navigate_action(r,goal,public,copy.deepcopy(state.get('navigation',{})))
    # A pulse followed by a release; no ground-truth label or auxiliary leg action.
    jump=bool(fields.get('jump'))
    x,y=goal
    if flip:x,y=28-x,15-y
    return np.asarray([x/14-1,y/7.5-1,.95 if fields.get('transform') else -.95,
        .95 if jump else -.95,.95 if math.dist((r['x'],r['y']),goal)<.3 else -.95,
        .95 if obs['autoaim']['valid'] else -.95],dtype=np.float32)


def collect(config,output,deadline,seconds=24.,seeds=(101,102),backend='warp_kinematic'):
    from .env import SentryEnv
    from .campaign import write_json
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    samples=[];targets=[];episodes=[]
    for seed in seeds:
        if time.monotonic()>=deadline:break
        cfg=copy.deepcopy(config);cfg.update(sim_backend=backend,episode_seconds=seconds,alternate_sides=False,
            team='red' if seed%2 else 'blue',action_trace=True)
        cfg['controllers']={cfg['team']+':7':dict(kind='learning')}
        env=SentryEnv(cfg);start=time.monotonic();trace=[]
        try:
            features,info=env.reset(seed=seed)
            while time.monotonic()<deadline:
                action=teacher_action(env.session.obs,env.session.controller.state)
                samples.append(features);targets.append(action)
                features,_,done,truncated,info=env.step(action)
                trace.append(dict(time=info['time'],position=[own_robot(env.session.obs,7)[k] for k in ('x','y','z')],
                    action=action.tolist(),execution=info['action_trace'],autoaim=env.session.obs['autoaim']))
                if done or truncated:break
            robot=own_robot(env.session.obs,7)
            row=dict(seed=seed,side=cfg['team'],backend=backend,wall_seconds=time.monotonic()-start,
                simulated_seconds=info['time'],position=[robot[k] for k in ('x','y','z')],
                final_navigation=info['action_trace']['navigation'],metrics=info['metrics'])
            episodes.append(row)
            write_json(output/f'probe-{seed}-{backend}.json',dict(result=row,trace=trace))
            print({'kind':'sentry-navigation-probe',**row},flush=True)
        finally:env.close()
    if samples:
        np.savez_compressed(output/'demonstrations.npz',observations=np.asarray(samples),targets=np.asarray(targets))
    write_json(output/'probes.json',episodes)
    return output/'demonstrations.npz'


def initialize_actor(model,path,deadline,epochs=300):
    import torch
    from .sentry_v3 import schema
    with np.load(path) as data:
        observations=data['observations'];targets=data['targets']
    if observations.shape[1]!=schema()['obs_size'] or targets.shape[1]!=6:raise ValueError('v3 demonstration schema mismatch')
    x=torch.as_tensor(observations,dtype=torch.float32,device=model.device)
    y=torch.as_tensor(targets,dtype=torch.float32,device=model.device)
    weights=torch.ones_like(y)
    for axis in (2,3,4,5):
        positive=y[:,axis]>0
        if positive.any():weights[positive,axis]=min(200.,float(len(y)/positive.sum())*.5)
    optimizer=torch.optim.Adam(list(model.policy.mlp_extractor.policy_net.parameters())+list(model.policy.action_net.parameters()),lr=5e-4)
    first=None;loss=None;steps=0
    for _ in range(epochs):
        if time.monotonic()>=deadline:break
        predicted=model.policy.action_net(model.policy.mlp_extractor.policy_net(x))
        loss=((predicted-y).square()*weights).mean()
        if first is None:first=float(loss.detach())
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.policy.parameters(),.5);optimizer.step();steps+=1
    with torch.no_grad():prediction=model.policy.action_net(model.policy.mlp_extractor.policy_net(x))
    positives={str(axis):dict(count=int((y[:,axis]>.5).sum()),recall=float((prediction[y[:,axis]>.5,axis]>.5).float().mean())
        if (y[:,axis]>.5).any() else None) for axis in (2,3,4,5)}
    return dict(samples=len(x),epochs=steps,loss_before=first,loss_after=float(loss.detach()) if loss is not None else None,positive_actions=positives,
                source='public_observation_teacher',action_size=6)


def retain_demonstrations(model):
    """Small actor-only BC term after PPO; the executor still receives actor goals."""
    import torch
    if not hasattr(model,'_sentry_guidance_cache'):
        with np.load(model.sentry_guidance_path) as data:
            x=torch.as_tensor(data['observations'],dtype=torch.float32,device=model.device)
            y=torch.as_tensor(data['targets'],dtype=torch.float32,device=model.device)
        weights=torch.ones_like(y);weights[:,:2]=.25
        for axis in (2,3,4,5):
            positive=y[:,axis]>.5
            if positive.any():weights[positive,axis]=min(200.,float(len(y)/positive.sum())*.5)
        model._sentry_guidance_cache=(x,y,weights)
    x,y,weights=model._sentry_guidance_cache
    # Uniform minibatches often omit one-frame jump pulses entirely.
    rare=torch.where((y[:,3]>.5) | (y[:,2]>.5))[0]
    rare=rare[torch.randperm(len(rare),device=model.device)[:16]]
    indices=torch.unique(torch.cat((torch.randperm(len(x),device=model.device)[:240],rare)))
    predicted=model.policy.action_net(model.policy.mlp_extractor.policy_net(x[indices]))
    loss=((predicted-y[indices]).square()*weights[indices]).mean()
    model.policy.optimizer.zero_grad()
    (.1*loss).backward()
    torch.nn.utils.clip_grad_norm_(model.policy.parameters(),model.max_grad_norm)
    model.policy.optimizer.step()
    return float(loss.detach())
