"""Serial all-model training with one real-time budget and atomic progress.

Preparation and validation run separately. The clock here includes worker
imports, environment initialization, PPO, saving and process cleanup.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from rmuc2027_core import sdk
from rmuc2027_core.models import default_team_lineup
from .session import load_config


def models():
    return {int(k):d for k,d in sdk.DESIGNS.items()
            if d.get('seasons',{}).get('2027',{}).get('legal') and d.get('role') in ('heavy','infantry','sentry')}


def weight(design):
    return 2 if design.get('chassis_type')=='wheel_legged' or design.get('transform_allowed') else 1


def allocations(selected, seconds, reserve=120.):
    if not math.isfinite(seconds) or seconds<=reserve or seconds>7200:
        raise ValueError('Campaign requires reserve < wall_seconds <= 7200')
    catalog=models()
    if len(set(selected))!=len(selected) or not selected or any(m not in catalog for m in selected):
        raise ValueError('Select unique legal ground models')
    total=sum(weight(catalog[m]) for m in selected)
    return {m:(seconds-reserve)*weight(catalog[m])/total for m in selected}


def configuration(model, allowance):
    design=models()[model]
    slot={'heavy':1,'infantry':3,'sentry':7}[design['role']]
    lineup=default_team_lineup()
    lineup[str(slot)]=model
    cfg=load_config(dict(slot=slot,team='red',policy_version=2,navigation_mode='adaptive',
        sim_backend='warp_kinematic',sim_device='cuda:0',rl_seconds=.1,learning_control_hz=10,
        fixed_policy_hz=10,drones_enabled=False,timed_assembly=True,episode_seconds=420,
        lineups={t:dict(lineup) for t in ('red','blue')},
        github_backup_repo=None,checkpoint_seconds=60.,curriculum_seconds=allowance*(.6 if weight(design)==2 else .4)))
    cfg['ppo'].update(n_steps=64,batch_size=64,n_epochs=4,learning_rate=3e-4,
                      gamma=.995**.2,gae_lambda=.95**.2)
    return cfg


def write_json(path, value):
    pending=path.with_suffix('.tmp')
    with pending.open('w') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(pending,path)


def stop_worker(process, grace):
    if process.poll() is None:
        os.killpg(process.pid,signal.SIGINT)
        try:process.wait(timeout=max(0.,grace))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait()
    # The worker may have exited before its child processes did.
    try:os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:pass


def run(output, selected, seconds=7200., num_envs=8, reserve=120., device='cpu', resume=False):
    output=Path(output)
    if resume:
        report=json.loads((output/'campaign.json').read_text())
        if report['status']=='running':
            raise ValueError('Stop the existing campaign before resuming it')
        seconds=report['wall_budget_seconds'];reserve=report['reserve_seconds']
        num_envs=report['num_envs'];device=report['policy_device'];selected=report['models']
        budgets={int(k):v for k,v in report['allocations'].items()}
        report.setdefault('attempts',list(report['results']))
        # Recover a child interrupted before the older scheduler recorded it.
        known={r['model'] for r in report['attempts']}
        for model in selected:
            folder=output/str(model)
            if model not in known and folder.exists():
                paths=[folder/name for name in ('interrupted.zip','model.zip','latest.zip') if (folder/name).exists()]
                spent=max(0.,report['elapsed_seconds']-sum(r['wall_seconds'] for r in report['attempts']))
                report['attempts'].append(dict(model=model,status='interrupted',wall_seconds=spent,
                    checkpoint=str(paths[0]) if paths else None,training={}))
        elapsed=max(report['elapsed_seconds'],time.time()-report['started_unix'])
        report.setdefault('resumes',[]).append(dict(time_unix=time.time(),elapsed_seconds=elapsed))
    else:
        budgets=allocations(selected,seconds,reserve)
        output.mkdir(parents=True,exist_ok=False)
        (output/'configs').mkdir()
        for model,budget in budgets.items():
            write_json(output/'configs'/f'{model}.json',configuration(model,budget))
        report=dict(wall_budget_seconds=seconds,reserve_seconds=reserve,num_envs=num_envs,policy_device=device,physics_device='cuda:0',
                    started_unix=time.time(),models=selected,allocations=budgets,results=[],attempts=[])
        elapsed=0.
    configs=output/'configs'
    started=time.monotonic()-elapsed;deadline=started+seconds
    report['status']='running'
    write_json(output/'campaign.json',report)
    process=None
    try:
        for model in selected:
            attempts=[r for r in report['attempts'] if r['model']==model]
            if attempts and attempts[-1]['status']=='trained':continue
            spent=sum(r['wall_seconds'] for r in attempts)
            available=min(budgets[model]-spent,deadline-reserve-time.monotonic())
            if available<=15:
                report['results']=[r for r in report['results'] if r['model']!=model]+[dict(model=model,status='budget_exhausted',actor_updated=False)]
                continue
            run_dir=output/(str(model) if not attempts else f'{model}-attempt-{len(attempts)+1}')
            begin=time.monotonic()
            command=[sys.executable,'-m','rmuc2027_research.rl','train','--config',str(configs/f'{model}.json'),
                     '--run-dir',str(run_dir),'--num-envs',str(num_envs),'--device',device,
                     '--wall-seconds',str(max(.1,available-15.))]
            if attempts and attempts[-1].get('checkpoint'):
                command.extend(['--resume',attempts[-1]['checkpoint'],'--elapsed-wall-seconds',str(spent)])
            report['active_model']=model
            write_json(output/'campaign.json',report)
            with (output/f'{run_dir.name}.log').open('w') as log:
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:process.wait(timeout=max(.1,available-5.))
                except subprocess.TimeoutExpired:stop_worker(process,min(5.,max(0.,deadline-time.monotonic())))
                except BaseException:
                    stop_worker(process,min(5.,max(0.,deadline-time.monotonic())))
                    raise
                finally:stop_worker(process,0.)
            training_path=run_dir/'training.json'
            training=json.loads(training_path.read_text()) if training_path.exists() else {}
            checkpoints=[run_dir/name for name in ('model.zip','interrupted.zip','latest.zip') if (run_dir/name).exists()]
            row=dict(model=model,exit_code=process.returncode,wall_seconds=time.monotonic()-begin,
                     status='trained' if training.get('actor_updated') else 'incomplete',training=training,
                     checkpoint=str(checkpoints[0]) if checkpoints else None)
            report['attempts'].append(row)
            report['results']=[r for r in report['results'] if r['model']!=model]+[row]
            report['elapsed_seconds']=time.monotonic()-started
            write_json(output/'campaign.json',report)
            print(json.dumps(row,ensure_ascii=False),flush=True)
            process=None
        report['status']='finished'
    except KeyboardInterrupt:
        report['status']='interrupted'
        raise
    finally:
        if process is not None:
            stop_worker(process,min(5.,max(0.,deadline-time.monotonic())))
            training_path=run_dir/'training.json'
            training=json.loads(training_path.read_text()) if training_path.exists() else {}
            paths=[run_dir/name for name in ('model.zip','interrupted.zip','latest.zip') if (run_dir/name).exists()]
            row=dict(model=model,exit_code=process.returncode,wall_seconds=time.monotonic()-begin,
                     status='interrupted',training=training,checkpoint=str(paths[0]) if paths else None)
            report['attempts'].append(row)
            report['results']=[r for r in report['results'] if r['model']!=model]+[row]
        report.pop('active_model',None)
        report['elapsed_seconds']=time.monotonic()-started
        report['all_models_updated']=len(report['results'])==len(selected) and all(r.get('status')=='trained' for r in report['results'])
        write_json(output/'campaign.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wall-seconds',type=float,default=7200.)
    parser.add_argument('--num-envs',type=int,default=8)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu',help='PPO network device; physics remains on CUDA')
    parser.add_argument('--resume-campaign',action='store_true',help='Resume with the original global deadline and remaining model allowances')
    parser.add_argument('--models',type=int,nargs='+',default=sorted(models(),key=lambda m:(-weight(models()[m]),m)))
    args=parser.parse_args()
    if args.num_envs<1:parser.error('num-envs must be positive')
    run(args.output,args.models,args.wall_seconds,args.num_envs,device=args.device,resume=args.resume_campaign)


if __name__=='__main__':main()
