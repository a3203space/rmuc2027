"""Explicitly authorized main-only continuation with an absolute wall deadline.

The parent campaign remains immutable. Frozen anchor trials may be reused only
through v4_campaign's separately validated provenance checks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

from .campaign import write_json
from .v4_campaign import child, fit_radar, select_candidate, pool_config


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def prepare(parent, output, started, deadline, gradient_deadline):
    from .asymmetric_ppo import AsymmetricPPO
    from .checkpoints import save_checkpoint
    from .learning import experiment

    parent, output = Path(parent).resolve(), Path(output).resolve()
    package = json.loads((parent/'package.json').read_text())
    if not package.get('training_ready') or digest(package['path']) != package['sha256']:
        raise ValueError('Parent portable bundle must have completed integrity and readiness verification')
    if not started < time.time() < gradient_deadline < deadline or deadline-started > 7200:
        raise ValueError('Invalid or expired absolute continuation budget')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'budget.json', dict(started_unix=started, deadline_unix=deadline,
        gradient_deadline_unix=gradient_deadline, wall_budget_seconds=deadline-started,
        authorization='user: 再运行一个小时; from request receipt, not process start'))
    for name in ('preparation', 'radar-calibration', 'report-tools'):
        shutil.copytree(parent/name, output/name)
    for name in ('preflight.json', 'requirements-runtime.txt', 'runtime-environment.json'):
        shutil.copy2(parent/name, output/name)
    for role in ('main', 'infantry', 'heavy'):
        destination=output/role; destination.mkdir()
        for name in ('model.zip', 'initialized.zip', 'actor.json', 'experiment.json', 'resume-state.pt', 'samplers.json'):
            shutil.copy2(parent/role/name, destination/name)
        for name in ('guidance',):
            if (parent/role/name).is_dir():shutil.copytree(parent/role/name,destination/name)
        shutil.copy2(parent/role/'training.json', destination/'parent-training.json')
    for role in ('main', 'infantry', 'heavy'):
        destination=output/role
        metadata=json.loads((destination/'experiment.json').read_text())
        cfg=metadata['config']; cfg['auxiliary_pool']=pool_config(output)
        cfg['radar_classifier_path']=str(output/'radar-calibration/classifier.json')
        if role == 'main':
            cfg.update(episode_seconds=120, v4_capture_reward=.02)
            cfg['ppo'].update(learning_rate=1e-5, target_kl=.006, ent_coef=.001)
        write_json(destination/'experiment.json', experiment(cfg))
    # Preserve both policy and Adam tensors. Only explicit training scalars change.
    model=AsymmetricPPO.load(output/'main/model.zip',device='cpu')
    before=dict(learning_rate=float(model.lr_schedule(1.)),target_kl=model.target_kl,
        ent_coef=model.ent_coef,samples=model.num_timesteps,optimizer_steps=model.optimizer_steps)
    model.learning_rate=1e-5; model._setup_lr_schedule()
    model.target_kl=.006; model.ent_coef=.001
    save_checkpoint(model,output/'main/model.zip')
    del model
    record=dict(kind='explicit_main_only_continuation',parent_run=str(parent),
        parent_bundle_sha256=package['sha256'], parent_model_sha256=digest(parent/'main/model.zip'),
        parent_main_training=json.loads((parent/'main/training.json').read_text()),
        previous=before,learning_rate=1e-5,target_kl=.006,ent_coef=.001,
        episode_seconds=120,v4_capture_reward=.02,demonstration_updates=0,
        optimizer_reset=False,num_envs=2,helpers_frozen=True,
        reason='Preserve improved folding; reduce goal drift and reward actual self occupancy. Short training episodes broaden opening coverage; they are not full matches.',
        inherited_preflight=dict(source=str(parent/'preflight.json'),sha256=digest(parent/'preflight.json'),
            reason='Same host, devices, dependencies and two-worker topology; no redundant throughput benchmark.'))
    write_json(output/'interventions.json',[record])
    return record


def run(parent,output,started,deadline,gradient_deadline):
    output=Path(output).resolve(); parent=Path(parent).resolve()
    ancestry=prepare(parent,output,started,deadline,gradient_deadline)
    report=dict(status='training',started_unix=started,deadline_unix=deadline,
        gradient_deadline_unix=gradient_deadline,wall_budget_seconds=deadline-started,
        parent_run=str(parent),parent_bundle_sha256=ancestry['parent_bundle_sha256'],
        num_envs=2,stages=[],evaluations=[])
    write_json(output/'campaign.json',report)
    code=child(output,['train-stage','--output',str(output),'--role','main',
        '--deadline-unix',str(gradient_deadline),'--num-envs','2'],gradient_deadline,'main')
    report['stages'].append(dict(stage='main',exit_code=code))
    report['status']='evaluating';write_json(output/'campaign.json',report)
    (output/'evaluation').mkdir()
    jobs=[(backend,side) for backend in ('warp_kinematic','rapier','mjwarp') for side in ('red','blue')]
    # Allocate according to measured frozen-candidate runtime on this same host.
    weights=[max(1.,json.loads((parent/'evaluation'/f'{b}-{s}.json').read_text())[1]['wall_seconds']) for b,s in jobs]
    for index,(backend,side) in enumerate(jobs):
        if time.time()>=deadline-180:break
        until=time.time()+(deadline-180-time.time())*weights[index]/sum(weights[index:])
        code=child(output,['evaluate-pair','--output',str(output),'--backend',backend,'--side',side,
            '--deadline-unix',str(until),'--reuse-anchor-from',str(parent)],until,f'evaluate-{backend}-{side}')
        report['evaluations'].append(dict(backend=backend,side=side,exit_code=code))
        write_json(output/'campaign.json',report)
    fit_radar(output,'radar-calibration-next')
    report['selection']=select_candidate(output)
    report['status']='packaging';write_json(output/'campaign.json',report)
    code=child(output,['package','--output',str(output)],deadline-1,'package')
    package_ok=code==0 and (output/'package.json').exists()
    if package_ok:report['bundle']=json.loads((output/'package.json').read_text())
    else:report['package_error']=dict(exit_code=code,log='package.log')
    complete=all(x.get('exit_code')==0 for x in report['stages']+report['evaluations']) and report['selection']['complete_pairs']==6
    report.update(status=('completed' if complete else 'completed_partial') if package_ok else 'failed_packaging',
        finished_unix=time.time(),elapsed_seconds=time.time()-started,within_wall_budget=time.time()<=deadline,
        all_stages_succeeded=complete)
    write_json(output/'campaign.json',report)
    print(json.dumps(report),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--started-unix',type=float,required=True)
    parser.add_argument('--deadline-unix',type=float,required=True)
    parser.add_argument('--gradient-deadline-unix',type=float,required=True)
    args=parser.parse_args()
    run(args.parent,args.output,args.started_unix,args.deadline_unix,args.gradient_deadline_unix)


if __name__=='__main__':main()
