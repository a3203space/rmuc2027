"""Continue a completed campaign with public skill guidance and a fixed deadline."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from rmuc2027_core import sdk
from .campaign import models,write_json,stop_worker
from .intervention import collect,correct
from .validation import frozen_model
from .learning import experiment


def load_deadline(path):
    value=json.loads(Path(path).read_text())['deadline_unix']
    if not math.isfinite(value):raise ValueError('Invalid real-time deadline')
    return time.monotonic()+max(0.,value-time.time())


def probe_seeds(design, round_id):
    # Curriculum selection uses both seed % 3 and seed % 4. Keep the
    # course unchanged when varying the seed after a failed probe.
    base=47+12*round_id
    return [base,base+1] if design.get('capabilities',{}).get('jump') else [base]


def probe_passed(episodes,seeds):
    expected={(seed,side) for seed in seeds for side in ('red','blue')}
    actual={(row['seed'],row['side']) for row in episodes}
    return actual==expected and len(episodes)==len(expected) and all(
        row['finished'] and row['course']['success'] and row['metrics']['tipped_seconds']==0
        for row in episodes)


def changed_probe_courses(row):
    probes=[r for r in row['rounds'] if r['kind']=='frozen-probe']
    if not probes:return False
    original={(e['side'],e['course']['kind']) for e in probes[0]['episodes']}
    return any({(e['side'],e['course']['kind']) for e in p['episodes']}!=original for p in probes[1:])


def select_probe(rounds):
    probes=[r for r in rounds if r['kind']=='frozen-probe']
    if not probes:return None
    def score(row):
        episodes=row['episodes']
        return (sum(e['finished'] and e['course']['success'] and e['metrics']['tipped_seconds']==0 for e in episodes),
                -sum(e['metrics']['tipped_seconds'] for e in episodes),
                -sum(e['metrics']['alive_seconds'] for e in episodes))
    return max(probes,key=score)


def prepare(initial, output, deadline_file, resume=False):
    initial,output=Path(initial),Path(output)
    output.mkdir(parents=True,exist_ok=resume)
    deadline=load_deadline(deadline_file)
    report=dict(status='preparing',results=[],errors=[],deadline_file=str(deadline_file))
    repairs={}
    if resume:
        report=json.loads((output/'preparation.json').read_text())
        report['status']='preparing'
        for row in report['results']:
            if changed_probe_courses(row) and not row.get('course_validation_repaired'):
                row['previous_pass_invalidated']='Probe seed changed curriculum kind between rounds'
                repairs[row['model']]=row
            else:
                selected=select_probe(row['rounds'])
                if selected:
                    row.update(checkpoint=selected['checkpoint'],verification_passed=selected['passed'])
        report.setdefault('resume_history',[]).append(dict(time_unix=time.time(),repair_models=list(repairs)))
    seen={r['model'] for r in report['results'] if r['model'] not in repairs}
    write_json(output/'preparation.json',report)
    while time.monotonic()<deadline-90:
        campaign=json.loads((initial/'campaign.json').read_text())
        for row in campaign['results']:
            model=row['model']
            if model in seen or not row.get('checkpoint'):continue
            if row['status'] not in ('trained','incomplete'):continue
            started=time.monotonic();folder=output/str(model)
            if folder.exists():folder=folder/f'repair-{time.time_ns()}'
            folder.mkdir(parents=True)
            design=models()[model]
            capable=design.get('chassis_type')=='wheel_legged' or design.get('transform_allowed')
            allowance=300. if design['role']=='heavy' else 180.
            end=min(deadline-90,started+allowance)
            try:
                repairing=repairs.get(model)
                checkpoint=repairing['checkpoint'] if repairing else row['checkpoint']
                actor,metadata=frozen_model(checkpoint,allow_sdk_transfer=True)
                if row['status']=='incomplete' and actor.num_timesteps==0:
                    raise ValueError('Incomplete initial checkpoint has no training samples')
                paths=[];rounds=[]
                if repairing:
                    paths=[Path(p) for p in metadata['config'].get('skill_guidance_datasets',[])]
                    current=Path(checkpoint).parent
                elif capable:
                    # Reuse the audited pilot's public observations for W3.
                    pilot=Path('data/research/locomotion-intervention-pilot-v2/3')
                    if model==3 and (pilot/'dagger/demonstrations.npz').exists():
                        paths=[pilot/'teacher/demonstrations.npz',pilot/'dagger/demonstrations.npz']
                    else:
                        paths=[collect(metadata['config'],folder/'teacher',backend='rapier',
                            deadline=min(end-20.,time.monotonic()+allowance*.5))]
                if not repairing:
                    current=folder/'warm-0'
                    fit=correct(checkpoint,current,paths,max_seconds=max(0.,min(20.,end-time.monotonic())))
                    rounds.append(dict(kind='initial-correction',fit=fit))
                passed=None
                if capable:
                    for round_id in range(3):
                        if time.monotonic()>=end-10:break
                        actor,metadata=frozen_model(current/'model.zip')
                        seeds=probe_seeds(design,round_id)
                        sample=collect(metadata['config'],folder/f'actor-states-{round_id}',
                            seeds=seeds,backend='rapier',actor=actor,deadline=end-5.)
                        outcomes=json.loads((sample.parent/'episodes.json').read_text())
                        passed=probe_passed(outcomes,seeds)
                        rounds.append(dict(kind='frozen-probe',checkpoint=str(current/'model.zip'),episodes=outcomes,passed=passed))
                        if passed or round_id==2:break
                        paths.append(sample)
                        following=folder/f'warm-{round_id+1}'
                        fit=correct(current/'model.zip',following,paths,max_seconds=max(0.,min(15.,end-time.monotonic())))
                        current=following
                        rounds.append(dict(kind='failure-state-correction',fit=fit))
                selected=select_probe(rounds)
                if selected:
                    current=Path(selected['checkpoint']).parent;passed=selected['passed']
                result=dict(model=model,status='ready',checkpoint=str(current/'model.zip'),
                            original_checkpoint=row['checkpoint'],rounds=rounds,wall_seconds=time.monotonic()-started,
                            verification_passed=passed,course_validation_repaired=bool(repairing),
                            initial_status=row['status'])
                if repairing:
                    result['previous_preparation']=repairing
                    report['results']=[r for r in report['results'] if r['model']!=model]
                report['results'].append(result)
                write_json(folder/'preparation.json',result)
                print(json.dumps(dict(model=model,status='ready',wall_seconds=result['wall_seconds'])),flush=True)
            except Exception as error:
                report['errors'].append(dict(model=model,error=f'{type(error).__name__}: {error}'))
                print(json.dumps(report['errors'][-1]),flush=True)
            seen.add(model)
            write_json(output/'preparation.json',report)
        if campaign['status']=='finished':break
        if campaign['status'] not in ('running','finished'):
            report['status']='initial_campaign_interrupted';break
        time.sleep(5)
    if report['status']=='preparing':report['status']='finished'
    report['all_models_prepared']=set(r['model'] for r in report['results'])==set(campaign['models'])
    write_json(output/'preparation.json',report)


def continuation_weight(model):
    design=models()[model]
    if design['role']=='heavy':return 4 if design.get('chassis_type')=='wheel_legged' else 3
    return 2 if design.get('chassis_type')=='wheel_legged' or design.get('transform_allowed') else 1


def continue_training(initial, prepared, output, deadline_file, resume=False):
    initial,prepared,output=Path(initial),Path(prepared),Path(output)
    deadline=load_deadline(deadline_file)
    if resume:
        previous=json.loads((output/'campaign.json').read_text())
        if previous['results'] or previous['status']!='waiting':
            raise ValueError('Only an unstarted waiting continuation can be resumed')
    output.mkdir(parents=True,exist_ok=resume)
    report=dict(status='waiting',deadline_file=str(deadline_file),results=[],attempts=[],
                physics_device='cuda:0',policy_device='cpu',intervention='public demonstrations + failed-state correction + PPO skill anchor')
    write_json(output/'campaign.json',report)
    original=json.loads((initial/'campaign.json').read_text());progress={}
    while time.monotonic()<deadline-90:
        original=json.loads((initial/'campaign.json').read_text())
        progress=json.loads((prepared/'preparation.json').read_text()) if (prepared/'preparation.json').exists() else {}
        if original['status']=='finished' and progress.get('status')=='finished':break
        if original['status'] not in ('running','finished'):
            raise RuntimeError('Resolve the initial campaign interruption before continuing')
        time.sleep(5)
    ready=progress.get('results',[])
    # Preserve enough time for full heavy assembly and subsequent learning.
    ready.sort(key=lambda row:(-continuation_weight(row['model']),row['model']))
    started=time.monotonic();report['status']='running';report['started_unix']=time.time()
    report['models']=[row['model'] for row in ready]
    report['missing_models']=sorted(set(original['models'])-set(report['models']))
    report['weighting']={row['model']:continuation_weight(row['model']) for row in ready}
    process=None
    try:
        for index,row in enumerate(ready):
            model=row['model']
            # An operator may improve a not-yet-started model while another
            # chassis is training. Resolve its prepared checkpoint at handoff.
            latest=json.loads((prepared/'preparation.json').read_text())
            row=next((r for r in latest['results'] if r['model']==model),row)
            available=deadline-90-time.monotonic()
            remaining=sum(continuation_weight(r['model']) for r in ready[index:])
            budget=available*continuation_weight(model)/remaining
            if budget<=20:
                report['results'].append(dict(model=model,status='budget_exhausted'));continue
            folder=output/str(model)
            metadata=json.loads((Path(row['checkpoint']).parent/'experiment.json').read_text())
            # Assembly may consume an entire match without installing the
            # launcher. Guarantee locomotion updates before attempting it.
            fraction=.7 if models()[model]['role']=='heavy' else .5
            config=dict(metadata['config'],curriculum_seconds=max(0.,budget-15.)*fraction)
            # Resume requires exact configuration provenance. Record an explicit
            # stage-schedule transition while preserving the checkpoint bytes.
            initialization=output/f'{model}-initialization';initialization.mkdir()
            shutil.copy2(row['checkpoint'],initialization/'model.zip')
            rebound=experiment(config)
            rebound['stage_schedule_transition']=dict(parent_checkpoint=row['checkpoint'],
                parent_experiment=metadata,previous_seconds=metadata['config'].get('curriculum_seconds',0.),
                current_seconds=config['curriculum_seconds'],weights_unchanged=True,
                training_sdk_hash=metadata['sdk_hash'],continuation_sdk_hash=rebound['sdk_hash'],
                sdk_transfer=metadata['sdk_hash']!=rebound['sdk_hash'])
            write_json(initialization/'experiment.json',rebound)
            config_path=output/f'{model}-config.json'
            write_json(config_path,config)
            command=[sys.executable,'-m','rmuc2027_research.rl','train','--config',str(config_path),
                     '--run-dir',str(folder),'--resume',str(initialization/'model.zip'),'--num-envs','1','--device','cpu',
                     '--wall-seconds',str(budget-15.)]
            begin=time.monotonic();report['active_model']=model
            write_json(output/'campaign.json',report)
            with (output/f'{model}.log').open('w') as log:
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:process.wait(timeout=max(.1,budget-5.))
                except subprocess.TimeoutExpired:stop_worker(process,min(5.,max(0.,deadline-time.monotonic())))
                except BaseException:
                    stop_worker(process,min(5.,max(0.,deadline-time.monotonic())));raise
                finally:stop_worker(process,0.)
            path=folder/'training.json'
            training=json.loads(path.read_text()) if path.exists() else {}
            candidates=[folder/name for name in ('model.zip','interrupted.zip','latest.zip') if (folder/name).exists()]
            result=dict(model=model,status='trained' if training.get('actor_updated') else 'incomplete',
                checkpoint=str(candidates[0]) if candidates else row['checkpoint'],
                guided_initialization=row['checkpoint'],training=training,
                wall_seconds=time.monotonic()-begin,exit_code=process.returncode)
            report['results'].append(result);report['attempts'].append(result)
            write_json(output/'campaign.json',report)
            print(json.dumps(result),flush=True);process=None
        report['status']='finished'
    except KeyboardInterrupt:
        report['status']='interrupted';raise
    except Exception as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:
        if process is not None:stop_worker(process,min(5.,max(0.,deadline-time.monotonic())))
        report.pop('active_model',None)
        report['wall_seconds']=time.monotonic()-started
        report['all_models_updated']=bool(ready) and not report['missing_models'] and len(report['results'])==len(ready) and all(r['status']=='trained' for r in report['results'])
        write_json(output/'campaign.json',report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','continue'])
    parser.add_argument('--initial',type=Path,required=True)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--deadline-file',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.stage=='prepare':prepare(args.initial,args.prepared,args.deadline_file,args.resume)
    else:
        if args.output is None:parser.error('continue requires --output')
        continue_training(args.initial,args.prepared,args.output,args.deadline_file,args.resume)


if __name__=='__main__':main()
