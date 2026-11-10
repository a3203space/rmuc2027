"""Authorized absolute-deadline continuation with independent, audited rounds.

Each round starts from one complete generation. A frozen Rapier canary controls
continuation only; deployment still requires the full paired acceptance suite.
Rejected branches remain on disk and are never counted as accepted coverage.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .campaign import write_json
from .v6_campaign import sha, child, refresh_manifest, seal_training_report, checkpoint_counters


def run_directory(master,suffix):
    """Keep playable runs at the existing catalog's one-directory depth."""
    if suffix!='final' and not (len(suffix)==3 and suffix[0]=='r' and suffix[1:].isdigit()):
        raise ValueError('Expected round rNN or final suffix')
    master=Path(master).resolve()
    return master.parent/(master.name+'-'+suffix)


def budget_from_authorization(record):
    raw=[record[key] for key in ('authorization_start_unix','deadline_unix','wall_budget_seconds')]
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in raw):
        raise ValueError('Authorization times must be numeric, not booleans')
    start,end,duration=map(float,raw)
    if not all(math.isfinite(v) for v in (start,end,duration)) or end<=start:
        raise ValueError('Finite increasing absolute authorization required')
    if abs(end-start-duration)>1e-6:
        raise ValueError('Authorization duration does not match its endpoints')
    if end-start<7200:raise ValueError('Long continuation needs a separate bounded authorization')
    return dict(started_unix=start,deadline_unix=end,wall_budget_seconds=end-start,
        training_deadline_unix=end-7200,evaluation_deadline_unix=end-1200,
        web_deadline_unix=end-720,other_workloads_may_continue=True,
        scope='prior run completion, preparation, PPO rounds, frozen checks, web, packaging and backups')


def apply_training_control(model,cfg,path):
    """Change a declared learning rate, preserving weights and Adam moments."""
    control=json.loads(Path(path).read_text())
    if set(control)!={'tactical_lr','reason'}:raise ValueError('Unsupported training control')
    value=control['tactical_lr']
    if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or not 0<value<=1e-5:
        raise ValueError('Tactical learning rate must be finite in (0, 1e-5]')
    if not isinstance(control['reason'],str) or not control['reason'].strip():raise ValueError('Intervention reason required')
    if cfg['ppo']['tactical_lr']!=value:raise ValueError('Runtime configuration and intervention disagree')
    old=model.tactical_lr;model.tactical_lr=float(value)
    for group in model.tactical_optimizer.param_groups:group['lr']=float(value)
    return dict(**control,previous_tactical_lr=old,weights_modified=False,adam_moments_reset=False,
        revive_and_critic_unchanged=True,applied_before_first_gradient=True)


def canary_gate(root,case,reference=None):
    from .v6_evaluation import _validate_row
    root=Path(root);rows={};failures=[]
    for member in ('baseline','candidate'):
        path=root/'evaluation'/(case['id']+'-'+member+'.json')
        try:row=json.loads(path.read_text())
        except (OSError,ValueError):row=None
        try:invalid=_validate_row(root,case,member,row)
        except (ValueError,KeyError,TypeError,OSError,AttributeError,OverflowError) as error:invalid=['unverifiable:'+str(error)]
        if invalid:failures.append(dict(member=member,reasons=invalid))
        rows[member]=row
    proposed_reference=None
    if not failures:
        old,new=rows['baseline'],rows['candidate']
        if old['identity']!=new['identity'] or old['prefix_identity']!=new['prefix_identity']:
            failures.append(dict(reasons=['paired_execution_or_initial_state_mismatch']))
        for member,row in rows.items():
            if row['metrics']['tipped_seconds']>0 or row['metrics']['policy_errors']:
                failures.append(dict(member=member,reasons=['tip_or_policy_error']))
        old_adv=old['control'][case['side']]-old['control']['red' if case['side']=='blue' else 'blue']
        new_adv=new['control'][case['side']]-new['control']['red' if case['side']=='blue' else 'blue']
        if old['metrics']['capture_seconds']<=0 or old_adv<=0:
            failures.append(dict(reasons=['baseline_did_not_exercise_capture_and_control']))
        if new['metrics']['capture_seconds']<old['metrics']['capture_seconds']-1:
            failures.append(dict(reasons=['capture_regression']))
        if new_adv<old_adv-1:failures.append(dict(reasons=['control_regression']))
        # One-second tolerance is relative to both the previous accepted model
        # and the first valid anchor; it must not compound once per round.
        proposed_reference=dict(case_id=case['id'],identity=old['identity'],
            prefix_identity=old['prefix_identity'],scenario_hash=old['scenario_hash'],
            capabilities_sha256=old['capabilities_sha256'],
            checkpoint_sha256=old['checkpoint_sha256'],capture_seconds=old['metrics']['capture_seconds'],
            control_advantage=old_adv,source_record=str(root/'evaluation'/(case['id']+'-baseline.json')))
        if reference is not None:
            keys=('case_id','identity','prefix_identity','scenario_hash','capabilities_sha256')
            if any(reference.get(key)!=proposed_reference[key] for key in keys):
                failures.append(dict(reasons=['guard_reference_identity_mismatch']))
            elif new['metrics']['capture_seconds']<reference['capture_seconds']-1 or new_adv<reference['control_advantage']-1:
                failures.append(dict(reasons=['original_guard_anchor_regression']))
        if (old['metrics']['capture_seconds']<=0 or old_adv<=0 or old['metrics']['tipped_seconds']>0
                or old['metrics']['policy_errors'] or old['identity']!=new['identity'] or old['prefix_identity']!=new['prefix_identity']):
            proposed_reference=None
    def capture(row):
        metrics=row.get('metrics') if isinstance(row,dict) else None
        return metrics.get('capture_seconds') if isinstance(metrics,dict) else None
    report=dict(passed=not failures,failures=failures,case_id=case['id'],
        baseline_capture=capture(rows['baseline']),candidate_capture=capture(rows['candidate']),
        reference=reference or proposed_reference,tolerance_seconds=1.,
        purpose='training continuation guard only; not deployment promotion',
        source_sha256={member:sha(root/'evaluation'/(case['id']+'-'+member+'.json'))
            for member in rows if (root/'evaluation'/(case['id']+'-'+member+'.json')).exists()})
    write_json(root/'canary.json',report);return report


def stop_owned(process):
    if process is None or process.poll() is not None:return
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait(timeout=2)


def wait_owned(process,deadline,maximum):
    """Every exit, including interruption, reaps this known child group."""
    try:
        process.wait(timeout=max(.01,min(maximum,deadline-time.time())))
    except subprocess.TimeoutExpired:
        pass
    finally:
        stop_owned(process)
    return process.returncode


def background(root,label,arguments):
    log=(root/(label+'.log')).open('a')
    try:
        return subprocess.Popen([sys.executable,'-u',*arguments],stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    finally:log.close()


def wait_previous(previous,deadline):
    """Preserve the previous experiment's own clock and its active validation."""
    previous=Path(previous);old=json.loads((previous/'budget.json').read_text())
    until=min(deadline,float(old['deadline_unix'])+15)
    while time.time()<until:
        if (previous/'campaign-final.json').is_file():return
        time.sleep(min(5,max(.01,until-time.time())))
    active=previous/'active-worker.json'
    if active.exists():
        row=json.loads(active.read_text());proc=Path('/proc')/str(row['pid'])
        if (proc/'cmdline').exists() and str(previous).encode() in (proc/'cmdline').read_bytes():
            raise RuntimeError('Previous owned worker still active after its deadline; refusing overlapping physical work')


def materialize(source,generation,output,budget,rejected=None):
    from .v6_continuation import materialize_generation
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'budget.json',budget)
    cfg,report=materialize_generation(source,generation,output,rejected_candidate=rejected,
        reason='Continue only from a complete generation; a failed or incomplete canary rejects its branch')
    write_json(output/'campaign.json',dict(status='initialized',budget=budget,initialization=report))
    return cfg


def configure_training(root,cfg,rejections):
    from .learning import experiment
    from .v6_sampling import TrainingCoordinator
    cfg=copy.deepcopy(cfg)
    lr=max(1.25e-6,1e-5*(.5**rejections))
    cfg['ppo']['tactical_lr']=lr
    # Longer ordinary windows observe holding/leaving a point and natural death
    # recovery. Evaluation always overrides this with its declared horizon.
    cfg['decision_window_seconds']=120.
    sources={name:sha(Path(__file__).with_name(name)) for name in ('v6_recovery.py','v6_curriculum.py')}
    cohort='v6-longrun-'+sources['v6_recovery.py'][:16]+'-'+sources['v6_curriculum.py'][:16]
    cfg['v6_recovery_curriculum']=dict(enabled=True,generation=cohort,
        recovery_probability=.4,failure_limit=2,cooldown_seconds=1800)
    coordinator=TrainingCoordinator(root/'sampler-coordinator.json',cfg['seed'])
    if not hasattr(coordinator,'configure_curriculum'):
        raise RuntimeError('Explicit recovery curriculum integration must be installed before long-run training')
    coordinator.configure_curriculum(cfg['v6_recovery_curriculum'],backend=cfg['sim_backend'])
    write_json(root/'training-curriculum-applied.json',dict(source_sha256=sources,
        curriculum=coordinator.snapshot()['curriculum'],source_baseline_generation_unchanged=True,
        probability_basis='episode lease attempts; actor samples and seconds are counted separately'))
    write_json(root/'main/experiment.json',experiment(cfg))
    control=dict(tactical_lr=lr,reason=f'Conservative segmented continuation; {rejections} rejected prior branches')
    write_json(root/'training-control.json',control)
    return control


def final_training_summary(final,rounds,anchor):
    """Account for accepted ancestry separately from rejected research work."""
    old=checkpoint_counters(anchor);new=checkpoint_counters(Path(final)/'main/model.zip')
    accepted=[row for row in rounds if row['accepted_for_continuation']]
    fields={'published_samples':'num_timesteps','published_accepted_actor_steps':'optimizer_steps',
            'published_completed_updates':'completed_updates'}
    differences={key:new[counter]-old[counter] for key,counter in fields.items()}
    if any(value<0 or value!=sum(row['training'][key] for row in accepted)
           for key,value in differences.items()):
        raise ValueError('Accepted lineage counters disagree with final checkpoint')
    return dict(status='accepted_lineage_frozen_for_evaluation',
        actual_samples=sum(row['training'].get('actual_samples',0) for row in accepted),
        observed_samples_all_branches=sum(row['training'].get('actual_samples',0) for row in rounds),
        rejected_branch_observed_samples=sum(row['training'].get('actual_samples',0) for row in rounds if not row['accepted_for_continuation']),
        total_samples=new['num_timesteps'],anchor_samples=old['num_timesteps'],
        accepted_actor_steps=differences['published_accepted_actor_steps'],
        completed_updates=differences['published_completed_updates'],
        rejected_actor_steps=sum(row['training'].get('rejected_actor_steps',0) for row in accepted),
        skill_unchanged=all(row['training']['published_skill_unchanged'] for row in accepted),
        demonstration_updates_during_ppo=0,physics_steps_in_final_run_training=0,
        published_checkpoint_sha256=sha(Path(final)/'main/model.zip'),**differences)


def archive_round_evidence(final,rounds,*,replay_byte_limit=512*1024*1024):
    """Copy frozen evidence, never checkpoints/Git trees or relabeled replay.

    Training replay is prioritized by accepted ancestry, then round recency and
    size. All omitted native replays retain a source digest/size/reason. The
    index is audit evidence, not a promise that every trial fits in the quota.
    """
    from rmuc2027_core.replay import ReplayArchive
    final=Path(final).resolve()
    if isinstance(replay_byte_limit,bool) or not isinstance(replay_byte_limit,int) or replay_byte_limit<0:
        raise ValueError('Nonnegative integer replay byte limit required')
    records=[];receipts=[];candidates=[];seen=set()
    identifiers=[row['round'] for row in rounds]
    if any(isinstance(i,bool) or not isinstance(i,int) or i<1 for i in identifiers) or len(set(identifiers))!=len(identifiers):
        raise ValueError('Round identifiers must be unique positive integers')

    def include(source,destination,entry):
        if destination in seen:raise ValueError('Evidence destination collision: '+str(destination))
        seen.add(destination)
        if destination.exists():
            if sha(destination)!=entry['sha256']:
                raise ValueError('Existing evidence hash collision: '+str(destination))
        else:
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,destination)
        if destination.stat().st_size!=entry['bytes'] or sha(destination)!=entry['sha256']:
            raise ValueError('Evidence changed while archiving: '+str(source))
        entry.update(included=True,destination=destination.relative_to(final).as_posix(),
            destination_sha256=entry['sha256'],raw_bytes_unchanged=True)

    fixed=('main/updates.jsonl','main/episodes.jsonl','main/training.json','main/experiment.json',
        'main/migration.json','main/checkpoint-ready.json','sampler-coordinator.json',
        'training-control.json','training-control-applied.json','training-curriculum-applied.json',
        'canary.json','run-results.json','preflight.json','runtime-environment.json','requirements-runtime.txt',
        'replays/training/recordings.jsonl')
    for row in rounds:
        source_root=Path(row['path']).resolve();tag=f"round-{row['round']:02d}"
        audit=final/'review/round-evidence'/tag
        paths=[source_root/name for name in fixed]
        for folder in ('main/completed-episodes','main/prefix-reports'):
            paths.extend(sorted((source_root/folder).glob('*.json')))
        for pattern in ('main-rapier-blue-*.json','main-rapier-blue-*.jsonl'):
            paths.extend(sorted((source_root/'evaluation').glob(pattern)))
        receipt=source_root/'github-backups/receipt.json'
        if receipt.is_file():paths.append(receipt)
        for path in paths:
            if not path.is_file() or path.is_symlink():continue
            relative=path.relative_to(source_root)
            entry=dict(round=row['round'],accepted_for_continuation=row['accepted_for_continuation'],
                kind='backup_receipt' if path==receipt else 'round_log_or_trace',source=str(path),
                sha256=sha(path),bytes=path.stat().st_size)
            # Copy just the receipt, never its Git tree, checkpoint ZIP or parts.
            target=audit/('backup-receipt.json' if path==receipt else relative)
            include(path,target,entry);records.append(entry)
            if path==receipt:
                try:value=json.loads(path.read_text())
                except (OSError,ValueError):value={}
                receipts.append(dict(round=row['round'],source=str(path),sha256=entry['sha256'],
                    destination=entry['destination'],url=value.get('url'),commit=value.get('commit'),
                    remote_verified=value.get('remote_verified') is True,
                    checkpoint_samples=(value.get('checkpoint') or {}).get('samples'),
                    checkpoint_sha256=((value.get('checkpoint') or {}).get('files') or {}).get('model.zip')))
        for kind in ('training','evaluation'):
            for path in sorted((source_root/'replays'/kind).glob('*.json')):
                if path.is_symlink():continue
                entry=dict(round=row['round'],accepted_for_continuation=row['accepted_for_continuation'],
                    kind='training_replay' if kind=='training' else 'canary_replay',
                    source=str(path),sha256=sha(path),bytes=path.stat().st_size,included=False)
                target=(final/'replays/training'/(tag+'--'+path.name) if kind=='training'
                    else audit/'replays/evaluation'/path.name)
                candidates.append((kind,row,path,target,entry))

    used=0;training_count=0
    candidates.sort(key=lambda item:(item[0]!='training',not item[1]['accepted_for_continuation'],
        -item[1]['round'],item[4]['bytes'],item[2].name))
    for kind,row,path,target,entry in candidates:
        if used+entry['bytes']>replay_byte_limit:
            entry['omission_reason']='native_replay_archive_byte_limit'
            records.append(entry);continue
        try:
            replay=ReplayArchive(path);metadata=replay.metadata
            expected='training' if kind=='training' else 'evaluation'
            if metadata.get('kind')!=expected:
                raise ValueError('Source folder and recorded replay kind disagree')
            if kind=='training':
                training=metadata.get('training_recording') or {}
                start=float(training['learner_start_time']);end=float(metadata['recording_end_time'])
                if not math.isfinite(start) or not math.isfinite(end) or end<=start or training.get('policy_changes_during_episode') is not True:
                    raise ValueError('No recorded nonempty learner-controlled interval')
                entry['learner_start_time']=start;entry['recording_end_time']=end
            if len(replay.offsets)<2:raise ValueError('No recorded transition')
            entry.update(recorded_kind=metadata['kind'],frames=len(replay.offsets),
                recording_complete=bool(metadata.get('recording_complete')),truncated=bool(metadata.get('truncated')),
                policy_update_range=metadata.get('policy_update_range'))
        except (OSError,ValueError,KeyError,TypeError) as error:
            entry['omission_reason']='invalid_or_nontraining_recording: '+str(error)
            records.append(entry);continue
        include(path,target,entry);records.append(entry)
        used+=entry['bytes'];training_count+=kind=='training'
    report=dict(version=1,raw_evidence_unchanged=True,records=records,backup_receipts=receipts,
        replay_byte_limit=replay_byte_limit,included_native_replay_bytes=used,
        included_training_replays=training_count,training_playback_available=training_count>0,
        training_playback_unavailable_reason=None if training_count else 'No valid nonempty training replay fit the fixed quota; evaluation is not relabeled.',
        excluded_recursive_content=['checkpoints','Git objects','ZIP files','backup parts'],
        selection_scope='Archived rejected-branch evidence is not accepted training coverage')
    (final/'review').mkdir(exist_ok=True)
    write_json(final/'review/round-evidence-index.json',report)
    return report


def shared_preflight_report(report,source):
    source=Path(source)
    return dict(report or dict(training_ready=False),shared_preflight=True,source_run=str(source),
        source_report_sha256=sha(source/'preflight.json') if (source/'preflight.json').is_file() else None,
        new_hardware_measurement=False,current_checkpoint_inference_rechecked=False,
        scope='Prior round real workload rates; final checkpoint receives separate relocated bundle validation')


def audit_final_backup(root):
    root=Path(root);path=root/'github-backups/receipt.json'
    try:receipt=json.loads(path.read_text())
    except (OSError,ValueError):receipt={}
    package=json.loads((root/'package.json').read_text()) if (root/'package.json').is_file() else {}
    ready=json.loads((root/'main/checkpoint-ready.json').read_text())
    checkpoint=receipt.get('checkpoint') or {};archive=receipt.get('final') or {}
    reasons=[]
    if receipt.get('remote_verified') is not True:reasons.append('remote_not_verified')
    if not package.get('sha256') or archive.get('sha256')!=package['sha256']:reasons.append('final_zip_hash_mismatch_or_absent')
    if checkpoint.get('files')!=ready['sha256'] or checkpoint.get('samples')!=ready['samples'] or checkpoint.get('update_id')!=ready['update_id']:
        reasons.append('published_generation_or_counters_mismatch')
    report=dict(verified=not reasons,reasons=reasons,receipt_sha256=sha(path) if path.is_file() else None,
        url=receipt.get('url'),commit=receipt.get('commit'),checkpoint_samples=checkpoint.get('samples'),
        final_zip_sha256=archive.get('sha256'),scope='Post-package external receipt, not a self-referential ZIP member')
    write_json(root/'backup-verification.json',report)
    return report


def evaluate(root,case,member,deadline,label=None):
    label=label or case['id']+'-'+member
    return child(root,'evaluate',deadline,(label,['--case',case['id'],'--member',member,
        '--deadline-unix',str(deadline-2)]))


def run(args):
    from .v6_evaluation import required_cases,select_candidate
    from rmuc2027_core.paths import ROOT
    root=args.output.resolve();authorization=json.loads((root/'authorization.json').read_text())
    budget=budget_from_authorization(authorization);deadline=budget['deadline_unix'];cutoff=budget['training_deadline_unix']
    if time.time()>=cutoff-600:raise TimeoutError('Authorization leaves insufficient training time')
    write_json(root/'budget.json',budget)
    status=dict(status='waiting_for_prior_run',budget=budget,rounds=[],previous_run=str(args.previous.resolve()),
        automatic_deployment=False,canary_case='main-rapier-blue',max_rounds=8)
    write_json(root/'campaign.json',status)
    wait_previous(args.previous,deadline)
    source=args.previous.resolve();generation='main/checkpoints/baseline';best_source=source;best_generation=generation
    anchor=source/'main/checkpoints/baseline/model.zip'
    rejected=str(source/'main/model.zip')
    cases=required_cases();canary=next(c for c in cases if c['id']=='main-rapier-blue')
    rejections=0;selected_envs=1;preflight=None;round_path=None;backup=None;web=None;reference=None
    try:
        for index in range(1,9):
            if time.time()>=cutoff-1800:break
            round_path=run_directory(root,f'r{index:02d}')
            cfg=materialize(best_source,best_generation,round_path,budget,rejected)
            control=configure_training(round_path,cfg,rejections)
            if index==1:
                status.update(status='preflight',active_run=str(round_path));write_json(root/'campaign.json',status)
                until=min(cutoff-1800,time.time()+1200)
                result=child(round_path,'preflight',until,('preflight',['--deadline-unix',str(until-5)]))
                preflight=json.loads((round_path/'preflight.json').read_text()) if (round_path/'preflight.json').exists() else {}
                if not preflight.get('training_ready'):
                    status['preflight_failure']=result;break
                selected_envs=int(preflight['selected_num_envs'])
            else:
                write_json(round_path/'preflight.json',shared_preflight_report(preflight,run_directory(root,'r01')))
            status.update(status='training',active_run=str(round_path),selected_num_envs=selected_envs)
            write_json(root/'campaign.json',status)
            write_json(round_path/'campaign.json',dict(status='training',budget=budget,control=control))
            backup=background(round_path,'backup',['-m','rmuc2027_research.rl.v6_backup','--run',str(round_path),
                '--deadline-unix',str(deadline-10),'--loop'])
            train_until=min(cutoff-1200,time.time()+3600)
            trained=child(round_path,'train',train_until,('train',['--deadline-unix',str(train_until-2),
                '--num-envs',str(selected_envs),'--training-control',str(round_path/'training-control.json')]))
            ready=seal_training_report(round_path)
            status['status']='canary';write_json(root/'campaign.json',status)
            write_json(round_path/'campaign.json',dict(status='evaluating',budget=budget))
            evaluations=[]
            for member in ('baseline','candidate'):
                remaining=cutoff-time.time()-5
                until=min(cutoff-2,time.time()+min(600,remaining/(2 if member=='baseline' else 1)))
                evaluations.append(dict(member=member,**evaluate(round_path,canary,member,until)))
            guard=canary_gate(round_path,canary,reference)
            if reference is None and guard['reference'] is not None:
                reference=guard['reference'];write_json(root/'canary-reference.json',reference)
            training=json.loads((round_path/'main/training.json').read_text())
            accepted=(guard['passed'] and training['published_samples']>0
                and training['published_accepted_actor_steps']>0 and training['published_skill_unchanged']
                and training['published_checkpoint_sha256']==ready['sha256']['model.zip'])
            row=dict(round=index,path=str(round_path),source_run=str(best_source),source_generation=best_generation,
                train=trained,training=training,checkpoint=ready,canary=guard,evaluations=evaluations,
                accepted_for_continuation=accepted,control=control)
            status['rounds'].append(row)
            if accepted:
                best_source,best_generation=round_path,ready['directory'];rejected=None
            else:
                rejections+=1;rejected=str(round_path/'main/model.zip')
            write_json(round_path/'run-results.json',dict(status='results_frozen',**row))
            write_json(round_path/'campaign.json',dict(status='results_frozen',**row))
            refresh_manifest(round_path)
            stop_owned(backup);backup=None
            # Force a bounded snapshot of the final accepted/rejected boundary.
            final_snapshot=background(round_path,'boundary-backup',['-m','rmuc2027_research.rl.v6_backup',
                '--run',str(round_path),'--deadline-unix',str(min(cutoff,time.time()+120))])
            wait_owned(final_snapshot,cutoff,120)
            status.update(best_source=str(best_source),best_generation=best_generation,rejected_branches=rejections)
            write_json(root/'campaign.json',status)

        final=run_directory(root,'final')
        materialize(best_source,best_generation,final,budget,rejected)
        shutil.copy2(anchor,final/'main/initialized.zip')
        shutil.copy2(source/'main/checkpoints/baseline/actor.json',final/'main/initialized-actor.json')
        # Final inference is compared against the original pre-extension anchor.
        # The current model/RNG/optimizers remain the accepted generation.
        write_json(final/'main/evaluation-anchor.json',dict(source_run=str(source),
            source_generation='main/checkpoints/baseline',model_sha256=sha(anchor),
            current_resume_generation=str(best_source/best_generation),
            scope='Original pre-extension inference anchor; optimizer resume remains the accepted generation'))
        write_json(final/'main/training.json',final_training_summary(final,status['rounds'],anchor))
        seal_training_report(final)
        # Freeze this host's actual dependencies; materialization intentionally
        # copies state/resources only and does not claim target-hardware tests.
        from .v6_preflight import freeze_environment
        freeze_environment(final)
        (final/'review').mkdir(exist_ok=True)
        write_json(final/'review/continuation-rounds.json',status)
        archived=archive_round_evidence(final,status['rounds'])
        write_json(final/'authorization.json',authorization)
        write_json(final/'required-cases.json',cases)
        write_json(final/'preflight.json',shared_preflight_report(preflight,run_directory(root,'r01')))
        refresh_manifest(final)
        report=dict(status='evaluating',budget=budget,evaluations=[],round_history=str(root/'campaign.json'),
            selected_continuation_source=str(best_source),selected_continuation_generation=best_generation,
            observed_samples_all_branches=sum(r['training'].get('actual_samples',0) for r in status['rounds']),
            accepted_model_samples=json.loads((final/'main/checkpoint-ready.json').read_text())['samples'],
            training_evidence=dict(index='review/round-evidence-index.json',
                training_playback_available=archived['training_playback_available'],
                included_native_replay_bytes=archived['included_native_replay_bytes']),
            no_overall_win_rate_claim=True)
        status.update(status='evaluating',active_run=str(final));write_json(root/'campaign.json',status)
        write_json(final/'campaign.json',report)
        backup=background(final,'backup',['-m','rmuc2027_research.rl.v6_backup','--run',str(final),
            '--deadline-unix',str(deadline-10),'--loop'])
        web=background(final,'web-watcher',[str(ROOT/'scripts/v6_web_watch.py'),'--run',str(final),
            '--start-service','--port',str(args.port),'--playwright-module',str(args.playwright_module)])
        jobs=[(case,member) for case in cases for member in ('baseline','candidate')]
        rates=(preflight or {}).get('evaluation_rates',{})
        def weight(c):
            return (c['seconds']+(60 if c['recovery_kind'] in ('free_ready','alive_weak') else 45 if c['recovery_kind'] else 0))/max(.05,rates.get(c['backend'],.2))
        for index,(case,member) in enumerate(jobs):
            remaining=budget['evaluation_deadline_unix']-time.time()
            if remaining<20:break
            until=time.time()+remaining*weight(case)/sum(weight(c) for c,_ in jobs[index:])
            report['evaluations'].append(dict(case_id=case['id'],member=member,**evaluate(final,case,member,until)))
            write_json(final/'campaign.json',report)
        report['selection']=select_candidate(final);refresh_manifest(final)
        report['status']='web_acceptance';write_json(final/'campaign.json',report)
        write_json(final/'web-acceptance-request.json',dict(run_id=final.name,
            deadline_unix=budget['web_deadline_unix'],required=['frozen_live_match','saved_training_or_evaluation_replay']))
        stop_owned(backup);backup=None
        while time.time()<budget['web_deadline_unix']-5 and not (final/'web-acceptance.json').exists():time.sleep(1)
        report['web_acceptance']=json.loads((final/'web-acceptance.json').read_text()) if (final/'web-acceptance.json').exists() else dict(status='not_completed')
        stop_owned(web);web=None
        report.update(status='results_frozen',results_frozen_unix=time.time(),training=json.loads((final/'main/training.json').read_text()))
        write_json(final/'campaign.json',report);write_json(final/'run-results.json',report)
        packaged=child(final,'package',deadline-100,('package',['--deadline-unix',str(deadline-105)]))
        report['package']=packaged
        if packaged.get('exit_code')==0 and time.time()<deadline-15:
            backup=background(final,'final-backup',['-m','rmuc2027_research.rl.v6_backup','--run',str(final),
                '--deadline-unix',str(deadline-3),'--final',str(final.with_suffix('.zip'))])
            report['final_backup_exit_code']=wait_owned(backup,deadline-2,deadline-time.time());backup=None
        report['final_backup_verification']=audit_final_backup(final)
        report.update(status='completed' if packaged.get('exit_code')==0 and not report['selection']['missing']
            and report['web_acceptance'].get('status')=='passed'
            and report['final_backup_verification']['verified'] else 'completed_partial',
            finished_unix=time.time(),within_wall_budget=time.time()<=deadline)
        write_json(final/'campaign-final.json',report)
        status.update(status=report['status'],final_run=str(final),finished_unix=time.time(),within_wall_budget=time.time()<=deadline)
        write_json(root/'campaign-final.json',status);write_json(root/'campaign.json',status)
    finally:
        stop_owned(backup);stop_owned(web)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8768)
    parser.add_argument('--playwright-module',type=Path,required=True)
    args=parser.parse_args()
    def interrupt(signum,_frame):
        raise KeyboardInterrupt('Signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupt)
    signal.signal(signal.SIGINT,interrupt)
    try:run(args)
    except BaseException as error:
        args.output.mkdir(parents=True,exist_ok=True)
        failure=dict(error=type(error).__name__+': '+str(error),time_unix=time.time())
        write_json(args.output/'orchestrator-error.json',failure)
        try:status=json.loads((args.output/'campaign.json').read_text())
        except (OSError,ValueError):status={}
        status.update(status='failed',failure=failure)
        write_json(args.output/'campaign.json',status)
        raise


if __name__=='__main__':main()
