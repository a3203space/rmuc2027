"""Declared v6 paired cases and finite-evidence promotion, never a win-rate claim."""
import copy
import hashlib
import json
import math
from pathlib import Path
import time

from .campaign import write_json
from .scenarios import ScenarioSpec

REPLAY_BYTE_LIMIT = 2 * 1024**3


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def _execution_identity(cfg, provenance):
    """Actual execution inputs; wall deadlines and training-only options differ."""
    from .session import resource
    excluded = {'_training_deadline', '_v6_sampler_path', '_v6_replay_dir', '_episode_log_dir',
                '_worker_id', 'ppo', 'github_backup_seconds'}
    normalized = {key: value for key, value in cfg.items() if key not in excluded}
    helpers = {role: _file_hash(resource(path)) for role, path in cfg.get('auxiliary_pool', {}).items()}
    normalized['auxiliary_pool'] = helpers
    if cfg.get('radar_classifier_path'):
        normalized['radar_classifier_path'] = _file_hash(resource(cfg['radar_classifier_path']))
    for key in ('friendly', 'opponent'):
        normalized[key] = provenance['sources'][key]
    runtime = {key: provenance.get(key) for key in
               ('sdk_hash', 'runtime_hash', 'engine_binary_sha256', 'rules_sha256', 'simulation', 'sources')}
    runtime['sentry_sources'] = {key: value for key, value in provenance.get('sentry_sources', {}).items()
        if not key.endswith(('/v6_training.py', '/v6_sampling.py'))}
    from rmuc2027_core.paths import ROOT
    runtime['control_pipeline'] = {str(path.relative_to(ROOT)):_file_hash(path) for path in (
        Path(__file__).with_name('session.py'), Path(__file__).with_name('env.py'),
        Path(__file__).parents[1]/'strategies/battle.py',
        ROOT/'packages/core/rmuc2027_core/engine.py', ROOT/'packages/core/rmuc2027_core/policy.py')}
    return dict(config_hash=_digest(normalized), execution_config=normalized, execution_hash=_digest(runtime), helper_hashes=helpers,
        radar_hash=normalized.get('radar_classifier_path'), sources=runtime)


def _prefix_identity(battle, info):
    prefix = info.get('recovery_prefix') or {}
    return dict(commands_hash=_digest(battle.actions), handoff_hash=_digest(battle.match._frame),
        legal_history_hash=prefix.get('legal_history_sha256'), origin=float(info['time']),
        prefix_kind=prefix.get('kind'), prepared=prefix.get('prepared', True))


def _replay_bytes(root):
    folder = Path(root)/'replays'
    return sum(path.stat().st_size for path in folder.rglob('*') if path.is_file()) if folder.exists() else 0


def _recording_would_exceed(battle, used_bytes, byte_limit=REPLAY_BYTE_LIMIT):
    # 10 seats, <=1 KiB each, plus static/metadata allowance. Stop recording
    # before writing past the shared cap; continue simulation as a censored
    # recording and never label its last stored frame as the trial's endpoint.
    estimate = sum(size for _, size in battle.frames.offsets)+len(battle.actions)*10240+1024**2
    return used_bytes+estimate > byte_limit


def _revive_advantage(old, new):
    paid = [x for x in (new.get('recovery') or {}).get('death_episodes', []) if x.get('observed_mode') == 'paid']
    free = [x for x in (old.get('recovery') or {}).get('death_episodes', []) if x.get('observed_mode') == 'free']
    if not paid or not free: return None
    a, b = free[0], paid[0]
    try:
        old_delay = float(a['revived_at'])-float(old['learner_start_time'])
        new_delay = float(b['revived_at'])-float(new['learner_start_time'])
        hp_gain = float(b['initial_hp'])-float(a['initial_hp'])
        cost = float(b['cost'])
    except (KeyError, TypeError, ValueError): return None
    if not all(math.isfinite(x) for x in (old_delay, new_delay, hp_gain, cost)) or min(old_delay, new_delay) < 0:
        return None
    return dict(seconds_saved=old_delay-new_delay, hp_gain=hp_gain, actual_paid_cost=cost,
                demonstrated=new_delay < old_delay and hp_gain > 0 and cost > 0)


def required_cases():
    cases=[]
    for backend in ('warp_kinematic','rapier'):
        for side in ('red','blue'):
            cases.append(dict(id=f'main-{backend}-{side}',kind='main',backend=backend,side=side,
                seed=81001 if side=='red' else 81002,seconds=60.,recovery_kind=None,
                scenario=ScenarioSpec(learner_side=side).to_dict()))
    for i,(kind,seconds) in enumerate((('free_ready',20.),('paid_early',20.),('unavailable',30.),('alive_weak',60.))):
        side='red' if i%2==0 else 'blue'
        cases.append(dict(id='recovery-'+kind,kind='recovery',backend='rapier',side=side,
            seed=91001+i,seconds=seconds,recovery_kind=kind,
            scenario=ScenarioSpec(learner_side=side).to_dict()))
    return cases


def trial(root,case,member,deadline_unix):
    import numpy as np
    from .env import SentryEnv
    from .portable import own_robot
    from .v6_ppo import HybridPPO
    from .v6_recording import save_snapshot_replay
    from .v4_evaluation import detect_tunnel_passages
    from .v4_campaign import _evaluation_diagnostics
    from .sentry_v6 import validate_action
    if member not in ('baseline','candidate'): raise ValueError('Unknown evaluation member')
    root=Path(root);folder=root/'evaluation';folder.mkdir(exist_ok=True)
    name=case['id']+'-'+member
    checkpoint=root/'main'/('initialized.zip' if member=='baseline' else 'model.zip')
    cfg=copy.deepcopy(json.loads((root/'main/experiment.json').read_text())['config'])
    for key in ('_v6_sampler_path','_v6_replay_dir','_episode_log_dir'):
        cfg.pop(key,None)
    cfg.update(evaluation=True,scenario_sampling=False,capability_sampling=False,
        sim_backend=case['backend'],scenario=case['scenario'],decision_window_seconds=case['seconds'],
        recovery_kind=case['recovery_kind'],action_trace=True,navigation_precompute=False,
        _training_deadline=time.monotonic()+max(0,deadline_unix-time.time())-2)
    capability_path=root/'capability-profiles'/f"evaluation-{case['side']}.json"
    capabilities=json.loads(capability_path.read_text())
    cfg['training_capabilities']=capabilities
    model=HybridPPO.load(checkpoint,device='cpu')
    env=SentryEnv(cfg,record_replay=True)
    row=dict(case_id=case['id'],member=member,backend=case['backend'],side=case['side'],
             seed=case['seed'],required_seconds=case['seconds'],horizon_completed=False,
             censored=True,prepared=False,checkpoint=str(checkpoint),full_match=False,
             checkpoint_sha256=_file_hash(checkpoint),scenario_hash=_digest(case['scenario']),
             capabilities_sha256=_file_hash(capability_path),
             actor_legality=dict(enforcement='session.select_goal',checked_decisions=0,invalid_decisions=0),
             execution_mask_fallbacks=0,recording_complete=False)
    trace=[];started=time.monotonic();info=None;origin=0.;recovered_at=None;active_hold=0.;capture_after=0.
    trace_path=folder/(name+'.jsonl')
    replay_path=root/'replays/evaluation'/(name+'.json')
    used_replay_bytes=_replay_bytes(root)-(replay_path.stat().st_size if replay_path.exists() else 0)
    try:
        obs,info=env.reset(seed=case['seed'],options={'scenario':case['scenario'],'capabilities':capabilities,
            'recovery_kind':case['recovery_kind']})
        origin=info['time'];row.update(prepared=True,prefix=info.get('recovery_prefix'),learner_start_time=origin)
        row['identity']=_execution_identity(env.session.config,env.session.battle.metadata['rl'])
        row['prefix_identity']=_prefix_identity(env.session.battle,info)
        with trace_path.open('w') as stream:
            while time.time()<deadline_unix-3:
                action,_=model.predict(obs,deterministic=True)
                try:
                    validated=validate_action(action)
                    if not bool(obs['revive_mask'][validated[6]]):
                        raise ValueError('Actor category violates its decision-time legal mask')
                except ValueError:
                    row['actor_legality']['invalid_decisions']+=1
                    raise
                previous_capture=info['metrics']['capture_seconds']
                previous_time=info['time']
                obs,_,done,truncated,info=env.step(action)
                row['actor_legality']['checked_decisions']+=1
                robot=own_robot(env.session.obs,7)
                private=env.session.battle.match._frame
                actual=next(r for r in private['robots'] if r['team']==case['side'] and r['slot']==7)
                dt=info['time']-previous_time
                active_hold=active_hold+dt if actual['alive'] and not actual.get('weak') else 0.
                if active_hold>=1-1e-8 and recovered_at is None:recovered_at=info['time']-.9
                if recovered_at is not None:capture_after+=info['metrics']['capture_seconds']-previous_capture
                point=dict(time=info['time'],position=[robot[k] for k in ('x','y','z')],
                    fold=robot.get('transform_progress'),action=action.tolist(),
                    navigation=info['action_trace']['navigation'],revival=env.session.controller.state.get('revive_execution'))
                point.update(_evaluation_diagnostics(env.session.obs,info,private,robot))
                trace.append(point);stream.write(json.dumps(point,allow_nan=False)+'\n');stream.flush()
                # An allowed 10 Hz choice may become unavailable before its
                # 20 Hz executor tick (another legal purchase won the race).
                # This fallback is not an illegal actor request.
                row['execution_mask_fallbacks']+=int(bool((point.get('revival') or {}).get('reason')))
                if env.session.battle.record_replay and _recording_would_exceed(env.session.battle,used_replay_bytes):
                    env.session.battle.record_replay=False
                    row['replay_quota_reached']=True
                if done or truncated:break
        passage=detect_tunnel_passages(trace)
        duration=info['time']-origin
        row.update(simulation_seconds=info['time'],decision_seconds=duration,
            horizon_completed=duration>=case['seconds']-1e-6,censored=duration<case['seconds']-1e-6,
            metrics=info['metrics'],recovery=info['recovery'],control=info['control'],
            recovered_at=recovered_at,recovery_delay=None if recovered_at is None else recovered_at-origin,
            capture_after_recovery=capture_after,passed_tunnel=passage['passed_tunnel'],passed_tunnel_evidence=passage)
    except Exception as error:
        row.update(error=type(error).__name__+': '+str(error),prefix=getattr(error,'report',row.get('prefix')))
    finally:
        if env.session.battle is not None and len(env.session.battle.frames)>1:
            try:
                complete=(row['horizon_completed'] and not row.get('replay_quota_reached')
                          and env.session.battle.frames[-1]['tick']==env.session.battle.match.tick)
                row['replay']=save_snapshot_replay(env.session.battle,replay_path,
                    metadata=dict(kind='evaluation',case_id=case['id'],checkpoint_member=member,
                                  truncated=not complete,recording_complete=complete,required_seconds=case['seconds']),
                    reason='evaluation_complete' if complete else 'recording_quota' if row.get('replay_quota_reached') else 'evaluation_censored')
                if _replay_bytes(root)>REPLAY_BYTE_LIMIT:
                    replay_path.unlink()
                    row['replay']=dict(recording_complete=False,omitted='shared_2GiB_quota')
                    complete=False
                row['recording_complete']=bool(complete)
            except Exception as error:row['replay_error']=str(error)
        env.close()
        if trace_path.exists():
            row['trace']=dict(path=str(trace_path),sha256=_file_hash(trace_path),frames=len(trace))
        row['wall_seconds']=time.monotonic()-started
        write_json(folder/(name+'.json'),row)
        print(json.dumps(dict(kind='v6-evaluation',**{k:v for k,v in row.items() if k!='replay'})),flush=True)
    return row


def _valid_hash(value):
    return isinstance(value, str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _validate_row(root, case, member, row):
    """Reject stale, renamed, incomplete or cross-run evidence before scoring."""
    from .v4_evaluation import detect_tunnel_passages
    reasons=[]
    if not isinstance(row, dict): return ['missing']
    expected=dict(case_id=case['id'],member=member,backend=case['backend'],side=case['side'],
                  seed=case['seed'],required_seconds=case['seconds'],scenario_hash=_digest(case['scenario']))
    if any(row.get(key)!=value for key,value in expected.items()): reasons.append('case_identity')
    if row.get('prepared') is not True or row.get('horizon_completed') is not True or row.get('censored') is not False or row.get('error'):
        reasons.append('incomplete_trial')
    checkpoint=Path(root)/'main'/('initialized.zip' if member=='baseline' else 'model.zip')
    if not checkpoint.is_file() or row.get('checkpoint_sha256')!=_file_hash(checkpoint): reasons.append('checkpoint_hash')
    capabilities=Path(root)/'capability-profiles'/f"evaluation-{case['side']}.json"
    if not capabilities.is_file() or row.get('capabilities_sha256')!=_file_hash(capabilities): reasons.append('capabilities_hash')
    identity=row.get('identity') or {};sources=identity.get('sources') or {};config=identity.get('execution_config') or {}
    if (identity.get('config_hash')!=_digest(config) or identity.get('execution_hash')!=_digest(sources)
            or any(not _valid_hash(sources.get(k)) for k in ('sdk_hash','runtime_hash','engine_binary_sha256','rules_sha256'))
            or not sources.get('sentry_sources') or not sources.get('simulation') or not sources.get('sources')):
        reasons.append('execution_identity')
    actual_options=dict(evaluation=True,policy_version=6,team=case['side'],sim_backend=case['backend'],
        scenario_sampling=False,capability_sampling=False,decision_window_seconds=case['seconds'],rl_seconds=.1)
    if any(config.get(key)!=value for key,value in actual_options.items()):reasons.append('evaluation_config')
    if set(identity.get('helper_hashes',{}))!={'infantry','heavy'} or not all(
            _valid_hash(x) for x in identity.get('helper_hashes',{}).values()): reasons.append('helper_identity')
    if not _valid_hash(identity.get('radar_hash')): reasons.append('radar_identity')
    try:
        from .session import resource
        current=json.loads((Path(root)/'main/experiment.json').read_text())['config']
        helper_hashes={key:_file_hash(resource(path)) for key,path in current['auxiliary_pool'].items()}
        radar_hash=_file_hash(resource(current['radar_classifier_path']))
        if identity.get('helper_hashes')!=helper_hashes or identity.get('radar_hash')!=radar_hash:
            reasons.append('frozen_pool_changed')
    except (KeyError,TypeError,ValueError,OSError):reasons.append('frozen_pool_unverifiable')
    prefix=row.get('prefix_identity') or {}
    if (any(not _valid_hash(prefix.get(k)) for k in ('commands_hash','handoff_hash'))
            or prefix.get('prepared') is not True or prefix.get('origin')!=row.get('learner_start_time')):
        reasons.append('prefix_identity')
    if case['recovery_kind']:
        from .v6_recovery import ALIASES
        expected_kind=ALIASES.get(case['recovery_kind'],case['recovery_kind'])
        if (prefix.get('prefix_kind')!=expected_kind or not _valid_hash(prefix.get('legal_history_hash'))
                or (row.get('prefix') or {}).get('prepared') is not True): reasons.append('recovery_prefix')
    legality=row.get('actor_legality') or {}
    expected_frames=round(case['seconds']/.1)
    if (legality.get('enforcement')!='session.select_goal' or legality.get('invalid_decisions')!=0
            or legality.get('checked_decisions')!=expected_frames): reasons.append('actor_legality')
    try:
        metrics=row['metrics'];recovery=row.get('recovery') or {}
        nonnegative=[metrics[key] for key in ('capture_seconds','tipped_seconds','policy_errors')]
        nonnegative += [recovery.get(key,0) for key in ('paid_coins','paid_revives','free_revives','recoveries')]
        nonnegative += [row.get('capture_after_recovery',0)]
        if row.get('recovery_delay') is not None:nonnegative.append(row['recovery_delay'])
        if (not all(isinstance(v,(int,float)) and math.isfinite(v) and v>=0 for v in nonnegative)
                or not all(math.isfinite(float(row['control'][side])) for side in ('red','blue'))
                or row.get('capture_after_recovery',0)>metrics['capture_seconds']+1e-6):
            reasons.append('invalid_metrics')
    except (KeyError,TypeError,ValueError):reasons.append('invalid_metrics')
    if row.get('recording_complete'):
        replay=row.get('replay') or {};meta=replay.get('metadata') or {}
        path=Path(root)/'replays/evaluation'/(case['id']+'-'+member+'.json')
        if (meta.get('recording_complete') is not True or meta.get('truncated') is not False
                or not path.is_file() or replay.get('sha256')!=_file_hash(path)
                or abs(float(replay.get('simulation_seconds',-1))-float(row.get('simulation_seconds',-2)))>1e-6):
            reasons.append('replay_completion_claim')
    trace_path=Path(root)/'evaluation'/(case['id']+'-'+member+'.jsonl')
    trace_meta=row.get('trace') or {}
    try:
        if not trace_path.is_file() or trace_meta.get('sha256')!=_file_hash(trace_path):
            reasons.append('trace_hash')
        else:
            trace=[json.loads(line) for line in trace_path.read_text().splitlines()]
            origin=float(row['learner_start_time'])
            if (len(trace)!=expected_frames or trace_meta.get('frames')!=expected_frames
                    or any(not math.isfinite(float(point['time'])) or abs(float(point['time'])-origin-(i+1)*.1)>1e-5
                           for i,point in enumerate(trace))
                    or abs(float(row['decision_seconds'])-case['seconds'])>1e-5): reasons.append('trace_horizon')
            if case['kind']=='main':
                actual=detect_tunnel_passages(trace)
                if row.get('passed_tunnel')!=actual['passed_tunnel'] or not actual['passed_tunnel']:
                    reasons.append('tunnel_evidence')
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError): reasons.append('invalid_trace')
    return reasons


def select_candidate(root):
    root=Path(root);cases=required_cases();pairs={};missing=[];failures=[];invalid={}
    for case in cases:
        rows=[]
        for member in ('baseline','candidate'):
            path=root/'evaluation'/(case['id']+'-'+member+'.json')
            try:value=json.loads(path.read_text()) if path.exists() else None
            except (OSError,json.JSONDecodeError):value=None
            try:reasons=_validate_row(root,case,member,value)
            except (KeyError,TypeError,ValueError,OSError,AttributeError,OverflowError):reasons=['invalid_record']
            if reasons:
                key=case['id']+':'+member
                missing.append(key);invalid[key]=reasons
                value=None
            rows.append(value)
        if all(rows):
            old,new=rows
            if any(old.get(key)!=new.get(key) for key in ('identity','prefix_identity','scenario_hash','capabilities_sha256')):
                failures.append(case['id']+':pair_identity_mismatch')
            else:pairs[case['id']]=rows
    controls=[0.,0.];captures=[0.,0.];improved=False
    for case in cases[:4]:
        if case['id'] not in pairs:continue
        old,new=pairs[case['id']];side=case['side'];other='blue' if side=='red' else 'red'
        if not old.get('passed_tunnel') or not new.get('passed_tunnel'):failures.append(case['id']+':tunnel')
        if any(row['metrics'].get('policy_errors',0) or row['metrics'].get('tipped_seconds',0)>1e-6 for row in (old,new)):
            failures.append(case['id']+':policy_or_tip')
        a,b=[row['control'][side]-row['control'][other] for row in (old,new)]
        x,y=[row['metrics']['capture_seconds'] for row in (old,new)]
        if b<a-1.-1e-6 or y<x-1.-1e-6:failures.append(case['id']+':per_case_regression')
        controls[0]+=a;controls[1]+=b;captures[0]+=x;captures[1]+=y
        improved|=b>=a+1.-1e-6 or y>=x+1.-1e-6
    if controls[1]<controls[0]-1e-6 or captures[1]<captures[0]-1e-6:failures.append('aggregate_regression')
    for case in cases[4:]:
        if case['id'] not in pairs:continue
        old,new=pairs[case['id']];r=new.get('recovery') or {};kind=case['recovery_kind']
        if any(row['metrics'].get('policy_errors',0) for row in (old,new)):failures.append(case['id']+':policy_error')
        if kind=='free_ready' and (r.get('free_revives',0)<1 or r.get('paid_coins',0)!=0 or r.get('recoveries',0)<1):
            failures.append('free_ready:free_and_recovery_not_demonstrated')
        if kind=='paid_early':
            advantage=_revive_advantage(old,new)
            if not advantage or not advantage['demonstrated'] or r.get('paid_coins',0)<=0 or r.get('paid_revives',0)<1:
                failures.append('paid_early:actual_advantage_not_demonstrated')
            elif advantage['seconds_saved']>=1-1e-6:improved=True
        if kind=='unavailable' and (r.get('free_revives',0)<1 or r.get('paid_coins',0)!=0 or r.get('paid_revives',0)!=0):
            failures.append('unavailable:free_without_payment_not_demonstrated')
        if kind=='alive_weak' and (r.get('recoveries',0)<1 or new.get('capture_after_recovery',0)<5.-1e-6):
            failures.append('alive_weak:recovery_and_capture_not_demonstrated')
        for metric in ('recovery_delay',):
            a,b=old.get(metric),new.get(metric)
            improved|=b is not None and (a is None or a-b>=1-1e-6)
    try:training=json.loads((root/'main/training.json').read_text())
    except (OSError,ValueError):training={}
    published_steps=training.get('published_accepted_actor_steps')
    if type(published_steps) is not int or published_steps<1:failures.append('no_accepted_ppo_update')
    if training.get('published_skill_unchanged') is not True:failures.append('frozen_skill_not_verified')
    candidate=root/'main/model.zip'
    if not candidate.is_file() or training.get('published_checkpoint_sha256')!=_file_hash(candidate):
        failures.append('published_checkpoint_hash_mismatch')
    promoted=not missing and not failures and improved
    result=dict(version=6,selected='model.zip' if promoted else 'initialized.zip',
        v6_paired_improvement_verified=promoted,complete_pairs=len(pairs),required_pairs=len(cases),
        missing=missing,invalid_evidence=invalid,failed_gates=failures,strict_improvement=improved,
        control=controls,capture_seconds=captures,historical_best_replaced=False,
        complete_recordings=sum(bool(row.get('recording_complete')) for rows in pairs.values() for row in rows),
        recording_quota_bytes=REPLAY_BYTE_LIMIT,
        reason='v6_paired_improvement' if promoted else 'retain_v6_migration_baseline_pending_evidence',
        evidence_scope='Eight declared finite paired cases, not overall win rate or complete lineup coverage')
    write_json(root/'selection.json',result)
    return result
