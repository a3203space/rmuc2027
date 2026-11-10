"""Strict frozen-anchor reuse; no simulation, fitting or permissive fallback."""
import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import zipfile

CAPTURE_REWARD_INSERTION = b"""        if 'v4_capture_reward' in config:
            capture_reward = config['v4_capture_reward']
            if (isinstance(capture_reward, bool) or not isinstance(capture_reward, (int, float))
                    or not math.isfinite(capture_reward) or capture_reward < 0):
                raise ValueError('v4_capture_reward must be finite and nonnegative')
            config['rewards']['capture_seconds'] = float(capture_reward)
"""


def _session_equivalent_hashes(path):
    raw=Path(path).read_bytes();hashes={hashlib.sha256(raw).hexdigest()}
    if raw.count(CAPTURE_REWARD_INSERTION)==1:
        hashes.add(hashlib.sha256(raw.replace(CAPTURE_REWARD_INSERTION,b'',1)).hexdigest())
    return hashes


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluation_config(root, backend, side):
    from .scenarios import ScenarioSpec
    root=Path(root)
    cfg=copy.deepcopy(json.loads((root/'main/experiment.json').read_text())['config'])
    cfg.update(sim_backend=backend,episode_seconds=60,scenario_sampling=False,evaluation=True,
        scenario=ScenarioSpec(learner_side=side).to_dict(),navigation_precompute=False,
        v4_capture_reward=0.,auxiliary_pool={role:str((root/role/'actor.json').resolve()) for role in ('infantry','heavy')
                        if (root/role/'actor.json').exists()},action_trace=True)
    return cfg


def _require(condition, message):
    if not condition: raise ValueError('Anchor reuse rejected: '+message)


def _runtime_evidence(source):
    """The original sealed source snapshot supplements backend-specific metadata."""
    from rmuc2027_core.paths import ROOT
    archive=source.with_suffix('.zip')
    _require(archive.is_file(),'original sealed source bundle is required')
    with zipfile.ZipFile(archive) as zipped:
        manifest_bytes=zipped.read('bundle-manifest.json')
        records=json.loads(manifest_bytes)['files']
    files=[]
    files.extend((ROOT/'packages/core/rmuc2027_core').glob('*.py'))
    files.extend((ROOT/'research/rmuc2027_research/strategies').glob('*.py'))
    rl=ROOT/'research/rmuc2027_research/rl'
    files.extend(rl/(name+'.py') for name in ('session','env','portable','navigation','critic_state',
        'sentry_v3','sentry_v4','sentry_v4_policy','aux_policy','privileged_aux','capabilities','scenarios',
        'asymmetric_ppo','v4_evaluation'))
    # Verify both optional physics implementations, regardless of selected pair.
    for backend in ('warp_kinematic','mjwarp'): files.extend((rl/backend).glob('*.py'))
    hashes={};allowed_deltas={}
    for path in files:
        relative='source/'+path.relative_to(ROOT).as_posix()
        _require(relative in records,'missing original execution source: '+relative)
        actual=digest(path)
        expected=records[relative]['sha256']
        if actual!=expected and path==rl/'session.py':
            _require(expected in _session_equivalent_hashes(path),'session change exceeds exact reward insertion')
            allowed_deltas[relative]='exact optional v4_capture_reward insertion; effective evaluation value forced to 0'
        else:_require(actual==expected,'execution source changed: '+relative)
        hashes[relative]=actual
    return dict(manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),execution_sources=hashes,
                allowed_source_deltas=allowed_deltas,
                excluded_training_only_sources=['v4_campaign.py','v4_training.py','v4_anchor_reuse.py',
                    'aux_training.py','radar_calibration.py','v4_bundle.py'])


def _signature(root, backend, side):
    from .session import provenance,load_config
    root=Path(root);metadata=json.loads((root/'main/experiment.json').read_text())
    cfg=load_config(evaluation_config(root,backend,side))
    current=provenance(cfg)
    # These are recorded at the original execution boundary, rather than
    # accepting two currently changed artifacts merely because they match.
    keys=('sources','sdk_hash','runtime_hash','sentry_sources','sentry_schema',
          'scenario_registry_hash','capability_schema_hash','engine_binary_sha256','rules_sha256')
    for key in keys:
        _require(key in metadata and metadata[key]==current[key],'recorded runtime mismatch: '+key)
    _require(metadata['navigation']['preparation_sha256']==current['navigation']['preparation_sha256'],
             'navigation source changed')
    for name in ('session.py','env.py','aux_policy.py','privileged_aux.py','capabilities.py','scenarios.py'):
        valid=metadata['training_sources'].get(name)==current['training_sources'].get(name)
        if name=='session.py' and not valid:
            valid=metadata['training_sources'].get(name) in _session_equivalent_hashes(Path(__file__).with_name(name))
        _require(valid,
                 'recorded execution source changed: '+name)
    helpers={role:digest(path) for role,path in cfg.get('auxiliary_pool',{}).items()}
    _require(set(helpers)=={'infantry','heavy'},'both frozen helper actors are required')
    for role,sha in helpers.items():
        _require(metadata.get('auxiliary_pool',{}).get(role,{}).get('sha256')==sha,'helper differs from original metadata: '+role)
    radar=cfg.get('radar_classifier_path')
    radar_hash=digest(radar) if radar else None
    _require(metadata.get('radar_classifier_sha256')==radar_hash,'radar differs from original metadata')
    normalized=copy.deepcopy(cfg)
    for key in list(normalized):
        if key=='ppo' or key.startswith('_') or key in ('checkpoint_seconds','github_backup_repo','github_backup_seconds'):
            normalized.pop(key)
    for key in ('friendly','opponent'):
        if key in normalized: normalized[key]={'content':current['sources'][key]}
    for key,spec in normalized.get('controllers',{}).items():
        if spec.get('submission'): spec['submission']={'content':current['sources']['controller:'+key]}
    normalized['auxiliary_pool']=helpers
    normalized['radar_classifier_path']=radar_hash
    dependencies={name:importlib.metadata.version(name) for name in metadata.get('dependencies',{})}
    _require(dependencies==metadata.get('dependencies'),'recorded inference dependency mismatch')
    return dict(config=normalized,runtime={key:current[key] for key in keys},
                simulation=current['simulation'],dependencies=dependencies)


def validate_anchor_reuse(root, source_run, backend, side):
    """Return a copied summary and exact trace bytes, or reject before copying."""
    root=Path(root).resolve();source=Path(source_run).resolve()
    _require(source!=root,'source and destination must differ')
    summary=source/'evaluation'/f'{backend}-{side}.json'
    summary_bytes=summary.read_bytes();rows=json.loads(summary_bytes)
    _require(isinstance(rows,list) and len(rows)==2,'source must contain exactly two ordered rows')
    for row,checkpoint in zip(rows,('initialized.zip','model.zip')):
        _require(row.get('checkpoint')==checkpoint and row.get('backend')==backend and row.get('side')==side,
                 'source checkpoint order/backend/side mismatch')
        _require(row.get('horizon_completed') is True and row.get('censored') is False and
                 abs(float(row.get('simulation_seconds',0))-60)<1e-8,'source pair is incomplete')
    checkpoint_hash=digest(source/'main/initialized.zip')
    _require(digest(root/'main/initialized.zip')==checkpoint_hash,'anchor checkpoint bytes differ')
    source_signature=_signature(source,backend,side);target_signature=_signature(root,backend,side)
    _require(source_signature==target_signature,'effective evaluation configuration/runtime differs')
    evidence=_runtime_evidence(source)
    trace_path=source/'evaluation'/f'{backend}-{side}-initialized.zip.jsonl'
    trace_bytes=trace_path.read_bytes();trace=[json.loads(line) for line in trace_bytes.splitlines() if line.strip()]
    _require(len(trace)==rows[0]['steps'] and len(trace)>0,'anchor trace length mismatch')
    previous=-math.inf
    for frame in trace:
        t=float(frame['time'])
        _require(math.isfinite(t) and t>previous,'invalid anchor trace timestamps');previous=t
        _require(len(frame['position'])==3 and all(math.isfinite(float(x)) for x in frame['position']),
                 'invalid anchor trace position')
        _require(len(frame['action'])==6 and all(math.isfinite(float(x)) for x in frame['action']) and
                 'navigation' in frame,'invalid anchor trace action/navigation')
    _require(abs(previous-60)<1e-8 and trace[-1]['position']==rows[0]['final_position'],'anchor trace endpoint mismatch')
    row=copy.deepcopy(rows[0]);row['reused']=True
    row['reuse']=dict(source_run=str(source),summary_sha256=hashlib.sha256(summary_bytes).hexdigest(),
        trace_sha256=hashlib.sha256(trace_bytes).hexdigest(),anchor_checkpoint_sha256=checkpoint_hash,
        evaluation_signature_sha256=hashlib.sha256(json.dumps(source_signature,sort_keys=True).encode()).hexdigest(),
        source_bundle_manifest_sha256=evidence['manifest_sha256'],
        allowed_source_deltas=evidence.get('allowed_source_deltas',{}),
        excluded_training_only_sources=evidence.get('excluded_training_only_sources',[]))
    return row,trace_bytes
