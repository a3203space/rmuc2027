"""Registered frozen research runs and canonical replay access for the Arena UI.

Catalog reads do not import Torch, create a simulator, or load arbitrary client
paths. A frozen match is created only by the explicit research start endpoint.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import re

from rmuc2027_core.paths import ROOT
from .scenarios import ScenarioSpec

FORMAT = 'rmuc-research-run-v1'
ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$')
SHA = re.compile(r'^[0-9a-f]{64}$')
BACKENDS = ('rapier', 'warp_kinematic')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def identifier(value):
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise ValueError('Invalid registered research identifier')
    return value


def member(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError('Research resource paths must be relative to their run')
    path = (root/name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Research resource is missing or outside its registered run')
    return path


def resource(root, record, *, verify=True):
    if not isinstance(record, dict) or not SHA.fullmatch(str(record.get('sha256', ''))):
        raise ValueError('Research resource requires a SHA256')
    path = member(root, record.get('path'))
    if verify and file_hash(path) != record['sha256']:
        raise ValueError('Research resource SHA256 mismatch')
    return path


class ResearchCatalog:
    def __init__(self, root=None):
        self.root = Path(root or ROOT/'data/research').resolve()
        self._replay_checks = {}

    def _runs(self):
        runs, errors, duplicate = {}, [], set()
        for path in sorted(self.root.glob('*/run-manifest.json')):
            try:
                if not path.resolve().is_relative_to(self.root):
                    raise ValueError('Research manifest is outside the registered root')
                raw = path.read_bytes()
                if len(raw) > 4*1024*1024:
                    raise ValueError('Research manifest is too large')
                data = json.loads(raw)
                if data.get('format') != FORMAT:
                    raise ValueError('Unsupported research run manifest')
                run_id = identifier(data.get('run_id'))
                if run_id in runs or run_id in duplicate:
                    runs.pop(run_id, None)
                    duplicate.add(run_id)
                    raise ValueError('Duplicate research run_id')
                for field in ('checkpoints', 'scenarios', 'capability_profiles', 'helpers'):
                    if not isinstance(data.get(field), dict):
                        raise ValueError('Research manifest is missing '+field)
                for key, row in data['checkpoints'].items():
                    identifier(key)
                    sha = row.get('sha256', '')
                    if not SHA.fullmatch(sha) or not key.endswith('-'+sha[:12]):
                        raise ValueError('Checkpoint IDs must end with their SHA256 prefix')
                runs[run_id] = (path.parent.resolve(), data, hashlib.sha256(raw).hexdigest())
            except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
                errors.append(dict(run_id=path.parent.name, error=str(error)))
        return runs, errors

    def get(self, run_id):
        identifier(run_id)
        runs, _ = self._runs()
        if run_id not in runs:
            raise ValueError('Unknown or invalid registered research run')
        return runs[run_id]

    def list_runs(self):
        runs, errors = self._runs()
        result = []
        for run_id, (_, data, digest) in runs.items():
            try:
                scenarios = []
                for key, row in data['scenarios'].items():
                    case = ScenarioSpec.from_dict(row)
                    if key != case.scenario_id:
                        raise ValueError('Scenario key does not match its hardware')
                    scenarios.append(dict(id=key, **case.to_dict(), lineups=case.lineups()))
                result.append(dict(run_id=run_id, title=str(data.get('title', run_id)), manifest_sha256=digest,
                    checkpoints=[dict(id=key, label=str(row.get('label', key)), sha256=row['sha256'],
                                      policy_version=row.get('policy_version', 6))
                                 for key, row in data['checkpoints'].items()],
                    scenarios=scenarios, capability_profiles=[dict(id=key, label=str(row.get('label', key)),
                        sha256=row.get('sha256'), learner_side=row.get('learner_side'))
                        for key, row in data['capability_profiles'].items()],
                    helpers={key:dict(sha256=row.get('sha256')) for key, row in data['helpers'].items()},
                    replay_count=len(data.get('replays', {})), backends=list(BACKENDS)))
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                errors.append(dict(run_id=run_id, error=str(error)))
        return dict(runs=result, errors=errors)

    def resolve_start(self, request, *, seed, duration):
        run_id = identifier(request['run_id'])
        directory, manifest, manifest_hash = self.get(run_id)
        backend = request.get('backend', 'rapier')
        if backend not in BACKENDS:
            raise ValueError('Research starts support Rapier or Warp only')
        checkpoint_id = identifier(request['checkpoint_id'])
        scenario_id = identifier(request['scenario_id'])
        profile_id = identifier(request.get('capability_profile_id', 'evaluation-default'))
        try:
            checkpoint = manifest['checkpoints'][checkpoint_id]
            case = ScenarioSpec.from_dict(manifest['scenarios'][scenario_id])
            profile_record = manifest['capability_profiles'][profile_id]
        except KeyError as error:
            raise ValueError('Unknown research checkpoint, scenario or capability profile') from error
        if case.scenario_id != scenario_id:
            raise ValueError('Scenario identifier mismatch')
        if profile_record.get('learner_side', case.learner_side) != case.learner_side:
            raise ValueError('Capability profile belongs to the other learner side')
        model_path = resource(directory, checkpoint)
        config_data = json.loads(resource(directory, manifest['runtime_config']).read_text())
        config = copy.deepcopy(config_data.get('config', config_data))
        version = checkpoint.get('policy_version', config.get('policy_version'))
        if version not in (4, 6) or config.get('policy_version') != version:
            raise ValueError('Checkpoint and runtime policy schema do not match')
        if set(manifest['helpers']) != {'infantry', 'heavy'}:
            raise ValueError('Both frozen helper actors are required')
        helpers = {role:str(resource(directory, row)) for role,row in manifest['helpers'].items()}
        capabilities = json.loads(resource(directory, profile_record).read_text())
        self._validate_capabilities(capabilities, case)
        # Configs may reference copied resources or repository-owned bundled bots.
        # Client input never participates in path resolution.
        def config_resource(name):
            path = Path(name)
            candidates = [path] if path.is_absolute() else [directory/path, ROOT/path]
            if not path.is_absolute() and path.parts and path.parts[0] == 'examples':
                candidates.append(ROOT/'research/strategies'/path)
            for candidate in candidates:
                candidate = candidate.resolve()
                if (candidate.is_relative_to(directory) or candidate.is_relative_to(ROOT)) and candidate.exists():
                    return str(candidate)
            raise ValueError('Frozen runtime policy resource is unavailable')
        for name in ('friendly', 'opponent'):
            if name not in config:
                raise ValueError('Frozen runtime is missing '+name)
            config[name] = config_resource(config[name])
        controllers = copy.deepcopy(config.get('controllers', {}))
        for controller in controllers.values():
            if controller.get('submission'):
                controller['submission'] = config_resource(controller['submission'])
        original = config.get('team', 'red')
        if original != case.learner_side:
            controllers = {('blue' if key.startswith('red:') else 'red')+':'+key.split(':')[1]:value
                           for key,value in controllers.items()}
        controllers[f'{case.learner_side}:7'] = {'kind':'learning'}
        if any(key != f'{case.learner_side}:7' and not key.endswith(':6') for key in controllers):
            raise ValueError('Frozen ground helpers and the strict enemy sentry cannot be overridden')
        for key in ('_training_deadline', '_episode_log_dir', 'github_backup_repo',
                    'sentry_guidance_path', 'skill_guidance_path', 'skill_guidance_datasets',
                    '_v6_sampler_path', '_v6_replay_dir', '_v6_run_id', 'recovery_kind',
                    '_worker_id', '_run_dir'):
            config.pop(key, None)
        config.update(policy_version=version, slot=7, team=case.learner_side, controllers=controllers,
            auxiliary_pool=helpers, scenario=case.to_dict(), lineups=case.lineups(),
            training_capabilities=capabilities, scenario_sampling=False, capability_sampling=False,
            evaluation=True, sim_backend=backend, episode_seconds=duration, curriculum_seconds=0,
            decision_window_seconds=duration,
            alternate_sides=False, action_trace=True, training_summary=False, seed=seed)
        if manifest.get('radar_classifier'):
            config['radar_classifier_path'] = str(resource(directory, manifest['radar_classifier']))
        elif config.get('radar_classifier_path'):
            raise ValueError('Frozen radar classifier requires an explicit hashed manifest resource')
        provenance = dict(run_id=run_id, manifest_sha256=manifest_hash, checkpoint_id=checkpoint_id,
            checkpoint_sha256=checkpoint['sha256'], checkpoint_label=checkpoint.get('label', checkpoint_id),
            scenario_id=scenario_id, capability_profile_id=profile_id,
            capability_sha256=profile_record['sha256'], backend=backend, policy_version=version,
            helpers={role:row['sha256'] for role,row in manifest['helpers'].items()},
            seed=seed, horizon_seconds=duration, frozen=True, gradient_updates=0)
        return dict(config=config, checkpoint=model_path, scenario=case.to_dict(), capabilities=capabilities,
                    provenance=provenance, title=manifest.get('title', run_id))

    @staticmethod
    def _validate_capabilities(profiles, case):
        lineups = case.lineups()
        expected = {f'{side}:{slot}' for side in ('red','blue') for slot in (1,3,4,7)}
        if not isinstance(profiles, dict) or set(profiles) != expected:
            raise ValueError('Frozen capabilities must include all eight ground seats')
        for seat,row in profiles.items():
            if not isinstance(row, dict):
                raise ValueError('Invalid frozen capability seat')
            side,slot = seat.split(':')
            if row.get('model') != lineups[side][slot]:
                raise ValueError('Capability profile hardware does not match the chosen scenario')
            for key in ('speed_reference_mps','motor_force_max_n','drive_efficiency'):
                value = row.get('mobility', {}).get(key)
                if isinstance(value, bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError('Invalid frozen mobility profile')
            for key in ('delay_s','sigma_deg','tau_s'):
                value = row.get('aim', {}).get(key)
                if isinstance(value, bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value < 0 or (key=='tau_s' and value==0):
                    raise ValueError('Invalid frozen aiming profile')
        learner = profiles[f'{case.learner_side}:7']
        if {seat for seat,row in profiles.items() if row.get('protected_learner')} != {f'{case.learner_side}:7'}:
            raise ValueError('Only the selected main 2773 may be protected')
        if not learner.get('protected_learner') or any(v != 1 for v in learner['mobility'].values()) or learner['aim']['delay_s'] or learner['aim']['sigma_deg']:
            raise ValueError('Main 2773 must retain its protected nominal capability profile')

    def list_replays(self, run_id):
        directory, manifest, _ = self.get(run_id)
        result = []
        for key,row in manifest.get('replays', {}).items():
            identifier(key)
            path = resource(directory,row,verify=False)
            result.append(dict(id=key, run_id=run_id, label=str(row.get('label',key)),
                sha256=row['sha256'], bytes=path.stat().st_size,
                kind=row.get('kind','evaluation'), backend=row.get('backend'),
                checkpoint_id=row.get('checkpoint_id'), scenario_id=row.get('scenario_id'),
                truncated=row.get('truncated'), seconds=row.get('seconds'),
                recording_end_reason=row.get('recording_end_reason')))
        return result

    def replay(self, run_id, replay_id):
        directory, manifest, _ = self.get(run_id)
        identifier(replay_id)
        try:
            row = manifest.get('replays', {})[replay_id]
        except KeyError as error:
            raise ValueError('Unknown registered research replay') from error
        path = resource(directory,row,verify=False)
        stat = path.stat();key = (str(path),stat.st_size,stat.st_mtime_ns,row['sha256'])
        if key not in self._replay_checks:
            resource(directory,row)
            from rmuc2027_core.replay import ReplayArchive
            archive = ReplayArchive(path)
            first = {**archive.static, **archive.frames(0,1)[0]}
            if not {'robots','projectiles','events','teams','tick','time'} <= first.keys():
                raise ValueError('Research playback requires canonical full frames, not a trajectory trace')
            last = archive.frames(len(archive.offsets)-1,1)[0]
            span = last['tick']-first['tick']
            if len(archive.offsets) != span+1 or abs((last['time']-first['time'])-span*.05) > 1e-5:
                raise ValueError('Research playback requires consecutive 20 Hz frames')
            if not archive.metadata.get('test_match'):
                raise ValueError('Research replay must be explicitly excluded from Elo')
            self._replay_checks[key] = True
        return path


class FrozenResearchMatch:
    """No-gradient SentryEnv adapter: one web step advances 0.1 simulated seconds."""
    def __init__(self, resolved, *, env_factory=None, model_loader=None):
        if env_factory is None:
            from .env import SentryEnv
            env_factory = SentryEnv
        if model_loader is None:
            if resolved['config']['policy_version'] == 6:
                from .v6_ppo import HybridPPO
                model_loader = HybridPPO.load
            else:
                from .asymmetric_ppo import AsymmetricPPO
                model_loader = AsymmetricPPO.load
        self.resolved = resolved
        self.model = model_loader(resolved['checkpoint'], device='cpu')
        self.env = env_factory(resolved['config'], record_replay=True)
        try:
            self.observation, _ = self.env.reset(seed=resolved['provenance']['seed'],
                options={'scenario':resolved['scenario'],'capabilities':resolved['capabilities']})
            self.battle = self.env.session.battle
            self.metadata = self.battle.metadata
            self.metadata.update(test_match=True, ranked=False, match_mode='research',
                                 research=copy.deepcopy(resolved['provenance']))
            side = resolved['config']['team']
            self.metadata['teams'][side].update(display_name=str(resolved['title'])+' · '+resolved['provenance']['checkpoint_label'],
                version=resolved['provenance']['checkpoint_sha256'])
            self.step_seconds = float(self.env.session.config['rl_seconds'])
            if abs(self.step_seconds-.1) > 1e-9:
                raise ValueError('Frozen research requires 10 Hz actor decisions')
            self.done = False
            self.terminated = self.truncated = False
        except BaseException:
            self.env.close()
            raise

    def frame(self):
        return self.battle.match.snapshot()

    def step(self):
        import numpy as np
        action, _ = self.model.predict(self.observation, deterministic=True)
        if not np.isfinite(action).all():
            raise ValueError('Nonfinite frozen research action')
        self.observation, _, self.terminated, self.truncated, _ = self.env.step(action)
        self.done = bool(self.terminated or self.truncated)
        return self.frame()

    def save(self, path, *, stopped=False):
        actual = float(self.frame()['time'])
        horizon = self.resolved['provenance']['horizon_seconds']
        self.metadata['recording'] = dict(kind='evaluation', frame_hz=20, actor_hz=10,
            requested_seconds=horizon, recorded_seconds=actual, full_match=actual >= 420-1e-6,
            horizon_completed=actual >= horizon-1e-6, truncated=bool(stopped or actual < horizon-1e-6),
            end_reason='user_stop' if stopped else 'horizon' if self.truncated else 'terminal')
        self.env.session.save_replay(path)

    def result(self):
        return self.battle.result()

    def close(self):
        self.env.close()
