"""Headless simulation session; the Gym adapter adds only spaces and seeding."""
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time
from types import SimpleNamespace

from rmuc2027_core.engine import DT, binary_path
from rmuc2027_core.paths import ROOT
from ..strategies.battle import Battle
from rmuc2027_core.policy import EncodedObservation, Policy
from rmuc2027_core.submissions import load_submission
from .portable import SdkNavigator, encode_observation, execute_goal, select_goal, fixed_opening, learning_ready, own_robot, policy_schema

DEFAULT_CONFIG = ROOT / 'configs/rl/sentry.json'


def load_config(source=None):
    config = json.loads(DEFAULT_CONFIG.read_text())
    supplied = json.loads(Path(source).read_text()) if isinstance(source, (str, Path)) else copy.deepcopy(source or {})
    for key, value in supplied.items():
        if key in ('rewards', 'ppo'):
            config[key].update(value)
        else:
            config[key] = value
    if config.get('lineups') is not None:
        from rmuc2027_core.models import normalize_team_lineup
        if set(config['lineups']) != {'red', 'blue'}:
            raise ValueError('lineups requires red and blue')
        config['lineups'] = {team: normalize_team_lineup(value, '2027')
                             for team, value in config['lineups'].items()}
    config.setdefault('policy_timeout_s', 5.0)
    config.setdefault('navigation_mode', 'legacy')
    if config['navigation_mode'] not in ('legacy', 'predictive', 'adaptive'):
        raise ValueError('Unknown navigation mode')
    policy_schema(config)
    if config.get('policy_version') in (3,4):
        if config['slot']!=7 or config['navigation_mode']!='adaptive' or config['rl_seconds']!=.1:
            raise ValueError('Policy v3 requires sentry slot 7, adaptive navigation and 10 Hz actor decisions')
        config['observation_profile']='sentry_referee_2027_v1'
        config['learning_control_hz']=20
        config['training_summary']=False
        if config.get('curriculum_seconds',0):raise ValueError('Policy v3 has no hidden curriculum goal override')
        if 'opponent' not in supplied:config['opponent']=config['friendly']
    if config.get('policy_version') == 4:
        config['rewards'].update(control=.01,outcome=10.,death=-.1,capture_seconds=0.)
        config.setdefault('scenario_sampling',True)
        config.setdefault('capability_sampling',True)
    if config.get('policy_version') == 5:
        if config['slot'] not in (1,3,4):raise ValueError('Auxiliary RL only controls infantry/heavy')
        config.update(observation_profile='sentry_referee_2027_v1',training_summary=False,
                      learning_control_hz=20,timed_assembly=False,navigation_mode='adaptive')
        config.setdefault('scenario_sampling',True)
        config.setdefault('capability_sampling',True)
    if config.get('policy_version', 1) == 2 and (config['navigation_mode'] != 'adaptive'
            or config.get('learning_control_hz', 10) != 10 or config['rl_seconds'] != .1):
        raise ValueError('Policy v2 requires adaptive navigation and 10 Hz decisions')
    config.setdefault('timed_assembly', False)
    if not isinstance(config['timed_assembly'], bool):
        raise ValueError('timed_assembly must be boolean')
    config.setdefault('drones_enabled', True)
    if not isinstance(config['drones_enabled'], bool):
        raise ValueError('drones_enabled must be boolean')
    config.setdefault('fixed_policy_hz', 10)
    config.setdefault('learning_control_hz', 10)
    if config['learning_control_hz'] not in (2, 5, 10, 20):
        raise ValueError('learning_control_hz must be 2, 5, 10 or 20')
    config.setdefault('training_summary', True)
    config.setdefault('action_trace', False)
    config.setdefault('navigation_precompute',None)
    if config['navigation_precompute'] is not None and not isinstance(config['navigation_precompute'],bool):
        raise ValueError('navigation_precompute must be boolean or null (automatic)')
    config.setdefault('github_backup_seconds',300.)
    interval=config['github_backup_seconds']
    if isinstance(interval,bool) or not isinstance(interval,(int,float)) or not math.isfinite(interval) or interval<=0:
        raise ValueError('github_backup_seconds must be finite and positive')
    if config['fixed_policy_hz'] not in (5, 10):
        raise ValueError('fixed_policy_hz must be 5 or 10')
    timeout = config['policy_timeout_s']
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('policy_timeout_s must be finite and positive')
    if config['map'] != 'rmuc2027' or config['slot'] not in (1, 3, 4, 7) or config['team'] not in ('red', 'blue'):
        raise ValueError('RL supports ground slots 1, 3, 4 and 7 on either side')
    for key in ('duration', 'episode_seconds', 'rl_seconds', 'grid_m'):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if not 0.1 <= config['rl_seconds'] <= 10 or not math.isclose(config['rl_seconds']/.1, round(config['rl_seconds']/.1)):
        raise ValueError('rl_seconds must be a multiple of 0.1 in [0.1, 10]')
    if not math.isclose(config['episode_seconds']/config['rl_seconds'], round(config['episode_seconds']/config['rl_seconds'])):
        raise ValueError('episode_seconds must be a multiple of rl_seconds')
    if not .1 <= config['grid_m'] <= 1 or not 0 < config['recover_enter'] < config['recover_exit'] <= 1:
        raise ValueError('Invalid grid spacing or recovery hysteresis')
    if config['duration'] != 420:
        raise ValueError('Keep the 420-second game horizon; shorten episode_seconds for training truncation')
    if not all(math.isfinite(float(v)) for v in config['rewards'].values()):
        raise ValueError('Reward weights must be finite')
    # The selected side follows team. The default file is convenient for red;
    # explicit non-default controller maps must name the desired side themselves.
    if 'controllers' not in supplied:
        config['controllers'] = {f"{config['team']}:{config['slot']}": {'kind': 'learning'}}
    for key, spec in config['controllers'].items():
        parts = key.split(':')
        if len(parts) != 2 or parts[0] not in ('red', 'blue') or parts[1] not in ('1', '3', '4', '6', '7'):
            raise ValueError('Controller keys must be canonical team:slot identifiers')
        if spec['kind'] not in ('learning', 'script', 'frozen') or (spec['kind'] != 'learning' and 'submission' not in spec):
            raise ValueError('Invalid controller assignment')
    if config.get('sim_backend', 'rapier') not in ('rapier', 'mjwarp', 'warp_kinematic'):
        raise ValueError('Unknown simulation backend')
    return config


def resource(path):
    path = Path(path)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == 'examples':
        return ROOT/'research/strategies'/path
    return ROOT/path


def baseline_module(config):
    manifest, entries = load_submission(resource(config['friendly']))
    source = entries[str(config['slot'])].read_text()
    namespace = {'__name__': 'fixed_sentry_execution'}
    # Only rewrite the public SDK import path; deployed source imports the
    # identical sdk.py staged by Policy. Never mutate process-wide sys.modules.
    exec(compile(source.replace('from sdk import ', 'from rmuc2027_core.sdk import '), str(entries[str(config['slot'])]), 'exec'), namespace)
    return SimpleNamespace(**namespace)


def provenance(config):
    result = {'config': config, 'sources': {}}
    result['training_sources']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ('session.py','env.py','learning.py','campaign.py','intervention.py','guided_campaign.py')}
    for name in ('friendly', 'opponent'):
        manifest, entries = load_submission(resource(config[name]))
        result['sources'][name] = {'version': manifest['version'], 'manifest': manifest,
            'hashes': {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in entries.items()}}
    for key, assignment in config['controllers'].items():
        if assignment['kind'] == 'learning':
            continue
        manifest, entries = load_submission(resource(assignment['submission']))
        result['sources']['controller:'+key] = {'version': manifest['version'], 'manifest': manifest,
            'hashes': {slot: hashlib.sha256(path.read_bytes()).hexdigest() for slot, path in entries.items()}}
    result['sdk_hash'] = hashlib.sha256((Path(__file__).resolve().parents[3]/'packages/core/rmuc2027_core/sdk.py').read_bytes()).hexdigest()
    result['runtime_hash'] = hashlib.sha256(Path(__file__).with_name('portable.py').read_bytes()).hexdigest()
    if config.get('policy_version') in (3,4):
        result['sentry_schema']=policy_schema(config)
        for name in ('sentry_training.py','sentry_campaign.py'):
            result['training_sources'][name]=hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        result['sentry_sources']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__).with_name('sentry_v3.py'), ROOT/'packages/core/rmuc2027_core/sentry_protocol.py',ROOT/'packages/core/rmuc2027_core/radar2027.py')}
        if config.get('policy_version')==4:
            for p in (Path(__file__).with_name('sentry_v4.py'),Path(__file__).with_name('sentry_v4_policy.py'),
                      Path(__file__).with_name('critic_state.py'),ROOT/'packages/core/rmuc2027_core/radar_inference.py'):
                if p.exists():result['sentry_sources'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    if config.get('policy_version') in (4,5):
        for name in ('v4_training.py','v4_campaign.py','asymmetric_ppo.py','aux_policy.py',
                     'aux_training.py','privileged_aux.py','capabilities.py','scenarios.py','radar_calibration.py'):
            result['training_sources'][name]=hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        from .scenarios import REGISTRY_HASH
        from .capabilities import SCHEMA_HASH
        result['scenario_registry_hash']=REGISTRY_HASH
        result['capability_schema_hash']=SCHEMA_HASH
        result['auxiliary_pool']={key:dict(path=str(value),sha256=hashlib.sha256(resource(value).read_bytes()).hexdigest())
                                  for key,value in config.get('auxiliary_pool',{}).items()}
        if config.get('radar_classifier_path'):
            result['radar_classifier_sha256']=hashlib.sha256(resource(config['radar_classifier_path']).read_bytes()).hexdigest()
    result['navigation']={'precompute':config.get('navigation_precompute') if config.get('navigation_precompute') is not None else config.get('sim_backend')=='warp_kinematic',
        'preparation_sha256':hashlib.sha256(Path(__file__).with_name('navigation.py').read_bytes()).hexdigest()}
    result['engine_binary_sha256'] = hashlib.sha256(binary_path().read_bytes()).hexdigest()
    result['rules_sha256'] = hashlib.sha256((ROOT/'configs/rules.json').read_bytes()).hexdigest()
    if config.get('sim_backend', 'rapier') == 'mjwarp':
        from importlib.metadata import version
        folder = Path(__file__).with_name('mjwarp')
        result['simulation'] = {'backend': 'mjwarp', 'device': config.get('sim_device', 'cuda:0'),
            'dt': 1/120, 'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.py')},
            'dependencies': {n: version(n) for n in ('mujoco', 'mujoco-warp', 'warp-lang')}}
    elif config.get('sim_backend') == 'warp_kinematic':
        from importlib.metadata import version
        from .warp_kinematic.protocol import FRAME_ENCODER
        from rmuc2027_core import engine as transport
        folder=Path(__file__).with_name('warp_kinematic')
        result['simulation']={'backend':'warp_kinematic','device':config.get('sim_device','cuda:0'),
            'dt':1/120,'experimental':True,'motion_protocol_version':2,
            'frame_encoder':FRAME_ENCODER,
            'pipe_reader':'posix-select' if transport._DIRECT_PIPE_READER else 'threaded',
            'policy_observation_cache':transport._CACHE_POLICY_OBSERVATIONS,
            'transport_sha256':hashlib.sha256((ROOT/'packages/core/rmuc2027_core/engine.py').read_bytes()).hexdigest(),
            'files':{name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
                     for name in ('__init__.py','arena.py','motion.py','contacts.py','terrain.py','traversal.py','dynamics.py','pose.py','protocol.py','uploads.py','batching.py','vector.py')},
            'dependencies':{'warp-lang':version('warp-lang')}}
        if FRAME_ENCODER=='orjson-numpy':result['simulation']['dependencies']['orjson']=version('orjson')
    else:
        result['simulation'] = {'backend': 'rapier', 'dt': 1/120}
    return result


class GoalController:
    """Policy-compatible external target latch; one instance per episode."""
    def __init__(self, config):
        self.config = config
        self.is_learning_controller = True
        precompute=config.get('navigation_precompute')
        if precompute is None:precompute=config.get('sim_backend')=='warp_kinematic'
        if precompute:
            from .navigation import PreparedNavigator
            self.navigator=PreparedNavigator(config['grid_m'])
        else:self.navigator = SdkNavigator(config['grid_m'])
        self.baseline = baseline_module(config)
        self.state, self.logs = {}, []
        self.disabled, self.calls, self.errors, self.consecutive_timeouts = False, 0, 0, 0

    def select(self, obs, action):
        select_goal(obs, self.state, action, self.navigator, self.config)

    def decide(self, observation):
        obs = json.loads(observation.payload) if isinstance(observation, EncodedObservation) else observation
        self.calls += 1
        opening = fixed_opening(obs, self.state, self.config, self.baseline)
        if opening is not None:
            return opening, None
        return execute_goal(obs, self.state, self.navigator, self.config, self.baseline), None

    def close(self):
        self.disabled = True
        self.navigator.grids.clear()


class OpeningDeadline(TimeoutError):
    """The real-time budget expired before a valid learning observation existed."""


class ReferenceSentryController(GoalController):
    """Deterministic six-action opponent with the same information boundary."""
    control_hz=20

    def decide(self,observation):
        obs=json.loads(observation.payload) if isinstance(observation,EncodedObservation) else observation
        if obs['tick']%2==0 or 'goal' not in self.state:
            me=own_robot(obs,7)
            from rmuc2027_core import sdk
            ctx=sdk.locomotion_context(me,obs,self.state.get('navigation',{}))
            goal=(12.,7.5)
            fold=ctx['roof'] is not None and ctx['roof']['gap']<1.5
            nav=self.state.get('navigation',{});route=nav.get('jump_route')
            jump=False
            if route:
                along=(me['x']-route['edge'])*route['direction']
                jump= -.9<along<-.1 and math.hypot(me['vx'],me['vy'])>.5
            self.select(obs,[goal[0]/14-1,goal[1]/7.5-1,1 if fold else -1,1 if jump else -1,-1,1])
        return super().decide(obs)


class OpeningIncomplete(RuntimeError):
    """The fixed opening reached match end without enabling learned control."""


class ParkedController:
    """Stationary peers for explicit locomotion courses, never match evaluation."""
    disabled = False
    consecutive_timeouts = 0
    logs = ()

    def __init__(self, slot):
        self.slot = str(slot)

    def decide(self, observation):
        return {self.slot: {'move':[0., 0.], 'brake':True}}, None

    def close(self):
        self.disabled = True


class ArenaSession:
    """Standard-library core used directly by diagnostic CLI and by Gymnasium.

    Actor inputs come only from scoped observations. Full frames are consumed
    separately by reward/metric accounting and optional replay recording.
    """
    def __init__(self, config=None, record_replay=False):
        self.config = load_config(config)
        self.deadline = self.config.pop('_training_deadline', None)
        self.base_config=copy.deepcopy(self.config)
        from .scenarios import ScenarioSampler
        from .capabilities import CapabilitySampler
        sampler_seed=self.config['seed']+1000003*self.config.get('_worker_id',0)
        self.scenario_sampler=ScenarioSampler(sampler_seed)
        self.capability_sampler=CapabilitySampler(sampler_seed+7919)
        self.scenario=None
        self.episode_index=0
        self.calibration_recorder=None
        if self.config.get('policy_version') in (4,5):
            from .radar_calibration import RadarCalibrationRecorder
            self.calibration_recorder=RadarCalibrationRecorder()
        self.record_replay = record_replay
        self.rng = random.Random(self.config['seed'])
        self.battle = None
        self.controller = None
        self.done = True
        self.requested_stage = 'course' if config and self.config.get('curriculum_seconds', 0)>0 else 'match'

    def reset(self, seed=None, options=None):
        self.close()
        if seed is not None:
            self.rng.seed(int(seed))
        self.seed = int(seed) if seed is not None else self.rng.randrange(2**31)
        self.config=copy.deepcopy(self.base_config)
        cfg = self.config
        self.scenario=None
        options=options or {}
        if set(options)-{'scenario','capabilities'}:raise ValueError('Unknown reset options')
        from .scenarios import ScenarioSpec
        explicit=options.get('scenario',cfg.get('scenario'))
        if explicit is not None or cfg.get('scenario_sampling'):
            self.scenario=ScenarioSpec.from_dict(explicit) if explicit is not None else self.scenario_sampler.sample()
            original=cfg['team']
            cfg.update(team=self.scenario.learner_side,lineups=self.scenario.lineups())
            if original!=cfg['team']:
                cfg['controllers']={('blue' if key.startswith('red:') else 'red')+':'+key.split(':')[1]:spec for key,spec in cfg['controllers'].items()}
            cfg['controllers'][f"{cfg['team']}:{cfg['slot']}"]={'kind':'learning'}
        self.episode_index+=1
        self.stage = self.requested_stage
        if cfg.get('alternate_sides') and self.scenario is None:
            original=cfg['team'];cfg['team']='red' if self.seed%2==0 else 'blue'
            if original!=cfg['team']:
                if cfg.get('lineups'):cfg['lineups']={t:cfg['lineups']['blue' if t=='red' else 'red'] for t in ('red','blue')}
                cfg['controllers']={('blue' if key.startswith('red:') else 'red')+':'+key.split(':')[1]:spec for key,spec in cfg['controllers'].items()}
        self.training_capabilities=copy.deepcopy(options.get('capabilities',cfg.get('training_capabilities',{})))
        if not self.training_capabilities and cfg.get('capability_sampling'):
            self.training_capabilities=self.capability_sampler.sample(cfg['lineups'],cfg['team'])
        learning = f"{cfg['team']}:{cfg['slot']}"
        assignments = cfg['controllers']
        if assignments.get(learning, {}).get('kind') != 'learning' or sum(v['kind'] == 'learning' for v in assignments.values()) != 1:
            raise ValueError('Exactly one learning controller must own the selected robot')
        if cfg.get('policy_version')==5:
            from .privileged_aux import AuxController
            self.controller=AuxController(cfg)
        else:self.controller = GoalController(cfg)
        overrides = {(cfg['team'], cfg['slot']): self.controller}
        if cfg.get('policy_version') in (3,4,5):
            other='blue' if cfg['team']=='red' else 'red'
            ref_cfg={**cfg,'slot':7,'policy_version':3}
            overrides[(other,7)]=ReferenceSentryController(ref_cfg)
            if cfg.get('policy_version')==5:overrides[(cfg['team'],7)]=ReferenceSentryController(ref_cfg)
        if cfg.get('auxiliary_pool'):
            from .privileged_aux import AuxController
            from .aux_policy import load_portable
            for team in ('red','blue'):
                for slot in (1,3,4):
                    role='heavy' if slot==1 else 'infantry'
                    if (team,slot) not in overrides and role in cfg['auxiliary_pool']:
                        model=load_portable(resource(cfg['auxiliary_pool'][role]))
                        overrides[(team,slot)]=AuxController({**cfg,'team':team,'slot':slot},model=model)
        if self.stage == 'course':
            for team in ('red', 'blue'):
                for slot in (1, 3, 4, 6, 7):
                    if (team, slot) not in overrides:
                        overrides[(team, slot)] = ParkedController(slot)
        try:
            for key, spec in assignments.items():
                team, slot = key.split(':')
                if team not in ('red', 'blue'):
                    raise ValueError('Unknown controller team')
                if spec['kind'] == 'learning' or self.stage == 'course':
                    continue
                if spec['kind'] not in ('script', 'frozen'):
                    raise ValueError('Controller kind must be learning, script or frozen')
                manifest, entries = load_submission(resource(spec['submission']))
                path = entries[str(int(slot))]
                overrides[(team, int(slot))] = Policy(path, self.seed*100+int(slot), 'trusted-local', cfg['policy_timeout_s'],
                    expected_digest=manifest['hashes'][manifest['entries'][str(int(slot))]])
            sources = (resource(cfg['friendly']), resource(cfg['opponent']))
            if self.stage == 'course':
                sources = (resource(cfg['friendly']), resource(cfg['friendly']))
            red, blue = sources if cfg['team'] == 'red' else sources[::-1]
            self.battle = Battle(red, blue, self.seed, cfg['duration'], cfg['map'], sandbox='trusted-local', timeout=cfg['policy_timeout_s'],
                controller_overrides=overrides, record_replay=self.record_replay, side_lineups=cfg.get('lineups'),
                navigation_mode=cfg['navigation_mode'], battle_settings={'drones_enabled': cfg['drones_enabled'], 'timed_assembly': cfg['timed_assembly'] and self.stage!='course'}, physics_backend=cfg.get('sim_backend', 'rapier'), sim_device=cfg.get('sim_device', 'cuda:0'), fixed_policy_hz=cfg['fixed_policy_hz'], learning_control_hz=cfg['learning_control_hz'],observation_profile=cfg.get('observation_profile'),training_capabilities=self.training_capabilities,
                radar_classifier=json.loads(resource(cfg['radar_classifier_path']).read_text()) if cfg.get('radar_classifier_path') else None)
            self.battle.match.training_summary = cfg['training_summary'] and not self.record_replay
            self.battle.metadata['rl'] = provenance(cfg)
            # These are development experiments, never ranked submissions.
            self.battle.metadata['test_match'] = True
            self.obs = self.battle.observation(cfg['team'], [cfg['slot']])
            if self.stage == 'course':
                from rmuc2027_core import sdk
                robot = own_robot(self.obs, cfg['slot'])
                design = sdk._design(robot)
                # Start from the actual spawn; no teleports or invisible force.
                # Basic travel alternates with the chassis's ability approach.
                capable = design.get('transform_allowed') or design.get('capabilities', {}).get('step_climb_m', 0)>0
                travel_goal=(7.,10.) if cfg.get('guided_locomotion') else (7.8,13.9)
                goal = (12., 7.5) if capable and self.seed%3 else travel_goal
                kind='ability' if capable and self.seed%3 else 'travel'
                if design.get('capabilities',{}).get('jump') and self.seed%4==0:
                    kind='flat_jump'
                    goal=(3.,6.15) if cfg['slot']==7 else (5.8,6.15)
                if cfg['team']=='blue':
                    goal = (28-goal[0], 15-goal[1])
                if cfg.get('guided_locomotion') and kind=='flat_jump':
                    goal=(robot['x'],robot['y'])
                self.controller.state.update(course_goal=list(goal), goal=list(goal), goal_time=self.obs['time'],course_kind=kind)
                self.course_best = math.dist((robot['x'], robot['y']), goal)
                self.course_hold = 0.
                self.course_airborne=False
                self.course_jump_complete=False
                self.course_requires_unfold=bool(cfg.get('guided_locomotion') and kind=='ability' and design.get('transform_allowed'))
            self.opening_seconds = 0.0
            opening_started = time.monotonic()
            last_opening_report = opening_started
            while cfg.get('policy_version')!=5 and self.stage!='course' and not learning_ready(own_robot(self.obs, cfg['slot']), self.controller.state):
                now = time.monotonic()
                if self.deadline is not None and now >= self.deadline:
                    raise OpeningDeadline('Training wall-time budget expired during fixed heavy opening')
                if now-last_opening_report >= 15:
                    print(json.dumps({'kind':'fixed-opening-progress', 'slot':cfg['slot'],
                        'simulation_seconds':self.battle.match.time, 'wall_seconds':now-opening_started,
                        'launcher':own_robot(self.obs, cfg['slot']).get('launcher'),
                        'position':[own_robot(self.obs,cfg['slot']).get(k) for k in ('x','y','z')],
                        'learning_samples':0}), flush=True)
                    last_opening_report = now
                if self.battle.match.status == 'finished' or self.battle.match.time >= cfg['episode_seconds']:
                    raise OpeningIncomplete('Heavy opening ended before launcher installation; no match RL samples collected')
                self.battle.step()
                if self.battle.diagnostics:
                    raise RuntimeError(f'Invalid fixed opening: {self.battle.diagnostics[-1]}')
                self.obs = self.battle.observation(cfg['team'], [cfg['slot']])
                self.opening_seconds = self.battle.match.time
            self.battle.held_commands[cfg['team']].pop(str(cfg['slot']), None)
            self.metrics = {'alive_seconds': 0.0, 'weak_seconds': 0.0, 'deaths': 0, 'revives': 0,
                'takeover_seconds': {mode: 0.0 for mode in ('dead', 'recover', 'tipped', 'unreachable')},
                'capture_seconds': 0.0, 'policy_errors': 0}
            if cfg.get('policy_version',1) in (2,3,4,5):
                self.metrics.update(path_m=0.,tipped_seconds=0.,jump_commands=0,fold_reversals=0,rejected_skills=0)
            if cfg.get('policy_version')==3:
                enemy='blue' if cfg['team']=='red' else 'red'
                self.objective_zones=[z for z in self.obs['arena']['zones'] if z.get('owner')==enemy and z.get('kind') in ('outpost','central','fortress','base')]
                self.objective_best=self.objective_gap(own_robot(self.obs,7))
                self.metrics['objective_gap_m']=self.objective_best
            self.last_fold=None
            self.reward_totals = dict.fromkeys(cfg['rewards'], 0.0)
            self.done = False
            return self.features(), self.info({})
        except BaseException:
            for controller in overrides.values():
                controller.close()
            self.close()
            raise

    def features(self):
        if self.config.get('policy_version')==5:
            from .privileged_aux import encode
            return encode(self.obs)
        actor=encode_observation(self.obs, self.controller.state, self.config['slot'], self.config['rl_seconds'], self.config)
        if self.config.get('policy_version')==4:
            from .critic_state import encode
            return {'actor':actor,'critic':encode(self.battle.match._frame,self.config['team'],self.battle.match.training_capabilities)}
        return actor

    def sampler_state(self):
        return dict(episode_index=self.episode_index,scenario=self.scenario_sampler.state_dict(),
                    capabilities=self.capability_sampler.state_dict(),rng_state=self.rng.getstate())

    def restore_sampler_state(self,state):
        from .scenarios import ScenarioSampler,_tuples
        from .capabilities import CapabilitySampler
        self.scenario_sampler=ScenarioSampler.from_state_dict(state['scenario'])
        self.capability_sampler=CapabilitySampler.from_state_dict(state['capabilities'])
        self.episode_index=int(state['episode_index']);self.rng.setstate(_tuples(state['rng_state']))

    def objective_gap(self,robot):
        from rmuc2027_core.sdk import point_in_polygon
        x,y=robot['x'],robot['y'];distances=[]
        for z in self.objective_zones:
            polygon=z.get('polygon')
            if not polygon:
                distances.append(max(0.,math.hypot(x-z['x'],y-z['y'])-z.get('radius',0)));continue
            if point_in_polygon(x,y,polygon):distances.append(0.);continue
            for a,b in zip(polygon,polygon[1:]+polygon[:1]):
                dx,dy=b[0]-a[0],b[1]-a[1]
                t=max(0.,min(1.,((x-a[0])*dx+(y-a[1])*dy)/max(1e-12,dx*dx+dy*dy)))
                distances.append(math.hypot(x-a[0]-t*dx,y-a[1]-t*dy))
        return min(distances)

    def step(self, action):
        if self.done or self.battle is None:
            raise RuntimeError('reset() is required before step()')
        try:
            return self._step(action)
        except BaseException:
            self.close()
            raise

    def _step(self, action):
        cfg, battle = self.config, self.battle
        if cfg.get('policy_version') in (3,4):self.controller.state['control_steps']=[]
        self.controller.select(self.obs, action)
        team, other = cfg['team'], 'blue' if cfg['team'] == 'red' else 'red'
        terms = dict.fromkeys(cfg['rewards'], 0.0)
        if cfg.get('policy_version', 1) in (2,3,4):
            terms['locomotion'] = 0.
        sampled_seconds=0.0
        for _ in range(round(cfg['rl_seconds']/DT)):
            before = battle.match._frame
            previous = next(r for r in before['robots'] if r['team'] == team and r['slot'] == cfg['slot'])
            frame = battle.step()
            robot = next(r for r in frame['robots'] if r['team'] == team and r['slot'] == cfg['slot'])
            if cfg.get('policy_version') in (3,4):
                trace=copy.deepcopy(self.controller.state.get('execution_trace',{}))
                trace.update(result_time=frame['time'],result=dict(grounded=robot.get('grounded'),
                    fold=robot.get('transform_progress'),yaw_rate=robot.get('angular_velocity',[0,0,0])[2]))
                self.controller.state['control_steps'].append(trace)
                skills=self.controller.state.get('navigation',{}).get('executed_skills',{})
                self.metrics['jump_commands']+=int(bool(skills.get('jump')))
                fold=skills.get('transform')
                if fold is not None:
                    self.metrics['fold_reversals']+=int(self.last_fold is not None and self.last_fold!=fold)
                    self.last_fold=fold
            dt = frame['time']-before['time']
            sampled_seconds+=dt
            delta = ((frame['teams'][team]['control']-frame['teams'][other]['control'])
                -(before['teams'][team]['control']-before['teams'][other]['control']))
            terms['control'] += cfg['rewards']['control']*delta
            died = previous['alive'] and not robot['alive']
            self.metrics['deaths'] += int(died)
            self.metrics['revives'] += int(not previous['alive'] and robot['alive'])
            if cfg.get('policy_version', 1) in (2,3,4,5):
                self.metrics['path_m']+=math.hypot(robot['x']-previous['x'],robot['y']-previous['y'])
                self.metrics['tipped_seconds']+=dt*float(robot.get('tipped',False))
                if cfg.get('policy_version') in (2,3):terms['locomotion'] -= float(robot.get('tipped', False) and not previous.get('tipped', False))
                if cfg.get('policy_version')==3 and robot['alive']:
                    # One-time progress credit towards public capture regions; no
                    # target override or reward-state features enter the actor.
                    gap=self.objective_gap(robot)
                    terms['locomotion']+=.2*max(0.,self.objective_best-gap)
                    self.objective_best=min(self.objective_best,gap)
                    self.metrics['objective_gap_m']=gap
                if self.stage == 'course':
                    jump_was_complete=self.course_jump_complete
                    if not robot.get('grounded',True) and self.controller.state.get('navigation',{}).get('last_jump',-100.)>=0:
                        self.course_airborne=True
                    if self.course_airborne and robot.get('grounded') and robot.get('upright',1.)>=.95:
                        self.course_jump_complete=True
                    if (cfg.get('guided_locomotion') and self.controller.state['course_kind']=='flat_jump'
                            and self.course_jump_complete and not jump_was_complete):
                        terms['locomotion'] += 1.
                    gap = math.dist((robot['x'], robot['y']), self.controller.state['course_goal'])
                    terms['locomotion'] += .1*max(0., self.course_best-gap)-.01*dt
                    self.course_best = min(self.course_best, gap)
                    settled = (gap<.25 and robot.get('grounded') and robot.get('upright', 1.)>=.95
                               and math.hypot(robot.get('vx', 0.), robot.get('vy', 0.))<.2)
                    settled &= self.controller.state['course_kind']!='flat_jump' or self.course_jump_complete
                    if self.course_requires_unfold:
                        settled &= robot.get('transform_progress',0.)<=.01
                    self.course_hold = self.course_hold+dt if settled else 0.
            terms['death'] += cfg['rewards']['death']*died
            self.metrics['alive_seconds'] += dt*robot['alive']
            self.metrics['weak_seconds'] += dt*bool(robot.get('weak'))
            mode = self.controller.state.get('mode', 'policy')
            if mode in self.metrics['takeover_seconds']:
                self.metrics['takeover_seconds'][mode] += dt
            occupying = robot['alive'] and not robot.get('weak') and any(
                robot['id'] in frame['zones'][zone].get('occupants', [])
                for p in frame['positions'].values() if p['owner'] == other
                for zone in p['zones'])
            self.metrics['capture_seconds'] += dt*occupying
            terms['capture_seconds'] += cfg['rewards']['capture_seconds']*dt*occupying
            if frame['status'] == 'finished' or frame['time']+1e-8 >= cfg['episode_seconds'] or self.stage=='course' and (frame['time']>=40. or self.course_hold>=2.):
                break
        terminated = battle.match.status == 'finished'
        truncated = not terminated and battle.match.time+1e-8 >= cfg['episode_seconds']
        if self.stage == 'course':
            truncated |= battle.match.time>=40. or self.course_hold>=2. or self.requested_stage!=self.stage
            # Competition rewards have no meaning against parked course peers.
            terms = {key: (value if key=='locomotion' else 0.) for key,value in terms.items()}
            if self.course_hold>=2.:
                terms['locomotion'] += 5.
        if cfg.get('policy_version', 1) in (2,3,4):
            nav=self.controller.state.get('navigation',{})
            rejected=len(nav.get('rejected',[]))
            if cfg.get('policy_version') in (2,3):terms['locomotion'] -= .05*rejected
            self.metrics['rejected_skills']+=rejected
            skills=nav.get('executed_skills',{})
            if cfg.get('policy_version')==2:self.metrics['jump_commands']+=int(bool(skills.get('jump')))
            fold=skills.get('transform')
            if cfg.get('policy_version')==2 and fold is not None:
                self.metrics['fold_reversals']+=int(self.last_fold is not None and self.last_fold!=fold)
                self.last_fold=fold
        if terminated:
            terms['outcome'] = cfg['rewards']['outcome']*(0 if battle.match.winner is None else 1 if battle.match.winner == team else -1)
        self.done = terminated or truncated
        if self.done and self.scenario is not None and not cfg.get('evaluation'):
            self.scenario_sampler.record(self.scenario,valid_steps=round(battle.match.time/cfg['rl_seconds']),
                seconds=battle.match.time,completed=True,failure=bool(self.metrics['deaths'] or self.metrics.get('tipped_seconds',0)>1))
        self.obs = battle.observation(team, [cfg['slot']])
        if self.calibration_recorder is not None and not cfg.get('evaluation'):
            self.calibration_recorder.observe(battle.match,f'role{cfg["slot"]}-worker{cfg.get("_worker_id",0)}-seed{self.seed}-episode{self.episode_index}')
        self.metrics['policy_errors'] = len(battle.diagnostics)
        if battle.diagnostics:
            raise RuntimeError(f'Fixed controller error; rollout is invalid: {battle.diagnostics[-1]}')
        for name, value in terms.items():
            self.reward_totals[name] = self.reward_totals.get(name, 0.)+value
        reward = sum(terms.values())
        if not math.isfinite(reward):
            raise ValueError('Non-finite reward')
        info=self.info(terms)
        info['sampled_simulation_seconds']=sampled_seconds
        return self.features(), reward, terminated, truncated, info

    def info(self, terms):
        battle, cfg = self.battle, self.config
        return {'stage': self.stage, 'opening_simulation_seconds': self.opening_seconds, 'seed': self.seed, 'tick': battle.match.tick, 'time': battle.match.time,
            'course':dict(kind=self.controller.state['course_kind'],success=self.course_hold>=2.,closest_goal_m=self.course_best,
                          jump_complete=self.course_jump_complete) if self.stage=='course' else None,
            'winner': battle.match.winner, 'reason': battle.match.reason,
            'bases': {team: {'hp': battle.match.bases[team]['hp'], 'max_hp': battle.match.bases[team]['max_hp']} for team in ('red', 'blue')},
            'control': {t: battle.match.teams[t]['control'] for t in ('red', 'blue')},
            'metrics': copy.deepcopy(self.metrics), 'reward_terms': dict(terms),
            'reward_totals': dict(self.reward_totals),
            'action_trace': copy.deepcopy({key: self.controller.state.get(key) for key in
                ('supplied_action', 'clipped_action', 'proposed_goal', 'goal', 'executed_goal', 'executed_action', 'mode', 'navigation', 'execution_trace', 'control_steps')})}

    def save_replay(self, path):
        if self.battle is None:
            raise RuntimeError('No active match')
        if self.battle.match.status != 'finished':
            self.battle.match.finish(None, 'stopped')
            self.battle.frames[-1] = self.battle.match.snapshot()
        self.battle.save(path)

    def close(self):
        self.done = True
        battle, self.battle = self.battle, None
        try:
            if battle is not None:
                battle.close()
        finally:
            if battle is not None:
                battle.frames.file.close()
            if self.controller is not None:
                self.controller.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
