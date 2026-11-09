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
from .portable import SdkNavigator, encode_observation, execute_goal, select_goal, fixed_opening, learning_ready, own_robot

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
    config.setdefault('timed_assembly', False)
    if not isinstance(config['timed_assembly'], bool):
        raise ValueError('timed_assembly must be boolean')
    config.setdefault('drones_enabled', True)
    if not isinstance(config['drones_enabled'], bool):
        raise ValueError('drones_enabled must be boolean')
    config.setdefault('fixed_policy_hz', 10)
    config.setdefault('learning_control_hz', 10)
    if config['learning_control_hz'] not in (2, 5, 10):
        raise ValueError('learning_control_hz must be 2, 5 or 10')
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
            'dt':1/120,'experimental':True,
            'frame_encoder':FRAME_ENCODER,
            'pipe_reader':'posix-select' if transport._DIRECT_PIPE_READER else 'threaded',
            'policy_observation_cache':transport._CACHE_POLICY_OBSERVATIONS,
            'transport_sha256':hashlib.sha256((ROOT/'packages/core/rmuc2027_core/engine.py').read_bytes()).hexdigest(),
            'files':{name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
                     for name in ('__init__.py','arena.py','motion.py','contacts.py','terrain.py','traversal.py','pose.py','protocol.py','uploads.py','batching.py','vector.py')},
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


class ArenaSession:
    """Standard-library core used directly by diagnostic CLI and by Gymnasium.

    Actor inputs come only from scoped observations. Full frames are consumed
    separately by reward/metric accounting and optional replay recording.
    """
    def __init__(self, config=None, record_replay=False):
        self.config = load_config(config)
        self.deadline = self.config.pop('_training_deadline', None)
        self.record_replay = record_replay
        self.rng = random.Random(self.config['seed'])
        self.battle = None
        self.controller = None
        self.done = True

    def reset(self, seed=None):
        self.close()
        if seed is not None:
            self.rng.seed(int(seed))
        self.seed = int(seed) if seed is not None else self.rng.randrange(2**31)
        cfg = self.config
        learning = f"{cfg['team']}:{cfg['slot']}"
        assignments = cfg['controllers']
        if assignments.get(learning, {}).get('kind') != 'learning' or sum(v['kind'] == 'learning' for v in assignments.values()) != 1:
            raise ValueError('Exactly one learning controller must own the selected robot')
        self.controller = GoalController(cfg)
        overrides = {(cfg['team'], cfg['slot']): self.controller}
        try:
            for key, spec in assignments.items():
                team, slot = key.split(':')
                if team not in ('red', 'blue'):
                    raise ValueError('Unknown controller team')
                if spec['kind'] == 'learning':
                    continue
                if spec['kind'] not in ('script', 'frozen'):
                    raise ValueError('Controller kind must be learning, script or frozen')
                manifest, entries = load_submission(resource(spec['submission']))
                path = entries[str(int(slot))]
                overrides[(team, int(slot))] = Policy(path, self.seed*100+int(slot), 'trusted-local', cfg['policy_timeout_s'],
                    expected_digest=manifest['hashes'][manifest['entries'][str(int(slot))]])
            sources = (resource(cfg['friendly']), resource(cfg['opponent']))
            red, blue = sources if cfg['team'] == 'red' else sources[::-1]
            self.battle = Battle(red, blue, self.seed, cfg['duration'], cfg['map'], sandbox='trusted-local', timeout=cfg['policy_timeout_s'],
                controller_overrides=overrides, record_replay=self.record_replay, side_lineups=cfg.get('lineups'),
                navigation_mode=cfg['navigation_mode'], battle_settings={'drones_enabled': cfg['drones_enabled'], 'timed_assembly': cfg['timed_assembly']}, physics_backend=cfg.get('sim_backend', 'rapier'), sim_device=cfg.get('sim_device', 'cuda:0'), fixed_policy_hz=cfg['fixed_policy_hz'], learning_control_hz=cfg['learning_control_hz'])
            self.battle.match.training_summary = cfg['training_summary'] and not self.record_replay
            self.battle.metadata['rl'] = provenance(cfg)
            # These are development experiments, never ranked submissions.
            self.battle.metadata['test_match'] = True
            self.obs = self.battle.observation(cfg['team'], [cfg['slot']])
            self.opening_seconds = 0.0
            opening_started = time.monotonic()
            last_opening_report = opening_started
            while not learning_ready(own_robot(self.obs, cfg['slot']), self.controller.state):
                now = time.monotonic()
                if self.deadline is not None and now >= self.deadline:
                    raise OpeningDeadline('Training wall-time budget expired during fixed heavy opening')
                if now-last_opening_report >= 15:
                    print(json.dumps({'kind':'fixed-opening-progress', 'slot':cfg['slot'],
                        'simulation_seconds':self.battle.match.time, 'wall_seconds':now-opening_started,
                        'launcher':own_robot(self.obs, cfg['slot']).get('launcher'), 'learning_samples':0}), flush=True)
                    last_opening_report = now
                if self.battle.match.status == 'finished' or self.battle.match.time >= cfg['episode_seconds']:
                    raise RuntimeError('Heavy opening ended before launcher installation; no RL samples collected')
                self.battle.step()
                if self.battle.diagnostics:
                    raise RuntimeError(f'Invalid fixed opening: {self.battle.diagnostics[-1]}')
                self.obs = self.battle.observation(cfg['team'], [cfg['slot']])
                self.opening_seconds = self.battle.match.time
            self.battle.held_commands[cfg['team']].pop(str(cfg['slot']), None)
            self.metrics = {'alive_seconds': 0.0, 'weak_seconds': 0.0, 'deaths': 0, 'revives': 0,
                'takeover_seconds': {mode: 0.0 for mode in ('dead', 'recover', 'tipped', 'unreachable')},
                'capture_seconds': 0.0, 'policy_errors': 0}
            self.reward_totals = dict.fromkeys(cfg['rewards'], 0.0)
            self.done = False
            return self.features(), self.info({})
        except BaseException:
            for controller in overrides.values():
                controller.close()
            self.close()
            raise

    def features(self):
        return encode_observation(self.obs, self.controller.state, self.config['slot'], self.config['rl_seconds'])

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
        self.controller.select(self.obs, action)
        team, other = cfg['team'], 'blue' if cfg['team'] == 'red' else 'red'
        terms = dict.fromkeys(cfg['rewards'], 0.0)
        sampled_seconds=0.0
        for _ in range(round(cfg['rl_seconds']/DT)):
            before = battle.match._frame
            previous = next(r for r in before['robots'] if r['team'] == team and r['slot'] == cfg['slot'])
            frame = battle.step()
            robot = next(r for r in frame['robots'] if r['team'] == team and r['slot'] == cfg['slot'])
            dt = frame['time']-before['time']
            sampled_seconds+=dt
            delta = ((frame['teams'][team]['control']-frame['teams'][other]['control'])
                -(before['teams'][team]['control']-before['teams'][other]['control']))
            terms['control'] += cfg['rewards']['control']*delta
            died = previous['alive'] and not robot['alive']
            self.metrics['deaths'] += int(died)
            self.metrics['revives'] += int(not previous['alive'] and robot['alive'])
            terms['death'] += cfg['rewards']['death']*died
            self.metrics['alive_seconds'] += dt*robot['alive']
            self.metrics['weak_seconds'] += dt*bool(robot.get('weak'))
            mode = self.controller.state.get('mode', 'policy')
            if mode != 'policy':
                self.metrics['takeover_seconds'][mode] += dt
            occupying = robot['alive'] and not robot.get('weak') and any(
                robot['id'] in frame['zones'][zone].get('occupants', [])
                for p in frame['positions'].values() if p['owner'] == other
                for zone in p['zones'])
            self.metrics['capture_seconds'] += dt*occupying
            terms['capture_seconds'] += cfg['rewards']['capture_seconds']*dt*occupying
            if frame['status'] == 'finished' or frame['time']+1e-8 >= cfg['episode_seconds']:
                break
        terminated = battle.match.status == 'finished'
        truncated = not terminated and battle.match.time+1e-8 >= cfg['episode_seconds']
        if terminated:
            terms['outcome'] = cfg['rewards']['outcome']*(0 if battle.match.winner is None else 1 if battle.match.winner == team else -1)
        self.done = terminated or truncated
        self.obs = battle.observation(team, [cfg['slot']])
        self.metrics['policy_errors'] = len(battle.diagnostics)
        if battle.diagnostics:
            raise RuntimeError(f'Fixed controller error; rollout is invalid: {battle.diagnostics[-1]}')
        for name, value in terms.items():
            self.reward_totals[name] += value
        reward = sum(terms.values())
        if not math.isfinite(reward):
            raise ValueError('Non-finite reward')
        info=self.info(terms)
        info['sampled_simulation_seconds']=sampled_seconds
        return self.features(), reward, terminated, truncated, info

    def info(self, terms):
        battle, cfg = self.battle, self.config
        return {'opening_simulation_seconds': self.opening_seconds, 'seed': self.seed, 'tick': battle.match.tick, 'time': battle.match.time,
            'winner': battle.match.winner, 'reason': battle.match.reason,
            'bases': {team: {'hp': battle.match.bases[team]['hp'], 'max_hp': battle.match.bases[team]['max_hp']} for team in ('red', 'blue')},
            'control': {t: battle.match.teams[t]['control'] for t in ('red', 'blue')},
            'metrics': copy.deepcopy(self.metrics), 'reward_terms': dict(terms),
            'reward_totals': dict(self.reward_totals),
            'action_trace': copy.deepcopy({key: self.controller.state.get(key) for key in
                ('supplied_action', 'clipped_action', 'proposed_goal', 'goal', 'executed_goal', 'executed_action', 'mode')})}

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
