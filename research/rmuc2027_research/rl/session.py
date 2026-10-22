"""Headless simulation session; the Gym adapter adds only spaces and seeding."""
import copy
import hashlib
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

from rmuc2027_core.engine import DT, binary_path
from rmuc2027_core.paths import ROOT
from ..strategies.battle import Battle
from rmuc2027_core.policy import EncodedObservation, Policy
from rmuc2027_core.submissions import load_submission
from .portable import SdkNavigator, encode_observation, execute_goal, select_goal

DEFAULT_CONFIG = ROOT / 'configs/rl/sentry.json'


def load_config(source=None):
    config = json.loads(DEFAULT_CONFIG.read_text())
    supplied = json.loads(Path(source).read_text()) if isinstance(source, (str, Path)) else copy.deepcopy(source or {})
    for key, value in supplied.items():
        if key in ('rewards', 'ppo'):
            config[key].update(value)
        else:
            config[key] = value
    if config['map'] != 'rmuc2027' or config['slot'] != 7 or config['team'] not in ('red', 'blue'):
        raise ValueError('v1 trains the RMUC 2027 slot-7 sentry, on either side')
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
    if config['team'] == 'blue' and config['controllers'] == {'red:7': {'kind': 'learning'}}:
        config['controllers'] = {'blue:7': {'kind': 'learning'}}
    for key, spec in config['controllers'].items():
        parts = key.split(':')
        if len(parts) != 2 or parts[0] not in ('red', 'blue') or parts[1] not in ('1', '3', '4', '6', '7'):
            raise ValueError('Controller keys must be canonical team:slot identifiers')
        if spec['kind'] not in ('learning', 'script', 'frozen') or (spec['kind'] != 'learning' and 'submission' not in spec):
            raise ValueError('Invalid controller assignment')
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
    result['engine_binary_sha256'] = hashlib.sha256(binary_path().read_bytes()).hexdigest()
    result['rules_sha256'] = hashlib.sha256((ROOT/'configs/rules.json').read_bytes()).hexdigest()
    return result


class GoalController:
    """Policy-compatible external target latch; one instance per episode."""
    def __init__(self, config):
        self.config = config
        self.navigator = SdkNavigator(config['grid_m'])
        self.baseline = baseline_module(config)
        self.state, self.logs = {}, []
        self.disabled, self.calls, self.errors, self.consecutive_timeouts = False, 0, 0, 0

    def select(self, obs, action):
        select_goal(obs, self.state, action, self.navigator, self.config)

    def decide(self, observation):
        obs = json.loads(observation.payload) if isinstance(observation, EncodedObservation) else observation
        self.calls += 1
        return execute_goal(obs, self.state, self.navigator, self.config, self.baseline), None

    def close(self):
        self.disabled = True
        self.navigator.grids.clear()


class ArenaSession:
    """Standard-library core used directly by diagnostic CLI and by Gymnasium.

    Actor inputs come only from scoped observations. Full frames are consumed
    separately by reward/metric accounting and optional replay recording.
    """
    def __init__(self, config=None, record_replay=False):
        self.config = load_config(config)
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
            raise ValueError('Exactly one learning controller must own the selected sentry')
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
                overrides[(team, int(slot))] = Policy(path, self.seed*100+int(slot), 'trusted-local', .2,
                    expected_digest=manifest['hashes'][manifest['entries'][str(int(slot))]])
            sources = (resource(cfg['friendly']), resource(cfg['opponent']))
            red, blue = sources if cfg['team'] == 'red' else sources[::-1]
            self.battle = Battle(red, blue, self.seed, cfg['duration'], cfg['map'], sandbox='trusted-local',
                controller_overrides=overrides, record_replay=self.record_replay)
            self.battle.metadata['rl'] = provenance(cfg)
            # These are development experiments, never ranked submissions.
            self.battle.metadata['test_match'] = True
            self.obs = self.battle.observation(cfg['team'], [cfg['slot']])
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
        for _ in range(round(cfg['rl_seconds']/DT)):
            before = battle.match._frame
            previous = next(r for r in before['robots'] if r['team'] == team and r['slot'] == cfg['slot'])
            frame = battle.step()
            robot = next(r for r in frame['robots'] if r['team'] == team and r['slot'] == cfg['slot'])
            dt = frame['time']-before['time']
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
        return self.features(), reward, terminated, truncated, self.info(terms)

    def info(self, terms):
        battle, cfg = self.battle, self.config
        return {'seed': self.seed, 'tick': battle.match.tick, 'time': battle.match.time,
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
