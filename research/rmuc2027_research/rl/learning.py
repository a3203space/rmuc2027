"""Optional SB3/PyTorch implementation, imported only for train/evaluate."""
import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import time
from functools import partial

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .env import SentryEnv, make_env
from .portable import OBS_SIZE, OBS_VERSION
from .session import provenance


def device_name(requested):
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA explicitly requested, but torch.cuda.is_available() is false; use --device cpu')
    print(json.dumps({'device': requested, 'torch': torch.__version__, 'cuda_available': torch.cuda.is_available()}), flush=True)
    return requested


def vector_env(config, count):
    if count < 1:
        raise ValueError('num-envs must be positive')
    factories = [partial(make_env, config) for _ in range(count)]
    vec = DummyVecEnv(factories) if count == 1 else SubprocVecEnv(factories, start_method='spawn')
    vec.seed(config['seed'])
    return vec


class TimedPPO(PPO):
    def train(self):
        started = time.perf_counter()
        super().train()
        seconds = time.perf_counter()-started
        self.optimization_seconds = getattr(self, 'optimization_seconds', 0.0)+seconds


class TraceCallback(BaseCallback):
    def __init__(self, path):
        super().__init__()
        self.path = Path(path)
        self.sampling_seconds = 0.0
        self.last_metrics = []

    def _on_training_start(self):
        self.stream = self.path.open('w')

    def _on_rollout_start(self):
        self.started = time.perf_counter()

    def _on_rollout_end(self):
        self.sampling_seconds += time.perf_counter()-self.started

    def _on_step(self):
        for index, info in enumerate(self.locals['infos']):
            # SB3 stores self.locals['actions'] and its original log probability
            # in RolloutBuffer. Do not replace them with clipped/snapped goals.
            self.stream.write(json.dumps({'env': index, 'total_steps': self.num_timesteps,
                'sampled_action': self.locals['actions'][index].tolist(),
                'environment_action': self.locals['clipped_actions'][index].tolist(),
                **info}, allow_nan=False, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value.item())+'\n')
        self.last_metrics = self.locals['infos']
        return True

    def _on_training_end(self):
        self.stream.close()


def _actor_parameters(model):
    return {name: value.detach().cpu().clone() for name, value in model.policy.named_parameters()
        if 'policy_net' in name or 'action_net' in name}


def experiment(config):
    from .. import __version__
    return {**provenance(config), 'obs_version': OBS_VERSION, 'obs_size': OBS_SIZE,
        'arena_version': __version__, 'dependencies': {name: importlib.metadata.version(name) for name in
            ('torch', 'gymnasium', 'stable-baselines3', 'numpy')}}


def train(config, output, steps, num_envs=1, device='cpu', resume=None):
    device_name(device)
    torch.set_num_threads(1)
    torch.manual_seed(config['seed'])
    torch.use_deterministic_algorithms(True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    metadata = experiment(config)
    if resume:
        previous = json.loads((Path(resume).parent/'experiment.json').read_text())
        for key in ('config', 'sources', 'sdk_hash', 'obs_version', 'obs_size'):
            if metadata[key] != previous[key]:
                raise ValueError(f'Resume incompatibility: {key}')
        for key in ('runtime_hash', 'engine_binary_sha256', 'rules_sha256'):
            if previous.get(key, metadata[key]) != metadata[key]:
                raise ValueError(f'Resume incompatibility: {key}')
    (output/'experiment.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    vec = vector_env(config, num_envs)
    callback = TraceCallback(output/'actions.jsonl')
    started = time.perf_counter()
    model = None
    try:
        if resume:
            model = TimedPPO.load(resume, env=vec, device=device)
        else:
            model = TimedPPO('MlpPolicy', vec, seed=config['seed'], device=device, verbose=1,
                policy_kwargs={'net_arch': {'pi': [64, 64], 'vf': [64, 64]}}, **config['ppo'])
        block = model.n_steps*num_envs
        if steps <= 0 or steps % block:
            raise ValueError(f'Steps must be a positive multiple of rollout size {block}; no hidden overshoot')
        before = _actor_parameters(model)
        previous_steps, previous_updates = model.num_timesteps, model._n_updates
        model.optimization_seconds = 0.0
        model.learn(total_timesteps=steps, callback=callback, reset_num_timesteps=resume is None)
        after = _actor_parameters(model)
        delta = math.sqrt(sum(float(torch.sum((after[name]-value)**2)) for name, value in before.items()))
        if not math.isfinite(delta) or delta <= 0:
            raise RuntimeError('Actor did not receive a finite, nonzero parameter update')
        model.save(output/'model.zip')
        # Check predictions and optimizer restoration from the file just saved.
        loaded = TimedPPO.load(output/'model.zip', device=device)
        check_obs = np.zeros((OBS_SIZE,), dtype=np.float32)
        assert np.array_equal(model.predict(check_obs, deterministic=True)[0], loaded.predict(check_obs, deterministic=True)[0])
        assert len(loaded.policy.optimizer.state_dict()['state']) > 0
        losses = {str(key): float(value) for key, value in model.logger.name_to_value.items()
                  if key.startswith('train/') and np.isscalar(value)}
        if not all(math.isfinite(value) for value in losses.values()):
            raise RuntimeError('Non-finite PPO training statistic')
        report = {'actual_steps': model.num_timesteps-previous_steps, 'total_steps': model.num_timesteps,
            'gradient_epochs': model._n_updates-previous_updates, 'actor_parameter_l2_change': delta,
            'reload_prediction_equal': True, 'optimizer_reloaded': True,
            'num_envs': num_envs, 'device': str(model.device),
            'wall_seconds': time.perf_counter()-started, 'sampling_seconds': callback.sampling_seconds,
            'optimization_seconds': model.optimization_seconds,
            'rl_steps_per_second': steps/max(callback.sampling_seconds, 1e-9),
            'simulation_seconds_per_second': steps*config['rl_seconds']/max(callback.sampling_seconds, 1e-9),
            'losses': losses, 'model_sha256': hashlib.sha256((output/'model.zip').read_bytes()).hexdigest()}
        (output/'training.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        return report
    except KeyboardInterrupt:
        if model is not None:
            model.save(output/'interrupted.zip')
        raise
    finally:
        if hasattr(callback, 'stream') and not callback.stream.closed:
            callback.stream.close()
        vec.close()


def evaluate(model_path, output, seeds, sides, duration=420, device='cpu'):
    device_name(device)
    torch.set_num_threads(1)
    model_path, output = Path(model_path), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((model_path.parent/'experiment.json').read_text())
    current = experiment(metadata['config'])
    for key in ('sources', 'sdk_hash', 'obs_version', 'obs_size'):
        if current[key] != metadata[key]:
            raise ValueError(f'Evaluation incompatibility: {key}')
    for key in ('runtime_hash', 'engine_binary_sha256', 'rules_sha256'):
        if metadata.get(key, current[key]) != current[key]:
            raise ValueError(f'Evaluation incompatibility: {key}')
    model = PPO.load(model_path, device=device)
    results = []
    started = time.perf_counter()
    for seed in seeds:
        for side in sides:
            config = copy.deepcopy(metadata['config'])
            old_side = config['team']
            config.update(team=side, episode_seconds=duration)
            config['controllers'] = {f'{side if key.split(":")[0] == old_side else ("blue" if side == "red" else "red")}:{key.split(":")[1]}': value
                                     for key, value in config['controllers'].items()}
            env = SentryEnv(config, record_replay=True)
            try:
                obs, _ = env.reset(seed=seed)
                terminated = truncated = False
                total_reward = 0.0
                with (output/f'{seed}-{side}-actions.jsonl').open('w') as stream:
                    while not (terminated or truncated):
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = env.step(action)
                        total_reward += reward
                        stream.write(json.dumps(info, allow_nan=False)+'\n')
                outcome = ('win' if info['winner'] == side else 'draw' if info['winner'] is None else 'loss') if terminated else 'truncated'
                replay = output/f'{seed}-{side}.json'
                env.session.save_replay(replay)
                from rmuc2027_core.replay import verify_replay
                verified = verify_replay(replay)
                record = {**info, 'side': side, 'outcome': outcome, 'terminated': terminated,
                    'truncated': truncated, 'training_reward': total_reward, 'replay': str(replay), 'verified': verified}
                results.append(record)
                print(json.dumps(record), flush=True)
            finally:
                env.close()
    report = {'games': results, 'wall_seconds': time.perf_counter()-started,
        'summary': {key: sum(r['outcome'] == key for r in results) for key in ('win', 'draw', 'loss', 'truncated')}}
    (output/'evaluation.json').write_text(json.dumps(report, indent=2))
    return report
