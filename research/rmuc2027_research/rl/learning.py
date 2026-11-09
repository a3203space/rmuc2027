"""Optional SB3/PyTorch implementation, imported only for train/evaluate."""
from .checkpoints import save_checkpoint
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
from functools import partial

import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .env import SentryEnv, make_env
from .portable import OBS_SIZE, OBS_VERSION
from .session import provenance, OpeningDeadline


def device_name(requested):
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA explicitly requested, but torch.cuda.is_available() is false; use --device cpu')
    print(json.dumps({'device': requested, 'torch': torch.__version__, 'cuda_available': torch.cuda.is_available()}), flush=True)
    return requested


def vector_env(config, count):
    if count < 1:
        raise ValueError('num-envs must be positive')
    factories = [partial(make_env, config) for _ in range(count)]
    if config.get('sim_backend', 'rapier') == 'mjwarp':
        from .mjwarp.vector import WarpVecEnv
        vec = WarpVecEnv(factories)
    elif config.get('sim_backend') == 'warp_kinematic' and count>1:
        from .warp_kinematic.vector import WarpVecEnv
        vec = WarpVecEnv(factories)
    else:
        vec = DummyVecEnv(factories) if count == 1 or config.get('slot') == 1 else SubprocVecEnv(factories, start_method='spawn')
    vec.seed(config['seed'])
    return vec


class TimedPPO(PPO):
    def apply_learning_rate_control(self):
        """Apply an operator request between complete PPO rollouts only."""
        path = getattr(self, 'learning_rate_control_path', None)
        if not path or not Path(path).exists():
            return
        try:
            request = json.loads(Path(path).read_text())
            value = request['learning_rate']
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError('learning_rate must be finite and positive')
            value = float(value)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(json.dumps({'kind': 'learning-rate-control-error', 'error': str(exc)}), flush=True)
            return
        previous = float(self.lr_schedule(self._current_progress_remaining))
        if value == previous:
            return
        self.learning_rate = value
        self.lr_schedule = lambda _: value
        event = {'kind': 'learning-rate-change', 'total_samples': self.num_timesteps,
                 'old': previous, 'new': value, 'reason': str(request.get('reason', 'operator request')),
                 'time_unix': time.time()}
        with Path(path).with_name('learning-rate-events.jsonl').open('a') as stream:
            stream.write(json.dumps(event)+'\n')
        print(json.dumps(event), flush=True)

    def train(self):
        self.apply_learning_rate_control()
        started = time.perf_counter()
        super().train()
        seconds = time.perf_counter()-started
        self.optimization_seconds = getattr(self, 'optimization_seconds', 0.0)+seconds
        path = getattr(self, 'update_log_path', None)
        if path:
            row = {'kind': 'ppo-update', 'total_samples': self.num_timesteps,
                   'gradient_epochs': self._n_updates, 'optimization_seconds': seconds,
                   'learning_rate': float(self.lr_schedule(self._current_progress_remaining)),
                   'behavior': getattr(self, 'last_rollout_behavior', {}),
                   'losses': {str(k): float(v) for k, v in self.logger.name_to_value.items()
                              if k.startswith('train/') and np.isscalar(v)}}
            with Path(path).open('a') as stream:
                stream.write(json.dumps(row)+'\n')


class TraceCallback(BaseCallback):
    def __init__(self, path, deadline=None, detailed=True, backup_seconds=300.):
        super().__init__()
        self.detailed = detailed
        self.path = Path(path)
        self.sampling_seconds = 0.0
        self.last_metrics = []
        self.deadline = deadline
        self.last_checkpoint = time.perf_counter()
        self.last_backup = self.last_checkpoint
        self.backup = None
        self.backup_seconds = backup_seconds
        self.timed_out = False
        self.last_progress = time.perf_counter()
        self.progress_started = self.last_progress

    def _on_training_start(self):
        self.stream = self.path.open('w')

    def _on_rollout_start(self):
        self.started = time.perf_counter()
        self.rollout_behavior = {'samples': 0, 'moving_commands': 0, 'modes': {}}

    def _on_rollout_end(self):
        self.sampling_seconds += time.perf_counter()-self.started
        self.model.last_rollout_behavior = copy.deepcopy(self.rollout_behavior)

    def _on_step(self):
        for index, info in enumerate(self.locals['infos']):
            trace = info.get('action_trace', {})
            action = trace.get('executed_action') or {}
            # execute_goal stores a single robot command, not the team map.
            move = action.get('move', [0., 0.])
            self.rollout_behavior['samples'] += 1
            self.rollout_behavior['moving_commands'] += int(math.hypot(*move) > .05)
            mode = trace.get('mode') or 'unknown'
            self.rollout_behavior['modes'][mode] = self.rollout_behavior['modes'].get(mode, 0)+1
            if self.locals['dones'][index]:
                row = {'env': index, 'total_samples': self.num_timesteps,
                       **{key: info.get(key) for key in ('seed', 'time', 'winner', 'reason', 'metrics',
                                                       'reward_totals', 'control', 'bases')}}
                with (self.path.parent/'episodes.jsonl').open('a') as episodes:
                    episodes.write(json.dumps(row)+'\n')
        for index, info in enumerate(self.locals['infos'] if self.detailed else []):
            # SB3 stores self.locals['actions'] and its original log probability
            # in RolloutBuffer. Do not replace them with clipped/snapped goals.
            self.stream.write(json.dumps({'env': index, 'total_steps': self.num_timesteps,
                'sampled_action': self.locals['actions'][index].tolist(),
                'environment_action': self.locals['clipped_actions'][index].tolist(),
                **info}, allow_nan=False, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value.item())+'\n')
        self.last_metrics = self.locals['infos']
        now = time.perf_counter()
        if now-self.last_progress >= 15:
            elapsed = now-self.progress_started
            block = self.model.n_steps*self.training_env.num_envs
            rollout_steps = int(self.locals.get('n_steps', 0))+1
            print(json.dumps({'kind': 'training-progress', 'total_samples': self.num_timesteps,
                'elapsed_seconds': elapsed, 'samples_per_second': self.num_timesteps/max(elapsed, 1e-9),
                'gradient_epochs': self.model._n_updates, 'rollout_samples': rollout_steps*self.training_env.num_envs,
                'rollout_capacity': block}), flush=True)
            if not self.detailed:
                self.stream.write(json.dumps({'kind': 'summary', 'total_steps': self.num_timesteps,
                    'gradient_epochs': self.model._n_updates, 'metrics': self.last_metrics})+'\n')
                self.stream.flush()
            self.last_progress = now
        if now-self.last_checkpoint >= 300:
            save_checkpoint(self.model, self.path.parent/'latest.zip')
            self.stream.flush()
            self.last_checkpoint = now
            if self.backup is not None and now-self.last_backup >= self.backup_seconds:
                self.backup.submit(self.path.parent/'latest.zip', {'total_steps':self.num_timesteps,
                    'gradient_epochs':self.model._n_updates,'elapsed_seconds':now-self.progress_started,
                    'learning_rate':float(self.model.lr_schedule(self.model._current_progress_remaining))})
                self.last_backup = now
        self.timed_out = self.deadline is not None and now >= self.deadline
        return not self.timed_out

    def _on_training_end(self):
        if self.timed_out:
            self.sampling_seconds += time.perf_counter()-self.started
        if not self.detailed:
            self.stream.write(json.dumps({"kind":"final", "total_steps":self.num_timesteps, "gradient_epochs":self.model._n_updates, "metrics":self.last_metrics})+"\n")
        self.stream.close()


def _actor_parameters(model):
    return {name: value.detach().cpu().clone() for name, value in model.policy.named_parameters()
        if 'policy_net' in name or 'action_net' in name}


def experiment(config):
    from .. import __version__
    return {**provenance(config), 'obs_version': OBS_VERSION, 'obs_size': OBS_SIZE,
        'arena_version': __version__, 'dependencies': {name: importlib.metadata.version(name) for name in
            ('torch', 'gymnasium', 'stable-baselines3', 'numpy')}}


def train(config, output, steps, num_envs=1, device='cpu', resume=None, wall_seconds=None):
    if wall_seconds is not None and (not math.isfinite(wall_seconds) or wall_seconds <= 0):
        raise ValueError('wall-seconds must be finite and positive')
    device_name(device)
    torch.set_num_threads(1)
    torch.manual_seed(config['seed'])
    torch.use_deterministic_algorithms(True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    metadata = experiment(config)
    if resume:
        previous = json.loads((Path(resume).parent/'experiment.json').read_text())
        for key in ('config', 'sources', 'sdk_hash', 'obs_version', 'obs_size', 'simulation'):
            if metadata[key] != previous[key]:
                raise ValueError(f'Resume incompatibility: {key}')
        if metadata['navigation']!=previous.get('navigation'):
            raise ValueError('Resume incompatibility: navigation')
        for key in ('runtime_hash', 'engine_binary_sha256', 'rules_sha256'):
            if previous.get(key, metadata[key]) != metadata[key]:
                raise ValueError(f'Resume incompatibility: {key}')
    (output/'experiment.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    started = time.perf_counter()
    runtime_config = dict(config)
    if wall_seconds is not None:
        runtime_config['_training_deadline'] = time.monotonic()+wall_seconds
    vec = vector_env(runtime_config, num_envs)
    callback = TraceCallback(output/'actions.jsonl', started+wall_seconds if wall_seconds is not None else None,
                             detailed=config.get('action_trace', False),backup_seconds=config.get('github_backup_seconds',300.))
    model = None
    if config.get('github_backup_repo'):
        from .backup import ReleaseBackup
        callback.backup = ReleaseBackup(output, config['github_backup_repo'])
    try:
        if resume:
            model = TimedPPO.load(resume, env=vec, device=device)
        else:
            model = TimedPPO('MlpPolicy', vec, seed=config['seed'], device=device, verbose=1,
                policy_kwargs={'net_arch': {'pi': [64, 64], 'vf': [64, 64]}}, **config['ppo'])
        model.learning_rate_control_path = str((output/'learning-rate.json').resolve())
        model.update_log_path = str((output/'updates.jsonl').resolve())
        block = model.n_steps*num_envs
        if steps is None:
            steps = block * 1000000000 if wall_seconds is not None else 256
        if steps <= 0 or steps % block:
            raise ValueError(f'Steps must be a positive multiple of rollout size {block}; no hidden overshoot')
        before = _actor_parameters(model)
        previous_steps, previous_updates = model.num_timesteps, model._n_updates
        model.optimization_seconds = 0.0
        model.learn(total_timesteps=steps, callback=callback, reset_num_timesteps=resume is None)
        after = _actor_parameters(model)
        delta = math.sqrt(sum(float(torch.sum((after[name]-value)**2)) for name, value in before.items()))
        if not math.isfinite(delta) or (delta <= 0 and wall_seconds is None):
            raise RuntimeError('Actor did not receive a finite, nonzero parameter update')
        save_checkpoint(model, output/'model.zip')
        # Check predictions and optimizer restoration from the file just saved.
        loaded = TimedPPO.load(output/'model.zip', device=device)
        check_obs = np.zeros((OBS_SIZE,), dtype=np.float32)
        assert np.array_equal(model.predict(check_obs, deterministic=True)[0], loaded.predict(check_obs, deterministic=True)[0])
        optimizer_restored = len(loaded.policy.optimizer.state_dict()['state']) > 0
        if model._n_updates > 0:
            assert optimizer_restored
        losses = {str(key): float(value) for key, value in model.logger.name_to_value.items()
                  if key.startswith('train/') and np.isscalar(value)}
        if not all(math.isfinite(value) for value in losses.values()):
            raise RuntimeError('Non-finite PPO training statistic')
        report = {'actual_steps': model.num_timesteps-previous_steps, 'total_steps': model.num_timesteps,
            'gradient_epochs': model._n_updates-previous_updates, 'actor_parameter_l2_change': delta,
            'reload_prediction_equal': True, 'optimizer_reloaded': optimizer_restored,
            'actor_updated': delta > 0, 'requested_wall_seconds': wall_seconds,
            'stop_reason': 'wall_time_limit' if callback.timed_out else 'step_limit',
            'num_envs': num_envs, 'device': str(model.device),
            'wall_seconds': time.perf_counter()-started, 'sampling_seconds': callback.sampling_seconds,
            'optimization_seconds': model.optimization_seconds,
            'rl_steps_per_second': (model.num_timesteps-previous_steps)/max(callback.sampling_seconds, 1e-9),
            'simulation_seconds_per_second': (model.num_timesteps-previous_steps)*config['rl_seconds']/max(callback.sampling_seconds, 1e-9),
            'losses': losses, 'model_sha256': hashlib.sha256((output/'model.zip').read_bytes()).hexdigest()}
        (output/'training.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        return report
    except OpeningDeadline:
        if model is None:
            raise
        save_checkpoint(model, output/'model.zip')
        report = {'stop_reason':'wall_time_limit_during_opening',
                  'total_steps':model.num_timesteps, 'actual_steps':model.num_timesteps-previous_steps,
                  'gradient_epochs':model._n_updates-previous_updates,
                  'wall_seconds':time.perf_counter()-started,
                  'requested_wall_seconds':wall_seconds, 'opening_incomplete':True}
        (output/'training.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        return report
    except KeyboardInterrupt:
        if model is not None:
            save_checkpoint(model, output/'interrupted.zip')
        raise
    finally:
        if callback.backup is not None:
            try:
                candidates = [output/name for name in ('model.zip','interrupted.zip','latest.zip') if (output/name).exists()]
                if candidates:
                    checkpoint = max(candidates,key=lambda p:p.stat().st_mtime_ns)
                    callback.backup.submit(checkpoint, {'total_steps':getattr(model,'num_timesteps',0), 'final':True})
            except Exception as error:
                print(json.dumps({'kind':'backup-error','error':type(error).__name__}),flush=True)
            finally:
                callback.backup.close()
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
    if current['navigation']!=metadata.get('navigation'):
        raise ValueError('Training runtime changed: navigation')
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
            config.update(team=side, episode_seconds=duration, sim_backend='rapier')
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
