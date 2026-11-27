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
from typing import NamedTuple

from gymnasium import spaces
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import FloatSchedule, explained_variance
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .env import SentryEnv, make_env
from .portable import OBS_SIZE, OBS_VERSION
# Re-exported: these validators are pure config arithmetic and live in a
# torch-free module so a shell preflight can run them without importing torch.
from .preflight import rollout_block, check_steps, check_action_std
from .session import provenance, OpeningDeadline, result_label


def device_name(requested):
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA explicitly requested, but torch.cuda.is_available() is false; use --device cpu')
    print(json.dumps({'device': requested, 'torch': torch.__version__, 'cuda_available': torch.cuda.is_available()}), flush=True)
    return requested


def vector_env(config, count, shards=1):
    if count < 1:
        raise ValueError('num-envs must be positive')
    if shards < 1:
        raise ValueError('sim-shards must be positive')
    base_config = dict(config)
    stagger = bool(base_config.pop('_stagger_initial_horizons', False))
    if (stagger and count > 1 and base_config.get('slot', 7) != 1
            and base_config.get('episode_seconds') and base_config.get('rl_seconds')):
        ep_s = float(base_config['episode_seconds'])
        rl_s = float(base_config['rl_seconds'])
        ep_steps = max(1, round(ep_s / rl_s))
        factories = []
        for i in range(count):
            h_steps = max(1, round(ep_steps * (i + 1) / count))
            env_cfg = {**base_config, '_initial_episode_seconds': round(h_steps * rl_s, 6)}
            factories.append(partial(make_env, env_cfg))
    else:
        factories = [partial(make_env, base_config) for _ in range(count)]
    if base_config.get('sim_backend', 'rapier') == 'mjwarp':
        from .mjwarp.vector import WarpVecEnv
        vec = WarpVecEnv(factories)
    elif base_config.get('sim_backend') == 'warp_kinematic' and count>1:
        effective_shards = min(shards, count)
        if effective_shards > 1:
            from .warp_kinematic.vector import ShardedWarpVecEnv
            vec = ShardedWarpVecEnv(factories, shards=effective_shards)
        else:
            from .warp_kinematic.vector import WarpVecEnv
            vec = WarpVecEnv(factories)
    else:
        vec = DummyVecEnv(factories) if count == 1 or base_config.get('slot') == 1 else SubprocVecEnv(factories, start_method='spawn')
    vec.seed(base_config['seed'])
    return vec


class MaskedRolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_values: torch.Tensor
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    policy_mask: torch.Tensor


class MaskedRolloutBuffer(RolloutBuffer):
    """RolloutBuffer carrying a per-step policy_mask (1.0=policy, 0.0=takeover)."""

    def reset(self) -> None:
        self.policy_masks = np.ones((self.buffer_size, self.n_envs), dtype=np.float32)
        self._pending_mask = None
        super().reset()

    def set_pending_mask(self, mask) -> None:
        self._pending_mask = np.asarray(mask, dtype=np.float32).reshape(self.n_envs)

    def add(self, obs, action, reward, episode_start, value, log_prob) -> None:
        if self._pending_mask is not None:
            self.policy_masks[self.pos] = self._pending_mask
            self._pending_mask = None
        else:
            self.policy_masks[self.pos] = 1.0
        super().add(obs, action, reward, episode_start, value, log_prob)

    def get(self, batch_size=None):
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if not self.generator_ready:
            for tensor in (
                'observations',
                'actions',
                'values',
                'log_probs',
                'advantages',
                'returns',
                'policy_masks',
            ):
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs
        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(self, batch_inds, env=None):
        data = (
            self.observations[batch_inds],
            self.actions[batch_inds].astype(np.float32, copy=False),
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.policy_masks[batch_inds].flatten(),
        )
        return MaskedRolloutBufferSamples(*tuple(map(self.to_torch, data)))


class _PolicyMaskCallback(BaseCallback):
    """Forwards ArenaSession's `info['policy_executed']` to MaskedRolloutBuffer."""

    def _on_step(self) -> bool:
        buf = getattr(self.model, 'rollout_buffer', None)
        if isinstance(buf, MaskedRolloutBuffer):
            infos = self.locals.get('infos') or ()
            mask = [float(bool(info.get('policy_executed', True))) for info in infos]
            if len(mask) == buf.n_envs:
                buf.set_pending_mask(mask)
        return True


class TanhActorCriticPolicy(ActorCriticPolicy):
    """Gaussian actor whose deterministic mean is bounded in (-1, 1)^2 via Tanh."""

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        if not isinstance(self.action_dist, DiagGaussianDistribution):
            raise NotImplementedError('TanhActorCriticPolicy requires DiagGaussianDistribution')
        self.action_net = nn.Sequential(self.action_net, nn.Tanh())


class TimedPPO(PPO):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('rollout_buffer_class', MaskedRolloutBuffer)
        super().__init__(*args, **kwargs)

    def learn(self, *args, callback=None, **kwargs):
        mask_cb = _PolicyMaskCallback()
        if callback is None:
            wrapped = mask_cb
        elif isinstance(callback, list):
            wrapped = [mask_cb, *callback]
        else:
            wrapped = CallbackList([mask_cb, callback])
        return super().learn(*args, callback=wrapped, **kwargs)

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

    def apply_action_std_control(self):
        """Pull the exploration width back to the action box on operator request.

        `check_action_std` guards the configured value, but the width is a
        learned parameter: the 22h run started at the 1.0 default and the very
        first update took it to 1.969, ending at 2.219 - so every one of its
        440 updates sampled with a std larger than the [-1, 1] box and roughly
        half of every sample was clipped. The gradient then described actions
        the environment never executed, and nothing in `losses` showed it.

        This is a manual channel, not an automatic clamp, for the same reason
        the learning-rate one is: silently correcting a policy that is drifting
        hides the drift. Between two requests the curve stays honest.
        """
        path = getattr(self, 'action_std_control_path', None)
        if not path or not Path(path).exists():
            return
        try:
            request = json.loads(Path(path).read_text())
            bound = float(request['max_std'])
            if not math.isfinite(bound) or not 0 < bound <= 1.0:
                raise ValueError('max_std must be finite and in (0, 1]')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(json.dumps({'kind': 'action-std-control-error', 'error': str(exc)}), flush=True)
            return
        with torch.no_grad():
            before_vec = self.policy.log_std.data.detach().clone()
            before = float(torch.exp(before_vec).mean())
            before_max = float(torch.exp(before_vec).max())
            # log_std is the parameter; the bound is in std space, so the clamp
            # has to happen on log(bound) or it would clamp ~0.0 to ~0.0.
            torch.clamp_(self.policy.log_std.data, max=math.log(bound))
            after_vec = self.policy.log_std.data.detach()
            after = float(torch.exp(after_vec).mean())
            after_max = float(torch.exp(after_vec).max())
        if torch.equal(after_vec, before_vec):
            return
        event = {'kind': 'action-std-clamp', 'total_samples': self.num_timesteps,
                 'before': before, 'after': after,
                 'before_max': before_max, 'after_max': after_max, 'max_std': bound,
                 'reason': str(request.get('reason', 'operator request')), 'time_unix': time.time()}
        with Path(path).with_name('action-std-events.jsonl').open('a') as stream:
            stream.write(json.dumps(event)+'\n')
        print(json.dumps(event), flush=True)

    def _train_masked(self):
        """PPO update masking Actor PG/entropy/KL over policy_executed steps while training Critic on all steps."""
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = self.clip_range_vf(self._current_progress_remaining) if self.clip_range_vf is not None else None

        entropy_losses, pg_losses, value_losses, clip_fractions = [], [], [], []
        approx_kl_divs = []
        loss = torch.zeros((), device=self.device)
        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()

                mask = getattr(rollout_data, 'policy_mask', None)
                if mask is None:
                    mask = torch.ones_like(rollout_data.advantages)
                mask = mask.float()
                mask_sum = mask.sum()
                mask_count = float(mask_sum.item())

                advantages = rollout_data.advantages
                if self.normalize_advantage and mask_count > 1.0:
                    adv_mean = (advantages * mask).sum() / mask_sum
                    adv_var = (((advantages - adv_mean) ** 2) * mask).sum() / (mask_sum - 1.0)
                    advantages = (advantages - adv_mean) / (torch.sqrt(adv_var.clamp_min(0.0)) + 1e-8)

                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                if mask_count > 0.0:
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -(torch.min(policy_loss_1, policy_loss_2) * mask).sum() / mask_sum
                    pg_losses.append(policy_loss.item())
                    clip_fraction = (((torch.abs(ratio - 1) > clip_range).float() * mask).sum() / mask_sum).item()
                    clip_fractions.append(clip_fraction)
                    if entropy is None:
                        entropy_loss = -((-log_prob) * mask).sum() / mask_sum
                    else:
                        entropy_loss = -(entropy * mask).sum() / mask_sum
                    entropy_losses.append(entropy_loss.item())
                    with torch.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl_div = ((((torch.exp(log_ratio) - 1) - log_ratio) * mask).sum() / mask_sum).cpu().numpy()
                        approx_kl_divs.append(approx_kl_div)
                else:
                    policy_loss = torch.zeros((), device=values.device)
                    entropy_loss = torch.zeros((), device=values.device)
                    approx_kl_div = np.array(0.0, dtype=np.float32)

                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                if mask_count > 0.0 and self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f'Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}')
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        mask_frac = (
            float(np.mean(self.rollout_buffer.policy_masks))
            if hasattr(self.rollout_buffer, 'policy_masks') else 1.0
        )
        self.logger.record('train/entropy_loss', float(np.mean(entropy_losses)) if entropy_losses else 0.0)
        self.logger.record('train/policy_gradient_loss', float(np.mean(pg_losses)) if pg_losses else 0.0)
        self.logger.record('train/value_loss', float(np.mean(value_losses)) if value_losses else 0.0)
        self.logger.record('train/approx_kl', float(np.mean(approx_kl_divs)) if approx_kl_divs else 0.0)
        self.logger.record('train/clip_fraction', float(np.mean(clip_fractions)) if clip_fractions else 0.0)
        self.logger.record('train/policy_mask_fraction', mask_frac)
        self.logger.record('train/loss', float(loss.item()))
        self.logger.record('train/explained_variance', explained_var)
        if hasattr(self.policy, 'log_std'):
            self.logger.record('train/std', torch.exp(self.policy.log_std).mean().item())
        self.logger.record('train/n_updates', self._n_updates, exclude='tensorboard')
        self.logger.record('train/clip_range', clip_range)
        if clip_range_vf is not None:
            self.logger.record('train/clip_range_vf', clip_range_vf)

    def train(self):
        self.apply_learning_rate_control()
        self.apply_action_std_control()
        started = time.perf_counter()
        self._train_masked()
        seconds = time.perf_counter()-started
        self.optimization_seconds = getattr(self, 'optimization_seconds', 0.0)+seconds
        path = getattr(self, 'update_log_path', None)
        if path:
            row = {'kind': 'ppo-update', 'total_samples': self.num_timesteps,
                   'gradient_epochs': self._n_updates, 'optimization_seconds': seconds,
                   'learning_rate': float(self.lr_schedule(self._current_progress_remaining)),
                   'action_std': action_std(self),
                   'behavior': getattr(self, 'last_rollout_behavior', {}),
                   'losses': {str(k): float(v) for k, v in self.logger.name_to_value.items()
                              if k.startswith('train/') and np.isscalar(v)}}
            with Path(path).open('a') as stream:
                stream.write(json.dumps(row)+'\n')


def action_std(model):
    """The policy's per-dimension exploration scale, or None if it has none.

    Logged because it is the one PPO state that can quietly invalidate an
    action bound: the policy samples N(mean, std) and the env clips the result,
    so a std that drifts up sends half of every sample into the clip. The
    gradient then points at samples the environment never actually executed,
    and nothing in `losses` shows it - the loss gets *smaller* as the policy
    stops being able to tell clipping from a real preference. It is read from
    the state-independent log_std so the value is comparable across updates.
    """
    log_std = getattr(model.policy, 'log_std', None)
    if log_std is None:
        return None
    return float(torch.exp(log_std.detach().float().mean()))


def _json_default(value):
    """Make observations serialisable without losing their structure.

    The per-step infos carry numpy arrays, numpy scalars and occasionally
    torch scalars. json.dumps has no default encoder for any of them, so a run
    configured with action_trace=false crashed the moment the summary line was
    written - after the model had already been training for minutes.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


class TraceCallback(BaseCallback):
    def __init__(self, path, deadline=None, detailed=True, backup_seconds=300.):
        super().__init__()
        self.detailed = detailed
        self.path = Path(path)
        self.sampling_seconds = 0.0
        # The first rollout is not sampling: 16 envs, warp's lazy compilation
        # and the navigation mesh all happen inside it. On 1988 rollouts that
        # is noise; on one it is the whole denominator, which is how a 25
        # steps/s run reports 12. The legacy `sampling_seconds` stays exactly
        # as defined so a long run is still comparable with the 22h baseline,
        # and the split below is what the steady rate is read from.
        self.init_seconds = 0.0
        self.sampling_seconds_excluding_first = 0.0
        self.rollouts = 0
        self.saw_rollout_end = False
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
        self.saw_rollout_end = False
        self.rollout_behavior = {'samples': 0, 'moving_commands': 0, 'modes': {}}

    def _on_rollout_end(self):
        seconds = time.perf_counter()-self.started
        self.sampling_seconds += seconds
        if not self.rollouts:
            self.init_seconds = seconds
        else:
            self.sampling_seconds_excluding_first += seconds
        self.rollouts += 1
        self.saw_rollout_end = True
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
                                                       'reward_totals', 'control', 'bases', 'opponent',
                                                       'witness')}}
                with (self.path.parent/'episodes.jsonl').open('a') as episodes:
                    episodes.write(json.dumps(row, default=_json_default)+'\n')
        for index, info in enumerate(self.locals['infos'] if self.detailed else []):
            # SB3 stores self.locals['actions'] and its original log probability
            # in RolloutBuffer. Do not replace them with clipped/snapped goals.
            self.stream.write(json.dumps({'env': index, 'total_steps': self.num_timesteps,
                'sampled_action': self.locals['actions'][index].tolist(),
                'environment_action': self.locals['clipped_actions'][index].tolist(),
                **info}, allow_nan=False, default=_json_default)+'\n')
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
                    'gradient_epochs': self.model._n_updates, 'metrics': self.last_metrics},
                    default=_json_default)+'\n')
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
            # A deadline that lands mid-rollout leaves the last block unclosed.
            # It is still sampled wall time, so it belongs in the same bucket a
            # closed rollout would have gone to - otherwise a --wall-seconds
            # that happens to cut the first rollout short reports init as zero.
            seconds = time.perf_counter()-self.started
            self.sampling_seconds += seconds
            if not self.rollouts:
                self.init_seconds += seconds
            else:
                self.sampling_seconds_excluding_first += seconds
        if not self.detailed:
            self.stream.write(json.dumps({"kind":"final", "total_steps":self.num_timesteps, "gradient_epochs":self.model._n_updates, "metrics":self.last_metrics}, default=_json_default)+"\n")
        self.stream.close()


def _actor_parameters(model):
    return {name: value.detach().cpu().clone() for name, value in model.policy.named_parameters()
        if 'policy_net' in name or 'action_net' in name}


def _ppo_kwargs(config):
    """`config['ppo']` minus the keys that are checked here, not by SB3.

    `action_std` is ours: SB3 has no such argument and would raise TypeError
    the moment someone set it in a config file, so it is stripped after
    `check_action_std` has validated it.
    """
    return {key: value for key, value in config['ppo'].items() if key != 'action_std'}


def experiment(config):
    from .. import __version__
    return {**provenance(config), 'obs_version': OBS_VERSION, 'obs_size': OBS_SIZE,
        'arena_version': __version__, 'dependencies': {name: importlib.metadata.version(name) for name in
            ('torch', 'gymnasium', 'stable-baselines3', 'numpy')}}


def train(config, output, steps, num_envs=1, device='cpu', resume=None, wall_seconds=None, init_from=None, sim_shards=1):
    if resume and init_from:
        raise ValueError('--resume and --init-from are mutually exclusive')
    if sim_shards < 1:
        raise ValueError('sim-shards must be positive')
    if wall_seconds is not None and (not math.isfinite(wall_seconds) or wall_seconds <= 0):
        raise ValueError('wall-seconds must be finite and positive')
    check_action_std(config)
    # Rollout arithmetic is pure config, so it is checked here rather than
    # after the environment exists: a mismatch used to be reported only once
    # N operators had spawned, and the traceback pointed at the step loop
    # instead of at the flag that caused it.
    check_steps(config, num_envs, steps)
    device_name(device)
    torch.set_num_threads(1)
    torch.manual_seed(config['seed'])
    torch.use_deterministic_algorithms(True)
    output = Path(output)
    # A resumed experiment owns its directory already: experiment.json, the
    # logs and the checkpoint being resumed all live there, so the run must
    # reopen it rather than create a fresh one. init_from deliberately does not
    # get this treatment - starting over in a new directory is what keeps the
    # previous run's artifacts read-only and its counters untouched.
    output.mkdir(parents=True, exist_ok=bool(resume))
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
        # Keep the original provenance when it matches: rewriting it would make
        # the resumed run look like a new one to evaluate/compose comparisons.
        metadata = previous
    (output/'experiment.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    started = time.perf_counter()
    runtime_config = dict(config)
    if wall_seconds is not None:
        runtime_config['_training_deadline'] = time.monotonic()+wall_seconds
    if num_envs > 1:
        runtime_config['_stagger_initial_horizons'] = True
    vec = vector_env(runtime_config, num_envs) if sim_shards == 1 else vector_env(runtime_config, num_envs, shards=sim_shards)
    callback = TraceCallback(output/'actions.jsonl', started+wall_seconds if wall_seconds is not None else None,
                             detailed=config.get('action_trace', False),backup_seconds=config.get('github_backup_seconds',300.))
    # Everything expensive that is not a rollout: spawning the operators,
    # warping, the mesh, and building the policy. Measured separately so that
    # `sampling_seconds` is never blamed for it on a short run.
    setup_seconds = time.perf_counter()-started
    model = None
    if config.get('github_backup_repo'):
        from .backup import ReleaseBackup
        callback.backup = ReleaseBackup(output, config['github_backup_repo'])
    try:
        if resume:
            model = TimedPPO.load(resume, env=vec, device=device)
        elif init_from:
            # Seed a fresh experiment from existing weights while applying the
            # new config's PPO hyperparameters (n_steps, batch_size, lr, ent_coef...).
            # If the reward definition changed vs the source experiment, reset the
            # Critic and Adam state so stale value scales do not corrupt updates.
            init_custom = {**_ppo_kwargs(config), 'rollout_buffer_class': MaskedRolloutBuffer}
            if 'learning_rate' in init_custom:
                init_custom['lr_schedule'] = FloatSchedule(init_custom['learning_rate'])
            model = TimedPPO.load(init_from, env=vec, device=device, custom_objects=init_custom)
            prev_exp_path = Path(init_from).parent / 'experiment.json'
            if prev_exp_path.is_file():
                prev_rewards = json.loads(prev_exp_path.read_text()).get('config', {}).get('rewards')
                if prev_rewards is not None and prev_rewards != config['rewards']:
                    for module in model.policy.mlp_extractor.value_net.modules():
                        if isinstance(module, nn.Linear):
                            nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
                            nn.init.constant_(module.bias, 0.0)
                    nn.init.orthogonal_(model.policy.value_net.weight, gain=1.0)
                    nn.init.constant_(model.policy.value_net.bias, 0.0)
                    model.policy.optimizer.state.clear()
            model.num_timesteps = 0
            model._total_timesteps = 0
            model._num_timesteps_at_start = 0
            model._n_updates = 0
            model._episode_num = 0
            model.ep_info_buffer.clear()
            model.ep_success_buffer.clear()
            model._current_progress_remaining = 1.0
            # Force learn() to reset the env instead of replaying a stale obs.
            model._last_obs = None
            model._last_episode_starts = None
        else:
            init_std = check_action_std(config)
            model = TimedPPO(
                TanhActorCriticPolicy,
                vec,
                seed=config['seed'],
                device=device,
                verbose=1,
                policy_kwargs={
                    'net_arch': {'pi': [64, 64], 'vf': [64, 64]},
                    'log_std_init': math.log(init_std),
                },
                **_ppo_kwargs(config),
            )
        model.learning_rate_control_path = str((output/'learning-rate.json').resolve())
        model.action_std_control_path = str((output/'action-std.json').resolve())
        model.update_log_path = str((output/'updates.jsonl').resolve())
        block = model.n_steps*num_envs
        if steps is None:
            steps = block * 1000000000 if wall_seconds is not None else 256
        # A configured `n_steps` that disagrees with the env count would make the
        # startup check above describe the wrong block, so restate the invariant
        # against what the model actually holds. A None `steps` is synthesised
        # here and can therefore be checked only now.
        if steps <= 0 or steps % block:
            raise ValueError(f'Steps must be a positive multiple of rollout size {block}; no hidden overshoot')
        before = _actor_parameters(model)
        callback.block = block
        previous_steps, previous_updates = model.num_timesteps, model._n_updates
        model.optimization_seconds = 0.0
        # Starting exploration width. A run that begins at the bound is already
        # clipping every other sample, so it is reported rather than assumed
        # away; the per-update trace in updates.jsonl shows where it drifts.
        start_std = action_std(model)
        model.learn(total_timesteps=steps, callback=callback, reset_num_timesteps=resume is None)
        end_std = action_std(model)
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
            'num_envs': num_envs, 'sim_shards': getattr(vec, 'num_shards', 1), 'device': str(model.device),
            'wall_seconds': time.perf_counter()-started, 'sampling_seconds': callback.sampling_seconds,
            'optimization_seconds': model.optimization_seconds,
            'rl_steps_per_second': (model.num_timesteps-previous_steps)/max(callback.sampling_seconds, 1e-9),
            'simulation_seconds_per_second': (model.num_timesteps-previous_steps)*config['rl_seconds']/max(callback.sampling_seconds, 1e-9),
            # Same run, denominator with the first rollout taken out. `None`
            # rather than a number when there is no second rollout: one
            # rollout has no steady state to report, and a fabricated rate is
            # what produced the 12.57 steps/s that looked like a regression.
            'setup_seconds': setup_seconds, 'init_seconds': callback.init_seconds,
            'sampling_seconds_excluding_first': callback.sampling_seconds_excluding_first,
            'rl_rollouts': callback.rollouts,
            'rl_steps_per_second_excluding_first': (
                (model.num_timesteps-previous_steps-block)/callback.sampling_seconds_excluding_first
                if callback.rollouts > 1 and callback.sampling_seconds_excluding_first > 0 else None),
            'losses': losses, 'model_sha256': hashlib.sha256((output/'model.zip').read_bytes()).hexdigest(),
            'action_std': {'start': start_std, 'end': end_std,
                           'bound': 1.0, 'clipping': end_std is not None and end_std >= 1.0}}
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


def evaluate(model_path, output, seeds, sides, duration=420, device='cpu', backend=None):
    device_name(device)
    torch.set_num_threads(1)
    model_path, output = Path(model_path), Path(output)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise ValueError(f'evaluate output directory already exists: {output}; '
                         'pick another --run-dir or move the old one aside') from None
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
            # The reported outcome has to come from the same simulator the run
            # trained on. Pinning 'rapier' here once meant an evaluation could be
            # scored on a backend the policy had never seen; --backend still wins
            # when the caller asks for a specific one.
            config.update(team=side, episode_seconds=duration, sim_backend=backend or config['sim_backend'])
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
                outcome = result_label(info['winner'], side, terminated)
                replay = output/f'{seed}-{side}.json'
                env.session.save_replay(replay)
                if config['sim_backend'] == 'rapier':
                    from rmuc2027_core.replay import verify_replay
                    verified = verify_replay(replay)
                else:
                    verified = {'skipped': True, 'backend': config['sim_backend']}
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
