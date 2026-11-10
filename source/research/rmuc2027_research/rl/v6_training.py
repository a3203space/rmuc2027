"""Exact v4 -> v6 actor/Adam migration and complete resumable research state.

Migration introduces an explicit, documented revival prior, not demonstrations.
The old on-policy rollout is never reused with the new observation/action law.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

from . import sentry_v4
from .asymmetric_ppo import AsymmetricPPO
from .sentry_v6_policy import SentryV6Policy, TACTICAL_AXES, REVIVE_PRIOR
from .v6_ppo import HybridPPO


def configuration(seed=20261111):
    from .v4_training import configuration as previous_configuration
    cfg = previous_configuration('main', seed)
    cfg.update(policy_version=6, episode_seconds=120, v4_capture_reward=.02)
    cfg['ppo'].update(learning_rate=1e-5, target_kl=.006, ent_coef=.001,
        tactical_lr=1e-5, revive_lr=3e-4, critic_lr=5e-5,
        tactical_target_kl=.006, near_roof_target_kl=.003, revive_target_kl=.01)
    return cfg


def _expanded(value, target_shape, column_map):
    if tuple(value.shape) == tuple(target_shape):
        return value.detach().clone()
    if value.ndim == 2 and value.shape[1] == 739 and tuple(target_shape) == (value.shape[0], 775):
        result = value.new_zeros(target_shape)
        result[:, column_map] = value
        return result
    raise ValueError(f'Unsupported migration shape {tuple(value.shape)} -> {tuple(target_shape)}')


def _copy_optimizer(source, target, source_policy, target_policy, column_map, *, tactical=False):
    source_names = {id(parameter): name for name, parameter in source_policy.named_parameters()}
    source_states = {source_names[id(parameter)]: state for parameter, state in source.state.items()}
    target_names = {id(parameter): name for name, parameter in target_policy.named_parameters()}
    copied = []
    for group in target.param_groups:
        for parameter in group['params']:
            name = target_names[id(parameter)]
            source_name = 'log_std' if name == 'tactical_log_std' else name
            if source_name not in source_states:
                continue  # An untouched source parameter legitimately has no Adam state.
            new_state = {}
            for key, value in source_states[source_name].items():
                if not torch.is_tensor(value):
                    new_state[key] = copy.deepcopy(value)
                elif value.ndim == 0:
                    new_state[key] = value.detach().clone().to(parameter.device)
                elif name == 'tactical_log_std':
                    new_state[key] = value[list(TACTICAL_AXES)].detach().clone().to(parameter.device)
                else:
                    new_state[key] = _expanded(value, parameter.shape, column_map).to(parameter.device)
            target.state[parameter] = new_state
            copied.append(name)
    # Preserve optimizer dynamics (betas/eps/etc), while honoring the newly
    # declared branch learning rates and the new, disjoint parameter groups.
    if len(source.param_groups) != 1 or len(target.param_groups) != 1:
        raise ValueError('Expected a single named-parameter group per branch')
    for key, value in source.param_groups[0].items():
        if key not in ('params', 'lr', 'initial_lr'):
            target.param_groups[0][key] = copy.deepcopy(value)
    return copied


def migrate_from_v4(source_zip, env, config=None, *, resume_folder=None, device='cpu'):
    from . import sentry_v6
    source_zip = Path(source_zip)
    config = copy.deepcopy(config if config is not None else configuration())
    old = AsymmetricPPO.load(source_zip, device=device)
    old_names = [item[0] for item in sentry_v4.FEATURES]
    new_names = [item[0] for item in sentry_v6.FEATURES]
    if len(set(new_names)) != 775 or set(old_names) - set(new_names):
        raise ValueError('v6 registry must uniquely preserve every v4 feature')
    column_map = [new_names.index(name) for name in old_names]
    if column_map != list(range(739)) or tuple(sentry_v6.SKILL_INDICES) != tuple(sentry_v4.SKILL_INDICES):
        raise ValueError('v6 migration requires the unchanged v4 prefix and skill37 contract')
    kwargs = dict(config.get('ppo', {}))
    kwargs.setdefault('max_grad_norm', .5)
    model = HybridPPO(SentryV6Policy, env, seed=config.get('seed', 20261111), device=device,
                      verbose=0, **kwargs)
    state = model.policy.state_dict()
    migrated = []
    for name, value in old.policy.state_dict().items():
        if name == 'log_std':
            state['tactical_log_std'] = value[list(TACTICAL_AXES)].clone()
            state['skill_log_std'] = value[2:4].clone()
        elif name in state:
            state[name] = _expanded(value, state[name].shape, column_map)
            migrated.append(name)
        else:
            raise ValueError(f'Unrecognized source policy parameter/buffer {name}')
    model.policy.load_state_dict(state, strict=True)
    tactical = _copy_optimizer(old.actor_optimizer, model.tactical_optimizer, old.policy,
                               model.policy, column_map, tactical=True)
    critic = _copy_optimizer(old.critic_optimizer, model.critic_optimizer, old.policy,
                             model.policy, column_map)
    counters = ('num_timesteps', '_n_updates', 'optimizer_steps', 'optimized_samples',
        'on_policy_rollout_samples', 'completed_updates', 'last_completed_samples',
        '_num_timesteps_at_start', '_total_timesteps', '_current_progress_remaining')
    for name in counters:
        if hasattr(old, name):
            setattr(model, name, copy.deepcopy(getattr(old, name)))
    model.tactical_optimizer_steps = getattr(old, 'optimizer_steps', 0)
    model.critic_optimizer_steps = getattr(old, 'optimizer_steps', 0)
    model.revive_optimizer_steps = 0
    model.rollout_buffer.reset()
    model._last_obs = None
    model._last_episode_starts = None
    model.migration_source_sha256 = hashlib.sha256(source_zip.read_bytes()).hexdigest()
    report = dict(version=6, source_sha256=model.migration_source_sha256,
        source_schema_hash=sentry_v4.SCHEMA_HASH, schema_hash=sentry_v6.SCHEMA_HASH,
        old_actor_size=739, actor_size=775, critic_size=454, action_size=7,
        copied_state=migrated, tactical_adam_copied=tactical, critic_adam_copied=critic,
        new_columns=list(range(739, 775)), new_columns_and_moments_zero=True,
        frozen_skill=True, frozen_skill_std_axes=[2, 3], revive_prior=list(REVIVE_PRIOR),
        revive_adam='fresh', rollout_reused=False,
        parity_scope='alive six-dimensional Gaussian; confirmed-dead execution and revival explicitly change',
        source_num_timesteps=old.num_timesteps)
    if resume_folder is not None:
        restored = restore_state(resume_folder, model.get_env())
        report['resume'] = restored['sampler_restore']
        report['rng_restored'] = bool(restored.get('rng_restored'))
    else:
        report['rng_restored'] = False
    return model, report


def save_state(model, vec, folder, name='model.zip'):
    from .v4_training import save_state as old_save_state
    old_save_state(model, vec, folder, name)
    folder = Path(folder)
    state = torch.load(folder / 'resume-state.pt', map_location='cpu', weights_only=False)
    state.update(policy_version=6, optimizers=['tactical_optimizer', 'revive_optimizer', 'critic_optimizer'],
                 frozen_skill=True, frozen_skill_std_axes=[2, 3], rollout_reused=False,
                 gradient_epochs=model._n_updates, optimizer_steps=getattr(model, 'optimizer_steps', 0),
                 tactical_optimizer_steps=getattr(model, 'tactical_optimizer_steps', 0),
                 revive_optimizer_steps=getattr(model, 'revive_optimizer_steps', 0),
                 critic_optimizer_steps=getattr(model, 'critic_optimizer_steps', 0))
    pending = folder / 'resume-state.v6.tmp'
    torch.save(state, pending)
    os.replace(pending, folder / 'resume-state.pt')


def restore_state(folder, vec=None):
    """Read source state without mutating the source checkpoint or its reports."""
    from .v4_training import plan_sampler_restore
    folder = Path(folder)
    path = folder / 'resume-state.pt'
    state = torch.load(path, map_location='cpu', weights_only=False) if path.exists() else {}
    raw = (folder / 'samplers.json').read_bytes() if (folder / 'samplers.json').exists() else b'[]'
    samplers, report = plan_sampler_restore(json.loads(raw), vec.num_envs if vec else 1,
        num_timesteps=state.get('num_timesteps', 0), source_sha256=hashlib.sha256(raw).hexdigest())
    if vec is not None:
        for index, item in enumerate(samplers):
            vec.env_method('restore_sampler_state', item, indices=index)
        if samplers:
            vec._seeds = [None] * vec.num_envs
    if state:
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch_cpu'])
        cuda = state.get('torch_cuda', [])
        if cuda and torch.cuda.is_available() and len(cuda) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda)
    state['rng_restored'] = path.exists()
    state['sampler_restore'] = report
    return state
