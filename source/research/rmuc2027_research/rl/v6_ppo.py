"""Hybrid on-policy PPO with independent branch optimizers and exact KL guards."""
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance

from .sentry_v4 import FEATURE_INDEX
from .sentry_v6_policy import TACTICAL_AXES


def near_roof(actor):
    """A local geometric window; a distant planned passage alone is insufficient."""
    at = lambda name: actor[..., FEATURE_INDEX[name]]
    return ((at('roof.valid') > .5) &
        (at('roof.along').abs() * 5 <= at('roof.half_length') * 5 + 1.5) &
        (at('roof.across').abs() * 2 <= at('roof.half_width') * 2 + .6))


class HybridPPO(PPO):
    def __init__(self, *args, tactical_lr=1e-5, revive_lr=3e-4, critic_lr=5e-5,
                 tactical_target_kl=.006, near_roof_target_kl=.003, revive_target_kl=.01,
                 **kwargs):
        self.tactical_lr, self.revive_lr, self.critic_lr = tactical_lr, revive_lr, critic_lr
        self.tactical_target_kl = tactical_target_kl
        self.near_roof_target_kl = near_roof_target_kl
        self.revive_target_kl = revive_target_kl
        super().__init__(*args, **kwargs)

    def _setup_model(self):
        super()._setup_model()
        groups = [list(self.policy.tactical_parameters()), list(self.policy.revive_parameters()),
                  list(self.policy.critic_parameters())]
        ids = [set(map(id, group)) for group in groups]
        if any(ids[i] & ids[j] for i in range(3) for j in range(i)):
            raise ValueError('v6 optimizer parameter groups must be disjoint')
        if any(not p.requires_grad for group in groups for p in group):
            raise ValueError('Frozen skills must be excluded from every optimizer')
        rates = (self.tactical_lr, self.revive_lr, self.critic_lr)
        if not all(math.isfinite(rate) and rate > 0 for rate in rates):
            raise ValueError('Optimizer learning rates must be finite and positive')
        if not all(math.isfinite(value) and value > 0 for value in
                   (self.tactical_target_kl, self.near_roof_target_kl, self.revive_target_kl)):
            raise ValueError('KL guard thresholds must be finite and positive')
        self.tactical_optimizer, self.revive_optimizer, self.critic_optimizer = [
            torch.optim.Adam(group, lr=rate, eps=1e-5) for group, rate in zip(groups, rates)]

    def _get_torch_save_params(self):
        return ['policy', 'tactical_optimizer', 'revive_optimizer', 'critic_optimizer'], []

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ['update_log_path', 'completed_update_hook', 'training_deadline']

    @torch.no_grad()
    def exact_kl(self, reference, observations):
        old = reference.get_distribution(observations)
        new = self.policy.get_distribution(observations)
        gaussian = torch.distributions.kl_divergence(old.normal, new.normal)[..., list(TACTICAL_AXES)].sum(-1)
        eligible = old.dead & (old.mask.sum(-1) > 1)
        categorical = torch.distributions.kl_divergence(old.categorical, new.categorical)
        alive = ~old.dead
        nearby = near_roof(observations['actor']) & alive
        def mean(values, mask):
            return float(values[mask].mean()) if mask.any() else 0.
        return dict(tactical_kl=mean(gaussian, alive), near_roof_kl=mean(gaussian, nearby),
                    revive_kl=mean(categorical, eligible), alive_samples=int(alive.sum()),
                    near_roof_samples=int(nearby.sum()), revive_samples=int(eligible.sum()))

    def _kl_exceeded(self, metrics):
        return any(not math.isfinite(metrics[key]) or metrics[key] > 1.5 * threshold
            for key, threshold in (('tactical_kl', self.tactical_target_kl),
                                  ('near_roof_kl', self.near_roof_target_kl),
                                  ('revive_kl', self.revive_target_kl)))

    def actor_step(self, loss, reference, guard_observations, step_observations=None):
        """One atomic T+revive step. A failed guard restores parameters AND Adam."""
        groups = (list(self.policy.tactical_parameters()), list(self.policy.revive_parameters()))
        optimizers = (self.tactical_optimizer, self.revive_optimizer)
        saved_parameters = [[p.detach().clone() for p in group] for group in groups]
        saved_optimizers = [copy.deepcopy(optimizer.state_dict()) for optimizer in optimizers]
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norms = [float(torch.nn.utils.clip_grad_norm_(group, self.max_grad_norm)) for group in groups]
        mask = (step_observations if step_observations is not None else guard_observations)['revive_mask']
        active = (bool((mask[:, 1] == 0).any()), bool(((mask[:, 1] > 0) & (mask.sum(-1) > 1)).any()))
        for optimizer, enabled in zip(optimizers, active):
            # A zero gradient alone does not stop old Adam momentum. An absent
            # branch receives no parameter OR optimizer-state update.
            if enabled:
                optimizer.step()
        try:
            metrics = self.exact_kl(reference, guard_observations)
        except ValueError:
            # A numerical proposal may fail distribution validation before a
            # finite KL can be computed; it still must roll back atomically.
            metrics = dict(tactical_kl=math.inf, near_roof_kl=math.inf, revive_kl=math.inf,
                           alive_samples=0, near_roof_samples=0, revive_samples=0)
        rejected = self._kl_exceeded(metrics) or not all(math.isfinite(norm) for norm in norms)
        if rejected:
            with torch.no_grad():
                for group, saved in zip(groups, saved_parameters):
                    for parameter, value in zip(group, saved):
                        parameter.copy_(value)
            for optimizer, state in zip(optimizers, saved_optimizers):
                optimizer.load_state_dict(state)
                optimizer.zero_grad(set_to_none=True)
        metrics.update(actor_step_accepted=not rejected, tactical_gradient_norm=norms[0], revive_gradient_norm=norms[1],
                       tactical_step=active[0] and not rejected, revive_step=active[1] and not rejected)
        return metrics

    def train(self):
        if getattr(self, 'sentry_guidance_path', None) or getattr(self, 'skill_guidance_datasets', None):
            raise ValueError('v6 PPO cannot contain hidden demonstration gradients')
        self.policy.set_training_mode(True)
        for optimizer, rate in ((self.tactical_optimizer, self.tactical_lr),
                                (self.revive_optimizer, self.revive_lr), (self.critic_optimizer, self.critic_lr)):
            for group in optimizer.param_groups:
                group['lr'] = rate
        buffer = self.rollout_buffer
        observations = {key: torch.as_tensor(value if buffer.generator_ready else buffer.swap_and_flatten(value),
                         device=self.device) for key, value in buffer.observations.items()}
        # A fixed old-policy reference for the complete current rollout, not a
        # moving minibatch reference. Masks stay inside the stored observations.
        reference = copy.deepcopy(self.policy).eval()
        reference.requires_grad_(False)
        clip = self.clip_range(self._current_progress_remaining)
        vf_clip = self.clip_range_vf(self._current_progress_remaining) if self.clip_range_vf else None
        metrics = {key: [] for key in ('policy_gradient_loss', 'value_loss', 'entropy_loss', 'approx_kl',
                                       'clip_fraction', 'tactical_kl', 'near_roof_kl', 'revive_kl')}
        started = time.monotonic()
        optimizer_steps = tactical_steps = revive_steps = critic_steps = sample_visits = rejected_steps = 0
        stop = False
        for epoch in range(self.n_epochs):
            epoch_used = False
            for data in buffer.get(self.batch_size):
                if getattr(self, 'training_deadline', math.inf) <= time.monotonic():
                    stop = True
                    break
                values, log_prob, entropy = self.policy.evaluate_actions(data.observations, data.actions)
                values = values.flatten()
                advantages = data.advantages.detach()
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                log_ratio = log_prob - data.old_log_prob
                ratio = log_ratio.exp()
                policy_loss = -torch.min(advantages * ratio, advantages * ratio.clamp(1 - clip, 1 + clip)).mean()
                entropy_loss = -entropy.mean()
                with torch.no_grad():
                    approx_kl = float(((ratio - 1) - log_ratio).mean())
                    fraction = float((torch.abs(ratio - 1) > clip).float().mean())
                predicted = values if vf_clip is None else data.old_values + (values - data.old_values).clamp(-vf_clip, vf_clip)
                value_loss = F.mse_loss(data.returns.detach(), predicted)
                step = self.actor_step(policy_loss + self.ent_coef * entropy_loss, reference, observations,
                                       data.observations)
                # Privileged value gradients never enter actor clipping or its KL guard.
                self.critic_optimizer.zero_grad(set_to_none=True)
                (self.vf_coef * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(list(self.policy.critic_parameters()), self.max_grad_norm)
                self.critic_optimizer.step()
                critic_steps += 1
                epoch_used = True
                tactical_steps += int(step['tactical_step'])
                revive_steps += int(step['revive_step'])
                if step['actor_step_accepted']:
                    optimizer_steps += 1
                    sample_visits += len(data.actions)
                else:
                    rejected_steps += 1
                    stop = True
                row = dict(policy_gradient_loss=policy_loss.item(), value_loss=value_loss.item(),
                    entropy_loss=entropy_loss.item(), approx_kl=approx_kl, clip_fraction=fraction, **step)
                for key in metrics:
                    metrics[key].append(row[key])
                if stop:
                    break
            self._n_updates += int(epoch_used)
            if stop:
                break
        self.optimizer_steps = getattr(self, 'optimizer_steps', 0) + optimizer_steps
        self.tactical_optimizer_steps = getattr(self, 'tactical_optimizer_steps', 0) + tactical_steps
        self.revive_optimizer_steps = getattr(self, 'revive_optimizer_steps', 0) + revive_steps
        self.critic_optimizer_steps = getattr(self, 'critic_optimizer_steps', 0) + critic_steps
        self.rejected_actor_steps = getattr(self, 'rejected_actor_steps', 0) + rejected_steps
        self.optimized_samples = getattr(self, 'optimized_samples', 0) + sample_visits
        self.on_policy_rollout_samples = getattr(self, 'on_policy_rollout_samples', 0) + buffer.buffer_size * self.n_envs
        self.completed_updates = getattr(self, 'completed_updates', 0) + int(critic_steps > 0)
        if critic_steps:
            self.last_completed_samples = self.num_timesteps
        losses = {key: float(np.mean(value)) if value else 0. for key, value in metrics.items()}
        for key, value in losses.items():
            self.logger.record('train/' + key, value)
        self.logger.record('train/n_updates', self._n_updates)
        self.logger.record('train/explained_variance', explained_variance(buffer.values.flatten(), buffer.returns.flatten()))
        row = dict(kind='ppo-update', policy_version=6, total_samples=self.num_timesteps,
            optimized_samples=self.optimized_samples, optimizer_steps=optimizer_steps,
            tactical_optimizer_steps=tactical_steps, revive_optimizer_steps=revive_steps,
            critic_optimizer_steps=critic_steps, rejected_actor_steps=rejected_steps,
            optimizer_sample_visits=sample_visits, on_policy_rollout_samples=self.on_policy_rollout_samples,
            completed_updates=self.completed_updates, gradient_epochs=self._n_updates,
            optimization_seconds=time.monotonic() - started, demonstration_updates=0, losses=losses,
            frozen_skill=True, kl_guard='exact_old_to_new_full_rollout_poststep_rollback')
        if getattr(self, 'update_log_path', None):
            with Path(self.update_log_path).open('a') as stream:
                stream.write(json.dumps(row) + '\n')
        hook = getattr(self, 'completed_update_hook', None)
        if hook:
            hook(self, row)
