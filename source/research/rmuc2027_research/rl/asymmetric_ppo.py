"""PPO with detached advantages and independently optimized actor/critic.

Uses SB3's Dict rollout collection, including terminal critic bootstrap for
time-limit truncations. Critic gradients cannot alter the actor's clip norm.
"""
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance


class AsymmetricPPO(PPO):
    def _setup_model(self):
        super()._setup_model()
        actor=list(self.policy.actor_parameters());critic=list(self.policy.critic_parameters())
        if {id(p) for p in actor}&{id(p) for p in critic}:raise ValueError('Actor and critic share trainable parameters')
        self.actor_optimizer=torch.optim.Adam(actor,lr=self.lr_schedule(1),eps=1e-5)
        self.critic_optimizer=torch.optim.Adam(critic,lr=self.lr_schedule(1),eps=1e-5)

    def _get_torch_save_params(self):
        return ['policy','actor_optimizer','critic_optimizer'],[]

    def _excluded_save_params(self):
        return super()._excluded_save_params()+['update_log_path','completed_update_hook','training_deadline']

    def train(self):
        if getattr(self,'sentry_guidance_path',None) or getattr(self,'skill_guidance_datasets',None):
            raise ValueError('v4 PPO updates cannot contain hidden demonstration gradients')
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor_optimizer,self.critic_optimizer])
        clip=self.clip_range(self._current_progress_remaining)
        vf_clip=self.clip_range_vf(self._current_progress_remaining) if self.clip_range_vf else None
        actor_params=list(self.policy.actor_parameters());critic_params=list(self.policy.critic_parameters())
        metrics={k:[] for k in ('policy_gradient_loss','value_loss','entropy_loss','approx_kl','clip_fraction')}
        started=time.monotonic();stop=False;optimizer_steps=sample_visits=0
        for epoch in range(self.n_epochs):
            for data in self.rollout_buffer.get(self.batch_size):
                values,log_prob,entropy=self.policy.evaluate_actions(data.observations,data.actions)
                values=values.flatten();advantages=data.advantages.detach()
                if self.normalize_advantage and len(advantages)>1:
                    advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
                log_ratio=log_prob-data.old_log_prob
                ratio=log_ratio.exp()
                policy_loss=-torch.min(advantages*ratio,advantages*ratio.clamp(1-clip,1+clip)).mean()
                entropy_loss=-entropy.mean() if entropy is not None else log_prob.mean()
                with torch.no_grad():
                    kl=((log_ratio.exp()-1)-log_ratio).mean().item()
                    fraction=(torch.abs(ratio-1)>clip).float().mean().item()
                if self.target_kl is not None and kl>1.5*self.target_kl:
                    stop=True;break
                predicted=values if vf_clip is None else data.old_values+(values-data.old_values).clamp(-vf_clip,vf_clip)
                value_loss=F.mse_loss(data.returns.detach(),predicted)
                self.actor_optimizer.zero_grad(set_to_none=True)
                (policy_loss+self.ent_coef*entropy_loss).backward()
                torch.nn.utils.clip_grad_norm_(actor_params,self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.zero_grad(set_to_none=True)
                (self.vf_coef*value_loss).backward()
                torch.nn.utils.clip_grad_norm_(critic_params,self.max_grad_norm)
                self.critic_optimizer.step()
                optimizer_steps+=1;sample_visits+=len(data.actions)
                for key,value in zip(metrics,(policy_loss.item(),value_loss.item(),entropy_loss.item(),kl,fraction)):
                    metrics[key].append(value)
            self._n_updates+=1
            if stop:break
        self.optimizer_steps=getattr(self,'optimizer_steps',0)+optimizer_steps
        self.optimized_samples=getattr(self,'optimized_samples',0)+sample_visits
        self.on_policy_rollout_samples=getattr(self,'on_policy_rollout_samples',0)+self.rollout_buffer.buffer_size*self.n_envs
        self.completed_updates=getattr(self,'completed_updates',0)+1
        self.last_completed_samples=self.num_timesteps
        for name,values in metrics.items():self.logger.record('train/'+name,float(np.mean(values)) if values else 0.)
        self.logger.record('train/n_updates',self._n_updates)
        self.logger.record('train/explained_variance',explained_variance(self.rollout_buffer.values.flatten(),self.rollout_buffer.returns.flatten()))
        row=dict(kind='ppo-update',total_samples=self.num_timesteps,optimized_samples=self.optimized_samples,
                 optimizer_steps=optimizer_steps,optimizer_sample_visits=sample_visits,
                 on_policy_rollout_samples=self.on_policy_rollout_samples,
                 completed_updates=self.completed_updates,gradient_epochs=self._n_updates,
                 optimization_seconds=time.monotonic()-started,demonstration_updates=0,
                 losses={key:float(np.mean(value)) if value else 0. for key,value in metrics.items()})
        if getattr(self,'update_log_path',None):
            with Path(self.update_log_path).open('a') as stream:stream.write(json.dumps(row)+'\n')
        hook=getattr(self,'completed_update_hook',None)
        if hook:hook(self,row)
