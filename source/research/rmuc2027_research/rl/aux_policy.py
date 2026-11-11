"""Masked hybrid PPO policy; discrete choices are never rounded Gaussians."""
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal
from stable_baselines3.common.policies import ActorCriticPolicy

from . import privileged_aux as contract


def _conditional(name, actions):
    if name == 'rescue': return actions.get('pickup') == 1
    if name == 'arm_mode': return (actions.get('rescue') > 0) | (actions.get('pickup') == 1)
    if name == 'arm2_mode': return actions.get('pickup') == 2
    return None


def _active(action):
    """Tensor counterpart of contract.active_heads, used for both PPO and BC."""
    c = {k: action[:, contract.ACTION_INDEX[k]].long() for k in contract.CAT_NAMES}
    active = torch.ones_like(action)
    def set_(name, flag): active[:, contract.ACTION_INDEX[name]] = flag.to(action.dtype)
    for k in ('move_x', 'move_y'): set_(k, c['motion'] == 0)
    for k in ('goal_x', 'goal_y'): set_(k, c['motion'] == 1)
    for name, flag in (('heading', c['turn'] == 1), ('yaw_rate', c['turn'] == 2),
                       ('power', c['power_mode'] == 1), ('leg_height', c['leg_mode'] == 1),
                       ('jump_strength', c['jump'] == 1), ('aim_yaw', c['target'] != 0),
                       ('aim_pitch', c['target'] != 0), ('ammo_batches', c['ammo'] != 0)):
        set_(name, flag)
    for arm in ('arm', 'arm2'):
        for name in contract.CONTINUOUS:
            if name.startswith(arm+'_'): set_(name, c[arm+'_mode'] == (3 if '_joint' in name else 2))
    return active


class HybridDistribution:
    def __init__(self, mean, log_std, logits, mask):
        self.mean = mean
        self.normal = Normal(mean, log_std.exp().expand_as(mean))
        self.logits = logits
        self.mask = mask

    def _categorical(self, name, offset, choices):
        size = len(contract.CATEGORIES[name])
        allowed = self.mask[:, contract.N_CONT+offset:contract.N_CONT+offset+size] > .5
        forced = _conditional(name, choices)
        if forced is not None:
            allowed = allowed.clone(); allowed[forced] = False; allowed[forced, 0] = True
        if not torch.all(allowed.any(dim=1)): raise ValueError('Empty auxiliary categorical mask')
        return Categorical(logits=self.logits[:, offset:offset+size].masked_fill(~allowed, -1e9))

    def get_actions(self, deterministic=False):
        action = torch.zeros((self.mean.shape[0], len(contract.ACTION_NAMES)), device=self.mean.device)
        action[:, :contract.N_CONT] = torch.tanh(self.mean if deterministic else self.normal.rsample())
        choices = {}; offset = 0
        for name in contract.CAT_NAMES:
            dist = self._categorical(name, offset, choices)
            choice = dist.logits.argmax(dim=-1) if deterministic else dist.sample()
            action[:, contract.ACTION_INDEX[name]] = choice
            choices[name] = choice; offset += len(contract.CATEGORIES[name])
        active = _active(action)
        action[:, :contract.N_CONT] *= self.mask[:, :contract.N_CONT] * active[:, :contract.N_CONT]
        return action

    def log_prob_parts(self, actions):
        raw = actions[:, :contract.N_CONT].clamp(-1+1e-6, 1-1e-6)
        unsquashed = torch.atanh(raw)
        # Exact transformed normal density, including tanh's Jacobian.
        continuous = (self.normal.log_prob(unsquashed)-torch.log1p(-raw.square()))
        continuous *= self.mask[:, :contract.N_CONT]
        discrete = []; choices = {}; offset = 0
        for name in contract.CAT_NAMES:
            value = actions[:, contract.ACTION_INDEX[name]].long()
            dist = self._categorical(name, offset, choices)
            discrete.append(dist.log_prob(value)); choices[name] = value
            offset += len(contract.CATEGORIES[name])
        return torch.cat((continuous, torch.stack(discrete, dim=1)), dim=1) * _active(actions)

    def log_prob(self, actions): return self.log_prob_parts(actions).sum(dim=1)

    def entropy(self):
        # PPO uses -log_prob as its sample entropy estimate for transformed laws.
        return None


class AuxPolicy(ActorCriticPolicy):
    """SB3-compatible network with one model shared by both infantry seats.

    Role/model/side are represented by the contract, while per-episode state and
    masks stay in each seat's environment. Heavy uses a separate model instance.
    """
    def _build(self, lr_schedule):
        size = self.observation_space['actor'].shape[0]
        self.actor_net = nn.Sequential(nn.Linear(size, 256), nn.Tanh(), nn.Linear(256, 128), nn.Tanh())
        self.critic_net = nn.Sequential(nn.Linear(size, 256), nn.Tanh(), nn.Linear(256, 128), nn.Tanh())
        self.mean_net = nn.Linear(128, contract.N_CONT)
        self.logits_net = nn.Linear(128, sum(map(len, contract.CATEGORIES.values())))
        self.value_net = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.full((contract.N_CONT,), -1.))
        for net in (self.actor_net, self.critic_net):
            for layer in net:
                if isinstance(layer, nn.Linear): nn.init.orthogonal_(layer.weight, math.sqrt(2)); nn.init.zeros_(layer.bias)
        for net in (self.mean_net, self.logits_net): nn.init.orthogonal_(net.weight, .01); nn.init.zeros_(net.bias)
        nn.init.orthogonal_(self.value_net.weight, 1.); nn.init.zeros_(self.value_net.bias)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def get_distribution(self, obs):
        latent = self.actor_net(obs['actor'].float())
        return HybridDistribution(self.mean_net(latent), self.log_std,
                                  self.logits_net(latent), obs['action_mask'].float())

    def forward(self, obs, deterministic=False):
        dist = self.get_distribution(obs)
        actions = dist.get_actions(deterministic)
        return actions, self.predict_values(obs), dist.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        dist = self.get_distribution(obs)
        return self.predict_values(obs), dist.log_prob(actions), None

    def predict_values(self, obs): return self.value_net(self.critic_net(obs['actor'].float()))


class AuxLayerNormPolicy(AuxPolicy):
    """Stable auxiliary architecture; the original class remains loadable.

    Dense, mostly constant validity features can saturate an unnormalized tanh
    encoder during imitation. Normalize each hidden layer independently; this
    changes neither the observation contract nor PPO's hybrid action density.
    """
    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        for name in ('actor_net', 'critic_net'):
            original = getattr(self, name)
            setattr(self, name, nn.Sequential(original[0], nn.LayerNorm(256), nn.Tanh(),
                                             original[2], nn.LayerNorm(128), nn.Tanh()))
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)


def export_actor(policy, *, family=None):
    """JSON-serializable actor only: no value net, optimizer or training inputs."""
    policy = getattr(policy, 'policy', policy)
    def linear(layer): return {'weight': layer.weight.detach().cpu().tolist(), 'bias': layer.bias.detach().cpu().tolist()}
    payload = {'format': 'rmuc-aux-actor-v1', 'schema': contract.schema(), 'family': family,
            'layers': [linear(layer) for layer in policy.actor_net if isinstance(layer, nn.Linear)],
            'mean': linear(policy.mean_net), 'logits': linear(policy.logits_net)}
    norms = [layer for layer in policy.actor_net if isinstance(layer, nn.LayerNorm)]
    if norms:
        payload.update(format='rmuc-aux-actor-v2', architecture='layernorm_tanh_v1',
                       normalizations=[dict(**linear(layer), eps=layer.eps) for layer in norms])
    return payload


class PortableAuxActor:
    """Read-only NumPy inference, safe to share across independent seat controllers."""
    def __init__(self, payload):
        version = payload.get('format')
        if version not in ('rmuc-aux-actor-v1', 'rmuc-aux-actor-v2') or payload.get('schema') != contract.schema():
            raise ValueError('Auxiliary actor schema mismatch')
        self.family = payload.get('family')
        def layer(value):
            w, b = np.asarray(value['weight'], dtype=np.float32), np.asarray(value['bias'], dtype=np.float32)
            if not np.isfinite(w).all() or not np.isfinite(b).all(): raise ValueError('Non-finite actor weights')
            w.setflags(write=False); b.setflags(write=False)
            return w, b
        self.layers = tuple(layer(value) for value in payload['layers'])
        self.mean = layer(payload['mean']); self.logits = layer(payload['logits'])
        self.normalizations = ()
        if version == 'rmuc-aux-actor-v2':
            if payload.get('architecture') != 'layernorm_tanh_v1': raise ValueError('Unknown auxiliary architecture')
            values = payload.get('normalizations', [])
            if len(values) != len(self.layers): raise ValueError('Missing layer normalization')
            normals = []
            for value, (weight, _) in zip(values, self.layers):
                gamma, beta = layer(value); eps = float(value['eps'])
                if gamma.shape != (weight.shape[0],) or beta.shape != gamma.shape or not math.isfinite(eps) or eps <= 0:
                    raise ValueError('Invalid auxiliary layer normalization')
                normals.append((gamma, beta, eps))
            self.normalizations = tuple(normals)

    def predict(self, observation, deterministic=True, **kwargs):
        if not deterministic: raise ValueError('Frozen auxiliary pool uses deterministic actors')
        x = np.asarray(observation['actor'], dtype=np.float32)
        mask = np.asarray(observation['action_mask'], dtype=np.float32)
        batched = x.ndim == 2
        if not batched: x, mask = x[None, :], mask[None, :]
        # Small per-seat batches should not launch a BLAS thread team inside
        # each Battle controller thread. NumPy's direct contraction is compiled
        # and avoids both Python scalar loops and nested BLAS oversubscription.
        def linear(value, layer):
            return np.einsum('bi,oi->bo', value, layer[0], optimize=False)+layer[1]
        for index, layer in enumerate(self.layers):
            x = linear(x, layer)
            if self.normalizations:
                gamma, beta, eps = self.normalizations[index]
                x = (x-x.mean(axis=-1, keepdims=True))/np.sqrt(x.var(axis=-1, keepdims=True)+eps)
                x = x*gamma+beta
            x = np.tanh(x)
        continuous = np.tanh(linear(x, self.mean))
        logits = linear(x, self.logits)
        action = np.zeros((len(x), len(contract.ACTION_NAMES)), np.float32)
        action[:, :contract.N_CONT] = continuous
        for row in range(len(x)):
            choices = {}; offset = 0
            for name in contract.CAT_NAMES:
                size = len(contract.CATEGORIES[name]); allowed = mask[row, contract.N_CONT+offset:contract.N_CONT+offset+size] > .5
                conditional = contract.conditional_allowed(name, choices)
                if conditional is not None: allowed = np.asarray([i in conditional for i in range(size)])
                if not allowed.any(): raise ValueError('Empty auxiliary mask')
                value = int(np.argmax(np.where(allowed, logits[row, offset:offset+size], -np.inf)))
                choices[name] = value; action[row, contract.ACTION_INDEX[name]] = value; offset += size
            active = contract.active_heads(action[row])
            action[row, :contract.N_CONT] *= mask[row, :contract.N_CONT]*active[:contract.N_CONT]
        return (action if batched else action[0]), None


_PORTABLE_CACHE = {}


def load_portable(path):
    path = Path(path).resolve(); content = path.read_bytes()
    import hashlib
    key = (str(path), hashlib.sha256(content).hexdigest())
    if key not in _PORTABLE_CACHE:
        _PORTABLE_CACHE[key] = PortableAuxActor(json.loads(content))
    return _PORTABLE_CACHE[key]
