"""Rule-observed hybrid sentry actor, with frozen v4 locomotion skills.

The seventh Box coordinate transports an exact categorical sample. The first
six coordinates retain SB3's unsquashed Gaussian latent / execution-clip
contract. A legally confirmed death replaces that Gaussian by a zero Dirac
action; invalid/stale HP is never interpreted as death.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical, Normal
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from . import sentry_v4
from .sentry_v4_policy import _mlp

ACTOR_SIZE = 775
CRITIC_SIZE = 454
TACTICAL_AXES = (0, 1, 4, 5)
REVIVE_PRIOR = (0., .1, -4.)


class ExpandedLinear(nn.Linear):
    """Keep the old 739-column arithmetic intact at zero-column migration."""
    def forward(self, inputs):
        return (F.linear(inputs[..., :739], self.weight[:, :739].contiguous(), self.bias) +
                F.linear(inputs[..., 739:], self.weight[:, 739:], None))


class HybridFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        expected = {'actor': (ACTOR_SIZE,), 'critic': (CRITIC_SIZE,), 'revive_mask': (3,)}
        if set(observation_space.spaces) != set(expected) or any(
                observation_space.spaces[k].shape != shape for k, shape in expected.items()):
            raise ValueError('v6 requires actor775, critic454 and revive_mask3')
        super().__init__(observation_space, ACTOR_SIZE + CRITIC_SIZE)

    def forward(self, observations):
        return torch.cat((observations['actor'], observations['critic']), dim=-1)


class V6Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.tactical = _mlp(ACTOR_SIZE, 96, 4)
        self.tactical[0] = ExpandedLinear(ACTOR_SIZE, 96)
        self.skill = _mlp(37, 64, 2)
        self.register_buffer('skill_indices', torch.tensor(sentry_v4.SKILL_INDICES, dtype=torch.long))
        self.skill.requires_grad_(False)

    def forward(self, actor):
        tactical = self.tactical(actor)
        skill = self.skill(actor.index_select(-1, self.skill_indices))
        return torch.cat((tactical[..., :2], skill, tactical[..., 2:]), dim=-1)


class V6Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_net = V6Actor()
        self.value_net = nn.Sequential(nn.Linear(CRITIC_SIZE, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh())
        self.latent_dim_pi, self.latent_dim_vf = 6, 128

    def forward_actor(self, features):
        return self.policy_net(features[..., :ACTOR_SIZE])

    def forward_critic(self, features):
        return self.value_net(features[..., ACTOR_SIZE:])

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)


class HybridDistribution:
    """Independent Gaussian and masked categorical, including dead Dirac rows."""
    def __init__(self, means, log_std, logits, mask):
        if mask.shape != logits.shape or not torch.isfinite(mask).all() or not (
                ((mask == 0) | (mask == 1)).all() and (mask[..., 0] == 1).all()):
            raise ValueError('revive_mask must be binary with wait always legal')
        if ((mask[..., 2] > 0) & (mask[..., 1] == 0)).any():
            raise ValueError('paid cannot be legal without confirmed-dead/free eligibility')
        self.mask = mask.bool()
        self.dead = self.mask[..., 1]
        self.normal = Normal(means, log_std.exp().expand_as(means))
        self.distribution = self.normal
        self.categorical = Categorical(logits=logits.masked_fill(~self.mask, -torch.inf))

    def get_actions(self, deterministic=False):
        continuous = self.normal.mean if deterministic else self.normal.sample()
        continuous = torch.where(self.dead[..., None], torch.zeros_like(continuous), continuous)
        category = self.categorical.logits.argmax(-1) if deterministic else self.categorical.sample()
        return torch.cat((continuous, category[..., None].to(continuous.dtype)), -1)

    def mode(self):
        return self.get_actions(True)

    def sample(self):
        return self.get_actions(False)

    def log_prob_parts(self, actions):
        categories = actions[..., 6]
        if not torch.isfinite(actions).all() or ((categories != categories.round()) |
                (categories < 0) | (categories > 2)).any():
            raise ValueError('v6 category is an exact integer sample, never a rounded Gaussian')
        categories = categories.long()
        if not self.mask.gather(-1, categories[..., None]).all():
            raise ValueError('Stored categorical action violates its stored legal mask')
        if (actions[self.dead, :6] != 0).any():
            raise ValueError('Confirmed-dead continuous action must be zero')
        gaussian = self.normal.log_prob(actions[..., :6]).sum(-1)
        gaussian = torch.where(self.dead, torch.zeros_like(gaussian), gaussian)
        return gaussian, self.categorical.log_prob(categories)

    def log_prob(self, actions):
        continuous, revive = self.log_prob_parts(actions)
        return continuous + revive

    def entropy_parts(self):
        continuous = self.normal.entropy().sum(-1)
        return torch.where(self.dead, torch.zeros_like(continuous), continuous), self.categorical.entropy()

    def entropy(self):
        continuous, revive = self.entropy_parts()
        return continuous + revive


class SentryV6Policy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        kwargs['features_extractor_class'] = HybridFeatures
        kwargs['share_features_extractor'] = True
        if kwargs.get('use_sde'):
            raise ValueError('v6 uses an explicit independent Gaussian, not gSDE')
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        if self.action_space.shape != (7,):
            raise ValueError('v6 requires the seven-coordinate hybrid transport Box')
        self.mlp_extractor = V6Extractor().to(self.device)
        self.revive_net = _mlp(ACTOR_SIZE, 16, 3).to(self.device)
        self.value_net = nn.Linear(128, 1).to(self.device)
        self.action_net = nn.Identity()
        self.tactical_log_std = nn.Parameter(torch.full((4,), self.log_std_init, device=self.device))
        self.register_buffer('skill_log_std', torch.full((2,), self.log_std_init, device=self.device))
        for network in (self.mlp_extractor, self.revive_net, self.value_net):
            network.apply(lambda module: self.init_weights(module, 2 ** .5))
        for head in (self.mlp_extractor.policy_net.tactical[-1], self.mlp_extractor.policy_net.skill[-1]):
            nn.init.orthogonal_(head.weight, gain=.01)
            nn.init.zeros_(head.bias)
        nn.init.zeros_(self.revive_net[-1].weight)
        with torch.no_grad():
            self.revive_net[-1].bias.copy_(torch.tensor(REVIVE_PRIOR, device=self.device))
        # Compatibility only: HybridPPO owns the three actual optimizers.
        self.optimizer = self.optimizer_class(self.tactical_parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    @property
    def log_std(self):
        return torch.cat((self.tactical_log_std[:2], self.skill_log_std, self.tactical_log_std[2:]))

    def tactical_parameters(self):
        yield from self.mlp_extractor.policy_net.tactical.parameters()
        yield self.tactical_log_std

    def revive_parameters(self):
        yield from self.revive_net.parameters()

    def actor_parameters(self):
        yield from self.tactical_parameters()
        yield from self.revive_parameters()

    def critic_parameters(self):
        yield from self.mlp_extractor.value_net.parameters()
        yield from self.value_net.parameters()

    def get_distribution(self, obs):
        actor = obs['actor']
        return HybridDistribution(self.mlp_extractor.policy_net(actor), self.log_std,
            self.revive_net(actor), obs['revive_mask'])

    def predict_values(self, obs):
        return self.value_net(self.mlp_extractor.value_net(obs['critic']))

    def forward(self, obs, deterministic=False):
        distribution = self.get_distribution(obs)
        actions = distribution.get_actions(deterministic)
        return actions, self.predict_values(obs), distribution.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        distribution = self.get_distribution(obs)
        return self.predict_values(obs), distribution.log_prob(actions), distribution.entropy()


def export_actor(policy):
    from . import sentry_v6
    def layers(network):
        return [dict(weight=m.weight.detach().cpu().tolist(), bias=m.bias.detach().cpu().tolist())
            for m in network if isinstance(m, nn.Linear)]
    actor = policy.mlp_extractor.policy_net
    return dict(version=6, schema_hash=sentry_v6.SCHEMA_HASH,
        skill_indices=list(sentry_v4.SKILL_INDICES), log_std=policy.log_std.detach().cpu().tolist(),
        architecture=dict(tactical_hidden=[96, 96], skill_hidden=[64, 64], revive_hidden=[16, 16]),
        tactical=layers(actor.tactical), skill=layers(actor.skill), revive=layers(policy.revive_net),
        revive_prior=list(REVIVE_PRIOR), frozen_skill=True,
        dead_continuous='zero_dirac', gaussian='latent_unsquashed_execution_clip')
