"""SB3-compatible policy with explicit actor/critic and locomotion input isolation.

Observation: Dict(actor=Box(739), critic=Box(N)). The concatenation below is only
transport: separate slices enter separate networks, with no shared trainable
feature extractor. Call actor_parameters()/critic_parameters() for independent
optimizers; this module performs no optimizer steps or gradient clipping.
"""
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from . import sentry_v4

TACTICAL_HIDDEN = 96
SKILL_HIDDEN = 64


class AsymmetricFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        actor = observation_space.spaces['actor'].shape
        critic = observation_space.spaces['critic'].shape
        if actor != (sentry_v4.schema()['obs_size'],) or len(critic) != 1:
            raise ValueError('Expected flat v4 actor and privileged critic Dict observations')
        super().__init__(observation_space, actor[0]+critic[0])
        self.actor_size, self.critic_size = actor[0], critic[0]

    def forward(self, observations):
        return torch.cat((observations['actor'], observations['critic']), dim=-1)


def _mlp(input_size, hidden, output_size):
    return nn.Sequential(nn.Linear(input_size, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, output_size))


class BranchedActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.tactical = _mlp(sentry_v4.schema()['obs_size'], TACTICAL_HIDDEN, 4)
        self.skill = _mlp(37, SKILL_HIDDEN, 2)
        self.register_buffer('skill_indices', torch.tensor(sentry_v4.SKILL_INDICES, dtype=torch.long))

    def forward(self, actor):
        tactical = self.tactical(actor)
        skill = self.skill(actor.index_select(-1, self.skill_indices))
        return torch.cat((tactical[..., :2], skill, tactical[..., 2:]), dim=-1)


class AsymmetricExtractor(nn.Module):
    def __init__(self, critic_size):
        super().__init__()
        self.actor_size = sentry_v4.schema()['obs_size']
        self.policy_net = BranchedActor()
        self.value_net = nn.Sequential(nn.Linear(critic_size, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh())
        self.latent_dim_pi, self.latent_dim_vf = 6, 128

    def forward_actor(self, features):
        return self.policy_net(features[..., :self.actor_size])

    def forward_critic(self, features):
        return self.value_net(features[..., self.actor_size:])

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)


class SentryV4Policy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        kwargs['features_extractor_class'] = AsymmetricFeatures
        kwargs['share_features_extractor'] = True
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = AsymmetricExtractor(self.features_extractor.critic_size).to(self.device)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        # The two branches already produce the six means. No final dense layer
        # may reconnect clock-sensitive tactical outputs into fold/jump outputs.
        self.action_net = nn.Identity()
        for head in (self.mlp_extractor.policy_net.tactical[-1], self.mlp_extractor.policy_net.skill[-1]):
            nn.init.orthogonal_(head.weight, gain=.01)
            nn.init.zeros_(head.bias)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def actor_parameters(self):
        yield from self.mlp_extractor.policy_net.parameters()
        yield self.log_std

    def critic_parameters(self):
        yield from self.mlp_extractor.value_net.parameters()
        yield from self.value_net.parameters()


def export_actor(policy):
    def layers(net):
        return [dict(weight=m.weight.detach().cpu().tolist(), bias=m.bias.detach().cpu().tolist())
            for m in net if isinstance(m, nn.Linear)]
    actor = policy.mlp_extractor.policy_net
    return dict(version=4, schema_hash=sentry_v4.SCHEMA_HASH,
        skill_indices=list(sentry_v4.SKILL_INDICES),
        architecture=dict(tactical_hidden=[m.out_features for m in actor.tactical if isinstance(m,nn.Linear)][:-1],
                          skill_hidden=[m.out_features for m in actor.skill if isinstance(m,nn.Linear)][:-1]),
        tactical=layers(actor.tactical), skill=layers(actor.skill))


portable_forward = sentry_v4.portable_forward
