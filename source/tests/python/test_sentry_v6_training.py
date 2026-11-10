"""CPU-only toy optimizer checks; no match simulation or research training."""
import copy
from pathlib import Path
import random
import tempfile
import unittest

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from rmuc2027_research.rl.asymmetric_ppo import AsymmetricPPO
from rmuc2027_research.rl.sentry_v4 import FEATURE_INDEX, SKILL_INDICES
from rmuc2027_research.rl.sentry_v4_policy import SentryV4Policy
from rmuc2027_research.rl.sentry_v6_policy import SentryV6Policy, REVIVE_PRIOR, export_actor
from rmuc2027_research.rl.v6_ppo import HybridPPO, near_roof
from rmuc2027_research.rl.v6_training import migrate_from_v4, save_state, restore_state


class ToyEnv(gym.Env):
    def __init__(self, version=6):
        self.version = version
        spaces = dict(actor=gym.spaces.Box(-np.inf, np.inf, (775 if version == 6 else 739,), np.float32),
                      critic=gym.spaces.Box(-np.inf, np.inf, (454,), np.float32))
        if version == 6:
            spaces['revive_mask'] = gym.spaces.Box(0, 1, (3,), np.float32)
        self.observation_space = gym.spaces.Dict(spaces)
        self.action_space = (gym.spaces.Box(np.array([-1] * 6 + [0], np.float32),
            np.array([1] * 6 + [2], np.float32)) if version == 6 else gym.spaces.Box(-1, 1, (6,), np.float32))
        self.t = 0

    def _obs(self):
        actor = np.zeros(775 if self.version == 6 else 739, np.float32)
        actor[FEATURE_INDEX['self.hp']] = .2 if self.t % 2 == 0 else 0.
        actor[FEATURE_INDEX['packet.0201.valid']] = 1
        actor[FEATURE_INDEX['packet.020D.valid']] = 1
        actor[FEATURE_INDEX['ack.paid_revive']] = 1
        result = dict(actor=actor, critic=np.full(454, self.t, np.float32))
        if self.version == 6:
            result['revive_mask'] = np.array([1, self.t % 2, self.t % 2], np.float32)
        return result

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        return self._obs(), 1., False, self.t == 4, {}

    def sampler_state(self):
        return dict(session=dict(episode_index=9), gym_rng=copy.deepcopy(self.np_random.bit_generator.state))

    def restore_sampler_state(self, state):
        self.restored = copy.deepcopy(state)

    def calibration_rows(self):
        return []

    def execution_state(self):
        return dict(reset_on_resume=True)


def make_model(version=6, **overrides):
    vec = DummyVecEnv([lambda: ToyEnv(version)])
    kwargs = dict(n_steps=4, batch_size=4, n_epochs=1, normalize_advantage=False,
                  gamma=.9, ent_coef=.001, seed=17, device='cpu')
    kwargs.update(overrides)
    model = (HybridPPO(SentryV6Policy, vec, **kwargs) if version == 6 else
             AsymmetricPPO(SentryV4Policy, vec, learning_rate=1e-5, **kwargs))
    model.set_logger(configure(folder=None, format_strings=[]))
    return model, vec


def observations(dead=(False, True, False, True)):
    rng = np.random.default_rng(89)
    actor = rng.normal(0, .1, (len(dead), 775)).astype(np.float32)
    for index, is_dead in enumerate(dead):
        actor[index, FEATURE_INDEX['self.hp']] = 0 if is_dead else .2
        for command in ('0201', '020D'):
            actor[index, FEATURE_INDEX[f'packet.{command}.valid']] = 1
            actor[index, FEATURE_INDEX[f'packet.{command}.age']] = 0
        actor[index, FEATURE_INDEX['ack.paid_revive']] = 1
    return dict(actor=torch.tensor(actor), critic=torch.tensor(rng.normal(size=(len(dead), 454)).astype(np.float32)),
                revive_mask=torch.tensor([[1, int(value), int(value)] for value in dead], dtype=torch.float32))


def fill_update(model, target=1.):
    obs = observations()
    buffer = model.rollout_buffer
    buffer.reset()
    for index in range(4):
        row = {key: value[index:index + 1] for key, value in obs.items()}
        with torch.no_grad():
            actions, values, logp = model.policy(row, deterministic=True)
        buffer.add({key: value.numpy() for key, value in row.items()}, actions.numpy(),
                   np.array([1.]), np.array([index == 0]), values, logp)
    buffer.compute_returns_and_advantage(torch.zeros(1), np.array([False]))
    buffer.advantages[:] = np.array([.4, -.8, 1.2, -1.6], np.float32).reshape(4, 1)
    buffer.returns[:] = target
    model.num_timesteps += 4
    model._current_progress_remaining = 1.
    np.random.seed(123)
    model.train()


def assert_optimizer_equal(test, left, right):
    old, new = left.state_dict(), right.state_dict()
    test.assertEqual(old['param_groups'], new['param_groups'])
    test.assertEqual(set(old['state']), set(new['state']))
    for index, state in old['state'].items():
        for key, value in state.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(value, new['state'][index][key], rtol=0, atol=0)
            else:
                test.assertEqual(value, new['state'][index][key])


class V6TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_exact_hybrid_logprob_entropy_and_masked_dead_dirac(self):
        model, vec = make_model()
        try:
            obs = observations((False, True))
            dist = model.policy.get_distribution(obs)
            actions = dist.mode()
            self.assertEqual(actions[1, 6].item(), 1)
            self.assertTrue(torch.equal(actions[1, :6], torch.zeros(6)))
            cont, categorical = dist.log_prob_parts(actions)
            torch.testing.assert_close(dist.log_prob(actions), cont + categorical)
            self.assertEqual(cont[1].item(), 0)
            self.assertEqual(dist.entropy_parts()[0][1].item(), 0)
            self.assertEqual(categorical[0].item(), 0)
            torch.testing.assert_close(dist.categorical.probs[1], torch.tensor(REVIVE_PRIOR).softmax(-1))
            actions[1, 6] = .9
            with self.assertRaisesRegex(ValueError, 'integer'):
                dist.log_prob(actions)
            actions[1, 6] = 1
            actions[1, 0] = .1
            with self.assertRaisesRegex(ValueError, 'zero'):
                dist.log_prob(actions)
            obs['revive_mask'][1] = torch.tensor([1., 1., 0.])
            dist = model.policy.get_distribution(obs)
            for _ in range(100):
                self.assertNotEqual(dist.sample()[1, 6].item(), 2)
        finally:
            vec.close()

    def test_frozen_skill_and_std_and_independent_critic_clip(self):
        first, vec1 = make_model()
        second, vec2 = make_model()
        try:
            frozen = {name: value.clone() for name, value in first.policy.state_dict().items()
                      if '.skill.' in name or name == 'skill_log_std'}
            fill_update(first, 1.)
            fill_update(second, 1e9)
            for left, right in zip(first.policy.actor_parameters(), second.policy.actor_parameters()):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            for name, value in frozen.items():
                torch.testing.assert_close(first.policy.state_dict()[name], value, rtol=0, atol=0)
            optimized = {id(p) for optimizer in (first.tactical_optimizer, first.revive_optimizer,
                first.critic_optimizer) for group in optimizer.param_groups for p in group['params']}
            self.assertFalse(optimized & {id(p) for p in first.policy.mlp_extractor.policy_net.skill.parameters()})
            self.assertEqual(first.optimizer_steps, 1)
            self.assertTrue(first.revive_optimizer.state)
        finally:
            vec1.close()
            vec2.close()

    def test_poststep_kl_rejection_restores_both_adam_states(self):
        model, vec = make_model()
        try:
            fill_update(model)
            reference = copy.deepcopy(model.policy)
            obs = observations()
            before = [parameter.clone() for parameter in model.policy.actor_parameters()]
            tactical = copy.deepcopy(model.tactical_optimizer)
            revive = copy.deepcopy(model.revive_optimizer)
            model.tactical_target_kl = model.near_roof_target_kl = model.revive_target_kl = 1e-15
            actions = model.policy.get_distribution(obs).mode()
            _, logp, _ = model.policy.evaluate_actions(obs, actions)
            result = model.actor_step(-logp.mean(), reference, obs)
            self.assertFalse(result['actor_step_accepted'])
            for left, right in zip(before, model.policy.actor_parameters()):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            assert_optimizer_equal(self, tactical, model.tactical_optimizer)
            assert_optimizer_equal(self, revive, model.revive_optimizer)
        finally:
            vec.close()

    def test_inactive_branch_cannot_move_through_old_adam_momentum(self):
        model, vec = make_model()
        try:
            fill_update(model)
            for dead, name, parameters in (
                    (False, 'revive_optimizer', model.policy.revive_parameters),
                    (True, 'tactical_optimizer', model.policy.tactical_parameters)):
                obs = observations((dead, dead))
                reference = copy.deepcopy(model.policy)
                before = [parameter.clone() for parameter in parameters()]
                optimizer = copy.deepcopy(getattr(model, name))
                actions = model.policy.get_distribution(obs).mode()
                _, logp, entropy = model.policy.evaluate_actions(obs, actions)
                result = model.actor_step(-logp.mean() - .001 * entropy.mean(), reference, obs)
                self.assertTrue(result['actor_step_accepted'])
                for left, right in zip(before, parameters()):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                assert_optimizer_equal(self, optimizer, getattr(model, name))
        finally:
            vec.close()

    def test_v4_migration_zero_columns_adam_moments_and_alive_exact_parity(self):
        old, old_vec = make_model(4)
        target_vec = DummyVecEnv([lambda: ToyEnv(6)])
        try:
            # A few toy backward operations seed actual Adam momentum; no rollout/match.
            for optimizer, parameters in ((old.actor_optimizer, old.policy.actor_parameters()),
                                           (old.critic_optimizer, old.policy.critic_parameters())):
                optimizer.zero_grad()
                sum(parameter.square().sum() for parameter in parameters).backward()
                optimizer.step()
            old.num_timesteps = 123
            old.completed_updates = 3
            with tempfile.TemporaryDirectory() as folder:
                source = Path(folder) / 'old.zip'
                old.save(source)
                cfg = dict(seed=11, ppo=dict(n_steps=4, batch_size=4, n_epochs=1))
                migrated, report = migrate_from_v4(source, target_vec, cfg)
                obs = observations((False, False, False, False))
                old_obs = dict(actor=obs['actor'][:, :739].contiguous(), critic=obs['critic'])
                with torch.no_grad():
                    old_dist = old.policy.get_distribution(old_obs)
                    new_dist = migrated.policy.get_distribution(obs)
                torch.testing.assert_close(old_dist.distribution.mean, new_dist.normal.mean, rtol=0, atol=0)
                torch.testing.assert_close(old.policy.log_std, migrated.policy.log_std, rtol=0, atol=0)
                torch.testing.assert_close(old.policy.predict_values(old_obs), migrated.policy.predict_values(obs), rtol=0, atol=0)
                weight = migrated.policy.mlp_extractor.policy_net.tactical[0].weight
                self.assertEqual(torch.count_nonzero(weight[:, 739:]).item(), 0)
                state = migrated.tactical_optimizer.state[weight]
                self.assertEqual(torch.count_nonzero(state['exp_avg'][:, 739:]).item(), 0)
                old_weight = old.policy.mlp_extractor.policy_net.tactical[0].weight
                torch.testing.assert_close(state['exp_avg'][:, :739], old.actor_optimizer.state[old_weight]['exp_avg'], rtol=0, atol=0)
                self.assertEqual(migrated.num_timesteps, 123)
                self.assertFalse(report['rollout_reused'])
                self.assertFalse(migrated.rollout_buffer.full)
                self.assertIsNone(migrated._last_obs)
        finally:
            old_vec.close()
            target_vec.close()

    def test_three_optimizers_rng_and_next_update_resume(self):
        original, vec = make_model()
        try:
            fill_update(original)
            with tempfile.TemporaryDirectory() as directory:
                save_state(original, vec, directory)
                expected = (random.random(), np.random.random(), torch.rand(1))
                restored = HybridPPO.load(Path(directory) / 'model.zip', env=vec, device='cpu')
                restored.set_logger(configure(folder=None, format_strings=[]))
                state = restore_state(directory, vec)
                self.assertEqual(random.random(), expected[0])
                self.assertEqual(np.random.random(), expected[1])
                torch.testing.assert_close(torch.rand(1), expected[2], rtol=0, atol=0)
                self.assertTrue(state['frozen_skill'])
                for name in ('tactical_optimizer', 'revive_optimizer', 'critic_optimizer'):
                    assert_optimizer_equal(self, getattr(original, name), getattr(restored, name))
                fill_update(original)
                fill_update(restored)
                for left, right in zip(original.policy.parameters(), restored.policy.parameters()):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
        finally:
            vec.close()

    def test_critic_and_recovery_history_do_not_enter_skill(self):
        model, vec = make_model()
        try:
            obs = observations((False, False))
            before = model.policy.get_distribution(obs).normal.mean
            changed = {key: value.clone() for key, value in obs.items()}
            changed['critic'] *= 1e8
            torch.testing.assert_close(before, model.policy.get_distribution(changed).normal.mean, rtol=0, atol=0)
            excluded = [i for i in range(775) if i not in SKILL_INDICES]
            changed['actor'][:, excluded] += 100
            after = model.policy.get_distribution(changed).normal.mean
            torch.testing.assert_close(before[:, 2:4], after[:, 2:4], rtol=0, atol=0)
            changed['actor'][:, FEATURE_INDEX['roof.valid']] = 1
            changed['actor'][:, FEATURE_INDEX['roof.along']] = 100
            self.assertFalse(near_roof(changed['actor']).any())
        finally:
            vec.close()

    def test_rollout_contains_masks_and_timeout_uses_terminal_critic(self):
        model, vec = make_model()
        try:
            calls = []
            def values(obs):
                calls.extend(obs['critic'][:, 0].tolist())
                return obs['critic'][:, :1]
            model.policy.predict_values = values
            _, callback = model._setup_learn(4, callback=lambda locals_, globals_: True)
            callback.on_training_start({}, {})
            model.collect_rollouts(vec, callback, model.rollout_buffer, n_rollout_steps=4)
            self.assertIn(4, calls)
            np.testing.assert_allclose(model.rollout_buffer.rewards[:, 0], [1, 1, 1, 4.6])
            np.testing.assert_array_equal(model.rollout_buffer.observations['revive_mask'][:, 0, 1], [0, 1, 0, 1])
            np.testing.assert_array_equal(model.rollout_buffer.actions[[1, 3], 0, :6], np.zeros((2, 6)))
            self.assertTrue(np.isin(model.rollout_buffer.actions[:, 0, 6], [0, 1, 2]).all())
        finally:
            vec.close()

    def test_export_numpy_parity_for_alive_and_dead(self):
        from rmuc2027_research.rl.sentry_v6 import portable_forward
        model, vec = make_model()
        try:
            obs = observations((False, True))
            with torch.no_grad():
                expected = model.policy.get_distribution(obs).mode().numpy()
            actor = export_actor(model.policy)
            for features, result in zip(obs['actor'].numpy(), expected):
                np.testing.assert_allclose(portable_forward(features.tolist(), actor), result, rtol=1e-5, atol=1e-6)
        finally:
            vec.close()


if __name__ == '__main__':
    unittest.main()
