"""Opponent-pool sampling contract; no neural libraries or engine needed."""
import unittest

import numpy as np

from rmuc2027_research.rl.session import load_config, opponent_for, provenance

try:
    import stable_baselines3  # noqa: F401  (imports torch; unavailable in some envs)
    _HAVE_SB3 = True
except ImportError:
    _HAVE_SB3 = False

BASELINE = 'examples/claude_opus55_xhigh_2027_hold'
OTHER = 'examples/dsv41_1017_heavy'
THIRD = 'examples/step5_2027_echo'


_ABSENT = object()


def config_with(pool=_ABSENT):
    supplied = {'opponent': BASELINE}
    if pool is not _ABSENT:
        supplied['opponent_pool'] = pool
    return load_config(supplied)


class RotationTests(unittest.TestCase):
    def test_seed_rotation_is_pure(self):
        # seed % len(pool): no rng, so workers and replays agree exactly.
        config = config_with([BASELINE, OTHER, THIRD])
        self.assertEqual([opponent_for(config, seed) for seed in (0, 1, 2, 3, 12)],
                         [BASELINE, OTHER, THIRD, BASELINE, BASELINE])

    def test_pool_absent_keeps_opponent(self):
        config = config_with()
        self.assertEqual(opponent_for(config, 12345), BASELINE)
        record = provenance(config)
        self.assertNotIn('opponent_pool', record)
        self.assertNotIn('active_opponent', record)

    def test_baseline_occupies_first_pool_slot_semantics(self):
        # The pool is required to contain the baseline, so the seed % n == 0
        # slot stays the original fixed opponent.
        config = config_with([BASELINE, OTHER])
        for seed in (0, 2, 24):
            self.assertEqual(opponent_for(config, seed), BASELINE)

    def test_rotation_covers_every_pool_member(self):
        pool = [BASELINE, OTHER, THIRD]
        config = config_with(pool)
        picked = {opponent_for(config, seed) for seed in range(len(pool))}
        self.assertEqual(picked, set(pool))

    def test_weighted_rotation_matches_exact_deterministic_ratios(self):
        pool = [BASELINE, OTHER, THIRD]
        config = load_config({'opponent': BASELINE, 'opponent_pool': pool, 'opponent_pool_weights': [1.0, 3.0, 2.0]})
        picks = [opponent_for(config, seed) for seed in range(6)]
        self.assertEqual(picks[0], BASELINE)
        self.assertEqual(picks.count(BASELINE), 1)
        self.assertEqual(picks.count(OTHER), 3)
        self.assertEqual(picks.count(THIRD), 2)
        self.assertEqual(opponent_for(config, 7), opponent_for(config, 1))


class ValidationTests(unittest.TestCase):
    def test_pool_must_contain_baseline(self):
        with self.assertRaises(ValueError):
            config_with([OTHER, THIRD])

    def test_pool_duplicates_rejected(self):
        with self.assertRaises(ValueError):
            config_with([BASELINE, BASELINE])

    def test_empty_pool_rejected(self):
        # An empty pool must fail loudly rather than silently degrading to a
        # fixed opponent, which would look like pooling was enabled.
        for bad in ([], None):
            with self.assertRaises(ValueError):
                config_with(bad)

    def test_non_string_pool_rejected(self):
        with self.assertRaises(ValueError):
            config_with([BASELINE, 7])

    def test_invalid_pool_weights_rejected(self):
        with self.assertRaises(ValueError):
            load_config({'opponent': BASELINE, 'opponent_pool_weights': [1.0]})
        for bad in ([1.0], [1.0, 0.0], [1.0, -2.0], [1.0, True], [1.0, float('nan')]):
            with self.assertRaises(ValueError):
                load_config({'opponent': BASELINE, 'opponent_pool': [BASELINE, OTHER], 'opponent_pool_weights': bad})


class ProvenanceTests(unittest.TestCase):
    def test_sources_ignore_per_episode_opponent(self):
        # sources['opponent'] is the baseline in both cases; the per-episode
        # pick lives only in active_opponent. evaluate/export/resume compare
        # sources, so this invariant is what keeps them working.
        config = config_with([BASELINE, OTHER])
        training = provenance(config)
        replay = provenance(config, OTHER)
        self.assertEqual(training['sources'], replay['sources'])
        self.assertEqual(training['active_opponent'], BASELINE)
        self.assertEqual(replay['active_opponent'], OTHER)

    def test_pool_entries_are_recorded_in_config_order(self):
        pool = [BASELINE, OTHER, THIRD]
        record = provenance(config_with(pool))
        self.assertEqual(list(record['opponent_pool']), pool)
        for path, entry in record['opponent_pool'].items():
            self.assertTrue(entry['version'], path)
            self.assertTrue(entry['hashes'], path)

    def test_record_is_stable_across_calls(self):
        config = config_with([BASELINE, OTHER])
        self.assertEqual(provenance(config), provenance(config))


class SeedTransportTests(unittest.TestCase):
    @unittest.skipUnless(_HAVE_SB3, 'stable-baselines3/torch unavailable')
    def test_vec_seeds_are_sequential(self):
        # First-episode seeds come from VecEnv.seed(), not from the session rng.
        # Locks the assumption behind opponent_for when num_envs > 1.
        import gymnasium as gym
        from stable_baselines3.common.vec_env import DummyVecEnv

        class Stub(gym.Env):
            # SB3 >= 2 refuses non-Gymnasium envs in DummyVecEnv, so the stub
            # has to subclass gym.Env rather than stay a bare object.
            observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=float)
            action_space = gym.spaces.Discrete(1)

            def reset(self, seed=None, options=None):
                super().reset(seed=seed)
                return np.zeros(1, dtype=float), {}

            def step(self, action):
                return np.zeros(1, dtype=float), 0.0, True, False, {}

        vec = DummyVecEnv([Stub, Stub])
        try:
            self.assertEqual(list(vec.seed(42)), [42, 43])
            vec.seed(42)
            # VecEnv.reset returns a single stacked array of shape
            # (num_envs, *obs_shape), not a (obs, info) tuple: unpacking it
            # would slice the batch dimension and hide the second env.
            observations = vec.reset()
            self.assertEqual(observations.shape, (2, 1))
            self.assertEqual(observations.tolist(), [[0.0], [0.0]])
            # Seeds are consumed once: a second reset carries no seed, so the
            # per-episode opponent keeps coming from the episode's own seed.
            self.assertEqual(vec._seeds, [None, None])
        finally:
            vec.close()


class ResourceTests(unittest.TestCase):
    def test_unknown_pool_member_fails_when_recorded(self):
        # load_config only validates shape; existence is proven when the
        # episode records provenance, so a typo cannot pass silently.
        config = config_with([BASELINE, 'examples/does-not-exist'])
        self.assertEqual(opponent_for(config, 1), 'examples/does-not-exist')
        with self.assertRaises(Exception):
            provenance(config)


if __name__ == '__main__':
    unittest.main()
