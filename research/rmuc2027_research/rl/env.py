"""Gymnasium adapter; install research/requirements-rl.txt to enable this module."""
import gymnasium as gym
import numpy as np

from .portable import OBS_SIZE
from .session import ArenaSession


def make_env(config):
    # Keep the spawned sampler's factory outside learning.py so a sampler
    # imports Gym/numpy, without importing torch or allocating a network.
    return SentryEnv(config)


class SentryEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, config=None, record_replay=False):
        self.session = ArenaSession(config, record_replay)
        self.observation_space = gym.spaces.Box(-1, 1, shape=(OBS_SIZE,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        try:
            super().reset(seed=seed)
            if options:
                raise ValueError('v1 uses immutable environment configuration; reset options must be empty')
            # Gym owns the RNG: seeded resets reproduce; unseeded resets advance it.
            match_seed = int(seed) if seed is not None else int(self.np_random.integers(0, 2**31))
            obs, info = self.session.reset(match_seed)
            return np.asarray(obs, dtype=np.float32), info
        except BaseException:
            self.close()
            raise

    def step(self, action):
        try:
            if np.asarray(action).shape != (2,):
                raise ValueError('Expected action shape (2,)')
            obs, reward, terminated, truncated, info = self.session.step([float(v) for v in action])
            return np.asarray(obs, dtype=np.float32), reward, terminated, truncated, info
        except BaseException:
            self.close()
            raise

    def close(self):
        self.session.close()

    def process_ids(self):
        """Owned child PIDs for sampler diagnostics; never part of Actor input."""
        battle = self.session.battle
        return [] if battle is None else [battle.match.process.pid]+[
            policy.process.pid for _, _, policy in battle.policies if hasattr(policy, 'process')]
