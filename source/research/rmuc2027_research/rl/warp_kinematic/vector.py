"""SB3 vector adapter: CPU policies concurrently feed one batched GPU worker."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from . import batching


class WarpVecEnv(DummyVecEnv):
    def __init__(self, factories):
        self.physics = self.pool = None
        self.envs = []
        self._closed = self._failed = False
        if batching.active_service is not None:
            raise RuntimeError('Only one kinematic vector environment may be active per process')

        def create(factory):
            env = factory()
            # DummyVecEnv assigns its list only after every factory succeeds.
            self.envs.append(env)
            return env

        try:
            self.physics = batching.PhysicsService(len(factories))
            self.pool = ThreadPoolExecutor(max_workers=len(factories))
            batching.active_service = self.physics
            super().__init__([lambda factory=factory: create(factory) for factory in factories])
        except BaseException as error:
            self._abort(error)
            raise

    def _check_open(self):
        if self._closed:
            raise RuntimeError('WarpVecEnv is closed')

    def _abort(self, error):
        self._failed = True
        try:
            self.close()
        except BaseException as cleanup_error:
            if hasattr(error, 'add_note'):
                error.add_note(f'Cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}')
                for note in getattr(cleanup_error, '__notes__', ()):
                    error.add_note(note)

    def _submit(self, function, *args, **kwargs):
        try:
            return self.pool.submit(function, *args, **kwargs)
        except BaseException as error:
            # submit may queue work before Thread.start fails, without returning
            # its Future. The partially scheduled round cannot be reused.
            self._abort(error)
            raise

    def step_async(self, actions):
        self._check_open()
        super().step_async(actions)

    def _step_one(self,i):
        try:
            return self.envs[i].step(self.actions[i])
        finally:
            self.physics.worker_done()

    def step_wait(self):
        self._check_open()
        self.physics.begin_round(len(self.envs))
        futures = [self._submit(self._step_one,i) for i in range(len(self.envs))]
        results = [f.result() for f in futures]
        resets={i:self._submit(self.envs[i].reset)
                for i,(_,_,terminated,truncated,_) in enumerate(results) if terminated or truncated}
        for i, (obs, reward, terminated, truncated, info) in enumerate(results):
            self.buf_rews[i] = reward
            self.buf_dones[i] = terminated or truncated
            self.buf_infos[i] = info
            self.buf_infos[i]['TimeLimit.truncated'] = truncated and not terminated
            if self.buf_dones[i]:
                self.buf_infos[i]['terminal_observation'] = obs
                obs, self.reset_infos[i] = resets[i].result()
            self._save_obs(i, obs)
        return self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos)

    def reset(self):
        self._check_open()
        futures=[self._submit(env.reset,seed=self._seeds[i],options=self._options[i])
                 for i,env in enumerate(self.envs)]
        for i,future in enumerate(futures):
            obs,self.reset_infos[i]=future.result()
            self._save_obs(i,obs)
        self._reset_seeds();self._reset_options()
        return self._obs_from_buf()

    def close(self):
        if self._closed:
            return
        self._closed = True
        error = None

        def cleanup(function):
            nonlocal error
            try:
                function()
            except BaseException as cleanup_error:
                if error is None:
                    error = cleanup_error
                elif hasattr(error, 'add_note'):
                    error.add_note(f'Cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}')

        # A failed submit can leave workers waiting for an absent round peer.
        # Wake physics before joining; healthy workers still finish normally.
        if self._failed and self.physics is not None:
            cleanup(self.physics.close)
        if self.pool is not None:
            cleanup(lambda: self.pool.shutdown(wait=True, cancel_futures=self._failed))
        seen = set()
        for env in self.envs:
            if id(env) not in seen:
                seen.add(id(env))
                cleanup(env.close)
        if not self._failed and self.physics is not None:
            cleanup(self.physics.close)
        if batching.active_service is self.physics:
            batching.active_service = None
        if error is not None:
            raise error
