"""SB3 vector adapter: CPU policies concurrently feed one batched GPU worker."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from . import batching


class WarpVecEnv(DummyVecEnv):
    def __init__(self, factories):
        if batching.active_service is not None:
            raise RuntimeError('Only one kinematic vector environment may be active per process')
        self.physics = batching.PhysicsService(len(factories))
        self.pool = ThreadPoolExecutor(max_workers=len(factories))
        batching.active_service = self.physics
        try:
            super().__init__(factories)
        except BaseException:
            self.pool.shutdown(wait=True)
            self.physics.close()
            batching.active_service = None
            raise

    def _step_one(self,i):
        try:
            return self.envs[i].step(self.actions[i])
        finally:
            self.physics.worker_done()

    def step_wait(self):
        self.physics.begin_round(len(self.envs))
        futures = [self.pool.submit(self._step_one,i) for i in range(len(self.envs))]
        results = [f.result() for f in futures]
        resets={i:self.pool.submit(self.envs[i].reset)
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
        futures=[self.pool.submit(env.reset,seed=self._seeds[i],options=self._options[i])
                 for i,env in enumerate(self.envs)]
        for i,future in enumerate(futures):
            obs,self.reset_infos[i]=future.result()
            self._save_obs(i,obs)
        self._reset_seeds();self._reset_options()
        return self._obs_from_buf()

    def close(self):
        try:
            self.pool.shutdown(wait=True)
            super().close()
        finally:
            self.physics.close()
            batching.active_service = None
