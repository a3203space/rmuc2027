"""Gymnasium adapter; install research/requirements-rl.txt to enable this module."""
import gymnasium as gym
import numpy as np
import json
import os
from pathlib import Path
import time

from .portable import policy_schema
from .session import ArenaSession


def make_env(config):
    # Keep the spawned sampler's factory outside learning.py so a sampler
    # imports Gym/numpy, without importing torch or allocating a network.
    return SentryEnv(config)


class SentryEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, config=None, record_replay=False):
        self.session = ArenaSession(config, record_replay)
        schema = policy_schema(self.session.config)
        bound = np.inf if schema['version'] in (3,4,5,6) else 1
        actor=gym.spaces.Box(-bound, bound, shape=(schema['obs_size'],), dtype=np.float32)
        if schema['version'] in (4,6):
            from .critic_state import SIZE
            self.observation_space=gym.spaces.Dict(actor=actor,critic=gym.spaces.Box(-np.inf,np.inf,shape=(SIZE,),dtype=np.float32))
            if schema['version']==6:
                self.observation_space.spaces['revive_mask']=gym.spaces.Box(0,1,shape=(3,),dtype=np.float32)
        elif schema['version']==5:
            from .privileged_aux import observation_space
            self.observation_space=observation_space()
        else:self.observation_space=actor
        if schema['version']==5:
            from .privileged_aux import action_space
            self.action_space=action_space()
        elif schema['version']==6:
            self.action_space=gym.spaces.Box(np.asarray([-1]*6+[0],np.float32),np.asarray([1]*6+[2],np.float32),dtype=np.float32)
        else:self.action_space = gym.spaces.Box(-1, 1, shape=(schema['action_size'],), dtype=np.float32)

    @staticmethod
    def _array(obs):
        return {k:np.asarray(v,dtype=np.float32) for k,v in obs.items()} if isinstance(obs,dict) else np.asarray(obs,dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        try:
            super().reset(seed=seed)
            # Gym owns the RNG: seeded resets reproduce; unseeded resets advance it.
            match_seed = int(seed) if seed is not None else int(self.np_random.integers(0, 2**31))
            for attempt in range(3):
                try:
                    obs, info = self.session.reset(match_seed,options) if options else self.session.reset(match_seed)
                    return self._array(obs), info
                except Exception as error:
                    if self.session.config.get('policy_version')!=6:raise
                    from .v6_recovery import RecoveryPreparationError
                    if (not isinstance(error,RecoveryPreparationError) or options
                            or self.session.config.get('evaluation') or attempt==2):raise
                    directory=self.session.config.get('_episode_log_dir')
                    if directory:
                        path=Path(directory);path.mkdir(parents=True,exist_ok=True)
                        (path/f'prefix-failed-{os.getpid()}-{time.time_ns()}.json').write_text(
                            json.dumps(dict(kind='not_prepared',report=getattr(error,'report',str(error))),allow_nan=False))
        except BaseException:
            self.close()
            raise

    def step(self, action):
        try:
            if np.asarray(action).shape != self.action_space.shape:
                raise ValueError(f'Expected action shape {self.action_space.shape}')
            obs, reward, terminated, truncated, info = self.session.step([float(v) for v in action])
            directory=self.session.config.get('_episode_log_dir')
            if directory and (terminated or truncated):
                # VecEnv resets before returning a terminal transition. A slow
                # next opening may hit the wall deadline before the callback
                # sees it, so retain the completed episode here as well.
                path=Path(directory);path.mkdir(parents=True,exist_ok=True)
                output=path/f'{os.getpid()}-{time.time_ns()}.json'
                pending=output.with_suffix('.tmp')
                pending.write_text(json.dumps(dict(info=info,terminated=terminated,
                    truncated=truncated,recorded_unix=time.time()),allow_nan=False))
                os.replace(pending,output)
            return self._array(obs), reward, terminated, truncated, info
        except BaseException:
            self.close()
            raise

    def close(self):
        self.session.close()

    def set_training_stage(self, stage):
        self.session.requested_stage = stage

    def sampler_state(self):
        return dict(session=self.session.sampler_state(),gym_rng=self.np_random.bit_generator.state)

    def restore_sampler_state(self,state):
        self.session.restore_sampler_state(state['session'])
        self.np_random.bit_generator.state=state['gym_rng']

    def execution_state(self):
        match=self.session.battle.match if self.session.battle else None
        return dict(resume_action='discard_and_start_new_episode',
                    aim_references=match._aim_references if match else {},
                    capability_drift=match.capability_drift.state_dict() if match and match.capability_drift else None,
                    aim_executors={key:value.state_dict() for key,value in match._aim_executors.items()} if match else {})

    def calibration_rows(self):
        recorder=self.session.calibration_recorder
        return recorder.rows() if recorder is not None else []

    def mark_policy_update(self,update_id,samples):
        self.session.mark_policy_update(update_id,samples)

    def process_ids(self):
        """Owned child PIDs for sampler diagnostics; never part of Actor input."""
        battle = self.session.battle
        return [] if battle is None else [battle.match.process.pid]+[
            policy.process.pid for _, _, policy in battle.policies if hasattr(policy, 'process')]
