"""Toy optimizer/rollout checks only; no match simulation or research training."""
import copy
from pathlib import Path
import tempfile
import types
import unittest

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from rmuc2027_research.rl.asymmetric_ppo import AsymmetricPPO
from rmuc2027_research.rl.sentry_v4 import SKILL_INDICES, schema
from rmuc2027_research.rl.sentry_v4_policy import SentryV4Policy
from rmuc2027_research.rl.v4_training import save_state, restore_state


class ToyEnv(gym.Env):
    def __init__(self, terminated=False):
        self.observation_space=gym.spaces.Dict(dict(
            actor=gym.spaces.Box(-np.inf,np.inf,(schema()['obs_size'],),np.float32),
            critic=gym.spaces.Box(-np.inf,np.inf,(4,),np.float32)))
        self.action_space=gym.spaces.Box(-1,1,(6,),np.float32)
        self.terminated=terminated;self.t=0;self.restored=None

    def _obs(self):
        return dict(actor=np.zeros(schema()['obs_size'],np.float32),
                    critic=np.array([self.t,2*self.t,0,1],np.float32))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed);self.t=0
        return self._obs(),{}

    def step(self, action):
        self.t+=1;done=self.t==2
        return self._obs(),1.,done and self.terminated,done and not self.terminated,{}

    def sampler_state(self):
        return dict(session=dict(episode_index=9),gym_rng=copy.deepcopy(self.np_random.bit_generator.state))

    def restore_sampler_state(self, value):
        self.restored=copy.deepcopy(value)

    def calibration_rows(self):
        return []

    def execution_state(self):
        return dict(reset_on_resume=True)


def model(terminated=False):
    vec=DummyVecEnv([lambda:ToyEnv(terminated)])
    result=AsymmetricPPO(SentryV4Policy,vec,n_steps=4,batch_size=4,n_epochs=1,
        learning_rate=1e-3,gamma=.9,gae_lambda=.95,normalize_advantage=False,
        max_grad_norm=.1,ent_coef=.01,vf_coef=.5,seed=17,device='cpu',verbose=0)
    result.set_logger(configure(folder=None,format_strings=[]))
    return result,vec


def one_update(algorithm, value_target=1.):
    """Same fixed actor inputs, actions, old logp and advantages for both runs."""
    rng=np.random.default_rng(87);buffer=algorithm.rollout_buffer;buffer.reset()
    for index in range(4):
        obs=dict(actor=rng.normal(0,.1,(1,schema()['obs_size'])).astype(np.float32),
                 critic=rng.normal(size=(1,4)).astype(np.float32))
        tensor={k:torch.as_tensor(v) for k,v in obs.items()}
        actions=np.full((1,6),.2,np.float32)
        with torch.no_grad():
            values,logp,_=algorithm.policy.evaluate_actions(tensor,torch.as_tensor(actions))
        buffer.add(obs,actions,np.ones(1,np.float32),np.array([index==0]),values,logp)
    buffer.compute_returns_and_advantage(torch.zeros(1),np.array([False]))
    buffer.advantages[:]=np.array([.4,.8,1.2,1.6],np.float32).reshape(4,1)
    buffer.returns[:]=value_target
    algorithm.num_timesteps=4;algorithm._current_progress_remaining=1.
    np.random.seed(123);algorithm.train()


class AsymmetricTrainingTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_huge_critic_gradient_cannot_change_actor_update_or_clip_norm(self):
        first,vec1=model();second,vec2=model()
        try:
            before=[p.detach().clone() for p in first.policy.actor_parameters()]
            one_update(first,1.);one_update(second,1e9)
            for left,right in zip(first.policy.actor_parameters(),second.policy.actor_parameters()):
                torch.testing.assert_close(left,right,rtol=0,atol=0)
            self.assertTrue(any(not torch.equal(old,new) for old,new in zip(before,first.policy.actor_parameters())))
            self.assertEqual(first.optimizer_steps,1)
            self.assertEqual(first.optimized_samples,4)
            self.assertGreater(second.logger.name_to_value['train/value_loss'],1e16)
        finally:vec1.close();vec2.close()

    def test_both_optimizer_moments_and_sampler_rng_survive_save_restore(self):
        original,vec=model();restored=None
        try:
            one_update(original)
            with tempfile.TemporaryDirectory() as directory:
                save_state(original,vec,directory)
                expected_numpy=np.random.random()
                restored=AsymmetricPPO.load(Path(directory)/'model.zip',env=vec,device='cpu')
                restored.set_logger(configure(folder=None,format_strings=[]))
                for name in ('actor_optimizer','critic_optimizer'):
                    old=getattr(original,name).state_dict();new=getattr(restored,name).state_dict()
                    self.assertTrue(old['state']);self.assertEqual(old['param_groups'],new['param_groups'])
                    self.assertEqual(set(old['state']),set(new['state']))
                    for parameter,state in old['state'].items():
                        for key,value in state.items():
                            if isinstance(value,torch.Tensor):torch.testing.assert_close(value,new['state'][parameter][key],rtol=0,atol=0)
                            else:self.assertEqual(value,new['state'][parameter][key])
                state=restore_state(directory,vec)
                self.assertEqual(np.random.random(),expected_numpy)
                self.assertEqual(state['resume_boundary'],'new_episode_after_completed_update')
                self.assertEqual(state['partial_rollout_discarded'],0)
                self.assertEqual(vec.envs[0].restored['session']['episode_index'],9)
                self.assertEqual(restored.num_timesteps,4);self.assertEqual(restored.optimizer_steps,1)
                # The next equal update must also agree, not merely saved tensors.
                one_update(original);one_update(restored)
                for left,right in zip(original.policy.parameters(),restored.policy.parameters()):
                    torch.testing.assert_close(left,right,rtol=0,atol=0)
        finally:vec.close()

    def test_dict_timeout_bootstraps_terminal_critic_state_not_reset_state(self):
        for terminated in (False,True):
            algorithm,vec=model(terminated)
            try:
                calls=[]
                def predict_values(policy,obs):
                    calls.extend(obs['critic'][:,0].detach().cpu().tolist())
                    return obs['critic'][:,:1]
                algorithm.policy.predict_values=types.MethodType(predict_values,algorithm.policy)
                _,callback=algorithm._setup_learn(4,callback=lambda locals_,globals_:True)
                callback.on_training_start({}, {})
                success=algorithm.collect_rollouts(vec,callback,algorithm.rollout_buffer,n_rollout_steps=4)
                self.assertTrue(success)
                expected=[1.,1. if terminated else 2.8,1.,1. if terminated else 2.8]
                np.testing.assert_allclose(algorithm.rollout_buffer.rewards[:,0],expected,rtol=1e-6)
                if not terminated:self.assertEqual(calls.count(2.),2)
                else:self.assertNotIn(2.,calls)
                self.assertEqual(algorithm.num_timesteps,4)
                self.assertFalse(hasattr(algorithm,'optimizer_steps'))
            finally:vec.close()

    def test_private_critic_and_tactical_history_cannot_reach_skill_branch(self):
        algorithm,vec=model()
        try:
            actor=torch.randn(3,schema()['obs_size']);critic=torch.randn(3,4)
            initial=algorithm.policy.get_distribution(dict(actor=actor,critic=critic)).distribution.mean
            private_changed=algorithm.policy.get_distribution(dict(actor=actor,critic=critic*1e8)).distribution.mean
            torch.testing.assert_close(initial,private_changed,rtol=0,atol=0)
            history_changed=actor.clone();keep=set(SKILL_INDICES)
            history_changed[:,[i for i in range(actor.shape[1]) if i not in keep]]+=100.
            after=algorithm.policy.get_distribution(dict(actor=history_changed,critic=critic*1e8)).distribution.mean
            torch.testing.assert_close(initial[:,2:4],after[:,2:4],rtol=0,atol=0)
        finally:vec.close()

    def test_kl_rejection_does_not_count_optimizer_steps_and_guidance_is_rejected(self):
        algorithm,vec=model()
        try:
            one_update(algorithm)
            prior_steps=algorithm.optimizer_steps;prior_samples=algorithm.optimized_samples
            algorithm.target_kl=1e-6;algorithm.rollout_buffer.log_probs+=10.
            algorithm.train()
            self.assertEqual(algorithm.optimizer_steps,prior_steps)
            self.assertEqual(algorithm.optimized_samples,prior_samples)
            algorithm.sentry_guidance_path='forbidden-hidden-demonstrations.npz'
            with self.assertRaisesRegex(ValueError,'demonstration'):algorithm.train()
        finally:vec.close()


if __name__=='__main__':unittest.main()
