"""Full rule/policy sampler parity including independent episode resets."""
import unittest
from unittest.mock import patch
import numpy as np
from rmuc2027_research.rl.env import SentryEnv
from rmuc2027_research.rl.learning import vector_env
from rmuc2027_research.rl.session import load_config
from rmuc2027_research.rl.warp_kinematic import batching


class VectorIntegrationTests(unittest.TestCase):
    def test_batched_rule_rollouts_match_independent_sessions_through_reset(self):
        config=load_config('configs/rl/sentry-mjwarp-hour.json')
        config.update(sim_backend='warp_kinematic',episode_seconds=2.)
        solo=[SentryEnv(config) for _ in range(2)]
        # Solo adapters must be created before a vector service becomes active.
        expected=[env.reset(seed=config['seed']+i)[0] for i,env in enumerate(solo)]
        vec=vector_env(config,2)
        try:
            np.testing.assert_array_equal(vec.reset(),expected)
            rng=np.random.default_rng(617)
            resets=0
            for _ in range(9):
                actions=rng.uniform(-1,1,(2,2)).astype(np.float32)
                obs,rewards,dones,infos=vec.step(actions)
                for i,env in enumerate(solo):
                    ref,reward,term,trunc,info=env.step(actions[i])
                    self.assertEqual(dones[i],term or trunc)
                    self.assertEqual(infos[i]['sampled_simulation_seconds'],info['sampled_simulation_seconds'])
                    self.assertAlmostEqual(rewards[i],reward,places=6)
                    if term or trunc:
                        np.testing.assert_array_equal(infos[i]['terminal_observation'],ref)
                        with patch.object(batching,'active_service',None):
                            ref,_=env.reset()
                        resets+=1
                    np.testing.assert_array_equal(obs[i],ref)
            self.assertGreaterEqual(resets,4)
            self.assertEqual(vec.physics.max_batch,2)
            owned=[pid for env in vec.envs for pid in env.process_ids()]
        finally:
            vec.close()
            for env in solo:env.close()
        self.assertIsNone(batching.active_service)
        self.assertFalse(vec.physics.thread.is_alive())
        from pathlib import Path
        self.assertTrue(all(not Path(f'/proc/{pid}').exists() for pid in owned))

    def test_sharded_vector_env_matches_independent_sessions_and_cleans_up(self):
        from pathlib import Path
        from rmuc2027_research.rl.warp_kinematic.vector import ShardedWarpVecEnv
        config=load_config('configs/rl/sentry-mjwarp-hour.json')
        config.update(sim_backend='warp_kinematic',episode_seconds=2.)
        with self.assertRaises(ValueError):
            vector_env(config,2,shards=0)
        solo=[SentryEnv(config) for _ in range(4)]
        expected=[env.reset(seed=config['seed']+i)[0] for i,env in enumerate(solo)]
        vec=vector_env(config,4,shards=2)
        try:
            self.assertIsInstance(vec,ShardedWarpVecEnv)
            self.assertEqual(vec.num_shards,2)
            self.assertEqual(len(vec.processes),2)
            np.testing.assert_array_equal(vec.reset(),expected)
            rng=np.random.default_rng(617)
            resets=0
            for _ in range(9):
                actions=rng.uniform(-1,1,(4,2)).astype(np.float32)
                obs,rewards,dones,infos=vec.step(actions)
                for i,env in enumerate(solo):
                    ref,reward,term,trunc,info=env.step(actions[i])
                    self.assertEqual(dones[i],term or trunc)
                    self.assertEqual(infos[i]['sampled_simulation_seconds'],info['sampled_simulation_seconds'])
                    self.assertAlmostEqual(rewards[i],reward,places=6)
                    if term or trunc:
                        np.testing.assert_array_equal(infos[i]['terminal_observation'],ref)
                        ref,_=env.reset()
                        resets+=1
                    np.testing.assert_array_equal(obs[i],ref)
            self.assertGreaterEqual(resets,8)
            self.assertEqual(vec.physics.max_batch,2)
            self.assertGreater(vec.physics.world_steps,0)
            self.assertEqual(len(vec.get_attr('action_space')),4)
            shard_pids=[p.pid for p in vec.processes]
            owned=vec.process_ids()
            self.assertGreater(len(owned),0)
        finally:
            vec.close()
            for env in solo:env.close()
        self.assertIsNone(batching.active_service)
        self.assertTrue(all(not Path(f'/proc/{pid}').exists() for pid in shard_pids+owned))
