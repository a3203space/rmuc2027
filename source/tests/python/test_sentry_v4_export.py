"""Standalone v4 smoke with random weights, no optimizer steps or training."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import struct
import tempfile
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

from rmuc2027_core import sdk
from rmuc2027_core.engine import Match
from rmuc2027_core.policy import Policy
from rmuc2027_core.worker import SOURCE_LIMIT
from rmuc2027_core.submissions import load_submission
from rmuc2027_research.rl import critic_state, sentry_v4
from rmuc2027_research.rl.asymmetric_ppo import AsymmetricPPO
from rmuc2027_research.rl.exporting import export_model, _pack_v4_actor
from rmuc2027_research.rl.learning import experiment
from rmuc2027_research.rl.sentry_v4_policy import SentryV4Policy, export_actor
from rmuc2027_research.rl.v4_training import configuration


class SpacesOnlyEnv(gym.Env):
    observation_space=gym.spaces.Dict(dict(
        actor=gym.spaces.Box(-np.inf,np.inf,(sentry_v4.schema()['obs_size'],),np.float32),
        critic=gym.spaces.Box(-np.inf,np.inf,(critic_state.SIZE,),np.float32)))
    action_space=gym.spaces.Box(-1,1,(6,),np.float32)

    def reset(self, *, seed=None, options=None):
        raise AssertionError('Export smoke must not collect training data')

    def step(self, action):
        raise AssertionError('Export smoke must not train')


class StandaloneV4ExportTests(unittest.TestCase):
    def test_random_actor_standalone_torch_parity_and_native_policy_steps(self):
        torch.set_num_threads(1)
        config=configuration();config.update(sim_backend='rapier',scenario_sampling=False,
            capability_sampling=False,training_summary=False)
        model=AsymmetricPPO(SentryV4Policy,SpacesOnlyEnv(),device='cpu',seed=73,verbose=0,**config['ppo'])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);checkpoint=root/'model.zip'
            model.save(checkpoint)
            (root/'experiment.json').write_text(json.dumps(experiment(config)))
            metadata=export_model(checkpoint,root/'submission')
            self.assertLess(metadata['actor_parity_max_error'],1e-5)
            _,entries=load_submission(root/'submission')
            self.assertLessEqual(entries['7'].stat().st_size,SOURCE_LIMIT)
            self.assertGreater(SOURCE_LIMIT-entries['7'].stat().st_size,30*1024)
            self.assertEqual(metadata['actor_architecture'],dict(tactical_hidden=[96,96],skill_hidden=[64,64]))
            namespace={'__name__':'standalone_v4_export_smoke','__package__':''}
            # In-process inspect of the exact emitted source; isolated Policy
            # below independently receives only its copied public SDK/runtime.
            with patch.dict(sys.modules,{'sdk':sdk}):
                exec(compile(entries['7'].read_text(),str(entries['7']),'exec'),namespace)
            features=np.random.default_rng(901).normal(0,.3,(8,sentry_v4.schema()['obs_size'])).astype(np.float32)
            with torch.no_grad():expected=model.policy.mlp_extractor.policy_net(torch.as_tensor(features)).numpy()
            actual=np.asarray([namespace['mlp_forward'](row,namespace['_LAYERS']) for row in features])
            np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-5)
            self.assertEqual(namespace['sentry_v4'].SCHEMA_HASH,sentry_v4.SCHEMA_HASH)
            self.assertEqual(namespace['_LAYERS'],export_actor(model.policy))
            restored=[v for name in ('tactical','skill') for layer in namespace['_LAYERS'][name]
                for v in [*[x for row in layer['weight'] for x in row],*layer['bias']]]
            self.assertEqual(hashlib.sha256(struct.pack('<%df'%len(restored),*restored)).hexdigest(),metadata['actor_float32_sha256'])
            invalid=copy.deepcopy(namespace['_LAYERS']);invalid['skill'][0]['bias'][0]=float('nan')
            with self.assertRaisesRegex(ValueError,'finite'):_pack_v4_actor(invalid)
            lineups={side:[dict(slot=int(slot),model=number) for slot,number in lineup.items()]
                for side,lineup in config['lineups'].items()}
            for team in ('red','blue'):
                with Match(seed=73,lineups=lineups,observation_profile=sentry_v4.PROFILE) as match:
                    policy=Policy(entries['7'],73,'trusted-local',5.)
                    state={}
                    try:
                        for tick in range(8):
                            obs=match.observation(team,[7])
                            obs.update(navigation_mode='adaptive',policy_dt=.05,motion_trace=True)
                            actor_features=namespace['encode_observation'](obs,state,7,.1,config)
                            with torch.no_grad():torch_action=model.policy.mlp_extractor.policy_net(torch.tensor([actor_features],dtype=torch.float32))[0].numpy()
                            pure_action=namespace['mlp_forward'](actor_features,namespace['_LAYERS'])
                            np.testing.assert_allclose(pure_action,torch_action,atol=1e-6,rtol=1e-5)
                            expected_action,state=namespace['act'](copy.deepcopy(obs),state)
                            action,error=policy.decide(obs)
                            self.assertIsNone(error, f'{team} tick {tick}: {error}; {list(policy.logs)}')
                            # JSON transport turns SDK tuples into lists.
                            self.assertEqual(action,json.loads(json.dumps(expected_action)))
                            self.assertEqual(set(action),{'7'})
                            self.assertIn('navigation',policy.navigation_trace)
                            self.assertEqual(state['execution_trace']['execution_trace_version'],2)
                            match.step({team:action})
                        self.assertEqual(policy.errors,0);self.assertEqual(policy.calls,8)
                    finally:policy.close()
            self.assertEqual(model.num_timesteps,0);self.assertEqual(model._n_updates,0)
            self.assertFalse(model.actor_optimizer.state);self.assertFalse(model.critic_optimizer.state)


if __name__=='__main__':unittest.main()
