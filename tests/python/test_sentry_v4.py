import copy
import math
import unittest

import numpy as np
import gymnasium as gym
import torch

from rmuc2027_core.engine import Match
from rmuc2027_research.rl import sentry_v3, sentry_v4
from rmuc2027_research.rl.sentry_v4_policy import SentryV4Policy, export_actor


class SentryV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with Match(observation_profile=sentry_v3.PROFILE) as match:cls.obs=match.observation('red',[7])

    def test_registry_prefix_and_skill_information_boundary(self):
        a=sentry_v4.encode(self.obs,{})
        self.assertEqual(len(a),739);self.assertEqual(len(sentry_v4.SKILL_INDICES),37)
        self.assertEqual(a[:393],sentry_v3.encode(self.obs,{}))
        changed=copy.deepcopy(self.obs);changed['remaining']=1
        for source in changed['referee']['sources'].values():source['age']=.7
        changed['radar_inference']['team']={'advantage':1,'confidence':1,'valid':True}
        b=sentry_v4.encode(changed,{})
        self.assertNotEqual(a,b)
        self.assertEqual([a[i] for i in sentry_v4.SKILL_INDICES],[b[i] for i in sentry_v4.SKILL_INDICES])
        self.assertTrue(all(math.isfinite(x) for x in a))

    def test_stale_peer_name_does_not_create_current_wait_feedback(self):
        a=sentry_v4.local_feedback(self.obs,{'navigation':{'wait_for_peer':'red-3'}})
        self.assertFalse(a['navigation.waiting_peer'])

    def test_inference_age_v2_mask_bound_and_deterministic_v1_migration(self):
        obs=copy.deepcopy(self.obs)
        obs['radar_inference']=dict(enemies={'3':dict(age=999.,speed_valid=False),
            '4':dict(age=30.,speed_valid=True)},team=dict(age=999.,valid=False),
            robots={'ally':{'7':dict(pressure_age=999.,pressure_valid=False,
                contribution_age=8.,contribution_valid=True,local_age=40.,local_valid=True)}})
        encoded=sentry_v4.encode(obs,{})
        checks={'estimate.enemy.3.age':0.,'estimate.enemy.4.age':1.,'estimate.team.age':0.,
            'estimate.ally.7.pressure_age':0.,'estimate.ally.7.contribution_age':.8,'estimate.ally.7.local_age':1.}
        for name,expected in checks.items():self.assertAlmostEqual(encoded[sentry_v4.FEATURE_INDEX[name]],expected)
        legacy=list(encoded)
        for name,value in {'estimate.enemy.3.age':99.9,'estimate.enemy.4.age':3.,'estimate.team.age':99.9,
            'estimate.ally.7.pressure_age':99.9,'estimate.ally.7.local_age':4.}.items():
            legacy[sentry_v4.FEATURE_INDEX[name]]=value
        migrated=sentry_v4.migrate_encoding_v1(legacy)
        self.assertEqual(migrated,encoded)
        self.assertEqual(migrated[:393],legacy[:393])
        self.assertEqual([migrated[i] for i in sentry_v4.SKILL_INDICES],[legacy[i] for i in sentry_v4.SKILL_INDICES])
        self.assertEqual(sentry_v4.normalize_inference_ages(list(migrated)),migrated)
        self.assertEqual(sentry_v4.schema()['encoding_version'],2)
        with self.assertRaises(ValueError):sentry_v4.migrate_encoding_v1(legacy,'wrong-hash')

    def test_prediction_blue_canonical_coordinates(self):
        o=copy.deepcopy(self.obs);o['team']='blue'
        o['radar_inference']['predictions']={'3':{'future':{'1':{'x':20,'y':10,'valid':True,'confidence':.5}}}}
        features=sentry_v4.encode(o,{})
        self.assertAlmostEqual(features[sentry_v4.FEATURE_INDEX['prediction.enemy.3.future.1.x']],8/28)
        self.assertAlmostEqual(features[sentry_v4.FEATURE_INDEX['prediction.enemy.3.future.1.y']],5/15)

    def test_trace_distinguishes_legacy_threshold_latch_and_effective_transform(self):
        obs=copy.deepcopy(self.obs);robot=obs['robots'][0]
        robot.update(model_slot=2773,x=3.,y=7.,vx=0.,vy=0.,transform_progress=0.)
        obs['arena']=dict(obs['arena'],obstacles=[],terrain=[])
        state=dict(goal=[3.,7.],sentry_intent=dict(fold=.4,jump=-1.,spin=-1.,fire=-1.))
        sentry_v4.execute(obs,state,None,{})
        trace=state['execution_trace']
        self.assertEqual(trace['execution_trace_version'],2)
        self.assertFalse(trace['requested']['fold'])  # Preserved v3 >.5 schema.
        self.assertEqual(trace['decoded_fold_request'],1)  # Executor >.3.
        self.assertTrue(trace['fold_latched']);self.assertTrue(trace['effective_transform'])
        self.assertEqual(trace['executed']['fold'],0.)  # Body has not moved yet.


class SentryV4PolicyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4);torch.set_num_threads(1)
        obs=gym.spaces.Dict({'actor':gym.spaces.Box(-np.inf,np.inf,(739,),np.float32),
            'critic':gym.spaces.Box(-np.inf,np.inf,(17,),np.float32)})
        self.policy=SentryV4Policy(obs,gym.spaces.Box(-1,1,(6,),np.float32),lambda _:1e-4)

    def test_no_critic_to_actor_or_tactical_to_skill_connection(self):
        actor=torch.randn(2,739);critic=torch.randn(2,17,requires_grad=True)
        mean=self.policy.get_distribution({'actor':actor,'critic':critic}).distribution.mean
        other=self.policy.get_distribution({'actor':actor,'critic':critic*100}).distribution.mean
        self.assertTrue(torch.equal(mean,other))
        grad=torch.autograd.grad(mean.sum(),critic,allow_unused=True)[0]
        self.assertTrue(grad is None or torch.count_nonzero(grad)==0)
        changed=actor.clone();keep=set(sentry_v4.SKILL_INDICES)
        for i in range(739):
            if i not in keep:changed[:,i]+=100
        after=self.policy.get_distribution({'actor':changed,'critic':critic}).distribution.mean
        self.assertTrue(torch.equal(mean[:,2:4],after[:,2:4]))

    def test_parameter_ownership_and_critic_gradient_isolation(self):
        actor={id(p) for p in self.policy.actor_parameters()};critic={id(p) for p in self.policy.critic_parameters()}
        self.assertFalse(actor & critic)
        self.assertEqual(actor | critic,{id(p) for p in self.policy.parameters()})
        value=self.policy.predict_values({'actor':torch.randn(2,739),'critic':torch.randn(2,17)})
        value.sum().backward()
        self.assertTrue(all(p.grad is None for p in self.policy.actor_parameters()))
        self.assertTrue(any(p.grad is not None for p in self.policy.critic_parameters()))

    def test_portable_actor_export_parity_and_no_critic_weights(self):
        x=torch.randn(1,739);mean=self.policy.get_distribution({'actor':x,'critic':torch.randn(1,17)}).distribution.mean
        artifact=export_actor(self.policy)
        self.assertEqual(set(artifact),{'version','schema_hash','skill_indices','architecture','tactical','skill'})
        self.assertEqual(artifact['architecture'],dict(tactical_hidden=[96,96],skill_hidden=[64,64]))
        output=sentry_v4.portable_forward(x[0].tolist(),artifact)
        np.testing.assert_allclose(output,mean[0].detach().numpy(),atol=2e-6,rtol=2e-6)


if __name__=='__main__':unittest.main()
