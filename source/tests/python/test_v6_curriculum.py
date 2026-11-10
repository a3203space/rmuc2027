"""Pure state tests: no engine, training, policy prediction or physics."""
import copy
import json
import random
import unittest
from rmuc2027_research.rl import v6_curriculum as c


def config(**kw):
    return dict(enabled=True, generation='prefix5-trainingA', **kw)


def report(kind='free_ready', prepared=False, reason='navigation_stalled'):
    return dict(kind=kind, prepared=prepared, reason=reason, learning_samples=0,
                version=5, simulation_seconds=10., wall_seconds=20.)


class CurriculumTests(unittest.TestCase):
    def test_explicit_enable_and_generation(self):
        self.assertIsNone(c.normalize_config(None))
        self.assertIsNone(c.normalize_config({'enabled': False}))
        with self.assertRaises(ValueError): c.normalize_config({'enabled': True})
        with self.assertRaises(ValueError): c.normalize_config(config(failure_limit=True))
        with self.assertRaises(ValueError): c.normalize_config(config(cooldown_seconds=float('nan')))

    def test_default_probability_is_about_sixty_forty_and_reproducible(self):
        s=c.configure(None,config())
        def sequence():
            rng=random.Random(18)
            return [c.choose(s,rng,backend='rapier',side='red',now=0)[0] for _ in range(10000)]
        a=sequence();self.assertEqual(a,sequence())
        self.assertTrue(5800<a.count(None)<6200)
        self.assertEqual(s['config']['probability_basis'],'episode_lease_attempts')

    def test_two_failures_open_only_matching_kind_backend_side(self):
        s=c.configure(None,config(recovery_probability=1))
        for i in (1,2):c.record_prefix(s,lease_id=i,kind='free_ready',backend='rapier',side='red',report=report(),now=100+i)
        self.assertEqual(s['scopes']['free_ready/rapier/red']['open_until'],1902.)
        rng=random.Random(1)
        self.assertNotIn('free_ready',[c.choose(s,rng,backend='rapier',side='red',now=103)[0] for _ in range(100)])
        self.assertIn('free_ready',[c.choose(s,rng,backend='rapier',side='blue',now=103)[0] for _ in range(100)])
        self.assertIn('free_ready',[c.choose(s,rng,backend='rapier',side='red',now=1902)[0] for _ in range(100)])
        self.assertTrue(all(x['actor_samples']==0 and not x['policy_failure'] for x in s['attempts']))

    def test_all_breakers_return_normal_and_success_clears_streak(self):
        s=c.configure(None,config(recovery_probability=1,failure_limit=1))
        for i,kind in enumerate(c.KINDS):c.record_prefix(s,lease_id=i,kind=kind,backend='rapier',side='red',report=report(kind),now=0)
        self.assertEqual(c.choose(s,random.Random(2),backend='rapier',side='red',now=1),(None,'all_recovery_breakers_open'))
        c.record_prefix(s,lease_id=8,kind='free_ready',backend='rapier',side='red',report=report(prepared=True,reason='prepared'),now=1801)
        self.assertEqual(s['scopes']['free_ready/rapier/red']['consecutive_failures'],0)

    def test_prefix_report_idempotence_alias_and_conflict(self):
        s=c.configure(None,config());r=report('paid_beneficial')
        first=c.record_prefix(s,lease_id=1,kind='paid_early',backend='rapier',side='red',report=r,now=1)
        self.assertEqual(first,c.record_prefix(s,lease_id=1,kind='paid_early',backend='rapier',side='red',report=r,now=2))
        self.assertEqual(len(s['attempts']),1)
        with self.assertRaises(ValueError):c.record_prefix(s,lease_id=1,kind='paid_early',backend='rapier',side='red',report=report('paid_early',True),now=2)
        with self.assertRaises(ValueError):c.record_prefix(s,lease_id=2,kind='paid_early',backend='rapier',side='red',report={**r,'learning_samples':1},now=2)

    def test_same_generation_resume_and_new_generation_preserves_ancestor(self):
        s=c.configure(None,config());c.record_prefix(s,lease_id=1,kind='free_ready',backend='rapier',side='red',report=report(),now=1)
        restored=json.loads(json.dumps(s));self.assertEqual(c.configure(restored,config()),s)
        newer=c.configure(restored,dict(enabled=True,generation='prefix6-trainingB'))
        self.assertEqual(newer['ancestor'],s);self.assertEqual(newer['attempts'],[]);self.assertEqual(newer['scopes'],{})
        with self.assertRaises(ValueError):c.configure(s,config(recovery_probability=.5))

    def test_actual_actor_steps_separate_from_zero_sample_prefixes(self):
        s=c.configure(None,config());c.add_actor_progress(s,None,600,60.);c.add_actor_progress(s,'paid_beneficial',1200,120.)
        self.assertEqual(s['actor_steps']['normal'],600);self.assertEqual(s['actor_steps']['paid_early'],1200)
        self.assertEqual(s['attempts'],[])
        with self.assertRaises(ValueError):c.add_actor_progress(s,None,-1,0)

if __name__=='__main__':unittest.main()
