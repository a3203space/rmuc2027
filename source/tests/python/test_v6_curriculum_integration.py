"""State-only integration tests; no engine process, training or physics."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rmuc2027_research.rl import env as envmod
from rmuc2027_research.rl import v6_sampling as sampler
from rmuc2027_research.rl.v6_recovery import RecoveryPreparationError


class IntegrationTests(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
 def coordinator(self):return sampler.TrainingCoordinator(Path(self.tmp.name)/'coordinator.json',13)
 def settings(self):return dict(enabled=True,generation='prefix5-A',recovery_probability=1,failure_limit=2,cooldown_seconds=1800)
 def fail(self,c,worker):
  lease=c.acquire(worker);self.assertIsNotNone(lease['recovery_kind'])
  c.progress(worker,lease['id'],valid_steps=0,seconds=0,close=True,reason='interrupted')
  r=dict(kind=lease['recovery_kind'],prepared=False,reason='navigation_stalled',learning_samples=0,simulation_seconds=10,wall_seconds=20,version=5)
  c.report_prefix(worker,r,now=100,log_path='full-failure.json');return lease,r
 def test_default_unchanged_even_when_curriculum_state_exists(self):
  c=self.coordinator();self.assertIsNone(c.acquire(0)['recovery_kind']);c.release_interrupted_leases()
  c.configure_curriculum(self.settings(),backend='rapier');c2=self.coordinator()
  self.assertEqual(c2.acquire(0)['recovery_kind'],'free_ready')
 def test_failed_closed_lease_audited_without_policy_failure_and_normal_fallback(self):
  c=self.coordinator();c.configure_curriculum(self.settings(),backend='rapier');lease,r=self.fail(c,0)
  c.request_normal_fallback(0);next_=c.acquire(0);self.assertIsNone(next_['recovery_kind'])
  state=c.snapshot();self.assertEqual(state['history'][-1]['reason'],'prefix_not_prepared:navigation_stalled')
  self.assertFalse(state['history'][-1]['failure']);self.assertEqual(state['counts'][lease['scenario_id']]['valid_steps'],0)
  self.assertEqual(state['curriculum']['attempts'][0]['actor_samples'],0)
 def test_same_generation_restores_and_other_generation_requires_no_active_leases(self):
  c=self.coordinator();c.configure_curriculum(self.settings(),backend='rapier');self.fail(c,0)
  d=self.coordinator();d.configure_curriculum(self.settings(),backend='rapier');self.assertEqual(len(d.snapshot()['curriculum']['attempts']),1)
  d.acquire(0)
  with self.assertRaises(ValueError):d.configure_curriculum({**self.settings(),'generation':'new'},backend='rapier')
 def test_progress_delta_and_report_idempotent(self):
  c=self.coordinator();c.configure_curriculum(self.settings(),backend='rapier');lease=c.acquire(0)
  r=dict(kind=lease['recovery_kind'],prepared=True,reason='prepared',learning_samples=0)
  c.report_prefix(0,r,now=0);c.report_prefix(0,r,now=1)
  for _ in range(2):c.progress(0,lease['id'],valid_steps=10,seconds=1)
  s=c.snapshot();self.assertEqual(sum(s['curriculum']['actor_steps'].values()),10);self.assertEqual(len(s['curriculum']['attempts']),1)
 def test_env_reports_failed_prefix_after_session_closed_it_then_normal_reset(self):
  c=self.coordinator();settings=self.settings()
  class FakeSession:
   def __init__(self):
    self.config=dict(policy_version=6,seed=13,_worker_id=0,sim_backend='rapier',v6_recovery_curriculum=settings,_episode_log_dir=self_tmp)
    self.v6_coordinator=c;self.calls=0
   def reset(self,*args,**kw):
    self.calls+=1;lease=c.acquire(0)
    if self.calls==1:
     c.progress(0,lease['id'],valid_steps=0,seconds=0,close=True,reason='interrupted')
     raise RecoveryPreparationError(dict(kind=lease['recovery_kind'],prepared=False,reason='navigation_stalled',learning_samples=0))
    assert lease['recovery_kind'] is None
    return [0.],{'recovery_prefix':None}
   def close(self):pass
  self_tmp=self.tmp.name;fake=FakeSession()
  with patch.object(envmod,'ArenaSession',return_value=fake),patch.object(envmod,'policy_schema',return_value={'version':6,'obs_size':775}):
   env=envmod.SentryEnv();obs,info=env.reset(seed=3)
  self.assertEqual(fake.calls,2);self.assertEqual(len(c.snapshot()['curriculum']['attempts']),1)
  self.assertEqual(len(list(Path(self.tmp.name).glob('prefix-failed-*.json'))),1)

if __name__=='__main__':unittest.main()
