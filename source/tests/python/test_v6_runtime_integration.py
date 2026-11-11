import copy
import tempfile
import unittest
from pathlib import Path

from rmuc2027_research.rl.v6_sampling import TrainingCoordinator, RECOVERY_KINDS
from rmuc2027_research.rl.v6_rewards import RecoveryAccounting
from rmuc2027_research.rl.v6_evaluation import required_cases, select_candidate
from rmuc2027_research.rl.session import load_config


class RuntimeTests(unittest.TestCase):
    def test_coordinator_leases_progress_and_resume_do_not_invent_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            a=TrainingCoordinator(Path(directory)/'samplers.json',91)
            first=a.acquire(0);second=a.acquire(1)
            self.assertNotEqual(first['scenario_id'],second['scenario_id'])
            a.progress(0,first['id'],valid_steps=12,seconds=1.2)
            a.progress(0,first['id'],valid_steps=12,seconds=1.2)
            snapshot=a.snapshot()
            self.assertEqual(snapshot['counts'][first['scenario_id']]['valid_steps'],12)
            a.release_interrupted_leases()
            restored=TrainingCoordinator(a.path).snapshot()
            self.assertFalse(restored['leases'])
            self.assertEqual(restored['coverage']['short_completed']['scenarios'],0)
            self.assertEqual(restored['coverage']['valid_steps']['scenarios'],1)

    def test_recovery_curriculum_is_not_permanently_tied_to_blue(self):
        with tempfile.TemporaryDirectory() as directory:
            scheduler=TrainingCoordinator(Path(directory)/'samplers.json',92)
            sides={kind:set() for kind in RECOVERY_KINDS}
            for _ in range(16):
                job=scheduler.acquire(0)
                if job['recovery_kind']:sides[job['recovery_kind']].add(job['scenario']['learner_side'])
                scheduler.progress(0,job['id'],valid_steps=0,seconds=0,close=True)
            self.assertTrue(all(value=={'red','blue'} for value in sides.values()))

    def test_reward_cost_uses_actual_own_event_once_and_no_suicide_profit(self):
        robot=dict(id='red-7',alive=False,weak=True)
        tracker=RecoveryAccounting(robot)
        revived={**robot,'alive':True}
        frame=dict(tick=100,time=5.,events=[dict(type='respawn',robot='red-7',instant=True,cost=180,hp=400)])
        self.assertAlmostEqual(tracker.advance(robot,revived,frame,.05)['paid_coins'],-.18)
        self.assertEqual(tracker.advance(revived,revived,frame,.05)['paid_coins'],0)
        frame['events']=[dict(type='respawn',robot='red-3',instant=True,cost=900)]
        self.assertEqual(tracker.advance(revived,revived,frame,.05)['paid_coins'],0)
        active={**revived,'weak':False}
        rewards=[tracker.advance(active,active,dict(frame,events=[]),.05)['recovery'] for _ in range(60)]
        self.assertAlmostEqual(sum(rewards),.2)
        self.assertLessEqual(-1+sum(rewards)-tracker.paid_coins*.001,-.8)
        self.assertEqual(tracker.paid_revives,1)

    def test_v6_configuration_and_incomplete_manifest_never_promote(self):
        cfg=load_config(dict(policy_version=6,slot=7,navigation_mode='adaptive',rl_seconds=.1))
        self.assertEqual(cfg['rewards']['death'],-1)
        self.assertEqual(cfg['rewards']['capture_seconds'],.02)
        self.assertEqual(len(required_cases()),8)
        with tempfile.TemporaryDirectory() as directory:
            result=select_candidate(directory)
            self.assertFalse(result['v6_paired_improvement_verified'])
            self.assertEqual(len(result['missing']),16)


if __name__=='__main__':unittest.main()
