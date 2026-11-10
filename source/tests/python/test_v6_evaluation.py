"""Finite paired-evidence selection with synthetic records; no simulation."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from rmuc2027_research.rl import v6_evaluation as evaluation


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    for name in ('initialized.zip', 'model.zip'):
        path=root/'main'/name;path.parent.mkdir(exist_ok=True);path.write_bytes(name.encode())
    helpers={}
    for role in ('infantry','heavy'):
        path=root/(role+'.json');path.write_text(role);helpers[role]=str(path)
    radar=root/'radar.json';radar.write_text('radar')
    write(root/'main/experiment.json',dict(config=dict(auxiliary_pool=helpers,radar_classifier_path=str(radar))))
    helper_hashes={key:evaluation._file_hash(path) for key,path in helpers.items()}
    radar_hash=evaluation._file_hash(radar)
    for side in ('red','blue'):write(root/'capability-profiles'/f'evaluation-{side}.json',dict(side=side))
    write(root/'main/training.json',dict(accepted_actor_steps=1,skill_unchanged=True,
        published_accepted_actor_steps=1,published_skill_unchanged=True,
        published_checkpoint_sha256=evaluation._file_hash(root/'main/model.zip')))
    for case in evaluation.required_cases():
        for member in ('baseline','candidate'):
            origin=10. if case['kind']=='recovery' else 0.
            n=round(case['seconds']/.1)
            path=root/'evaluation'/(case['id']+'-'+member+'.jsonl');path.parent.mkdir(exist_ok=True)
            geometry=dict(id='roof',center=[10.,7.5],axis=[1.,0.],half=1.,width=2.)
            points=[dict(time=origin+(i+1)*.1,position=[8.+min(i,40)*.1,7.5,.2],fold=1.,
                         navigation=dict(passage=geometry)) for i in range(n)]
            path.write_text(''.join(json.dumps(point)+'\n' for point in points))
            config=dict(evaluation=True,policy_version=6,team=case['side'],sim_backend=case['backend'],
                scenario_sampling=False,capability_sampling=False,decision_window_seconds=case['seconds'],rl_seconds=.1)
            sources={key:evaluation._digest(key) for key in ('sdk_hash','runtime_hash','engine_binary_sha256','rules_sha256')}
            sources.update(sentry_sources={'sentry_v6.py':evaluation._digest('v6')},
                simulation=dict(backend=case['backend']),sources=dict(friendly='frozen',opponent='frozen'))
            identity=dict(config_hash=evaluation._digest(config),execution_config=config,
                execution_hash=evaluation._digest(sources),sources=sources,helper_hashes=helper_hashes,radar_hash=radar_hash)
            kind='paid_beneficial' if case['recovery_kind']=='paid_early' else case['recovery_kind']
            prefix_identity=dict(commands_hash=evaluation._digest(case['id']),handoff_hash=evaluation._digest(case['id']+'frame'),
                legal_history_hash=evaluation._digest('legal') if kind else None,origin=origin,prefix_kind=kind,prepared=True)
            recovery=dict(free_revives=1,paid_revives=0,paid_coins=0,recoveries=1,
                          death_episodes=[dict(observed_mode='free',revived_at=origin+12,initial_hp=30,cost=0)])
            if kind=='paid_beneficial' and member=='candidate':
                recovery.update(free_revives=0,paid_revives=1,paid_coins=100,
                    death_episodes=[dict(observed_mode='paid',revived_at=origin+2,initial_hp=300,cost=100)])
            checkpoint=root/'main'/('initialized.zip' if member=='baseline' else 'model.zip')
            row=dict(case_id=case['id'],member=member,backend=case['backend'],side=case['side'],seed=case['seed'],
                required_seconds=case['seconds'],scenario_hash=evaluation._digest(case['scenario']),
                prepared=True,horizon_completed=True,censored=False,checkpoint_sha256=evaluation._file_hash(checkpoint),
                capabilities_sha256=evaluation._file_hash(root/'capability-profiles'/f"evaluation-{case['side']}.json"),
                identity=identity,prefix_identity=prefix_identity,prefix=dict(prepared=True,kind=kind) if kind else None,
                learner_start_time=origin,simulation_seconds=origin+case['seconds'],decision_seconds=case['seconds'],
                actor_legality=dict(enforcement='session.select_goal',invalid_decisions=0,checked_decisions=n),
                trace=dict(sha256=evaluation._file_hash(path),frames=n),passed_tunnel=True,
                metrics=dict(capture_seconds=7 if member=='candidate' else 6,tipped_seconds=0,policy_errors=0),
                control={case['side']:11 if member=='candidate' else 10,'blue' if case['side']=='red' else 'red':0},
                recovery=recovery,recovery_delay=8 if member=='candidate' else 10,capture_after_recovery=5,
                recording_complete=False,execution_mask_fallbacks=0)
            write(path.with_suffix('.json'),row)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.directory=TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name);fixture(self.root)

    def mutate(self, case, change, member='candidate'):
        path=self.root/'evaluation'/(case+'-'+member+'.json')
        row=json.loads(path.read_text());change(row);write(path,row)

    def result(self):return evaluation.select_candidate(self.root)

    def test_all_sixteen_complete_matched_trials_and_real_goals_promote(self):
        result=self.result()
        self.assertTrue(result['v6_paired_improvement_verified'],result)
        self.assertEqual(result['complete_pairs'],8)
        self.assertEqual(result['complete_recordings'],0)  # Separate from complete performance traces.

    def test_count_alone_wrong_seed_or_prefix_cannot_form_a_pair(self):
        self.mutate('main-rapier-red',lambda r:r.update(seed=999))
        result=self.result()
        self.assertFalse(result['v6_paired_improvement_verified'])
        self.assertIn('case_identity',result['invalid_evidence']['main-rapier-red:candidate'])
        self.mutate('recovery-paid_early',lambda r:r['prefix_identity'].update(commands_hash=evaluation._digest('other')))
        self.assertIn('recovery-paid_early:pair_identity_mismatch',self.result()['failed_gates'])

    def test_checkpoint_helpers_radar_and_execution_sources_are_bound(self):
        (self.root/'model.zip').write_bytes(b'irrelevant')  # Wrong path cannot change checkpoint identity.
        self.assertTrue(self.result()['v6_paired_improvement_verified'])
        (self.root/'infantry.json').write_text('different frozen helper')
        result=self.result()
        self.assertFalse(result['v6_paired_improvement_verified'])
        self.assertIn('frozen_pool_changed',result['invalid_evidence']['main-rapier-red:candidate'])
        (self.root/'main/model.zip').write_bytes(b'changed checkpoint')
        self.assertIn('checkpoint_hash',self.result()['invalid_evidence']['main-rapier-red:candidate'])

    def test_source_hash_difference_refuses_pair_without_feature_performance_selection(self):
        def changed(row):
            row['identity']['sources']['sdk_hash']=evaluation._digest('new-sdk')
            row['identity']['execution_hash']=evaluation._digest(row['identity']['sources'])
        self.mutate('main-rapier-red',changed)
        self.assertIn('main-rapier-red:pair_identity_mismatch',self.result()['failed_gates'])

    def test_missing_trace_tail_and_fake_passage_flags_do_not_count(self):
        case='main-rapier-red';path=self.root/'evaluation'/(case+'-candidate.jsonl')
        rows=path.read_text().splitlines();path.write_text('\n'.join(rows[:-1])+'\n')
        self.mutate(case,lambda r:r['trace'].update(sha256=evaluation._file_hash(path)))
        result=self.result();self.assertIn('trace_horizon',result['invalid_evidence'][case+':candidate'])
        case='main-rapier-blue';path=self.root/'evaluation'/(case+'-candidate.jsonl')
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:row['position'][0]=8.
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        self.mutate(case,lambda r:r['trace'].update(sha256=evaluation._file_hash(path)))
        self.assertIn('tunnel_evidence',self.result()['invalid_evidence'][case+':candidate'])

    def test_paid_advantage_is_relative_to_each_handoff_and_requires_actual_cost(self):
        old=dict(learner_start_time=10,recovery=dict(death_episodes=[dict(observed_mode='free',revived_at=20,initial_hp=30,cost=0)]))
        new=dict(learner_start_time=100,recovery=dict(death_episodes=[dict(observed_mode='paid',revived_at=102,initial_hp=300,cost=100)]))
        advantage=evaluation._revive_advantage(old,new)
        self.assertTrue(advantage['demonstrated']);self.assertEqual(advantage['seconds_saved'],8)
        new['recovery']['death_episodes'][0]['cost']=0
        self.assertFalse(evaluation._revive_advantage(old,new)['demonstrated'])

    def test_legal_paid_race_fallback_is_not_actor_illegality(self):
        self.mutate('recovery-paid_early',lambda r:r.update(execution_mask_fallbacks=20))
        self.assertTrue(self.result()['v6_paired_improvement_verified'])
        self.mutate('recovery-paid_early',lambda r:r['actor_legality'].update(invalid_decisions=1))
        self.assertFalse(self.result()['v6_paired_improvement_verified'])

    def test_per_case_regression_cannot_hide_in_aggregate_or_recovery_gains(self):
        self.mutate('main-warp_kinematic-red',lambda r:r['metrics'].update(capture_seconds=4,capture_after_recovery=0))
        self.mutate('main-warp_kinematic-red',lambda r:r.update(capture_after_recovery=0))
        self.assertIn('main-warp_kinematic-red:per_case_regression',self.result()['failed_gates'])

    def test_recovery_and_training_gates_require_successful_operations(self):
        self.mutate('recovery-free_ready',lambda r:r['recovery'].update(recoveries=0))
        self.mutate('recovery-unavailable',lambda r:r['recovery'].update(paid_coins=100,paid_revives=1))
        self.mutate('recovery-alive_weak',lambda r:r.update(capture_after_recovery=4.9))
        write(self.root/'main/training.json',dict(accepted_actor_steps=10,skill_unchanged=True,
            published_accepted_actor_steps=0,published_skill_unchanged=False,
            published_checkpoint_sha256=evaluation._file_hash(self.root/'main/model.zip')))
        gates=self.result()['failed_gates']
        self.assertIn('free_ready:free_and_recovery_not_demonstrated',gates)
        self.assertIn('unavailable:free_without_payment_not_demonstrated',gates)
        self.assertIn('alive_weak:recovery_and_capture_not_demonstrated',gates)
        self.assertIn('no_accepted_ppo_update',gates);self.assertIn('frozen_skill_not_verified',gates)

    def test_only_published_update_report_for_exact_candidate_can_pass(self):
        path=self.root/'main/training.json';report=json.loads(path.read_text())
        del report['published_accepted_actor_steps']
        report['accepted_actor_steps']=1000
        write(path,report)
        self.assertIn('no_accepted_ppo_update',self.result()['failed_gates'])
        report['published_accepted_actor_steps']=1
        report['published_checkpoint_sha256']=evaluation._digest('another-checkpoint')
        write(path,report)
        self.assertIn('published_checkpoint_hash_mismatch',self.result()['failed_gates'])

    def test_quota_partial_replay_must_never_claim_complete(self):
        battle=SimpleNamespace(frames=SimpleNamespace(offsets=[(0,200)]),actions=[{}])
        self.assertTrue(evaluation._recording_would_exceed(battle,0,byte_limit=100))
        self.mutate('main-rapier-red',lambda r:r.update(recording_complete=True,replay=dict(metadata=dict(recording_complete=False))))
        self.assertIn('replay_completion_claim',self.result()['invalid_evidence']['main-rapier-red:candidate'])


if __name__=='__main__':unittest.main()
