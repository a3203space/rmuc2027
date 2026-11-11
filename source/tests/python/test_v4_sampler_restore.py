"""State-only sampler migration checks: no environment, optimizer or rollout."""
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np

from rmuc2027_research.rl.capabilities import CapabilitySampler
from rmuc2027_research.rl.scenarios import ScenarioSampler,canonical_hash
from rmuc2027_research.rl.v4_training import plan_sampler_restore,restore_state


def sampler(seed,starts=3):
    scenarios=ScenarioSampler(seed)
    for index in range(starts):
        case=scenarios.sample()
        scenarios.record(case,valid_steps=index+1,seconds=60. if index else 2.,completed=index>0,failure=index==0)
    capabilities=CapabilitySampler(seed+7919);capabilities.episodes=starts
    for _ in range(starts):capabilities.rng.random()
    return dict(session=dict(episode_index=starts,scenario=scenarios.state_dict(),
        capabilities=capabilities.state_dict(),rng_state=random.Random(seed).getstate()),
        gym_rng=np.random.default_rng(seed).bit_generator.state)


def totals(states):
    result={}
    for state in states:
        for key,row in state['session']['scenario']['counts'].items():
            values=result.setdefault(key,{})
            for name,value in row.items():values[name]=values.get(name,0)+value
    return result


class SamplerRestoreTests(unittest.TestCase):
    def assert_migrated(self,original,workers):
        before=copy.deepcopy(original)
        migrated,report=plan_sampler_restore(original,workers,num_timesteps=10310)
        self.assertEqual(original,before)
        self.assertEqual(totals(original),totals(migrated))
        self.assertEqual(report['ancestor_states'],original)
        self.assertEqual(report['recorded_counts_before'],report['recorded_counts_after'])
        self.assertEqual(report['session_episode_total_before'],report['session_episode_total_after'])
        self.assertEqual(report['capability_episode_total_before'],report['capability_episode_total_after'])
        self.assertFalse(report['bitwise_sampler_restore']);self.assertFalse(report['migration_adds_observed_coverage'])
        started=set().union(*(set(s['session']['scenario']['queue'][:s['session']['scenario']['cursor']]) for s in original))
        for state in migrated:
            scene=ScenarioSampler.from_state_dict(state['session']['scenario'])
            CapabilitySampler.from_state_dict(state['session']['capabilities'])
            self.assertEqual(set(scene.queue[:scene.cursor]),started)
            self.assertEqual(set(scene.queue),set(original[0]['session']['scenario']['queue']))
        self.assertEqual(len({s['session']['scenario']['seed'] for s in migrated}),workers)
        self.assertEqual(len({s['session']['capabilities']['seed'] for s in migrated}),workers)
        return migrated,report

    def test_one_to_many_has_independent_streams_without_multiplying_history(self):
        result,report=self.assert_migrated([sampler(7)],4)
        queues=[s['session']['scenario']['queue'][s['session']['scenario']['cursor']:] for s in result]
        self.assertEqual(len({tuple(q) for q in queues}),4)
        for kind in ('scenario','capabilities'):
            self.assertEqual(len({canonical_hash(s['session'][kind]['rng_state']) for s in result}),4)
        self.assertIn('worker_topology_changed',report['reasons'])

    def test_many_to_one_preserves_all_ancestors_counts_failures_and_started_prefixes(self):
        original=[sampler(7,2),sampler(17,4),sampler(27,5)]
        result,report=self.assert_migrated(original,1)
        failures=result[0]['session']['scenario']['failures']
        for state in original:
            for key,weight in state['session']['scenario']['failures'].items():self.assertGreaterEqual(failures[key],weight)
        self.assertEqual(len(report['original_cursors']),3)

    def test_legacy_clones_migrate_even_when_worker_count_is_unchanged(self):
        state=sampler(7);clone=copy.deepcopy(state)
        first=next(iter(clone['session']['scenario']['counts']))
        clone['session']['scenario']['counts'][first]['valid_steps']+=5
        _,report=self.assert_migrated([state,clone],2)
        self.assertIn('legacy_cloned_scenario_streams',report['reasons'])
        self.assertIn('legacy_cloned_capability_streams',report['reasons'])
        self.assertTrue(report['historical_clone_records_preserved_without_deduplication'])

    def test_unique_same_topology_is_exact_and_migrations_are_deterministic(self):
        states=[sampler(7),sampler(17)]
        restored,report=plan_sampler_restore(states,2)
        self.assertEqual(restored,states);self.assertTrue(report['bitwise_sampler_restore'])
        self.assertEqual(report['mode'],'exact_sampler_state')
        first=plan_sampler_restore(states,3,num_timesteps=50)
        self.assertEqual(first,plan_sampler_restore(states,3,num_timesteps=50))
        self.assertNotEqual(first[0],plan_sampler_restore(states,3,num_timesteps=51)[0])

    def test_registry_mismatch_is_rejected_before_migration(self):
        state=sampler(7);state['session']['scenario']['registry_hash']='wrong'
        with self.assertRaisesRegex(ValueError,'registry/schema'):plan_sampler_restore([state],2)

    def test_restore_writes_audit_without_rewriting_original_sampler_checkpoint(self):
        class Vector:
            num_envs=2
            def __init__(self):self.restored={};self._seeds=[7,17]
            def env_method(self,name,value,indices):
                if name!='restore_sampler_state':raise AssertionError(name)
                self.restored[indices]=copy.deepcopy(value)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'samplers.json';path.write_text(json.dumps([sampler(7)]))
            original=path.read_bytes();vec=Vector();result=restore_state(directory,vec)
            self.assertEqual(path.read_bytes(),original)
            self.assertEqual(vec._seeds,[None,None]);self.assertEqual(set(vec.restored),{0,1})
            report=json.loads((Path(directory)/'sampler-restore.json').read_text())
            self.assertEqual(report['source_sampler_sha256'],hashlib.sha256(original).hexdigest())
            self.assertEqual(report['ancestor_states'],json.loads(original))
            self.assertEqual(result['sampler_restore']['mode'],'migrated')
            previous=(Path(directory)/'sampler-restore.json').read_bytes()
            second=restore_state(directory,vec)['sampler_restore']
            archived=Path(directory)/second['previous_restore_report']['path']
            self.assertEqual(archived.read_bytes(),previous)
            self.assertEqual(path.read_bytes(),original)


if __name__=='__main__':unittest.main()
