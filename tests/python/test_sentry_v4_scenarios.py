import collections
import json
import unittest

from rmuc2027_research.rl.scenarios import ScenarioSampler, ScenarioSpec, scenario_registry


class ScenarioTests(unittest.TestCase):
    def test_exact_registry_and_learner_hardware_on_both_sides(self):
        cases = scenario_registry()
        self.assertEqual(len(cases), 486)
        self.assertEqual(len({c.ordered_id for c in cases}), 243)
        self.assertEqual(len({c.quantity_id for c in cases}), 108)
        sizes = collections.Counter(collections.Counter(c.quantity_id for c in cases).values())
        self.assertEqual(sizes, {2: 27, 4: 54, 8: 27})
        for case in cases:
            lineups = case.lineups()
            self.assertEqual(lineups[case.learner_side]['7'], 2773)
            other = 'blue' if case.learner_side == 'red' else 'red'
            self.assertEqual(lineups[other]['7'], case.enemy_sentry)
            self.assertEqual(ScenarioSpec.from_dict(json.loads(json.dumps(case.to_dict()))), case)

    def test_first_group_pass_balanced_then_exact_full_coverage(self):
        sampler = ScenarioSampler(301)
        first = [sampler.sample() for _ in range(108)]
        self.assertEqual(len({c.quantity_id for c in first}), 108)
        self.assertEqual(collections.Counter(c.learner_side for c in first), {'red': 54, 'blue': 54})
        cases = first + [sampler.sample() for _ in range(486-108)]
        self.assertEqual(len({c.scenario_id for c in cases}), 486)
        self.assertEqual(sampler.coverage()['full_completed']['scenarios'], 0)

    def test_rng_queue_and_failure_replay_resume_exactly(self):
        sampler = ScenarioSampler(11, failure_fraction=1.)
        cases = [sampler.sample() for _ in range(486)]
        sampler.record(cases[3], valid_steps=600, seconds=60, completed=True, failure=True)
        restored = ScenarioSampler.from_state_dict(json.loads(json.dumps(sampler.state_dict())))
        self.assertEqual([sampler.next_episode() for _ in range(8)], [restored.next_episode() for _ in range(8)])
        self.assertEqual(restored.sample().scenario_id, cases[3].scenario_id)
        self.assertEqual(sampler.coverage()['short_completed']['scenarios'], 1)
        self.assertEqual(sampler.coverage()['full_completed']['scenarios'], 0)
        with self.assertRaises(ValueError):
            sampler.record(cases[3], failure=True, split='test')

    def test_reject_schema_and_queue_tampering(self):
        sampler = ScenarioSampler()
        state = sampler.state_dict(); state['queue'][1] = state['queue'][0]
        with self.assertRaises(ValueError):
            ScenarioSampler.from_state_dict(state)
        state = sampler.state_dict(); state['registry_hash'] = 'changed'
        with self.assertRaises(ValueError):
            ScenarioSampler.from_state_dict(state)
        with self.assertRaises(ValueError):
            ScenarioSpec(enemy_sentry=2731)


if __name__ == '__main__':
    unittest.main()
