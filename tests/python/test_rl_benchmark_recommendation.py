import unittest
from rmuc2027_research.rl.__main__ import recommend_envs


class BenchmarkRecommendationTests(unittest.TestCase):
    def test_aggregate_throughput_does_not_override_per_environment_gate(self):
        rows=[dict(num_envs=1,per_env_simulation_seconds_per_second=5.1,rl_steps_per_second=10.2),
              dict(num_envs=2,per_env_simulation_seconds_per_second=3.7,rl_steps_per_second=14.8)]
        self.assertEqual(recommend_envs(rows),1)
        self.assertIsNone(recommend_envs(rows,6.))

    def test_profiled_measurement_is_not_a_recommendation(self):
        row=dict(num_envs=1,per_env_simulation_seconds_per_second=7.,rl_steps_per_second=14.,profiled=True)
        self.assertIsNone(recommend_envs([row]))
