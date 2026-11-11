import json
import math
import unittest

from rmuc2027_research.rl.capabilities import AIM_PROFILES, MOBILITY_FIELDS, AimNoiseExecutor, CapabilitySampler, SlowCapabilityDrift
from rmuc2027_research.rl.scenarios import ScenarioSpec


class CapabilitiesTests(unittest.TestCase):
    def test_learner_protected_and_resume_sampler(self):
        sampler = CapabilitySampler(31)
        for side in ('red', 'blue'):
            lineups = ScenarioSpec(learner_side=side).lineups()
            profiles = sampler.sample(lineups, side)
            self.assertEqual(len(profiles), 8)
            learner = profiles[side+':7']
            self.assertEqual(learner['mobility'], dict.fromkeys(MOBILITY_FIELDS, 1.))
            self.assertEqual(learner['aim']['sigma_deg'], 0.)
            self.assertEqual(learner['aim']['delay_s'], 0.)
            other = 'red' if side == 'blue' else 'blue'
            self.assertFalse(profiles[other+':7']['protected_learner'])
            self.assertEqual(profiles[other+':7']['observation_scope'], 'strict_sentry')
        restored = CapabilitySampler.from_state_dict(json.loads(json.dumps(sampler.state_dict())))
        self.assertEqual(sampler.sample(lineups, side), restored.sample(lineups, side))

    def test_delay_starts_invalid_then_uses_old_reference(self):
        executor = AimNoiseExecutor(dict(delay_s=.15, sigma_deg=0., tau_s=.15))
        self.assertFalse(executor.step(0., .1, .2)['ready'])
        executor.step(.05, .3, .4)
        executor.step(.10, .5, .6)
        result = executor.step(.15, .7, .8)
        self.assertTrue(result['ready'])
        self.assertAlmostEqual(result['aim'], .1)
        self.assertAlmostEqual(result['pitch'], .2)
        self.assertEqual(executor.step(.15, 9., 9.), result)
        with self.assertRaises(ValueError):
            executor.step(.1, 0, 0)

    def test_noise_state_round_trip_and_angular_units(self):
        executor = AimNoiseExecutor('weak', 98)
        for i in range(10):
            executor.step(i*.05, .4, -.1)
        restored = AimNoiseExecutor.from_state_dict(json.loads(json.dumps(executor.state_dict())))
        for i in range(10, 30):
            self.assertEqual(executor.step(i*.05, .4, -.1), restored.step(i*.05, .4, -.1))
        self.assertLess(abs(executor.noise[0]), math.radians(20))
        self.assertEqual(AIM_PROFILES['weak']['sigma_deg'], 2.5)

    def test_main_has_no_additional_noise(self):
        executor = AimNoiseExecutor('learner')
        for i in range(5):
            result = executor.step(i*.05, .5, -.2)
            self.assertTrue(result['ready'])
            self.assertEqual(result['noise_yaw_rad'], 0.)
            self.assertEqual(result['noise_pitch_rad'], 0.)
            self.assertAlmostEqual(result['aim'], .5)

    def test_variable_delay_retains_history_and_noise_state(self):
        executor = AimNoiseExecutor(dict(delay_s=0., sigma_deg=0., tau_s=.15))
        for i in range(21):
            executor.step(i*.05, i*.01, .0)
        executor.set_profile(dict(delay_s=.3, sigma_deg=0., tau_s=.15))
        result = executor.step(1.05, .21, 0.)
        self.assertTrue(result['ready'])
        self.assertAlmostEqual(result['aim'], .15)
        self.assertAlmostEqual(result['age_s'], .3)
        executor.noise = [.02, -.01]
        executor.set_profile('weak')
        self.assertEqual(executor.noise, [.02, -.01])

    def test_slow_drift_protects_allies_and_has_no_target_boundary_jump(self):
        base = CapabilitySampler(5).sample(ScenarioSpec().lineups(), 'red')
        drift = SlowCapabilityDrift(base, 17, 'red')
        before = drift.update(14.99)
        at = drift.update(15.)
        self.assertEqual(before, at)
        later = drift.update(20.)
        for seat in base:
            if seat.startswith('red:'):
                self.assertEqual(later[seat], base[seat])
            else:
                self.assertNotEqual(later[seat]['mobility'], base[seat]['mobility'])
                self.assertLessEqual(later[seat]['nominal_drive_efficiency']*later[seat]['mobility']['drive_efficiency'], 1.)
        restored = SlowCapabilityDrift.from_state_dict(json.loads(json.dumps(drift.state_dict())))
        self.assertEqual(drift.update(55.), restored.update(55.))

    def test_drift_is_not_step_count_dependent_or_compounded(self):
        base = CapabilitySampler(8).sample(ScenarioSpec().lineups(), 'blue')
        small = SlowCapabilityDrift(base, 91, 'blue')
        large = SlowCapabilityDrift(base, 91, 'blue')
        for i in range(1201):
            actual = small.update(i*.05)
        expected = large.update(60.)
        for seat in base:
            for field in MOBILITY_FIELDS:
                self.assertAlmostEqual(actual[seat]['mobility'][field], expected[seat]['mobility'][field], places=12)
        final = large.update(3600.)
        for seat in large.seats:
            self.assertLessEqual(final[seat]['mobility']['speed_reference_mps'], 1.1*base[seat]['mobility']['speed_reference_mps'])
            self.assertGreaterEqual(final[seat]['mobility']['speed_reference_mps'], .9*base[seat]['mobility']['speed_reference_mps'])


if __name__ == '__main__':
    unittest.main()
