"""Optional real MJWarp lifecycle tests; no stand-in simulation."""
import importlib.util
import unittest
from rmuc2027_research.rl.mjwarp import BatchPhysics

AVAILABLE = all(importlib.util.find_spec(n) for n in ('mujoco', 'mujoco_warp', 'warp', 'numpy'))
XML = '''<mujoco><worldbody><geom type="plane" size="10 10 .1"/>
<body pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
</worldbody></mujoco>'''


class ValidationTests(unittest.TestCase):
    def test_invalid_batch_and_timestep_rejected_before_loading_optional_dependencies(self):
        for count in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                BatchPhysics(XML, count)
        for dt in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                BatchPhysics(XML, 2, timestep=dt)

    @unittest.skipUnless(AVAILABLE, 'MJWarp optional dependencies not installed')
    def test_real_cpu_worlds_reset_independently(self):
        import numpy as np
        with BatchPhysics(XML, 2, device='cpu') as batch:
            for _ in range(10):
                batch.step(np.zeros((2, 0)))
            before = batch.state()
            self.assertLess(before['qpos'][0, 2], 1.)
            batch.reset([0])
            after = batch.state()
            self.assertAlmostEqual(float(after['qpos'][0, 2]), 1., places=5)
            np.testing.assert_array_equal(before['qpos'][1], after['qpos'][1])
            np.testing.assert_array_equal(before['qvel'][1], after['qvel'][1])
            with self.assertRaises(ValueError):
                batch.step(np.zeros((1, 0)))
        with self.assertRaises(RuntimeError):
            batch.state()
