import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, BrokenBarrierError
from unittest.mock import patch

from rmuc2027_core import sdk


class AimCacheConcurrencyTests(unittest.TestCase):
    def test_simultaneous_eviction_does_not_remove_the_same_entry_twice(self):
        class OverlappingEvictions(dict):
            barrier = Barrier(2)

            def pop(self, key, *args):
                # Without an atomic eviction, both workers select the same key
                # before either removes it. A serialized first worker times out.
                try:
                    self.barrier.wait(timeout=0.2)
                except BrokenBarrierError:
                    pass
                return super().pop(key, *args)

        robot = dict(role='infantry', x=0., y=0., z=0.)
        targets = [dict(x=4. + i, y=0., aim_z=100.) for i in range(2)]
        with patch.object(sdk, '_AIM_CACHE', OverlappingEvictions({i: (0., False) for i in range(256)})) as cache:
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda target: sdk._ballistic_solution(robot, target), targets))
            self.assertEqual(results, [(sdk.math.pi / 4, False)] * 2)
            self.assertEqual(len(cache), 256)

    def test_parallel_hits_and_evictions_match_uncached_ballistics(self):
        robot = dict(role='infantry', x=0., y=0., z=0., vx=.2, vy=.1)
        targets = [dict(x=2. + i * .031, y=.5, aim_z=.3) for i in range(320)]
        expected = []
        with patch.object(sdk, '_AIM_CACHE', {}) as cache:
            for target in targets:
                cache.clear()
                expected.append(sdk._ballistic_solution(robot, target))
            cache.clear()
            with ThreadPoolExecutor(max_workers=8) as pool:
                actual = list(pool.map(lambda target: sdk._ballistic_solution(robot, target), targets * 2))
            self.assertEqual(actual, expected * 2)
            self.assertLessEqual(len(cache), 256)


if __name__ == '__main__':
    unittest.main()
