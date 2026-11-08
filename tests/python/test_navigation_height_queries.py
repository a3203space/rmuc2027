import inspect
import random
import unittest
from rmuc2027_core import sdk


class HeightQueryTests(unittest.TestCase):
    def test_navigation_height_matches_public_ground_query(self):
        graph=sdk._prepare_chapter4(sdk._REFERENCE_ARENA,'red',False)
        height=inspect.getclosurevars(graph[2]).nonlocals['height']
        rng=random.Random(904)
        for _ in range(2000):
            x,y=rng.uniform(0,28),rng.uniform(0,15)
            self.assertAlmostEqual(height(x,y),sdk.terrain_height(sdk._REFERENCE_ARENA,x,y),places=12)
