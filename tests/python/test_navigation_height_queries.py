import inspect
import random
import unittest
from rmuc2027_core import sdk


class HeightQueryTests(unittest.TestCase):
    def test_clearance_keeps_both_footprint_edges_on_terrain(self):
        clear=sdk._prepare_chapter4(sdk._REFERENCE_ARENA,'red',False,.43)[2]
        start=(20.275321976693597,7.340985194291465)
        end=(21.270798334970294,7.842168235694187)
        # The centerline is traversable here; the full chassis crosses a
        # terrain edge. The strict footprint check must remain stronger.
        self.assertTrue(clear(start,end,(),False))
        self.assertFalse(clear(start,end,(),True))
        for strict in (False,True):
            self.assertTrue(clear(start,start,(),strict))

    def test_navigation_height_matches_public_ground_query(self):
        graph=sdk._prepare_chapter4(sdk._REFERENCE_ARENA,'red',False)
        height=inspect.getclosurevars(graph[2]).nonlocals['height']
        rng=random.Random(904)
        for _ in range(2000):
            x,y=rng.uniform(0,28),rng.uniform(0,15)
            self.assertAlmostEqual(height(x,y),sdk.terrain_height(sdk._REFERENCE_ARENA,x,y),places=12)
