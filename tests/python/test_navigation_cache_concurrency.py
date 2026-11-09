import math
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from rmuc2027_core import sdk


class RouteCacheConcurrencyTests(unittest.TestCase):
    def test_parallel_evictions_keep_exact_costs_and_bounded_cache(self):
        class CheckedCache(dict):
            def get(self,*args):
                if not sdk._NAV_ROUTE_LOCK.locked():raise AssertionError('unlocked cache lookup')
                return super().get(*args)
            def pop(self,*args):
                if not sdk._NAV_ROUTE_LOCK.locked():raise AssertionError('unlocked cache eviction')
                return super().pop(*args)
            def __setitem__(self,*args):
                if not sdk._NAV_ROUTE_LOCK.locked():raise AssertionError('unlocked cache insertion')
                return super().__setitem__(*args)
        def clear(p,goal,near,strict):return p==(2.,0.)
        graph=([(0.,0.),(1.,0.),(2.,0.)],[[(1,1.)],[(0,1.),(2,1.)],[(1,1.)]],clear,None)
        goals=[(3.+i*.001,1.) for i in range(512)]*2
        with patch.object(sdk,'_NAV_ROUTE_COSTS',CheckedCache()) as cache:
            with ThreadPoolExecutor(max_workers=8) as pool:
                results=list(pool.map(lambda goal:sdk._route_costs(graph,goal,{},True),goals))
            self.assertLessEqual(len(cache),128)
            for goal,cost in zip(goals,results):
                tail=math.dist((2.,0.),goal)
                self.assertEqual(cost,{2:tail,1:tail+1.,0:tail+2.})
            # Returned distances remain usable after their key is evicted.
            self.assertEqual(results[0],results[512])

    def test_repeated_key_returns_same_immutable_by_convention_cost_table(self):
        def clear(p,goal,near,strict):return True
        graph=([(0.,0.)],[[]],clear,None)
        with patch.object(sdk,'_NAV_ROUTE_COSTS',{}):
            with ThreadPoolExecutor(max_workers=8) as pool:
                tables=list(pool.map(lambda _:sdk._route_costs(graph,(3.,4.),{},True),range(64)))
            self.assertTrue(all(table is tables[0] for table in tables))
            self.assertEqual(tables[0],{0:5.})


if __name__=='__main__':unittest.main()
