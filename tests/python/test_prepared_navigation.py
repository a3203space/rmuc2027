import random
import unittest
from concurrent.futures import ThreadPoolExecutor
from rmuc2027_core import sdk
from rmuc2027_research.rl.navigation import PreparedNavigator
from rmuc2027_research.rl.portable import SdkNavigator
from rmuc2027_research.rl.session import load_config,provenance


class PreparedNavigationTests(unittest.TestCase):
    def test_preparation_preserves_goals_across_teams_fold_and_hull(self):
        rng=random.Random(1081)
        reference=SdkNavigator();prepared=PreparedNavigator()
        for team,folded,radius in [('red',False,.39),('blue',True,.39),('red',True,.7)]:
            for start in [(2.5,5.),(12.,7.5)]:
                robot=dict(team=team,x=start[0],y=start[1],radius=radius,transform_progress=float(folded))
                for _ in range(5):
                    proposed=[rng.uniform(0,28),rng.uniform(0,15)]
                    self.assertEqual(prepared.resolve_goal(robot,proposed,sdk._REFERENCE_ARENA),
                                     reference.resolve_goal(robot,proposed,sdk._REFERENCE_ARENA))
        self.assertGreater(prepared.preparation_seconds,0.)

    def test_table_is_shared_across_reset_but_excludes_dynamic_state(self):
        robot=dict(team='red',x=2.5,y=5.,radius=.391,transform_progress=0.)
        first=PreparedNavigator();first.resolve_goal(robot,[24.,14.],sdk._REFERENCE_ARENA)
        second=PreparedNavigator();second.resolve_goal(robot,[1.,1.],sdk._REFERENCE_ARENA)
        self.assertIs(first.precomputed_reachability,second.precomputed_reachability)
        self.assertEqual(second.preparation_seconds,0.)
        with self.assertRaises(TypeError):second.precomputed_reachability['corrupt']=True
        reference=SdkNavigator()
        for x,y in [(4.,13.),(12.,7.5),(24.,10.)]:
            moved=dict(robot,x=x,y=y)
            self.assertEqual(second.resolve_goal(moved,[14.,7.5],sdk._REFERENCE_ARENA),
                             reference.resolve_goal(moved,[14.,7.5],sdk._REFERENCE_ARENA))

    def test_concurrent_preparation_builds_once_and_returns_equal_goals(self):
        robot=dict(team='blue',x=25.5,y=10.,radius=.392,transform_progress=0.)
        navs=[PreparedNavigator(),PreparedNavigator()]
        with ThreadPoolExecutor(max_workers=2) as pool:
            answers=list(pool.map(lambda n:n.resolve_goal(robot,[2.,2.],sdk._REFERENCE_ARENA),navs))
        self.assertEqual(answers[0],answers[1])
        self.assertEqual(sum(n.preparation_seconds>0 for n in navs),1)
        self.assertIs(navs[0].precomputed_reachability,navs[1].precomputed_reachability)

    def test_config_validation_and_recorded_source(self):
        for invalid in (0,1,'true',[]):
            with self.assertRaises(ValueError):load_config({'navigation_precompute':invalid})
        config=load_config({'sim_backend':'warp_kinematic'})
        navigation=provenance(config)['navigation']
        self.assertTrue(navigation['precompute'])
        self.assertEqual(len(navigation['preparation_sha256']),64)
        self.assertFalse(provenance(load_config({'sim_backend':'warp_kinematic','navigation_precompute':False}))['navigation']['precompute'])
