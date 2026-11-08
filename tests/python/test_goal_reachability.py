import random
import unittest
from rmuc2027_core import sdk
from rmuc2027_research.rl.portable import SdkNavigator


class GoalReachabilityTests(unittest.TestCase):
    def test_component_query_selects_same_goal_as_sdk_shortest_path(self):
        rng=random.Random(222)
        navigator=SdkNavigator(.25)
        arena=sdk._REFERENCE_ARENA
        for team,x,y in [('red',2.5,5.),('blue',25.5,10.),('red',12.,7.5)]:
            for folded in (False,True):
                robot=dict(team=team,x=x,y=y,radius=.39,transform_progress=float(folded))
                graph=sdk._prepare_chapter4(arena,team,folded,robot['radius']+.04)
                for _ in range(4):
                    proposed=[rng.uniform(0,28),rng.uniform(0,15)]
                    actual=navigator.resolve_goal(robot,proposed,arena)
                    key=(arena['layout_revision'],team,folded,.39,.25)
                    points=sorted(navigator.grids[key],key=lambda p:((p[0]-proposed[0])**2+(p[1]-proposed[1])**2,p))
                    start=(x,y);near=graph[3](start,True);expected=None
                    for goal in points:
                        if any(graph[2](start,goal,near,strict) or
                               sdk._route(graph,start,goal,near,set(),strict) is not None for strict in (True,False)):
                            expected=list(goal);break
                    self.assertEqual(actual,expected,(robot,proposed))

    def test_large_hull_goals_use_execution_clearance(self):
        arena=sdk._REFERENCE_ARENA
        navigator=SdkNavigator(.25)
        small=sdk._prepare_chapter4(arena,'red',False)
        large=sdk._prepare_chapter4(arena,'red',False,.74)
        # Select real map points admitted by the old fixed margin but rejected
        # by this chassis. Test actual goal output, not only the graph argument.
        candidates=[(i*.25,j*.25) for i in range(3,109) for j in range(3,57)
                    if not small[3]((i*.25,j*.25)) and large[3]((i*.25,j*.25))]
        self.assertTrue(candidates)
        robot=dict(team='red',x=2.5,y=5.,radius=.7,transform_progress=0.)
        for proposed in candidates[::max(1,len(candidates)//12)]:
            goal=navigator.resolve_goal(robot,proposed,arena)
            self.assertIsNotNone(goal)
            self.assertFalse(large[3](goal),(proposed,goal))
