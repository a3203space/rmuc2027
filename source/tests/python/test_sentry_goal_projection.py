"""Public-geometry regression for the projection/strict-execution contract.

The small audited fixture is independent of research run directories. It uses
public map geometry and our own recorded robot pose/footprint/fold only. These
tests construct no Match, expose no opponent state and execute no physics.
"""
import copy
import json
from pathlib import Path
import unittest

from rmuc2027_core import sdk
from rmuc2027_research.rl.portable import SdkNavigator


FIXTURE=Path(__file__).resolve().parents[1]/'fixtures/rl_navigation_regression.json'


class SentryGoalProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture=json.loads(FIXTURE.read_text())
        cls.arena=copy.deepcopy(cls.fixture['arena'])
        # SDK graphs are keyed by layout revision. Isolate this frozen geometry
        # from graphs prewarmed from the current rulebook during SDK import.
        cls.arena['layout_revision']+='-goal-projection-regression-fixture'

    def graph(self,robot):
        return sdk._prepare_chapter4(self.arena,robot['team'],
            robot['transform_progress']>=.99,robot['radius']+.04)

    def path(self,robot,goal):
        return sdk._complete_route(self.graph(robot),(robot['x'],robot['y']),tuple(goal))

    def test_recorded_relaxed_only_goal_is_not_accepted_by_strict_projection(self):
        case=next(c for c in self.fixture['cases'] if c['name']=='candidate-25')
        robot=case['robot'];old=case['historical_projected_goal'];graph=self.graph(robot)
        self.assertEqual(old,[9.25,7.25])
        self.assertEqual(self.path(robot,old),[])
        self.assertFalse(any(graph[2](node,old,(),True) for node in graph[0]))
        self.assertTrue(any(graph[2](node,old,(),False) for node in graph[0]))
        goal=SdkNavigator().resolve_goal(robot,case['proposed_goal'],self.arena)
        self.assertEqual(goal,case['expected_strict_goal'])
        self.assertNotEqual(goal,old)
        self.assertTrue(self.path(robot,goal))

    def test_recorded_approach_folded_exit_and_capture_goals_preserve_expected_outputs(self):
        navigator=SdkNavigator()
        for case in self.fixture['cases']:
            with self.subTest(case=case['name']):
                goal=navigator.resolve_goal(case['robot'],case['proposed_goal'],self.arena)
                self.assertEqual(goal,case['expected_strict_goal'])
                self.assertTrue(self.path(case['robot'],goal))

    def test_every_returned_path_segment_keeps_strict_footprint_clearance_on_both_sides(self):
        for case in self.fixture['cases']:
            for mirror in (False,True):
                robot=copy.deepcopy(case['robot']);proposed=list(case['proposed_goal'])
                if mirror:
                    robot.update(team='red',x=self.arena['width']-robot['x'],y=self.arena['height']-robot['y'])
                    proposed=[self.arena['width']-proposed[0],self.arena['height']-proposed[1]]
                with self.subTest(case=case['name'],mirror=mirror):
                    goal=SdkNavigator().resolve_goal(robot,proposed,self.arena)
                    self.assertIsNotNone(goal)
                    graph=self.graph(robot);path=self.path(robot,goal)
                    self.assertTrue(path)
                    self.assertFalse(graph[3](tuple(goal)))
                    start=(robot['x'],robot['y'])
                    for index,end in enumerate(path):
                        ignore=graph[3](start,True) if index==0 else {}
                        self.assertTrue(graph[2](start,end,ignore,True))
                        start=end


if __name__=='__main__':unittest.main()
