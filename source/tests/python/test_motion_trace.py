import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
from rmuc2027_research.rl.mjwarp.diagnose import MotionTrace


class TraceTests(TestCase):
    def test_commanded_stall_is_distinct_from_stationary_guard(self):
        b = SimpleNamespace(metadata={}, policies=[], held_commands={'red':{'3':{'move':[1,0]},'7':{'move':[0,0]}}})
        robots = [dict(id=f'red-{slot}',slot=slot,team='red',role='infantry',x=1,y=2,z=0,alive=True,powered=True) for slot in (3,7)]
        with TemporaryDirectory() as tmp, patch('rmuc2027_research.rl.session.provenance',return_value={}):
            trace = MotionTrace(Path(tmp)/'trace',b,{})
            for tick in range(61):
                trace.record({'tick':tick,'time':tick*.05,'robots':robots})
            trace.close()
            report = json.loads((Path(tmp)/'trace/summary.json').read_text())
            self.assertEqual(len(report['alerts']),1)
            self.assertEqual(report['alerts'][0]['robot'],'red-3')
            self.assertEqual(report['alerts'][0]['kind'],'commanded_without_progress')
            self.assertFalse(b.motion_trace)
            self.assertTrue((Path(tmp)/'trace/trajectories.svg').exists())

    def test_zero_command_far_from_live_goal_is_reported(self):
        nav = {'navigation': {'3': {'time':0.,'goal':[17.2,7.5]}, '7':{'time':0.,'goal':[10.,7.5]}}}
        policy = SimpleNamespace(navigation_trace=nav)
        b = SimpleNamespace(metadata={},policies=[('red',None,policy)],held_commands={'red':{'3':{'move':[0,0]},'7':{'move':[0,0]}}})
        robots = [dict(id=f'red-{slot}',slot=slot,team='red',role='infantry',x=10.,y=7.5,z=0.,alive=True,powered=True) for slot in (3,7)]
        with TemporaryDirectory() as tmp, patch('rmuc2027_research.rl.session.provenance',return_value={}):
            trace=MotionTrace(Path(tmp)/'trace',b,{})
            for tick in range(61):
                for value in nav['navigation'].values():
                    value['time']=tick*.05
                trace.record({'tick':tick,'time':tick*.05,'robots':robots})
            trace.close()
            report=json.loads((Path(tmp)/'trace/summary.json').read_text())
            self.assertEqual([(a['robot'],a['kind']) for a in report['alerts']], [('red-3','zero_command_before_goal')])
