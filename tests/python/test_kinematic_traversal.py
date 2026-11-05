import copy
import math
import unittest
import numpy as np
from rmuc2027_core.sdk import _REFERENCE_ARENA, terrain_height
from rmuc2027_research.rl.warp_kinematic.arena import ArenaMotion
from test_kinematic_adapter import request


def launch_request(x, heading, model=2731):
    command=request(model=model)
    command['arena']=copy.deepcopy(_REFERENCE_ARENA)
    command['robots'][0].update(pos=[x,7.5,terrain_height(_REFERENCE_ARENA,x,7.5)],
        heading=heading,heading_goal=heading,jump=1.)
    return command


def advance(adapter,command,frames=40):
    airborne=False
    trace=[]
    for tick in range(frames):
        command['sequence']=tick
        frame=adapter.step(command)
        airborne |= not np.asarray(frame['grounded']).all()
        robot=command['robots'][0]
        robot.update(pos=frame['positions'][-1][0],velocity=frame['velocities'][-1][0],
                     heading=frame['headings'][-1][0],folded=frame['folded'][-1][0])
        trace.append((list(robot['pos']),int(adapter.jump_status.numpy()[0])))
    return airborne,trace


class TraversalTests(unittest.TestCase):
    def test_mirrored_central_entry_lands_on_platform(self):
        for x,heading,target in [(8.6,0.,10.85),(19.4,math.pi,17.15)]:
            with self.subTest(start=x):
                adapter=ArenaMotion();command=launch_request(x,heading)
                airborne,trace=advance(adapter,command)
                self.assertTrue(airborne,trace)
                self.assertEqual(trace[-1][1],2,trace)
                self.assertAlmostEqual(trace[-1][0][0],target,places=3)
                self.assertAlmostEqual(trace[-1][0][2],terrain_height(_REFERENCE_ARENA,target,7.5),places=4)
                adapter.close()

    def test_non_jump_chassis_cannot_use_link(self):
        adapter=ArenaMotion();command=launch_request(8.6,0.,model=2732)
        airborne,trace=advance(adapter,command,3)
        self.assertFalse(airborne)
        self.assertEqual(trace[-1][1],0)
        adapter.close()

    def test_obstructed_arc_rejects_launch(self):
        adapter=ArenaMotion();command=launch_request(8.6,0.)
        command['arena']['obstacles'].append(dict(x=9.,y=7.,w=.1,h=1.,bottom=0.,height=3.))
        airborne,trace=advance(adapter,command,3)
        self.assertFalse(airborne)
        self.assertEqual(trace[-1][1],3)
        adapter.close()

    def test_occupied_landing_rejects_launch(self):
        adapter=ArenaMotion();command=launch_request(8.6,0.)
        blocker=copy.deepcopy(command['robots'][0]);blocker.update(id='blue-4',jump=0.,pos=[10.85,7.5,.5])
        command['robots'].append(blocker)
        frame=adapter.step(command)
        self.assertTrue(np.asarray(frame['grounded']).all())
        self.assertEqual(adapter.jump_status.numpy()[0],3)
        adapter.close()
