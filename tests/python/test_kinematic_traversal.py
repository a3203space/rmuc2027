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
    def test_dead_airborne_robot_settles_without_horizontal_jump(self):
        adapter=ArenaMotion();command=launch_request(8.6,0.)
        frame=adapter.step(command)
        robot=command['robots'][0]
        robot.update(pos=frame['positions'][-1][0],velocity=frame['velocities'][-1][0],alive=False,enabled=False)
        initial=np.asarray(robot['pos']).copy();zs=[initial[2]]
        for tick in range(1,25):
            command['sequence']=tick;frame=adapter.step(command)
            zs.extend(np.asarray(frame['positions'])[:,0,2])
            np.testing.assert_allclose(np.asarray(frame['positions'])[:,0,:2],np.tile(initial[:2],(6,1)),atol=1e-6)
            robot.update(pos=frame['positions'][-1][0],velocity=frame['velocities'][-1][0])
        self.assertTrue(np.asarray(frame['grounded']).all())
        self.assertTrue((np.diff(zs)<=1e-6).all())
        self.assertAlmostEqual(zs[-1],terrain_height(_REFERENCE_ARENA,*initial[:2]),places=4)
        adapter.close()

    def test_authoritative_teleport_cancels_old_jump(self):
        adapter=ArenaMotion();command=launch_request(8.6,0.)
        first=adapter.step(command)
        self.assertFalse(first['grounded'][-1][0])
        command['sequence']=1
        command['robots'][0].update(pos=[4.,4.,terrain_height(_REFERENCE_ARENA,4.,4.)],
                                    velocity=[0.,0.,0.],jump=0.)
        frame=adapter.step(command)
        np.testing.assert_allclose(np.asarray(frame['positions'])[:,:,0:2],[[[4.,4.]]]*6,atol=1e-6)
        self.assertTrue(np.asarray(frame['grounded']).all())
        self.assertEqual(adapter.jump_status.numpy()[0],0)
        adapter.close()

    def test_graph_matches_direct_launches_through_jump(self):
        adapters=[ArenaMotion(reuse_commands=False,packed_readback=False),ArenaMotion(),ArenaMotion(use_graph=True)]
        commands=[launch_request(8.6,0.) for _ in adapters]
        for tick in range(40):
            frames=[]
            for adapter,command in zip(adapters,commands):
                command['sequence']=tick
                frame=adapter.step(command);frames.append(frame)
                command['robots'][0].update(pos=frame['positions'][-1][0],velocity=frame['velocities'][-1][0],
                    heading=frame['headings'][-1][0],folded=frame['folded'][-1][0])
            for key in ('positions','velocities','headings','folded','grounded'):
                for frame in frames[1:]:
                    np.testing.assert_array_equal(frames[0][key],frame[key])
        self.assertIsNotNone(adapters[2].graph)
        for adapter in adapters:adapter.close()

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

    def test_new_obstruction_aborts_arc_without_ground_teleport(self):
        adapter=ArenaMotion();command=launch_request(8.6,0.)
        # Move an initially distant obstacle into the route after takeoff.
        # This exercises collision feedback, not just preflight rejection.
        command['arena']['obstacles'].append(dict(x=24.,y=1.,w=.1,h=1.,bottom=0.,height=3.))
        samples=[];states=[]
        for tick in range(45):
            command['sequence']=tick
            if tick==3:
                boxes=adapter.motion.boxes.numpy()
                boxes[adapter.physical_count-1]=[9.4,7.,9.5,8.]
                adapter.motion.boxes.assign(boxes)
                command['robots'][0]['jump']=0.
            frame=adapter.step(command)
            samples.extend(np.asarray(frame['positions'])[:,0,:].tolist())
            states.append(int(adapter.jump_status.numpy()[0]))
            command['robots'][0].update(pos=frame['positions'][-1][0],velocity=frame['velocities'][-1][0],
                heading=frame['headings'][-1][0],folded=frame['folded'][-1][0])
        samples=np.asarray(samples)
        self.assertIn(4,states,'Blocked flight must enter falling state')
        self.assertEqual(states[-1],3,'Landing after an interrupted jump is a failure')
        self.assertLessEqual(samples[:,0].max(),9.15+.001)
        self.assertLess(np.abs(np.diff(samples[:,2])).max(),.08,'No instantaneous ground snap')
        self.assertAlmostEqual(samples[-1,2],terrain_height(_REFERENCE_ARENA,*samples[-1,:2]),places=4)
        self.assertTrue(frame['grounded'][-1][0])
        adapter.close()

    def test_robot_entering_jump_route_does_not_get_crossed(self):
        adapter=ArenaMotion();command=launch_request(8.6,0.)
        blocker=copy.deepcopy(command['robots'][0])
        blocker.update(id='blue-4',team='blue',jump=0.,pos=[24.,1.,terrain_height(_REFERENCE_ARENA,24.,1.)])
        # Tall test hull ensures overlap throughout flight; isolated from the
        # real roster's design values so the test covers collision feedback.
        blocker['design']['muzzle_z_m']=2.
        command['robots'].append(blocker)
        states=[];path=[]
        for tick in range(45):
            command['sequence']=tick
            if tick==3:
                blocker['pos']=[9.8,7.5,terrain_height(_REFERENCE_ARENA,9.8,7.5)]
                command['robots'][0]['jump']=0.
            frame=adapter.step(command)
            states.append(int(adapter.jump_status.numpy()[0]))
            path.extend(np.asarray(frame['positions'])[:,0,:].tolist())
            for index,robot in enumerate(command['robots']):
                robot.update(pos=frame['positions'][-1][index],velocity=frame['velocities'][-1][index],
                    heading=frame['headings'][-1][index],folded=frame['folded'][-1][index])
        self.assertIn(4,states)
        self.assertEqual(states[-1],3)
        self.assertLessEqual(np.asarray(path)[:,0].max(),9.3+.001)
        self.assertLess(np.abs(np.diff(np.asarray(path)[:,2])).max(),.08)
        adapter.close()
