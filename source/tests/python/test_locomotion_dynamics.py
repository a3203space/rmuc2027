"""Version-two flight, attitude transport and world isolation."""
import copy
import unittest
import numpy as np
from rmuc2027_research.rl.warp_kinematic.arena import ArenaMotion
from rmuc2027_research.rl.warp_kinematic.protocol import MotionFrame
from test_kinematic_adapter import request


def command(vx=0., model=2731):
    value=request(model=model)
    value['version']=2
    value['arena']['terrain']=[]
    value['arena']['obstacles']=[]
    value['arena']['forbidden']=[]
    value['robots'][0].update(pos=[4.,4.,0.],velocity=[vx,0.,0.],move=[0.,0.],jump=1.,coast=True,
        quaternion=[0.,0.,0.,1.],angular_velocity=[0.,0.,0.],grounded=True)
    return value


def feedback(value, frame):
    value['sequence']+=1
    for i,r in enumerate(value['robots']):
        r.update(pos=frame['positions'][-1][i],velocity=frame['velocities'][-1][i],
            heading=frame['headings'][-1][i],folded=frame['folded'][-1][i],grounded=frame['grounded'][-1][i],
            quaternion=frame['rotations'][-1][i],angular_velocity=frame['angular_velocities'][-1][i])


class DynamicsTests(unittest.TestCase):
    def test_initial_airborne_powerless_body_falls_and_settles(self):
        a=ArenaMotion();c=command()
        c['robots'][0].update(pos=[4.,4.,2.],enabled=False,alive=False,grounded=False,jump=0.)
        try:
            first=a.step(c);feedback(c,first)
            self.assertLess(c['robots'][0]['pos'][2],2.)
            self.assertFalse(c['robots'][0]['grounded'])
            for _ in range(30):feedback(c,a.step(c))
            self.assertTrue(c['robots'][0]['grounded'])
            self.assertLess(c['robots'][0]['pos'][2],.3)
        finally:a.close()

    def test_angular_velocity_rotates_with_world_heading(self):
        import math
        outputs=[]
        for yaw in (0.,math.pi/2):
            a=ArenaMotion();c=command()
            c['robots'][0].update(heading=yaw,heading_goal=yaw,jump=0.,coast=False,
                velocity=[2*math.cos(yaw),2*math.sin(yaw),0.],
                quaternion=[0.,0.,math.sin(yaw/2),math.cos(yaw/2)])
            try:outputs.append(np.asarray(a.step(c)['angular_velocities'])[:,0])
            finally:a.close()
        expected=np.column_stack((-outputs[0][:,1],outputs[0][:,0],outputs[0][:,2]))
        np.testing.assert_allclose(outputs[1],expected,atol=1e-5)

    def test_attitude_and_jump_state_follow_world_through_reordering_and_teleport(self):
        from rmuc2027_research.rl.warp_kinematic.batching import BatchClient,WorldBatch
        commands=[command(1.),command(2.,2772)]
        clients=[BatchClient(None,'cuda:0') for _ in commands]
        solos=[ArenaMotion() for _ in commands]
        batches={}
        for client,value in zip(clients,commands):client.ids=tuple(r['id'] for r in value['robots'])
        try:
            for tick in range(28):
                order=[0] if tick%3==0 else [1,0] if tick%2 else [0,1]
                if tick==9:commands[0]['robots'][0].update(pos=[4.,4.,0.],velocity=[0.,0.,0.],grounded=True)
                if tick==12:commands[0]['robots'][0]['jump']=0.
                if tick==15:commands[0]['robots'][0]['jump']=1.
                key=tuple(order)
                if key not in batches:batches[key]=WorldBatch('cuda:0')
                frames=batches[key].step_many([(clients[i],commands[i]) for i in order])
                for i,actual in zip(order,frames):
                    expected=solos[i].step(copy.deepcopy(commands[i]))
                    for field in ('positions','velocities','rotations','angular_velocities','legs','grounded'):
                        np.testing.assert_array_equal(actual[field],expected[field],err_msg=str((tick,i,field)))
                    feedback(commands[i],actual)
        finally:
            for value in list(batches.values())+solos:value.close()

    def test_cuda_graph_preserves_extended_frame(self):
        engines=[ArenaMotion(use_graph=False),ArenaMotion(use_graph=True)]
        value=command(2.)
        try:
            for _ in range(24):
                frames=[a.step(copy.deepcopy(value)) for a in engines]
                for field in ('positions','velocities','rotations','angular_velocities','legs'):
                    np.testing.assert_array_equal(frames[0][field],frames[1][field])
                feedback(value,frames[0])
        finally:
            for a in engines:a.close()

    def test_jump_preserves_momentum_and_held_request_does_not_repeat(self):
        distances=[]
        for speed in (0.,2.):
            a=ArenaMotion('cuda:0');c=command(speed)
            try:
                heights=[];air=[]
                for _ in range(70):
                    f=a.step(c);feedback(c,f)
                    heights.append(f['positions'][-1][0][2]);air.append(not f['grounded'][-1][0])
                self.assertGreater(max(heights),.2)
                self.assertFalse(any(air[-20:]),'held jump must not relaunch after cooldown')
                distances.append(c['robots'][0]['pos'][0]-4.)
                self.assertEqual(f['version'],2)
            finally:a.close()
        self.assertLess(abs(distances[0]),.01)
        self.assertGreater(distances[1],1.)

    def test_ordinary_and_flat_jump_chassis_use_distinct_capabilities(self):
        for model,expected in ((303,False),(3,True),(304,True),(305,True)):
            a=ArenaMotion('cuda:0');c=command(model=model)
            try:
                f=a.step(c)
                self.assertEqual(not f['grounded'][-1][0],expected)
            finally:a.close()

    def test_invalid_orientation_rejected_before_exposing_frame(self):
        base=dict(ids=('r',),positions=np.zeros((6,1,3)),velocities=np.zeros((6,1,3)),
            headings=np.zeros((6,1)),folded=np.zeros((6,1)),grounded=np.ones((6,1),dtype=bool),
            rotations=np.tile([0.,0.,0.,1.],(6,1,1)),angular_velocities=np.zeros((6,1,3)),legs=np.zeros((6,1,2)))
        f=MotionFrame(**base)
        self.assertEqual(f.response(0)['version'],2)
        with self.assertRaises(ValueError):MotionFrame(**{**base,'rotations':np.zeros((6,1,4))})
        with self.assertRaises(ValueError):MotionFrame(**{**base,'legs':None})


if __name__=='__main__':unittest.main()
