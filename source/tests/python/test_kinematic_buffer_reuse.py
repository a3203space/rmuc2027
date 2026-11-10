import unittest
import numpy as np
import warp as wp
from rmuc2027_research.rl.warp_kinematic.arena import ArenaMotion,controls
from rmuc2027_research.rl.warp_kinematic.terrain import sample_heights
from rmuc2027_research.rl.warp_kinematic.motion import MotionBatch
from rmuc2027_research.rl.warp_kinematic.protocol import DT
from test_kinematic_traversal import launch_request


class RefreshingMotion(ArenaMotion):
    def initialize(self,request):
        super().initialize(request)
        advance=self.motion._advance
        def refreshed(dt,launch=wp.launch,**kwargs):return advance(dt,launch=launch)
        self.motion._advance=refreshed

    def _launch(self,kernel,*args,**kwargs):
        if kernel is controls:
            m=self.motion
            super()._launch(sample_heights,m.count,inputs=[self.terrain.data,m.pos,m.heights],device=m.device)
        return super()._launch(kernel,*args,**kwargs)


class BufferReuseTests(unittest.TestCase):
    def test_buffer_exchange_matches_copies_during_wall_contact_and_recovery(self):
        arguments=([[.21,0.]]*3,[[3.,4.,.2]]*3,[[-1.,-2.,0.,2.]])
        reference=MotionBatch(*arguments);optimized=MotionBatch(*arguments)
        for tick in range(30):
            for motion in (reference,optimized):
                motion.enabled.assign(np.asarray([1,int(tick<10 or tick>=20),1],dtype=np.int32))
            commands=[[.3,.2],[-.1,-.2],[0.,0.]]
            positions,velocities=reference.frame(commands)
            optimized.commands.assign(np.asarray(commands,dtype=np.float32))
            for index in range(6):
                optimized._advance(DT,ping_pong=True)
                actual_position,actual_velocity=optimized.state()
                self.assertEqual(actual_position.tobytes(),positions[index].tobytes())
                self.assertEqual(actual_velocity.tobytes(),velocities[index].tobytes())

    def test_reused_heights_and_buffers_match_uncached_trajectory(self):
        for graph in (False,True):
            adapters=[RefreshingMotion(),ArenaMotion(use_graph=graph)]
            commands=[launch_request(8.6,0.),launch_request(8.6,0.)]
            initial=[]
            try:
                for tick in range(42):
                    frames=[]
                    for adapter,command in zip(adapters,commands):
                        if tick==20:command['robots'][0].update(leg_height=.39,jump=0.)
                        if tick==30:command['robots'][0].update(move=[.2,.1],heading_goal=.3)
                        command['sequence']=tick;frame=adapter.step(command);frames.append(frame)
                        command['robots'][0].update(pos=frame['positions'][-1][0],velocity=frame['velocities'][-1][0],
                            heading=frame['headings'][-1][0],folded=frame['folded'][-1][0],grounded=frame['grounded'][-1][0])
                    self.assertEqual(frames[0],frames[1])
                    for field in ('positions','velocities','headings','folded'):
                        self.assertEqual(np.asarray(frames[0][field]).tobytes(),np.asarray(frames[1][field]).tobytes())
                    pointers=(adapters[1].motion.pos.ptr,adapters[1].motion.vel.ptr)
                    if tick==0:initial=pointers
                    self.assertEqual(pointers,initial)
            finally:
                for adapter in adapters:adapter.close()
