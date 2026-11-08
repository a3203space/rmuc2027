import copy
from concurrent.futures import ThreadPoolExecutor
import unittest
import numpy as np
from rmuc2027_research.rl.warp_kinematic.arena import ArenaMotion
from rmuc2027_research.rl.warp_kinematic.batching import BatchClient,PhysicsService,WorldBatch
from test_kinematic_adapter import request
from test_kinematic_traversal import launch_request


def feedback(command,frame):
    command['sequence']+=1
    for i,r in enumerate(command['robots']):
        r.update(pos=frame['positions'][-1][i],velocity=frame['velocities'][-1][i],
                 heading=frame['headings'][-1][i],folded=frame['folded'][-1][i],grounded=frame['grounded'][-1][i])


class WorldBatchTests(unittest.TestCase):
    def compare(self,commands,death=False,teleport=False):
        # Same IDs across worlds, different authority and asynchronous schedules.
        clients=[BatchClient(None,'cuda:0') for _ in commands]
        for c,r in zip(clients,commands):c.ids=tuple(b['id'] for b in r['robots'])
        solos=[ArenaMotion() for _ in commands];engines={}
        try:
            for tick in range(35):
                order=[0,1] if tick%2 else [1,0]
                if tick%3==0:order=[0]  # World 1 must not advance while absent.
                if tick==4 and death:commands[0]['robots'][0].update(enabled=False,alive=False)
                if tick==7 and teleport:
                    commands[0]['robots'][0].update(pos=[4.,4.,0.],velocity=[0.,0.,0.],jump=0.)
                items=[(clients[i],commands[i]) for i in order]
                key=tuple((i,commands[i]['robots'][0]['model']) for i in order)
                if key not in engines:engines[key]=WorldBatch('cuda:0')
                engine=engines[key]
                # A real service reuses slots for different clients with equal
                # static designs. Reverse ordering is intentionally included.
                actual=engine.step_many(items)
                for i,result in zip(order,actual):
                    expected=solos[i].step(copy.deepcopy(commands[i]))
                    for field in ('positions','velocities','headings','folded','grounded'):
                        np.testing.assert_array_equal(result[field],expected[field],err_msg=str((tick,i,field)))
                    feedback(commands[i],result)
        finally:
            for e in engines.values():e.close()
            for e in solos:e.close()

    def test_jump_state_follows_world_across_order_and_batch_size(self):
        a=launch_request(8.6,0.);b=copy.deepcopy(a)
        b['robots'][0].update(pos=[10.85,7.5,.5],jump=0.)
        self.compare([a,b])

    def test_disabled_fall_and_authoritative_teleport_match_solo(self):
        for death,teleport in ((True,False),(False,True)):
            with self.subTest(death=death):
                self.compare([launch_request(8.6,0.),launch_request(19.4,3.141592653589793)],death,teleport)

    def test_fold_and_roof_clearance_match_solo(self):
        a=request(fold=1.,roof=True);b=copy.deepcopy(a)
        a['robots'][0]['move']=[.5,0.];b['robots'][0].update(transform=True,move=[-.4,0.])
        self.compare([a,b])

    def test_service_merges_concurrent_worlds_and_closes(self):
        service=PhysicsService(2,wait_seconds=.02)
        clients=[BatchClient(service,'cuda:0') for _ in range(2)]
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures=[pool.submit(c.step,request()) for c in clients]
                frames=[f.result(timeout=30) for f in futures]
            self.assertEqual(service.max_batch,2)
            self.assertEqual(service.world_steps,2)
            self.assertEqual(frames[0],frames[1])
            later=request();later['sequence']=1;later['arena']=None
            later['robots'][0]['design']=None
            # First use of a singleton cache slot must recover both static
            # fields from the client, even if the rule request omits them.
            self.assertEqual(clients[1].step(later)['sequence'],1)
            clients[0].close()
            replacement=BatchClient(service,'cuda:0')
            self.assertEqual(replacement.step(request())['sequence'],0)
            self.assertIsNone(clients[0].state)
        finally:service.close()
        self.assertFalse(service.thread.is_alive())

    def test_worker_exception_is_forwarded_without_leaking_thread(self):
        service=PhysicsService(1);client=BatchClient(service,'cuda:0')
        bad=request();bad['robots'][0]['pos']=[float('nan'),5.,0.]
        try:
            with self.assertRaises(ValueError):client.step(bad)
            fresh=BatchClient(service,'cuda:0')
            self.assertEqual(fresh.step(request())['sequence'],0)
        finally:service.close()
        self.assertFalse(service.thread.is_alive())

    def test_round_continues_when_one_worker_finishes_without_request(self):
        service=PhysicsService(2);client=BatchClient(service,'cuda:0')
        service.begin_round(2)
        pool=ThreadPoolExecutor(max_workers=2)
        def empty():
            service.worker_done()
        def moving():
            try:
                command=request()
                for _ in range(3):
                    frame=client.step(command);feedback(command,frame)
                return frame
            finally:service.worker_done()
        try:
            a=pool.submit(empty);b=pool.submit(moving)
            a.result(timeout=10)
            self.assertEqual(b.result(timeout=10)['sequence'],2)
        finally:
            service.close();pool.shutdown(wait=True)
        self.assertFalse(service.thread.is_alive())
