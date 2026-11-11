"""CPU fake engines only: every enqueued request must complete or fail promptly."""
import queue
import builtins
import threading
import time
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import patch
from rmuc2027_research.rl.warp_kinematic import batching as warp_batch
from rmuc2027_research.rl.mjwarp import batching as mj_batch
from rmuc2027_research.rl.mjwarp import arena as mj_arena


class FakeEngine:
    def __init__(self,device):self.last_profile={}
    @staticmethod
    def signature(request):
        if request.get('bad_signature'):raise ValueError('bad signature')
        return 'same'
    def step_many(self,items,**kwargs):
        values=[i[1] if isinstance(i,tuple) else i for i in items]
        if any(v.get('fail') for v in values):raise ValueError('bad integration')
        return ['ok']*len(items)
    def close(self):pass


class FailCloseEngine(FakeEngine):
    def close(self):raise RuntimeError('cleanup failed')


class BadClient:
    device='cpu'
    design_signature='same'
    @property
    def scene_signature(self):raise ValueError('bad scene')


def client():return SimpleNamespace(device='cpu',scene_signature='same',design_signature='same')


class ServiceLifecycleTests(unittest.TestCase):
    def run_request(self,service,first,request):
        future=Future();service.queue.put((first,request,future))
        return future.result(timeout=.4)

    def test_warp_bad_scene_does_not_kill_service_or_strand_request(self):
        with patch.object(warp_batch,'WorldBatch',FakeEngine):
            service=warp_batch.PhysicsService(1)
            try:
                with self.assertRaisesRegex(ValueError,'bad scene'):self.run_request(service,BadClient(),{})
                self.assertEqual(self.run_request(service,client(),{}),'ok')
                self.assertTrue(service.thread.is_alive())
            finally:service.close()

    def test_warp_failed_cleanup_preserves_original_request_error(self):
        with patch.object(warp_batch,'WorldBatch',FailCloseEngine):
            service=warp_batch.PhysicsService(1)
            try:
                with self.assertRaisesRegex(ValueError,'bad integration'):self.run_request(service,client(),{'fail':True})
                self.assertEqual(self.run_request(service,client(),{}),'ok')
            finally:service.close()

    def test_mj_bad_signature_does_not_kill_service_or_strand_request(self):
        with patch.object(mj_arena,'ArenaPhysics',FakeEngine):
            service=mj_batch.PhysicsService(1)
            try:
                with self.assertRaisesRegex(ValueError,'bad signature'):self.run_request(service,'cpu',{'bad_signature':True})
                self.assertEqual(self.run_request(service,'cpu',{}),'ok')
                self.assertTrue(service.thread.is_alive())
            finally:service.close()

    def test_mj_failed_engine_is_evicted_even_when_cleanup_fails(self):
        with patch.object(mj_arena,'ArenaPhysics',FailCloseEngine):
            service=mj_batch.PhysicsService(1)
            try:
                with self.assertRaisesRegex(ValueError,'bad integration'):self.run_request(service,'cpu',{'fail':True})
                self.assertEqual(len(service.engines),0)
                self.assertEqual(self.run_request(service,'cpu',{}),'ok')
            finally:service.close()

    def test_mj_close_cannot_put_sentinel_before_inflight_submit(self):
        entered=threading.Event();release=threading.Event();errors=[];results=[]
        class ShortFuture(Future):
            def result(self,timeout=None):return super().result(timeout=.5)
        class GateQueue(queue.Queue):
            def put(self,item,*args,**kwargs):
                if isinstance(item,tuple):entered.set();release.wait(1.)
                return super().put(item,*args,**kwargs)
        with patch.object(mj_arena,'ArenaPhysics',FakeEngine),patch.object(mj_batch,'Future',ShortFuture),patch.object(mj_batch.queue,'Queue',GateQueue):
            service=mj_batch.PhysicsService(1)
            def submit():
                try:results.append(service.submit('cpu',{}))
                except BaseException as e:errors.append(e)
            submitting=threading.Thread(target=submit);submitting.start();self.assertTrue(entered.wait(.5))
            closing=threading.Thread(target=service.close);closing.start()
            time.sleep(.02);release.set();submitting.join(1.);closing.join(1.)
            self.assertFalse(submitting.is_alive());self.assertFalse(closing.is_alive())
            self.assertEqual(errors,[]);self.assertEqual(results,['ok'])
            with self.assertRaisesRegex(RuntimeError,'closed'):service.submit('cpu',{})
            service.close()

    def test_incomplete_batch_results_fail_instead_of_stranding_futures(self):
        class ShortEngine(FakeEngine):
            def step_many(self,items,**kwargs):return []
        for module,owner,attribute,first in ((warp_batch,warp_batch,'WorldBatch',client()),
                                             (mj_batch,mj_arena,'ArenaPhysics','cpu')):
            with self.subTest(module=module.__name__),patch.object(owner,attribute,ShortEngine):
                service=module.PhysicsService(1)
                try:
                    with self.assertRaisesRegex(RuntimeError,'incomplete result'):
                        self.run_request(service,first,{})
                    self.assertEqual(len(service.engines),0)
                    self.assertTrue(service.thread.is_alive())
                finally:service.close()

    def test_shutdown_wakes_warp_round_wait_and_completes_queued_work(self):
        with patch.object(warp_batch,'WorldBatch',FakeEngine):
            service=warp_batch.PhysicsService(2)
            service.begin_round(2)
            future=Future();service.queue.put((client(),{},future))
            closing=threading.Thread(target=service.close);closing.start();closing.join(.5)
            self.assertFalse(closing.is_alive())
            self.assertEqual(future.result(timeout=.1),'ok')
            with self.assertRaisesRegex(RuntimeError,'closed'):service.submit(client(),{})
            service.close()

    def test_mj_import_failure_closes_service_without_an_orphan_queue(self):
        original=builtins.__import__
        def importing(name,globals=None,locals=None,fromlist=(),level=0):
            if name=='arena' and level==1 and (globals or {}).get('__name__')==mj_batch.__name__:
                raise ImportError('test missing integrator')
            return original(name,globals,locals,fromlist,level)
        with patch.object(builtins,'__import__',side_effect=importing):
            service=mj_batch.PhysicsService(1);service.thread.join(.5)
            self.assertFalse(service.thread.is_alive())
            self.assertIsInstance(service.terminal_error,ImportError)
            with self.assertRaisesRegex(RuntimeError,'closed'):service.submit('cpu',{})
            service.close()

    def test_cleanup_diagnostics_keep_only_the_latest_32_errors(self):
        for module in (warp_batch,mj_batch):
            service=module.PhysicsService(1)
            try:
                for _ in range(100):service._dispose(FailCloseEngine('cpu'))
                self.assertEqual(service.cleanup_error_count,100)
                self.assertEqual(len(service.cleanup_errors),32)
            finally:service.close()

if __name__=='__main__':unittest.main()
