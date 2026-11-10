"""Thread/HTTP regressions only: no real models, physics or training."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from rmuc2027_core.policy import FrameLog
from rmuc2027_research.web.server import ArenaService, create_app


class WebResponsivenessTests(unittest.TestCase):
    def test_slow_step_allows_state_and_stop_without_publishing_half_a_frame(self):
        entered, release = threading.Event(), threading.Event()
        class Frozen:
            def __init__(self):
                self.done=False; self.metadata={}; self.battle=object(); self.step_seconds=.1
                self.saved_stopped=False; self.closed=False
            def frame(self):return {'tick':2 if self.done else 0}
            def step(self):
                entered.set()
                if not release.wait(2):raise TimeoutError('Test worker was not released')
                self.done=True
                return self.frame()
            def save(self, _path, *, stopped):self.saved_stopped=stopped
            def result(self):return {}
            def close(self):self.closed=True
        frozen=Frozen()
        with tempfile.TemporaryDirectory() as directory:
            service=ArenaService.__new__(ArenaService)
            service.lock=threading.RLock(); service.stopping=False; service.paused=False
            service.speed=1; service.revision=0; service.busy=True; service.error=None
            service.store=SimpleNamespace(directory=Path(directory), running=lambda *args:None,
                                          finish=lambda *args, **kwargs:None)
            with patch('rmuc2027_research.web.server.FrozenResearchMatch', return_value=frozen):
                worker=threading.Thread(target=service._run_research,args=('test',{}));worker.start()
                try:
                    self.assertTrue(entered.wait(1))
                    acquired=service.lock.acquire(timeout=.1)
                    self.assertTrue(acquired, 'Slow inference held the API state lock')
                    try:
                        self.assertEqual(service.frame['tick'],0)
                        service.control({'command':'stop'})
                    finally:
                        service.lock.release()
                finally:
                    release.set(); worker.join(2)
            self.assertFalse(worker.is_alive())
            self.assertIsNone(service.error)
            self.assertEqual(service.frame['tick'],2)
            self.assertTrue(frozen.saved_stopped)
            self.assertTrue(frozen.closed)
            self.assertFalse(service.busy)

    def test_events_cannot_advance_ahead_of_the_published_frame(self):
        frames=FrameLog({'tick':0,'time':0.,'events':[]})
        self.addCleanup(frames.file.close)
        frames.append({'tick':1,'time':.05,'events':[{'type':'next-frame-event'}]})
        service=ArenaService.__new__(ArenaService)
        service.__dict__.update(lock=threading.RLock(),strategy_library=SimpleNamespace(status=lambda:{'version':0}),
            strategy_version=0, revision=0, battle=SimpleNamespace(frames=frames), frame={'tick':0},
            metadata={'match_mode':'research'}, paused=False, speed=1, error=None,
            sandbox='test', match_id='test', busy=True, prematch=None)
        self.assertEqual(service.state()['events'],[])
        service.frame={'tick':1}
        self.assertEqual(service.state()['events'],[{'type':'next-frame-event'}])
        self.assertEqual(service.state(since=1)['events'],[])

    def test_active_research_replay_spool_is_not_exported_during_a_step(self):
        class FakeInitialMatch:
            def __init__(self,*args,**kwargs):pass
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def snapshot(self):return {'time':0.,'tick':0,'robots':[]}
        with tempfile.TemporaryDirectory() as directory:
            with patch('rmuc2027_research.web.server.Match',FakeInitialMatch):
                app=create_app(test_mode=True,data_root=Path(directory)/'web',research_root=Path(directory)/'research')
            service=app.state.arena
            service.busy=True;service.metadata={'match_mode':'research'};service.battle=object()
            client=TestClient(app)
            self.assertEqual(client.get('/api/replay').status_code,409)
            service.busy=False;service.battle=None
            self.assertEqual(client.get('/api/replay').status_code,404)


if __name__=='__main__':unittest.main()
