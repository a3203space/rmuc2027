import os
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import patch
from rmuc2027_core import engine


@unittest.skipUnless(os.name=='posix','POSIX pipe readiness required')
class PipeTransportTests(unittest.TestCase):
    def test_broken_transport_cannot_hang_flushing_full_stdin(self):
        process=subprocess.Popen([sys.executable,'-c','import sys,time; sys.stdout.buffer.write(b"x"*8000000); sys.stdout.flush(); time.sleep(5)'],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
        match=object.__new__(engine.Match);match.process=process;match.closed=False
        match._broken=True;match._reader=None;match.external_physics=None
        os.set_blocking(process.stdin.fileno(),False)
        try:
            while True:os.write(process.stdin.fileno(),b'x'*4096)
        except BlockingIOError:pass
        os.set_blocking(process.stdin.fileno(),True)
        process.stdin.write(b'pending')
        errors=[]
        def close():
            try:match._close_transport()
            except BaseException as error:errors.append(error)
        worker=threading.Thread(target=close,daemon=True);worker.start();worker.join(.5)
        completed=not worker.is_alive()
        if not completed:process.kill();worker.join(2)
        self.assertTrue(completed,'Broken transport must terminate the child before a buffered flush')
        self.assertFalse(errors)
        self.assertIsNotNone(process.poll())

    def pipe(self):
        read,write=os.pipe();reader=os.fdopen(read,'rb')
        self.addCleanup(reader.close);self.addCleanup(os.close,write)
        return engine._PosixLineReader(reader),write

    def test_partial_and_coalesced_lines_preserve_order(self):
        reader,write=self.pipe()
        os.write(write,b'{"one":')
        def finish():
            time.sleep(.02);os.write(write,b'1}\n{"two":2}\n')
        worker=threading.Thread(target=finish);worker.start()
        try:
            self.assertEqual(reader.read_line(timeout=1),b'{"one":1}\n')
            self.assertEqual(reader.read_line(timeout=1),b'{"two":2}\n')
        finally:worker.join()

    def test_partial_response_cannot_extend_the_deadline(self):
        reader,write=self.pipe()
        def partial():
            for _ in range(8):
                os.write(write,b' ');time.sleep(.02)
        worker=threading.Thread(target=partial);worker.start();start=time.monotonic()
        try:
            with self.assertRaisesRegex(RuntimeError,'timed out'):reader.read_line(timeout=.05)
            self.assertLess(time.monotonic()-start,.13)
        finally:worker.join()

    def test_size_limit_rejects_a_line_before_unbounded_buffering(self):
        reader,write=self.pipe();os.write(write,b'x'*256)
        with self.assertRaisesRegex(RuntimeError,'size limit'):reader.read_line(timeout=1,maximum=128)
        self.assertLessEqual(len(reader.pending),128)

    def test_large_line_and_read_ahead_survive_pipe_chunks(self):
        reader,write=self.pipe();payload=b'x'*100000+b'\nnext\n'
        worker=threading.Thread(target=os.write,args=(write,payload));worker.start()
        try:
            self.assertEqual(reader.read_line(timeout=1),payload[:-5])
            self.assertEqual(reader.read_line(timeout=1),b'next\n')
        finally:worker.join()

    def test_eof_returns_buffered_tail_then_empty(self):
        read,write=os.pipe()
        with os.fdopen(read,'rb') as stream:
            reader=engine._PosixLineReader(stream)
            os.write(write,b'tail');os.close(write)
            self.assertEqual(reader.read_line(timeout=1),b'tail')
            self.assertEqual(reader.read_line(timeout=1),b'')

    def test_direct_and_threaded_readers_preserve_frames_and_observation_batches(self):
        results=[]
        for direct in (False,True):
            with patch.object(engine,'_DIRECT_PIPE_READER',direct),engine.Match(seed=42,battle_settings={'drones_enabled':False}) as match:
                frames=[]
                for tick in range(50):frames.append(match.step({'red':{'3':{'move':[.1,.2],'aim':tick*.01},'4':{'transform':tick>20}}}))
                viewers=[('red',[3]),('blue',[7]),('red',[1,4])]
                results.append((frames,match.observations(viewers),match.digest()))
                reader=match._reader
            if reader is not None:self.assertFalse(reader.is_alive())
        self.assertEqual(results[0],results[1])

    def test_interrupted_observation_tail_cannot_be_used_as_a_later_response(self):
        with engine.Match(seed=42,battle_settings={'drones_enabled':False}) as match:
            original=match._read_line;calls=0
            def read():
                nonlocal calls
                calls+=1
                if calls>1:raise KeyboardInterrupt
                return original()
            with patch.object(match,'_read_line',side_effect=read):
                with self.assertRaises(KeyboardInterrupt):match.observations([('red',[3]),('blue',[7])])
            with self.assertRaisesRegex(RuntimeError,'interrupted'):match.digest()
