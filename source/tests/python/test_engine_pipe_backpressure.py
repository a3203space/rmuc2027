"""Fault-injected pipes only; no game physics, model or training."""
import os
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import Mock, patch

from rmuc2027_core import engine


@unittest.skipUnless(os.name == 'posix', 'POSIX pipe readiness required')
class PipeBackpressureTests(unittest.TestCase):
    def pipe(self):
        read, write = os.pipe()
        reader, writer = os.fdopen(read, 'rb', buffering=0), os.fdopen(write, 'wb', buffering=0)
        self.addCleanup(reader.close)
        self.addCleanup(writer.close)
        return reader, engine._PosixPipeWriter(writer)

    def test_full_pipe_write_has_a_deadline_before_response_read(self):
        _, writer = self.pipe()
        started = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, 'request write timed out'):
            writer.write_all(b'x' * 1048576, timeout=.05)
        self.assertLess(time.monotonic() - started, .4)

    def test_short_writes_preserve_payload_without_duplication(self):
        reader, writer = self.pipe()
        payload = bytes(range(256)) * 4096
        chunks = []
        def consume():
            remaining = len(payload)
            while remaining:
                chunk = reader.read(min(remaining, 3000))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
        worker = threading.Thread(target=consume, daemon=True)
        worker.start()
        writer.write_all(payload, timeout=2)
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(b''.join(chunks), payload)

    def test_request_timeout_poisoning_prevents_response_reuse_and_close_reaps_peer(self):
        peer = subprocess.Popen([sys.executable, '-c',
            'import time; print("ready", flush=True); time.sleep(10)'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        match = object.__new__(engine.Match)
        match.process = peer
        match.closed = False
        match._broken = False
        match._reader = None
        match.external_physics = None
        match._observation_cache = {}
        match.lock = threading.RLock()
        match._direct_writer = engine._PosixPipeWriter(peer.stdin)
        match._read_line = Mock(side_effect=AssertionError('Read must not be reached'))
        self.assertEqual(peer.stdout.readline(), b'ready\n')
        write = match._direct_writer.write_all
        started = time.monotonic()
        try:
            with patch.object(match._direct_writer, 'write_all', side_effect=lambda payload: write(payload, timeout=.05)):
                with self.assertRaisesRegex(RuntimeError, 'request write timed out'):
                    match._request({'op': 'snapshot', 'padding': 'x' * 1048576})
            self.assertTrue(match._broken)
            with self.assertRaisesRegex(RuntimeError, 'transport interrupted'):
                match.digest()
            match._read_line.assert_not_called()
        finally:
            match.close()
        self.assertIsNotNone(peer.poll())
        self.assertLess(time.monotonic() - started, 1)


if __name__ == '__main__':
    unittest.main()
