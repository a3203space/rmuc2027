"""Match lifecycle failures with Python pipe peers; no game physics or GPU."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import patch

from rmuc2027_core import engine


PEER = r'''
import json, sys, time
mode, rules_hash = sys.argv[1:]
request = sys.stdin.readline()
if not request:
    sys.exit()
if mode == 'bad_init':
    print('{broken', flush=True)
else:
    print(json.dumps({'robots': [], 'rules_hash': rules_hash}), flush=True)
if mode == 'teardown_sleep':
    time.sleep(60)
for line in sys.stdin:
    request = json.loads(line)
    if mode == 'bad_json':
        print('{broken', flush=True)
    elif mode == 'eof':
        break
    elif mode == 'timeout':
        time.sleep(5)
    elif mode in ('callback', 'callback_encoded'):
        print(json.dumps({'op': 'kinematic_frame' if mode == 'callback_encoded' else 'external_physics'}), flush=True)
        if not sys.stdin.readline():
            break
        print(json.dumps({'digest': 'ok'}), flush=True)
    elif mode in ('strict_callback', 'strict_encoded'):
        if request['op'] == 'step':
            encoded = mode == 'strict_encoded'
            print(json.dumps({'op': 'kinematic_frame' if encoded else 'external_physics'}), flush=True)
            reply = sys.stdin.readline()
            if not reply:
                break
            if json.loads(reply).get('op') != ('kinematic_frame_result' if encoded else 'external_physics_result'):
                sys.exit(4)
        print(json.dumps({'digest': 'ok'}), flush=True)
    elif mode == 'batch_tail':
        print(json.dumps({'controlled_ids': [['3'], ['7']]}), flush=True)
        print(json.dumps({'controlled_ids': ['3']}), flush=True)
        time.sleep(5)
    elif mode.startswith('framing_'):
        if request['op'] == 'observation':
            print(json.dumps({'controlled_ids': ['1']}), flush=True)
            continue
        if request['op'] == 'observations':
            print(json.dumps({'controlled_ids': [['3'], ['7']]}), flush=True)
            print(json.dumps({'controlled_ids': ['3']}), flush=True)
            tail = json.dumps({'controlled_ids': ['7']})
        else:
            tail = json.dumps({'digest': 'ok'})
        if mode == 'framing_complete':
            print(tail, flush=True)
        else:
            sys.stdout.write(tail[:-1] if mode == 'framing_partial_json' else tail)
            sys.stdout.flush()
            break
    elif request['op'] == 'invalid':
        print(json.dumps({'error': 'invalid request'}), flush=True)
    else:
        print(json.dumps({'digest': 'ok'}), flush=True)
'''


class Adapter:
    protocol = 'external_physics'

    def __init__(self, *, step_error=None, close_error=None):
        self.step_error = step_error
        self.close_error = close_error
        self.close_calls = 0
        self.entered = threading.Event()
        self.release = threading.Event()
        self.closed_event = threading.Event()
        self.active = False
        self.close_during_step = False

    def step(self, request):
        self.active = True
        self.entered.set()
        try:
            if self.step_error is not None:
                raise self.step_error
            if not self.release.wait(2):
                raise TimeoutError('Test did not release callback')
            if self.close_calls:
                raise RuntimeError('Adapter closed during step')
            return {'op': 'external_physics_result'}
        finally:
            self.active = False

    def close(self):
        self.close_during_step |= self.active
        self.close_calls += 1
        self.closed_event.set()
        if self.close_error is not None:
            raise self.close_error


@unittest.skipUnless(os.name == 'posix', 'POSIX fake pipe peers required')
class MatchLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.children = []
        self.actual_popen = subprocess.Popen
        self.rules_hash = hashlib.sha256((engine.ROOT / 'configs/rules.json').read_bytes()).hexdigest()
        self.addCleanup(self.cleanup_children)

    def cleanup_children(self):
        for child in self.children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=2)
            for stream in (child.stdin, child.stdout):
                if stream is not None:
                    stream.close()

    def spawn(self, mode):
        def factory(*args, **kwargs):
            process = self.actual_popen([sys.executable, '-u', '-c', PEER, mode, self.rules_hash], **kwargs)
            self.children.append(process)
            return process
        return factory

    def match(self, mode='ok', adapter=None, direct=True):
        with patch.object(engine, 'binary_path', return_value=Path(sys.executable)), \
             patch.object(engine.subprocess, 'Popen', side_effect=self.spawn(mode)), \
             patch.object(engine, '_DIRECT_PIPE_READER', direct):
            return engine.Match(physics_adapter=adapter)

    def assert_reaped(self):
        for child in self.children:
            self.assertIsNotNone(child.poll(), 'Constructor/close left its peer alive')
            self.assertTrue(child.stdin.closed)
            self.assertTrue(child.stdout.closed)

    def bounded_close(self, match):
        errors = []
        def close():
            try:
                match.close()
            except BaseException as error:
                errors.append(error)
            else:
                errors.append(None)
        worker = threading.Thread(target=close, daemon=True)
        worker.start()
        worker.join(1)
        rescued = worker.is_alive()
        if rescued:
            # Bypass the injected instance method; rescue only this test's peer.
            self.actual_popen.kill(match.process)
            worker.join(2)
        self.assertFalse(worker.is_alive(), 'Close did not unwind after killing its owned peer')
        return errors[0], rescued

    def test_interrupted_transport_close_can_retry_and_rejects_requests(self):
        for target, error_type in (('wait', KeyboardInterrupt), ('kill', OSError)):
            for direct in (True, False):
                with self.subTest(target=target, direct=direct):
                    adapter = Adapter()
                    match = self.match('teardown_sleep', adapter, direct=direct)
                    if target == 'kill':
                        match._broken = True
                    original = getattr(match.process, target)
                    injected = error_type('injected teardown interruption')
                    calls = []
                    def fail_once(*args, **kwargs):
                        calls.append((args, kwargs))
                        if len(calls) == 1:
                            raise injected
                        return original(*args, **kwargs)
                    with patch.object(match.process, target, side_effect=fail_once):
                        first, first_rescued = self.bounded_close(match)
                        self.assertIs(first, injected)
                        self.assertTrue(match.closed)
                        with self.assertRaisesRegex(RuntimeError, 'closed'):
                            match.digest()
                        second, second_rescued = self.bounded_close(match)
                        calls_before_repeat = len(calls)
                        match.close()
                        self.assertEqual(len(calls), calls_before_repeat)
                    self.assertIsNone(second)
                    self.assertFalse(first_rescued or second_rescued, 'Teardown blocked and needed watchdog rescue')
                    self.assertGreaterEqual(len(calls), 2)
                    if target == 'wait':
                        self.assertTrue(all(kwargs.get('timeout') is not None for _, kwargs in calls))
                    self.assertEqual(adapter.close_calls, 1)
                    if match._reader is not None:
                        self.assertFalse(match._reader.is_alive())
                    self.assert_reaped()

    def test_persistent_kill_failure_stays_bounded_and_can_later_retry(self):
        adapter = Adapter()
        match = self.match('teardown_sleep', adapter, direct=False)
        match._broken = True
        injected = OSError('injected persistent kill refusal')
        with patch.object(match.process, 'kill', side_effect=injected) as kill:
            for _ in range(2):
                observed, rescued = self.bounded_close(match)
                self.assertIs(observed, injected)
                self.assertFalse(rescued, 'Failed kill must not block closing threaded stdout')
                self.assertTrue(match.closed)
                self.assertIsNone(match.process.poll())
                with self.assertRaisesRegex(RuntimeError, 'closed'):
                    match.digest()
            self.assertEqual(kill.call_count, 2)
        observed, rescued = self.bounded_close(match)
        self.assertIsNone(observed)
        self.assertFalse(rescued)
        match.close()
        self.assertEqual(adapter.close_calls, 1)
        self.assertFalse(match._reader.is_alive())
        self.assert_reaped()

    def test_adapter_cleanup_cannot_start_a_new_request(self):
        adapter = Adapter()
        match = self.match(adapter=adapter)
        original = adapter.close
        errors = []
        def close_adapter():
            try:
                match.digest()
            except RuntimeError as error:
                errors.append(error)
            original()
        with patch.object(adapter, 'close', side_effect=close_adapter):
            match.close()
        self.assertEqual(len(errors), 1)
        self.assertIn('closed', str(errors[0]))
        self.assertEqual(adapter.close_calls, 1)
        self.assert_reaped()

    def test_binary_lookup_and_spawn_failure_release_owned_adapter(self):
        for stage in ('lookup', 'popen'):
            with self.subTest(stage=stage):
                adapter = Adapter()
                error = OSError('injected startup failure')
                with patch.object(engine, 'binary_path', side_effect=error if stage == 'lookup' else None,
                                  return_value=Path(sys.executable)), \
                     patch.object(engine.subprocess, 'Popen', side_effect=error) as popen:
                    with self.assertRaises(OSError) as raised:
                        engine.Match(physics_adapter=adapter)
                self.assertIs(raised.exception, error)
                self.assertEqual(adapter.close_calls, 1)
                if stage == 'lookup':
                    popen.assert_not_called()

    def test_reader_setup_failure_reaps_child_while_traceback_is_retained(self):
        adapter = Adapter()
        error = OSError('injected reader setup failure')
        retained = None
        with patch.object(engine, 'binary_path', return_value=Path(sys.executable)), \
             patch.object(engine.subprocess, 'Popen', side_effect=self.spawn('ok')), \
             patch.object(engine, '_PosixLineReader', side_effect=error):
            try:
                engine.Match(physics_adapter=adapter)
            except OSError as caught:
                retained = caught
        self.assertIs(retained, error)
        self.assertIsNotNone(retained.__traceback__)
        self.assertEqual(adapter.close_calls, 1)
        self.assert_reaped()

    def test_thread_start_failure_cleans_up_unstarted_reader(self):
        adapter = Adapter()
        error = RuntimeError('injected thread start failure')
        with patch.object(threading.Thread, 'start', side_effect=error):
            with self.assertRaises(RuntimeError) as raised:
                self.match(adapter=adapter, direct=False)
        self.assertIs(raised.exception, error)
        self.assertEqual(adapter.close_calls, 1)
        self.assert_reaped()

    def test_thread_start_interrupt_after_reader_enters_still_joins_reader(self):
        adapter = Adapter(close_error=RuntimeError('injected cleanup failure'))
        error = KeyboardInterrupt('injected interrupt after reader start')
        started_readers = []
        entered = threading.Event()
        actual_start = threading.Thread.start
        actual_read = engine.Match._read_output

        def read(stream, responses):
            entered.set()
            actual_read(stream, responses)

        def start_then_interrupt(reader):
            started_readers.append(reader)
            actual_start(reader)
            self.assertTrue(entered.wait(1))
            raise error

        retained = None
        with patch.object(engine.Match, '_read_output', staticmethod(read)), \
             patch.object(threading.Thread, 'start', start_then_interrupt):
            try:
                self.match(adapter=adapter, direct=False)
            except KeyboardInterrupt as caught:
                retained = caught
        self.assertIs(retained, error)
        self.assertIsNotNone(retained.__traceback__)
        self.assertEqual(len(started_readers), 1)
        self.assertFalse(started_readers[0].is_alive())
        self.assertEqual(adapter.close_calls, 1)
        if hasattr(error, 'add_note'):
            self.assertTrue(any('injected cleanup failure' in note for note in error.__notes__))
        self.assert_reaped()

    def test_constructor_cleanup_failure_does_not_mask_initial_error(self):
        adapter = Adapter(close_error=RuntimeError('injected cleanup failure'))
        with self.assertRaises(ValueError) as raised:
            self.match(mode='bad_init', adapter=adapter)
        self.assertNotIn('injected cleanup failure', str(raised.exception))
        if hasattr(raised.exception, 'add_note'):
            self.assertTrue(any('injected cleanup failure' in note for note in raised.exception.__notes__))
        self.assertEqual(adapter.close_calls, 1)
        self.assert_reaped()

    def test_invalid_json_eof_and_read_timeout_poison_then_close_reaps_peer(self):
        for mode in ('bad_json', 'eof', 'timeout'):
            with self.subTest(mode=mode):
                match = self.match(mode)
                read = match._direct_reader.read_line
                started = time.monotonic()
                try:
                    with patch.object(match._direct_reader, 'read_line', side_effect=lambda: read(timeout=.05)):
                        with self.assertRaises((ValueError, RuntimeError)):
                            match.digest()
                    with self.assertRaisesRegex(RuntimeError, 'transport interrupted'):
                        match.digest()
                finally:
                    match.close()
                self.assertLess(time.monotonic() - started, 1)
                self.assert_reaped()

    def test_native_error_response_preserves_synchronized_transport(self):
        match = self.match()
        try:
            with self.assertRaisesRegex(ValueError, 'invalid request'):
                match._request({'op': 'invalid'})
            self.assertEqual(match.digest(), 'ok')
        finally:
            match.close()
        self.assert_reaped()

    def test_partial_observation_batch_poisoning_reaps_peer_on_close(self):
        match = self.match('batch_tail')
        read = match._direct_reader.read_line
        try:
            with patch.object(match._direct_reader, 'read_line', side_effect=lambda: read(timeout=.05)):
                with self.assertRaisesRegex(RuntimeError, 'timed out'):
                    match.observations([('red', [3]), ('blue', [7])])
            with self.assertRaisesRegex(RuntimeError, 'transport interrupted'):
                match.digest()
        finally:
            match.close()
        self.assert_reaped()

    def test_eof_without_newline_rejects_requests_and_cached_observations(self):
        for direct in (True, False):
            for operation in ('request', 'batch'):
                for mode in ('framing_partial_json', 'framing_valid_json'):
                    with self.subTest(direct=direct, operation=operation, mode=mode):
                        adapter = Adapter()
                        match = self.match(mode, adapter, direct=direct)
                        try:
                            match.observation('red', [1])
                            cache_before = dict(match._observation_cache)
                            self.assertTrue(cache_before)
                            started = time.monotonic()
                            with self.assertRaisesRegex(RuntimeError, 'newline'):
                                if operation == 'request':
                                    match.digest()
                                else:
                                    match.observations([('red', [3]), ('blue', [7])])
                            self.assertLess(time.monotonic() - started, 1)
                            self.assertTrue(match._broken)
                            self.assertEqual(match._observation_cache, cache_before)
                            for query in (match.digest,
                                          lambda: match.observation('red', [1]),
                                          lambda: match.observations([('red', [1])]),
                                          lambda: match.observations([])):
                                with self.assertRaisesRegex(RuntimeError, 'transport interrupted'):
                                    query()
                        finally:
                            match.close()
                        self.assertEqual(adapter.close_calls, 1)
                        if match._reader is not None:
                            self.assertFalse(match._reader.is_alive())
                        self.assert_reaped()

    def test_complete_newline_frames_preserve_queries_and_batch_cache(self):
        for direct in (True, False):
            with self.subTest(direct=direct):
                match = self.match('framing_complete', direct=direct)
                try:
                    self.assertEqual(match.digest(), 'ok')
                    viewers = [('red', [3]), ('blue', [7])]
                    ids, lines = match.observations(viewers)
                    self.assertEqual(ids, [['3'], ['7']])
                    self.assertTrue(all(line.endswith(b'\n') for line in lines))
                    self.assertEqual([json.loads(line) for line in lines],
                                     [{'controlled_ids': ['3']}, {'controlled_ids': ['7']}])
                    with patch.object(match, '_write_payload', wraps=match._write_payload) as write:
                        self.assertEqual(match.observations(viewers), (ids, lines))
                        write.assert_not_called()
                    self.assertEqual(match.digest(), 'ok')
                    self.assertFalse(match._broken)
                finally:
                    match.close()
                if match._reader is not None:
                    self.assertFalse(match._reader.is_alive())
                self.assert_reaped()

    def test_callback_and_adapter_cleanup_failures_still_reap_peer(self):
        callback_error = ValueError('injected callback failure')
        adapter = Adapter(step_error=callback_error, close_error=RuntimeError('injected cleanup failure'))
        match = self.match('callback', adapter)
        with self.assertRaises(ValueError) as raised:
            match._request({'op': 'step'})
        self.assertIs(raised.exception, callback_error)
        with self.assertRaisesRegex(RuntimeError, 'transport interrupted'):
            match.digest()
        with self.assertRaisesRegex(RuntimeError, 'injected cleanup failure'):
            match.close()
        match.close()
        self.assertEqual(adapter.close_calls, 1)
        self.assert_reaped()

    def test_context_cleanup_failure_preserves_callback_error(self):
        callback_error = ValueError('injected callback failure')
        adapter = Adapter(step_error=callback_error, close_error=RuntimeError('injected cleanup failure'))
        with self.assertRaises(ValueError) as raised:
            with self.match('callback', adapter) as match:
                match._request({'op': 'step'})
        self.assertIs(raised.exception, callback_error)
        if hasattr(callback_error, 'add_note'):
            self.assertTrue(any('injected cleanup failure' in note for note in callback_error.__notes__))
        self.assertEqual(adapter.close_calls, 1)
        self.assert_reaped()

    def test_cleanup_diagnostic_failures_preserve_primary_and_close_resources(self):
        class UnprintableCleanupError(RuntimeError):
            def __str__(self):
                hook_calls.append('cleanup_str')
                raise diagnostic_error

        class BrokenNoteError(ValueError):
            def add_note(self, note):
                hook_calls.append('primary_add_note')
                raise diagnostic_error

        read_line = engine.Match._read_line
        for phase in ('constructor', 'context'):
            for hook in ('cleanup_str', 'primary_add_note'):
                for diagnostic_type in (RuntimeError, KeyboardInterrupt):
                    for direct in (True, False):
                        with self.subTest(phase=phase, hook=hook, diagnostic_type=diagnostic_type, direct=direct):
                            diagnostic_error = diagnostic_type('injected diagnostic failure')
                            hook_calls, matches = [], []
                            primary = BrokenNoteError('original failure') if hook == 'primary_add_note' else ValueError('original failure')
                            cleanup = UnprintableCleanupError() if hook == 'cleanup_str' else RuntimeError('cleanup failure')
                            adapter = Adapter(close_error=cleanup)

                            def fail_init(match):
                                matches.append(match)
                                read_line(match)
                                raise primary

                            observed = None
                            try:
                                if phase == 'constructor':
                                    with patch.object(engine.Match, '_read_line', fail_init):
                                        self.match(adapter=adapter, direct=direct)
                                else:
                                    match = self.match(adapter=adapter, direct=direct)
                                    matches.append(match)
                                    with match:
                                        raise primary
                            except BaseException as error:
                                observed = error
                            self.assertEqual(len(matches), 1)
                            match = matches[0]
                            self.assertTrue(match.closed)
                            match.close()
                            match.close()
                            self.assertEqual(adapter.close_calls, 1)
                            if match._reader is not None:
                                self.assertFalse(match._reader.is_alive())
                            self.assert_reaped()
                            self.assertEqual(hook_calls, [hook])
                            self.assertIs(observed, primary)
                            self.assertIsNotNone(primary.__traceback__)

    def test_cleanup_error_without_primary_still_propagates(self):
        for phase in ('explicit_close', 'normal_context'):
            for cleanup_type in (RuntimeError, KeyboardInterrupt):
                with self.subTest(phase=phase, cleanup_type=cleanup_type):
                    class CleanupError(cleanup_type):
                        def __str__(self):
                            raise AssertionError('Cleanup without a primary must not be formatted')

                    cleanup = CleanupError('original cleanup failure')
                    adapter = Adapter(close_error=cleanup)
                    match = self.match(adapter=adapter, direct=False)
                    with self.assertRaises(cleanup_type) as raised:
                        if phase == 'explicit_close':
                            match.close()
                        else:
                            with match:
                                pass
                    self.assertIs(raised.exception, cleanup)
                    match.close()
                    match.close()
                    self.assertEqual(adapter.close_calls, 1)
                    self.assertFalse(match._reader.is_alive())
                    self.assert_reaped()

    def test_reentrant_cleanup_closes_once_and_preserves_context_error_order(self):
        class ReentrantAdapter(Adapter):
            def close(self):
                super().close()
                self.match.close()
                raise cleanup_error

        for exceptional_body in (False, True):
            with self.subTest(exceptional_body=exceptional_body):
                cleanup_error = RuntimeError('injected reentrant cleanup failure')
                body_error = ValueError('injected context body failure')
                adapter = ReentrantAdapter()
                match = adapter.match = self.match(adapter=adapter)
                expected = body_error if exceptional_body else cleanup_error
                with self.assertRaises(type(expected)) as raised:
                    with match:
                        if exceptional_body:
                            raise body_error
                self.assertIs(raised.exception, expected)
                if exceptional_body and hasattr(body_error, 'add_note'):
                    self.assertTrue(any('injected reentrant cleanup failure' in note for note in body_error.__notes__))
                match.close()
                self.assertEqual(adapter.close_calls, 1)
                self.assert_reaped()

    def test_concurrent_close_waits_for_callback_and_closes_adapter_once(self):
        adapter = Adapter()
        match = self.match('callback', adapter)
        errors, responses = [], []
        def request():
            try:
                responses.append(match._request({'op': 'step'}))
            except BaseException as error:
                errors.append(error)
        def close():
            try:
                match.close()
            except BaseException as error:
                errors.append(error)
        requesting = threading.Thread(target=request, daemon=True)
        closers = [threading.Thread(target=close, daemon=True) for _ in range(2)]
        try:
            requesting.start()
            self.assertTrue(adapter.entered.wait(1))
            for closer in closers:
                closer.start()
            self.assertFalse(adapter.closed_event.wait(.05), 'Cleanup raced the active callback')
        finally:
            adapter.release.set()
            requesting.join(2)
            for closer in closers:
                if closer.ident is not None:
                    closer.join(3)
            match.close()
        self.assertFalse(requesting.is_alive())
        self.assertTrue(all(not closer.is_alive() for closer in closers))
        self.assertEqual(errors, [])
        self.assertEqual(responses, [{'digest': 'ok'}])
        self.assertFalse(adapter.close_during_step)
        self.assertEqual(adapter.close_calls, 1)
        self.assert_reaped()

    def test_callback_cannot_close_its_own_active_adapter(self):
        class ClosingAdapter(Adapter):
            def step(self, request):
                self.active = True
                try:
                    self.match.close()
                    return {'op': 'external_physics_result'}
                finally:
                    self.active = False

            def step_encoded(self, request):
                self.step(request)
                return b'{"op":"kinematic_frame_result"}\n'

        for mode in ('callback', 'callback_encoded'):
            for direct in (True, False):
                with self.subTest(mode=mode, direct=direct):
                    adapter = ClosingAdapter()
                    if mode == 'callback_encoded':
                        adapter.protocol = 'kinematic_frame'
                    match = adapter.match = self.match(mode, adapter, direct=direct)
                    caught = None
                    try:
                        try:
                            match._request({'op': 'step'})
                        except BaseException as error:
                            caught = error
                        self.assertFalse(adapter.close_during_step, 'Reentrant close disposed the active callback')
                        self.assertEqual(adapter.close_calls, 0)
                        self.assertIsInstance(caught, RuntimeError)
                        self.assertIn('active physics callback', str(caught))
                        self.assertTrue(match._broken)
                        self.assertFalse(match.closed)
                        self.assertEqual(match._physics_callback_depth, 0)
                    finally:
                        match.close()
                    self.assertEqual(adapter.close_calls, 1)
                    self.assert_reaped()

    def test_callback_rpc_reentry_is_rejected_before_writing(self):
        class QueryingAdapter(Adapter):
            def step(self, request):
                # A purely local read must remain usable within callbacks.
                self.local_snapshot = self.match.snapshot()
                try:
                    self.match.digest()
                except RuntimeError as error:
                    self.inner_error = error
                    if not self.catch_error:
                        raise
                return {'op': 'external_physics_result'}

            def step_encoded(self, request):
                self.step(request)
                return b'{"op":"kinematic_frame_result"}\n'

        for encoded in (False, True):
            for direct in (True, False):
                for catch_error in (False, True):
                    with self.subTest(encoded=encoded, direct=direct, catch_error=catch_error):
                        adapter = QueryingAdapter()
                        adapter.catch_error = catch_error
                        adapter.inner_error = None
                        if encoded:
                            adapter.protocol = 'kinematic_frame'
                        match = adapter.match = self.match('strict_encoded' if encoded else 'strict_callback', adapter, direct)
                        writes = []
                        actual_write = match._write_payload
                        def write(payload):
                            writes.append(json.loads(payload)['op'])
                            return actual_write(payload)
                        try:
                            with patch.object(match, '_write_payload', side_effect=write):
                                if catch_error:
                                    self.assertEqual(match._request({'op': 'step'}), {'digest': 'ok'})
                                    self.assertFalse(match._broken)
                                    self.assertEqual(match.digest(), 'ok')
                                    self.assertEqual(writes, ['step', 'kinematic_frame_result' if encoded else 'external_physics_result', 'digest'])
                                else:
                                    with self.assertRaises(RuntimeError) as raised:
                                        match._request({'op': 'step'})
                                    self.assertIs(raised.exception, adapter.inner_error)
                                    self.assertTrue(match._broken)
                                    with self.assertRaisesRegex(RuntimeError, 'transport interrupted'):
                                        match.digest()
                                    self.assertEqual(writes, ['step'])
                            self.assertIn('active physics callback', str(adapter.inner_error))
                            self.assertEqual(adapter.local_snapshot['robots'], [])
                            self.assertEqual(match._physics_callback_depth, 0)
                        finally:
                            match.close()
                            match.close()
                        self.assertEqual(adapter.close_calls, 1)
                        if match._reader is not None:
                            self.assertFalse(match._reader.is_alive())
                        self.assert_reaped()

    def test_other_thread_query_waits_for_physics_callback(self):
        adapter = Adapter()
        match = self.match('strict_callback', adapter, direct=False)
        writes, responses, errors = [], {}, []
        query_started = threading.Event()
        query_done = threading.Event()
        actual_write = match._write_payload
        def write(payload):
            writes.append(json.loads(payload)['op'])
            return actual_write(payload)
        def step():
            try:
                responses['step'] = match._request({'op': 'step'})
            except BaseException as error:
                errors.append(error)
        def query():
            query_started.set()
            try:
                responses['query'] = match.digest()
            except BaseException as error:
                errors.append(error)
            finally:
                query_done.set()
        stepping = threading.Thread(target=step, daemon=True)
        querying = threading.Thread(target=query, daemon=True)
        try:
            with patch.object(match, '_write_payload', side_effect=write):
                stepping.start()
                self.assertTrue(adapter.entered.wait(1))
                querying.start()
                self.assertTrue(query_started.wait(1))
                self.assertFalse(query_done.wait(.02))
                self.assertEqual(writes, ['step'])
                adapter.release.set()
                stepping.join(2)
                querying.join(2)
            self.assertFalse(stepping.is_alive() or querying.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(responses, {'step': {'digest': 'ok'}, 'query': 'ok'})
            self.assertEqual(writes, ['step', 'external_physics_result', 'digest'])
            self.assertEqual(match._physics_callback_depth, 0)
        finally:
            adapter.release.set()
            stepping.join(2)
            if querying.ident is not None:
                querying.join(2)
            match.close()
        self.assertEqual(adapter.close_calls, 1)
        self.assert_reaped()

    def test_failed_match_can_be_closed_and_recreated_without_state_or_reader_leak(self):
        for direct in (True, False):
            with self.subTest(direct=direct):
                failed = self.match('bad_json', direct=direct)
                with self.assertRaises(ValueError):
                    failed.digest()
                failed.close()
                if failed._reader is not None:
                    self.assertFalse(failed._reader.is_alive())
                fresh = self.match(direct=direct)
                self.assertEqual(fresh.digest(), 'ok')
                fresh.close()
                fresh.close()
                with self.assertRaisesRegex(RuntimeError, 'closed'):
                    fresh.digest()
                if fresh._reader is not None:
                    self.assertFalse(fresh._reader.is_alive())
                self.assert_reaped()


if __name__ == '__main__':
    unittest.main()
