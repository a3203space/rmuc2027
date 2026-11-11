"""CPU fake integration with real service, executor, and thread-start failures."""
import threading
import time
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rmuc2027_research.rl.warp_kinematic import batching, vector


class FakeEngine:
    calls = []

    def __init__(self, device):
        assert device == 'cpu'

    def step_many(self, items, **kwargs):
        self.calls.append(len(items))
        return ['ok'] * len(items)

    def close(self):
        pass


class Client:
    device = 'cpu'
    scene_signature = design_signature = 'same'


class Env(gym.Env):
    observation_space = spaces.Box(-100., 100., (1,), dtype=np.float32)
    action_space = spaces.Box(-1., 1., (1,), dtype=np.float32)

    def __init__(self, error=None, after=0, terminated=False, truncated=False, close_error=None):
        self.error, self.after = error, after
        self.terminated, self.truncated = terminated, truncated
        self.close_error = close_error
        self.requests = self.steps = self.closes = 0
        self.resets = []
        self.finished = threading.Event()

    def reset(self, *, seed=None, options=None):
        self.resets.append((seed, options))
        return np.array([-1.], np.float32), {'reset': len(self.resets)}

    def step(self, action):
        self.steps += 1
        try:
            for i in range(3):
                if self.error is not None and i == self.after:
                    raise self.error
                batching.active_service.submit(Client(), {})
                self.requests += 1
            return np.array([self.steps], np.float32), 2., self.terminated, self.truncated, {'step': self.steps}
        finally:
            self.finished.set()

    def close(self):
        self.closes += 1
        if self.close_error is not None:
            raise self.close_error


class VectorFailureCleanupTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNone(batching.active_service)
        FakeEngine.calls = []
        self.engine_patch = patch.object(batching, 'WorldBatch', FakeEngine)
        self.engine_patch.start()
        self.addCleanup(self.engine_patch.stop)

    def make_vec(self, envs):
        vec = vector.WarpVecEnv([lambda env=env: env for env in envs])
        self.addCleanup(self.rescue, vec)
        return vec

    @staticmethod
    def rescue(vec):
        # A regression must fail promptly, not leave a 300-second submit wait.
        if getattr(vec, 'physics', None) is not None:
            vec.physics.close()
        if getattr(vec, 'pool', None) is not None:
            vec.pool.shutdown(wait=True, cancel_futures=True)
        try:
            vec.close()
        except BaseException:
            pass  # Deliberately failing cleanup is checked by the calling test.

    def bounded(self, vec, operation):
        outcome = {}

        def run():
            try:
                outcome['result'] = operation()
            except BaseException as error:
                outcome['error'] = error

        start = time.monotonic()
        caller = threading.Thread(target=run, name='vector-test-caller', daemon=True)
        caller.start()
        caller.join(1.)
        prompt = not caller.is_alive()
        if not prompt:
            vec.physics.close()
            caller.join(2.)
        self.assertFalse(caller.is_alive(), 'operation did not exit after test rescue')
        outcome.update(prompt=prompt, elapsed=time.monotonic() - start)
        return outcome

    def assert_stopped(self, vec, envs):
        self.assertFalse(vec.physics.thread.is_alive())
        self.assertTrue(all(not thread.is_alive() for thread in vec.pool._threads))
        self.assertEqual([env.closes for env in envs], [1] * len(envs))
        self.assertIsNone(batching.active_service)

    def startup_failure(self, fail_on, operation='step', close_errors=False):
        envs = [Env(close_error=ValueError('cleanup one') if close_errors else None),
                Env(close_error=ValueError('cleanup two') if close_errors else None)]
        vec = self.make_vec(envs)
        vec.pool._thread_name_prefix = 'vector-test-pool'
        primary = RuntimeError("can't start new thread (controlled fault)")
        request_queued = threading.Event()
        original_put = vec.physics.queue.put
        original_start = threading.Thread.start
        snapshot = {}
        starts = 0

        def put(item, *args, **kwargs):
            result = original_put(item, *args, **kwargs)
            if item is not None and len(item) == 3:
                request_queued.set()
            return result

        def start(thread):
            nonlocal starts
            if thread.name.startswith('vector-test-pool'):
                starts += 1
                if starts == fail_on:
                    if fail_on == 2:
                        self.assertTrue(request_queued.wait(.5), 'first worker never submitted physics')
                    # Real ThreadPoolExecutor.submit already put this work item.
                    work = vec.pool._work_queue.get_nowait()
                    vec.pool._work_queue.put(work)
                    snapshot.update(future=work.future, live_threads=len(vec.pool._threads),
                                    integration_calls=len(FakeEngine.calls))
                    raise primary
            return original_start(thread)

        def invoke():
            with patch.object(vec.physics.queue, 'put', put), patch.object(threading.Thread, 'start', start):
                if operation == 'reset':
                    return vec.reset()
                vec.step_async(np.zeros((2, 1), np.float32))
                return vec.step_wait()

        stepping = self.bounded(vec, invoke)
        closing = self.bounded(vec, vec.close)
        self.assertIs(stepping.get('error'), primary)
        self.assertTrue(stepping['prompt'], stepping)
        self.assertTrue(closing['prompt'], closing)
        self.assertNotIn('error', closing)
        self.assertEqual(snapshot['live_threads'], fail_on - 1)
        self.assertEqual(snapshot['integration_calls'], 0)
        orphan = snapshot['future']
        self.assertTrue(orphan.done(), 'queued work without a returned Future must be settled')
        if not orphan.cancelled():
            # The surviving worker can pick it up as physics.close wakes it.
            self.assertIsInstance(orphan.exception(), RuntimeError)
        self.assertEqual(envs[1].requests, 0)
        self.assert_stopped(vec, envs)
        if fail_on == 2:
            self.assertTrue(envs[0].finished.is_set())
        if close_errors:
            self.assertIn('cleanup one', '\n'.join(getattr(primary, '__notes__', [])))
            self.assertIn('cleanup two', '\n'.join(getattr(primary, '__notes__', [])))
        for retry in (lambda: vec.step_async(np.zeros((2, 1), np.float32)), vec.step_wait, vec.reset):
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                retry()
        vec.close()
        self.assert_stopped(vec, envs)

    def test_second_thread_start_failure_after_real_submit_queues_work(self):
        self.startup_failure(2)

    def test_first_thread_start_failure_cancels_orphaned_work(self):
        self.startup_failure(1)

    def test_reset_thread_start_failure_is_also_terminal(self):
        self.startup_failure(1, operation='reset')

    def test_submit_error_survives_multiple_environment_cleanup_errors(self):
        self.startup_failure(2, close_errors=True)

    def test_auto_reset_submission_failure_also_closes_vector(self):
        envs = [Env(terminated=True), Env(terminated=True)]
        vec = self.make_vec(envs)
        primary = RuntimeError('post-queue auto-reset submission failed')
        original_adjust = vec.pool._adjust_thread_count
        submissions = 0

        def adjust():
            nonlocal submissions
            submissions += 1
            if submissions == 3:
                raise primary
            return original_adjust()

        vec.step_async(np.zeros((2, 1), np.float32))
        with patch.object(vec.pool, '_adjust_thread_count', adjust):
            result = self.bounded(vec, vec.step_wait)
        self.assertTrue(result['prompt'], result)
        self.assertIs(result.get('error'), primary)
        self.assertEqual([env.requests for env in envs], [3, 3])
        self.assert_stopped(vec, envs)
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            vec.reset()

    def test_worker_errors_still_let_every_peer_finish(self):
        for error_type in (ValueError, TimeoutError, SystemExit):
            for after in (0, 1):
                for failed_index in (0, 1):
                    with self.subTest(error=error_type.__name__, after=after, failed_index=failed_index):
                        primary = error_type('controlled worker failure')
                        envs = [Env(), Env()]
                        envs[failed_index] = Env(primary, after)
                        vec = self.make_vec(envs)
                        vec.step_async(np.zeros((2, 1), np.float32))
                        stepping = self.bounded(vec, vec.step_wait)
                        closing = self.bounded(vec, vec.close)
                        self.assertIs(stepping.get('error'), primary)
                        self.assertTrue(stepping['prompt'], stepping)
                        self.assertTrue(closing['prompt'], closing)
                        self.assertNotIn('error', closing)
                        self.assertEqual(envs[failed_index].requests, after)
                        self.assertEqual(envs[1 - failed_index].requests, 3)
                        self.assert_stopped(vec, envs)

    def test_successful_rounds_keep_reset_and_observation_semantics(self):
        envs = [Env(terminated=True), Env(truncated=True)]
        vec = self.make_vec(envs)
        vec.seed(41)
        vec.set_options([{'id': 0}, {'id': 1}])
        np.testing.assert_array_equal(vec.reset(), [[-1.], [-1.]])
        self.assertEqual([env.resets for env in envs], [[(41, {'id': 0})], [(42, {'id': 1})]])
        for step in (1, 2):
            obs, rews, dones, infos = vec.step(np.zeros((2, 1), np.float32))
            np.testing.assert_array_equal(obs, [[-1.], [-1.]])
            np.testing.assert_array_equal(rews, [2., 2.])
            np.testing.assert_array_equal(dones, [True, True])
            self.assertEqual([info['TimeLimit.truncated'] for info in infos], [False, True])
            for info in infos:
                np.testing.assert_array_equal(info['terminal_observation'], [step])
            self.assertEqual(vec.reset_infos, [{'reset': step + 1}] * 2)
        self.assertEqual([env.requests for env in envs], [6, 6])
        vec.close()
        self.assert_stopped(vec, envs)

    def test_pool_construction_failure_closes_started_service(self):
        primary = RuntimeError('pool constructor failed')
        vec = vector.WarpVecEnv.__new__(vector.WarpVecEnv)
        with patch.object(vector, 'ThreadPoolExecutor', side_effect=primary):
            with self.assertRaises(RuntimeError) as caught:
                vec.__init__([Env])
        self.addCleanup(self.rescue, vec)
        self.assertIs(caught.exception, primary)
        self.assertFalse(vec.physics.thread.is_alive())
        self.assertIsNone(batching.active_service)
        vec.close()

    def test_service_thread_start_failure_preserves_error_and_can_close(self):
        primary = RuntimeError('service thread start failed')
        vec = vector.WarpVecEnv.__new__(vector.WarpVecEnv)
        with patch.object(threading.Thread, 'start', side_effect=primary):
            with self.assertRaises(RuntimeError) as caught:
                vec.__init__([Env])
        self.assertIs(caught.exception, primary)
        self.assertIsNone(vec.physics)
        self.assertIsNone(vec.pool)
        self.assertIsNone(batching.active_service)
        vec.close()

    def test_factory_failure_closes_prior_environment_and_preserves_error(self):
        primary = ValueError('second factory failed')
        first = Env(close_error=RuntimeError('factory cleanup failed'))
        vec = vector.WarpVecEnv.__new__(vector.WarpVecEnv)

        def fail():
            raise primary

        with self.assertRaises(ValueError) as caught:
            vec.__init__([lambda: first, fail])
        self.addCleanup(self.rescue, vec)
        self.assertIs(caught.exception, primary)
        self.assertIn('factory cleanup failed', '\n'.join(getattr(primary, '__notes__', [])))
        self.assert_stopped(vec, [first])
        vec.close()
        self.assertEqual(first.closes, 1)

    def test_duplicate_factory_result_is_only_closed_once(self):
        env = Env()
        vec = vector.WarpVecEnv.__new__(vector.WarpVecEnv)
        with self.assertRaisesRegex(ValueError, 'same instance'):
            vec.__init__([lambda: env, lambda: env])
        self.addCleanup(self.rescue, vec)
        self.assert_stopped(vec, [env])
        vec.close()
        self.assertEqual(env.closes, 1)

    def test_repeated_close_does_not_clear_replacement_service(self):
        env = Env()
        old = self.make_vec([env])
        old.close()
        replacement = self.make_vec([Env()])
        old.close()
        self.assertIs(batching.active_service, replacement.physics)
        self.assertEqual(env.closes, 1)
        replacement.close()

    def test_rejected_second_vector_cannot_close_active_service(self):
        active = self.make_vec([Env()])
        rejected = vector.WarpVecEnv.__new__(vector.WarpVecEnv)
        with self.assertRaisesRegex(RuntimeError, 'Only one'):
            rejected.__init__([Env])
        rejected.close()
        self.assertIs(batching.active_service, active.physics)
        self.assertTrue(active.physics.thread.is_alive())
        active.close()

    def test_normal_close_preserves_first_cleanup_error_and_closes_every_env(self):
        primary = ValueError('first close failed')
        envs = [Env(close_error=primary), Env(close_error=RuntimeError('second close failed'))]
        vec = self.make_vec(envs)
        result = self.bounded(vec, vec.close)
        self.assertTrue(result['prompt'], result)
        self.assertIs(result.get('error'), primary)
        self.assertIn('second close failed', '\n'.join(getattr(primary, '__notes__', [])))
        self.assert_stopped(vec, envs)
        vec.close()
        self.assertEqual([env.closes for env in envs], [1, 1])


if __name__ == '__main__':
    unittest.main()
