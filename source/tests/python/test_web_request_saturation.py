"""Shared-loop CPU tests: asynchronous step waiters must not occupy AnyIO tokens."""
import asyncio
from contextlib import AsyncExitStack
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import anyio.to_thread
import httpx
import test_web_server_lifecycle_faults as lifecycle
from test_web_standard_step import SlowBattle


async def until(predicate, timeout=2):
    end = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= end:
            raise AssertionError("Expected async test boundary was not reached")
        await asyncio.sleep(.005)


class RequestSaturationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = lifecycle.WebServerLifecycleFaultTests()
        self.fixture.setUp()
        self.service, self.app = self.fixture.service, self.fixture.client.app
        self.service.referee_token = "cpu-test-organizer"
        self.stack = AsyncExitStack()
        await self.stack.enter_async_context(self.app.router.lifespan_context(self.app))
        self.loop_ids, self.limiter_ids = set(), set()
        self.active_http = self.max_http = 0

        async def traced_app(scope, receive, send):
            self.loop_ids.add(id(asyncio.get_running_loop()))
            self.limiter_ids.add(id(anyio.to_thread.current_default_thread_limiter()))
            self.active_http += 1
            self.max_http = max(self.max_http, self.active_http)
            try:
                await self.app(scope, receive, send)
            finally:
                self.active_http -= 1

        self.client = await self.stack.enter_async_context(httpx.AsyncClient(
            transport=httpx.ASGITransport(app=traced_app), base_url="http://testserver"))
        self.limiter = anyio.to_thread.current_default_thread_limiter()
        self.default_tokens = self.limiter.total_tokens
        self.limiter.total_tokens = 4
        self.tasks, self.gates, self.battles = [], [], []

    async def asyncTearDown(self):
        self.service.stopping = True
        for gate in self.gates:
            gate.release.set()
        if self.tasks:
            await asyncio.wait_for(asyncio.gather(*self.tasks, return_exceptions=True), timeout=5)
        await self.join_worker()
        await self.stack.aclose()
        self.limiter.total_tokens = self.default_tokens
        for battle in self.battles:
            battle.frames.file.close()
        self.fixture.doCleanups()

    async def join_worker(self):
        if self.service.thread:
            await asyncio.to_thread(self.service.thread.join, 2)
            self.assertFalse(self.service.thread.is_alive())

    def gate(self):
        gate = lifecycle.Gate()
        self.gates.append(gate)
        return gate

    async def start_battle(self, gate, **kwargs):
        battle = SlowBattle(gate, **kwargs)
        self.battles.append(battle)
        self.fixture.stack.enter_context(patch("rmuc2027_research.web.server.Battle", return_value=battle))
        self.assertEqual((await self.client.post("/api/start", json={"duration":1})).status_code, 200)
        await until(gate.entered.is_set)
        return battle

    def waiter(self, index):
        if index % 2:
            call = self.client.post("/api/referee", json={"kind":"yellow_both"},
                                    headers={"Authorization":"Bearer cpu-test-organizer"})
        else:
            call = self.client.get("/api/replay")
        task = asyncio.create_task(call)
        self.tasks.append(task)
        return task

    async def controls(self, *commands):
        tasks = [asyncio.create_task(self.client.get("/api/state"))]
        tasks += [asyncio.create_task(self.client.post("/api/control", json={"command":command}))
                  for command in commands]
        self.tasks.extend(tasks)
        start = time.monotonic()
        _, pending = await asyncio.wait(tasks, timeout=.2)
        self.assertFalse(pending, "Shared AnyIO pool prevented state/controls from entering")
        self.assertTrue(all(task.result().status_code == 200 for task in tasks))
        return time.monotonic() - start

    async def saturation_case(self, count=8, cancel_count=3):
        gate = self.gate()
        await self.start_battle(gate)
        requests = [self.waiter(i) for i in range(count)]
        # On old code this detects saturated sync workers; new code registers every waiter.
        await until(lambda: len(getattr(self.service, "_step_waiters", ())) == count
                    or self.limiter.borrowed_tokens == self.limiter.total_tokens)
        latency = await self.controls("stop")
        await until(lambda: len(self.service._step_waiters) == count)
        borrowed = self.limiter.borrowed_tokens
        for task in requests[:cancel_count]:
            task.cancel()
        await asyncio.gather(*requests[:cancel_count], return_exceptions=True)
        await until(lambda: len(self.service._step_waiters) == count - cancel_count)
        self.assertTrue(all(not task.done() for task in requests[cancel_count:]))
        gate.release.set()
        results = await asyncio.wait_for(asyncio.gather(*requests[cancel_count:]), timeout=3)
        await self.join_worker()
        await until(lambda: not self.service._step_waiters and self.limiter.borrowed_tokens == 0)
        statuses = [response.status_code for response in results]
        self.assertTrue(all(code in (200, 409) for code in statuses))
        return dict(tokens=self.limiter.total_tokens, waiters=count, cancelled=cancel_count,
                    borrowed_while_waiting=borrowed, state_stop_seconds=latency,
                    remaining_statuses=statuses, waiters_after=len(self.service._step_waiters),
                    borrowed_after=self.limiter.borrowed_tokens, worker_joined=True,
                    distinct_loops=len(self.loop_ids), distinct_limiters=len(self.limiter_ids),
                    maximum_inflight_http=self.max_http, active_http_after=self.active_http)

    async def test_more_waiters_than_tokens_allow_controls_and_independent_cancellation(self):
        result = await self.saturation_case()
        self.assertEqual(result["borrowed_while_waiting"], 0)

    async def test_disconnect_removes_only_its_waiter(self):
        gate = self.gate()
        await self.start_battle(gate)
        survivor = self.waiter(0)
        await until(lambda: len(self.service._step_waiters) == 1)
        messages = asyncio.Queue()
        await messages.put({"type":"http.request", "body":b"", "more_body":False})
        sent = []

        async def send(message):
            sent.append(message)

        scope = dict(type="http", asgi={"version":"3.0"}, http_version="1.1", method="GET",
                     scheme="http", path="/api/replay", raw_path=b"/api/replay", query_string=b"",
                     root_path="", headers=[(b"host", b"testserver")], client=("127.0.0.1", 1),
                     server=("testserver", 80))
        disconnected = asyncio.create_task(self.app(scope, messages.get, send))
        self.tasks.append(disconnected)
        await until(lambda: len(self.service._step_waiters) == 2)
        await messages.put({"type":"http.disconnect"})
        await asyncio.wait_for(disconnected, timeout=1)
        await until(lambda: len(self.service._step_waiters) == 1)
        self.assertFalse(survivor.done())
        await self.controls("stop")
        gate.release.set()
        self.assertEqual((await asyncio.wait_for(survivor, timeout=3)).status_code, 200)

    async def test_cancellation_during_admitted_export_removes_temporary_file(self):
        battle = self.fixture.finished_battle()
        self.service.store.finish("previous", {"reason":"stopped"})
        gate = self.gate()
        paths = []

        def save(path):
            paths.append(Path(path))
            gate()
            Path(path).write_text('{"source":"completed-export"}')

        battle.save = save
        task = self.waiter(0)
        await until(gate.entered.is_set)
        task.cancel()
        await self.controls("stop")
        gate.release.set()
        await asyncio.gather(task, return_exceptions=True)
        self.assertTrue(task.cancelled())
        self.assertTrue(paths)
        self.assertTrue(all(not path.exists() for path in paths))
        self.assertFalse(self.service._step_waiters)

    async def test_step_and_close_errors_notify_all_remaining_waiters(self):
        gate, close_gate = self.gate(), self.gate()
        await self.start_battle(gate, fail=True, close_gate=close_gate, close_error=True)
        requests = [self.waiter(i) for i in range(8)]
        await until(lambda: len(self.service._step_waiters) == 8)
        requests[0].cancel()
        await asyncio.gather(requests[0], return_exceptions=True)
        gate.release.set()
        await until(close_gate.entered.is_set)
        await self.controls("stop")
        self.assertTrue(all(not task.done() for task in requests[1:]))
        close_gate.release.set()
        results = await asyncio.wait_for(asyncio.gather(*requests[1:]), timeout=3)
        self.assertTrue(all(response.status_code in (200, 409) for response in results))
        await self.join_worker()
        self.assertFalse(self.service._step_waiters)
        self.assertFalse(self.service._step_in_progress)
        self.assertFalse(self.service.busy)

    async def test_old_waiters_recheck_battle_identity_after_handoff(self):
        old_gate = self.gate()
        old = await self.start_battle(old_gate)
        old.match.frame["generation"] = "old"
        requests = [self.waiter(0), self.waiter(1)]
        await until(lambda: len(self.service._step_waiters) == 2)
        notify = self.service._notify_step_waiters
        with patch.object(self.service, "_notify_step_waiters"):
            await self.controls("stop")
            old_gate.release.set()
            await self.join_worker()
            new_gate = self.gate()
            new = await self.start_battle(new_gate)
            new.match.frame["generation"] = "new"

            def delayed_notify():
                with self.service.lock:
                    notify()

            await asyncio.to_thread(delayed_notify)
            replay, referee = await asyncio.wait_for(asyncio.gather(*requests), timeout=1)
            self.assertEqual(replay.status_code, 200)
            self.assertEqual(replay.json()["frame"]["generation"], "old")
            self.assertEqual(referee.status_code, 409)
            self.assertEqual(new.referee_commands, [])
            await self.controls("stop")
            new_gate.release.set()
            await self.join_worker()

    async def test_notification_does_not_bypass_a_new_step_in_same_battle(self):
        first, second = self.gate(), self.gate()

        class TwoSteps(SlowBattle):
            def step(self):
                self.gate, self.gate_at = (first if self.steps == 0 else second), self.steps + 1
                return super().step()

        battle = TwoSteps(first)
        self.battles.append(battle)
        self.fixture.stack.enter_context(patch("rmuc2027_research.web.server.Battle", return_value=battle))
        await self.client.post("/api/start", json={"duration":1})
        await until(first.entered.is_set)
        requests = [self.waiter(0), self.waiter(1)]
        await until(lambda: len(self.service._step_waiters) == 2)
        notify = self.service._notify_step_waiters
        with patch.object(self.service, "_notify_step_waiters"):
            await self.controls("pause")
            first.release.set()
            await until(lambda: self.service.frame["tick"] == 1)
            await self.controls("pause")
            await until(second.entered.is_set)

            def delayed_notify():
                with self.service.lock:
                    notify()

            await asyncio.to_thread(delayed_notify)
            await until(lambda: len(self.service._step_waiters) == 2)
            self.assertTrue(all(not task.done() for task in requests))
            self.assertEqual(battle.save_calls, 0)
            self.assertEqual(battle.referee_commands, [])
        await self.controls("stop")
        second.release.set()
        responses = await asyncio.wait_for(asyncio.gather(*requests), timeout=3)
        self.assertEqual([response.status_code for response in responses], [200, 409])


if __name__ == "__main__":
    unittest.main()
