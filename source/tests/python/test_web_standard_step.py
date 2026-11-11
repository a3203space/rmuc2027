"""CPU fake standard Battle interleavings; no models, controllers or physics."""
from concurrent.futures import ThreadPoolExecutor, wait
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import patch

from rmuc2027_core.policy import FrameLog
import test_web_server_lifecycle_faults as lifecycle


class LiveMatch:
    def __init__(self):
        self.frame = dict(time=0., tick=0, robots=[], events=[], status="running", progress="initial")
        self.reason = None

    @property
    def status(self):
        return self.frame["status"]

    def snapshot(self):
        # Deliberately reuse live state to verify that publication detaches it.
        return self.frame

    def finish(self, _winner, reason):
        self.reason = reason
        self.frame["status"] = "finished"
        self.frame["events"] = [dict(type=reason, tick=self.frame["tick"])]


class SlowBattle(lifecycle.FakeBattle):
    def __init__(self, gate, *, gate_at=1, fail=False, close_gate=None, close_error=False):
        super().__init__()
        self.match = LiveMatch()
        self.frames = FrameLog(self.match.snapshot())
        self.metadata = dict(test_match=True, teams={}, phase=dict(value="initial"))
        self.gate, self.gate_at, self.fail, self.close_gate = gate, gate_at, fail, close_gate
        self.close_error = close_error
        self.in_step = False
        self.save_calls = 0
        self.referee_commands = []
        self.second_step = threading.Event()

    def step(self):
        self.steps += 1
        if self.steps == 2:
            self.second_step.set()
        self.in_step = True
        try:
            frame = self.match.frame
            frame.update(tick=self.steps, time=self.steps * .05, progress="partial",
                         events=[dict(type="unpublished", tick=self.steps)])
            self.metadata["phase"]["value"] = "partial"
            # Spool mutation/IO also happens before the whole step is published.
            with self.frames.lock:
                self.frames.append(frame)
                if self.steps == self.gate_at:
                    self.gate()
                if self.fail:
                    raise RuntimeError("injected standard step failure")
                frame.update(progress="complete", events=[dict(type="complete", tick=self.steps)])
                self.metadata["phase"]["value"] = "complete"
                self.frames[-1] = frame
            return frame
        finally:
            self.in_step = False

    def save(self, path):
        if self.in_step:
            raise AssertionError("Replay read a mutating Battle")
        self.save_calls += 1
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(dict(frame=self.match.snapshot(), commands=self.referee_commands)))

    def queue_referee(self, command):
        if self.in_step:
            raise AssertionError("Referee accessed a mutating Battle")
        self.referee_commands.append(command)

    def __exit__(self, *args):
        if self.close_gate:
            self.close_gate()
        self.closed = True
        if self.close_error:
            raise RuntimeError("injected standard close failure")
        return False


class StandardStepTests(unittest.TestCase):
    def setUp(self):
        self.case = lifecycle.WebServerLifecycleFaultTests()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.service, self.client = self.case.service, self.case.client
        self.gates, self.battles = [], []
        self.addCleanup(self.release_workers)

    def release_workers(self):
        self.service.stopping = True
        for gate in self.gates:
            gate.release.set()
        self.case.join_worker()
        self.assertFalse(self.service._step_in_progress, "Standard operation reservation leaked")
        for battle in self.battles:
            battle.frames.file.close()

    def gate(self):
        gate = lifecycle.Gate()
        self.gates.append(gate)
        return gate

    def start_battle(self, gate, **kwargs):
        battle = SlowBattle(gate, **kwargs)
        self.battles.append(battle)
        self.case.stack.enter_context(patch("rmuc2027_research.web.server.Battle", return_value=battle))
        self.assertEqual(self.case.start().status_code, 200)
        self.assertTrue(gate.entered.wait(1))
        return battle

    def controls(self, pool, *commands):
        requests = [pool.submit(self.client.get, "/api/state")]
        requests += [pool.submit(self.client.post, "/api/control", json=dict(command=value)) for value in commands]
        _, pending = wait(requests, timeout=.2)
        self.assertFalse(pending, "State/control requests blocked behind a standard step")
        for request in requests:
            self.assertEqual(request.result().status_code, 200)
        return requests[0].result().json()

    def test_slow_step_allows_state_pause_and_stop_and_preserves_terminal_frame(self):
        gate = self.gate()
        battle = self.start_battle(gate)
        with ThreadPoolExecutor(max_workers=3) as pool:
            try:
                state = self.controls(pool, "pause", "stop")
                self.assertEqual(state["frame"]["tick"], 0)
                self.assertEqual(state["events"], [])
                self.assertEqual(state["metadata"]["phase"]["value"], "initial")
                self.assertTrue(self.service.paused)
                self.assertTrue(self.service.stopping)
            finally:
                gate.release.set()
        self.case.join_worker()
        state = self.client.get("/api/state").json()
        self.assertEqual(battle.steps, 1)
        self.assertEqual(state["frame"]["status"], "finished")
        self.assertEqual(state["events"], [dict(type="stopped", tick=1)])
        self.assertFalse(state["busy"])

    def test_published_frame_events_and_metadata_stay_detached_during_next_step(self):
        gate = self.gate()
        battle = self.start_battle(gate, gate_at=2)
        with ThreadPoolExecutor(max_workers=2) as pool:
            try:
                state = self.controls(pool, "stop")
                self.assertEqual(state["frame"]["tick"], 1)
                self.assertEqual(state["frame"]["progress"], "complete")
                self.assertEqual(state["metadata"]["phase"]["value"], "complete")
                self.assertEqual(state["events"], [dict(type="complete", tick=1)])
                response = pool.submit(self.client.get, "/api/state?since=1").result(timeout=.2)
                self.assertEqual(response.json()["events"], [])
            finally:
                gate.release.set()
        self.case.join_worker()
        self.assertEqual(battle.steps, 2)
        # The already returned first frame must not change when step two completes.
        self.assertEqual(state["frame"]["tick"], 1)

    def test_replay_waiter_releases_state_lock_and_only_captures_whole_step(self):
        gate = self.gate()
        battle = self.start_battle(gate)
        with ThreadPoolExecutor(max_workers=4) as pool:
            replay = pool.submit(self.client.get, "/api/replay")
            try:
                self.controls(pool, "stop")
                self.assertFalse(replay.done())
                self.assertEqual(battle.save_calls, 0)
            finally:
                gate.release.set()
            response = replay.result(timeout=2)
        self.case.join_worker()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["frame"]["progress"], "complete")
        self.assertEqual(battle.steps, 1)

    def test_pause_waits_for_inflight_step_then_resume_advances(self):
        gate = self.gate()
        battle = self.start_battle(gate)
        with ThreadPoolExecutor(max_workers=3) as pool:
            try:
                self.controls(pool, "pause")
                replay = pool.submit(self.client.get, "/api/replay")
            finally:
                gate.release.set()
            self.assertEqual(replay.result(timeout=2).status_code, 200)
            self.assertFalse(battle.second_step.wait(.15), "Pause admitted another step")
            self.assertEqual(self.client.post("/api/control", json=dict(command="pause")).status_code, 200)
            self.assertTrue(battle.second_step.wait(1), "Resume did not admit the next step")
            self.assertEqual(self.client.post("/api/control", json=dict(command="stop")).status_code, 200)
        self.case.join_worker()

    def test_referee_waits_for_step_without_blocking_controls(self):
        gate = self.gate()
        battle = self.start_battle(gate)
        self.service.referee_token = "cpu-test-organizer"
        with ThreadPoolExecutor(max_workers=4) as pool:
            referee = pool.submit(self.client.post, "/api/referee", json=dict(kind="yellow_both"),
                                  headers={"Authorization": "Bearer cpu-test-organizer"})
            try:
                self.controls(pool, "pause")
                self.assertFalse(referee.done())
                self.assertEqual(battle.referee_commands, [])
            finally:
                gate.release.set()
            self.assertEqual(referee.result(timeout=2).status_code, 200)
        self.assertEqual(len(battle.referee_commands), 1)
        self.client.post("/api/control", json=dict(command="stop"))
        self.case.join_worker()

    def test_step_failure_unblocks_replay_after_cleanup_and_allows_retry(self):
        gate, close_gate = self.gate(), self.gate()
        battle = self.start_battle(gate, fail=True, close_gate=close_gate)
        with ThreadPoolExecutor(max_workers=4) as pool:
            replay = pool.submit(self.client.get, "/api/replay")
            try:
                self.controls(pool, "stop")
                gate.release.set()
                self.assertTrue(close_gate.entered.wait(1))
                self.controls(pool, "stop")
                self.assertFalse(replay.done())
                self.assertEqual(battle.save_calls, 0)
            finally:
                gate.release.set()
                close_gate.release.set()
            self.assertEqual(replay.result(timeout=2).status_code, 200)
        self.case.join_worker()
        self.assertFalse(self.service.busy)
        self.assertIn("injected standard step failure", self.service.error)
        with patch("rmuc2027_research.web.server.Battle", lifecycle.FakeBattle):
            self.assertEqual(self.case.start().status_code, 200)
            self.case.join_worker()
        self.assertIsNone(self.service.error)

    def test_event_window_and_terminal_replacement_keep_published_tick_semantics(self):
        self.service.speed = 1_000_000  # Skip wall-clock pacing for this fake history.
        gate = self.gate()
        self.start_battle(gate, gate_at=1002)
        with ThreadPoolExecutor(max_workers=2) as pool:
            try:
                state = self.controls(pool, "stop")
                self.assertEqual(state["frame"]["tick"], 1001)
                self.assertEqual([event["tick"] for event in state["events"]], list(range(2, 1002)))
                recent = pool.submit(self.client.get, "/api/state?since=999").result(timeout=.2).json()
                self.assertEqual([event["tick"] for event in recent["events"]], [1000, 1001])
            finally:
                gate.release.set()
        self.case.join_worker()
        state = self.client.get("/api/state").json()
        self.assertEqual([event["tick"] for event in state["events"]], list(range(3, 1003)))
        self.assertEqual(state["events"][-1], dict(type="stopped", tick=1002))

    def test_close_failure_notifies_replay_waiter_and_clears_reservation(self):
        gate, close_gate = self.gate(), self.gate()
        self.start_battle(gate, close_gate=close_gate, close_error=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            try:
                self.controls(pool, "stop")
                gate.release.set()
                self.assertTrue(close_gate.entered.wait(1))
                replay = pool.submit(self.client.get, "/api/replay")
                self.controls(pool, "stop")
                self.assertFalse(replay.done())
            finally:
                gate.release.set()
                close_gate.release.set()
            self.assertEqual(replay.result(timeout=2).status_code, 200)
        self.case.join_worker()
        self.assertFalse(self.service.busy)
        self.assertIn("injected standard close failure", self.service.error)


if __name__ == "__main__":
    unittest.main()
