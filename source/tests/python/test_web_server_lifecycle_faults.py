"""Bounded CPU fake-engine faults through the real FastAPI lifecycle routes."""
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from rmuc2027_research.web.server import create_app


class Gate:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self):
        self.entered.set()
        if not self.release.wait(5):
            raise TimeoutError("Fake operation was not released")


class FakeMatch:
    def __init__(self, *args, **kwargs):
        self.status = "running"
        self.reason = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def snapshot(self):
        return dict(time=0., tick=0, robots=[], events=[], status=self.status)

    def finish(self, _winner, reason):
        self.status, self.reason = "finished", reason


class FakeFrames:
    def __init__(self):
        self.lock = threading.RLock()
        self.events = [[]]

    def events_since(self, _since):
        return []

    def __setitem__(self, _index, _frame):
        pass


class FakeBattle:
    def __init__(self, *args, **kwargs):
        self.match = FakeMatch()
        self.frames = FakeFrames()
        self.metadata = dict(test_match=True, teams={})
        self.steps = 0
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True
        return False

    def step(self):
        self.steps += 1
        self.match.finish(None, "timeout")
        return self.match.snapshot()

    def result(self):
        return dict(reason=self.match.reason)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(source="new-battle", result=self.result())))


class WebServerLifecycleFaultTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch("rmuc2027_research.web.server.Match", FakeMatch))
        self.battles = []

        def new_battle(*args, **kwargs):
            battle = FakeBattle(*args, **kwargs)
            self.battles.append(battle)
            return battle

        # Every construction path is fake, including unexpected worker starts.
        self.stack.enter_context(patch("rmuc2027_research.web.server.Battle", side_effect=new_battle))
        self.stack.enter_context(patch("rmuc2027_core.deployment.deployment_settings", return_value={}))
        self.stack.enter_context(patch("rmuc2027_research.web.server.load_submission", return_value=({}, None)))
        app = create_app(test_mode=True, data_root=self.root / "web", research_root=self.root / "research")
        self.service = app.state.arena
        self.service.resolve_season = lambda args: (
            args.model_copy(update=dict(map="rmuc2027", red="red", blue="blue")), "2027")
        self.service.catalog = lambda *args, **kwargs: {
            side: dict(path=self.root, name=side, mode="team", version="fake") for side in ("red", "blue")}
        self.client = TestClient(app)
        self.stack.callback(self.client.close)
        self.stack.callback(self.join_worker)

    def join_worker(self):
        if self.service.thread is not None:
            self.service.thread.join(2)
            self.assertFalse(self.service.thread.is_alive(), "Fake match worker leaked")
        self.assertFalse(self.service.initializing, "Initialization reservation leaked")

    def start(self):
        return self.client.post("/api/start", json=dict(duration=1))

    def assert_responsive(self, pool, *, busy):
        state = pool.submit(self.client.get, "/api/state")
        stop = pool.submit(self.client.post, "/api/control", json=dict(command="stop"))
        _, pending = wait((state, stop), timeout=.2)
        self.assertFalse(pending, "State/stop did not both finish within 200 ms")
        self.assertEqual(state.result().status_code, 200)
        self.assertEqual(stop.result().status_code, 200)
        self.assertEqual(state.result().json()["busy"], busy)
        self.assertTrue(self.service.stopping)

    def finished_battle(self):
        battle = FakeBattle()
        battle.match.finish(None, "stopped")
        self.service.battle = battle
        self.service.match_id = "previous"
        self.service.metadata = battle.metadata
        self.service.store.create("previous", battle.metadata)
        return battle

    def test_research_close_keeps_reservation_without_blocking_state_or_stop(self):
        gate = Gate()
        battle = FakeBattle()
        metadata = dict(test_match=True, match_mode="research", teams={})
        frozen = Mock(done=True, battle=battle, metadata=metadata)
        frozen.frame.return_value = battle.match.snapshot()
        frozen.result.return_value = dict(reason="stopped")
        frozen.close.side_effect = gate
        self.service.busy = True
        self.service.match_id = "research"
        self.service.metadata = metadata
        self.service.store.create("research", metadata)
        with patch("rmuc2027_research.web.server.FrozenResearchMatch", return_value=frozen):
            with ThreadPoolExecutor(max_workers=4) as pool:
                worker = pool.submit(self.service._run_research, "research", {})
                try:
                    self.assertTrue(gate.entered.wait(1))
                    self.assert_responsive(pool, busy=True)
                    self.assertEqual(pool.submit(self.start).result(timeout=.2).status_code, 409)
                    self.assertIs(self.service.battle, battle)
                    self.assertFalse(worker.done())
                finally:
                    gate.release.set()
                worker.result(timeout=2)
        self.assertFalse(self.service.busy)
        self.assertIsNone(self.service.battle)
        self.assertIsNone(self.service.prematch)
        self.assertIsNone(self.service.error)
        frozen.close.assert_called_once()

    def test_research_close_error_releases_reservation_and_retains_prior_error(self):
        for save_error in (None, RuntimeError("save failed")):
            with self.subTest(save_error=save_error):
                match_id = "close-failure-" + str(bool(save_error))
                battle = FakeBattle()
                metadata = dict(test_match=True, match_mode="research", teams={})
                frozen = Mock(done=True, battle=battle, metadata=metadata)
                frozen.frame.return_value = battle.match.snapshot()
                frozen.result.return_value = dict(reason="stopped")
                frozen.save.side_effect = save_error
                frozen.close.side_effect = RuntimeError("close failed")
                self.service.busy, self.service.error = True, None
                self.service.store.create(match_id, metadata)
                with patch("rmuc2027_research.web.server.FrozenResearchMatch", return_value=frozen):
                    self.service._run_research(match_id, {})
                self.assertFalse(self.service.busy)
                self.assertIsNone(self.service.battle)
                self.assertIsNone(self.service.prematch)
                self.assertIn("Match cleanup failed: close failed", self.service.error)
                if save_error:
                    self.assertIn("save failed", self.service.error)
                self.assertEqual(self.client.get("/api/state").status_code, 200)
        self.assertEqual(self.start().status_code, 200)
        self.join_worker()

    def test_completed_replay_serves_immutable_archive_without_save(self):
        battle = self.finished_battle()
        battle.save = Mock(side_effect=AssertionError("Archive must not be re-exported"))
        archive = self.root / "archived.json"
        content = b'{"source":"archive", "immutable":true}\n'
        archive.write_bytes(content)
        self.service.store.finish("previous", dict(reason="stopped"), archive)
        response = self.client.get("/api/replay")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, content)
        battle.save.assert_not_called()

    def test_previous_archive_during_next_initialization_never_exports_old_battle(self):
        battle = self.finished_battle()
        battle.save = Mock(side_effect=AssertionError("Archive must not be re-exported"))
        archive = self.root / "archived.json"
        archive.write_text('{"source":"immutable-archive"}')
        self.service.store.finish("previous", dict(reason="stopped"), archive)
        previous_frame = self.service.frame
        init_gate = Gate()

        def initial_match(*args, **kwargs):
            init_gate()
            return FakeMatch(*args, **kwargs)

        with patch("rmuc2027_research.web.server.Match", side_effect=initial_match) as constructor:
            with ThreadPoolExecutor(max_workers=4) as pool:
                start = pool.submit(self.start)
                try:
                    self.assertTrue(init_gate.entered.wait(1))
                    self.assertTrue(self.service.initializing)
                    response = pool.submit(self.client.get, "/api/replay").result(timeout=.2)
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json(), dict(source="immutable-archive"))
                    battle.save.assert_not_called()
                    self.assert_responsive(pool, busy=True)
                    self.assertIs(self.service.frame, previous_frame)
                    self.assertIs(self.service.battle, battle)
                    self.assertEqual(pool.submit(self.start).result(timeout=.2).status_code, 409)
                    self.assertEqual(constructor.call_count, 1)
                finally:
                    init_gate.release.set()
                self.assertEqual(start.result(timeout=2).status_code, 200)
        self.join_worker()
        self.assertTrue(self.service.stopping)
        self.assertEqual(self.battles[-1].steps, 0)
        self.assertFalse(self.service.busy)

    def test_previous_fallback_during_next_initialization_is_unlocked_and_pinned(self):
        battle = self.finished_battle()
        self.service.store.finish("previous", dict(reason="stopped"))
        init_gate, save_gate = Gate(), Gate()
        paths = []

        def initial_match(*args, **kwargs):
            init_gate()
            return FakeMatch(*args, **kwargs)

        def save(path):
            paths.append(Path(path))
            save_gate()
            Path(path).write_text('{"source":"previous-battle"}')

        battle.save = Mock(side_effect=save)
        with patch("rmuc2027_research.web.server.Match", side_effect=initial_match) as constructor:
            with ThreadPoolExecutor(max_workers=4) as pool:
                start = pool.submit(self.start)
                try:
                    self.assertTrue(init_gate.entered.wait(1))
                    self.assertTrue(self.service.initializing)
                    replay = pool.submit(self.client.get, "/api/replay")
                    self.assertTrue(save_gate.entered.wait(1))
                    self.assert_responsive(pool, busy=True)
                    self.assertEqual(pool.submit(self.start).result(timeout=.2).status_code, 409)
                    self.assertEqual(constructor.call_count, 1)
                    init_gate.release.set()
                    self.assertEqual(start.result(timeout=2).status_code, 200)
                    self.join_worker()
                    self.assertIsNot(self.service.battle, battle)
                    self.assertFalse(replay.done())
                finally:
                    save_gate.release.set()
                    init_gate.release.set()
                response = replay.result(timeout=2)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), dict(source="previous-battle"))
        battle.save.assert_called_once()
        self.assertFalse(paths[0].exists())
        self.assertTrue(self.service.stopping)
        self.assertEqual(self.battles[-1].steps, 0)

    def test_active_standard_replay_still_holds_state_lock_for_atomic_capture(self):
        battle = FakeBattle()
        self.service.busy = True
        self.service.battle, self.service.metadata = battle, battle.metadata
        gate = Gate()

        def save(path):
            gate()
            Path(path).write_text('{"source":"active-battle"}')

        battle.save = save
        with ThreadPoolExecutor(max_workers=1) as pool:
            replay = pool.submit(self.client.get, "/api/replay")
            try:
                self.assertTrue(gate.entered.wait(1))
                acquired = self.service.lock.acquire(blocking=False)
                if acquired:
                    self.service.lock.release()
                self.assertFalse(acquired, "Active standard replay must capture atomically")
            finally:
                gate.release.set()
            response = replay.result(timeout=2)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), dict(source="active-battle"))
        self.service.busy = False

    def test_completed_fallback_save_allows_state_stop_and_new_start(self):
        battle = self.finished_battle()
        self.service.store.finish("previous", dict(reason="stopped"))
        gate, paths = Gate(), []

        def save(path):
            paths.append(Path(path))
            gate()
            Path(path).write_text('{"source":"previous-battle"}')

        battle.save = Mock(side_effect=save)
        with ThreadPoolExecutor(max_workers=4) as pool:
            replay = pool.submit(self.client.get, "/api/replay")
            try:
                self.assertTrue(gate.entered.wait(1))
                self.assert_responsive(pool, busy=False)
                self.assertEqual(pool.submit(self.start).result(timeout=.2).status_code, 200)
                self.join_worker()
                self.assertIsNot(self.service.battle, battle)
                self.assertFalse(replay.done())
            finally:
                gate.release.set()
            response = replay.result(timeout=2)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), dict(source="previous-battle"))
        battle.save.assert_called_once()
        self.assertFalse(paths[0].exists(), "Temporary replay was not removed")

    def test_completed_fallback_pins_battle_before_archive_lookup(self):
        battle = self.finished_battle()
        self.service.store.finish("previous", dict(reason="stopped"))
        gate = Gate()

        def lookup(match_id):
            self.assertEqual(match_id, "previous")
            gate()
            return None

        battle.save = Mock(side_effect=lambda path: Path(path).write_text('{"source":"captured"}'))
        with patch.object(self.service.store, "replay_path", side_effect=lookup):
            with ThreadPoolExecutor(max_workers=2) as pool:
                replay = pool.submit(self.client.get, "/api/replay")
                try:
                    self.assertTrue(gate.entered.wait(1))
                    self.assertEqual(pool.submit(self.start).result(timeout=.2).status_code, 200)
                    self.join_worker()
                    self.assertIsNot(self.service.battle, battle)
                finally:
                    gate.release.set()
                response = replay.result(timeout=2)
        self.assertEqual(response.json(), dict(source="captured"))
        battle.save.assert_called_once()

    def test_completed_fallback_save_error_removes_temporary_file(self):
        battle = self.finished_battle()
        paths = []

        def save(path):
            paths.append(Path(path))
            Path(path).write_text("partial")
            raise RuntimeError("export failed")

        battle.save = save
        with self.assertRaisesRegex(RuntimeError, "export failed"):
            self.client.get("/api/replay")
        self.assertFalse(paths[0].exists())
        self.assertFalse(self.service.busy)

    def test_standard_initialization_is_reserved_and_stop_survives_all_slow_phases(self):
        for stage in ("construct", "snapshot", "close"):
            with self.subTest(stage=stage):
                self.join_worker()
                gate = Gate()

                class SlowMatch(FakeMatch):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        if stage == "construct":
                            gate()

                    def snapshot(self):
                        if stage == "snapshot":
                            gate()
                        return super().snapshot()

                    def __exit__(self, *args):
                        if stage == "close":
                            gate()
                        return False

                with patch("rmuc2027_research.web.server.Match", side_effect=SlowMatch) as constructor:
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        start = pool.submit(self.start)
                        try:
                            self.assertTrue(gate.entered.wait(1))
                            self.assert_responsive(pool, busy=True)
                            if self.service.match_id is None:
                                replay = pool.submit(self.client.get, "/api/replay").result(timeout=.2)
                                self.assertEqual(replay.status_code, 404)
                            conflict = pool.submit(self.start).result(timeout=.2)
                            self.assertEqual(conflict.status_code, 409)
                            self.assertEqual(constructor.call_count, 1)
                            self.assertFalse(start.done())
                        finally:
                            gate.release.set()
                        response = start.result(timeout=2)
                self.assertEqual(response.status_code, 200)
                self.join_worker()
                self.assertTrue(self.service.stopping)
                self.assertFalse(self.service.busy)
                self.assertEqual(self.battles[-1].steps, 0)
                self.assertTrue(self.battles[-1].closed)
                self.assertEqual(self.service.store.match(response.json()["match_id"])["status"], "stopped")

    def test_standard_initialization_failure_rolls_back_and_can_retry(self):
        for stage in ("construct", "snapshot", "close"):
            with self.subTest(stage=stage):
                previous = (self.service.frame, self.service.battle, self.service.match_id,
                            self.service.metadata, self.service.thread)
                count = len(self.service.store.matches())

                class FailedMatch(FakeMatch):
                    def __init__(self, *args, **kwargs):
                        super().__init__(*args, **kwargs)
                        if stage == "construct":
                            raise RuntimeError("initialization failed")

                    def snapshot(self):
                        if stage == "snapshot":
                            raise RuntimeError("initialization failed")
                        return super().snapshot()

                    def __exit__(self, *args):
                        if stage == "close":
                            raise RuntimeError("initialization failed")
                        return False

                with patch("rmuc2027_research.web.server.Match", FailedMatch):
                    response = self.start()
                self.assertEqual(response.status_code, 409)
                self.assertIn("initialization failed", response.json()["detail"])
                self.assertFalse(self.service.busy)
                self.assertFalse(self.service.initializing)
                self.assertIn("initialization failed", self.service.error or "")
                self.assertEqual((self.service.frame, self.service.battle, self.service.match_id,
                                  self.service.metadata, self.service.thread), previous)
                self.assertEqual(len(self.service.store.matches()), count)
                self.assertEqual(self.start().status_code, 200)
                self.join_worker()
                self.assertIsNone(self.service.error)
                self.assertEqual(self.battles[-1].steps, 1)

    def test_standard_initialization_failure_after_stop_releases_reservation(self):
        gate = Gate()

        def construct(*args, **kwargs):
            gate()
            raise RuntimeError("initialization failed after stop")

        with patch("rmuc2027_research.web.server.Match", side_effect=construct):
            with ThreadPoolExecutor(max_workers=3) as pool:
                start = pool.submit(self.start)
                try:
                    self.assertTrue(gate.entered.wait(1))
                    self.assert_responsive(pool, busy=True)
                finally:
                    gate.release.set()
                self.assertEqual(start.result(timeout=2).status_code, 409)
        self.assertTrue(self.service.stopping)
        self.assertFalse(self.service.busy)
        self.assertFalse(self.service.initializing)
        self.assertIsNone(self.service.thread)
        self.assertFalse(self.battles)
        self.assertEqual(self.start().status_code, 200)
        self.join_worker()
        self.assertFalse(self.service.stopping)
        self.assertEqual(self.battles[-1].steps, 1)

    def test_standard_admission_errors_roll_back_reservation_and_published_state(self):
        for stage in ("journal", "thread_construct", "thread_start"):
            with self.subTest(stage=stage):
                previous = (self.service.frame, self.service.battle, self.service.match_id,
                            self.service.metadata, self.service.thread)
                count = len(self.service.store.matches())
                with ExitStack() as faults:
                    if stage == "journal":
                        faults.enter_context(patch.object(self.service.store, "create", side_effect=RuntimeError(stage)))
                    else:
                        thread_factory = Mock()
                        if stage == "thread_construct":
                            thread_factory.side_effect = RuntimeError(stage)
                        else:
                            thread_factory.return_value.start.side_effect = RuntimeError(stage)
                        # Replace only the server's reference, not threading used by TestClient.
                        faults.enter_context(patch("rmuc2027_research.web.server.threading",
                                                   SimpleNamespace(Thread=thread_factory)))
                    response = self.start()
                self.assertEqual(response.status_code, 409)
                self.assertEqual(response.json()["detail"], stage)
                self.assertFalse(self.service.busy)
                self.assertFalse(self.service.initializing)
                self.assertEqual(self.service.error, stage)
                self.assertEqual((self.service.frame, self.service.battle, self.service.match_id,
                                  self.service.metadata, self.service.thread), previous)
                records = self.service.store.matches()
                self.assertEqual(len(records), count + (stage == "thread_start"))
                if stage == "thread_start":
                    self.assertEqual(records[0]["status"], "error")
                    self.assertEqual(records[0]["error"], stage)
                self.assertEqual(self.start().status_code, 200)
                self.join_worker()


if __name__ == "__main__":
    unittest.main()
