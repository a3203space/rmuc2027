"""LAN ownership, projection, actual Rust control, and replay compatibility."""
import copy
import importlib.util
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import subprocess

from fastapi.testclient import TestClient
from pydantic import ValidationError

from rmuc2027_core.replay import verify_replay
from rmuc2027_core.engine import Match
from rmuc2027_server import SLOTS
from rmuc2027_server.control import HumanController, baseline_helpers
from rmuc2027_server.projection import project
from rmuc2027_server.protocol import Input, Join, Operation, Profile, Seat
from rmuc2027_server.room import Room
from rmuc2027_server.server import create_app


class LobbyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.room = Room(owner_key="host-secret", directory=self.temp.name, duration=1, ai=False)
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.room.close)

    def player(self, host=False, team="red", slot=3):
        result = self.room.join(Join(nickname="玩家", host_key="host-secret" if host else None))
        token = result["token"]
        generation = self.room.attach(token)
        self.room.select(token, Seat(team=team, slot=slot))
        self.room.ready(token, True)
        return token, generation

    def test_host_auth_seat_conflict_and_required_ready(self):
        host, _ = self.player(host=True)
        other = self.room.join(Join(nickname="另一人"))["token"]
        self.assertNotIn("token", self.room.state()["players"][0])
        with self.assertRaisesRegex(ValueError, "占用"):
            self.room.select(other, Seat(team="red", slot=3))
        with self.assertRaisesRegex(ValueError, "房主"):
            self.room.start(other)
        with self.assertRaisesRegex(ValueError, "准备"):
            self.room.start(host)
        self.room.leave(other)
        self.room.start(host)

    def test_reconnect_requires_resume_and_expiration(self):
        host, generation = self.player(host=True)
        self.room.status = "running"
        self.room.match_id = "game"
        self.room.detach(host, generation)
        self.room.attach(host)
        self.assertTrue(self.room.player(host).needs_resume)
        with self.assertRaisesRegex(ValueError, "活动连接"):
            self.room.attach(host)
        # A stale socket's finally handler cannot disconnect a newer connection.
        self.room.detach(host, generation)
        self.assertTrue(self.room.player(host).connected)
        self.room.detach(host, generation + 1)
        self.room.player(host).disconnected -= 31
        self.room.expire()
        self.assertEqual(self.room.status, "closed")

    def test_match_identity_monotonic_sequences_and_no_team_override(self):
        host, _ = self.player(host=True)
        self.room.status = "running"
        self.room.match_id = "game"
        with self.assertRaises(ValueError):
            self.room.submit(host, Input(match_id="old", seq=1, yaw=0, pitch=0))
        self.assertEqual(self.room.submit(host, Input(match_id="game", seq=1, yaw=0, pitch=0))["status"], "accepted")
        self.assertEqual(self.room.submit(host, Input(match_id="game", seq=1, yaw=1, pitch=0))["status"], "duplicate")
        with self.assertRaises(ValidationError):
            Input(match_id="game", seq=2, yaw=float("nan"), pitch=0)
        with self.assertRaises(ValidationError):
            Input(match_id="game", seq=2, yaw=0, pitch=0, team="blue")
        with self.assertRaises(ValueError):
            self.room.select(host, Seat(team="blue", slot=3))

    def test_concurrent_seat_claim_has_exactly_one_owner(self):
        tokens = [self.room.join(Join(nickname=str(index)))["token"] for index in range(2)]
        def claim(token):
            try:
                self.room.select(token, Seat(team="red", slot=4))
                return True
            except ValueError:
                return False
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.assertEqual(sum(pool.map(claim, tokens)), 1)

    def test_timeout_detection_does_not_restart_the_ai_grace_period(self):
        token, generation = self.player(host=True)
        self.room.player(token).seen -= 3.1
        self.room.detach(token, generation)
        self.assertTrue(self.room.state()["players"][0]["ai"])
        self.room.status = "running"
        self.room.attach(token)
        self.assertTrue(self.room.state()["players"][0]["ai"])

    def test_http_surface_and_websocket(self):
        # uvloop uses eventfd for thread wakeups, which also works in environments
        # that disallow socketpair.send; Windows uses the standard asyncio loop.
        options = {"use_uvloop": True} if importlib.util.find_spec("uvloop") else {}
        with TestClient(create_app(self.room), backend_options=options) as client:
            self.assertEqual(client.get("/api/lan/v1/health").json()["protocol"], 1)
            self.assertEqual(client.post("/api/submissions", json={}).status_code, 404)
            self.assertEqual(client.post("/api/lan/v1/start").status_code, 401)
            self.assertEqual(client.post("/api/lan/v1/join", json={"nickname": "x", "protocol": 2}).status_code, 422)
            self.assertEqual(client.post("/api/lan/v1/join", json={"nickname": "x"}, headers={"Origin": "http://other.test"}).status_code, 403)
            response = client.post("/api/lan/v1/join", json={"nickname": "网络玩家", "host_key": "host-secret"})
            token = response.json()["token"]
            with client.websocket_connect("/api/lan/v1/session") as socket:
                socket.send_json({"type": "auth", "protocol": 1, "token": token})
                value = socket.receive_json()
                self.assertEqual(value["type"], "snapshot")
                self.assertTrue(value["room"]["players"][0]["connected"])
                socket.send_json({"type": "ping"})

    def test_ten_websocket_sessions_get_their_own_seats(self):
        options = {"use_uvloop": True} if importlib.util.find_spec("uvloop") else {}
        with TestClient(create_app(self.room), backend_options=options) as client, ExitStack() as stack:
            sockets = []
            host = None
            for index, (team, slot) in enumerate((team, slot) for team in ("red", "blue") for slot in SLOTS):
                body = {"nickname": f"player-{index}"}
                if index == 0:
                    body["host_key"] = "host-secret"
                joined = client.post("/api/lan/v1/join", json=body).json()
                token = joined["token"]
                host = host or token
                socket = stack.enter_context(client.websocket_connect("/api/lan/v1/session"))
                socket.send_json({"type": "auth", "protocol": 1, "token": token})
                socket.receive_json()
                headers = {"Authorization": "Bearer " + token}
                self.assertEqual(client.post("/api/lan/v1/seat", json={"team": team, "slot": slot}, headers=headers).status_code, 200)
                self.assertEqual(client.post("/api/lan/v1/ready", json={"ready": True}, headers=headers).status_code, 200)
                sockets.append((socket, team, slot))
            started = client.post("/api/lan/v1/start", headers={"Authorization": "Bearer " + host})
            self.assertEqual(started.status_code, 200)
            match_id = started.json()["match_id"]
            for socket, team, slot in sockets:
                for _ in range(100):
                    value = socket.receive_json()
                    if value.get("player"):
                        break
                self.assertEqual((value["player"]["team"], value["player"]["slot"]), (team, slot))
                socket.send_json({"type": "input", "match_id": match_id, "seq": 1, "yaw": 0, "pitch": 0})
                self.assertNotIn("host_key", json.dumps(value))


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.match = Match(map_name="rmuc2027")
        self.addCleanup(self.match.close)
        self.frame = self.match.snapshot()
        self.obs = self.match.observation("red", [3])

    def test_radar_while_enemy_outside_camera(self):
        self.obs["vision"]["3"]["enemies"] = []
        value = project(self.frame, self.obs, "red", 3, "game")
        self.assertEqual(len(value["tactical"]["robots"]), 10)
        self.assertFalse(any(r["team"] == "blue" for r in value["scene"]["robots"]))
        radar = next(r for r in value["tactical"]["robots"] if r["team"] == "blue")
        self.assertIn("ammo", radar)
        self.assertIn("buffs", radar)
        self.assertIn("armor_hit_times", radar)
        self.assertNotIn("heat", radar)
        self.assertIn("total_resources", value["tactical"]["teams"]["blue"])

    def test_visible_enemy_does_not_expose_private_state(self):
        enemy = next(r for r in self.frame["robots"] if r["team"] == "blue")
        self.obs["vision"]["3"]["enemies"] = [enemy["id"]]
        value = project(self.frame, self.obs, "red", 3, "game")
        scene = next(r for r in value["scene"]["robots"] if r["id"] == enemy["id"])
        for key in ("heat", "ammo", "battery_remaining_wh", "pending_ammo", "drones", "posture_time"):
            self.assertNotIn(key, scene)

    def test_heat_logistics_and_yellow_card_have_distinct_scope(self):
        own = next(r for r in self.frame["robots"] if r["id"] == "red-3")
        own["heat"], own["overheated"] = 260, True
        value = project(self.frame, self.obs, "red", 3, "game")
        self.assertTrue(value["display"]["camera_off"])
        self.assertIn("tactical", value)
        self.assertEqual(value["scene"]["robots"], [])
        self.frame["bases"]["red"]["hp"] = 0
        value = project(self.frame, self.obs, "red", 3, "game")
        self.assertEqual(value["display"]["resolution_scale"], .5)
        own["fpv_blocked_until"] = 5
        value = project(self.frame, self.obs, "red", 3, "game")
        self.assertTrue(value["display"]["ui_blocked"])
        self.assertNotIn("tactical", value)
        self.assertNotIn("scene", value)
        self.assertNotIn("hp", value["self"])

    def test_hangar_has_no_drone_camera(self):
        value = project(self.frame, self.match.observation("red", [6]), "red", 6, "game")
        self.assertEqual(value["display"]["reason"], "hangar")
        self.assertIsNone(value["display"]["camera"])
        self.assertEqual(value["scene"]["robots"], [])


class RustMatchTests(unittest.TestCase):
    def test_host_shutdown_reaps_engine_and_both_ai_workers(self):
        processes = []
        original = subprocess.Popen
        def spawn(*args, **kwargs):
            process = original(*args, **kwargs)
            processes.append(process)
            return process
        with tempfile.TemporaryDirectory() as directory, patch("subprocess.Popen", side_effect=spawn):
            room = Room(owner_key="key", directory=directory, duration=1, ai=True)
            self.addCleanup(room.close)
            token = room.join(Join(nickname="房主", host_key="key"))["token"]
            room.attach(token)
            room.select(token, Seat(team="red", slot=3))
            room.ready(token, True)
            room.start(token)
            deadline = time.monotonic() + 10
            while room.status == "starting" and time.monotonic() < deadline:
                time.sleep(.01)
            room.close()
            self.assertFalse(room.runner.thread.is_alive())
            self.assertGreaterEqual(len(processes), 3)
            self.assertTrue(all(process.poll() is not None for process in processes))

    def test_same_room_can_start_a_fresh_match_without_old_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            room = Room(owner_key="key", directory=directory, duration=1, realtime=False, ai=False)
            self.addCleanup(room.close)
            token = room.join(Join(nickname="房主", host_key="key"))["token"]
            room.attach(token)
            room.select(token, Seat(team="red", slot=3))
            ids = []
            for _ in range(2):
                room.ready(token, True)
                room.start(token)
                ids.append(room.match_id)
                room.runner.thread.join(timeout=5)
                self.assertEqual(room.status, "finished")
                self.assertIsNone(room.error)
                self.assertEqual(room.player(token).seq, -1)
            self.assertNotEqual(ids[0], ids[1])
            self.assertEqual(len(list(Path(directory).glob("*.json"))), 2)
            self.assertEqual(list(Path(directory).glob("*.part")), [])

    def test_wheel_legged_chassis_can_steer_without_moving_the_gimbal(self):
        with Match(map_name="rmuc2027", duration=1) as match:
            controller = HumanController(baseline_helpers())
            for seq in range(20):
                observation = match.compact_observation("red", [3])
                robot = next(r for r in observation["robots"] if r["slot"] == 3)
                action = controller.action(observation, robot, Input(match_id="test", seq=seq, strafe=1, yaw=0, pitch=.04))
                match.step({"red": {"3": action}})
            robot = next(r for r in match.robots if r["id"] == "red-3")
            self.assertGreater(robot["heading"], .2)
            self.assertLess(min(robot["aim"], 6.283185307179586 - robot["aim"]), .1)

    def test_real_movement_ai_and_verifiable_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            room = Room(owner_key="key", directory=directory, duration=2, ai=True)
            self.addCleanup(room.close)
            token = room.join(Join(nickname="房主", host_key="key"))["token"]
            room.attach(token)
            room.select(token, Seat(team="red", slot=3))
            room.performance(token, Profile(chassis="power", weapon="burst"))
            room.ready(token, True)
            room.start(token)
            start = time.monotonic()
            position = None
            while room.status in ("starting", "running") and time.monotonic() - start < 20:
                if room.status == "running":
                    room.submit(token, Input(match_id=room.match_id, seq=room.player(token).seq + 1, forward=1, yaw=0, pitch=0))
                    snapshot = room.snapshot(token)["player"]
                    if snapshot and position is None:
                        position = snapshot["self"]["x"]
                time.sleep(.02)
            room.runner.thread.join(timeout=2)
            self.assertIsNone(room.error)
            self.assertEqual(room.status, "finished")
            self.assertGreater(room.snapshot(token)["player"]["self"]["x"], position + .1)
            robot = room.snapshot(token)["player"]["self"]
            self.assertEqual((robot["chassis_profile"], robot["weapon_profile"]), ("power", "burst"))
            replay = Path(directory) / room.result["replay"]
            metadata = json.loads(replay.read_text())["metadata"]
            self.assertEqual(set(metadata["teams"]), {"red", "blue"})
            self.assertEqual(metadata["match_mode"], "lan")
            self.assertEqual(verify_replay(replay)["verified_ticks"], 40)
            self.assertEqual(room.result["policy_errors"], 0)
            self.assertEqual(room.result["metrics"]["steps"], 40)

    def test_stale_input_stops_firing_and_movement(self):
        with tempfile.TemporaryDirectory() as directory:
            room = Room(owner_key="key", directory=directory, duration=1, ai=False)
            self.addCleanup(room.close)
            token = room.join(Join(nickname="房主", host_key="key"))["token"]
            room.attach(token)
            room.select(token, Seat(team="red", slot=3))
            room.ready(token, True)
            room.start(token)
            deadline = time.monotonic() + 10
            while room.status == "starting" and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertEqual(room.status, "running")
            room.submit(token, Input(match_id=room.match_id, seq=1, forward=1, yaw=0, pitch=0, fire=True))
            room.runner.thread.join(timeout=5)
            self.assertEqual(room.status, "finished")
            replay = json.loads((Path(directory) / room.result["replay"]).read_text())
            self.assertTrue(any(a["red"]["3"].get("fire") for a in replay["actions"][:7]))
            for action in replay["actions"][9:]:
                self.assertEqual(action["red"]["3"]["move"], [0, 0])
                self.assertFalse(action["red"]["3"].get("fire"))


if __name__ == "__main__":
    unittest.main()
