"""Thread-safe lobby and a single authoritative, real-time match loop."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
import copy
import json
import math
import secrets
import threading
import time
import uuid

from rmuc2027_core.engine import DT, Match
from rmuc2027_core.models import mirrored_lineups
from rmuc2027_core.paths import ROOT, data_path
from rmuc2027_core.policy import FrameLog, Policy
from rmuc2027_core.submissions import load_submission
from . import LINEUP, PROTOCOL, SLOTS
from .control import HumanController, baseline_helpers
from .projection import project
from .protocol import Input


@dataclass
class Player:
    token: str
    nickname: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    team: str | None = None
    slot: int | None = None
    ready: bool = False
    profile: dict = field(default_factory=lambda: {"chassis": "hp", "weapon": "cooling"})
    connected: bool = False
    seen: float = field(default_factory=time.monotonic)
    disconnected: float | None = None
    generation: int = 0
    seq: int = -1
    processed_seq: int = -1
    input_at: float = 0.
    intent: Input | None = None
    needs_resume: bool = False
    operations: list = field(default_factory=list)
    receipts: list = field(default_factory=list)


class Room:
    def __init__(self, owner_key=None, directory=None, *, duration=420, realtime=True, ai=True):
        self.owner_key = owner_key or secrets.token_urlsafe(32)
        self.directory = Path(directory or data_path("replays/lan"))
        self.duration, self.realtime, self.ai = duration, realtime, ai
        self.lock = threading.RLock()
        self.players = {}
        self.owner = None
        self.status, self.error = "lobby", None
        self.match_id = ""
        self.revision = 0
        self.runner = None
        self.latest = {}
        self.static = None
        self.result = None
        self.history = []
        self.stop_event = threading.Event()

    def player(self, token):
        if token not in self.players:
            raise ValueError("会话不存在或已过期，请重新加入")
        return self.players[token]

    def join(self, request):
        with self.lock:
            self.expire()
            if self.status == "closed":
                raise ValueError("房间已关闭")
            if request.token:
                p = self.player(request.token)
                return {"token": p.token, "player_id": p.id, "room": self.state()}
            if request.host_key and not secrets.compare_digest(request.host_key, self.owner_key):
                raise ValueError("房主口令不正确")
            if len(self.players) >= 10:
                raise ValueError("房间已满（最多 10 人）")
            token = secrets.token_urlsafe(32)
            p = Player(token, request.nickname)
            self.players[token] = p
            if request.host_key and secrets.compare_digest(request.host_key, self.owner_key):
                if self.owner is not None:
                    del self.players[token]
                    raise ValueError("房主会话已建立，请使用原会话重连")
                self.owner = token
            self.changed("join", p)
            return {"protocol": PROTOCOL, "token": token, "player_id": p.id, "room": self.state()}

    def changed(self, kind, p=None):
        self.revision += 1
        self.history.append({"kind": kind, "player": p.id if p else None,
                             "team": p.team if p else None, "slot": p.slot if p else None,
                             "tick": self.runner.tick if self.runner else 0})

    def state(self):
        with self.lock:
            now = time.monotonic()
            return {"protocol": PROTOCOL, "status": self.status, "revision": self.revision,
                    "match_id": self.match_id, "season": "2027", "slots": list(SLOTS),
                    "lineup": LINEUP, "duration": self.duration, "error": self.error,
                    "result": copy.deepcopy(self.result), "manual_sentry": True,
                    "players": [{"id": p.id, "nickname": p.nickname, "team": p.team, "slot": p.slot,
                                 "host": p.token == self.owner, "ready": p.ready, "connected": p.connected,
                                 "profile": p.profile,
                                 "needs_resume": p.needs_resume,
                                 "ai": p.needs_resume or (not p.connected and now - p.seen >= 3),
                                 "reserved_seconds": max(0, round(30 - (now - (p.disconnected if p.disconnected is not None else p.seen)))) if not p.connected else 0}
                                for p in self.players.values()]}

    def select(self, token, seat):
        with self.lock:
            p = self.player(token)
            self.expire()
            if self.status not in ("lobby", "finished", "running"):
                raise ValueError("当前不能选择席位")
            if self.status == "running" and p.slot is not None and (p.team, p.slot) != (seat.team, seat.slot):
                raise ValueError("比赛中不能换队或换席位")
            if any(other is not p and (other.team, other.slot) == (seat.team, seat.slot) for other in self.players.values()):
                raise ValueError("该席位已被占用或预留")
            p.team, p.slot, p.ready = seat.team, seat.slot, False
            p.intent, p.input_at = None, 0
            self.changed("seat", p)
            return self.state()

    def ready(self, token, value):
        with self.lock:
            p = self.player(token)
            if self.status not in ("lobby", "finished") or p.slot is None:
                raise ValueError("请先在大厅选择席位")
            p.ready = value
            self.changed("ready", p)
            return self.state()

    def performance(self, token, profile):
        with self.lock:
            p = self.player(token)
            if self.status not in ("lobby", "finished") or p.slot not in (3, 4):
                raise ValueError("仅步兵可在准备阶段选择性能")
            p.profile = profile.model_dump()
            p.ready = False
            self.changed("profile", p)
            return self.state()

    def attach(self, token):
        with self.lock:
            p = self.player(token)
            if p.connected:
                raise ValueError("该会话已有活动连接")
            reconnect = p.disconnected is not None
            p.connected, p.seen, p.disconnected = True, time.monotonic(), None
            p.generation += 1
            p.intent, p.input_at = None, 0
            p.needs_resume = reconnect and self.status == "running"
            self.changed("reconnect" if reconnect else "connect", p)
            return p.generation

    def detach(self, token, generation):
        with self.lock:
            p = self.players.get(token)
            if p and p.generation == generation:
                p.connected, p.disconnected = False, time.monotonic()
                p.intent, p.input_at = None, 0
                p.operations.clear()
                self.changed("disconnect", p)

    def touch(self, token):
        with self.lock:
            self.player(token).seen = time.monotonic()

    def expire(self):
        now = time.monotonic()
        for token, p in list(self.players.items()):
            if not p.connected and now - (p.disconnected if p.disconnected is not None else p.seen) > 30:
                if token == self.owner:
                    self.status = "closed"
                    self.stop_event.set()
                self.changed("expire", p)
                self.latest.pop(p.id, None)
                if self.runner:
                    self.runner.controllers.pop(p.id, None)
                del self.players[token]

    def leave(self, token):
        with self.lock:
            p = self.player(token)
            if token == self.owner:
                self.status = "closed"
                self.stop_event.set()
            self.changed("leave", p)
            self.latest.pop(p.id, None)
            if self.runner:
                self.runner.controllers.pop(p.id, None)
            del self.players[token]
            return self.state()

    def start(self, token):
        with self.lock:
            if token != self.owner:
                raise ValueError("只有房主可以开局")
            if self.status not in ("lobby", "finished") or (self.runner and self.runner.thread.is_alive()):
                raise ValueError("比赛尚未结束")
            humans = list(self.players.values())
            if not humans or not all(p.connected and p.ready and p.slot is not None for p in humans):
                raise ValueError("所有玩家需要连接、选位并准备")
            self.stop_event.clear()
            self.match_id = uuid.uuid4().hex
            self.latest, self.static, self.result, self.error = {}, None, None, None
            self.status = "starting"
            for p in humans:
                p.seq, p.processed_seq, p.intent, p.input_at = -1, -1, None, 0
                p.operations.clear()
                p.receipts.clear()
                p.needs_resume = False
            self.changed("start")
            self.runner = LanMatchRunner(self)
            self.runner.thread.start()
            return self.state()

    def submit(self, token, message):
        with self.lock:
            p = self.player(token)
            if not p.connected or p.slot is None or self.status != "running" or message.match_id != self.match_id:
                raise ValueError("当前会话不能控制该场比赛")
            if message.seq <= p.seq:
                return {"type": "ack", "seq": message.seq, "status": "duplicate"}
            p.seq, p.seen = message.seq, time.monotonic()
            if isinstance(message, Input):
                if not message.active and p.intent and p.intent.active and self.runner:
                    controller = self.runner.controllers.get(p.id)
                    if controller:
                        controller.mode = ""
                p.intent, p.input_at = message, p.seen
            else:
                if len(p.operations) >= 8:
                    raise ValueError("操作过于频繁")
                p.operations.append(message)
            return {"type": "ack", "seq": message.seq, "status": "accepted"}

    def snapshot(self, token, static=False):
        with self.lock:
            p = self.player(token)
            value = {"type": "snapshot", "room": self.state(), "player_id": p.id,
                     "ack_seq": p.processed_seq, "player": self.latest.get(p.id), "operations": p.receipts[-8:]}
            if static and self.static:
                value["static"] = self.static
            return value

    def close(self):
        with self.lock:
            self.status = "closed"
            self.stop_event.set()
        if self.runner:
            self.runner.thread.join(timeout=15)


class LanMatchRunner:
    def __init__(self, room):
        self.room = room
        self.thread = threading.Thread(target=self.run, name="arena-lan-match", daemon=True)
        self.tick = 0
        self.controllers = {}
        self.observations = {}
        self.metrics = {"steps": 0, "step_ms": [], "late_ai": 0}

    def run(self):
        room = self.room
        match, policies, pool, log = None, {}, ThreadPoolExecutor(max_workers=2), None
        actions, pending, held = [], {}, {"red": {}, "blue": {}}
        diagnostics = []
        try:
            lineups = mirrored_lineups(LINEUP, "2027")
            match = Match(seed=42, duration=room.duration, map_name="rmuc2027", lineups=lineups)
            initial = match.snapshot()
            log = FrameLog(initial)
            helpers = baseline_helpers()
            if room.ai:
                for index, team in enumerate(("red", "blue")):
                    source = ROOT / "packages/core/bots/default"
                    manifest, entries = load_submission(source)
                    policies[team] = Policy(entries["team"], 4200 + index, "trusted-local", timeout=.2,
                                            expected_digest=manifest["hashes"]["strategy.py"], stop_event=room.stop_event)
                # Stage the first decisions before the clock starts so AI infantry
                # can choose their performance at tick zero, just like humans.
                initial_jobs = {team: pool.submit(policy.decide, match.compact_observation(team))
                                for team, policy in policies.items()}
                for team, future in initial_jobs.items():
                    held[team], error = future.result()
                    if error:
                        diagnostics.append({"type": "policy_error", "team": team, "tick": 0, "message": error})
            with room.lock:
                room.static = {key: initial[key] for key in ("arena", "robot_models", "simulation_assumptions")}
                room.status = "running" if not room.stop_event.is_set() else "closed"
                room.changed("running")
            deadline = time.monotonic()
            while match.status != "finished" and not room.stop_event.is_set():
                started = time.monotonic()
                frame = match.snapshot()
                self.tick = frame["tick"]
                observations = {}
                for team, job in list(pending.items()):
                    future, submitted, tick = job
                    if future.done():
                        result, error = future.result()
                        if error:
                            diagnostics.append({"type": "policy_error", "team": team, "tick": self.tick, "message": error})
                        # Do not apply decisions after their intended 200 ms window.
                        held[team] = result if self.tick - tick <= 4 else {}
                        del pending[team]
                    elif self.tick - tick > 4:
                        held[team] = {}
                        self.metrics["late_ai"] += 1
                if self.tick > 0 and self.tick % 2 == 0:
                    for team, policy in policies.items():
                        if team not in pending and not policy.disabled:
                            obs = observations[(team, None)] = match.compact_observation(team)
                            pending[team] = (pool.submit(policy.decide, obs), started, self.tick)
                commands = copy.deepcopy(held)
                processed = {}
                with room.lock:
                    room.expire()
                    players = list(room.players.values())
                    for p in players:
                        if p.slot is None:
                            continue
                        robot = next(r for r in frame["robots"] if r["team"] == p.team and r["slot"] == p.slot)
                        obs = self.observations.get((p.team, p.slot))
                        if obs is None:
                            obs = match.compact_observation(p.team, [p.slot])
                        controller = self.controllers.setdefault(p.id, HumanController(helpers))
                        processed[p.id] = p.seq
                        # A disconnect stops the robot for 3 s, then yields to AI.
                        manual = (p.connected and not p.needs_resume) or (not p.connected and started - p.seen < 3)
                        for operation in p.operations:
                            try:
                                if operation.name == "resume":
                                    p.needs_resume = False
                                    p.intent, p.input_at = None, 0
                                else:
                                    controller.operation(operation.name, operation.value, robot)
                                receipt = {"seq": operation.seq, "status": "applied", "tick": self.tick}
                            except ValueError as exc:
                                receipt = {"seq": operation.seq, "status": "rejected", "message": str(exc), "tick": self.tick}
                            p.receipts.append(receipt)
                        p.operations.clear()
                        p.receipts = p.receipts[-8:]
                        intent = p.intent
                        fresh_transport = intent is not None and started - p.input_at <= .25 and p.connected
                        fresh = fresh_transport and intent.active and not p.needs_resume
                        if manual:
                            if not fresh:
                                controller.last_jump = False
                                intent = Input(match_id=room.match_id, seq=max(0, p.seq), yaw=robot["aim"], pitch=robot["pitch"], active=False)
                                if not fresh_transport or p.needs_resume:
                                    controller.mode = ""
                            if robot.get("red_card"):
                                command = {"move": [0, 0], "brake": True, "fire": False}
                            elif p.needs_resume:
                                command = {"move": [0, 0], "brake": True, "fire": False}
                            elif not fresh and not controller.pending and not (fresh_transport and (controller.mode or robot["role"] == "aerial")):
                                command = {"move": [0, 0], "brake": True, "fire": False}
                            else:
                                command = controller.action(obs, robot, intent)
                            commands[p.team][str(p.slot)] = command
                    # Copying player projections is deferred until after physics.
                    if self.tick == 0:
                        for p in players:
                            if p.slot in (3, 4):
                                commands[p.team].setdefault(str(p.slot), {})["profile"] = dict(p.profile)
                frame = match.step(commands)
                self.tick = frame["tick"]
                actions.append(commands)
                log.append(frame, match.raw_frame)
                with room.lock:
                    for p in room.players.values():
                        if p.slot is None:
                            continue
                        observation = match.compact_observation(p.team, [p.slot])
                        self.observations[(p.team, p.slot)] = observation
                        view = project(frame, observation, p.team, p.slot, room.match_id)
                        controller = self.controllers.get(p.id)
                        if controller:
                            controller.observe(frame, view["self"])
                        p.processed_seq = processed.get(p.id, p.processed_seq)
                        view["control"] = {"needs_resume": p.needs_resume,
                                           "status": "判罚遮挡中" if view["display"]["ui_blocked"] else controller.status if controller and p.connected and not p.needs_resume else "AI 接管中",
                                           "ack_seq": p.processed_seq, "manual": p.connected and not p.needs_resume}
                        room.latest[p.id] = view
                    room.revision += 1
                elapsed = (time.monotonic() - started) * 1000
                self.metrics["steps"] += 1
                self.metrics["step_ms"].append(elapsed)
                if room.realtime:
                    deadline += DT
                    # Do not try to catch up an unbounded backlog after a host stall.
                    if time.monotonic() - deadline > .25:
                        deadline = time.monotonic()
                    room.stop_event.wait(max(0, deadline - time.monotonic()))
            if match.status != "finished":
                match.finish(None, "stopped")
                log[-1] = match.snapshot()
            final = match.snapshot()
            with room.lock:
                if room.status != "closed":
                    room.status = "settling"
                    room.changed("settling")
            result = {"winner": final["winner"], "reason": final["reason"], "time": final["time"],
                      "teams": final["teams"], "digest": match.digest(), "policy_errors": len(diagnostics)}
            samples = sorted(self.metrics.pop("step_ms"))
            self.metrics["step_p95_ms"] = samples[min(len(samples) - 1, int(len(samples) * .95))] if samples else 0
            self.metrics["step_max_ms"] = samples[-1] if samples else 0
            metadata = {"match_mode": "lan", "ranked": False, "api_version": 3, "engine": "rust", "season": "2027",
                        "map": "rmuc2027", "seed": 42, "duration": room.duration, "lineups": lineups,
                        "rules": initial["rules"], "roster": list(SLOTS), "policy_interval": 2,
                        "teams": {team: {"name": "lan-" + team, "display_name": "局域网" + ("红方" if team == "red" else "蓝方"),
                                         "version": room.match_id, "mode": "squad"} for team in ("red", "blue")},
                        "engine_hash": initial["engine_hash"], "rules_hash": initial["rules_hash"], "dt": DT,
                        "manual_sentry": True, "radar_decode": "always_success", "seats": room.history,
                        "metrics": self.metrics}
            room.directory.mkdir(parents=True, exist_ok=True)
            replay_path = room.directory / (room.match_id + ".json")
            # FrameLog is a disk spool; stream frames rather than materializing a match twice.
            partial = replay_path.with_suffix(".json.part")
            with partial.open("w", encoding="utf-8") as stream:
                stream.write('{"format":"rmuc-replay-v4","metadata":')
                json.dump(metadata, stream, ensure_ascii=False)
                stream.write(',"static":')
                json.dump(log.static, stream, ensure_ascii=False)
                stream.write(',"frames":[')
                for index, raw in enumerate(log.encoded()):
                    if room.stop_event.is_set() and room.status == "closed":
                        raise InterruptedError("房间已关闭，回放保存已取消")
                    if index:
                        stream.write(",")
                    stream.write(raw.decode())
                stream.write('],"actions":')
                json.dump(actions, stream, ensure_ascii=False)
                stream.write(',"diagnostics":')
                json.dump(diagnostics, stream, ensure_ascii=False)
                stream.write(',"result":')
                json.dump(result, stream, ensure_ascii=False)
                stream.write("}")
            partial.replace(replay_path)
            with room.lock:
                room.result = {**result, "replay": replay_path.name, "metrics": self.metrics}
        except BaseException as exc:
            with room.lock:
                room.error = str(exc)
        finally:
            room.stop_event.set()
            if "partial" in locals():
                try:
                    partial.unlink(missing_ok=True)
                except OSError:
                    pass
            # Policy.decide has a bounded deadline, then terminate its subprocess tree.
            pool.shutdown(wait=True, cancel_futures=True)
            for policy in policies.values():
                try:
                    policy.close()
                except (OSError, RuntimeError):
                    pass
            if match:
                match.close()
            with room.lock:
                if room.status != "closed":
                    room.status = "finished"
                for p in room.players.values():
                    p.ready = False
                room.changed("finished")
