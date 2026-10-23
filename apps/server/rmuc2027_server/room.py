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
from rmuc2027_core.models import normalize_team_lineup
from rmuc2027_core.paths import ROOT, data_path
from rmuc2027_core.policy import FrameLog, Policy
from rmuc2027_core.submissions import load_submission
from . import LINEUP, PROTOCOL, SLOTS
from .control import HumanController, baseline_helpers
from .projection import project
from .hud import actions as operator_actions, rules as hud_rules
from .protocol import Input
from .catalog import Catalog

INTRO_SECONDS = 38.3
PREMATCH = ("starting", "loading", "intro", "countdown")


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
    stage_loaded: bool = False
    fire_press_seq: int = 0
    fire_presses: list = field(default_factory=list)


class Room:
    def __init__(self, owner_key=None, directory=None, *, duration=420, realtime=True, ai=True, ceremony=True):
        self.owner_key = owner_key or secrets.token_urlsafe(32)
        self.directory = Path(directory or data_path("replays/lan"))
        self.duration, self.realtime, self.ai, self.ceremony = duration, realtime, ai, ceremony
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
        self.catalog = Catalog()
        self.catalog_revision = 0
        self.seat_configs = {team: {str(slot): {"model": LINEUP[str(slot)], "strategy": "baseline"} for slot in SLOTS} for team in ("red", "blue")}
        self.team_configs = {team: {"mode": "independent", "strategy": "baseline"} for team in ("red", "blue")}
        self.stage_at, self.stage_duration = 0., 0.
        self.stage_error = ""
        self.presentation = None
        self.policy_status = {}

    def editable(self, token):
        self.player(token)
        if self.status not in ("lobby", "finished"): raise ValueError("仅准备阶段可以修改配置")

    def invalidate_ready(self):
        for p in self.players.values(): p.ready = False

    def lineup(self, team):
        return {slot: c["model"] for slot, c in self.seat_configs[team].items()}

    def configure_seat(self, token, config):
        with self.lock:
            self.editable(token)
            occupant = next((p for p in self.players.values() if (p.team, p.slot) == (config.team, config.slot)), None)
            if (occupant and occupant.token != token) or (not occupant and token != self.owner):
                raise ValueError("只能配置自己的席位，房主可配置空席位")
            proposed = {**self.lineup(config.team), str(config.slot): config.model}
            normalize_team_lineup(proposed, '2027')
            if not self.catalog.compatible(config.strategy, config.slot, config.model): raise ValueError("单机器人策略与型号不兼容")
            self.seat_configs[config.team][str(config.slot)] = {"model": config.model, "strategy": config.strategy}
            self.invalidate_ready(); self.changed('seat_config')
            return self.state()

    def configure_team(self, token, config):
        with self.lock:
            self.editable(token)
            if token != self.owner: raise ValueError("只有房主可以配置队伍策略")
            if config.mode == 'team': self.catalog.validate_team(config.strategy, self.lineup(config.team))
            self.team_configs[config.team] = {"mode": config.mode, "strategy": config.strategy}
            self.invalidate_ready(); self.changed('team_config')
            return self.state()

    def import_strategy(self, token, files):
        with self.lock:
            self.editable(token)
            if token != self.owner: raise ValueError("只有房主可以导入策略")
            result = self.catalog.import_bundle(files)
            self.catalog_revision += 1
            return {**result, 'revision': self.catalog_revision}

    def stage_ready(self, token, request):
        with self.lock:
            p = self.player(token)
            if self.status != 'loading' or request.match_id != self.match_id: raise ValueError('资源准备阶段已变化')
            if request.error: self.stage_error = p.nickname + '：' + request.error
            else: p.stage_loaded = True
            return self.state()

    def skip_intro(self, token):
        with self.lock:
            if token != self.owner: raise ValueError('只有房主可以跳过入场')
            if self.status != 'intro': raise ValueError('当前不是入场阶段')
            self.stage_at -= self.stage_duration
            return self.state()

    def stage(self, status, duration):
        with self.lock:
            if self.status == 'closed': return
            self.status, self.stage_at, self.stage_duration = status, time.monotonic(), duration
            self.changed(status)

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
                    "lineup": self.lineup('red'), "lineups": {t: self.lineup(t) for t in ('red', 'blue')}, "duration": self.duration, "error": self.error,
                    "seat_configs": copy.deepcopy(self.seat_configs), "team_configs": copy.deepcopy(self.team_configs),
                    "catalog_revision": self.catalog_revision,
                    "stage_elapsed": max(0, time.monotonic()-self.stage_at) if self.stage_at else 0,
                    "stage_duration": self.stage_duration, "policy_status": dict(self.policy_status),
                    "result": copy.deepcopy(self.result), "manual_sentry": True,
                    "players": [{"id": p.id, "nickname": p.nickname, "team": p.team, "slot": p.slot,
                                 "host": p.token == self.owner, "ready": p.ready, "connected": p.connected,
                                 "profile": p.profile,
                                 "stage_loaded": p.stage_loaded,
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
            p.fire_presses.clear()
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
            p.fire_presses.clear()
            p.needs_resume = reconnect and self.status == "running"
            self.changed("reconnect" if reconnect else "connect", p)
            return p.generation

    def detach(self, token, generation):
        with self.lock:
            p = self.players.get(token)
            if p and p.generation == generation:
                p.connected, p.disconnected = False, time.monotonic()
                p.intent, p.input_at = None, 0
                p.fire_presses.clear()
                p.operations.clear()
                if self.status in PREMATCH: self.stage_error = p.nickname + ' 在开赛前断开连接'
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
            if self.status in PREMATCH: self.stage_error = p.nickname + ' 在开赛前离开房间'
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
            for team, config in self.team_configs.items():
                if config['mode'] == 'team': self.catalog.validate_team(config['strategy'], self.lineup(team))
                else:
                    for slot, c in self.seat_configs[team].items():
                        if not self.catalog.compatible(c['strategy'], int(slot), c['model']): raise ValueError('席位策略与型号不兼容')
            self.match_id = uuid.uuid4().hex
            self.latest, self.static, self.result, self.error = {}, None, None, None
            self.status = "starting"
            self.stage_at, self.stage_duration, self.stage_error, self.presentation = time.monotonic(), 60., '', None
            self.policy_status = {}
            for p in humans:
                p.seq, p.processed_seq, p.intent, p.input_at = -1, -1, None, 0
                p.fire_press_seq = 0
                p.fire_presses.clear()
                p.operations.clear()
                p.receipts.clear()
                p.needs_resume = False
                p.stage_loaded = False
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
                if not message.active or p.needs_resume:
                    p.fire_presses.clear()
                elif message.fire_press_seq > p.fire_press_seq:
                    # Preserve presses even when a following mouse-up replaces
                    # the latest intent before the physics thread samples it.
                    count = min(message.fire_press_seq - p.fire_press_seq, 8 - len(p.fire_presses))
                    p.fire_presses.extend(range(p.fire_press_seq + 1, p.fire_press_seq + count + 1))
                p.fire_press_seq = max(p.fire_press_seq, message.fire_press_seq)
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
            if self.status in PREMATCH and self.presentation:
                value['presentation'] = self.presentation
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
        match, policies, pool, log = None, {}, ThreadPoolExecutor(max_workers=10), None
        actions, pending, held, bindings = [], {}, {}, {}
        diagnostics = []
        running = False
        try:
            lineups = {team: [{'slot': int(slot), 'model': model} for slot, model in room.lineup(team).items()] for team in ('red', 'blue')}
            match = Match(seed=42, duration=room.duration, map_name="rmuc2027", lineups=lineups)
            initial = match.snapshot()
            log = FrameLog(initial)
            helpers = baseline_helpers()
            with room.lock:
                room.static = {key: initial[key] for key in ("arena", "robot_models", "simulation_assumptions")}
                room.presentation = copy.deepcopy({k: v for k, v in initial.items() if k not in room.static})
                # Prepare-stage cards show the exact referee performance that
                # will be applied to the human's first tick without advancing time.
                profiles = hud_rules()['profiles']
                for p in room.players.values():
                    if p.slot not in (3, 4): continue
                    r = next(r for r in room.presentation['robots'] if (r['team'], r['slot']) == (p.team, p.slot))
                    chassis = profiles['infantry_chassis'][p.profile['chassis']][0]
                    weapon = profiles['infantry_weapon'][p.profile['weapon']][0]
                    r.update(hp=chassis['hp'], max_hp=chassis['hp'], power_limit=chassis['power_w'],
                             max_heat=weapon['max_heat'], cooling_rate=weapon['cooling'],
                             chassis_profile=p.profile['chassis'], weapon_profile=p.profile['weapon'])
                room.stage('loading', 60)
            if room.ai:
                specs = []
                for team in ('red', 'blue'):
                    config = room.team_configs[team]
                    if config['mode'] == 'team':
                        key = config['strategy']
                        manifest, entries = load_submission(room.catalog.sources[key])
                        if manifest['mode'] == 'team': specs.append((team, tuple(SLOTS), entries['team'], manifest))
                        else:
                            for slot in SLOTS: specs.append((team, (slot,), entries[str(slot)], manifest))
                    else:
                        for slot in SLOTS:
                            key = room.seat_configs[team][str(slot)]['strategy']
                            manifest, entries = load_submission(room.catalog.sources[key])
                            path = next(iter(entries.values()))
                            specs.append((team, (slot,), path, manifest))
                for index, (team, slots, path, manifest) in enumerate(specs):
                    name = team + ':' + ','.join(map(str, slots))
                    bindings[name] = (team, slots)
                    policies[name] = Policy(path, 4200 + index, 'trusted-local', timeout=.2,
                        expected_digest=manifest['hashes'][path.name], deadline=room.stage_at+60, stop_event=room.stop_event)
                    room.policy_status[name] = '就绪'
                initial_jobs = {name: pool.submit(policy.decide, match.compact_observation(bindings[name][0], list(bindings[name][1]))) for name, policy in policies.items()}
                for name, future in initial_jobs.items():
                    held[name], error = future.result()
                    if error: raise ValueError('策略初始化失败：' + str(error))
            for policy in policies.values(): policy.deadline = None
            if room.ceremony:
                while not room.stop_event.is_set():
                    with room.lock:
                        if room.stage_error: raise ValueError(room.stage_error)
                        if all(p.stage_loaded and p.connected for p in room.players.values()): break
                        if time.monotonic()-room.stage_at >= 60: raise ValueError('资源准备超时，请检查客户端模型与图形环境')
                    room.stop_event.wait(.05)
                for phase, duration in (('intro', INTRO_SECONDS), ('countdown', 5)):
                    room.stage(phase, duration)
                    while not room.stop_event.is_set() and time.monotonic()-room.stage_at < duration:
                        if room.stage_error: raise ValueError(room.stage_error)
                        room.stop_event.wait(.05)
            if room.stop_event.is_set(): return
            room.stage('running', 0)
            running = True
            deadline = time.monotonic()
            while match.status != "finished" and not room.stop_event.is_set():
                started = time.monotonic()
                frame = match.snapshot()
                self.tick = frame["tick"]
                for name, job in list(pending.items()):
                    future, submitted, tick = job
                    if future.done():
                        result, error = future.result()
                        if error:
                            diagnostics.append({"type": "policy_error", "team": bindings[name][0], "tick": self.tick, "message": error})
                            room.policy_status[name] = str(error)
                        held[name] = result if self.tick-tick <= 4 and not policies[name].disabled else {}
                        del pending[name]
                    elif self.tick-tick > 4:
                        held[name] = {}
                        self.metrics['late_ai'] += 1
                if self.tick > 0 and self.tick % 2 == 0:
                    for name, policy in policies.items():
                        if name not in pending and not policy.disabled:
                            team, slots = bindings[name]
                            pending[name] = (pool.submit(policy.decide, match.compact_observation(team, list(slots))), started, self.tick)
                commands = {'red': {}, 'blue': {}}
                for name, result in held.items():
                    team, slots = bindings[name]
                    for slot in slots:
                        if str(slot) in result: commands[team][str(slot)] = copy.deepcopy(result[str(slot)])
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
                        if robot.get('transform_allowed') and controller.folded is None:
                            controller.folded = robot.get('transform_progress', 0) >= .5
                        processed[p.id] = p.seq
                        # A disconnect stops the robot for 3 s, then yields to AI.
                        manual = (p.connected and not p.needs_resume) or (not p.connected and started - p.seen < 3)
                        for operation in p.operations:
                            try:
                                if operation.name == "resume":
                                    p.needs_resume = False
                                    p.intent, p.input_at = None, 0
                                    p.fire_presses.clear()
                                    controller.stop_combat()
                                    controller.folded = robot.get('transform_progress', 0) >= .5 if robot.get('transform_allowed') else None
                                else:
                                    if p.needs_resume:
                                        raise ValueError("请先恢复人工控制")
                                    offered = next((a for a in operator_actions(frame, robot) if a["name"] == operation.name and a["value"] == operation.value), None)
                                    if offered and not offered["enabled"]:
                                        raise ValueError(offered["reason"])
                                    if robot.get("red_card") or robot.get("fpv_blocked_until", 0) > frame["time"]:
                                        raise ValueError("判罚期间无法执行操作")
                                    controller.operation(operation.name, operation.value, robot)
                                receipt = {"seq": operation.seq, "status": "applied", "name": operation.name, "value": operation.value, "tick": self.tick}
                            except ValueError as exc:
                                receipt = {"seq": operation.seq, "status": "rejected", "name": operation.name, "value": operation.value, "message": str(exc), "tick": self.tick}
                            p.receipts.append(receipt)
                        p.operations.clear()
                        p.receipts = p.receipts[-8:]
                        intent = p.intent
                        fresh_transport = intent is not None and started - p.input_at <= .25 and p.connected
                        fresh = fresh_transport and intent.active and not p.needs_resume
                        blind = (not robot['alive'] or robot.get('red_card')
                                 or robot.get('fpv_blocked_until', 0) > frame['time']
                                 or robot.get('overheated', False))
                        if not fresh or blind:
                            p.fire_presses.clear()
                            controller.stop_combat()
                        if manual:
                            if not fresh:
                                controller.last_jump = False
                                intent = Input(match_id=room.match_id, seq=max(0, p.seq), yaw=robot["aim"], pitch=robot["pitch"], active=False)
                                if not fresh_transport or p.needs_resume:
                                    controller.mode = ""
                            if robot.get('red_card') or robot.get('fpv_blocked_until', 0) > frame['time']:
                                controller.pending.clear()
                                controller.mode = ''
                                command = {"move": [0, 0], "brake": True, "fire": False}
                            elif blind and robot['role'] != 'aerial':
                                # Camera loss stops combat, but the dead operator
                                # must still be able to purchase revival/ammo.
                                controller.mode = ''
                                stopped = Input(match_id=room.match_id, seq=max(0, p.seq),
                                                yaw=robot['aim'], pitch=robot['pitch'], active=False)
                                command = controller.action(obs, robot, stopped) if controller.pending else {"move": [0, 0], "brake": True, "fire": False}
                            elif p.needs_resume:
                                command = {"move": [0, 0], "brake": True, "fire": False}
                            elif not fresh and not controller.pending and not (fresh_transport and (controller.mode or robot["role"] == "aerial" or intent.transform is not None)):
                                command = {"move": [0, 0], "brake": True, "fire": False}
                            else:
                                press = p.fire_presses.pop(0) if fresh and p.fire_presses else None
                                command = controller.action(obs, robot, intent, press)
                            if robot.get('transform_allowed') and controller.folded is not None and not robot.get('red_card') and not p.needs_resume:
                                command['transform'] = controller.folded
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
                        if controller and not view['display']['ui_blocked']:
                            view['control']['aim_assist'] = controller.aim.snapshot()
                            view['control']['last_press_seq'] = controller.last_press_seq
                            view['control']['last_press_result'] = controller.last_press_result
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
                      "teams": final["teams"], "digest": match.digest(), "policy_errors": len(diagnostics),
                      "robots": [{k:r.get(k) for k in ("id","team","slot","model_slot","design_name","role","kills","damage","hp","max_hp")} for r in final["robots"]]}
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
                        "players": [{"id": p.id, "nickname": p.nickname, "team": p.team, "slot": p.slot} for p in room.players.values()],
                        "strategies": {key: dict(room.catalog.entries[key]) for key in {c["strategy"] for seats in room.seat_configs.values() for c in seats.values()} | {c["strategy"] for c in room.team_configs.values()}},
                        "metrics": self.metrics, "seat_configs": room.seat_configs, "team_configs": room.team_configs,
                        "presentation": {"intro_seconds": INTRO_SECONDS, "countdown_seconds": 5, "enabled": room.ceremony, "robots": room.presentation["robots"] if room.presentation else None}}
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
                    room.status = "finished" if running else "lobby"
                room.presentation = None
                for p in room.players.values():
                    p.ready = False
                room.changed("finished")
