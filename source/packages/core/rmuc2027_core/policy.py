"""Policy isolation, deadlines, action ownership, and replay recording."""
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from collections.abc import Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import shutil
from .platform_support import kill_process_tree
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from .paths import ROOT
from .engine import Match, POLICY_INTERVAL, DT, SLOT_IDS, RULES_PROFILE, STATIC_FRAME_KEYS
from .communication import TeamCommunication, validate_messages
from .worker import MEMORY_LIMIT

IMAGE = "rmuc2027-policy:0.1"
CPU_LIMIT = 1.0
# Seconds for module import and global initialization before the first decision.
STARTUP_LIMIT = {"docker": 60, "trusted-local": 15}
# A stalled process is stopped after three consecutive missed decision windows.
POLICY_TIMEOUT_LIMIT = 3

ACTION_FIELDS = frozenset({"move", "aim", "pitch", "fire", "power", "profile", "buy_ammo",
    "remote_heal", "revive", "deploy", "support", "altitude", "pick_unit", "assembly",
    "assembly_insert", "assembly_translate", "assembly_rotate", "assembly_confirm", "rune_start",
    "heading", "crouch", "rescue", "yaw_rate", "brake", "leg_height", "jump", "lean", "arm",
    "arm2", "pickup_arm", "transform", "install_launcher", "assembly_core", "assembly_lift",
    "assembly_flip", "assembly_rotate2", "posture", "posture_enhanced", "drones", "sentry_cmd"})


def validate_actions(actions, controlled, has_messages=False, *, dynamic_messages=False,
                     robots=None, team=None, observation_profile=None, roster=None):
    """One protocol boundary for sandbox policies AND in-process controllers.

    Engine rules remain authoritative for range/contact/timing. This boundary
    checks ownership, finite types, physical role capabilities and message scope.
    It grants no extra actuation privileges to omniscient auxiliary actors.
    """
    if not isinstance(actions, dict): raise ValueError('Invalid action object')
    if not set(actions) <= {str(k) for k in controlled}: raise ValueError('Attempt to control an unowned robot')
    fields = ACTION_FIELDS | ({'messages'} if has_messages else set())
    boolean = {'fire', 'remote_heal', 'revive', 'deploy', 'support', 'pick_unit', 'assembly_insert',
               'assembly_translate', 'assembly_confirm', 'rune_start', 'crouch', 'brake', 'transform',
               'install_launcher', 'assembly_lift', 'assembly_flip', 'posture_enhanced'}
    numeric = {'pitch', 'power', 'altitude', 'assembly_rotate', 'assembly_rotate2', 'heading',
               'yaw_rate', 'leg_height', 'lean'}
    def number(v): return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    def vector(v, sizes): return isinstance(v, (tuple, list)) and len(v) in sizes and all(number(x) for x in v)
    for slot, action in actions.items():
        if not isinstance(action, dict) or not set(action) <= fields: raise ValueError('Action contains unsupported fields')
        if len(json.dumps(action, allow_nan=False).encode()) > 1024: raise ValueError('Each robot action must be <=1 KiB')
        for name in boolean & action.keys():
            if not isinstance(action[name], bool): raise ValueError(f'{name} must be boolean')
        for name in numeric & action.keys():
            if not number(action[name]): raise ValueError(f'{name} must be finite')
        if 'aim' in action and action['aim'] is not None and not number(action['aim']): raise ValueError('aim must be finite or null')
        if 'jump' in action and not isinstance(action['jump'], bool) and not number(action['jump']): raise ValueError('jump must be finite')
        if 'move' in action and not vector(action['move'], (2, 3)): raise ValueError('move must contain finite numbers')
        for name in ('assembly', 'assembly_core', 'pickup_arm'):
            if name in action and (not isinstance(action[name], int) or isinstance(action[name], bool) or action[name] < 0): raise ValueError(f'{name} must be a nonnegative integer')
        if 'buy_ammo' in action:
            order = action['buy_ammo']
            if not isinstance(order, dict) or set(order)-{'amount', 'remote'}: raise ValueError('Invalid ammo order')
            if 'remote' in order and not isinstance(order['remote'], bool): raise ValueError('Ammo remote must be boolean')
            if 'amount' in order and (not isinstance(order['amount'], int) or isinstance(order['amount'], bool) or order['amount'] < 0): raise ValueError('Ammo amount must be nonnegative integer')
        for name in ('arm', 'arm2'):
            if name not in action: continue
            arm = action[name]
            if not isinstance(arm, dict) or set(arm)-{'stow', 'joints', 'tip', 'dir', 'frame', 'roll', 'grip'}: raise ValueError('Invalid arm command')
            for key, length in (('joints', 7), ('tip', 3), ('dir', 3)):
                if key in arm and not vector(arm[key], (length,)): raise ValueError('Invalid arm vector')
            if 'frame' in arm and arm['frame'] not in ('world', 'robot'): raise ValueError('Invalid arm frame')
            if 'stow' in arm and not isinstance(arm['stow'], bool): raise ValueError('Invalid arm stow flag')
            for key in ('roll', 'grip'):
                if key in arm and not number(arm[key]): raise ValueError('Invalid arm scalar')
        if 'drones' in action:
            if not isinstance(action['drones'], dict): raise ValueError('Invalid drone commands')
            for index, command in action['drones'].items():
                if not str(index).isdigit() or not isinstance(command, dict) or set(command)-{'move', 'goto', 'launch', 'dock', 'land'}: raise ValueError('Invalid drone command')
                for key in ('move', 'goto'):
                    if key in command and not vector(command[key], (3,)): raise ValueError('Invalid drone vector')
                for key in ('launch', 'dock', 'land'):
                    if key in command and not isinstance(command[key], bool): raise ValueError('Invalid drone flag')
        if 'messages' in action:
            validate_messages(action['messages'], slot, dynamic_messages)
            if observation_profile in ('privileged_aux_2027_v1', 'outpost_drone_script_v1') and action['messages']:
                raise ValueError('Privileged auxiliary/drone controllers cannot send messages')
        if observation_profile == 'privileged_aux_2027_v1' and set(action) & {'deploy', 'support', 'altitude', 'posture', 'posture_enhanced', 'drones', 'sentry_cmd'}:
            raise ValueError('Auxiliary profile attempted unsupported role operation')
        if robots is None: continue
        robot = next((r for r in robots if str(r['slot']) == slot and (team is None or r['team'] == team)), None)
        if robot is None: raise ValueError('Unknown controlled robot')
        # Training summaries deliberately omit immutable hardware fields. Those
        # fields come from Battle's initial engine roster, never policy output or
        # a privileged observation added to the restricted actor's view.
        if roster is not None:
            static = roster.get((robot['team'], slot))
            if static is None: raise ValueError('Missing trusted action roster')
            robot = {**robot, **static}
        if 'role' not in robot:
            # RMUC 2027 has fixed roles per registered seat. Model-dependent
            # capabilities below still require authoritative hardware context.
            role = {1: 'heavy', 3: 'infantry', 4: 'infantry', 6: 'aerial', 7: 'sentry'}.get(int(slot))
            if role is None: raise ValueError('Unknown 2027 role slot')
            robot = {**robot, 'role': role}
        heavy = {'pick_unit', 'assembly', 'assembly_insert', 'assembly_translate', 'assembly_rotate', 'assembly_rotate2', 'assembly_confirm', 'rescue', 'arm', 'arm2', 'pickup_arm', 'install_launcher', 'assembly_core', 'assembly_lift', 'assembly_flip'}
        if robot['role'] != 'heavy' and any(action.get(k) for k in heavy): raise ValueError('Heavy-only action')
        if robot['role'] != 'sentry' and set(action) & {'posture', 'posture_enhanced', 'sentry_cmd'}: raise ValueError('Sentry-only action')
        if robot['role'] != 'aerial' and 'drones' in action: raise ValueError('Aerial-only action')
        if robot['role'] != 'infantry' and action.get('profile'): raise ValueError('Infantry-only profile')
        if 'transform' in action and not robot.get('transform_allowed', False): raise ValueError('Model cannot transform')
        if robot.get('chassis_type') != 'wheel_legged' and ('leg_height' in action or action.get('jump') or action.get('lean')): raise ValueError('Wheel-legged action on another chassis')
        if 'arm2' in action and robot.get('arm_count', 0) < 2: raise ValueError('Second arm unavailable')
    return actions


class FrameLog(Sequence):
    """Spool immutable frames to disk instead of retaining a full match's heap.

    Keeping thousands of nested frame dictionaries triggers long GC pauses that
    count against agents' wall-clock deadlines. Only offsets and events stay in
    memory; the public sequence interface remains available to replay callers.
    """
    def __init__(self, initial):
        self.file = tempfile.TemporaryFile(mode="w+b")
        self.offsets, self.events = [], []
        self.lock = threading.RLock()
        self.static = {key:initial[key] for key in STATIC_FRAME_KEYS if key in initial}
        self.append(initial)

    def __len__(self):
        return len(self.offsets)

    def __del__(self):
        if hasattr(self, "file"):
            self.file.close()

    def _write(self, frame, raw=None):
        if raw is None:
            raw = json.dumps({key:value for key,value in frame.items() if key not in self.static}, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
        self.file.seek(0, 2)
        offset = self.file.tell()
        self.file.write(raw)
        return offset, len(raw)

    def append(self, frame, raw=None):
        """`raw` is the frame already encoded without its static fields, if available."""
        with self.lock:
            self.offsets.append(self._write(frame, raw))
            self.events.append(list(frame["events"]))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self))) ]
        with self.lock:
            offset, size = self.offsets[index]
            self.file.seek(offset)
            return {**self.static,**json.loads(self.file.read(size))}

    def __setitem__(self, index, frame):
        with self.lock:
            # Preserve indexing semantics while appending a replacement record.
            self.offsets[index] = self._write(frame)
            self.events[index] = list(frame["events"])

    def events_since(self, tick):
        with self.lock:
            start = max(1, tick + 1, len(self) - 1000)
            return [event for events in self.events[start:] for event in events]

    def encoded(self):
        for index in range(len(self)):
            with self.lock:
                offset, size = self.offsets[index]
                self.file.seek(offset)
                raw = self.file.read(size)
            yield raw


class EncodedObservation:
    """An engine observation line forwarded to a policy without decoding it."""
    def __init__(self, payload, controlled_ids, has_messages):
        self.payload, self.controlled_ids, self.has_messages = payload, controlled_ids, has_messages


class Policy:
    def __init__(self, path, seed, sandbox="docker", timeout=0.2, expected_digest=None, deadline=None, stop_event=None, dynamic_messages=False):
        self.timeout, self.sandbox, self.disabled = timeout, sandbox, False
        self.deadline, self.stop_event = deadline, stop_event
        self.dynamic_messages = dynamic_messages
        self.logs, self.lines, self.calls, self.errors = deque(maxlen=32), queue.Queue(maxsize=2), 0, 0
        self.pending_response, self.consecutive_timeouts = None, 0
        self.name, self.temp = "rmuc-" + uuid.uuid4().hex[:16], tempfile.TemporaryDirectory(prefix="rmuc-policy-")
        stage = Path(self.temp.name)
        stage.chmod(0o755)
        source = path.read_bytes()
        self.digest = hashlib.sha256(source).hexdigest()
        if expected_digest is not None and self.digest != expected_digest:
            self.temp.cleanup()
            raise ValueError("Submitted strategy changed before staging")
        (stage / "strategy.py").write_bytes(source)
        (stage / "strategy.py").chmod(0o444)
        worker = Path(__file__).with_name("worker.py")
        if sandbox == "docker":
            command = ["docker", "run", "--rm", "--name", self.name, "--network=none", "--read-only",
                       "--cap-drop=ALL", "--security-opt=no-new-privileges", f"--memory={MEMORY_LIMIT}", f"--memory-swap={MEMORY_LIMIT}",
                       f"--cpus={CPU_LIMIT}", "--pids-limit=1", "--user=65534:65534", "--ipc=none",
                       "--ulimit", "nofile=64:64", "--mount", f"type=bind,source={stage},target=/submission,readonly",
                       "-i", IMAGE, "/submission/strategy.py", str(seed)]
        elif sandbox == "trusted-local":
            # Copy only the public runtime. Local mode is for trusted developer code.
            shutil.copyfile(worker, stage / "worker.py")
            shutil.copyfile(worker.with_name("sdk.py"), stage / "sdk.py")
            shutil.copyfile(ROOT / "configs/rules.json", stage / "rules.json")
            command = [sys.executable, "-I", str(stage / "worker.py"), str(stage / "strategy.py"), str(seed)]
        else:
            self.temp.cleanup()
            raise ValueError("Unsupported sandbox")
        env = {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", ""), "PYTHONHASHSEED": "0",
               "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUTF8": "1", "LANG": "C.UTF-8"}
        # Docker client needs its configured socket; never propagate arbitrary secrets.
        for key in ("DOCKER_HOST", "DOCKER_CONTEXT", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "USERPROFILE", "LOCALAPPDATA", "APPDATA"):
            if key in os.environ:
                env[key] = os.environ[key]
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                            cwd=stage, env=env, start_new_session=True)
        except BaseException:
            self.temp.cleanup()
            raise
        threading.Thread(target=self._read_output, daemon=True).start()
        threading.Thread(target=self._read_errors, daemon=True).start()
        try:
            ready_until = min(time.monotonic() + STARTUP_LIMIT[sandbox], deadline or float("inf"))
            while True:
                remaining = ready_until - time.monotonic()
                if remaining <= 0 or (stop_event and stop_event.is_set()):
                    raise TimeoutError("Policy initialization deadline reached")
                try:
                    ready = self.lines.get(timeout=min(.1, remaining))
                    break
                except queue.Empty:
                    continue
            if json.loads(ready) != {"ready": True}:
                raise ValueError("Invalid ready response")
        except BaseException as exc:
            details = " ".join(self.logs)[-1500:]
            self.close()
            raise RuntimeError(f"Policy did not initialize: {details or str(exc)}") from exc

    def _read_output(self):
        try:
            while True:
                line = self.process.stdout.readline(16386)
                if not line:
                    break
                if len(line) > 16385 or not line.endswith(b"\n"):
                    line = b'{"error":"oversized protocol output"}\n'
                try:
                    self.lines.put_nowait(line)
                except queue.Full:
                    break
        finally:
            try:
                self.lines.put_nowait(b'{"error":"policy process exited"}\n')
            except queue.Full:
                pass

    def _read_errors(self):
        while True:
            chunk = self.process.stderr.read(512)
            if not chunk:
                return
            self.logs.append(chunk.decode("utf-8", errors="replace"))

    def decide(self, observation):
        if self.disabled:
            return {}, None
        self.calls += 1
        try:
            if isinstance(observation, EncodedObservation):
                payload, controlled, has_messages = observation.payload, set(observation.controlled_ids), observation.has_messages
            else:
                payload = json.dumps(observation, allow_nan=False, separators=(",", ":")).encode() + b"\n"
                controlled, has_messages = set(observation["controlled_ids"]), "messages" in observation
            started = time.monotonic()
            if (self.deadline is not None and started >= self.deadline) or (self.stop_event and self.stop_event.is_set()):
                return {}, None
            until = min(started + self.timeout, self.deadline if self.deadline is not None else float("inf"))
            if self.pending_response is not None:
                # Only one request may be in flight. Discard its late response before
                # sending fresh observations, preserving private state without ever
                # applying old actions to a new tick or building a stdin backlog.
                self._receive(until)
            if time.monotonic() >= until:
                raise queue.Empty
            self.pending_response = controlled, has_messages
            self.process.stdin.write(payload)
            self.process.stdin.flush()
            actions = self._receive(until)
            self.consecutive_timeouts = 0
            return actions, None
        except queue.Empty:
            self.errors += 1
            self.consecutive_timeouts += 1
            message = "decision exceeded deadline"
            self.logs.append(message)
            if self.consecutive_timeouts >= POLICY_TIMEOUT_LIMIT:
                self.close()
            return {}, message
        except Exception as exc:
            self.errors += 1
            message = str(exc)
            self.logs.append(message)
            self.close()
            return {}, message

    def _receive(self, until):
        remaining = until - time.monotonic()
        if remaining <= 0:
            raise queue.Empty
        raw = self.lines.get(timeout=remaining)
        controlled, has_messages = self.pending_response
        self.pending_response = None
        response = json.loads(raw, parse_constant=lambda s: (_ for _ in ()).throw(ValueError("non-finite JSON")))
        if response.get("error"):
            raise ValueError(response["error"])
        self.navigation_trace = response.get("navigation_trace", {})
        actions = response.get("actions")
        return validate_actions(actions, controlled, has_messages,
                                dynamic_messages=self.dynamic_messages)

    def close(self):
        self.disabled = True
        if hasattr(self, "process"):
            if self.sandbox == "docker":
                # Killing docker's client alone does not always kill its container.
                try:
                    subprocess.run(["docker", "rm", "-f", self.name], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL, timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    pass
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
            if self.process.poll() is None:
                try:
                    if self.sandbox == "docker":
                        self.process.kill()
                    else:
                        kill_process_tree(self.process)
                except (ProcessLookupError, PermissionError):
                    # Container deletion above is authoritative. Never turn a
                    # completed match into a failure by killing an exited group.
                    try:
                        self.process.terminate()
                    except (ProcessLookupError, PermissionError):
                        pass
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
            for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass
        self.temp.cleanup()


