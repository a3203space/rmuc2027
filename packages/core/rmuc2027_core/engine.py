"""Thin Python transport to Rust; game rules live in simulation/src/lib.rs."""
import copy
import hashlib
import json
import os
from pathlib import Path
import queue
import select
import subprocess
import threading
import time

_DIRECT_PIPE_READER=os.name=='posix'
_CACHE_POLICY_OBSERVATIONS=True


class _PosixLineReader:
    """Bounded, deadline-aware pipe reads without a forwarding thread."""
    def __init__(self, stream):
        self.fd=stream.fileno();self.pending=bytearray()
        os.set_blocking(self.fd,False)

    def read_line(self,timeout=10.,maximum=4*1024*1024):
        deadline=time.monotonic()+timeout
        while True:
            newline=self.pending.find(b'\n')
            if newline>=0:
                if newline+1>maximum:raise RuntimeError('Rust engine response exceeds size limit')
                line=bytes(self.pending[:newline+1]);del self.pending[:newline+1]
                return line
            if len(self.pending)>=maximum:raise RuntimeError('Rust engine response exceeds size limit')
            remaining=deadline-time.monotonic()
            if remaining<=0. or not select.select([self.fd],[],[],remaining)[0]:
                raise RuntimeError('Rust engine response timed out')
            try:chunk=os.read(self.fd,min(65536,maximum-len(self.pending)))
            except BlockingIOError:continue
            if not chunk:
                line=bytes(self.pending);self.pending.clear()
                return line
            self.pending.extend(chunk)

try:
    import orjson as _response_json
except ImportError:
    _response_json = json
RESPONSE_JSON_DECODER = _response_json.__name__


def decode_response_json(payload):
    """Decode trusted engine JSON with the available, tested response decoder."""
    return _response_json.loads(payload)

from .paths import ROOT
DT = 0.05
POLICY_INTERVAL = 2
_config = json.loads((ROOT / "configs" / "rules.json").read_text(encoding="utf-8"))
SPECS = _config["specs"]
ROLES = ("heavy", "infantry", "infantry", "aerial", "sentry")
SLOT_IDS = (1, 3, 4, 6, 7)
RULES_PROFILE = _config["version"]
MATCH_DURATION = 420
# Frame fields that never change during a match; compact engine steps leave them out.
STATIC_FRAME_KEYS = ("arena", "rulebook_rules", "simulation_assumptions", "robot_models")


def arena_map(name="rmuc2027"):
    try:
        return copy.deepcopy(_config["maps"][name])
    except KeyError:
        raise ValueError("Unknown map") from None


def binary_path():
    path = Path(os.environ.get("RMUC_SIM_BINARY", ROOT / "target/release" / ("rmuc-sim.exe" if os.name == "nt" else "rmuc-sim")))
    if not path.is_file():
        raise RuntimeError("Build Rust first: cargo build --release --manifest-path packages/simulation/Cargo.toml")
    return path


class Match:
    def __init__(self, seed=42, duration=420, map_name="rmuc2027", replay_version=None, lineups=None, battle_settings=None, physics_adapter=None, observation_profile=None, training_capabilities=None, radar_classifier=None):
        self.training_summary = False
        self.training_capabilities = copy.deepcopy(training_capabilities or {})
        self.training_capabilities_initial=copy.deepcopy(self.training_capabilities)
        self.capability_drift=None
        self._aim_executors = {}
        self._aim_references = {}
        if self.training_capabilities:
            from rmuc2027_research.rl.capabilities import AimNoiseExecutor,SlowCapabilityDrift
            protected=[seat for seat,p in self.training_capabilities.items() if p.get('protected_learner')]
            if protected:
                self.capability_drift=SlowCapabilityDrift(self.training_capabilities,int(seed)+104729,
                    learner_side=protected[0].split(':')[0])
            for seat, profile in self.training_capabilities.items():
                team, slot = seat.split(':')
                self._aim_executors[seat] = AimNoiseExecutor(profile['aim'], profile.get('aim_seed',int(seed)*1000+(100 if team=='blue' else 0)+int(slot)))
        if observation_profile not in (None,'sentry_referee_2027_v1'):
            raise ValueError('Unknown observation profile')
        self.sentry_gateway = None
        self.external_physics = physics_adapter
        environment = dict(os.environ)
        environment.pop('RMUC_EXTERNAL_PHYSICS', None)
        environment.pop('RMUC_KINEMATIC_PHYSICS', None)
        if self.external_physics is not None:
            mode = getattr(self.external_physics, 'protocol', 'external_physics')
            if mode not in ('external_physics', 'kinematic_frame'):
                raise ValueError('Unknown physics adapter protocol')
            environment['RMUC_KINEMATIC_PHYSICS' if mode == 'kinematic_frame' else 'RMUC_EXTERNAL_PHYSICS'] = '1'
        self.lock = threading.RLock()
        self._observation_cache = {}
        if replay_version is not None:
            raise ValueError('Legacy engines are not included; historical 2027 replays remain viewable')
        binary = binary_path()
        self.process = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, env=environment)
        self.closed = False
        self._broken = False
        self._responses = queue.Queue()
        # The reader must not retain Match itself: callers also use temporary
        # Match().observation(...) instances whose destructor closes the engine.
        self._reader = None
        self._direct_reader=_PosixLineReader(self.process.stdout) if _DIRECT_PIPE_READER else None
        if self._direct_reader is None:
            self._reader=threading.Thread(target=self._read_output,args=(self.process.stdout,self._responses),daemon=True)
            self._reader.start()
        try:
            self._frame = self._request({"op": "init", "seed": int(seed), "duration": duration, "map": map_name, "lineups": lineups, "battle_settings": battle_settings, "observation_profile":observation_profile, "training_capabilities":self.training_capabilities or None})
            self._aim_references={f'{r["team"]}:{r["slot"]}':[r['aim'],r['pitch']] for r in self._frame['robots']
                                  if f'{r["team"]}:{r["slot"]}' in self._aim_executors}
            self._static = {key: self._frame[key] for key in STATIC_FRAME_KEYS if key in self._frame}
            # The latest compact step's JSON, for callers that store frames without static fields.
            self.raw_frame = None
            rule_path = ROOT / "configs/rules.json"
            expected = hashlib.sha256(rule_path.read_bytes()).hexdigest()
            if self._frame.get("rules_hash") != expected:
                raise RuntimeError("Rust engine contains old rules; rebuild packages/simulation/Cargo.toml")
            if observation_profile:
                from .sentry_protocol import SentryGateway
                self.sentry_gateway=SentryGateway(self._frame)
                if radar_classifier is not None:
                    from .radar_inference import ChassisClassifier
                    for inference in self.sentry_gateway.inference.values():inference.classifier=ChassisClassifier(radar_classifier)
        except BaseException:
            self.close()
            raise

    def _request(self, request, raw=False):
        with self.lock:
            self._ensure_available()
            # Only explicit, pure engine queries preserve the current-state cache.
            # Unknown operations invalidate conservatively, including failed writes.
            if request.get('op') not in ('observation','observations','operator_observation','snapshot','digest','terrain_probe','sentry_sensors'):
                self._observation_cache.clear()
            payload = json.dumps(request, allow_nan=False, separators=(",", ":")).encode() + b"\n"
            try:
                self.process.stdin.write(payload)
                self.process.stdin.flush()
                line = self._read_line()
                value = _response_json.loads(line)
                while value.get('op') in ('external_physics', 'kinematic_frame'):
                    if self.external_physics is None:
                        raise RuntimeError('Unexpected external physics request')
                    encoded=getattr(self.external_physics,'step_encoded',None)
                    if value.get('op')=='kinematic_frame' and callable(encoded):
                        payload=encoded(value)
                        if not isinstance(payload,bytes) or not payload.endswith(b'\n') or payload.count(b'\n')!=1:
                            raise ValueError('Encoded physics response must be one newline-terminated JSON frame')
                    else:
                        result = self.external_physics.step(value)
                        payload=json.dumps(result,allow_nan=False,separators=(",", ":")).encode()+b'\n'
                    self.process.stdin.write(payload)
                    self.process.stdin.flush()
                    line = self._read_line()
                    value = _response_json.loads(line)
            except BaseException:
                # A late response must never be mistaken for the next request's
                # reply, including a diagnostic digest requested during close.
                self._broken = True
                raise
            if value.get("error"):
                raise ValueError(value["error"])
            return (value, line) if raw else value

    def _ensure_available(self):
        if self.closed:
            raise RuntimeError("Rust engine is closed")
        if self._broken:
            raise RuntimeError("Rust engine transport interrupted; close and reset the match")

    @staticmethod
    def _viewer(team, slots):
        # Snapshot caller-owned lists before both the cache key and RPC are built.
        frozen = tuple(slots) if type(slots) in (list, tuple) else slots
        valid = (type(team) is str and team in ('red','blue') and
                 (frozen is None or type(frozen) is tuple and
                  all(type(slot) is int and slot in SLOT_IDS for slot in frozen)))
        return frozen, (team, frozen) if valid else None

    def _remember_observation(self, key, ids, line):
        if _CACHE_POLICY_OBSERVATIONS and key is not None:
            if key not in self._observation_cache and len(self._observation_cache) >= 32:
                del self._observation_cache[next(iter(self._observation_cache))]
            # Only immutable engine bytes/ids are retained. Policy augmentation
            # and caller mutation must never enter another viewer's observation.
            self._observation_cache[key] = (tuple(ids), line)

    @staticmethod
    def _read_output(stream, responses):
        try:
            while True:
                line = stream.readline(4 * 1024 * 1024)
                responses.put(line)
                if not line:
                    return
        except (OSError, ValueError):
            responses.put(b"")

    def _read_line(self):
        if self._direct_reader is not None:
            line=self._direct_reader.read_line()
        else:
            try:
                line = self._responses.get(timeout=10)
            except queue.Empty:
                raise RuntimeError("Rust engine response timed out") from None
        if not line:
            raise RuntimeError("Rust engine exited unexpectedly")
        return line

    def __getattr__(self, name):
        if name in ("tick", "time", "status", "winner", "reason", "seed", "duration", "arena", "robots", "bases", "outposts", "teams", "zones", "events", "projectiles"):
            return self._frame[name]
        raise AttributeError(name)

    def step(self, commands):
        if self.capability_drift:
            self.training_capabilities=self.capability_drift.update(self._frame['time'])
            for seat,executor in self._aim_executors.items():executor.set_profile(self.training_capabilities[seat]['aim'])
        if self._aim_executors:
            commands=copy.deepcopy(commands)
            for robot in self._frame['robots']:
                team, slot=robot['team'],str(robot['slot'])
                executor=self._aim_executors.get(f'{team}:{slot}')
                if executor is not None:
                    action=commands.setdefault(team,{}).setdefault(slot,{})
                    aim=action.get('aim');pitch=action.get('pitch')
                    desired=self._aim_references[f'{team}:{slot}']
                    if aim is not None:desired[0]=aim
                    if pitch is not None:desired[1]=pitch
                    result=executor.step(self._frame['time'],*desired)
                    action['aim']=result['aim'] if result['ready'] else robot['aim']
                    action['pitch']=result['pitch'] if result['ready'] else robot['pitch']
                    if not result['ready']:action['fire']=False
        if self.sentry_gateway:
            if self.training_summary:raise ValueError('Strict sentry profile requires full referee frames')
            commands={**commands,**{team:{**commands.get(team,{}),'7':self.sentry_gateway.command(team,commands.get(team,{}).get('7',{}))} for team in ('red','blue')}}
            commands['radar_effects']=self.sentry_gateway.radar.effects(self._frame)
        frame, line = self._request({"op": "training_step" if self.training_summary else "step", "commands": commands, "static": False,
                                    "training_capabilities":self.training_capabilities or None}, raw=True)
        self._frame = dict(sorted({**frame, **self._static}.items()))
        if self.sentry_gateway:self.sentry_gateway.update(self._frame)
        self.raw_frame = line.rstrip(b"\n")
        # Frames are treated as immutable once returned; callers may still extend their events.
        return {**self._frame, "events": list(self._frame["events"])}

    def observation(self, team, slots=None):
        with self.lock:
            self._ensure_available()
            slots,key=self._viewer(team,slots)
            cached=self._observation_cache.get(key) if _CACHE_POLICY_OBSERVATIONS else None
            if cached is not None:
                return _response_json.loads(cached[1])
            if self.sentry_gateway and (slots is None or 7 in slots):
                if slots != (7,):raise ValueError('Strict sentry requires its own single-slot controller')
                sensors=self._request({'op':'sentry_sensors','team':team})
                value=self.sentry_gateway.observation(team,sensors['targets'])
                line=json.dumps(value,allow_nan=False,separators=(',',':')).encode()+b'\n'
                self._remember_observation(key,['7'],line)
                return value
            value,line=self._request({"op":"observation","team":team,"slots":slots},raw=True)
            self._remember_observation(key,value['controlled_ids'],line)
            return value

    def compact_observation(self, team, slots=None):
        """Reuse the immutable map for local callers making per-player observations."""
        if self.sentry_gateway:return self.observation(team,slots)
        value = self._request({"op": "observation", "team": team, "slots": slots, "static": False})
        return {"arena": self._static["arena"], **value}

    def operator_observation(self, team, slot):
        """Own optical view including a wreck; never pass this observation to policies."""
        value = self._request({"op": "operator_observation", "team": team, "slot": slot})
        return {"arena": self._static["arena"], **value}

    def observations(self, viewers):
        """Encoded observations for (team, slots) viewers in a single engine round trip.

        Returns each observation's controlled ids and its JSON line, which policies can
        receive as is instead of being decoded and encoded again.
        """
        with self.lock:
            self._ensure_available()
            if self.sentry_gateway:
                # Separate strict and legacy recipients, preserving the batch result contract.
                values=[self.observation(team,slots) for team,slots in viewers]
                return [v['controlled_ids'] for v in values],[json.dumps(v,allow_nan=False,separators=(',',':')).encode()+b'\n' for v in values]
            viewers=[(team,*self._viewer(team,slots)) for team,slots in viewers]
            results=[self._observation_cache.get(key) if _CACHE_POLICY_OBSERVATIONS else None for _,_,key in viewers]
            missing=[i for i,result in enumerate(results) if result is None]
            if not missing:
                return [list(result[0]) for result in results], [result[1] for result in results]
            header = self._request({"op": "observations", "requests": [{"team": viewers[i][0], "slots": viewers[i][1]} for i in missing]})
            # The engine builds every observation before replying, so the lines follow at once.
            try:
                lines=[self._read_line() for _ in missing]
                controlled=header['controlled_ids']
                if len(controlled)!=len(missing):
                    raise RuntimeError('Invalid observation batch size')
                for i,ids,line in zip(missing,controlled,lines):
                    results[i]=(tuple(ids),line)
                    self._remember_observation(viewers[i][2],ids,line)
                return [list(result[0]) for result in results], [result[1] for result in results]
            except BaseException:
                self._broken=True
                raise

    def snapshot(self):
        return self._request({"op": "snapshot"}) if self.training_summary else copy.deepcopy(self._frame)

    def digest(self):
        return self._request({"op": "digest"})["digest"]

    def finish(self, winner, reason):
        self._frame = self._request({"op": "finish", "winner": winner, "reason": reason})

    def close(self):
        adapter = self.external_physics
        self.external_physics = None
        try:
            if adapter is not None:
                adapter.close()
        finally:
            self._close_transport()

    def _close_transport(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self._broken:
                # Do this before closing buffered stdin: its pending flush
                # could otherwise block behind an unread response tail.
                self.process.kill()
            try:
                self.process.stdin.close()
            except (OSError, ValueError):
                pass
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        finally:
            if self._reader is not None:self._reader.join(timeout=2)
            self.process.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        if hasattr(self, "closed"):
            self.close()
