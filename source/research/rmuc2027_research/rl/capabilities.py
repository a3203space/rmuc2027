"""Sampled hardware multipliers, bounded enemy drift and delayed OU aiming.

These are execution parameters, not privileged actor observations. Application
to physics is owned by the environment. No projectile-spread setting is changed.
"""
from collections import deque
import copy
import math
import random

from .scenarios import canonical_hash, _tuples

VERSION = 1
MOBILITY_FIELDS = ("speed_reference_mps", "motor_force_max_n", "drive_efficiency")
MOBILITY_PROFILES = {"low": dict(zip(MOBILITY_FIELDS, (.75, .8, .9))),
                     "nominal": dict(zip(MOBILITY_FIELDS, (1., 1., 1.))),
                     "high": dict(zip(MOBILITY_FIELDS, (1.2, 1.15, 1.05)))}
AIM_PROFILES = {"strong": {"delay_s": .05, "sigma_deg": .25, "tau_s": .15},
                "medium": {"delay_s": .15, "sigma_deg": 1., "tau_s": .15},
                "weak": {"delay_s": .3, "sigma_deg": 2.5, "tau_s": .15},
                "learner": {"delay_s": 0., "sigma_deg": 0., "tau_s": .15}}
PROBABILITIES = (.25, .5, .25)
DRIFT = dict(target_interval_s=15., time_constant_s=20., speed_fraction=.10,
             force_fraction=.08, efficiency_fraction=.03, aim_sigma_fraction=.20,
             aim_delay_delta_s=.05, aim_history_s=2., scope="enemy_ground_only")
SCHEMA_HASH = canonical_hash(dict(version=VERSION, mobility=MOBILITY_PROFILES,
                                 aim=AIM_PROFILES, probabilities=PROBABILITIES, drift=DRIFT))


class CapabilitySampler:
    def __init__(self, seed=42):
        self.seed = int(seed); self.rng = random.Random(self.seed); self.episodes = 0

    def sample(self, lineups, learner_side):
        from rmuc2027_core.models import robot_models
        if learner_side not in ("red", "blue") or set(lineups) != {"red", "blue"}:
            raise ValueError("Capabilities require both sides and a canonical learner side")
        if lineups[learner_side].get("7", lineups[learner_side].get(7)) != 2773:
            raise ValueError("The protected learner must be 2773")
        result = {}
        designs = robot_models()
        for team in ("red", "blue"):
            for slot in (1, 3, 4, 7):
                learner = team == learner_side and slot == 7
                mobility = "nominal" if learner else self.rng.choices(tuple(MOBILITY_PROFILES), PROBABILITIES)[0]
                aim = "learner" if learner else self.rng.choices(("strong", "medium", "weak"), PROBABILITIES)[0]
                model = lineups[team].get(str(slot), lineups[team].get(slot))
                result[f"{team}:{slot}"] = dict(mobility_profile=mobility, mobility=copy.deepcopy(MOBILITY_PROFILES[mobility]),
                    aim_profile=aim, aim=copy.deepcopy(AIM_PROFILES[aim]), aim_seed=self.rng.randrange(2**31),
                    protected_learner=learner, observation_scope="strict_sentry" if slot == 7 else "helper",
                    calibration_scale=1., calibration_source="fixed_unfitted; 2026 envelopes are validation only",
                    model=model, nominal_drive_efficiency=float(designs[str(model)]["drive_efficiency"]))
        self.episodes += 1
        return result

    def state_dict(self):
        return dict(version=VERSION, schema_hash=SCHEMA_HASH, seed=self.seed, episodes=self.episodes,
                    rng_state=self.rng.getstate())

    @classmethod
    def from_state_dict(cls, state):
        if state.get("version") != VERSION or state.get("schema_hash") != SCHEMA_HASH:
            raise ValueError("Capability table changed; explicit migration required")
        result = cls(state["seed"]); result.episodes = int(state["episodes"])
        if result.episodes < 0:
            raise ValueError("Invalid capability sampler count")
        result.rng.setstate(_tuples(state["rng_state"]))
        return result


class SlowCapabilityDrift:
    """Bounded enemy hardware/aim drift around the initially sampled profile.

    update integrates exactly between 15 s target changes, so reducing the control
    step does not change target RNG draws or multiply the drift cumulatively.
    """
    def __init__(self, base_profiles, seed=42, learner_side="red"):
        if learner_side not in ("red", "blue"):
            raise ValueError("Unknown learner side")
        self.base = copy.deepcopy(base_profiles)
        self.current = copy.deepcopy(base_profiles)
        self.targets = copy.deepcopy(base_profiles)
        self.learner_side = learner_side; self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.last_time = 0.; self.next_target_time = DRIFT["target_interval_s"]
        self.seats = []
        for seat, profile in sorted(self.base.items()):
            team, slot = seat.split(":")
            if team not in ("red", "blue") or slot not in ("1", "3", "4", "6", "7"):
                raise ValueError("Unknown capability seat")
            nominal = float(profile.get("nominal_drive_efficiency", .85))
            if not math.isfinite(nominal) or not 0 < nominal <= 1:
                raise ValueError("Invalid nominal drive efficiency")
            if nominal * profile["mobility"]["drive_efficiency"] > 1 + 1e-9:
                raise ValueError("Initial effective efficiency exceeds one")
            if team != learner_side and int(slot) in (1, 3, 4, 7) and not profile.get("protected_learner", False):
                self.seats.append(seat)

    def _new_targets(self):
        for seat in self.seats:
            base, target = self.base[seat], self.targets[seat]
            for key, fraction in zip(MOBILITY_FIELDS, (DRIFT["speed_fraction"], DRIFT["force_fraction"], DRIFT["efficiency_fraction"])):
                target["mobility"][key] = base["mobility"][key] * (1 + self.rng.uniform(-fraction, fraction))
            limit = 1 / float(base.get("nominal_drive_efficiency", .85))
            target["mobility"]["drive_efficiency"] = min(limit, target["mobility"]["drive_efficiency"])
            target["aim"]["sigma_deg"] = base["aim"]["sigma_deg"] * (1 + self.rng.uniform(-DRIFT["aim_sigma_fraction"], DRIFT["aim_sigma_fraction"]))
            target["aim"]["delay_s"] = max(0., base["aim"]["delay_s"] + self.rng.uniform(-DRIFT["aim_delay_delta_s"], DRIFT["aim_delay_delta_s"]))

    def _advance(self, until):
        alpha = -math.expm1(-(until-self.last_time) / DRIFT["time_constant_s"])
        for seat in self.seats:
            for group, fields in (("mobility", MOBILITY_FIELDS), ("aim", ("sigma_deg", "delay_s"))):
                for field in fields:
                    now, target = self.current[seat][group][field], self.targets[seat][group][field]
                    self.current[seat][group][field] = now + alpha * (target-now)
        self.last_time = until

    def update(self, time_s):
        time_s = float(time_s)
        if not math.isfinite(time_s) or time_s < self.last_time:
            raise ValueError("Capability drift time must be finite and monotonic")
        while self.next_target_time <= time_s:
            self._advance(self.next_target_time)
            self._new_targets()
            self.next_target_time += DRIFT["target_interval_s"]
        self._advance(time_s)
        return copy.deepcopy(self.current)

    def state_dict(self):
        return dict(version=VERSION, schema_hash=SCHEMA_HASH, seed=self.seed, learner_side=self.learner_side,
                    base=copy.deepcopy(self.base), current=copy.deepcopy(self.current), targets=copy.deepcopy(self.targets),
                    rng_state=self.rng.getstate(), last_time=self.last_time, next_target_time=self.next_target_time)

    @classmethod
    def from_state_dict(cls, state):
        if state.get("version") != VERSION or state.get("schema_hash") != SCHEMA_HASH:
            raise ValueError("Capability drift schema changed")
        result = cls(state["base"], state["seed"], state["learner_side"])
        if set(state["current"]) != set(result.base) or set(state["targets"]) != set(result.base):
            raise ValueError("Capability drift seats changed")
        result.current = copy.deepcopy(state["current"]); result.targets = copy.deepcopy(state["targets"])
        result.last_time = float(state["last_time"]); result.next_target_time = float(state["next_target_time"])
        if not 0 <= result.last_time < result.next_target_time or not math.isfinite(result.next_target_time):
            raise ValueError("Invalid drift clock")
        result.rng.setstate(_tuples(state["rng_state"]))
        return result


class AimNoiseExecutor:
    """Delay the aim reference, then apply exact discrete OU angular error.

    sigma_deg is stationary standard deviation per angular axis; tau_s is the
    correlation time. Repeated calls at the same time neither advance noise nor
    replace the already latched reference. Run this at the executor's 20 Hz.
    """
    def __init__(self, profile="learner", seed=42):
        self.set_profile(profile)
        self.rng = random.Random(int(seed)); self.queue = deque(); self.last_time = None
        self.noise = [0., 0.]; self.last_result = None

    def set_profile(self, profile):
        """Change the profile without resetting correlated error or aim history."""
        profile = AIM_PROFILES[profile] if isinstance(profile, str) else profile
        value = {key: float(profile[key]) for key in ("delay_s", "sigma_deg", "tau_s")}
        if (not all(math.isfinite(v) for v in value.values()) or not 0 <= value["delay_s"] <= DRIFT["aim_history_s"]
                or value["sigma_deg"] < 0 or value["tau_s"] <= 0):
            raise ValueError("Invalid delayed OU aim profile")
        self.profile = value

    def step(self, time_s, aim_rad, pitch_rad):
        values = tuple(map(float, (time_s, aim_rad, pitch_rad)))
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Aim reference must be finite")
        time_s, aim_rad, pitch_rad = values
        if self.last_time is not None and time_s < self.last_time:
            raise ValueError("Aim executor time went backwards; reset the episode")
        if self.last_time == time_s:
            return copy.deepcopy(self.last_result)
        dt = .05 if self.last_time is None else time_s - self.last_time
        decay = math.exp(-dt / self.profile["tau_s"])
        sigma = math.radians(self.profile["sigma_deg"])
        innovation = sigma * math.sqrt(-math.expm1(-2 * dt / self.profile["tau_s"]))
        self.noise = [decay * value + innovation * self.rng.gauss(0., 1.) for value in self.noise]
        self.queue.append((time_s, aim_rad, pitch_rad))
        target = time_s - self.profile["delay_s"]
        while self.queue and self.queue[0][0] < time_s - DRIFT["aim_history_s"] - 1e-9:
            self.queue.popleft()
        stamp, aim, pitch = next((row for row in reversed(self.queue) if row[0] <= target + 1e-9), self.queue[0])
        ready = stamp <= target + 1e-9
        # No usable delayed reference at startup: callers must reject firing.
        yaw = aim if self.noise[0] == 0 else (aim + self.noise[0] + math.pi) % math.tau - math.pi
        result = dict(ready=ready, aim=yaw,
                      pitch=pitch + self.noise[1], age_s=time_s-stamp,
                      noise_yaw_rad=self.noise[0], noise_pitch_rad=self.noise[1])
        self.last_time = time_s; self.last_result = result
        return copy.deepcopy(result)

    def state_dict(self):
        return dict(version=VERSION, schema_hash=SCHEMA_HASH, profile=dict(self.profile),
                    rng_state=self.rng.getstate(), queue=list(self.queue), last_time=self.last_time,
                    noise=list(self.noise), last_result=copy.deepcopy(self.last_result))

    @classmethod
    def from_state_dict(cls, state):
        if state.get("version") != VERSION or state.get("schema_hash") != SCHEMA_HASH:
            raise ValueError("Aim executor schema changed")
        result = cls(state["profile"])
        result.rng.setstate(_tuples(state["rng_state"]))
        result.queue = deque(tuple(map(float, row)) for row in state["queue"])
        result.noise = list(map(float, state["noise"]))
        result.last_time = state["last_time"]
        result.last_result = copy.deepcopy(state["last_result"])
        if len(result.noise) != 2 or not all(math.isfinite(v) for v in result.noise):
            raise ValueError("Invalid OU state")
        return result
