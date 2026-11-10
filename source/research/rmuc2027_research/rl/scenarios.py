"""Relative-side sentry scenarios and resumable, coverage-first sampling.

Sampling a case records a start, never a completed training/evaluation episode.
The learner always owns model 2773, including when its colour changes.
"""
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
import random

VERSION = 1
INFANTRY_MODELS = (2731, 2732, 303)
SENTRY_MODELS = (2772, 2773, 2771)
KINDS = {2731: "W", 2732: "T", 303: "O", 2772: "W", 2773: "T", 2771: "O"}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _tuples(value):
    return tuple(map(_tuples, value)) if isinstance(value, (tuple, list)) else value


@dataclass(frozen=True)
class ScenarioSpec:
    friendly_infantry: tuple = (2731, 2732)
    enemy_infantry: tuple = (2731, 2732)
    enemy_sentry: int = 2773
    learner_side: str = "red"

    def __post_init__(self):
        for key in ("friendly_infantry", "enemy_infantry"):
            values = tuple(getattr(self, key))
            if len(values) != 2 or any(type(v) is not int or v not in INFANTRY_MODELS for v in values):
                raise ValueError("Each infantry pair must use the three representative legal models")
            object.__setattr__(self, key, values)
        if type(self.enemy_sentry) is not int or self.enemy_sentry not in SENTRY_MODELS:
            raise ValueError("Unsupported enemy sentry model")
        if self.learner_side not in ("red", "blue"):
            raise ValueError("Unknown learner side")

    @property
    def ordered_id(self):
        return "f{}-{}_e{}-{}_s{}".format(*self.friendly_infantry, *self.enemy_infantry, self.enemy_sentry)

    @property
    def scenario_id(self):
        return self.ordered_id + "_" + self.learner_side

    @property
    def quantity_id(self):
        def count(pair):
            return "".join(str(sum(KINDS[m] == kind for m in pair)) for kind in "WTO")
        return "f" + count(self.friendly_infantry) + "_e" + count(self.enemy_infantry) + "_s" + KINDS[self.enemy_sentry]

    def lineups(self, heavy_model=2711, aerial_model=2761):
        from rmuc2027_core.models import normalize_team_lineup
        friendly = dict(zip(("3", "4"), self.friendly_infantry), **{"1": heavy_model, "6": aerial_model, "7": 2773})
        enemy = dict(zip(("3", "4"), self.enemy_infantry), **{"1": heavy_model, "6": aerial_model, "7": self.enemy_sentry})
        other = "blue" if self.learner_side == "red" else "red"
        return {self.learner_side: normalize_team_lineup(friendly, "2027"),
                other: normalize_team_lineup(enemy, "2027")}

    def to_dict(self):
        return {**asdict(self), "scenario_id": self.scenario_id, "ordered_id": self.ordered_id,
                "quantity_id": self.quantity_id}

    @classmethod
    def from_dict(cls, value):
        result = cls(**{key: value[key] for key in cls.__dataclass_fields__})
        for key in ("scenario_id", "ordered_id", "quantity_id"):
            if key in value and value[key] != getattr(result, key):
                raise ValueError("Scenario identifier does not match its hardware")
        return result


def scenario_registry(include_sides=True):
    sides = ("red", "blue") if include_sides else ("red",)
    return tuple(ScenarioSpec((a, b), (c, d), s, side)
                 for a, b, c, d, s, side in itertools.product(
                     INFANTRY_MODELS, INFANTRY_MODELS, INFANTRY_MODELS, INFANTRY_MODELS,
                     SENTRY_MODELS, sides))


REGISTRY_HASH = canonical_hash([s.to_dict() for s in scenario_registry()])


class ScenarioSampler:
    """Visit every quantity group first, then every ordered case and colour.

    Following the first coverage pass, sample quantity groups uniformly, mixed
    with replay of *training* failures. Callers must not record held-out results
    here. state_dict contains JSON-safe RNG, queue cursor and separate counters.
    """
    def __init__(self, seed=42, failure_fraction=.3):
        if not math.isfinite(failure_fraction) or not 0 <= failure_fraction <= 1:
            raise ValueError("Invalid failure replay probability")
        self.seed, self.failure_fraction = int(seed), float(failure_fraction)
        self.rng = random.Random(self.seed)
        self.cases = {s.scenario_id: s for s in scenario_registry()}
        self.groups = defaultdict(list)
        for case in self.cases.values():
            self.groups[case.quantity_id].append(case.scenario_id)
        groups = list(self.groups)
        self.rng.shuffle(groups)
        remaining = {}
        # Explicit colour alternation makes the 108 first representatives balanced.
        for index, group in enumerate(groups):
            items = list(self.groups[group]); self.rng.shuffle(items)
            side = "red" if index % 2 == 0 else "blue"
            first = next(item for item in items if self.cases[item].learner_side == side)
            remaining[group] = [first] + [item for item in items if item != first]
        self.queue = []
        while any(remaining.values()):
            for group in groups:
                if remaining[group]:
                    self.queue.append(remaining[group].pop(0))
        self.cursor = 0
        self.counts = {}
        self.failures = {}

    def sample(self):
        if self.cursor < len(self.queue):
            key = self.queue[self.cursor]; self.cursor += 1
        elif self.failures and self.rng.random() < self.failure_fraction:
            keys = sorted(self.failures)
            key = self.rng.choices(keys, weights=[self.failures[k] for k in keys], k=1)[0]
        else:
            group = self.rng.choice(sorted(self.groups))
            key = self.rng.choice(self.groups[group])
        row = self.counts.setdefault(key, dict(started=0, valid_steps=0, seconds=0., short_completed=0, full_completed=0))
        row["started"] += 1
        return self.cases[key]

    def next_episode(self):
        case = self.sample()
        return {"scenario": case.to_dict(), "seed": self.rng.randrange(2**31)}

    def record(self, scenario, *, valid_steps=0, seconds=0., completed=False, failure=False, split="train"):
        if split != "train":
            raise ValueError("Held-out results must not update the training sampler")
        key = scenario.scenario_id if isinstance(scenario, ScenarioSpec) else str(scenario)
        if key not in self.cases or type(valid_steps) is not int or valid_steps < 0 or not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid training episode accounting")
        row = self.counts.setdefault(key, dict(started=0, valid_steps=0, seconds=0., short_completed=0, full_completed=0))
        row["valid_steps"] += valid_steps; row["seconds"] += float(seconds)
        row["short_completed"] += int(bool(completed) and seconds >= 60 - 1e-6)
        row["full_completed"] += int(bool(completed) and seconds >= 420 - 1e-6)
        if failure:
            self.failures[key] = min(100., self.failures.get(key, 0.) + 1.)
        elif key in self.failures:
            self.failures[key] *= .9
            if self.failures[key] < .1:
                self.failures.pop(key)

    def coverage(self):
        def observed(field):
            keys = [k for k, row in self.counts.items() if row[field] > 0]
            return {"scenarios": len(keys), "ordered": len({self.cases[k].ordered_id for k in keys}),
                    "quantity_groups": len({self.cases[k].quantity_id for k in keys})}
        return {"registered": {"scenarios": 486, "ordered": 243, "quantity_groups": 108},
                **{field: observed(field) for field in ("started", "valid_steps", "short_completed", "full_completed")}}

    def state_dict(self):
        return json.loads(json.dumps(dict(version=VERSION, registry_hash=REGISTRY_HASH, seed=self.seed,
            failure_fraction=self.failure_fraction, rng_state=self.rng.getstate(), queue=self.queue,
            cursor=self.cursor, counts=self.counts, failures=self.failures), allow_nan=False))

    @classmethod
    def from_state_dict(cls, state):
        if state.get("version") != VERSION or state.get("registry_hash") != REGISTRY_HASH:
            raise ValueError("Scenario registry/schema changed; explicit migration required")
        result = cls(state["seed"], state["failure_fraction"])
        if len(state["queue"]) != len(result.cases) or set(state["queue"]) != set(result.cases):
            raise ValueError("Invalid coverage queue")
        if type(state["cursor"]) is not int or not 0 <= state["cursor"] <= len(result.cases):
            raise ValueError("Invalid coverage cursor")
        if not set(state["counts"]) <= result.cases.keys() or not set(state["failures"]) <= result.cases.keys():
            raise ValueError("Unknown scenario in sampler state")
        result.queue = list(state["queue"]); result.cursor = state["cursor"]
        result.counts = json.loads(json.dumps(state["counts"], allow_nan=False))
        result.failures = dict(state["failures"])
        if any(not math.isfinite(v) or v <= 0 for v in result.failures.values()):
            raise ValueError("Invalid failure replay weights")
        result.rng.setstate(_tuples(state["rng_state"]))
        return result
