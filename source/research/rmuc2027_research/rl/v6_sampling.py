"""A persisted coordinator for distinct worker leases and honest v6 coverage.

Only training workers use this file. Evaluation never records a failure here.
Progress is cumulative per lease, so checkpointing and episode close cannot
double count the same observed samples. No physics state is serialized.
"""
from contextlib import contextmanager
import copy
import fcntl
import json
import os
from pathlib import Path
import random
import time

from . import v6_curriculum

from .scenarios import REGISTRY_HASH, ScenarioSpec, scenario_registry, _tuples

RECOVERY_KINDS = ('free_ready', 'paid_early', 'unavailable', 'alive_weak')


class TrainingCoordinator:
    def __init__(self, path, seed=20261111):
        self.path = Path(path)
        self.seed = int(seed)
        self.curriculum_active = False
        self.curriculum_backend = None
        self.cases = {x.scenario_id: x for x in scenario_registry()}

    @contextmanager
    def transaction(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.path.exists():
                state = json.loads(self.path.read_text())
                if state.get('version') != 1 or state.get('registry_hash') != REGISTRY_HASH:
                    raise ValueError('Unsupported v6 coordinator state')
            else:
                state = dict(version=1, registry_hash=REGISTRY_HASH, seed=self.seed,
                             rng_state=random.Random(self.seed).getstate(), sequence=0,
                             counts={}, failures={}, leases={}, history=[])
            yield state
            pending = self.path.with_suffix('.tmp')
            with pending.open('w') as stream:
                json.dump(state, stream, allow_nan=False)
                stream.flush(); os.fsync(stream.fileno())
            os.replace(pending, self.path)

    def configure_curriculum(self, config, *, backend='warp_kinematic'):
        """Opt-in per process; absent config preserves the original sampler."""
        normalized = v6_curriculum.normalize_config(config)
        self.curriculum_active = normalized is not None
        self.curriculum_backend = backend
        if normalized is not None:
            with self.transaction() as state:
                if any(r.get('curriculum_generation') != normalized['generation']
                       for r in state['leases'].values()):
                    raise ValueError('Close old-generation leases before enabling a new curriculum')
                state['curriculum'] = v6_curriculum.configure(state.get('curriculum'), config)
        return self.curriculum_active

    def request_normal_fallback(self, worker):
        """After one failed preparation, the same reset gets real normal play."""
        if self.curriculum_active:
            with self.transaction() as state:
                state.setdefault('normal_fallback_workers', {})[str(worker)] = True

    def report_prefix(self, worker, report, *, log_path=None, now=None):
        if not self.curriculum_active:
            return None
        with self.transaction() as state:
            worker = str(worker)
            lease_id = state.get('last_lease_by_worker', {}).get(worker)
            lease = state['leases'].get(worker)
            if lease is None or lease['id'] != lease_id:
                lease = next((r for r in reversed(state['history']) if r['id'] == lease_id), None)
            if lease is None or lease.get('worker_id') != worker or not lease.get('recovery_kind'):
                raise ValueError('No recovery lease matches the prefix report')
            if not report.get('prepared') and (lease['valid_steps'] or lease['seconds']):
                raise ValueError('Failed preparation already contains actor samples')
            event = v6_curriculum.record_prefix(state['curriculum'], lease_id=lease['id'],
                kind=lease['recovery_kind'], backend=lease['recovery_backend'],
                side=lease['scenario']['learner_side'], report=report,
                now=time.time() if now is None else now, log_path=log_path)
            lease['prefix_outcome'] = event
            if not report.get('prepared'):
                lease['reason'] = 'prefix_not_prepared:'+str(report.get('reason'))
                lease['failure'] = False
            return event

    def initialize_history(self, samplers):
        """Import the ancestor's recorded counts once, never infer completion."""
        with self.transaction() as state:
            if state.get('ancestor_imported'):
                return
            for item in samplers:
                old = item.get('session', {}).get('scenario', {})
                for key, row in old.get('counts', {}).items():
                    if key not in self.cases:
                        raise ValueError('Ancestor uses an unknown scenario')
                    target = state['counts'].setdefault(key, self.empty_counts())
                    for field in target:
                        target[field] += row.get(field, 0)
                for key, weight in old.get('failures', {}).items():
                    state['failures'][key] = max(state['failures'].get(key, 0), weight)
            state['ancestor_imported'] = True
            state['ancestor_counts_preserved_without_deduplication'] = True

    @staticmethod
    def empty_counts():
        return dict(started=0, valid_steps=0, seconds=0., short_completed=0, full_completed=0)

    def acquire(self, worker):
        worker = str(worker)
        with self.transaction() as state:
            if worker in state['leases']:
                raise RuntimeError('Worker must close its previous training lease')
            rng = random.Random(); rng.setstate(_tuples(state['rng_state']))
            busy = {row['scenario_id'] for row in state['leases'].values()}
            free = [key for key in self.cases if key not in busy]
            failures = [key for key in free if state['failures'].get(key, 0) > 0]
            replay = bool(failures) and rng.random() < .3
            if replay:
                key = rng.choices(failures, [state['failures'][x] for x in failures])[0]
            else:
                # Prioritize completed coverage, then starts; unfinished cases
                # remain eligible, but simultaneous duplication is prohibited.
                def priority(key):
                    row = state['counts'].get(key, self.empty_counts())
                    return row['short_completed'], row['started']
                minimum = min(map(priority, free))
                eligible = [key for key in free if priority(key) == minimum]
                side = 'red' if (state['sequence']//2 + state['sequence']//8) % 2 == 0 else 'blue'
                balanced = [key for key in eligible if self.cases[key].learner_side == side]
                key = rng.choice(balanced or eligible)
            sequence = state['sequence']; state['sequence'] += 1
            kind = RECOVERY_KINDS[(sequence // 2) % 4] if sequence % 2 else None
            curriculum_reason = None
            if self.curriculum_active:
                if state.setdefault('normal_fallback_workers', {}).pop(worker, False):
                    kind, curriculum_reason = None, 'failed_prefix_normal_fallback'
                else:
                    kind, curriculum_reason = v6_curriculum.choose(state['curriculum'], rng,
                        backend=self.curriculum_backend, side=self.cases[key].learner_side, now=time.time())
            lease = dict(id=sequence, scenario_id=key, scenario=self.cases[key].to_dict(),
                         seed=rng.randrange(2**31), recovery_kind=kind,
                         source='training_failure' if replay else 'coverage',
                         valid_steps=0, seconds=0.)
            if self.curriculum_active:
                lease.update(worker_id=worker, recovery_backend=self.curriculum_backend,
                    curriculum_generation=state['curriculum']['config']['generation'],
                    curriculum_reason=curriculum_reason)
                state.setdefault('last_lease_by_worker', {})[worker] = sequence
            state['counts'].setdefault(key, self.empty_counts())['started'] += 1
            state['leases'][worker] = lease
            state['rng_state'] = rng.getstate()
            return copy.deepcopy(lease)

    def progress(self, worker, lease_id, *, valid_steps, seconds, completed=False,
                 failure=False, close=False, reason=None):
        if valid_steps < 0 or seconds < 0:
            raise ValueError('Negative training progress')
        with self.transaction() as state:
            lease = state['leases'].get(str(worker))
            if lease is None or lease['id'] != lease_id:
                raise ValueError('Unknown training lease')
            if valid_steps < lease['valid_steps'] or seconds + 1e-9 < lease['seconds']:
                raise ValueError('Training progress cannot regress')
            row = state['counts'][lease['scenario_id']]
            if self.curriculum_active:
                v6_curriculum.add_actor_progress(state['curriculum'], lease['recovery_kind'],
                    valid_steps-lease['valid_steps'], seconds-lease['seconds'])
            row['valid_steps'] += valid_steps - lease['valid_steps']
            row['seconds'] += seconds - lease['seconds']
            lease.update(valid_steps=valid_steps, seconds=seconds)
            if completed and not close:
                raise ValueError('Completion must close a lease')
            if close:
                row['short_completed'] += int(completed and seconds >= 60 - 1e-6)
                row['full_completed'] += int(completed and seconds >= 420 - 1e-6 and not lease['recovery_kind'])
                key = lease['scenario_id']
                if failure:
                    state['failures'][key] = min(100., state['failures'].get(key, 0.) + 1)
                elif completed and key in state['failures']:
                    state['failures'][key] *= .9
                state['history'].append(dict(**lease, completed=bool(completed), failure=bool(failure), reason=reason))
                del state['leases'][str(worker)]

    def release_interrupted_leases(self):
        """Resume from fresh episodes; already recorded partial progress remains."""
        with self.transaction() as state:
            for lease in state['leases'].values():
                state['history'].append(dict(**lease, completed=False, failure=False, reason='resume_discard_partial_episode'))
            state['leases'] = {}

    def snapshot(self):
        with self.transaction() as state:
            result = copy.deepcopy(state)
        result['coverage'] = {}
        for field in ('started', 'valid_steps', 'short_completed', 'full_completed'):
            keys = [k for k, row in result['counts'].items() if row[field] > 0]
            result['coverage'][field] = dict(scenarios=len(keys),
                quantity_groups=len({self.cases[k].quantity_id for k in keys}),
                ordered=len({self.cases[k].ordered_id for k in keys}))
        return result
