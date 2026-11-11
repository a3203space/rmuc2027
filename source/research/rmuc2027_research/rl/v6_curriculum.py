"""Training-only recovery sampling and persistent preparation circuit breakers.

No policy inputs, simulator commands, rewards or success labels are constructed
here. A failed preparation contributes zero actor samples and never becomes a
policy failure. All clocks are explicit UTC seconds for portable checkpoints.
"""
import copy
import hashlib
import json
import math

KINDS = ('free_ready', 'paid_early', 'unavailable', 'alive_weak')
ALIASES = {'paid_beneficial': 'paid_early'}


def canonical_kind(kind):
    return ALIASES.get(kind, kind)


def normalize_config(value):
    if not value or not value.get('enabled', False):
        return None
    generation = value.get('generation')
    if not isinstance(generation, str) or not generation:
        raise ValueError('Enabled recovery curriculum requires an explicit generation')
    probability = value.get('recovery_probability', .4)
    failures = value.get('failure_limit', 2)
    cooldown = value.get('cooldown_seconds', 1800.)
    if (isinstance(probability, bool) or not isinstance(probability, (float, int))
            or not math.isfinite(probability) or not 0 <= probability <= 1):
        raise ValueError('Invalid recovery probability')
    if isinstance(failures, bool) or not isinstance(failures, int) or failures < 1:
        raise ValueError('Invalid recovery failure limit')
    if (isinstance(cooldown, bool) or not isinstance(cooldown, (float, int))
            or not math.isfinite(cooldown) or cooldown <= 0):
        raise ValueError('Invalid recovery cooldown')
    return dict(enabled=True, generation=generation, recovery_probability=float(probability),
                failure_limit=failures, cooldown_seconds=float(cooldown),
                probability_basis='episode_lease_attempts', scope='kind/backend/side')


def configure(previous, value):
    config = normalize_config(value)
    if config is None:
        return previous
    if previous and previous.get('config') == config:
        return previous
    if previous and previous.get('config', {}).get('generation') == config['generation']:
        raise ValueError('A curriculum generation cannot change its sampling/breaker contract')
    return dict(version=1, config=config, scopes={}, attempts=[],
                actor_steps={key: 0 for key in ('normal', *KINDS)},
                actor_seconds={key: 0. for key in ('normal', *KINDS)},
                ancestor=copy.deepcopy(previous) if previous else None,
                ancestor_reused_for_breaker=False)


def scope_key(kind, backend, side):
    kind = canonical_kind(kind)
    if kind not in KINDS or backend not in ('warp_kinematic', 'rapier', 'mjwarp') or side not in ('red', 'blue'):
        raise ValueError('Invalid recovery preparation scope')
    return f'{kind}/{backend}/{side}'


def choose(state, rng, *, backend, side, now):
    """60/40 default *lease probabilities*; actual actor-step shares are audited."""
    if not state or not state.get('config', {}).get('enabled'):
        raise ValueError('Legacy sampling must bypass the optional curriculum')
    available = [kind for kind in KINDS if state['scopes'].get(scope_key(kind, backend, side), {}).get('open_until', 0) <= now]
    if rng.random() >= state['config']['recovery_probability']:
        return None, 'normal_probability'
    if not available:
        return None, 'all_recovery_breakers_open'
    return rng.choice(available), 'recovery_probability'


def record_prefix(state, *, lease_id, kind, backend, side, report, now, log_path=None):
    """Record genuine preparation evidence once; never infer an RL outcome."""
    key = scope_key(kind, backend, side)
    if canonical_kind(report.get('kind')) != canonical_kind(kind):
        raise ValueError('Prefix report does not match its training lease')
    if report.get('learning_samples', 0) != 0:
        raise ValueError('Preparation cannot contain actor learning samples')
    prepared = report.get('prepared')
    if not isinstance(prepared, bool):
        raise ValueError('Prefix report needs explicit prepared boolean')
    digest = hashlib.sha256(json.dumps(report, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    existing = next((x for x in state['attempts'] if x['lease_id'] == lease_id), None)
    if existing:
        if existing['report_sha256'] != digest:
            raise ValueError('Conflicting repeated prefix outcome')
        return copy.deepcopy(existing)
    row = state['scopes'].setdefault(key, dict(attempts=0, prepared=0, failed=0,
        consecutive_failures=0, open_until=0., last_reason=None))
    row['attempts'] += 1
    row['prepared' if prepared else 'failed'] += 1
    row['last_reason'] = report.get('reason')
    if prepared:
        row.update(consecutive_failures=0, open_until=0.)
    else:
        row['consecutive_failures'] += 1
        if row['consecutive_failures'] >= state['config']['failure_limit']:
            row['open_until'] = float(now)+state['config']['cooldown_seconds']
    event = dict(lease_id=lease_id, generation=state['config']['generation'], scope=key,
        prepared=prepared, reason=report.get('reason'), actor_samples=0, policy_failure=False,
        simulation_seconds=float(report.get('simulation_seconds', 0)),
        wall_seconds=float(report.get('wall_seconds', 0)), recorded_unix=float(now),
        report_sha256=digest, report_path=log_path, prefix_version=report.get('version'),
        open_until=row['open_until'])
    state['attempts'].append(event)
    return copy.deepcopy(event)


def add_actor_progress(state, kind, steps, seconds):
    if steps < 0 or seconds < 0:
        raise ValueError('Negative actor progress')
    category = canonical_kind(kind) if kind else 'normal'
    if category not in state['actor_steps']:
        raise ValueError('Invalid actor progress category')
    state['actor_steps'][category] += int(steps)
    state['actor_seconds'][category] += float(seconds)
