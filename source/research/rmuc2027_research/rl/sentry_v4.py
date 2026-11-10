"""Versioned legal sentry features; the original 393 features remain untouched.

The locomotion branch consumes only SKILL_INDICES. Radar estimates, referee
clocks, and privileged critic observations cannot enter that branch.
"""
import hashlib
import json
import math

try:
    from . import sentry_v3
except ImportError:
    pass  # Export bundles supply the embedded v3 module.
try:
    import sdk as _sdk
except ImportError:
    from rmuc2027_core import sdk as _sdk

PROFILE = 'sentry_referee_2027_v1'
ENCODING_VERSION = 2
ENCODING_V1_SCHEMA_HASH = '7de7fc26838ae930b873e25c6fec427338c40be2cd48dc857671d3e28db1dfa8'
ACTION_NAMES = sentry_v3.ACTION_NAMES
PHASES = ('route', 'approach', 'fold', 'cross', 'exit', 'runup', 'launch', 'airborne', 'settle', 'recover')
LOCAL_FIELDS = (
    ('roof.valid', 1), ('roof.along', 5), ('roof.across', 2),
    ('roof.half_length', 5), ('roof.half_width', 2), ('roof.standing_clearance', 1),
    ('roof.heading_sin', 1), ('roof.heading_cos', 1), ('roof.can_unfold', 1),
    ('motion.forward', 10), ('motion.lateral', 10), ('motion.vertical', 5), ('motion.yaw_rate', math.tau),
    ('navigation.passage_active', 1), ('navigation.fold_latched', 1),
    ('navigation.fold_dwell_remaining', 1), ('navigation.brake_active', 1), ('navigation.waiting_peer', 1),
) + tuple(('navigation.phase.' + phase, 1) for phase in PHASES)
SPEED_FIELDS = (('speed_lb', 10), ('robust_speed', 10), ('speed_valid', 1),
    ('speed_confidence', 1), ('n_valid', 20), ('age', 10),
    ('p_wheel', 1), ('p_fold', 1), ('p_omni', 1), ('class_unknown', 1))
TEAM_FIELDS = (('own_effect', 100), ('enemy_effect', 100), ('advantage', 1), ('confidence', 1),
    ('n_own', 30), ('n_enemy', 30), ('age', 10), ('valid', 1))
UNKNOWN_REASONS = ('missing_hp', 'missing_position', 'incomplete_candidates', 'multiple_candidates',
    'health_reset', 'status_ambiguous', 'flight_or_geometry', 'insufficient_support')
ROBOT_FIELDS = (
    ('pressure_rate', 100), ('pressure_confidence', 1), ('pressure_n', 30), ('pressure_age', 10), ('pressure_valid', 1),
    ('contribution_rate', 100), ('contribution_confidence', 1), ('contribution_n', 100),
    ('contribution_exposure', 100), ('contribution_age', 10), ('contribution_valid', 1),
    ('local_advantage', 1), ('local_confidence', 1), ('local_age', 10), ('local_valid', 1),
    ('candidate_count', 4), ('candidates_complete', 1), ('geometry_unique', 1),
) + tuple(('unknown.' + reason, 1) for reason in UNKNOWN_REASONS)
PREDICTION_FIELDS = (
    ('tunnel_probability', 1), ('tunnel_valid', 1), ('tunnel_confidence', 1),
    ('step_probability', 1), ('step_valid', 1), ('step_confidence', 1),
) + tuple((f'future.{h}.{key}', scale) for h in (1, 2, 3)
    for key, scale in (('x', 28), ('y', 15), ('valid', 1), ('confidence', 1))) + (
    ('route.straight', 1), ('route.detour', 1), ('route.tunnel', 1), ('route.step', 1),
    ('route.valid', 1), ('route.confidence', 1))

FEATURES = list(sentry_v3.FEATURES)
FEATURES += [(name, scale, 'public_geometry/local_execution') for name, scale in LOCAL_FIELDS]
SPEED_SOURCES = {'speed_lb': 'radar_0301/historical_observed_peak_not_current_bound',
    'robust_speed': 'radar_0301/current_smoothed_motion_proxy_not_true_max'}
FEATURES += [(f'estimate.enemy.{slot}.{name}', scale, SPEED_SOURCES.get(name, 'radar_0301'))
    for slot in (3, 4, 7) for name, scale in SPEED_FIELDS]
FEATURES += [('estimate.team.' + name, scale, 'radar_0301') for name, scale in TEAM_FIELDS]
FEATURES += [(f'estimate.{owner}.{slot}.{name}', scale, 'radar_0301') for owner in ('ally', 'enemy')
    for slot in (1, 3, 4, 7) for name, scale in ROBOT_FIELDS]
FEATURES += [(f'prediction.enemy.{slot}.{name}', scale, 'radar_0301') for slot in (3, 4, 7) for name, scale in PREDICTION_FIELDS]
AGE_VALIDITY = {f'estimate.enemy.{slot}.age': f'estimate.enemy.{slot}.speed_valid' for slot in (3,4,7)}
AGE_VALIDITY['estimate.team.age'] = 'estimate.team.valid'
AGE_VALIDITY.update({f'estimate.{owner}.{slot}.{kind}_age': f'estimate.{owner}.{slot}.{kind}_valid'
    for owner in ('ally','enemy') for slot in (1,3,4,7) for kind in ('pressure','contribution','local')})
# This encoding change affects only appended legal inference fields. A missing
# estimate has age 0 and valid=False; a valid age is min(age/divisor, 1).
FEATURES = [(name,scale,'radar_0301/bounded_valid_age_v2' if name in AGE_VALIDITY else source)
    for name,scale,source in FEATURES]
FEATURE_INDEX = {name: i for i, (name, _, _) in enumerate(FEATURES)}
AGE_VALIDITY_INDICES = tuple((FEATURE_INDEX[age],FEATURE_INDEX[valid]) for age,valid in AGE_VALIDITY.items())
SKILL_NAMES = tuple(name for name, _ in LOCAL_FIELDS) + (
    'self.alive', 'self.powered', 'self.grounded', 'self.upright', 'self.tipped',
    'self.fold_progress', 'self.can_fold', 'self.can_jump', 'nav.jump_cooldown')
SKILL_INDICES = tuple(FEATURE_INDEX[name] for name in SKILL_NAMES)
SCHEMA_HASH = hashlib.sha256(json.dumps(dict(features=FEATURES, skill_names=SKILL_NAMES,
    inference_version=1, encoding_version=ENCODING_VERSION), separators=(',', ':')).encode()).hexdigest()
assert len(FEATURES) == len(FEATURE_INDEX) == 739
assert len(SKILL_INDICES) == 37


def schema():
    return dict(version=4, obs_size=len(FEATURES), action_size=6, profile=PROFILE,
        schema_hash=SCHEMA_HASH, skill_obs_size=37, inference_version=1, encoding_version=ENCODING_VERSION)


def normalize_inference_ages(values):
    """Apply the v2 valid mask and bounded age ratio to an encoded feature row."""
    for age,valid in AGE_VALIDITY_INDICES:
        values[age] = min(1.,max(0.,float(values[age]))) if values[valid] > .5 else 0.
    return values


def migrate_encoding_v1(features, source_schema_hash=ENCODING_V1_SCHEMA_HASH):
    """Deterministically migrate an old 739-row; labels and skill37 stay exact.

    Call once per saved NPZ observation, preserving every target. This operates
    on previously encoded values, so no observation or hidden state is needed.
    Old checkpoints still require explicit refitting/validation on v2 inputs.
    """
    if source_schema_hash != ENCODING_V1_SCHEMA_HASH:
        raise ValueError('Unrecognized source sentry feature encoding')
    if len(features) != len(FEATURES) or not all(math.isfinite(float(v)) for v in features):
        raise ValueError('Expected a finite v1 sentry feature row')
    return normalize_inference_ages(list(features))


def _get(record, dotted, default=0.):
    for part in dotted.split('.'):
        if not isinstance(record, dict): return default
        record = record.get(part, record.get(int(part), default) if part.isdigit() else default)
    return record if isinstance(record, (int, float, bool)) else default


def local_feedback(obs, state):
    robot = next(r for r in obs['robots'] if r['slot'] == 7)
    nav = state.get('navigation', {})
    ctx = _sdk.locomotion_context(robot, obs, nav)
    d = _sdk._design(robot)
    roof = ctx['roof']
    passage = nav.get('passage')
    if passage:
        wall = next((w for w in obs['arena'].get('obstacles', []) if w.get('id') == passage['id']), None)
        if wall:
            axis = passage['axis']; center = passage['center']
            dx, dy = robot['x'] - center[0], robot['y'] - center[1]
            roof = dict(wall=wall, axis=axis, along=dx*axis[0]+dy*axis[1], across=-dx*axis[1]+dy*axis[0])
    a = robot['heading']; vx, vy = robot.get('vx', 0.), robot.get('vy', 0.)
    duration = max(.001, d.get('transform_time_s', .8))
    out = {'roof.valid': roof is not None, 'roof.can_unfold': ctx['can_unfold'],
        'motion.forward': vx*math.cos(a)+vy*math.sin(a),
        'motion.lateral': -vx*math.sin(a)+vy*math.cos(a), 'motion.vertical': robot.get('vz', 0.),
        'motion.yaw_rate': robot.get('angular_velocity', [0, 0, 0])[2],
        'navigation.passage_active': passage is not None,
        'navigation.fold_latched': nav.get('fold_intent', robot.get('transform_progress', 0.) > .5),
        'navigation.fold_dwell_remaining': max(0., min(1., (nav.get('fold_since', -100.)+duration-obs['time'])/duration)),
        'navigation.brake_active': bool(state.get('executed_action', {}).get('brake')),
        'navigation.waiting_peer': bool(nav.get('waiting_peer_active', False))}
    if roof:
        wall = roof['wall']; heading = math.atan2(roof['axis'][1], roof['axis'][0])-a
        out.update({'roof.along': roof['along'], 'roof.across': roof['across'],
            'roof.half_length': wall['w']/2, 'roof.half_width': wall['h']/2,
            'roof.standing_clearance': wall.get('bottom', 0.)-(robot.get('z', 0.)+d.get('muzzle_z_m', .4)+.06),
            'roof.heading_sin': math.sin(heading), 'roof.heading_cos': math.cos(heading)})
    for phase in PHASES: out['navigation.phase.'+phase] = nav.get('phase', 'route') == phase
    return out


def encode(obs, state):
    values = sentry_v3.encode(obs, state)
    local = local_feedback(obs, state)
    values += [float(local.get(name, 0.))/scale for name, scale in LOCAL_FIELDS]
    estimates = obs.get('radar_inference', {})
    for slot in (3, 4, 7):
        record = estimates.get('enemies', {}).get(str(slot), {})
        values += [float(_get(record, name, name == 'class_unknown'))/scale for name, scale in SPEED_FIELDS]
    values += [float(_get(estimates.get('team', {}), name))/scale for name, scale in TEAM_FIELDS]
    for owner in ('ally', 'enemy'):
        for slot in (1, 3, 4, 7):
            record = estimates.get('robots', {}).get(owner, {}).get(str(slot), {})
            values += [float(_get(record, name, name.startswith('unknown.')))/scale for name, scale in ROBOT_FIELDS]
    for slot in (3, 4, 7):
        record = estimates.get('predictions', {}).get(str(slot), {})
        for name, scale in PREDICTION_FIELDS:
            value = _get(record, name)
            # Radar coordinates are world coordinates; rotate valid predictions into the actor frame.
            if obs['team'] == 'blue' and name.startswith('future.'):
                _, horizon, key = name.split('.')
                if _get(record, f'future.{horizon}.valid'):
                    if key == 'x': value = 28-value
                    elif key == 'y': value = 15-value
            values.append(float(value)/scale)
    if len(values) != len(FEATURES) or not all(math.isfinite(v) for v in values):
        raise ValueError('Non-finite or incompatible v4 actor observation')
    return normalize_inference_ages(values)


def execute(obs, state, navigator, config):
    result = sentry_v3.execute(obs, state, navigator, config)
    trace = state.get('execution_trace', {})
    nav = state.get('navigation', {})
    raw = state.get('sentry_intent', {}).get('fold', -1.)
    trace.update(execution_trace_version=2, raw_fold=float(raw),
        decoded_fold_request=1 if raw > .3 else -1 if raw < -.3 else 0,
        fold_latched=bool(nav.get('fold_intent', False)),
        effective_transform=bool(state.get('executed_action', {}).get('transform', False)),
        safety_override='ceiling' if 'ceiling' in nav.get('rejected', []) else None)
    return result


def portable_forward(features, actor):
    """Torch-free branched deterministic actor mean, exported without a critic."""
    if len(features) != len(FEATURES) or actor.get('schema_hash') != SCHEMA_HASH:
        raise ValueError('Incompatible v4 actor export')
    def mlp(x, layers):
        for i, layer in enumerate(layers):
            x = [sum(a*b for a, b in zip(row, x))+bias for row, bias in zip(layer['weight'], layer['bias'])]
            if i+1 < len(layers): x = [math.tanh(v) for v in x]
        return x
    tactical = mlp(features, actor['tactical'])
    skill = mlp([features[i] for i in SKILL_INDICES], actor['skill'])
    return [tactical[0], tactical[1], skill[0], skill[1], tactical[2], tactical[3]]
