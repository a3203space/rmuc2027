"""Legal recovery memory plus six unchanged locomotion/tactical outputs and a
three-way revive choice. The portable functions use only the standard library.
"""
import hashlib
import json
import math
import numbers

try:
    from . import recovery_belief, sentry_v3, sentry_v4
except ImportError:
    pass  # Exporter injects these exact public modules into its namespace.

PROFILE = sentry_v4.PROFILE
ACTION_NAMES = (*sentry_v4.ACTION_NAMES, 'revive')
REVIVE_CATEGORIES = ('wait', 'free', 'paid')
FEATURES = list(sentry_v4.FEATURES) + [('recovery.'+name, scale, 'legal_received_history_v1')
                                    for name, scale in recovery_belief.FEATURES]
FEATURE_INDEX = {name: index for index, (name, _, _) in enumerate(FEATURES)}
SKILL_NAMES, SKILL_INDICES = sentry_v4.SKILL_NAMES, sentry_v4.SKILL_INDICES
SCHEMA_HASH = hashlib.sha256(json.dumps(dict(features=FEATURES, actions=ACTION_NAMES,
    revive_categories=REVIVE_CATEGORIES, belief_schema=recovery_belief.SCHEMA_HASH,
    skill_names=SKILL_NAMES), separators=(',', ':')).encode()).hexdigest()
assert len(FEATURES) == 775 and len(SKILL_INDICES) == 37


def schema():
    return dict(version=6, obs_size=775, action_size=7, profile=PROFILE,
        schema_hash=SCHEMA_HASH, critic_size=454, skill_obs_size=37,
        belief_version=1, revive_mask_size=3, revive_categories=list(REVIVE_CATEGORIES))


def validate_revive_action(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError('Revive action must be an integer category, not a rounded Gaussian')
    value = float(value)
    if not math.isfinite(value) or not value.is_integer() or not 0 <= value <= 2:
        raise ValueError('Revive action must be wait=0, free=1 or paid=2')
    return int(value)


def validate_action(action):
    if len(action) != 7: raise ValueError('Expected six continuous actions and one revive category')
    result = []
    for value in action[:6]:
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
            raise ValueError('Continuous sentry actions must be finite numbers')
        result.append(min(1., max(-1., float(value))))
    return result+[validate_revive_action(action[6])]


def revive_mask(obs):
    if obs.get('observation_profile') != PROFILE: raise ValueError('Strict sentry observation required')
    robot = next(r for r in obs['robots'] if r['slot'] == 7)
    dead = recovery_belief.fresh_packet(obs, '0x0201') and robot.get('hp') == 0
    paid = dead and recovery_belief.fresh_packet(obs, '0x020D') and bool(obs['referee'].get('ack', {}).get('paid_revive'))
    # Free is a confirmation intent: it remains legal before progress is ready.
    return [True, bool(dead), bool(paid)]


def v6_feature_masks(features):
    if len(features) != 775 or not all(math.isfinite(float(x)) for x in features):
        raise ValueError('Expected finite v6 actor features')
    def field(name): return float(features[FEATURE_INDEX[name]])
    def fresh(command, limit):
        return field('packet.'+command+'.valid') > .5 and -1e-8 <= field('packet.'+command+'.age')*5 <= limit+1e-8
    dead = fresh('0201', .25) and field('self.hp') == 0
    paid = dead and fresh('020D', 2.05) and field('ack.paid_revive') > .5
    return [True, bool(dead), bool(paid)]


def observe(obs, state):
    state['recovery_belief'] = recovery_belief.observe(obs, state.get('recovery_belief'))
    return state['recovery_belief']


def record_tx(state, time, word, transport_accepted=None):
    if 'recovery_belief' not in state: raise ValueError('Observe legal data before recording a transmission')
    recovery_belief.record_tx(state['recovery_belief'], time, word, transport_accepted)


def encode(obs, state):
    memory = observe(obs, state)
    result = sentry_v4.encode(obs, state)+recovery_belief.features(memory, obs['time'])
    if v6_feature_masks(result) != revive_mask(obs):
        raise ValueError('Legal revive mask and encoded packet freshness disagree')
    return result


def resource_command_v6(obs, state, intent, moving=False):
    """Keep existing resource counter semantics; only RL owns the revive bits.

    Even without fresh ACKs, a wait word clears the previous confirmation. It
    uses our cached counters and creates no new purchase while ACK is absent.
    """
    choice = validate_revive_action(intent)
    memory = observe(obs, state)
    robot = next(r for r in obs['robots'] if r['slot'] == 7)
    mask = revive_mask(obs)
    effective = choice if mask[choice] else 0
    resource_obs = obs
    if not recovery_belief.fresh_packet(obs, '0x020D'):
        resource_obs = dict(obs, referee=dict(obs['referee'], ack={}))
    word = sentry_v3.resource_command(robot, resource_obs, state, moving)
    if word is None:
        counters = state.get('referee_commands', {})
        word = (int(counters.get('local_ammo', 0)) << 2 | int(counters.get('remote_ammo', 0)) << 13
                | int(counters.get('remote_heal', 0)) << 17 | (3 if moving else 1) << 21)
    word = (word & ~3) | effective
    # One local record per executor tick. Repeating a word never increments a
    # purchase counter; the existing 0x0301 limiter owns the actual 20 Hz wire.
    record_tx(state, obs['time'], word)
    state['revive_execution'] = dict(selected=choice, effective=effective,
        mask=list(mask), reason=None if effective == choice else 'legal_packet_unavailable',
        word=word, time=obs['time'], operation_acknowledged=False)
    return word


def execute(obs, state, navigator, config):
    observe(obs, state)
    choice = validate_revive_action(state.get('revive_intent', 0))
    # A dead-row Gaussian is a Dirac noop. If resurrection occurs during the
    # first 20 Hz substep, its zero coordinates must not turn into a centre-map
    # goal on the second substep before the next 10 Hz actor decision.
    hold = bool(state.get('action_selected_dead', False))
    missing = object()
    goal, intent = state.get('goal', missing), state.get('sentry_intent', missing)
    if hold:
        state['goal'] = None
        state['sentry_intent'] = dict(fold=-1., jump=-1., spin=-1., fire=-1.)
    try:
        result = sentry_v3.execute(obs, state, navigator, config,
            resource_builder=lambda robot, observation, memory, moving:
                resource_command_v6(observation, memory, choice, moving))
    finally:
        if hold:
            for key, value in (('goal', goal), ('sentry_intent', intent)):
                if value is missing: state.pop(key, None)
                else: state[key] = value
    trace = state.get('execution_trace', {})
    nav = state.get('navigation', {})
    raw = state.get('sentry_intent', {}).get('fold', -1.)
    trace.update(execution_trace_version=3, raw_fold=float(raw),
        decoded_fold_request=1 if raw > .3 else -1 if raw < -.3 else 0,
        fold_latched=bool(nav.get('fold_intent', False)),
        effective_transform=bool(state.get('executed_action', {}).get('transform', False)),
        safety_override='ceiling' if 'ceiling' in nav.get('rejected', []) else None,
        revive=state['revive_execution'], continuous_waiting_for_actor=hold)
    return result


def portable_forward(features, actor):
    if actor.get('schema_hash') != SCHEMA_HASH: raise ValueError('Incompatible v6 actor export')
    mask = v6_feature_masks(features)
    def mlp(values, layers):
        for index, layer in enumerate(layers):
            values = [sum(a*b for a, b in zip(row, values))+bias
                      for row, bias in zip(layer['weight'], layer['bias'])]
            if index+1 < len(layers): values = [math.tanh(v) for v in values]
        return values
    tactical = mlp(features, actor['tactical'])
    skill = mlp([features[i] for i in SKILL_INDICES], actor['skill'])
    logits = mlp(features, actor['revive'])
    choice = max((index for index in range(3) if mask[index]), key=lambda index: logits[index])
    continuous = [tactical[0], tactical[1], skill[0], skill[1], tactical[2], tactical[3]]
    if mask[1]: continuous = [0.]*6
    return continuous+[choice]
