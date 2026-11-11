"""Conservative, JSON-safe recovery memory from received legal sentry data only.

Evidence is dated at its source, never at packet receipt. Possibility flags are
not calibrated probabilities. This module deliberately accepts no Match/frame,
reward dictionary, private weak flag, or referee respawn clock.
"""
import copy
import hashlib
import json
import math

VERSION = 1
PROFILE = 'sentry_referee_2027_v1'
LIFE = ('dead', 'normal_weak', 'paid_weak', 'active')
RECOVERY = ('own_base', 'own_supply', 'red_outpost', 'blue_outpost')
FEATURES = [(f'life_possible.{name}', 1) for name in LIFE]
FEATURES += [('life_known', 1), ('history_valid', 1), ('evidence_age', 420), ('has_gap', 1)]
FEATURES += [(f'{name}.{key}', 1 if key == 'valid' else 420)
             for name in ('dead_elapsed', 'revive_elapsed') for key in ('lower', 'upper', 'valid')]
FEATURES += [(f'revive_type.{name}', 1) for name in ('normal', 'paid', 'unknown')]
FEATURES += [(f'{name}.{key}', 1 if key == 'valid' else 30)
             for name in ('paid_weak_remaining', 'protection_remaining') for key in ('lower', 'upper', 'valid')]
FEATURES += [(f'recovery.{name}.{key}', 1) for name in RECOVERY for key in ('possible', 'proven')]
FEATURES += [(f'last_{name}_evidence.{key}', 1 if key == 'valid' else 420)
             for name in ('normal', 'clear') for key in ('age', 'valid')]
FEATURES += [('contradiction', 1)]
SCHEMA_HASH = hashlib.sha256(json.dumps(FEATURES, separators=(',', ':')).encode()).hexdigest()
assert len(FEATURES) == 36


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def fresh_packet(obs, command):
    meta = obs.get('referee', {}).get('sources', {}).get(command, {})
    if not meta.get('valid'): return False
    age = meta.get('age')
    if not finite(age) or age < -1e-8: return False
    limit = {'0x0201': .25, '0x020D': 2.05, '0x0209': 2/3+.05, '0x0101': 2.05}.get(command, 2.05)
    source = meta.get('source_time')
    return age <= limit+1e-8 and (source is None or finite(source) and source <= obs['time']+1e-8)


def new_state(team, now=0.):
    return dict(version=VERSION, schema_hash=SCHEMA_HASH, team=team, last_now=now,
        history_valid=now <= 1e-8, has_gap=now > 1e-8, contradiction=False,
        seen={}, hp=None, hp_time=None, life=list(LIFE), evidence_time=None,
        death_interval=None, revive_interval=None, revive_types=[],
        requested_modes=[], last_tx=None, tx_count=0, last_normal=None, last_clear=None,
        rfid_zero_after={}, recovery={name: dict(possible=False, proven=False) for name in RECOVERY})


def state_dict(memory):
    result = copy.deepcopy(memory)
    json.dumps(result, allow_nan=False)
    return result


def from_state_dict(value):
    result = state_dict(value)
    if result.get('version') != VERSION or result.get('schema_hash') != SCHEMA_HASH:
        raise ValueError('Recovery memory schema mismatch')
    if result.get('team') not in ('red', 'blue') or not finite(result.get('last_now')):
        raise ValueError('Invalid recovery memory identity/time')
    if not set(result.get('life', ())) <= set(LIFE): raise ValueError('Invalid recovery life possibilities')
    return result


def record_tx(memory, time, word, transport_accepted=None):
    """Remember our attempted word; transport acceptance is not operation ACK."""
    if not finite(time) or type(word) is not int or not 0 <= word < 1 << 25:
        raise ValueError('Invalid local referee command record')
    old = memory.get('last_tx')
    if old and old['time'] == time and old['word'] == word: return
    memory['last_tx'] = dict(time=time, word=word, transport_accepted=transport_accepted)
    memory['tx_count'] += 1
    if memory.get('hp') == 0 and transport_accepted is not False:
        modes = set(memory['requested_modes'])
        if word & 1: modes.add('normal')
        if word & 2: modes.add('paid')
        memory['requested_modes'] = sorted(modes)


def _mark_normal(memory, source, *, clear=False):
    # A shot and fatal hit can share one timestamp; the shot cannot overturn
    # a same-frame HP=0 packet. Only strictly later evidence may do that.
    if memory['hp_time'] is not None and source <= memory['hp_time']+1e-8 and memory['hp'] == 0:
        return
    if memory['last_normal'] is None or source > memory['last_normal']:
        memory['last_normal'] = source
    if clear: memory['last_clear'] = source
    if (source >= (memory['evidence_time'] if memory['evidence_time'] is not None else -1.)
            and source >= memory['last_now']-.75):
        memory['life'] = ['active']
        memory['evidence_time'] = source


def observe(obs, memory=None):
    if obs.get('observation_profile') != PROFILE: raise ValueError('Recovery memory requires legal sentry observation')
    now = obs['time']; team = obs['team']
    if not finite(now) or now < 0 or team not in ('red', 'blue'): raise ValueError('Invalid recovery observation clock')
    if memory is None: memory = new_state(team, now)
    if memory['team'] != team: raise ValueError('Recovery state belongs to another seat')
    if now < memory['last_now']-1e-8:
        # A true episode reset must not carry the previous life's timers.
        memory.clear(); memory.update(new_state(team, now))
    memory['last_now'] = now
    me = next(r for r in obs['robots'] if r['slot'] == 7 and r.get('team', team) == team)
    sources = obs.get('referee', {}).get('sources', {})
    def new_packet(command):
        if not fresh_packet(obs, command): return None
        source = sources[command].get('source_time')
        if not finite(source): return None  # Age alone cannot order historical events.
        previous = memory['seen'].get(command)
        if previous is not None and source <= previous+1e-8: return None
        memory['seen'][command] = source
        return source
    source = new_packet('0x0201')
    if source is not None:
        hp = me.get('hp')
        if not finite(hp) or hp < 0: raise ValueError('Invalid legal HP')
        old_hp, old_time = memory['hp'], memory['hp_time']
        if old_time is not None and source-old_time > .25+1e-8:
            memory['history_valid'] = False; memory['has_gap'] = True
        if hp == 0:
            if int(me.get('power_outputs', 0)) & 4: memory['contradiction'] = True
            if old_hp != 0:
                memory.update(death_interval=[old_time if old_time is not None else source, source],
                    revive_interval=None, revive_types=[], requested_modes=[], last_normal=None, last_clear=None)
                memory['recovery'] = {name: dict(possible=False, proven=False) for name in RECOVERY}
                memory['rfid_zero_after'] = {}
            memory['life'] = ['dead']; memory['evidence_time'] = source
        elif old_hp == 0:
            modes = list(memory['requested_modes']) if memory['history_valid'] else []
            memory['revive_interval'] = [old_time, source]
            memory['revive_types'] = modes
            memory['life'] = (['normal_weak'] if modes == ['normal'] else
                              ['paid_weak'] if modes == ['paid'] else ['normal_weak', 'paid_weak']) + ['active']
            memory['evidence_time'] = source
        elif old_hp is None:
            memory['life'] = ['active'] if source <= 1e-8 else ['normal_weak', 'paid_weak', 'active']
            memory['evidence_time'] = source
        memory['hp'], memory['hp_time'] = hp, source
        # Official chain: weakness locks shooter; locked shooter has no supply.
        # The reverse implication is deliberately forbidden (ammo/heat/etc.).
        if hp > 0 and int(me.get('power_outputs', 0)) & 4: _mark_normal(memory, source)
    if not fresh_packet(obs, '0x0201'):
        memory['has_gap'] = True; memory['history_valid'] = False
        memory['life'] = list(LIFE)
    source = new_packet('0x0209')
    if source is not None:
        cards = me.get('rfid', [])
        card = lambda index: index < len(cards) and bool(cards[index])
        detected = dict(own_base=card(2 if team == 'red' else 3), own_supply=card(0),
                        red_outpost=card(8), blue_outpost=card(9))
        control = obs.get('referee', {}).get('control', {})
        control_valid = fresh_packet(obs, '0x0101')
        revive = memory['revive_interval']
        post_revive = revive is not None and source > revive[1]+1e-8 and memory['hp'] is not None and memory['hp'] > 0
        natural = memory['revive_types'] == ['normal']
        paid_over = memory['revive_types'] == ['paid'] and revive is not None and source >= revive[1]+3.05-1e-8
        for name, value in detected.items():
            possible = value and (not name.endswith('outpost') or not control_valid or control.get(name) == 0)
            zero = memory.setdefault('rfid_zero_after', {}).get(name)
            # A positive RFID bit can retain a pre-death contact for 2 s.
            # Establish a new post-revive detection, not merely a new packet.
            new_contact = post_revive and (source > revive[1]+2+1e-8
                or zero is not None and zero > revive[1] and source > zero)
            proven = new_contact and value and name in ('own_base', 'own_supply') and (natural or paid_over)
            memory['recovery'][name] = dict(possible=possible, proven=proven)
            if proven: _mark_normal(memory, source, clear=True)
            if post_revive and not value: memory['rfid_zero_after'][name] = source
        # A zero detection history only excludes recent contact within the
        # referee's 2 s release window; it cannot establish an arbitrary past.
        if (post_revive and natural and not any(detected.values()) and source-2 <= revive[0]
                and memory['last_normal'] is None):
            memory['life'] = ['normal_weak']; memory['evidence_time'] = source
    for event in obs.get('referee', {}).get('events', []):
        source = event.get('source_time')
        if event.get('command') != 0x207 or not finite(source) or source > now+1e-8: continue
        key = 'shot'
        if source > memory['seen'].get(key, -1.):
            memory['seen'][key] = source
            _mark_normal(memory, source)
    revive = memory['revive_interval']
    if (revive is not None and memory['revive_types'] == ['paid'] and fresh_packet(obs, '0x0201')
            and memory['hp'] > 0 and now >= revive[1]+3.05-1e-8):
        _mark_normal(memory, now, clear=True)
    return memory


def features(memory, now):
    if not finite(now): raise ValueError('Invalid recovery feature clock')
    out = {f'life_possible.{name}': name in memory['life'] for name in LIFE}
    out.update(life_known=len(memory['life']) == 1, history_valid=memory['history_valid'],
        evidence_age=max(0., now-memory['evidence_time']) if memory['evidence_time'] is not None else 0.,
        has_gap=memory['has_gap'], contradiction=memory['contradiction'])
    for name, interval in (('dead_elapsed', memory['death_interval']), ('revive_elapsed', memory['revive_interval'])):
        out[name+'.valid'] = interval is not None
        if interval is not None:
            out[name+'.lower'], out[name+'.upper'] = max(0., now-interval[1]), max(0., now-interval[0])
    modes = memory['revive_types']
    for name in ('normal', 'paid'): out['revive_type.'+name] = name in modes
    out['revive_type.unknown'] = len(modes) != 1
    revive = memory['revive_interval']
    if revive is not None and modes == ['paid']:
        out.update({'paid_weak_remaining.lower': max(0., revive[0]+3-now),
                    'paid_weak_remaining.upper': max(0., revive[1]+3.05-now), 'paid_weak_remaining.valid': True})
    if revive is not None:
        if modes == ['paid']: lower, upper = revive[0]+3, revive[1]+3.05
        elif modes == ['normal']:
            lower, upper = revive[0]+10, revive[1]+30
            if memory['last_clear'] is not None:
                upper = min(upper, max(revive[1]+10, memory['last_clear']))
        else: lower, upper = revive[0]+3, revive[1]+30.05
        out.update({'protection_remaining.lower': max(0., lower-now),
                    'protection_remaining.upper': max(0., upper-now), 'protection_remaining.valid': True})
    for name, values in memory['recovery'].items():
        for key, value in values.items(): out[f'recovery.{name}.{key}'] = value
    for name in ('normal', 'clear'):
        value = memory['last_'+name]
        out[f'last_{name}_evidence.valid'] = value is not None
        out[f'last_{name}_evidence.age'] = max(0., now-value) if value is not None else 0.
    return [min(1., max(0., float(out.get(name, 0))/scale)) for name, scale in FEATURES]
