"""Versioned, per-seat privileged auxiliary policy contract.

Only infantry/heavy may use this profile.  Decisions include the whole match,
including the heavy opening; navigation, ballistics and the engine's arm IK are
executors, never tactical substitutes.  This module deliberately imports no ML
runtime so sampling workers and exported controllers can use the same contract.
"""
import copy
import hashlib
import json
import math

from rmuc2027_core import sdk

PROFILE = 'privileged_aux_2027_v1'
VERSION = 5
SLOTS = (1, 3, 4, 6, 7)
CONTINUOUS = ('move_x', 'move_y', 'goal_x', 'goal_y', 'aim_yaw', 'aim_pitch',
              'heading', 'yaw_rate', 'power', 'leg_height', 'lean', 'jump_strength',
              'assembly_rotate', 'assembly_rotate2') + tuple(
    f'{arm}_{key}' for arm in ('arm', 'arm2')
    for key in ('x', 'y', 'z', 'dir_yaw', 'dir_pitch', 'roll', 'grip',
                'joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6'))
CATEGORIES = {
    'motion': ('direct', 'goal'), 'turn': ('hold', 'heading', 'yaw_rate'),
    'brake': ('coast', 'brake'), 'power_mode': ('default', 'explicit'),
    'transform': ('hold', 'unfold', 'fold'), 'jump': ('off', 'on'),
    'leg_mode': ('default', 'height'),
    'target': ('hold', 'manual', 'enemy1', 'enemy3', 'enemy4', 'enemy6', 'enemy7', 'base', 'outpost',
               'drone1', 'drone2', 'drone3', 'drone4'),
    'fire': ('off', 'on'), 'ammo': ('none', 'local', 'remote'),
    'ammo_batches': (1, 2, 5, 10), 'remote_heal': ('off', 'on'),
    'revive': ('off', 'on'), 'chassis_profile': ('keep', 'hp', 'power'),
    'weapon_profile': ('keep', 'burst', 'cooling'), 'rune_start': ('off', 'on'),
    'pickup': ('none', 'arm', 'arm2'), 'rescue': ('none', 1, 3, 4, 6, 7),
    'assembly': (0, 1, 2, 3, 4), 'assembly_core': (0, 1, 2),
    'assembly_insert': ('off', 'on'), 'assembly_lift': ('off', 'on'),
    'assembly_flip': ('off', 'on'), 'assembly_confirm': ('off', 'on'),
    'install_launcher': ('off', 'on'),
    'arm_mode': ('automatic', 'stow', 'tip', 'joints'),
    'arm2_mode': ('automatic', 'stow', 'tip', 'joints'),
}
CAT_NAMES = tuple(CATEGORIES)
ACTION_NAMES = CONTINUOUS + CAT_NAMES
HEAD_NAMES = ACTION_NAMES
ACTION_INDEX = {name: i for i, name in enumerate(ACTION_NAMES)}
N_CONT = len(CONTINUOUS)
MASK_SIZE = N_CONT + sum(map(len, CATEGORIES.values()))

# Fixed slots, masks and explicit categorical codes make missing/dead entities
# distinct. Raw privileged observations retain the complete snapshot as well.
ROBOT_NUMERIC = {
    'x': 28, 'y': 15, 'z': 3, 'vx': 10, 'vy': 10, 'vz': 10,
    'hp': 1000, 'max_hp': 1000, 'ammo': 500, 'heat': 500, 'max_heat': 500,
    'cooling_rate': 100, 'power_limit': 200, 'power_w': 200, 'motor_power_w': 200,
    'buffer_energy_j': 60, 'supercap_energy_j': 1800, 'battery_remaining_wh': 200,
    'muzzle_speed_mps': 30, 'mass_kg': 40, 'radius': 1, 'speed_reference_mps': 10,
    'level': 10, 'experience': 1000, 'damage': 5000, 'kills': 10,
    'alive': 1, 'powered': 1, 'grounded': 1, 'upright': 1, 'tipped': 1,
    'weak': 1, 'invulnerable': 1, 'overheated': 1, 'red_card': 1,
    'launcher_progress': 10, 'carried_units': 2, 'arm_count': 2,
    'transform_progress': 1, 'crouched': 1, 'cooldown': 2,
    'respawn_progress': 30, 'respawn_required': 30, 'last_damage_time': 420,
    'posture_ready_at': 420, 'pending_heal': 420, 'power_boost_until': 420,
    'rescue_contact': 1, 'self_righting': 1, 'model_slot': 3000,
    'planar_speed': 10, 'hp_fraction': 1, 'heat_fraction': 1,
}
ROBOT_ENUMS = {
    'role': ('heavy', 'infantry', 'aerial', 'sentry'),
    'chassis_type': ('mecanum', 'omni', 'swerve', 'wheel_legged', 'hangar'),
    'launcher': ('locked', 'unlocked', 'installing', 'fitted', 'installed'),
    'posture': ('attack', 'defense', 'move'),
    'arm_mode': ('stow', 'manual', 'pick', 'assembly', 'rescue'),
    'arm2_mode': ('stow', 'manual', 'pick', 'assembly', 'rescue'),
    'chassis_profile': ('hp', 'power'), 'weapon_profile': ('burst', 'cooling'),
}
ROBOT_ARRAYS = {'angular_velocity': (3, 12), 'quaternion': (4, 1), 'legs': (2, 1),
                'arm_joints': (7, math.pi), 'arm2_joints': (7, math.pi),
                'arm_tip': (3, 30), 'arm2_tip': (3, 30), 'wheel_loads_n': (4, 400)}
TEAM_FIELDS = ('resources', 'control', 'purchased_17', 'purchased_42', 'energy_units_left',
               'damage', 'kills', 'total_coins', 'score')
ASSEMBLY_FIELDS = ('difficulty', 'started', 'ready_at', 'retry_at', 'permanent_lock',
                   'inserted', 'lifted', 'flipped', 'rotation', 'target_rotation',
                   'confirmed', 'confirm_progress_s', 'inserted_units', 'attempt',
                   'buffer_until', 'repeat_ready_at', 'interrupted')
ASSEMBLY_STAGES = ('idle', 'moving', 'inserting', 'complete', 'failed')
CAPABILITY_FIELDS = ('speed_reference_multiplier', 'force_multiplier', 'efficiency_multiplier',
                     'speed_reference_mps', 'motor_force_max_n', 'drive_efficiency',
                     'aim_delay_s', 'aim_sigma_deg', 'aim_tau_s', 'mobility_tier', 'aim_tier')
FEATURES = [('time', 420), ('remaining', 420), ('self_slot', 7)]
for owner in ('ally', 'enemy'):
    for slot in SLOTS:
        prefix = f'{owner}.{slot}.'
        FEATURES += [(prefix+'present', 1)] + [(prefix+k, d) for k, d in ROBOT_NUMERIC.items()]
        FEATURES += [(prefix+a+'.'+trig, 1) for a in ('heading', 'aim', 'pitch') for trig in ('sin', 'cos')]
        FEATURES += [(prefix+k+'.'+str(v), 1) for k, values in ROBOT_ENUMS.items() for v in values]
        FEATURES += [(prefix+k+'.'+str(i), scale) for k, (size, scale) in ROBOT_ARRAYS.items() for i in range(size)]
        FEATURES += [(prefix+'buff.'+k, 1) for k in ('heal', 'cooling', 'defense', 'vulnerability', 'attack')]
        FEATURES += [(prefix+'cap.'+k, 1) for k in CAPABILITY_FIELDS]
    FEATURES += [(owner+'.team.'+k, 5000) for k in TEAM_FIELDS]
    FEATURES += [(owner+'.completed.'+str(i), 1) for i in range(1, 5)]
    for kind in ('base', 'outpost'):
        FEATURES += [(owner+'.'+kind+'.'+k, scale) for k, scale in (('x', 28), ('y', 15), ('hp', 7500), ('max_hp', 7500))]
        FEATURES += [(owner+'.'+kind+'.controller', 1)]
    for i in range(4):
        FEATURES += [(f'{owner}.drone.{i}.{k}', scale) for k, scale in
                     (('present', 1), ('x', 28), ('y', 15), ('z', 3), ('vx', 10), ('vy', 10),
                      ('vz', 10), ('flying', 1), ('integrity', 10), ('battery_wh', 10))]
    FEATURES += [(owner+'.assembly.'+k, 420 if k.endswith(('_at', '_until')) else 100) for k in ASSEMBLY_FIELDS]
    FEATURES += [(owner+'.assembly.stage.'+k, 1) for k in ASSEMBLY_STAGES]
    FEATURES += [(owner+'.assembly.'+k+'.'+str(i), 30) for k in ('pose', 'current_pose') for i in range(6)]
    FEATURES += [(owner+'.assembly.second.'+k, 100) for k in ('inserted', 'lifted', 'flipped', 'rotation', 'target_rotation', 'confirm', 'confirmed')]
    FEATURES += [(owner+'.rune.'+k, 420) for k in ('active_until', 'started', 'ready_at', 'angle', 'phase', 'hits', 'activations')]
# Nearby projectiles matter to dodge decisions. Deterministic closest-first slots
# include presence and velocity; raw observation still includes every projectile.
for i in range(32):
    FEATURES += [(f'projectile.{i}.{k}', scale) for k, scale in
                 (('present', 1), ('enemy', 1), ('x', 28), ('y', 15), ('z', 3),
                  ('vx', 30), ('vy', 30), ('vz', 30), ('damage', 100), ('remaining', 40))]
for i in range(32):
    FEATURES += [(f'zone.{i}.{k}', scale) for k, scale in
                 (('present', 1), ('x', 28), ('y', 15), ('radius', 5), ('owner', 1), ('occupants', 10))]
VALUE_FEATURES = tuple(FEATURES)
FEATURES += [('valid.'+name, 1) for name, _ in VALUE_FEATURES]
SCHEMA_HASH = hashlib.sha256(json.dumps([FEATURES, CONTINUOUS, CATEGORIES], separators=(',', ':')).encode()).hexdigest()


def schema():
    return dict(version=VERSION, profile=PROFILE, obs_size=len(FEATURES),
                action_size=len(ACTION_NAMES), mask_size=MASK_SIZE,
                head_size=len(HEAD_NAMES), schema_hash=SCHEMA_HASH)


def own_robot(obs):
    slot = int(obs['controlled_ids'][0])
    return next(r for r in obs['robots'] if r['team'] == obs['team'] and r['slot'] == slot)


def build_observation(snapshot, team, slot, *, capabilities=None):
    if team not in ('red', 'blue') or int(slot) == 7:
        raise ValueError('Privileged auxiliary profile excludes both sentries')
    robots = snapshot['robots']
    robot = next((r for r in robots if r['team'] == team and r['slot'] == int(slot)), None)
    if robot is None or robot['role'] not in ('infantry', 'heavy'):
        raise ValueError('Privileged auxiliary profile requires one infantry/heavy seat')
    obs = copy.deepcopy(snapshot)
    obs.update(api_version=3, observation_profile=PROFILE, team=team,
               controlled_ids=[str(slot)], robots=[copy.deepcopy(r) for r in robots if r['team'] == team],
               enemies=[copy.deepcopy(r) for r in robots if r['team'] != team],
               training_capabilities=copy.deepcopy(capabilities or {}),
               message_inboxes={str(slot): []}, messages=[], vision={})
    obs.setdefault('remaining', max(0, obs.get('duration', 420)-obs.get('time', 0)))
    obs['enemy_drones'] = [copy.deepcopy(d) for r in robots if r['team'] != team for d in r.get('drones', [])]
    # Each recipient gets an independent object. No privileged object enters the
    # Match observation cache or a teammate's message/radar channel.
    return obs


def _num(value):
    return float(value) if isinstance(value, (int, float, bool)) and math.isfinite(value) else 0.


def _capabilities(obs, robot):
    caps = obs.get('training_capabilities', {})
    seats = caps.get('seats', caps)
    value = seats.get(robot['id'], seats.get(f"{robot['team']}:{robot['slot']}", {}))
    if 'mobility' in value or 'aim' in value:
        mobility, aim = value.get('mobility', {}), value.get('aim', {})
        value = dict(value, speed_reference_multiplier=mobility.get('speed_reference_mps', 1),
                     force_multiplier=mobility.get('motor_force_max_n', 1),
                     efficiency_multiplier=mobility.get('drive_efficiency', 1),
                     aim_delay_s=aim.get('delay_s', 0), aim_sigma_deg=aim.get('sigma_deg', 0),
                     aim_tau_s=aim.get('tau_s', .15))
    return value


def encode(obs):
    if obs.get('observation_profile') != PROFILE:
        raise ValueError('Auxiliary encoder requires explicit privileged profile')
    me = own_robot(obs)
    if me['role'] not in ('infantry', 'heavy') or me['slot'] == 7:
        raise ValueError('Sentries cannot use auxiliary encoder')
    team = obs['team']; enemy = 'blue' if team == 'red' else 'red'
    flip = team == 'blue'
    def xy(x, y): return (28-x, 15-y) if flip else (x, y)
    f = dict(time=obs.get('time', 0), remaining=obs.get('remaining', 0), self_slot=me['slot'])
    for owner, side, rows in (('ally', team, obs['robots']), ('enemy', enemy, obs['enemies'])):
        for r in rows:
            p = f'{owner}.{r["slot"]}.'; f[p+'present'] = 1
            for k in ROBOT_NUMERIC:
                if k in r: f[p+k] = _num(r[k])
            f[p+'planar_speed'] = math.hypot(r.get('vx', 0), r.get('vy', 0))
            f[p+'hp_fraction'] = r.get('hp', 0)/max(1, r.get('max_hp', 1))
            f[p+'heat_fraction'] = r.get('heat', 0)/max(1, r.get('max_heat', 1))
            f[p+'x'], f[p+'y'] = xy(r.get('x', 0), r.get('y', 0))
            if flip:
                for k in ('vx', 'vy'):
                    if p+k in f: f[p+k] *= -1
            for k in ('heading', 'aim', 'pitch'):
                if k not in r: continue
                a = _num(r.get(k)) + (math.pi if flip and k != 'pitch' else 0)
                f[p+k+'.sin'], f[p+k+'.cos'] = math.sin(a), math.cos(a)
            for k, values in ROBOT_ENUMS.items():
                if k not in r: continue
                for v in values: f[p+k+'.'+str(v)] = r.get(k) == v
            for k, (size, _) in ROBOT_ARRAYS.items():
                values = list(r.get(k) or [])
                if flip and k in ('arm_tip', 'arm2_tip') and len(values) >= 2:
                    values[:2] = xy(*values[:2])
                for i, v in enumerate(values[:size]): f[p+k+'.'+str(i)] = _num(v)
            for k, v in r.get('buffs', {}).items(): f[p+'buff.'+k] = _num(v)
            cap = _capabilities(obs, r)
            for k in CAPABILITY_FIELDS:
                if k in cap: f[p+'cap.'+k] = _num(cap[k])
        ts = obs.get('teams', {}).get(side, {})
        for k in TEAM_FIELDS:
            if k in ts: f[owner+'.team.'+k] = _num(ts[k])
        for k in ts.get('completed_difficulties', []): f[owner+'.completed.'+str(k)] = 1
        for kind, source in (('base', 'bases'), ('outpost', 'outposts')):
            value = obs.get(source, {}).get(side, {})
            for k in ('x', 'y', 'hp', 'max_hp'): f[owner+'.'+kind+'.'+k] = _num(value.get(k))
            f[owner+'.'+kind+'.x'], f[owner+'.'+kind+'.y'] = xy(value.get('x', 0), value.get('y', 0))
            f[owner+'.'+kind+'.controller'] = 1 if value.get('owner', side) == team else -1
        drones = [drone for r in rows for drone in r.get('drones', [])]
        for i, drone in enumerate(drones[:4]):
            p = f'{owner}.drone.{i}.'; f[p+'present'] = 1; f[p+'flying'] = drone.get('state') == 'flying'
            for k in ('x', 'y', 'z', 'vx', 'vy', 'vz', 'integrity', 'battery_wh'): f[p+k] = _num(drone.get(k))
            f[p+'x'], f[p+'y'] = xy(drone.get('x', 0), drone.get('y', 0))
            if flip:
                f[p+'vx'] *= -1; f[p+'vy'] *= -1
        a = obs.get('assemblies', {}).get(side, {})
        for k in ASSEMBLY_FIELDS: f[owner+'.assembly.'+k] = _num(a.get(k))
        for k in ASSEMBLY_STAGES: f[owner+'.assembly.stage.'+k] = a.get('stage') == k
        for k in ('pose', 'current_pose'):
            for i, v in enumerate((a.get(k) or [])[:6]): f[owner+'.assembly.'+k+'.'+str(i)] = _num(v)
        for k, v in (a.get('second') or {}).items(): f[owner+'.assembly.second.'+k] = _num(v)
        for k, v in obs.get('runes', {}).get(side, {}).items(): f[owner+'.rune.'+k] = _num(v)
    bullets = sorted(obs.get('projectiles', []), key=lambda p: ((p['x']-me['x'])**2+(p['y']-me['y'])**2, str(p.get('id', ''))))
    for i, p in enumerate(bullets[:32]):
        prefix = f'projectile.{i}.'; f[prefix+'present'] = 1; f[prefix+'enemy'] = p.get('team') != team
        for k, v in p.items(): f[prefix+k] = _num(v)
        f[prefix+'x'], f[prefix+'y'] = xy(p['x'], p['y'])
        if flip:
            for k in ('vx', 'vy'): f[prefix+k] *= -1
    for i, z in enumerate(sorted(obs.get('arena', {}).get('zones', []), key=lambda z: z['id'])[:32]):
        p = f'zone.{i}.'; f[p+'present'] = 1
        f[p+'x'], f[p+'y'] = xy(z['x'], z['y'])
        f[p+'radius'] = z.get('radius', 0); f[p+'owner'] = 1 if z.get('owner') == team else -1 if z.get('owner') == enemy else 0
        f[p+'occupants'] = len(obs.get('zones', {}).get(z['id'], {}).get('occupants', []))
    for name, _ in VALUE_FEATURES: f['valid.'+name] = name in f
    return {'actor': [_num(f.get(k))/scale for k, scale in FEATURES], 'action_mask': action_mask(obs)}


def action_mask(obs):
    r = own_robot(obs); d = sdk._design(r); alive = bool(r.get('alive'))
    active = alive and not r.get('red_card', False)
    heavy = r['role'] == 'heavy'; infantry = r['role'] == 'infantry'
    legged = d.get('chassis_type') == 'wheel_legged'
    arms = int(r.get('arm_count', d.get('arm_count', 0)))
    continuous = {k: float(active) for k in CONTINUOUS}
    for k in ('leg_height', 'lean', 'jump_strength'): continuous[k] *= legged
    for arm, count in (('arm', 1), ('arm2', 2)):
        for k in CONTINUOUS:
            if k.startswith(arm+'_'): continuous[k] *= heavy and arms >= count
    for k in ('assembly_rotate', 'assembly_rotate2'): continuous[k] *= heavy
    masks = {name: [True]*len(options) for name, options in CATEGORIES.items()}
    def enable(name, allowed):
        masks[name] = [i in allowed for i in range(len(CATEGORIES[name]))]
        if not any(masks[name]): masks[name][0] = True
    if not active:
        for name in CAT_NAMES: enable(name, {0})
    enable('revive', {0, 1} if not alive and not r.get('red_card') else {0})
    if active:
        enable('transform', {0, 1, 2} if d.get('transform_allowed') else {0})
        enable('jump', {0, 1} if legged and d.get('capabilities', {}).get('jump') and r.get('grounded') else {0})
        enable('leg_mode', {0, 1} if legged else {0})
        enable('chassis_profile', {0, 1, 2} if infantry and obs.get('tick', 0) == 0 else {0})
        enable('weapon_profile', {0, 1, 2} if infantry and obs.get('tick', 0) == 0 else {0})
        enable('rune_start', {0, 1} if infantry and not r.get('weak') else {0})
        enable('fire', {0, 1} if r.get('ammo', 0) > 0 and r.get('powered', True) and not r.get('overheated') and (not heavy or r.get('launcher') in ('fitted', 'installed')) else {0})
        enable('pickup', set(range(min(arms, 2)+1)) if heavy and not r.get('weak') and r.get('carried_units', 0) < 2 else {0})
        candidates = {0}
        if heavy and arms and not r.get('weak'):
            candidates.update(i for i, slot in enumerate(CATEGORIES['rescue']) if i and any(p['slot'] == slot and p['id'] != r['id'] and p.get('tipped') and p['role'] != 'aerial' for p in obs['robots']))
        enable('rescue', candidates)
        a = obs.get('assemblies', {}).get(obs['team'], {})
        completed = obs.get('teams', {}).get(obs['team'], {}).get('completed_difficulties', [])
        allowed = {0}
        if heavy and not r.get('weak') and a.get('stage', 'idle') in ('idle', 'complete', 'failed'):
            unlocks = sdk.rules(obs).get('assembly', {}).get('unlock_times_s', [0, 0, 0, 0])
            allowed.update(k for k in range(1, 5) if (k == 1 or k-1 in completed) and (k != 4 or 4 not in completed) and r.get('carried_units', 0) >= (2 if k == 4 else 1) and obs.get('time', 0) >= unlocks[k-1])
        enable('assembly', allowed)
        enable('assembly_core', {0, 1, 2} if heavy and a.get('difficulty') == 4 and arms == 2 else {0, 1} if heavy and a.get('difficulty') == 4 else {0})
        assembling = heavy and not r.get('weak') and a.get('stage') == 'inserting'
        for name in ('assembly_insert', 'assembly_lift', 'assembly_flip', 'assembly_confirm'):
            enable(name, {0, 1} if assembling else {0})
        enable('install_launcher', {0, 1} if heavy and not r.get('weak') and r.get('launcher') in ('unlocked', 'installing') else {0})
        for arm, count in (('arm', 1), ('arm2', 2)): enable(arm+'_mode', {0, 1, 2, 3} if heavy and arms >= count else {0})
        targets = {0, 1}
        for i, name in enumerate(CATEGORIES['target']):
            if name.startswith('enemy') and any(p['slot'] == int(name[5:]) and p.get('alive') for p in obs['enemies']): targets.add(i)
            if name.startswith('drone'):
                drones = obs.get('enemy_drones', []); index = int(name[5:])-1
                if index < len(drones) and drones[index].get('state') == 'flying': targets.add(i)
        other = 'blue' if obs['team'] == 'red' else 'red'
        for name, key in (('base', 'bases'), ('outpost', 'outposts')):
            if obs.get(key, {}).get(other, {}).get('hp', 0) > 0: targets.add(CATEGORIES['target'].index(name))
        enable('target', targets)
    return [continuous[k] for k in CONTINUOUS] + [float(v) for name in CAT_NAMES for v in masks[name]]


def split_mask(mask):
    result = {}; cursor = N_CONT
    for name, values in CATEGORIES.items():
        result[name] = mask[cursor:cursor+len(values)]; cursor += len(values)
    return mask[:N_CONT], result


def conditional_allowed(name, choices):
    """Choice-dependent legality used identically during sampling/evaluation."""
    if name == 'rescue' and choices.get('pickup') == 1: return {0}
    if name == 'arm_mode' and (choices.get('rescue', 0) > 0 or choices.get('pickup') == 1): return {0}
    if name == 'arm2_mode' and choices.get('pickup') == 2: return {0}
    return None


def active_heads(action):
    c = {k: int(action[ACTION_INDEX[k]]) for k in CAT_NAMES}
    active = {k: True for k in ACTION_NAMES}
    for k in ('move_x', 'move_y'): active[k] = c['motion'] == 0
    for k in ('goal_x', 'goal_y'): active[k] = c['motion'] == 1
    active['heading'] = c['turn'] == 1; active['yaw_rate'] = c['turn'] == 2
    active['power'] = c['power_mode'] == 1; active['leg_height'] = c['leg_mode'] == 1
    active['jump_strength'] = c['jump'] == 1
    active['aim_yaw'] = active['aim_pitch'] = c['target'] != 0
    active['ammo_batches'] = c['ammo'] != 0
    for arm in ('arm', 'arm2'):
        for k in CONTINUOUS:
            if k.startswith(arm+'_'): active[k] = c[arm+'_mode'] == (3 if '_joint' in k else 2)
    return [float(active[k]) for k in ACTION_NAMES]


def neutral_action():
    return [0.] * len(ACTION_NAMES)


def decode(action, obs, state=None):
    if len(action) != len(ACTION_NAMES) or not all(math.isfinite(float(v)) for v in action):
        raise ValueError('Invalid auxiliary action vector')
    if obs.get('observation_profile') != PROFILE: raise ValueError('Auxiliary profile required')
    state = {} if state is None else state
    mask, categories = split_mask(action_mask(obs)); choices = {}
    for name in CAT_NAMES:
        raw = float(action[ACTION_INDEX[name]]); value = int(raw)
        if raw != value or value < 0 or value >= len(CATEGORIES[name]):
            raise ValueError('Auxiliary categories must be exact integer choices')
        allowed = conditional_allowed(name, choices)
        if not categories[name][value] or allowed is not None and value not in allowed:
            value = 0  # stale state may disable a previously sampled request
        choices[name] = value
    v = {k: max(-1., min(1., float(action[i]))) * mask[i] for i, k in enumerate(CONTINUOUS)}
    r = own_robot(obs); d = sdk._design(r); flip = obs['team'] == 'blue'
    fields = {}; state['mode'] = 'policy'
    if not r.get('alive'):
        if choices['revive']: fields['revive'] = True
        return fields
    if choices['motion']:
        goal = [(v['goal_x']+1)*14, (v['goal_y']+1)*7.5]
        if flip: goal = [28-goal[0], 15-goal[1]]
        # Explicit fold/jump intent: the navigator executes geometry, it cannot
        # pick high-level assembly/resources, targets or tactical goals.
        intent = {'transform': {0: 0., 1: -1., 2: 1.}[choices['transform']],
                  'jump': float(bool(choices['jump']))}
        fields.update(sdk.navigate_action(r, goal, obs, state.setdefault('navigation', {}), locomotion_intent=intent))
        state['goal'] = goal
    else:
        sign = -1 if flip else 1
        fields['move'] = [sign*v['move_x'], sign*v['move_y']]
    fields['brake'] = bool(choices['brake'])
    if choices['turn'] == 1: fields['heading'] = v['heading']*math.pi + (math.pi if flip else 0)
    elif choices['turn'] == 2: fields['yaw_rate'] = v['yaw_rate']*12
    if choices['power_mode']: fields['power'] = (v['power']+1)*100
    if choices['transform'] and not choices['motion']: fields['transform'] = choices['transform'] == 2
    if d.get('chassis_type') == 'wheel_legged':
        if choices['leg_mode']:
            lo, hi = d.get('leg_min_m', .1), d.get('leg_max_m', .5)
            fields['leg_height'] = lo+(v['leg_height']+1)*.5*(hi-lo)
        fields['lean'] = v['lean']*.35
        if not choices['motion']:
            fields['jump'] = (v['jump_strength']+1)*.5 if choices['jump'] else 0.
    target_name = CATEGORIES['target'][choices['target']]
    if target_name == 'manual':
        fields.update(aim=v['aim_yaw']*math.pi+(math.pi if flip else 0), pitch=v['aim_pitch']*math.pi/2)
    elif target_name != 'hold':
        other = 'blue' if obs['team'] == 'red' else 'red'
        if target_name.startswith('enemy'):
            target = next(p for p in obs['enemies'] if p['slot'] == int(target_name[5:]))
        elif target_name.startswith('drone'):
            drone = obs['enemy_drones'][int(target_name[5:])-1]
            target = dict(drone, aim_z=drone.get('z', 0))
        else:
            target = dict(obs['bases' if target_name == 'base' else 'outposts'][other], kind=target_name)
        fields.update(aim=sdk.aim_at(r, target)+v['aim_yaw']*.2,
                      pitch=sdk.aim_pitch(r, target)+v['aim_pitch']*.2)
    fields['fire'] = bool(choices['fire'])
    for key in ('remote_heal', 'revive', 'rune_start', 'install_launcher',
                'assembly_insert', 'assembly_lift', 'assembly_flip', 'assembly_confirm'):
        if choices[key]: fields[key] = True
    if choices['ammo']:
        remote = choices['ammo'] == 2; kind = '42mm' if r['role'] == 'heavy' else '17mm'
        spec = sdk.rules(obs)['economy']['ammo'][kind]
        batch = int(spec['remote_batch' if remote else 'local_batch'])
        fields['buy_ammo'] = {'remote': remote, 'amount': batch*CATEGORIES['ammo_batches'][choices['ammo_batches']]}
    profile = {}
    if choices['chassis_profile']: profile['chassis'] = CATEGORIES['chassis_profile'][choices['chassis_profile']]
    if choices['weapon_profile']: profile['weapon'] = CATEGORIES['weapon_profile'][choices['weapon_profile']]
    if profile: fields['profile'] = profile
    if choices['pickup']: fields.update(pick_unit=True, pickup_arm=choices['pickup']-1)
    if choices['rescue']: fields['rescue'] = CATEGORIES['rescue'][choices['rescue']]
    if choices['assembly']: fields['assembly'] = choices['assembly']
    if r['role'] == 'heavy' and obs.get('assemblies', {}).get(obs['team'], {}).get('stage') in ('moving', 'inserting'):
        fields['assembly_core'] = choices['assembly_core']
        fields['assembly_rotate'] = v['assembly_rotate']*180
        if int(r.get('arm_count', 0)) == 2: fields['assembly_rotate2'] = v['assembly_rotate2']*180
    for arm in ('arm', 'arm2'):
        mode = choices[arm+'_mode']; design = d.get(arm, {})
        if mode == 1: fields[arm] = {'stow': True}
        elif mode == 2:
            yaw, pitch = v[arm+'_dir_yaw']*math.pi, v[arm+'_dir_pitch']*math.pi/2
            fields[arm] = dict(tip=[v[arm+'_x']*1.5, v[arm+'_y']*1.5, (v[arm+'_z']+1)*.8],
                dir=[math.cos(yaw)*math.cos(pitch), math.sin(yaw)*math.cos(pitch), math.sin(pitch)],
                frame='robot', roll=v[arm+'_roll']*math.pi, grip=(v[arm+'_grip']+1)*.5)
        elif mode == 3:
            limits = design.get('limits', [[-math.pi, math.pi]]*7)
            if not isinstance(limits, list) or len(limits) != 7: limits = [[-math.pi, math.pi]]*7
            fields[arm] = {'joints': [lo+(v[f'{arm}_joint{i}']+1)*.5*(hi-lo) for i, (lo, hi) in enumerate(limits)]}
    # Preserve the common 1 KiB protocol even for simultaneous dual-arm poses.
    # Micrometre/radian command precision is finer than physical integration.
    def compact(value):
        if isinstance(value, float): return round(value, 6)
        if isinstance(value, dict): return {k: compact(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)): return [compact(v) for v in value]
        return value
    fields = compact(fields)
    state['execution_trace'] = {'requested_action': list(action), 'executed_action': copy.deepcopy(fields)}
    return fields


def observation_space():
    import gymnasium as gym
    import numpy as np
    return gym.spaces.Dict({'actor': gym.spaces.Box(-np.inf, np.inf, (len(FEATURES),), np.float32),
                           'action_mask': gym.spaces.Box(0, 1, (MASK_SIZE,), np.float32)})


def action_space():
    import gymnasium as gym
    import numpy as np
    return gym.spaces.Box(np.array([-1.]*N_CONT+[0.]*len(CAT_NAMES), np.float32),
                         np.array([1.]*N_CONT+[len(v)-1 for v in CATEGORIES.values()], np.float32))


class AuxController:
    observation_profile = PROFILE
    control_hz = 20

    def __init__(self, config=None, model=None):
        self.config = dict(config or {}); self.model = model
        family = getattr(model, 'family', None)
        expected = 'heavy' if self.config.get('slot', 3) == 1 else 'infantry'
        if family is not None and family != expected:
            raise ValueError('Auxiliary frozen actor family does not match controlled seat')
        self.is_learning_controller = model is None
        self.state = {}; self.logs = []; self.disabled = False
        self.calls = self.errors = self.consecutive_timeouts = 0
        self.action = neutral_action()

    def select(self, obs, action):
        if obs.get('observation_profile') != PROFILE: raise ValueError('Auxiliary observation required')
        self.action = list(action)

    def decide(self, observation):
        from rmuc2027_core.policy import EncodedObservation
        obs = json.loads(observation.payload) if isinstance(observation, EncodedObservation) else observation
        self.calls += 1
        if self.model is not None and (obs.get('tick', 0) % 2 == 0 or 'prediction' not in self.state):
            import numpy as np
            inputs = {k: np.asarray(v, dtype=np.float32) for k, v in encode(obs).items()}
            action, _ = self.model.predict(inputs, deterministic=True)
            self.select(obs, action); self.state['prediction'] = True
        return {obs['controlled_ids'][0]: decode(self.action, obs, self.state)}, None

    def close(self):
        self.disabled = True; self.state.clear()


class OutpostDroneController:
    """Fixed aircraft role: strike the enemy field's outpost, with no messages."""
    observation_profile = 'outpost_drone_script_v1'
    control_hz = 10
    is_learning_controller = False
    disabled = False
    consecutive_timeouts = 0
    logs = ()

    def decide(self, observation):
        from rmuc2027_core.policy import EncodedObservation
        obs = json.loads(observation.payload) if isinstance(observation, EncodedObservation) else observation
        team = obs['team']; other = 'blue' if team == 'red' else 'red'
        outpost = obs.get('outposts', {}).get(other, {})
        plates = outpost.get('armor_plates', []) if outpost.get('hp', 0) > 0 and outpost.get('owner', other) != team else []
        actions = {}
        for r in obs['robots']:
            if str(r['slot']) not in obs['controlled_ids'] or r.get('role') != 'aerial': continue
            commands = {}
            for index, drone in enumerate(r.get('drones', [])):
                if not plates or drone.get('state') in ('disabled', 'spent'): continue
                plate = min(plates, key=lambda p: sum((drone.get(k, 0)-p['center'][i])**2 for i, k in enumerate(('x', 'y', 'z'))))
                phase, point = sdk.strike_plan(drone, plate)
                commands[str(index+1)] = {'move' if phase == 'strike' else 'goto': list(point)}
            actions[str(r['slot'])] = {'drones': commands}
        return actions, None

    def close(self): self.disabled = True
