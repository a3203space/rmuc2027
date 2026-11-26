"""Shared, standard-library actor input, goal decoding, execution and inference.

This exact source is embedded in exported submissions. No referee snapshot,
reward or info dictionary is accepted by the feature encoder.
"""
import math

try:
    import sdk as _sdk
except ImportError:
    from rmuc2027_core import sdk as _sdk

OBS_VERSION = 3
OBS_SIZE = 145
SLOTS = (1, 3, 4, 6, 7)
# Slot 6 carries the aerial system, which parks in a hangar and never drives:
# the rulebook only ever marks the four ground robots, so it gets no capability
# bits and its three trailing entries stay zero.
GROUND_SLOTS = (1, 3, 4, 7)
MODES = ("policy", "dead", "recover", "tipped", "unreachable")
# Divisor for the two `control` features. The engine's control score is
# uncapped, so the encoder has to pick a denominator, and the old hard-coded
# 1680 was never observed: a full game of the recorded red-side pool peaked at
# 863. Dividing by 1680 put every real value below 0.52 and then clipped, so the
# feature spent its whole life in the bottom half of the range and the network
# saw a dead input. 1000 is the measured peak rounded up, which keeps the
# normal path unclipped while still bounding the tail. Export bakes the value
# into the submission, so the divisor travels with the policy.
CONTROL_SCALE = 1000.0


def _ratio(value, maximum):
    return float(value) / max(1.0, float(maximum))


def capability_bits(robot):
    """Three orthogonal movement traits of one robot, from the public design table.

    `chassis_type` alone cannot answer the question that matters in combat.
    Mecanum covers both 2732/301/2773 (transform into a low profile and drive
    through the tunnel) and 4/7 (do neither), so an attacker cannot use the
    wheel type to predict where an opponent will reappear. The rulebook makes
    the distinction behavioural instead: crossing the tunnel over the top
    grants 25% defence for 30 s, while driving through it grants 1.5x barrel
    cooling for 120 s (rulebook table 5-27), so an opponent that can take the
    shortcut at all is the tactically different one. The 200 mm terrace next to
    the goal (rulebook 4.3.1) is the mirror case: models that report jump are
    the ones that can hop it, which is why 2731 carries step_climb_m == 0.2.
    Omni drive is kept separately because it strafes and flanks in a way the
    other two cannot, independent of either terrain feature.
    """
    design = _sdk.DESIGNS.get(str(robot.get('model_slot') or robot['slot']), {})
    capabilities = design.get('capabilities', {})
    return [float(bool(design.get('transform_allowed')) and bool(capabilities.get('tunnel'))),
            float(bool(capabilities.get('jump'))),
            float(design.get('chassis_type') == 'omni')]


def own_robot(obs, slot=7):
    if obs['controlled_ids'] != [str(slot)]:
        raise ValueError('Actor requires exactly its own slot-scoped observation')
    return next(r for r in obs['robots'] if r['slot'] == slot)


def encode_observation(obs, state, slot=7, rl_seconds=1.0, witness=None, control_scale=None):
    """145 bounded features, with blue coordinates rotated into the red frame.

    self 18 + four allies * 7 + five visible enemy slots * 11 (8 + 3 movement
    traits) + public match 6 + four public control positions * 4 + controller
    memory 8 + v3 tactical state 14 (sentry supply grant due 1 + enhanced posture
    budgets 3 + crossing cooling/defense remaining 2 + own/enemy outpost hp 2 +
    central card occupancy 4 + own fortress active 1 + team 17mm cap left 1).

    `witness` is an optional `terrain_witness.TerrainWitness`. Exported
    submissions embed `TerrainWitness(use_prior=False)` as well, so training
    and deployment read aligned observations.

    `control_scale` is the divisor for the two `control` features; None means
    `CONTROL_SCALE`. It is threaded through as an argument rather than read from
    a module global because the same source is embedded in exported
    submissions, where the trained run's value gets baked in - a module global
    would be the compile-time default there and silently rescale the input
    under a policy that learned against the configured one.
    """
    if control_scale is None:
        control_scale = CONTROL_SCALE
    control_scale = max(1e-6, float(control_scale))
    me = own_robot(obs, slot)
    team = obs['team']
    other = 'blue' if team == 'red' else 'red'
    w, h = obs['arena']['width'], obs['arena']['height']
    sign = 1 if team == 'red' else -1

    def xy(x, y):
        return (2*x/w-1)*sign, (2*y/h-1)*sign

    def motion(r):
        return [*xy(r['x'], r['y']), r.get('vx', 0)*sign/10,
                r.get('vy', 0)*sign/10, _ratio(r['hp'], r['max_hp'])]

    heading = me.get('heading', 0) + (math.pi if sign < 0 else 0)
    values = motion(me) + [me.get('ammo', 0)/500,
        _ratio(me.get('heat', 0), me.get('max_heat', 1)),
        me.get('energy', me.get('buffer_energy_j', 0))/250,
        float(me['alive']), float(me.get('weak', False)), float(me.get('tipped', False)),
        *[float(me.get('posture') == p) for p in ('move', 'attack', 'defense')],
        _ratio(me.get('respawn_progress', 0), me.get('respawn_required', 1)),
        float(me.get('powered', True)), math.sin(heading), math.cos(heading)]
    friends = {r['slot']: r for r in obs['robots']}
    for s in SLOTS:
        if s == slot:
            continue
        r = friends.get(s)
        values += motion(r) + [float(r['alive']), float(r.get('weak', False))] if r else [0.0]*7
    enemies = {r['slot']: r for r in obs['enemies']}
    for s in SLOTS:
        r = enemies.get(s)
        if r is None:
            values += [0.0] * 11
            continue
        values += [1.0, *motion(r), float(r['alive']), float(r.get('invulnerable', False))]
        if s in GROUND_SLOTS:
            values += witness.bits(s, r) if witness is not None else capability_bits(r)
        else:
            values += [0.0, 0.0, 0.0]
    values += [obs['remaining']/420,
        obs['teams'][team].get('control', 0)/control_scale,
        obs['teams'][other].get('control', 0)/control_scale,
        _ratio(obs['bases'][team]['hp'], obs['bases'][team]['max_hp']),
        _ratio(obs['bases'][other]['hp'], obs['bases'][other]['max_hp']),
        obs['teams'][team].get('resources', 0)/5000]
    for owner, kind in ((team, 'outpost'), (other, 'outpost'), (team, 'base'), (other, 'base')):
        p = obs['positions'][owner+'_'+kind]
        values += [float(p['controller'] == team), float(p['controller'] == other),
                   float(p['control'] == 'full'), p['progress']/5]
    goal = state.get('goal')
    values += list(xy(*goal)) if goal else [0.0, 0.0]
    values += [float(state.get('mode', 'policy') == mode) for mode in MODES]
    values += [(obs['time']-state.get('goal_time', obs['time']))/rl_seconds]
    # OBS v3 (+14): tactical awareness missing from the 131-dim layout.
    grants_due = (
        max(0, int(obs['time'] // 60.0) - int(me.get('supply_grants_taken', 0) or 0)) / 3.0
        if me.get('role') == 'sentry'
        else 0.0
    )
    enhanced_times = me.get('posture_enhanced_time') or (0.0, 0.0, 0.0)
    posture_budget = [
        max(0.0, (15.0 - float(enhanced_times[i])) / 15.0) if i < len(enhanced_times) else 1.0
        for i in range(3)
    ]
    cooling_rem = max(0.0, float(me.get('crossing_cooling_until') or 0.0) - float(obs['time'])) / 120.0
    defense_rem = max(0.0, float(me.get('crossing_defense_until') or 0.0) - float(obs['time'])) / 30.0
    outposts = obs.get('outposts') or {}
    outpost_hp = [
        _ratio(outposts[k]['hp'], outposts[k].get('max_hp', 1)) if k in outposts else 0.0
        for k in (team, other)
    ]
    zones = obs.get('zones') or {}
    ally_ids = {r.get('id') for r in obs.get('robots', ()) if r.get('slot') != slot and r.get('id')}
    friend_ids = {r.get('id') for r in obs.get('robots', ()) if r.get('id')}
    enemy_prefix = f'{other}-'
    own_central_occ = (zones.get(f'{team}_central') or {}).get('occupants') or ()
    other_central_occ = (zones.get(f'{other}_central') or {}).get('occupants') or ()
    ally_on_own_central = float(any(rid in ally_ids for rid in own_central_occ))
    enemy_on_own_central = float(any(str(rid).startswith(enemy_prefix) for rid in own_central_occ))
    ally_on_other_central = float(any(rid in friend_ids for rid in other_central_occ))
    enemy_on_other_central = float(any(str(rid).startswith(enemy_prefix) for rid in other_central_occ))
    positions = obs.get('positions') or {}
    own_fortress_active = float((positions.get(f'{team}_outpost') or {}).get('controller') != team)
    cap_left = max(0.0, 1.0 - float(obs['teams'][team].get('purchased_17', 0)) / 1000.0)
    values += [
        grants_due, *posture_budget, cooling_rem, defense_rem,
        *outpost_hp, ally_on_own_central, enemy_on_own_central,
        ally_on_other_central, enemy_on_other_central, own_fortress_active, cap_left,
    ]
    if len(values) != OBS_SIZE or not all(math.isfinite(v) for v in values):
        raise ValueError('Non-finite or incompatible actor observation')
    return [max(-1.0, min(1.0, float(v))) for v in values]


class NavigationBackend:
    """Replaceable public-geometry projection and navigation interface."""
    def resolve_goal(self, robot, proposed, arena):
        raise NotImplementedError

    def command(self, robot, goal, observation):
        raise NotImplementedError


class SdkNavigator(NavigationBackend):
    """SDK static visibility graph; no moving-robot avoidance or prediction.

    Grid legality and reachability are stricter than SDK's overlapping-goal
    fallback: targets inside inflated obstacles are never deliberately accepted.
    Caches contain public static geometry only and are owned by this navigator.
    """
    def __init__(self, grid_m=0.25):
        self.grid_m = grid_m
        self.grids = {}
        self.graph_components = {}
        self.reachability = {}
        self.precomputed_reachability = {}

    def _ordered_goals(self, points, proposed, key):
        return sorted(points, key=lambda p: ((p[0]-proposed[0])**2+(p[1]-proposed[1])**2, p))

    def resolve_goal(self, robot, proposed, arena):
        folded = robot.get('transform_progress', 0) >= .99
        # Use the same hull clearance as SDK command execution. Otherwise a
        # large chassis can select a legal-looking goal inside its own margin.
        graph = _sdk._prepare_chapter4(arena, robot['team'], folded,
                                      robot.get('radius', .28)+.04)
        _, _, clear, inside = graph
        key = (arena['layout_revision'], robot['team'], folded, robot.get('radius', .28), self.grid_m)
        if key not in self.grids:
            margin = max(.36, robot.get('radius', .28)+.08)
            g = self.grid_m
            self.grids[key] = [(i*g, j*g)
                for i in range(1, math.ceil(arena['width']/g))
                for j in range(1, math.ceil(arena['height']/g))
                if margin <= i*g <= arena['width']-margin
                and margin <= j*g <= arena['height']-margin and not inside((i*g, j*g))]
        start = (robot['x'], robot['y'])
        near = inside(start, True)
        graph_key=(id(graph),clear)
        if graph_key not in self.graph_components:
            nodes,edges,_,_=graph
            components=[-1]*len(nodes)
            for index in range(len(nodes)):
                if components[index]>=0:continue
                components[index]=index;pending=[index]
                while pending:
                    node=pending.pop()
                    for other,_ in edges[node]:
                        if components[other]<0:
                            components[other]=index;pending.append(other)
            groups={}
            for index,component in enumerate(components):groups.setdefault(component,[]).append(nodes[index])
            self.graph_components[graph_key]=sorted(groups.items(),key=lambda item:-len(item[1]))
        groups=self.graph_components[graph_key]
        start_components={}
        points = self._ordered_goals(self.grids[key], proposed, key)
        for goal in points:
            # Match the SDK's strict/relaxed static-route fallbacks. Dynamic
            # contacts may still obstruct a physically valid static route.
            for strict in (True, False):
                if clear(start,goal,near,strict):
                    return list(goal)
                for component,nodes in groups:
                    start_key=(strict,component)
                    if start_key not in start_components:
                        start_components[start_key]=any(clear(start,p,near,strict) for p in nodes)
                    if not start_components[start_key]:continue
                    reach_key=(graph_key,goal,strict,component)
                    if reach_key in self.precomputed_reachability:
                        if self.precomputed_reachability[reach_key]:return list(goal)
                        continue
                    if reach_key not in self.reachability:
                        reachable=any(clear(p,goal,(),strict) for p in nodes)
                        if len(self.reachability)>=4096:self.reachability.pop(next(iter(self.reachability)))
                        self.reachability[reach_key]=reachable
                    if self.reachability[reach_key]:return list(goal)
        return None

    def command(self, robot, goal, observation):
        return _sdk.navigate(robot, goal, observation['arena'])


def select_goal(obs, state, action, navigator, config):
    if len(action) != 2 or not all(math.isfinite(float(v)) for v in action):
        raise ValueError('Expected two finite normalized coordinates')
    clipped = [max(-1.0, min(1.0, float(v))) for v in action]
    w, h = obs['arena']['width'], obs['arena']['height']
    proposed = [(clipped[0]+1)*w/2, (clipped[1]+1)*h/2]
    if obs['team'] == 'blue':
        proposed = [w-proposed[0], h-proposed[1]]
    robot = own_robot(obs, config['slot'])
    folded = robot.get('transform_progress', 0) >= 0.99
    prev_goal = state.get('goal')
    prev_proposed = state.get('_goal_proposed')
    mode = state.get('mode', 'policy')
    same_fold = bool(state.get('_goal_folded', False)) == folded
    if prev_goal is not None and mode in ('dead', 'tipped') and same_fold:
        goal = list(prev_goal)
        locked_proposed = list(prev_proposed) if prev_proposed is not None else list(proposed)
    else:
        candidate = navigator.resolve_goal(robot, proposed, obs['arena'])
        if (
            prev_goal is not None
            and candidate is not None
            and same_fold
            and (
                math.hypot(candidate[0] - prev_goal[0], candidate[1] - prev_goal[1]) <= GOAL_HYSTERESIS_M
                or (
                    prev_proposed is not None
                    and math.hypot(proposed[0] - prev_proposed[0], proposed[1] - prev_proposed[1]) <= PROPOSED_HYSTERESIS_M
                )
            )
        ):
            goal = list(prev_goal)
            locked_proposed = list(prev_proposed) if prev_proposed is not None else list(proposed)
        else:
            goal = candidate
            locked_proposed = list(proposed)
            if prev_goal is None or goal is None or math.hypot(goal[0] - prev_goal[0], goal[1] - prev_goal[1]) > 1e-6:
                state['goal_settled'] = False
    state.update(goal=goal, goal_time=obs['time'], supplied_action=list(action),
                 clipped_action=clipped, proposed_goal=proposed,
                 _goal_folded=folded, _goal_proposed=locked_proposed)


def held_fire_safe(robot, seconds):
    """Budget all possible 17 mm shots before the next decision, without cooling credit."""
    # The referee can emit at most one shot per 50 ms game tick.
    ticks = max(1, math.ceil(seconds / .05 - 1e-9))
    interval = max(.05, float(robot.get('trigger_interval_s', .05)))
    spacing = max(1, math.ceil(interval / .05 - 1e-9))
    shots = min(int(robot['ammo']), 1 + (ticks - 1) // spacing)
    # Posture can change in this same command, so never assume the old no-heat buff persists.
    heat = 10
    return robot['heat'] + shots * heat <= robot['max_heat'] + 1e-8


def learning_ready(robot, state):
    """Latch hero takeover only after the referee confirms launcher installation."""
    if robot['role'] != 'heavy':
        return True
    if robot.get('launcher') == 'installed':
        if not state.get('hero_active'):
            for key in ('navigation', 'goal', 'goal_time', 'goal_tick', 'control_tick', 'held_actions', 'execution'):
                state.pop(key, None)
        state['hero_active'] = True
    return bool(state.get('hero_active'))


def fixed_opening(obs, state, config, baseline):
    if config['slot'] != 1:
        return None
    if learning_ready(own_robot(obs, config['slot']), state):
        return None
    actions, memory = baseline.act(obs, state.get('opening', {}))
    state['opening'] = memory
    slot = str(config['slot'])
    actions[slot] = _sdk.timed_assembly_action(own_robot(obs, config['slot']), actions.get(slot, {}),
                                              obs, state.setdefault('opening_adapter', {}))
    return actions


GOAL_STOP_RADIUS_M = 0.30
GOAL_RESUME_RADIUS_M = 0.48
GOAL_HYSTERESIS_M = 1.15
PROPOSED_HYSTERESIS_M = 1.0
MOVE_POSTURE_DISTANCE_M = 3.5
POSTURE_INDEX = {"attack": 0, "defense": 1, "move": 2}
ENHANCED_BUDGET_S = 14.9
REMOTE_BUY_MIN_COINS = 200.0
TEAM_17MM_BUY_LIMIT = 900
FREE_GRANT_NEAR_M = 8.0
RECOVER_COMMIT_M = 8.0
ATTACK_ENTRY_MIN_AMMO = 25


def _segment_box_clear_3d(ax, ay, az, bx, by, bz, wall):
    """Return True when the 3D segment (a -> b) misses the static wall slab."""
    rx, ry, rw, rh = wall['x'], wall['y'], wall['w'], wall['h']
    cx, cy = rx + rw * 0.5, ry + rh * 0.5
    rot = float(wall.get('rotation', 0.0) or 0.0)
    c, s = math.cos(rot), math.sin(rot)
    dx, dy, dz = bx - ax, by - ay, bz - az
    ox, oy = ax - cx, ay - cy
    lx = cx + ox * c + oy * s
    ly = cy - ox * s + oy * c
    ldx = dx * c + dy * s
    ldy = -dx * s + dy * c
    lo, hi = 0.0, 1.0
    z0 = float(wall.get('bottom', 0.0) or 0.0)
    z1 = z0 + float(wall.get('height', 10.0) or 10.0)
    for pos, d, a, b in ((lx, ldx, rx, rx + rw), (ly, ldy, ry, ry + rh), (az, dz, z0, z1)):
        if abs(d) < 1e-12:
            if pos < a or pos > b:
                return True
        else:
            t0, t1 = (a - pos) / d, (b - pos) / d
            if t0 > t1:
                t0, t1 = t1, t0
            lo = max(lo, t0)
            hi = min(hi, t1)
            if lo > hi:
                return True
    return False


def _structure_los_clear(obs, eye, target_pt):
    ax, ay, az = eye
    bx, by, bz = target_pt
    for wall in (obs.get('arena') or {}).get('obstacles', ()):
        if not _segment_box_clear_3d(ax, ay, az, bx, by, bz, wall):
            return False
    return True


def _structure_aim_point(robot, struct, visible_faces=None):
    plates = struct.get('armor_plates') or ()
    if visible_faces is not None:
        plates = [p for p in plates if p.get('face') in visible_faces]
    if plates:
        eye_z = float(robot.get('z', 0.0)) + float(robot.get('muzzle_height_m', 0.45)) + 0.08
        best = None
        for p in plates:
            center = p['center']
            normal = p['normal']
            diff = (robot['x'] - center[0], robot['y'] - center[1], eye_z - center[2])
            dist = max(1e-3, math.sqrt(sum(v * v for v in diff)))
            score = sum(a * b for a, b in zip(normal, diff)) / dist
            if best is None or score > best[0]:
                best = (score, center)
        if best is not None and best[0] > 0.0:
            return tuple(best[1]), dict(struct, armor_plates=[p for p in plates if tuple(p['center']) == tuple(best[1])])
    return (struct['x'], struct['y'], float(struct.get('z', 0.0)) + 0.45), struct


def _visible_structure_target(obs, robot, ctx, reach=10.0):
    eye = (
        robot['x'],
        robot['y'],
        float(robot.get('z', 0.0)) + float(robot.get('muzzle_height_m', 0.45)) + 0.08,
    )
    vision_slot = (obs.get('vision') or {}).get(str(robot['slot']), {})
    vision_range = min(reach, float(vision_slot.get('range_m', reach)))
    has_outpost_vision = 'vision' in obs and str(robot['slot']) in obs['vision'] and 'outposts' in vision_slot

    outpost = ctx.hostile_outpost()
    if outpost:
        dist = math.hypot(outpost['x'] - robot['x'], outpost['y'] - robot['y'])
        visible_faces = (vision_slot.get('outposts') or {}).get(outpost['site'])
        if dist <= vision_range and (not has_outpost_vision or visible_faces):
            aim_pt, refined = _structure_aim_point(robot, outpost, set(visible_faces) if visible_faces else None)
            if _sdk.ballistic_reachable(robot, refined) and _structure_los_clear(obs, eye, aim_pt):
                return refined

    if ctx.enemy_base_vulnerable():
        base = dict(obs['bases'][ctx.enemy], kind='base')
        dist = math.hypot(base['x'] - robot['x'], base['y'] - robot['y'])
        if dist <= vision_range:
            aim_pt, refined = _structure_aim_point(robot, base)
            if _sdk.ballistic_reachable(robot, refined) and _structure_los_clear(obs, eye, aim_pt):
                return refined
    return None


def _select_weapon_shot(obs, robot, ctx, baseline, reach=10.0):
    try:
        shot = baseline.weapon(robot, ctx, reach=reach, structures=False, drones_too=True)
    except TypeError:
        target = baseline.enemy_target(robot, ctx, reach) if hasattr(baseline, 'enemy_target') else None
        if target is None and robot['role'] != 'heavy' and hasattr(baseline, 'drone_target'):
            target = baseline.drone_target(robot, ctx)
        shot = {} if target is None else None
        if target is not None:
            aim = _sdk.aim_at(robot, target)
            heavy = robot['role'] == 'heavy'
            aligned = abs((aim - robot['aim'] + math.pi) % math.tau - math.pi) < (0.08 if heavy else 0.14)
            fire = aligned and _sdk.can_fire(robot) and (heavy or robot['heat'] + 20 <= robot['max_heat'])
            shot = {'aim': aim, 'pitch': _sdk.aim_pitch(robot, target), 'fire': fire}
    if shot:
        return shot
    struct = _visible_structure_target(obs, robot, ctx, reach=reach)
    if struct is None:
        return {}
    aim = _sdk.aim_at(robot, struct)
    heavy = robot['role'] == 'heavy'
    aligned = abs((aim - robot['aim'] + math.pi) % math.tau - math.pi) < (0.08 if heavy else 0.14)
    fire = aligned and _sdk.can_fire(robot) and (heavy or robot['heat'] + 20 <= robot['max_heat'])
    return {'aim': aim, 'pitch': _sdk.aim_pitch(robot, struct), 'fire': fire}


def _bump_zone_for(robot, arena, margin_x=0.0, margin_y=0.0):
    """Return the matching speed-bump zone if `robot` is within its expanded bounds."""
    rx, ry = float(robot['x']), float(robot['y'])
    for terrain in (arena or {}).get('terrain') or ():
        if terrain.get('kind') != 'bumps':
            continue
        x0 = float(terrain['x']) - margin_x
        y0 = float(terrain['y']) - margin_y
        x1 = float(terrain['x']) + float(terrain['w']) + margin_x
        y1 = float(terrain['y']) + float(terrain['h']) + margin_y
        if x0 <= rx <= x1 and y0 <= ry <= y1:
            return terrain
    return None


def _bump_axis_heading(robot, bump_zone):
    """Return the heading aligned with the bump zone's traversal axis closest to current heading."""
    hdg_wrap = (float(robot.get('heading', 0.0)) + math.pi) % math.tau - math.pi
    if str(bump_zone.get('axis', 'x')) == 'x':
        return 0.0 if abs(hdg_wrap) <= math.pi * 0.5 else math.pi
    return math.pi * 0.5 if hdg_wrap >= 0.0 else -math.pi * 0.5


def _unstick_fields(robot, fields, goal, obs, state, on_bumps):
    """Back off and re-align when translational progress stalls while driving."""
    u = state.setdefault('unstick', {'trail': [], 'tries': 0})
    now = float(obs['time'])
    if now < float(u.get('until', -1.0)):
        return {'move': list(u['escape']), 'heading': u['heading']}
    trail = u['trail']
    trail.append([now, float(robot['x']), float(robot['y']), float(robot.get('heading', 0.0))])
    while len(trail) > 1 and now - trail[0][0] > 1.75:
        trail.pop(0)
    elapsed = now - trail[0][0]
    if on_bumps is not None:
        axis = str(on_bumps.get('axis', 'x'))
        moved = abs(float(robot['x']) - trail[0][1]) if axis == 'x' else abs(float(robot['y']) - trail[0][2])
    else:
        moved = math.hypot(float(robot['x']) - trail[0][1], float(robot['y']) - trail[0][2])
    turned = abs((float(robot.get('heading', 0.0)) - trail[0][3] + math.pi) % math.tau - math.pi)
    if moved > 0.45:
        u['tries'] = 0
    if elapsed < 1.35 or moved > 0.12 or (on_bumps is None and turned > 0.15):
        return fields
    u['tries'] = int(u.get('tries', 0)) + 1
    dx, dy = fields['move']
    if math.hypot(dx, dy) < 0.05 and goal is not None:
        dx, dy = float(goal[0]) - float(robot['x']), float(goal[1]) - float(robot['y'])
    if on_bumps is not None:
        axis = str(on_bumps.get('axis', 'x'))
        if axis == 'x':
            rx = -1.0 if dx >= 0.0 else 1.0
            mid_y = float(on_bumps['y']) + 0.5 * float(on_bumps['h'])
            if abs(float(robot['y']) - mid_y) > 0.35:
                ry = 0.18 if float(robot['y']) < mid_y else -0.18
            else:
                ry = 0.18 if (u['tries'] % 2) else -0.18
        else:
            ry = -1.0 if dy >= 0.0 else 1.0
            mid_x = float(on_bumps['x']) + 0.5 * float(on_bumps['w'])
            if abs(float(robot['x']) - mid_x) > 0.35:
                rx = 0.18 if float(robot['x']) < mid_x else -0.18
            else:
                rx = 0.18 if (u['tries'] % 2) else -0.18
        hold_heading = _bump_axis_heading(robot, on_bumps)
        norm = max(1e-9, math.hypot(rx, ry))
        escape = [0.95 * rx / norm, 0.95 * ry / norm]
        u.update(escape=escape, until=now + 1.1, heading=hold_heading, bump_boost_until=now + 3.8)
    else:
        offset = 0.9 if (robot['slot'] + u['tries']) % 2 else -0.9
        angle = math.atan2(dy, dx) + math.pi + offset
        escape = [0.75 * math.cos(angle), 0.75 * math.sin(angle)]
        u.update(escape=escape, until=now + 0.75, heading=float(robot.get('heading', 0.0)))
    trail.clear()
    return {'move': list(u['escape']), 'heading': u['heading']}


def execute_goal(obs, state, navigator, config, baseline):
    """Movement arbitration; weapon and resource operations remain rule modules."""
    robot = own_robot(obs, config['slot'])
    ctx = baseline.Context(obs, state.setdefault('execution', {}))
    fields = {}
    goal = state.get('goal')
    hp = _ratio(robot['hp'], robot['max_hp'])
    was_recovering = bool(state.get('recovering', False))
    supply = ctx.zone(obs['team'], 'supply')
    dist_to_supply = math.hypot(robot['x'] - supply[0], robot['y'] - supply[1])
    team_state = (obs.get('teams') or {}).get(obs['team'], {})
    coins = float(team_state.get('resources', 0.0) or 0.0)
    purchased_17 = int(team_state.get('purchased_17', 0) or 0)
    can_retreat = True
    if robot['role'] == 'sentry':
        central = ctx.zone(obs['team'], 'central')
        covered = any(
            r['alive'] and not r.get('weak') and r['slot'] != robot['slot'] and r['role'] != 'aerial'
            and math.hypot(r['x'] - central[0], r['y'] - central[1]) < 0.8
            for r in obs['robots']
        )
        can_retreat = (
            covered
            or ctx.controls(f"{obs['team']}_outpost", full=True)
            or (was_recovering and dist_to_supply <= RECOVER_COMMIT_M)
        )
    due_grants = int(obs['time'] // 60.0)
    taken_grants = int(robot.get('supply_grants_taken', 0) or 0)
    can_afford_remote = (purchased_17 <= TEAM_17MM_BUY_LIMIT) and (coins >= REMOTE_BUY_MIN_COINS)
    free_grant_due = (
        robot['role'] == 'sentry'
        and int(robot.get('ammo', 0)) == 0
        and not robot.get('pending_ammo')
        and due_grants > taken_grants
        and (not can_afford_remote or dist_to_supply <= FREE_GRANT_NEAR_M)
    )
    recovering = bool(
        robot.get('weak', False)
        or (can_retreat and (
            hp < config['recover_enter']
            or (was_recovering and hp < config['recover_exit'])
            or free_grant_due
        ))
    )
    state['recovering'] = recovering
    if not robot['alive'] or robot.get('red_card'):
        mode, goal = 'dead', None
        state['recovering'] = False
        state['goal_settled'] = False
        state.pop('navigation', None)
        state.pop('unstick', None)
        fields.update(baseline.revive_fields(robot, ctx))
    elif robot.get('tipped'):
        mode, goal = 'tipped', None
        state['goal_settled'] = False
        state.pop('unstick', None)
    elif recovering:
        mode = 'recover'
        state['goal_settled'] = False
        # The rule module chooses a recovery destination; use the same legality
        # checks as policy destinations rather than driving into a wall.
        goal = navigator.resolve_goal(robot, supply, obs['arena'])
    else:
        mode = 'policy' if goal is not None else 'unreachable'
    control_dt = 1.0 / max(1.0, float(config.get('learning_control_hz', 10)))
    nav_state = state.setdefault('navigation', {})
    in_manoeuvre = any(nav_state.get(k, {}).get('phase', 'approach') != 'approach' for k in ('entry', 'tunnel'))
    gap = math.hypot(robot['x'] - goal[0], robot['y'] - goal[1]) if goal is not None else 0.0
    on_bumps = _bump_zone_for(robot, obs.get('arena'), margin_x=0.35, margin_y=0.25)
    approaching_bumps = _bump_zone_for(robot, obs.get('arena'), margin_x=1.5, margin_y=0.5)
    if goal is None:
        state['goal_settled'] = False
        state.pop('unstick', None)
        fields['move'] = (0.0, 0.0)
    elif not in_manoeuvre and mode != 'recover' and (
        gap <= GOAL_STOP_RADIUS_M
        or (state.get('goal_settled') and gap <= GOAL_RESUME_RADIUS_M)
    ):
        state['goal_settled'] = True
        nav_state.pop('teammate_detour', None)
        nav_state.pop('teammate_wait_since', None)
        state.pop('unstick', None)
        fields.update(move=(0.0, 0.0), brake=True)
    else:
        state['goal_settled'] = False
        nav_obs = obs if 'policy_dt' in obs else dict(
            obs, policy_dt=control_dt, navigation_mode=config.get('navigation_mode', 'legacy'))
        fields.update(_sdk.navigate_action(robot, goal, nav_obs, nav_state))
        raw_move = fields['move']
        step_travel = float(robot.get('speed_reference_mps', 3.0)) * control_dt
        if (gap < step_travel
                and not in_manoeuvre
                and not any(fields.get(k) for k in ('jump', 'transform', 'brake'))
                and getattr(raw_move, 'heading', None) is None):
            mag = math.hypot(*raw_move)
            if mag > 1e-6:
                scale = min(1.0, gap / max(step_travel, 1e-6))
                fields['move'] = (raw_move[0] * scale, raw_move[1] * scale)
        if not in_manoeuvre and gap > GOAL_RESUME_RADIUS_M and not any(fields.get(k) for k in ('jump', 'transform', 'brake')):
            fields = _unstick_fields(robot, fields, goal, obs, state, on_bumps)
        elif 'unstick' in state:
            state['unstick']['trail'].clear()
    now = float(obs['time'])
    unstick_state = state.get('unstick') or {}
    in_unstick = now < float(unstick_state.get('until', -1.0))
    bump_boost = now < float(unstick_state.get('bump_boost_until', -1.0))
    if (
        robot['alive']
        and not in_manoeuvre
        and not in_unstick
        and not any(fields.get(k) for k in ('jump', 'transform', 'brake'))
    ):
        active_bump = on_bumps or (approaching_bumps if bump_boost else None)
        if active_bump is not None:
            fields['heading'] = _bump_axis_heading(robot, active_bump)
            if on_bumps is not None:
                mx, my = fields['move']
                orig_mag = math.hypot(mx, my)
                if orig_mag > 0.2:
                    if str(on_bumps.get('axis', 'x')) == 'x':
                        if abs(mx) > 0.1:
                            lim = 0.15 * abs(mx)
                            my = max(-lim, min(lim, my))
                    else:
                        if abs(my) > 0.1:
                            lim = 0.15 * abs(my)
                            mx = max(-lim, min(lim, mx))
                    new_mag = math.hypot(mx, my)
                    if new_mag > 1e-9:
                        fields['move'] = (mx * orig_mag / new_mag, my * orig_mag / new_mag)
    speed = math.hypot(float(robot.get('vx', 0.0) or 0.0), float(robot.get('vy', 0.0) or 0.0))
    cmd_mag = math.hypot(*fields['move'])
    supercap_ok = ('supercap_energy_j' not in robot) or float(robot.get('supercap_energy_j', 0.0) or 0.0) > 150.0
    if (
        robot['alive']
        and not nav_state.get('limited')
        and not any(fields.get(k) for k in ('jump', 'transform', 'brake'))
        and cmd_mag > 0.2
        and supercap_ok
        and (
            speed < 0.35
            or in_unstick
            or ((approaching_bumps is not None or bump_boost) and speed < 1.8)
        )
    ):
        fields['power'] = 200
    move = fields['move']
    if robot['alive']:
        shot = _select_weapon_shot(obs, robot, ctx, baseline, reach=10.0)
        fields.update(shot)
        if fields.get('fire') and not held_fire_safe(robot, control_dt):
            fields['fire'] = False
        skip_ammo_buy = (
            mode == 'recover'
            and robot['role'] == 'sentry'
            and due_grants > taken_grants
            and gap <= 4.5
        )
        if not skip_ammo_buy:
            fields.update(baseline.ammo_fields(robot, ctx, 8, 20, 10, 100) if robot['role'] == 'heavy'
                          else baseline.ammo_fields(robot, ctx, 40, 100, 100, 50))
        if robot['role'] == 'sentry':
            moving = math.hypot(*move) > 0.05
            threatened = obs['time'] - robot.get('last_damage_time', -99.0) < 2.0
            has_ammo = int(robot.get('ammo', 0)) > 0
            attack_ammo_ok = int(robot.get('ammo', 0)) >= (1 if robot.get('posture') == 'attack' else ATTACK_ENTRY_MIN_AMMO)
            combat_ready = attack_ammo_ok and bool(shot)
            live_duel = has_ammo and any(
                e['alive'] and not e.get('invulnerable')
                and math.hypot(e['x'] - robot['x'], e['y'] - robot['y']) <= 10.0
                for e in obs.get('enemies', ())
            )
            traverse_bumps = (
                (on_bumps is not None or approaching_bumps is not None or in_unstick or bump_boost)
                and goal is not None
                and not state.get('goal_settled')
                and gap > GOAL_RESUME_RADIUS_M
            )
            if traverse_bumps:
                wanted = 'move'
            elif recovering:
                wanted = 'defense' if (hp < 0.40 or gap <= 3.0) else 'move'
            elif hp < 0.65 and threatened:
                wanted = 'defense'
            elif combat_ready and not (moving and gap > MOVE_POSTURE_DISTANCE_M):
                wanted = 'attack'
            elif moving and gap > MOVE_POSTURE_DISTANCE_M:
                wanted = 'move'
            else:
                wanted = 'defense'
            enhanced_times = robot.get('posture_enhanced_time') or (0.0, 0.0, 0.0)
            idx = POSTURE_INDEX[wanted]
            budget_ok = idx < len(enhanced_times) and float(enhanced_times[idx]) < ENHANCED_BUDGET_S
            heat_ratio = _ratio(robot.get('heat', 0.0), robot.get('max_heat', 260.0))
            want_enhanced = bool(
                budget_ok and (
                    (wanted == 'defense' and hp < 0.55 and threatened)
                    or (wanted == 'attack' and live_duel and bool(shot) and heat_ratio >= 0.70)
                )
            )
            if obs['time'] >= (robot.get('posture_ready_at') or 0.0):
                if wanted != robot.get('posture'):
                    fields['posture'] = wanted
                if bool(robot.get('posture_enhanced', False)) != want_enhanced:
                    fields['posture_enhanced'] = want_enhanced
    command = _sdk.action(**fields)
    state.update(mode=mode, executed_goal=goal, executed_action=command)
    return {str(config['slot']): command}


def mlp_forward(features, layers):
    """Deterministic actor mean, from exported Linear/Tanh layers (no torch)."""
    values = features
    for index, layer in enumerate(layers):
        values = [sum(a*b for a, b in zip(row, values))+bias
                  for row, bias in zip(layer['weight'], layer['bias'])]
        act_fn = layer.get('activation', 'tanh' if index < len(layers) - 1 else 'linear')
        if act_fn == 'tanh':
            values = [math.tanh(v) for v in values]
    return values
