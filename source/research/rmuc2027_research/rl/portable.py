"""Shared, standard-library actor input, goal decoding, execution and inference.

This exact source is embedded in exported submissions. No referee snapshot,
reward or info dictionary is accepted by the feature encoder.
"""
import math
try:
    from . import sentry_v3, sentry_v4, sentry_v6
except (ImportError, KeyError):
    pass  # Exported submissions embed the same module below this source.

try:
    import sdk as _sdk
except ImportError:
    from rmuc2027_core import sdk as _sdk

OBS_VERSION = 1
OBS_SIZE = 116
SLOTS = (1, 3, 4, 6, 7)
MODES = ("policy", "dead", "recover", "tipped", "unreachable")
LOCOMOTION_PHASES = ('route', 'approach', 'fold', 'cross', 'exit', 'runup', 'launch', 'airborne', 'settle', 'recover')
LOCOMOTION_SIZE = 36


def policy_schema(config=None):
    version = (config or {}).get('policy_version', 1)
    if version == 6:
        return sentry_v6.schema()
    if version == 4:
        return sentry_v4.schema()
    if version == 5:
        from .privileged_aux import schema
        return schema()
    if version == 3:
        return sentry_v3.schema()
    if version not in (1, 2):
        raise ValueError('Unsupported policy version')
    return {'version': version, 'obs_size': OBS_SIZE+(LOCOMOTION_SIZE if version == 2 else 0),
            'action_size': 5 if version == 2 else 2}


def _ratio(value, maximum):
    return float(value) / max(1.0, float(maximum))


def own_robot(obs, slot=7):
    if obs['controlled_ids'] != [str(slot)]:
        raise ValueError('Actor requires exactly its own slot-scoped observation')
    return next(r for r in obs['robots'] if r['slot'] == slot)


def encode_observation(obs, state, slot=7, rl_seconds=1.0, config=None):
    """116 bounded features, with blue coordinates rotated into the red frame.

    self 18 + four allies * 7 + five visible enemy slots * 8 + public match 6
    + four public control positions * 4 + controller memory 8.
    """
    if policy_schema(config)['version'] == 6:
        return sentry_v6.encode(obs,state)
    if policy_schema(config)['version'] == 4:
        return sentry_v4.encode(obs,state)
    if policy_schema(config)['version'] == 3:
        return sentry_v3.encode(obs,state)
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
        values += [1.0, *motion(r), float(r['alive']), float(r.get('invulnerable', False))] if r else [0.0]*8
    values += [obs['remaining']/420,
        obs['teams'][team].get('control', 0)/1680,
        obs['teams'][other].get('control', 0)/1680,
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
    if len(values) != OBS_SIZE or not all(math.isfinite(v) for v in values):
        raise ValueError('Non-finite or incompatible actor observation')
    if policy_schema(config)['version'] == 2:
        values += locomotion_features(me, obs, state)
    if len(values) != policy_schema(config)['obs_size'] or not all(math.isfinite(v) for v in values):
        raise ValueError('Non-finite or incompatible locomotion observation')
    return [max(-1.0, min(1.0, float(v))) for v in values]


def locomotion_features(robot, obs, state):
    nav = state.get('navigation', {})
    ctx = _sdk.locomotion_context(robot, obs, nav)
    d = _sdk._design(robot)
    lo, hi = d.get('leg_min_m', 0.), d.get('leg_max_m', 1.)
    leg = lambda x: 2*(x-lo)/max(.01, hi-lo)-1
    legs = robot.get('legs') or [lo, lo]
    roof = ctx['roof'] or dict(gap=5., along=0., across=0.)
    a = robot.get('heading', 0.)
    vx, vy = robot.get('vx', 0.), robot.get('vy', 0.)
    angular = robot.get('angular_velocity', [0., 0., 0.])
    skills = nav.get('executed_skills', {})
    goal = state.get('goal') or [robot['x'], robot['y']]
    sign = 1. if obs['team']=='red' else -1.
    values = [float(ctx['can_jump']), float(ctx['can_fold']), float(ctx['legged']),
        float(d.get('leg_type')=='serial'), float(robot.get('grounded', True)), robot.get('upright', 1.),
        robot.get('vz', 0.)/5, leg(legs[0]), leg(legs[1]), robot.get('transform_progress', 0.),
        ctx['cooldown']/max(.1, d.get('jump_cooldown_s', 1.)), float(ctx['can_unfold']),
        roof['gap']/5, roof['along']/5, roof['across']/5,
        (vx*math.cos(a)+vy*math.sin(a))/5, (-vx*math.sin(a)+vy*math.cos(a))/5,
        angular[0]/5, angular[1]/5, leg(skills.get('leg_height') or lo),
        float(nav.get('fold_intent', False)), float(nav.get('jump_held', False)),
        sign*(goal[0]-robot['x'])/28, sign*(goal[1]-robot['y'])/15,
        *[float(nav.get('phase', 'route')==phase) for phase in LOCOMOTION_PHASES],
        float(bool(nav.get('rejected'))), float(state.get('course_kind')=='flat_jump')]
    return values


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
    schema = policy_schema(config)
    if len(action) != schema['action_size'] or not all(math.isfinite(float(v)) for v in action):
        raise ValueError(f"Expected {schema['action_size']} finite normalized actions")
    continuous = action[:6] if schema['version']==6 else action
    clipped = [max(-1.0, min(1.0, float(v))) for v in continuous]
    if schema['version']==6:
        category=sentry_v6.validate_revive_action(action[6])
        mask=sentry_v6.revive_mask(obs)
        if not mask[category]:
            raise ValueError('Revival category is masked by the current legal observation')
        clipped.append(float(category))
        state['revive_intent']=category
        state['action_selected_dead']=bool(mask[1])
    state.update(supplied_action=list(action), clipped_action=clipped)
    if schema['version'] in (3,4,6):
        state['sentry_intent'] = dict(zip(('fold','jump','spin','fire'),clipped[2:6]))
    if schema['version'] == 2:
        state['locomotion_intent'] = dict(zip(('transform', 'leg_height', 'jump'), clipped[2:]))
        if state.get('course_goal') is not None:
            state.update(goal=list(state['course_goal']), goal_time=obs['time'], proposed_goal=list(state['course_goal']))
            return
        if 'goal_time' in state and obs['time'] >= state['goal_time'] and obs['time']-state['goal_time'] < .5-1e-8:
            return
    w, h = obs['arena']['width'], obs['arena']['height']
    proposed = [(clipped[0]+1)*w/2, (clipped[1]+1)*h/2]
    if obs['team'] == 'blue':
        proposed = [w-proposed[0], h-proposed[1]]
    goal = navigator.resolve_goal(own_robot(obs, config['slot']), proposed, obs['arena'])
    state.update(goal=goal, goal_time=obs['time'], supplied_action=list(action),
                 clipped_action=clipped, proposed_goal=proposed)


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
    if state.get('course_goal') is not None:
        return None
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


def execute_goal(obs, state, navigator, config, baseline):
    if policy_schema(config)['version'] == 6:
        return sentry_v6.execute(obs,state,navigator,config)
    """Movement arbitration; weapon and resource operations remain rule modules."""
    if policy_schema(config)['version'] == 4:
        return sentry_v4.execute(obs,state,navigator,config)
    if policy_schema(config)['version'] == 3:
        return sentry_v3.execute(obs,state,navigator,config)
    robot = own_robot(obs, config['slot'])
    ctx = baseline.Context(obs, state.setdefault('execution', {}))
    fields = {}
    goal = state.get('goal')
    hp = _ratio(robot['hp'], robot['max_hp'])
    recovering = state.get('recovering', False)
    recovering = robot.get('weak', False) or hp < config['recover_enter'] or (recovering and hp < config['recover_exit'])
    state['recovering'] = bool(recovering)
    if not robot['alive'] or robot.get('red_card'):
        mode, goal = 'dead', None
        state.pop('navigation', None)
        fields.update(baseline.revive_fields(robot, ctx))
    elif robot.get('tipped'):
        mode, goal = 'tipped', None
    elif recovering:
        mode = 'recover'
        # The rule module chooses a recovery destination; use the same legality
        # checks as policy destinations rather than driving into a wall.
        goal = navigator.resolve_goal(robot, ctx.zone(obs['team'], 'supply'), obs['arena'])
    else:
        mode = 'policy' if goal is not None else 'unreachable'
    move = navigator.command(robot, goal, obs) if goal is not None else (0.0, 0.0)
    fields.update(_sdk.navigate_action(robot, goal, obs, state.setdefault('navigation', {}),
                  locomotion_intent=state.get('locomotion_intent'))
                  if goal is not None else {'move': move})
    move = fields['move']
    if robot['alive']:
        fields.update(baseline.weapon(robot, ctx, reach=10))
        if fields.get('fire') and not held_fire_safe(robot, 1 / config.get('learning_control_hz', 10)):
            fields['fire'] = False
        fields.update(baseline.ammo_fields(robot, ctx, 8, 20, 10, 100) if robot['role'] == 'heavy'
                      else baseline.ammo_fields(robot, ctx, 40, 100, 100, 50))
        moving = math.hypot(*move) > .05
        wanted = 'move' if moving else ('defense' if recovering else 'attack')
        if robot['role'] == 'sentry' and obs['time'] >= (robot.get('posture_ready_at') or 0) and wanted != robot.get('posture'):
            fields['posture'] = wanted
    command = _sdk.action(**fields)
    state.update(mode=mode, executed_goal=goal, executed_action=command)
    return {str(config['slot']): command}


def mlp_forward(features, layers):
    if isinstance(layers,dict) and layers.get('version')==6:
        return sentry_v6.portable_forward(features,layers)
    """Deterministic actor mean, from exported Linear/Tanh layers (no torch)."""
    if isinstance(layers,dict) and layers.get('version')==4:
        return sentry_v4.portable_forward(features,layers)
    values = features
    for index, layer in enumerate(layers):
        values = [sum(a*b for a, b in zip(row, values))+bias
                  for row, bias in zip(layer['weight'], layer['bias'])]
        if index < len(layers)-1:
            values = [math.tanh(v) for v in values]
    return values
