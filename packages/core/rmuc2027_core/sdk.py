"""Public strategy API; this is the only arena code mounted in a policy sandbox.

Implement act(observation, state) -> (actions, new_state).
Actions are keyed by robot slot, e.g. {"1": action(...)}. In independent modes
only your own controlled_ids slot is accepted. State is private and JSON serializable.
"""
import math
import json
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Literal, TypedDict


class RobotState(TypedDict, total=False):
    id: str
    slot: int
    model_slot: int
    team: str
    role: str
    x: float
    y: float
    z: float
    vx: float
    vy: float
    hp: float
    max_hp: float
    ammo: int
    aim: float
    pitch: float
    heading: float
    quaternion: list[float]
    heat: float
    energy: float
    alive: bool
    upright: float
    tipped: bool
    powered: bool
    legs: list[float]
    arm_mode: str
    arm_joints: list[float]
    arm_tip: list[float]
    rescue_contact: bool


class Action(TypedDict, total=False):
    move: list[float]
    aim: float | None
    pitch: float
    fire: bool
    power: float
    altitude: float
    profile: str | dict[str, str]
    buy_ammo: dict[str, Any]
    remote_heal: bool
    revive: bool
    deploy: bool
    support: bool
    pick_unit: bool
    assembly: int
    assembly_insert: bool
    assembly_translate: bool
    assembly_rotate: float
    assembly_rotate2: float
    assembly_confirm: bool
    rune_start: bool
    heading: float
    crouch: bool
    rescue: int | str
    yaw_rate: float
    brake: bool
    leg_height: float
    jump: float
    lean: float
    arm: dict[str, Any]
    messages: list[dict[str, Any]]
    # RMUC 2027
    install_launcher: bool
    assembly_core: int
    assembly_lift: bool
    assembly_flip: bool
    posture: str
    posture_enhanced: bool
    drones: dict[str, dict[str, Any]]


class _TeamInbox(TypedDict, total=False):
    # Team mode aggregates all controlled inboxes; single-robot modes keep one inbox.
    messages: list[dict[str, Any]]
    message_inboxes: dict[str,list[dict[str,Any]]]


class Observation(_TeamInbox):
    api_version: int
    seed: int
    tick: int
    time: float
    dt: float
    policy_dt: float
    remaining: float
    team: str
    controlled_ids: list[str]
    robots: list[RobotState]
    enemies: list[RobotState]
    vision: dict[str,dict[str,Any]]
    arena: dict[str, Any]
    bases: dict[str, Any]
    teams: dict[str, Any]
    zones: dict[str, Any]
    projectiles: list[dict[str, Any]]
    specs: dict[str, Any]
    referee: dict[str, Any]

API_VERSION = 3
TEAM_SLOTS = (1, 3, 4, 6, 7)
TEAM_SIZE = 5
RULES_PROFILE = "rmuc-2027-v1.0.0-arena-v1"
# RMUC 2027: heavy 1, infantry 3 and 4, aerial system 6, sentry 7.
TEAM_SLOTS_2027 = (1, 3, 4, 6, 7)
TEAM_SIZE_2027 = 5
RULES_PROFILE_2027 = "rmuc-2027-v1.0.0-arena-v1"

def _load_rules():
    """Rules and reference field of every season available to this runtime."""
    here = Path(__file__).resolve().parent
    for file in (here / "rules.json", *(parent / "configs/rules.json" for parent in here.parents)):
        if file.is_file():
            bundle=json.loads(file.read_text(encoding="utf-8"))
            seasons={}
            if 'rulebook' in bundle and 'rmuc2027' in bundle['maps']:
                seasons['2027']=(bundle['rulebook'],bundle['maps']['rmuc2027'])
            return seasons,bundle.get('robot_designs',{})
    file = here / "arena.json"
    if file.is_file():
        bundle=json.loads(file.read_text(encoding="utf-8"))
        return {bundle.get('season','2027'):(bundle["rulebook_rules"],bundle["map"])},bundle.get('robot_designs',{})
    raise RuntimeError("Public rules reference missing from SDK runtime")

_SEASONS, DESIGNS = _load_rules()
RULES_BY_SEASON = {season: value[0] for season, value in _SEASONS.items()}
ARENAS_BY_SEASON = {season: value[1] for season, value in _SEASONS.items()}
# RULES and RULES_2027 refer to the same RMUC 2027 rulebook.
RULES = RULES_BY_SEASON.get('2027') or next(iter(RULES_BY_SEASON.values()))
RULES_2027 = RULES_BY_SEASON.get('2027')
_REFERENCE_ARENA = ARENAS_BY_SEASON.get('2027') or next(iter(ARENAS_BY_SEASON.values()))


def season_of(observation):
    """The RMUC 2027 rules season of this match."""
    return observation.get('season') or observation.get('arena', {}).get('season') or '2027'


def rules(observation):
    """The full public rules of the observation's season."""
    return RULES_BY_SEASON[season_of(observation)]


def action(move=(0.0, 0.0), aim=None, fire=False, skill=False, pitch=0.04, upgrade=None, **operations):
    """Game action. Move is world x/y; aerial may also supply z intent.

    Additional operations: profile, power, altitude, buy_ammo, remote_heal,
    revive, deploy, support, pick_unit, assembly, assembly_insert,
    assembly_translate, assembly_rotate, assembly_rotate2, assembly_confirm, rune_start, rescue,
    yaw_rate, brake, leg_height, jump, lean, arm, messages (all control modes).
    rescue=<own slot> makes the heavy robot grip and right that tipped robot;
    arm={"tip":[x,y,z],"dir":[dx,dy,dz],"frame":"world"|"robot"} or
    {"joints":[7 values]} or {"stow":True} drives the heavy robot arm directly.
    The profile is selectable only on the first decision (tick 0).
    """
    if skill or upgrade is not None:
        raise ValueError("Unsupported action; use the documented game operations")
    allowed = {"profile", "power", "altitude", "buy_ammo", "remote_heal", "revive", "deploy", "support", "pick_unit", "assembly", "assembly_insert", "assembly_translate", "assembly_rotate", "assembly_confirm", "rune_start", "heading", "crouch", "rescue", "yaw_rate", "brake", "leg_height", "jump", "lean", "arm", "arm2", "pickup_arm", "transform", "messages",
               "install_launcher", "assembly_core", "assembly_lift", "assembly_flip", "assembly_rotate2", "posture", "posture_enhanced", "drones", "sentry_cmd"}
    if not set(operations) <= allowed:
        raise ValueError("Unknown action operation")
    # A move from `navigate` that backs a chassis off the road bumps keeps its heading.
    if getattr(move, "heading", None) is not None and not {"heading", "yaw_rate"} & set(operations):
        operations["heading"] = move.heading
    return {"move": list(move), "aim": aim, "pitch": pitch, "fire": fire, **operations}


def message(data, to="all"):
    """Build a team message: a teammate instance or broadcast in any control mode.

    Pass it via action(messages=[message(...)]) and read obs['messages'] on
    the next decision; Team mode can read obs['message_inboxes'][slot]. Each payload is JSON, <=512 bytes; at most four sends
    per decision, still within the robot action's total 1 KiB budget.
    """
    recipient = str(to)
    if recipient != "all" and not (recipient.isascii() and recipient.isdigit() and 0 < int(recipient) <= 2**32-1):
        raise ValueError("Message recipient must be all or an own robot slot")
    if len(json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()) > 512:
        raise ValueError("Message data must be <=512 bytes")
    return {"to": recipient, "data": data}


def referee_info(observation: Observation) -> dict[str, Any]:
    """Read the referee snapshot delivered with this decision, without network I/O.

    Both teams' health is public. Enemy position, aim, ammunition, heat,
    economy, source code and private strategy state are not exposed here.
    Event IDs can be retained in strategy state to deduplicate broadcasts.
    """
    return observation["referee"]


def _design(robot):
    return DESIGNS.get(str(robot.get("model_slot") or robot.get("slot")), {})


def is_42(robot):
    """True for 42 mm launchers (RMUC 2027 heavy robot)."""
    return robot["role"] == "heavy"


def muzzle_speed(robot):
    if robot["role"] == "heavy":
        return _design(robot).get("muzzle_speed_mps", 15.6)
    return _design(robot).get("muzzle_speed_mps", 25.0)


def aim_height(target):
    if "aim_z" in target:
        return target["aim_z"]
    if target.get("kind") == "rune_target":
        return target["z"]
    if "armor_open" in target:
        return target.get("z",0) + ((1.05 if RULES.get('terrain_geometry',{}).get('chapter4_dimensions') else .8) if target.get("kind") == "outpost" else .225 if target["armor_open"] else 1.043)
    return target.get("z", 0) + target.get('armor_center_z_m',.50)


def _rotate(q,vector):
    x,y,z,w=q;vx,vy,vz=vector
    tx=2*(y*vz-z*vy);ty=2*(z*vx-x*vz);tz=2*(x*vy-y*vx)
    return (vx+w*tx+y*tz-z*ty,vy+w*ty+z*tx-x*tz,vz+w*tz+x*ty-y*tx)


def _aim_target(robot,target):
    if 'aim_z' in target:
        return (target['x'],target['y'],target['aim_z'])
    if target.get('armor_plates'):
        candidates=[p for p in target['armor_plates'] if not p['heavy_only'] or is_42(robot)]
        if target.get('armor_open') and target.get('kind')!='outpost':
            lower=[p for p in candidates if p.get('weakspot')]
            if lower:candidates=lower
        def score(plate):
            point=plate['center'];normal=plate['normal']
            delta=(robot['x']-point[0],robot['y']-point[1],robot.get('z',0)+robot.get('muzzle_height_m',.45)-point[2])
            length=max(.001,math.sqrt(sum(v*v for v in delta)))
            return sum(a*b for a,b in zip(normal,delta))/length
        if candidates:return tuple(max(candidates,key=score)['center'])
    if target.get('chassis_size_m') and target.get('quaternion') and target.get('role')!='aerial':
        size=target['chassis_size_m'];q=target['quaternion'];best=None
        for face in range(4):
            angle=face*math.pi/2;nx,ny=math.cos(angle),math.sin(angle)
            distance_=(size[0] if face%2==0 else size[1])/2-.032
            height=target.get('armor_center_z_m',.265)
            if face%2:height=target.get('armor_side_center_z_m',height)
            offset=_rotate(q,(nx*distance_,ny*distance_,height))
            point=(target['x']+offset[0],target['y']+offset[1],target.get('z',0)+offset[2])
            normal=_rotate(q,(nx*math.cos(math.pi/12),ny*math.cos(math.pi/12),math.sin(math.pi/12)))
            difference=(robot['x']-point[0],robot['y']-point[1],robot.get('z',0)+robot.get('muzzle_height_m',.45)-point[2])
            score=sum(a*b for a,b in zip(normal,difference))/max(.001,math.sqrt(sum(v*v for v in difference)))
            if best is None or score>best[0]:best=(score,point)
        return best[1]
    return (target['x'],target['y'],aim_height(target))


_AIM_CACHE={}
_AIM_CACHE_LOCK=Lock()
def _ballistic_solution(robot,target,speed=None):
    speed=speed or muzzle_speed(robot);point=_aim_target(robot,target)
    dx,dy=point[0]-robot['x'],point[1]-robot['y'];distance_=max(.01,math.hypot(dx,dy));ux,uy=dx/distance_,dy/distance_
    inherited=robot.get('vx',0)*ux+robot.get('vy',0)*uy
    pivot=_rotate(robot.get('quaternion',(0,0,0,1)),(0,0,robot.get('muzzle_height_m',.25 if robot['role']=='aerial' else .55)))
    base_z=robot.get('z',0)+pivot[2];reach=robot.get('muzzle_reach_m',.3)
    key=tuple(round(v,3) for v in (distance_,point[2]-base_z,speed,inherited,robot.get('vz',0),reach))+(is_42(robot),)
    with _AIM_CACHE_LOCK:
        cached=_AIM_CACHE.get(key)
        if cached is not None:return cached
    radius=.02125 if is_42(robot) else .0084;mass=.0445 if is_42(robot) else .0032
    drag=.5*1.225*.47*math.pi*radius*radius/mass;dt=1/120
    def height(angle):
        x=reach*math.cos(angle);z=base_z+reach*math.sin(angle);vx=speed*math.cos(angle)+inherited;vz=speed*math.sin(angle)+robot.get('vz',0)
        for _ in range(540):
            v=math.hypot(vx,vz);ax=-drag*v*vx;az=-9.81-drag*v*vz
            xn=x+vx*dt+.5*ax*dt*dt;zn=z+vz*dt+.5*az*dt*dt
            if xn>=distance_:return z+(zn-z)*(distance_-x)/max(1e-8,xn-x)
            x,z=xn,zn;vx+=ax*dt;vz+=az*dt
            if vx<=0 or z<base_z-8:return None
        return None
    low,high=-math.pi/12,math.pi/4;h=height(high)
    reachable=h is not None and h>=point[2]
    if reachable:
        for _ in range(13):
            mid=(low+high)/2;value=height(mid)
            if value is None or value<point[2]:low=mid
            else:high=mid
    result=((low+high)/2 if reachable else math.pi/4,reachable)
    # Integration stays outside the lock. Publish atomically so parallel RL
    # controllers cannot evict the same oldest entry or race a cache hit.
    with _AIM_CACHE_LOCK:
        cached=_AIM_CACHE.get(key)
        if cached is not None:return cached
        if len(_AIM_CACHE)>=256:_AIM_CACHE.pop(next(iter(_AIM_CACHE)))
        _AIM_CACHE[key]=result
        return result


def aim_pitch(robot,target,speed=None):
    """Lower ballistic arc with drag, muzzle geometry and inherited velocity."""
    return _ballistic_solution(robot,target,speed)[0]


def ballistic_reachable(robot,target):
    return _ballistic_solution(robot,target)[1]


def can_fire(robot):
    if 0.0<robot.get('transform_progress',0.0)<1.0:return False
    if robot.get("drones") is not None and robot["role"]=="aerial":
        return False
    # The 2027 heavy robot fires only with its modular launcher on board.
    if robot["role"]=="heavy" and robot.get("launcher") not in ("fitted","installed"):
        return False
    if not robot["alive"] or robot.get("weak") or robot.get("overheated") or robot.get("heat_permanent_lock") or robot.get("overspeed_locked") or robot.get("overspeed_permanent_lock") or robot.get("launcher_fault") or robot.get("red_card"):
        return False
    heat = 100 if is_42(robot) else 10
    if robot.get("posture")=="attack" and robot.get("posture_enhanced"):heat=0
    return robot["ammo"]>0 and robot["heat"]+heat<=robot["max_heat"]+1e-8


def remote_heal_cost(elapsed_seconds):
    return 50 + math.ceil(elapsed_seconds/60*20)


def instant_revive_cost(robot, elapsed_seconds):
    return math.ceil(elapsed_seconds/60)*80 + robot["level"]*20


def intercept(robot, target, speed=None):
    """Lead a moving visible target; strategies can improve this prediction."""
    speed=speed or muzzle_speed(robot)
    time=distance(robot,target)/speed
    point=dict(target)
    point["x"] += target.get("vx",0)*time
    point["y"] += target.get("vy",0)*time
    return point


def tipped_teammates(observation):
    """Own ground robots lying on a side or roof (body up-axis below 60 degrees) that need
    a heavy robot. Serial-leg robots (``hip_angle`` present) right themselves while powered."""
    return [r for r in observation['robots'] if r.get('tipped')
            and not ('hip_angle' in r and r.get('alive') and r.get('powered', True))]


def rescue_point(engineer, target, standoff=.62):
    """Where the heavy robot should stop to grip a tipped robot: on the line from the
    robot toward the heavy robot, close enough for the arm (about 0.85 m tool reach)."""
    dx,dy=engineer['x']-target['x'],engineer['y']-target['y'];d=math.hypot(dx,dy)
    if d<1e-6:dx,dy,d=1.0,0.0,1.0
    return (target['x']+dx/d*standoff,target['y']+dy/d*standoff)


def sight_block_point(engineer, viewer, protected, height=None):
    """Arm tip that blocks `viewer`'s camera line to `protected` (both robot dicts):
    the point on that line nearest the heavy robot's mast, for action(arm={'tip': ...})."""
    eye=(viewer['x'],viewer['y'],viewer['z']+viewer.get('muzzle_height_m',.45)+.08)
    goal=(protected['x'],protected['y'],protected['z']+(height if height is not None else protected.get('armor_center_z_m',.3)))
    d=[g-e for g,e in zip(goal,eye)];length=sum(v*v for v in d) or 1e-9
    t=((engineer['x']-eye[0])*d[0]+(engineer['y']-eye[1])*d[1])/(d[0]*d[0]+d[1]*d[1] or 1e-9)
    t=max(0.0,min(1.0,t))
    return [eye[k]+d[k]*t for k in range(3)]


def fly_slope_lane(arena, team):
    """Fly-slope geometry at the end of `team`'s highway: lip (low end, flush with the
    highway) and crest of the 17 degree ramp, which climbs toward the opponent's half,
    and the landing pad across the 650 mm gap. Two-wheel-legged robots can also jump
    the gap in reverse, from the landing pad up onto the crest."""
    ramp=next(t for t in arena['terrain'] if t.get('id')==team+'_fly_slope')
    landing=next(t for t in arena['terrain'] if t.get('id')==team+'_fly_landing')
    xs=[p[0] for p in landing['polygon']];ys=[p[1] for p in landing['polygon']]
    return {'lip':ramp['ramp_start'],'crest':ramp['ramp_end'],'lip_height':ramp['base_elevation'],
            'crest_height':ramp['elevation'],'landing':[sum(xs)/len(xs),sum(ys)/len(ys)],'landing_height':landing['elevation']}


def toward(robot, point):
    dx, dy = point[0] - robot["x"], point[1] - robot["y"]
    distance = math.hypot(dx, dy)
    return (dx / distance, dy / distance) if distance > 0.05 else (0.0, 0.0)


def aim_at(robot, target):
    point=_aim_target(robot,target);dx,dy=point[0]-robot['x'],point[1]-robot['y'];time=math.hypot(dx,dy)/muzzle_speed(robot)
    return math.atan2(dy-robot.get('vy',0)*time,dx-robot.get('vx',0)*time)


def distance(a, b):
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def nearest_enemy(robot, observation):
    enemies = [r for r in observation["enemies"] if r["alive"] and r.get("targetable",r.get("role")!="aerial")]
    return min(enemies, key=lambda r: (distance(robot, r), r["id"]), default=None)


def navigate(robot, point, arena):
    """Small obstacle-aware helper; strategies can replace it with pathfinding.

    Follow the cheapest collision-free waypoint graph around expanded wall
    corners. The public geometry is identical for every strategy. Pass the
    result to `action`: a move backing a chassis off the road bumps also holds
    its heading there.
    """
    import heapq

    if robot["role"] == "aerial":
        if robot.get("drones") is not None:
            return (0.0, 0.0)  # a 2027 hangar is fixed; steer its aerial robots with `drones`
        lane=arena["flight_lanes"][robot["team"]]
        return _physical_toward(robot,(max(lane["min_x"],min(point[0],lane["max_x"])),lane["center_y"]))
    if arena.get("layout_revision"):
        return _chapter4_navigate(robot,point,arena)
    margin = robot.get("radius", 0.28) + 0.08
    for platform in arena.get("terrain", []):
        if platform.get("kind") != "platform":
            continue
        inside_goal = platform["x"] <= point[0] <= platform["x"] + platform["w"] and platform["y"] <= point[1] <= platform["y"] + platform["h"]
        inside_robot = platform["x"] <= robot["x"] <= platform["x"] + platform["w"] and platform["y"] <= robot["y"] <= platform["y"] + platform["h"]
        if inside_goal and not inside_robot:
            ramps = [t for t in arena["terrain"] if t.get("kind") == "ramp" and abs(t.get("elevation", 0) - platform["elevation"]) < .01]
            if ramps:
                ramp = min(ramps, key=lambda t: math.hypot(robot["x"] - (t["x"] + t["w"] / 2), robot["y"] - (t["y"] + t["h"] / 2)))
                in_ramp = ramp["x"] - .1 <= robot["x"] <= ramp["x"] + ramp["w"] + .1 and ramp["y"] <= robot["y"] <= ramp["y"] + ramp["h"]
                if not in_ramp:
                    point = (ramp["x"] + .1 if ramp["direction"] > 0 else ramp["x"] + ramp["w"] - .1, max(ramp["y"] + margin, min(point[1], ramp["y"] + ramp["h"] - margin)))
    obstacles=list(arena["obstacles"])
    for zone in arena.get("forbidden",[]):
        if zone["kind"] in ("base","pad","highway","dart_station") or zone.get("owner") not in (None,robot["team"]):
            obstacles.append(zone)
    walls = [{"x": w["x"] - margin, "y": w["y"] - margin,
              "w": w["w"] + margin * 2, "h": w["h"] + margin * 2}
             for w in obstacles]

    def clear(a, b):
        for w in walls:
            lo, hi = 0.0, 1.0
            for p, d, low, high in ((a[0], b[0] - a[0], w["x"], w["x"] + w["w"]),
                                    (a[1], b[1] - a[1], w["y"], w["y"] + w["h"])):
                if abs(d) < 1e-9:
                    if not low < p < high:
                        lo = 2
                        break
                else:
                    aa, bb = sorted(((low - p) / d, (high - p) / d))
                    lo, hi = max(lo, aa), min(hi, bb)
            if lo < hi and hi > 0 and lo < 1:
                return False
        return True

    start, goal = (robot["x"], robot["y"]), tuple(point)
    if clear(start, goal):
        return toward(robot, goal)
    nodes = [start, goal]
    for w in walls:
        for x in (w["x"] - 0.03, w["x"] + w["w"] + 0.03):
            for y in (w["y"] - 0.03, w["y"] + w["h"] + 0.03):
                if margin < x < arena["width"] - margin and margin < y < arena["height"] - margin:
                    nodes.append((x, y))
    queue, best = [(0.0, 0, [])], {0: 0.0}
    while queue:
        cost, node, path = heapq.heappop(queue)
        if cost > best[node]:
            continue
        if node == 1:
            return toward(robot, nodes[path[0]])
        for i, candidate in enumerate(nodes):
            if i == node or not clear(nodes[node], candidate):
                continue
            distance_ = math.dist(nodes[node], candidate)
            new_cost = cost + distance_
            if new_cost < best.get(i, float("inf")):
                best[i] = new_cost
                heapq.heappush(queue, (new_cost, i, path + [i]))
    return (0.0, 0.0)


def point_in_polygon(x, y, points):
    inside=False
    for a,b in zip(points,points[1:]+points[:1]):
        cross=(x-a[0])*(b[1]-a[1])-(y-a[1])*(b[0]-a[0])
        if abs(cross)<1e-8 and min(a[0],b[0])-1e-8<=x<=max(a[0],b[0])+1e-8 and min(a[1],b[1])-1e-8<=y<=max(a[1],b[1])+1e-8:return True
        if (a[1]>y)!=(b[1]>y) and x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]:inside=not inside
    return inside


def _point_in_prepared_polygon(x,y,edges):
    inside=False
    for ax,ay,by,dx,dy,x0,x1,y0,y1 in edges:
        cross=(x-ax)*dy-(y-ay)*dx
        if abs(cross)<1e-8 and x0<=x<=x1 and y0<=y<=y1:return True
        if (ay>y)!=(by>y) and x<dx*(y-ay)/dy+ax:inside=not inside
    return inside


def _terrain_surface(t,x,y):
    if t['kind']=='ramp' and t.get('ramp_start'):
        a,b=t['ramp_start'],t['ramp_end'];dx,dy=b[0]-a[0],b[1]-a[1]
        p=max(0,min(1,((x-a[0])*dx+(y-a[1])*dy)/(dx*dx+dy*dy)))
        return t.get('base_elevation',0)+(t['elevation']-t.get('base_elevation',0))*p
    if t['kind']=='bumps':
        p=((x-t['x'])/t['bump_pitch'])%1
        return t['elevation']+t['bump_height']*(1-math.cos(math.tau*p))/2
    if t.get('edge_ramp_m',0):
        points=t['polygon']
        d=min(abs((x-a[0])*(b[1]-a[1])-(y-a[1])*(b[0]-a[0]))/math.dist(a,b) for a,b in zip(points,points[1:]+points[:1]))
        return t.get('base_elevation',0)+(t['elevation']-t.get('base_elevation',0))*min(1,d/t['edge_ramp_m'])
    return t['elevation']


def terrain_height(arena,x,y):
    """Ground datum for chapter 4 polygons, sloped faces and road bumps."""
    value=0
    for t in arena['terrain']:
        if t['x']<=x<=t['x']+t['w'] and t['y']<=y<=t['y']+t['h'] and (not t.get('polygon') or point_in_polygon(x,y,t['polygon'])):
            value=max(value,_terrain_surface(t,x,y))
    return value+max(0,min(y,arena['height']-y))*math.tan(math.radians(arena.get('floor_slope_deg',0)))


_NAV_GRAPHS={}


def _prepare_chapter4(arena,team,folded=True,clearance=.36):
    # Roofs are overhead and do not block the route, except that in 2027 only a folded robot
    # fits under the 250-260 mm tunnel roofs; others route around the tunnels.
    under=folded or arena.get('season')!='2027'
    margin=max(.36,round(clearance,3))
    key=(arena['layout_revision'],team,under,margin)
    if key in _NAV_GRAPHS:return _NAV_GRAPHS[key]
    walls=[w for w in arena['obstacles'] if 'roof' not in w.get('id','') or (not under and '_tunnel_roof' in w['id'])]
    # The highway no-go area behind a fly slope is entered only by flying the ramp (R58).
    walls += [w for w in arena['forbidden'] if w['kind'] in ('base','pad','highway','dart_station') or w.get('owner') not in (None,team)]
    boxes=[]
    for w in walls:
        a=w.get('rotation',0);boxes.append((w['x']+w['w']/2,w['y']+w['h']/2,w['w']/2+margin,w['h']/2+margin,math.cos(a),math.sin(a)))
    # RMUC 2027 outposts stand on the central deck beside its kerb: route around their pedestals.
    if arena.get('season')=='2027':
        for x,y in arena['outposts'].values():boxes.append((x,y,.34+margin,.34+margin,1.0,0.0))
    bounded_boxes=[]
    for box in boxes:
        cx,cy,hw,hh,c,s=box
        ex,ey=abs(c)*hw+abs(s)*hh,abs(s)*hw+abs(c)*hh
        bounded_boxes.append((*box,cx-ex-1e-9,cx+ex+1e-9,cy-ey-1e-9,cy+ey+1e-9))
    # A small spatial index keeps height checks bounded during decision calls.
    tiles={}
    for t in arena['terrain']:
        x0,y0=t['x'],t['y'];x1,y1=x0+t['w'],y0+t['h'];polygon=t['polygon']
        rectangular=(len(polygon)==4 and set(map(tuple,polygon))=={(x0,y0),(x0,y1),(x1,y0),(x1,y1)}
                     and all(a[0]==b[0] or a[1]==b[1] for a,b in zip(polygon,polygon[1:]+polygon[:1])))
        edges=tuple((a[0],a[1],b[1],b[0]-a[0],b[1]-a[1],min(a[0],b[0])-1e-8,
                     max(a[0],b[0])+1e-8,min(a[1],b[1])-1e-8,max(a[1],b[1])+1e-8)
                    for a,b in zip(polygon,polygon[1:]+polygon[:1]))
        entry=(t,x0,y0,x1,y1,rectangular,edges)
        for x in range(math.floor(t['x']),math.ceil(t['x']+t['w'])+1):
            for y in range(math.floor(t['y']),math.ceil(t['y']+t['h'])+1):tiles.setdefault((x,y),[]).append(entry)
    floor_slope=math.tan(math.radians(1.5))
    @lru_cache(maxsize=32768)
    def height(x,y):
        h=0
        for t,x0,y0,x1,y1,rectangular,edges in tiles.get((math.floor(x),math.floor(y)),[]):
            if x0-1e-8<=x<=x1+1e-8 and y0-1e-8<=y<=y1+1e-8:
                if (rectangular and x0<=x<=x1 and y0<=y<=y1) or _point_in_prepared_polygon(x,y,edges):
                    h=max(h,_terrain_surface(t,x,y))
        return h+min(y,15-y)*floor_slope
    def inside(point,keep=False):
        """Inflated boxes containing a point; a robot pushed into one must still be able to leave.

        With `keep`, each box maps to the point's clearance from the bare obstacle (None
        when the point is within it), so `clear` lets the robot leave or pass that obstacle
        without closing in on it."""
        found={}
        for k,(cx,cy,hw,hh,c,s) in enumerate(boxes):
            ax,ay=(point[0]-cx)*c+(point[1]-cy)*s,-(point[0]-cx)*s+(point[1]-cy)*c
            if abs(ax)<hw and abs(ay)<hh:
                gap=max(abs(ax)-hw,abs(ay)-hh)+margin
                found[k]=gap-.01 if gap>.01 else None
        return found if keep else set(found)
    def clear(a,b,ignore=(),strict=True):
        dx,dy=b[0]-a[0],b[1]-a[1]
        xmin,xmax=min(a[0],b[0]),max(a[0],b[0])
        ymin,ymax=min(a[1],b[1]),max(a[1],b[1])
        for k,(cx,cy,hw,hh,c,s,bx0,bx1,by0,by1) in enumerate(bounded_boxes):
            if k in ignore:
                kept=ignore[k] if isinstance(ignore,dict) else None
                if kept is None:continue
                hw,hh=hw-margin+kept,hh-margin+kept
                ex,ey=abs(c)*hw+abs(s)*hh,abs(s)*hw+abs(c)*hh
                bx0,bx1,by0,by1=cx-ex-1e-9,cx+ex+1e-9,cy-ey-1e-9,cy+ey+1e-9
            if xmax<bx0 or xmin>bx1 or ymax<by0 or ymin>by1:
                continue
            ax,ay=(a[0]-cx)*c+(a[1]-cy)*s,-(a[0]-cx)*s+(a[1]-cy)*c
            vx,vy=dx*c+dy*s,-dx*s+dy*c;lo,hi=0,1
            for p,d,half in ((ax,vx,hw),(ay,vy,hh)):
                if abs(d)<1e-9:
                    if not -half<p<half:lo=2;break
                else:
                    aa,bb=sorted(((-half-p)/d,(half-p)/d));lo,hi=max(lo,aa),min(hi,bb)
            if lo<hi and hi>0 and lo<1:return False
        length=math.hypot(dx,dy);count=max(1,math.ceil(length/.2))
        # Strict checks also sample both sides of the footprint, so a route never
        # runs one side of the chassis off a ramp edge or platform cliff.
        old_center=height(a[0],a[1])
        if strict and length>1e-9:
            sx,sy=-dy/length*margin,dx/length*margin
            old_left=height(a[0]+sx,a[1]+sy);old_right=height(a[0]-sx,a[1]-sy)
            for i in range(1,count+1):
                px,py=a[0]+dx*i/count,a[1]+dy*i/count
                center=height(px,py)
                if abs(center-old_center)>.13:return False
                left=height(px+sx,py+sy);right=height(px-sx,py-sy)
                if abs(left-old_left)>.13 or abs(right-old_right)>.13 or max(center,left,right)-min(center,left,right)>.13:return False
                old_center,old_left,old_right=center,left,right
        else:
            for i in range(1,count+1):
                center=height(a[0]+dx*i/count,a[1]+dy*i/count)
                if abs(center-old_center)>.13:return False
                old_center=center
        return True
    nodes=[tuple(p) for p in arena['navigation']['nodes']]
    # Only short wall-end detours are needed in addition to the public ramp nodes.
    for cx,cy,hw,hh,c,s in boxes:
        if hw>2:continue
        for x in (-hw-.04,hw+.04):
            for y in (-hh-.04,hh+.04):
                p=(cx+x*c-y*s,cy+x*s+y*c)
                if .37<p[0]<27.63 and .37<p[1]<14.63:nodes.append(p)
    edges=[[] for _ in nodes]
    for i,a in enumerate(nodes):
        for j in range(i):
            if math.dist(a,nodes[j])<7 and clear(a,nodes[j]):
                d=math.dist(a,nodes[j]);edges[i].append((j,d));edges[j].append((i,d))
    value=(nodes,edges,clear,inside)
    _NAV_GRAPHS[key]=value
    return value


def _physical_toward(robot,point):
    direction=toward(robot,point);distance_=math.hypot(robot['x']-point[0],robot['y']-point[1])
    speed=robot.get('speed_reference_mps')
    if not speed:return direction
    scale=min(1,distance_*2/speed)
    return (direction[0]*scale,direction[1]*scale)


def _chapter4_navigate(robot,point,arena):
    graph=_prepare_chapter4(arena,robot['team'],robot.get('transform_progress',0.0)>=.99,
                            robot.get('radius',.28)+.04);nodes,_,clear,inside=graph
    start=(robot['x'],robot['y']);goal=tuple(point)
    # A goal inside the safety margin may relax that margin, never remove the
    # physical obstacle: the final segment must approach its reachable side.
    near_goal={k:max(0., gap or 0.) for k,gap in inside(goal,True).items()}
    guarded=inside(start,True)
    for strict in (True,False):
        for near_start in ((guarded,set(guarded)) if guarded else (guarded,)):
            ignore=dict(near_start) if isinstance(near_start,dict) else dict.fromkeys(near_start)
            for k,gap in near_goal.items():
                if k not in ignore:
                    ignore[k]=gap
                elif ignore[k] is not None:
                    ignore[k]=min(ignore[k],gap)
            if clear(start,goal,ignore,strict):return _square_to_bumps(robot,goal,arena)
            step=_route(graph,start,goal,near_start,near_goal,strict,.3 if arena.get('season')=='2027' else 0)
            if step is not None:return _square_to_bumps(robot,nodes[step],arena)
    return (0.0,0.0)


def _navigate_action_legacy(robot, point, observation, state):
    """Stateful chassis navigation. Keep state separately for each robot.

    Returns action fields, not a serialized command. Geometry and feedback come
    exclusively from public observations. Never share state between deployments.
    """
    arena = observation['arena']
    now = observation['time']
    model = str(robot.get('model_slot', robot.get('slot')))
    design = _design(robot)
    capabilities = design.get('capabilities', {})
    tunnel_chassis = bool(design.get('transform_allowed') and capabilities.get('tunnel'))
    stair_chassis = bool(capabilities.get('jump') and capabilities.get('step_climb_m', 0) > 0)
    serial_legs = stair_chassis and design.get('leg_type') == 'serial'
    parallel_legs = stair_chassis and not serial_legs
    if not robot.get('alive', True):
        state.clear()
        return {'move': (0., 0.)}
    if now < state.get('time', -1):
        state.clear()
    state['time'] = now
    if state.get('goal') != list(point) and robot.get('grounded', True):
        if 'tunnel' not in state and not (serial_legs and 'entry' in state):
            state.pop('entry', None)
            state.pop('failed', None)
        state['goal'] = list(point)
    if tunnel_chassis:
        flip = robot['team'] == 'blue'
        local = lambda p: (28-p[0], 15-p[1]) if flip else tuple(p)
        x, y = local((robot['x'], robot['y']))
        gx, gy = local(point)
        entry = state.get('tunnel')
        if entry is None and x < 10 and y < 13.4 and 10.5 < gx < 17.5 and not state.get('tunnel_done'):
            entry = state['tunnel'] = {'phase': 'approach', 'since': now}
        if entry and now-entry['since'] > 25:
            state.pop('tunnel', None)
            state['tunnel_done'] = True
            entry = None
        if entry:
            heading = -math.pi/2 if flip else math.pi/2
            if entry['phase'] == 'approach':
                target = local((9.32, 11.7))
                gap=math.dist((robot['x'],robot['y']),target)
                speed=math.hypot(robot.get('vx',0.),robot.get('vy',0.))
                decel=max(.3,min(design.get('brake_force_max_n',100.)/max(1.,robot.get('mass_kg',25.)),
                                 design.get('tire_friction',.5)*9.81)*.5)
                if gap>.12:
                    move=navigate(robot,target,arena)
                    if gap<max(2.,speed*speed/(2*decel)+.5):
                        safe_speed=min(1.2,max(.12,gap*1.2))
                        if speed>safe_speed+.15:
                            return {'move':(0.,0.),'brake':True}
                        norm=math.hypot(*move)
                        if norm>1e-8:
                            scale=min(norm,safe_speed/max(.1,robot.get('speed_reference_mps',3.9)))
                            fields={'move':(move[0]/norm*scale,move[1]/norm*scale)}
                            if getattr(move,'heading',None) is not None:fields['heading']=move.heading
                            return fields
                    return {'move':move}
                if speed>.2:
                    return {'move':(0.,0.),'brake':True}
                entry['phase'] = 'fold'
            error = abs((robot['heading']-heading+math.pi) % math.tau-math.pi)
            if robot.get('transform_progress', 0) < .99 or (entry['phase'] == 'fold' and error > .08):
                return {'move': (0., 0.), 'heading': heading, 'transform': True, 'brake': True}
            if entry['phase'] == 'fold' and abs(x-9.32) > .15 and y > 12.:
                return {'move': (0., .25 if flip else -.25),
                        'heading': heading, 'transform': True, 'brake': True}
            if entry['phase'] == 'fold' and (abs(x-9.32) > .06 or abs(robot.get('vx', 0)) > .12):
                correction = max(-.18, min(.18, (9.32-x)))
                return {'move': (-correction if flip else correction, 0.),
                        'heading': heading, 'transform': True, 'brake': True}
            entry['phase'] = 'cross'
            if y < 13.55:
                correction = max(-.3, min(.3, (9.32-x)*1.5))
                move = (-correction, -.6) if flip else (correction, .6)
                return {'move': move, 'heading': heading, 'transform': True, 'power': 120}
            state.pop('tunnel', None)
            state['tunnel_done'] = True
        roofs = [w for w in arena['obstacles'] if '_tunnel_roof' in w.get('id', '')]
        under = any(w['x']-.4 <= robot['x'] <= w['x']+w['w']+.4 and
                    w['y']-.4 <= robot['y'] <= w['y']+w['h']+.4 for w in roofs)
        return {'move': navigate(robot, point, arena), 'transform': under}
    if stair_chassis and arena.get('layout_revision') == 'chapter4-2027-v1':
        # The central bands and rims are explicit public geometry. Only this
        # chassis has a validated design for this approach; other legs keep theirs.
        flip = robot['team'] == 'blue'
        local = lambda p: (28-p[0], 15-p[1]) if flip else tuple(p)
        x, y = local((robot['x'], robot['y']))
        gx, gy = local(point)
        entry = state.get('entry')
        if entry is None and x < 10.3 and 10.5 < gx < 17.5 and 3 < gy < 12 and not state.get('failed'):
            entry = state['entry'] = {'phase': 'approach', 'since': now, 'jumps': 0, 'last_jump': -10.}
        if entry:
            if parallel_legs:
                landed = (entry['jumps'] > 0 and x > 10.3
                          and robot.get('grounded') and robot.get('upright', 0) > .95
                          and abs(robot.get('vz', 0)) < .2
                          and robot.get('z', 0) - entry.get('takeoff_z', robot.get('z', 0)) > .15)
                if landed:
                    entry.setdefault('landed_since', now)
                    if now-entry['landed_since'] >= .3:
                        state.pop('entry', None)
                        return {'move': (-.35 if flip else .35, 0.),
                                'heading': math.pi if flip else 0., 'jump': 0.}
                    return {'move': (0., 0.), 'brake': True, 'jump': 0.}
                entry.pop('landed_since', None)
                if (entry['jumps'] > 0 and entry['last_jump'] >= 0 and now-entry['last_jump'] > 2.
                        and x < 10.65 and robot.get('grounded')):
                    entry['phase'] = 'retreat'
                if entry['phase'] == 'retreat':
                    if x > 8.35:
                        return {'move': (.35 if flip else -.35, 0.),
                                'heading': math.pi if flip else 0., 'jump': 0.}
                    if entry['jumps'] >= 3:
                        state.pop('entry', None)
                        state['failed'] = True
                        return {'move': (0., 0.), 'brake': True}
                    entry.update(phase='approach', since=now, last_jump=-10.)
                    entry.pop('stable_since', None)
            if serial_legs and robot.get('tipped'):
                state.pop('entry', None)
                return {'move': (0., 0.), 'brake': True}
            if serial_legs and not robot.get('grounded', True):
                entry['phase'] = 'airborne'
                return {'move': (0., 0.), 'heading': math.pi if flip else 0., 'jump': 0.}
            if serial_legs and entry['phase'] == 'airborne':
                entry.update(phase='settle', landed_at=now)
            if serial_legs and entry['phase'] == 'settle':
                if now-entry['landed_at'] > 4.:
                    state.pop('entry', None)
                    state['failed'] = True
                    return {'move': (0., 0.), 'brake': True, 'jump': 0.}
                if now-entry['landed_at'] < 1. or robot.get('upright', 1.) < .95:
                    return {'move': (0., 0.), 'brake': True, 'jump': 0.}
                entry['phase'] = 'align'
            if x > 10.65 and robot.get('grounded'):
                state.pop('entry', None)
            elif now-entry['since'] > 18 or entry['jumps'] >= 3 and now-entry['last_jump'] > 2:
                state.pop('entry', None)
                state['failed'] = True
            else:
                lane = 7.5
                if entry['phase'] == 'approach':
                    target = local((8.0 if parallel_legs else 8.5, lane))
                    if math.dist((robot['x'], robot['y']), target) > .2:
                        return {'move': navigate(robot, target, arena)}
                    entry['phase'] = 'align'
                heading = math.pi if flip else 0.
                error = abs((robot['heading']-heading+math.pi) % math.tau-math.pi)
                fields = {'heading': heading, 'leg_height': _design(robot).get('leg_min_m', .12), 'brake': True}
                if error > .12 or (serial_legs and robot.get('upright', 1.) < .95):
                    fields['move'] = (0., 0.)
                    return fields
                fields['move'] = _physical_toward(robot, local((11., lane)))
                fields['power'] = 180
                launch_time = design.get('jump_duration_s', .16) + design.get('jump_tuck_s', .35) + .1
                if parallel_legs and 0 <= now-entry['last_jump'] < launch_time:
                    # Preserve run-up momentum without accelerating at the tyre
                    # patch while the legs extend; that traction pitches the hull.
                    fields.update(move=(0., 0.), brake=False, jump=0.)
                    return fields
                if robot.get('grounded') and abs(robot.get('vz', 0)) < .2 and robot.get('upright', 1.) > .95:
                    entry.setdefault('stable_since', now)
                else:
                    entry.pop('stable_since', None)
                ready = (not parallel_legs or entry['jumps'] == 0 or
                         now-entry.get('stable_since', now) >= .4)
                # Allow the chassis to rise before its nose reaches the stair rim.
                # At 10 Hz a threshold at 8.65 can issue the jump only at 8.8.
                takeoff_x = 8.55 if parallel_legs else 8.65
                forward_speed = -robot.get('vx', 0.) if flip else robot.get('vx', 0.)
                crossing_before_next_decision = (parallel_legs and x >= 8.1 and
                    observation.get('policy_dt', .1) >= .49 and
                    x + max(0., forward_speed)*observation['policy_dt'] > takeoff_x)
                if ready and entry['jumps']<3 and (x > takeoff_x or crossing_before_next_decision) and robot.get('grounded') and now-entry['last_jump'] > _design(robot).get('jump_cooldown_s', .9)+.15:
                    fields['jump'] = 1.
                    if parallel_legs:
                        fields.update(move=(0., 0.), brake=False)
                    entry.update(last_jump=now, jumps=entry['jumps']+1, takeoff_z=robot.get('z', 0.))
                return fields
    return {'move': navigate(robot, point, arena)}


class _HeldMove(tuple):
    """A `navigate` move that must not turn the chassis; `action` sends its heading."""
    def __new__(cls,move,heading):
        value=super().__new__(cls,move);value.heading=heading
        return value


def _square_to_bumps(robot,target,arena,lookahead=.75,tolerance=math.radians(12)):
    """Drive toward `target`, but turn on the spot first if the course runs onto road bumps
    within `lookahead` while the chassis does not face it, backing off the humps first if it
    is too close to turn. A chassis that strafes across the humps stalls, and it cannot turn
    once it stands on them. Robots cross the humps at full command: slowing there for a
    waypoint just past them stalls them as well."""
    move=_physical_toward(robot,target);heading=robot.get('heading')
    x,y=robot['x'],robot['y'];dx,dy=target[0]-x,target[1]-y;length=math.hypot(dx,dy)
    if length<1e-6 or not any(move):return move
    for t in arena['terrain']:
        if t['kind']!='bumps':continue
        x0,y0,x1,y1=t['x'],t['y'],t['x']+t['w'],t['y']+t['h']
        if x0<=x<=x1 and y0<=y<=y1:
            return move if x0<=target[0]<=x1 and y0<=target[1]<=y1 else (dx/length,dy/length)
        if heading is None:continue
        lo,hi=0,min(length,lookahead)
        for p,d,a,b in ((x,dx/length,x0,x1),(y,dy/length,y0,y1)):
            if abs(d)<1e-9:
                if not a<=p<=b:lo=math.inf
            else:
                s0,s1=sorted(((a-p)/d,(b-p)/d));lo,hi=max(lo,s0),min(hi,s1)
        if lo>hi:continue
        error=abs((math.atan2(dy,dx)-heading+math.pi)%math.tau-math.pi)
        # A two-wheeled chassis drives either way round; others turn to face the motion.
        if robot.get('chassis_type')=='wheel_legged':error=min(error,math.pi-error)
        if error>tolerance:
            # Turning would push a wheel sideways up the first hump, so a chassis nosed onto
            # it at an angle (a teammate rounding the kerb end beside it yaws it there) is
            # pinned: back straight out, holding its heading, until its turning circle clears.
            ax,ay=x-min(max(x,x0),x1),y-min(max(y,y0),y1);gap=math.hypot(ax,ay)
            if gap<robot.get('radius',0):return _HeldMove((ax/gap*.6,ay/gap*.6),heading)
            # A creeping command still sets the chassis yaw goal, so it turns nearly in place.
            return (dx/length*.02,dy/length*.02)
    return move


_NAV_ROUTE_COSTS={}
_NAV_ROUTE_LOCK=Lock()


def _route_costs(graph,goal,near_goal,strict):
    """Distances from public waypoints to a fixed goal, reused across decisions."""
    import heapq
    nodes,edges,clear,_=graph
    key=(id(graph),clear,goal,frozenset(near_goal.items()) if isinstance(near_goal,dict) else frozenset(near_goal),strict)
    with _NAV_ROUTE_LOCK:
        cached=_NAV_ROUTE_COSTS.get(key)
        if cached is not None:return cached
    distances={i:math.dist(p,goal) for i,p in enumerate(nodes) if clear(p,goal,near_goal,strict)}
    queue=[(cost,i) for i,cost in distances.items()];heapq.heapify(queue)
    while queue:
        cost,index=heapq.heappop(queue)
        if cost!=distances[index]:continue
        for other,d in edges[index]:
            if cost+d<distances.get(other,float('inf')):
                distances[other]=cost+d;heapq.heappush(queue,(cost+d,other))
    # Dijkstra stays outside the critical section. Recheck after computing so
    # concurrent hits/evictions are atomic without serializing route searches.
    with _NAV_ROUTE_LOCK:
        cached=_NAV_ROUTE_COSTS.get(key)
        if cached is not None:return cached
        if len(_NAV_ROUTE_COSTS)>=128:_NAV_ROUTE_COSTS.pop(next(iter(_NAV_ROUTE_COSTS)))
        _NAV_ROUTE_COSTS[key]=distances
        return distances


def _route(graph,start,goal,near_start,near_goal,strict,reached=0):
    """First visible waypoint on the cheapest complete route, or None. Waypoints closer
    than `reached` count as passed, so a robot standing on one heads for the next instead
    of creeping on the spot (on a 2027 slope it would slide back)."""
    nodes,_,clear,_=graph
    costs=_route_costs(graph,goal,near_goal,strict)
    candidates=sorted((math.dist(start,nodes[i])+cost,i) for i,cost in costs.items())
    if reached:
        ahead=[c for c in candidates if math.dist(start,nodes[c[1]])>=reached]
        candidates=ahead or candidates
    # The cost of every complete route is already known; test visibility in
    # increasing cost order instead of scanning every start/goal pair again.
    for _,i in candidates:
        if clear(start,nodes[i],near_start,strict):return i
    return None


# Prepare static, public geometry during strategy initialization, before the
# decision timer begins. This creates no threads, external reads or hidden data.
for _arena in ARENAS_BY_SEASON.values():
    if _arena.get('layout_revision'):
        for _team in ('red','blue'):
            for _folded in ((False,True) if _arena.get('season')=='2027' else (True,)):_prepare_chapter4(_arena,_team,_folded)


def resource_pickup_point(robot,arena):
    """A station approach that stays inside the zone and within the chosen arm workspace."""
    station=next(f for f in arena['field_geometry']['fixtures'] if f['kind']=='resource' and f['team']==robot['team'])
    point=station['pick_point'];direction=station['pick_dir'];reach=.48 if robot.get('arm_count',1)==2 else .60
    return (point[0]-direction[0]*reach,point[1]-direction[1]*reach)


def assembly_work_point(robot,arena):
    """Dock inside the plinth, away from its ramp edge, while keeping the core in reach."""
    zone=next(z for z in arena['zones'] if z['id']==robot['team']+'_assembly')
    origin=arena['assembly_frames'][robot['team']]['origin']
    dx,dy=origin[0]-zone['x'],origin[1]-zone['y']
    length=math.hypot(dx,dy) or 1.0
    docking_distance=.25*math.sqrt(2)
    return (zone['x']+docking_distance*dx/length,zone['y']+docking_distance*dy/length)


# ---- RMUC 2027 helpers ------------------------------------------------------------------

def workbench_point(robot, arena):
    """Where a 2027 heavy robot parks to fit its modular 42 mm launcher (install_launcher=True)."""
    bench=next(f for f in arena['field_geometry']['fixtures'] if f['kind']=='workbench' and f['team']==robot['team'])
    return tuple(bench['dock_point'])


def launcher_ready(robot):
    """The heavy robot's launcher is unlocked (by level-3 assembly or the time rule) but not yet fitted."""
    return robot.get('launcher') in ('unlocked','installing')


def position(observation, name):
    """A control position: {'owner','kind','control':'full'|'temporary','controller',...}.
    Names: red_outpost, red_base, blue_outpost, blue_base."""
    return observation.get('positions',{}).get(name)


def controls(observation, name, team=None):
    """True while `team` (default: own) controls the position, fully or temporarily."""
    p=position(observation,name);team=team or observation['team']
    return bool(p) and p['controller']==team


def control_degree(observation, team=None):
    team=team or observation['team']
    return observation['teams'].get(team,{}).get('control') or 0.0


def zone_center(arena, zone_id):
    """A point on the buff card: its reference point, which lies inside even oddly shaped
    cards such as the long central-highland strips."""
    z=next(z for z in arena['zones'] if z['id']==zone_id)
    return (z['x'],z['y'])


def drones(robot):
    """The aerial robots of a 2027 aerial system (hangar), with state, pose, battery and integrity."""
    return robot.get('drones') or []


def strike_plan(drone, plate, standoff=.9, team_heights=None):
    """A ramming run at an armor plate: hover at `standoff` in front of its face, then
    fly straight into it. Returns ('approach', point) until lined up, then ('strike', direction).
    Use as drones={'1':{'goto':point}} or drones={'1':{'move':direction}}."""
    n=plate['normal'];c=plate['center']
    flat=math.hypot(n[0],n[1]) or 1.0
    ux,uy=n[0]/flat,n[1]/flat
    stand=(c[0]+ux*standoff,c[1]+uy*standoff,c[2])
    lateral=abs((drone['x']-c[0])*(-uy)+(drone['y']-c[1])*ux)
    ahead=(drone['x']-c[0])*ux+(drone['y']-c[1])*uy
    if lateral<.05 and abs(drone['z']-c[2])<.06 and ahead>.15:
        return 'strike',(-ux,-uy,(c[2]-drone['z'])*2)
    return 'approach',stand


def hostile_plates(observation, kind):
    """Armor plates a 2027 aerial robot may strike: 'outpost' (any outpost not owned by
    this team and still standing) or 'base' (the enemy base)."""
    team=observation['team'];enemy='blue' if team=='red' else 'red'
    if kind=='base':
        base=observation['bases'][enemy]
        return base.get('armor_plates',[]) if base['hp']>0 else []
    plates=[]
    for owner,outpost in observation['outposts'].items():
        if outpost['hp']>0 and outpost.get('owner',owner)!=team:plates+=outpost.get('armor_plates',[])
    return plates


def _stationary_teammate_detour(robot, point, fields, observation, state):
    """A local waypoint around observed parked teammates; never changes the goal.

    Terrain manoeuvres own the controls once alignment/folding/jumping starts.
    This is shared by real-physics deployments and simplified training.
    """
    peers = [r for r in observation.get('robots', [])
             if r.get('id') != robot.get('id') and r.get('team') == robot.get('team')
             and r.get('role') != 'aerial' and math.hypot(r.get('vx',0.),r.get('vy',0.)) < .15]
    move = fields.get('move',(0.,0.))
    if (not peers or not robot.get('alive',True) or not robot.get('grounded',True)
            or robot.get('role') == 'aerial' or any(fields.get(k) for k in ('jump','brake','transform'))
            or any(state.get(k,{}).get('phase','approach')!='approach' for k in ('entry','tunnel'))
            or math.hypot(*move)<.05 or observation.get('arena',{}).get('layout_revision')!='chapter4-2027-v1'):
        state.pop('teammate_detour',None)
        state.pop('teammate_wait_since',None)
        return fields
    start=(robot['x'],robot['y']);radius=robot.get('radius',.28)
    _,_,clear,inside=_prepare_chapter4(observation['arena'],robot['team'],robot.get('transform_progress',0)>=.99,radius+.04)
    near=inside(start,True)
    def safe(target):
        if not clear(start,target,near):return False
        dx,dy=target[0]-start[0],target[1]-start[1];length=dx*dx+dy*dy
        for peer in peers:
            px,py=peer['x'],peer['y'];gap=radius+peer.get('radius',.28)+.04
            t=max(0.,min(1.,((px-start[0])*dx+(py-start[1])*dy)/max(1e-9,length)))
            closest=math.hypot(start[0]+t*dx-px,start[1]+t*dy-py)
            initial=math.hypot(start[0]-px,start[1]-py)
            if closest<min(gap,initial)-1e-6:return False
        return True
    now=observation.get('time',0.)
    detour=state.get('teammate_detour')
    if detour and (detour['goal']!=list(point) or now-detour['since']>4.
                   or math.dist(start,detour['point'])<.12 or not safe(detour['point'])):
        state.pop('teammate_detour',None);detour=None
    if detour is None:
        if math.hypot(robot.get('vx',0.),robot.get('vy',0.))>.12:
            state.pop('teammate_wait_since',None)
            return fields
        state.setdefault('teammate_wait_since',now)
        if now-state['teammate_wait_since']<.6:
            return fields
        norm=math.hypot(*move);direction=(move[0]/norm,move[1]/norm)
        blockers=[]
        lookahead=.15
        for peer in peers:
            dx,dy=peer['x']-start[0],peer['y']-start[1]
            forward=dx*direction[0]+dy*direction[1]
            lateral=abs(dx*direction[1]-dy*direction[0]);gap=radius+peer.get('radius',.28)+.12
            if 0<forward<gap+lookahead and lateral<gap:
                blockers.append((forward,gap,peer))
        if not blockers:return fields
        _,gap,_=min(blockers,key=lambda v:v[0])
        candidates=[]
        for sign in (1.,-1.):
            target=(start[0]-direction[0]*.1-direction[1]*gap*sign,
                    start[1]-direction[1]*.1+direction[0]*gap*sign)
            if safe(target):candidates.append(target)
        if not candidates:return fields
        target=min(candidates,key=lambda p:math.dist(p,point))
        detour=state['teammate_detour']={'point':target,'goal':list(point),'since':now}
    result=dict(fields);result['move']=_physical_toward(robot,detour['point'])
    if 'heading' in result:
        result['heading']=math.atan2(detour['point'][1]-start[1],detour['point'][0]-start[0])
    return result


def navigate_action(robot, point, observation, state, *, locomotion_intent=None):
    """Opt-in sampled local avoidance using only the controller's observation."""
    if observation.get('navigation_mode') == 'adaptive' or locomotion_intent is not None:
        return _adaptive_navigation(robot, point, observation, state, locomotion_intent)
    if observation['time']<state.get('time',-1):state.clear()
    design=_design(robot)
    armed_parallel=(robot.get('alive',True) and design.get('chassis_type')=='wheel_legged'
                    and design.get('arm_count',0)>0 and design.get('leg_type')!='serial'
                    and design.get('capabilities',{}).get('jump') and design.get('capabilities',{}).get('step_climb_m',0)>0)
    if armed_parallel and observation['arena'].get('layout_revision')=='chapter4-2027-v1':
        flip=robot['team']=='blue'
        gx,gy=(28-point[0],15-point[1]) if flip else point
        central=10.5<gx<17.5 and 3<gy<12
        if central:state['heavy_low_stance']=True
        else:
            state.pop('heavy_stage',None);state.pop('heavy_stage_ready',None)
        x=28-robot['x'] if flip else robot['x']
        if (central and observation.get('policy_dt',.1)>=.49 and x<10.3 and robot.get('grounded',True)
                and not state.get('entry',{}).get('jumps') and not state.get('heavy_stage_ready')):
            # At 2 Hz, a moving chassis can hit the stair before the next jump
            # decision. Establish a stopped run-up origin first. The low leg
            # stance also keeps an arm-carrying chassis balanced on the way.
            staging=(20.,7.5) if flip else (8.,7.5)
            if (math.dist((robot['x'],robot['y']),staging)>=.07 or
                    math.hypot(robot.get('vx',0.),robot.get('vy',0.))>=.2 or
                    not robot.get('grounded',True) or robot.get('upright',1.)<.95):
                fields=navigate_action(robot,staging,observation,state.setdefault('heavy_stage',{}))
                fields.setdefault('leg_height',design.get('leg_min_m',.12))
                state['time']=observation['time']
                return fields
            state['heavy_stage_ready']=True
    fields = _navigate_action_legacy(robot, point, observation, state)
    if (observation.get('navigation_mode') == 'predictive' and
            not any(k in state for k in ('entry', 'tunnel')) and
            not any(fields.get(k) for k in ('jump', 'brake', 'transform')) and
            robot.get('grounded', True) and robot.get('role') != 'aerial'):
        fields.update(_trajectory_move(robot, point, observation, state))
    fields = _stationary_teammate_detour(robot, point, fields, observation, state)
    if armed_parallel and state.get('heavy_low_stance'):
        # Keep the stable transport stance after landing; raising the loaded
        # chassis immediately can undo an otherwise successful arrival.
        fields.setdefault('leg_height',design.get('leg_min_m',.12))
    return predictive_navigation(robot, fields, observation, state)


def predictive_navigation(robot, fields, observation, state):
    fields = dict(fields)
    if observation.get('navigation_mode') != 'predictive':
        return fields
    state['limited'] = False
    # Terrain manoeuvres own their heading and thrust; do not steer them sideways.
    if (robot.get('role') == 'aerial' or robot.get('tipped') or
            not robot.get('grounded', True) or fields.get('jump') or
            fields.get('transform') or fields.get('brake')):
        return fields
    move = fields.get('move', (0., 0.))
    magnitude = math.hypot(*move)
    if magnitude < .001:
        return fields
    arena = observation['arena']
    graph = _prepare_chapter4(arena, robot['team'], robot.get('transform_progress', 0) >= .99,
                             robot.get('radius', .28)+.04)
    clear, inside = graph[2:]
    start = (robot['x'], robot['y'])
    near = inside(start, True)
    reference = robot.get('speed_reference_mps', 3.)
    design = _design(robot)
    mass = max(1., robot.get('mass_kg', design.get('mass_kg', 25.)))
    decel = max(.2, min(design.get('brake_force_max_n', 100.)/mass,
                        design.get('tire_friction', .5)*9.81)*.6)
    period = max(.05, observation.get('policy_dt', .1))
    current = (robot.get('vx', 0.), robot.get('vy', 0.))
    obstacles = [r for r in observation.get('robots', []) + observation.get('enemies', [])
                 if r.get('id') != robot.get('id') and r.get('role') != 'aerial']
    radius = robot.get('radius', .28)
    angle = math.atan2(move[1], move[0])
    best = None
    for offset in (0., -.261799, .261799, -.523599, .523599):
        for scale in (1., .75, .5):
            requested = reference*magnitude*scale
            target = (math.cos(angle+offset)*requested, math.sin(angle+offset)*requested)
            pos, velocity = start, current
            horizon = period + max(math.hypot(*current), requested)/decel
            valid = True
            for step in range(math.ceil(horizon/.05)):
                t = (step+1)*.05
                desired = target if t <= period else (0., 0.)
                dv = (desired[0]-velocity[0], desired[1]-velocity[1])
                length = math.hypot(*dv)
                gain = min(1., decel*.05/max(length, 1e-9))
                velocity = (velocity[0]+dv[0]*gain, velocity[1]+dv[1]*gain)
                nxt = (pos[0]+velocity[0]*.05, pos[1]+velocity[1]*.05)
                if not (radius < nxt[0] < arena['width']-radius and radius < nxt[1] < arena['height']-radius) or not clear(pos, nxt, near, observation.get('navigation_terrain_strict',True)):
                    valid = False
                    break
                # A parked peer may already lie inside the inflated safety
                # margin at spawn. Permit separation, never closer approach;
                # otherwise every escape command is rejected forever.
                if any(math.hypot(nxt[0]-r['x']-r.get('vx',0)*t,
                                  nxt[1]-r['y']-r.get('vy',0)*t)
                       < min(radius+r.get('radius',.28)+.08+.05*t,
                             math.hypot(start[0]-r['x'],start[1]-r['y'])
                             if math.hypot(r.get('vx',0.),r.get('vy',0.))<.15 else float('inf'))-1e-6
                       for r in obstacles):
                    valid = False
                    break
                pos = nxt
            if valid:
                score = scale - abs(offset)*.4 - abs(offset-state.get('avoid_offset',0))*.2
                if best is None or score > best[0]:
                    best = (score, offset, scale)
    if best is None:
        state['limited'] = True
        return {**fields, 'move': (0.,0.), 'brake': True, 'power': 0}
    _, offset, scale = best
    state['avoid_offset'] = offset
    state['limited'] = scale < 1 or offset != 0
    if state['limited']:
        fields.pop('power', None)
    fields['move'] = (math.cos(angle+offset)*magnitude*scale, math.sin(angle+offset)*magnitude*scale)
    return fields


def _complete_route(graph, start, goal):
    """Recover the route, allowing the reachable side of a near-obstacle goal."""
    nodes, edges, clear, inside = graph
    near = inside(start, True)
    near_goal = {k:max(0., gap or 0.) for k,gap in inside(goal, True).items()}
    def guards(a,b):
        ignored = dict(near) if a == start else {}
        if b == goal:
            for k,gap in near_goal.items():
                if k not in ignored: ignored[k]=gap
                elif ignored[k] is not None: ignored[k]=min(ignored[k],gap)
        return ignored
    if clear(start, goal, guards(start,goal), True):
        return [goal]
    costs = _route_costs(graph, goal, near_goal, True)
    index = _route(graph, start, goal, near, near_goal, True)
    if index is None:
        return []
    path, visited = [], set()
    while index not in visited:
        visited.add(index)
        path.append(nodes[index])
        if clear(nodes[index], goal, near_goal, True):
            path.append(goal)
            break
        choices = [(distance_ + costs.get(j, float('inf')), j)
                   for j, distance_ in edges[index] if costs.get(j, float('inf')) < costs[index]]
        if not choices:
            return []
        index = min(choices)[1]
    else:
        return []
    result, anchor = [], start
    while path:
        visible = [i for i, point in enumerate(path) if clear(anchor, point, guards(anchor,point), True)]
        if not visible:
            return []
        i = max(visible)
        anchor = path[i]
        result.append(anchor)
        path = path[i+1:]
    return result


def _trajectory_move(robot, goal, obs, state):
    graph = _prepare_chapter4(obs['arena'], robot['team'], robot.get('transform_progress',0) >= .99,
                             robot.get('radius', .28)+.04)
    start, goal = (robot['x'], robot['y']), tuple(goal)
    def segment_clear(point):
        ignored = graph[3](start,True)
        if point == goal:
            for k,gap in graph[3](goal,True).items():
                gap=max(0.,gap or 0.)
                if k not in ignored:ignored[k]=gap
                elif ignored[k] is not None:ignored[k]=min(ignored[k],gap)
        return graph[2](start,point,ignored,False)
    path = state.get('trajectory', [])
    # Keep a committed segment through small footprint-height changes near a
    # corner. Replanning it every tick can alternate the previous/next vertex.
    # The full route and predictive motion guard still check both wheel edges.
    if state.get('trajectory_goal') != list(goal) or not path or not segment_clear(path[0]):
        path = _complete_route(graph, start, goal)
        state['trajectory_goal'] = list(goal)
    while len(path)>1 and math.dist(start,path[0]) < .2 and segment_clear(path[1]):
        path = path[1:]
    state['trajectory'] = path
    if not path:
        return {'move': (0.,0.), 'brake': True}
    target = path[0]
    gap = math.dist(start,target)
    reference = robot.get('speed_reference_mps',3.)
    design = _design(robot)
    mass = max(1.,robot.get('mass_kg',design.get('mass_kg',25.)))
    decel = max(.2,min(design.get('brake_force_max_n',100.)/mass,design.get('tire_friction',.5)*9.81)*.6)
    # A half-second command must leave room to slow before a corner, not after it.
    period = max(.05,obs.get('policy_dt',.1))
    arrival = 0.
    if len(path)>1 and gap>1e-6:
        a = ((target[0]-start[0])/gap,(target[1]-start[1])/gap)
        length = math.dist(target,path[1])
        b = ((path[1][0]-target[0])/max(length,1e-9),(path[1][1]-target[1])/max(length,1e-9))
        turn = math.acos(max(-1.,min(1.,a[0]*b[0]+a[1]*b[1])))
        arrival = min(reference, math.sqrt(decel*.2/max(.05,math.sin(turn/2))))
    limit = max(0., math.sqrt((decel*period)**2+arrival**2+2*decel*max(0.,gap-.1))-decel*period)
    move = _square_to_bumps(robot,target,obs['arena'])
    if math.isclose(math.hypot(*move),.02,abs_tol=1e-8) and gap>.2:
        return {'move':move,'brake':True}
    # Intermediate corners need centimetre-level tracking: the general helper's
    # 5 cm dead zone can stop short of the next segment's visibility boundary.
    if len(path)>1 and gap<.2 and gap>1e-6:
        crawl=max(.12,min(.4,gap*2.))
        move=((target[0]-start[0])/gap*crawl/reference,
              (target[1]-start[1])/gap*crawl/reference)
    # Keep terrain-specific heading semantics and avoid changing a deliberate alignment.
    if isinstance(move,_HeldMove):
        return {'move': move, 'heading':move.heading}
    scale = min(1.,limit/max(reference*math.hypot(*move),1e-9))
    climbing=any(t['kind']=='ramp' and t.get('slope_deg',0)>=10
                 and point_in_polygon(robot['x'],robot['y'],t['polygon']) for t in obs['arena'].get('terrain',[]))
    if climbing and len(path)>1 and math.hypot(*move)>.001:
        # A proportional slowdown halfway up a steep entrance can request less
        # force than gravity. Carry the ramp crossing through its next vertex.
        norm=math.hypot(*move)
        scale=max(scale,.6/norm)
    on_bumps=any(t['kind']=='bumps' and t['x']<=start[0]<=t['x']+t['w']
                 and t['y']<=start[1]<=t['y']+t['h'] for t in obs['arena'].get('terrain',[]))
    if design.get('leg_type')=='serial' and any(move) and not on_bumps:
        scale=min(scale,1.5/max(reference*math.hypot(*move),1e-9))
    fields={'move': (move[0]*scale,move[1]*scale)}
    if climbing:fields['power']=120
    if design.get('arm_count',0)>0 and on_bumps:
        fields['power']=180
    return fields


def locomotion_context(robot, observation, state):
    """Public geometry and feedback shared by navigation and portable actors.

    Coordinates use the robot's actual frame, including rotated tunnel roofs.
    No referee state or backend-specific data is read here.
    """
    d = _design(robot)
    p = (robot['x'], robot['y'])
    radius = robot.get('radius', .28)
    roofs = []
    for wall in observation['arena'].get('obstacles', []):
        if 'tunnel_roof' not in wall.get('id', ''):
            continue
        angle = wall.get('rotation', 0.)
        axis = (math.cos(angle), math.sin(angle))
        center = (wall['x']+wall['w']/2, wall['y']+wall['h']/2)
        along = (p[0]-center[0])*axis[0]+(p[1]-center[1])*axis[1]
        across = -(p[0]-center[0])*axis[1]+(p[1]-center[1])*axis[0]
        gap = math.hypot(max(0., abs(along)-wall['w']/2-radius),
                         max(0., abs(across)-wall['h']/2-radius))
        roofs.append(dict(wall=wall, axis=axis, center=center, along=along,
                          across=across, gap=gap))
    roof = min(roofs, key=lambda r:r['gap']) if roofs else None
    under = [r for r in roofs if r['gap'] <= .04]
    standing_top = robot.get('z', 0.)+d.get('muzzle_z_m', .4)+.06
    can_unfold = not any(standing_top+.025 > r['wall'].get('bottom', 0.) for r in under)
    now = observation.get('time', 0.)
    cooldown = max(0., state.get('last_jump', -100.)+d.get('jump_cooldown_s', .9)-now)
    stable = (robot.get('grounded', True) and robot.get('upright', 1.) >= .95
              and abs(robot.get('vz', 0.)) < .2 and not robot.get('tipped', False))
    return dict(roof=roof, can_unfold=can_unfold, cooldown=cooldown, stable=stable,
                can_jump=bool(d.get('capabilities', {}).get('jump')),
                can_fold=bool(d.get('transform_allowed')),
                legged=d.get('chassis_type') == 'wheel_legged')


def _adaptive_navigation(robot, goal, obs, state, intent):
    """Capability-aware trajectory tracking with policy-owned skill decisions.

    Scripted users get the same controller with automatic skill requests. Policy
    users choose those requests; guards may stop motion but never invent a jump.
    Every state belongs to one robot and is serializable by the policy worker.
    """
    now = obs['time']
    if not robot.get('alive', True):
        state.clear()
        return {'move': (0., 0.)}
    if now < state.get('time', -1):
        state.clear()
    state['time'] = now
    d = _design(robot)
    ctx = locomotion_context(robot, obs, state)
    p = (robot['x'], robot['y'])
    speed = math.hypot(robot.get('vx', 0.), robot.get('vy', 0.))
    radius = robot.get('radius', .28)
    period = max(.05, obs.get('policy_dt', .1))
    reference = max(.1, robot.get('speed_reference_mps', d.get('speed_reference_mps', 3.)))
    decel = max(.3, min(d.get('brake_force_max_n', 100.)/max(1., robot.get('mass_kg', d.get('mass_kg', 25.))),
                         d.get('tire_friction', .5)*9.81)*.6)
    if ctx['stable']:
        state.setdefault('stable_since', now)
    else:
        state.pop('stable_since', None)
    if state.get('goal') != list(goal):
        previous_goal=state.get('goal')
        def beyond_central_approach(target):
            if target is None:return False
            x,y=target
            if robot['team']=='blue':x,y=obs['arena']['width']-x,obs['arena']['height']-y
            return x>10.5 and 3<y<12
        same_passage_destination=(beyond_central_approach(previous_goal)
                                  and beyond_central_approach(goal))
        state['goal'] = list(goal)
        state.pop('trajectory', None)
        # Complete an occupied passage or airborne manoeuvre before replanning.
        if ctx['can_unfold'] and robot.get('grounded', True):
            # Goal jitter beyond the same central approach must not turn a
            # safely exiting robot back through the passage it just crossed.
            if not same_passage_destination:
                state.pop('passage', None)
                state.pop('passed_passages', None)
            route=state.get('jump_route')
            same_crossing=(route is not None and (goal[0]-route['end'])*route['direction']>0
                and abs(goal[1]-route['lane'])<4.5)
            # A small tactical target change beyond the same step does not
            # invalidate a staged run-up. A target on this side still cancels it.
            if not same_crossing:state.pop('jump_route', None)
    fields = {'move': (0., 0.)}
    state['rejected'] = []
    state['waiting_peer_active'] = False
    phase = 'route'
    passage = state.get('passage')
    flip = robot['team'] == 'blue'
    local_x = obs['arena']['width']-p[0] if flip else p[0]
    local_goal = (obs['arena']['width']-goal[0], obs['arena']['height']-goal[1]) if flip else goal
    central = 10.5 < local_goal[0] < 17.5 and 3 < local_goal[1] < 12
    if ctx['can_fold'] and passage is None:
        roof = ctx['roof']
        # The road tunnel is the validated central approach. Other passages are
        # selected when the intended segment intersects their physical opening.
        candidates = [w for w in obs['arena'].get('obstacles', [])
                      if w.get('id') == robot['team']+'_road_tunnel_roof']
        if central and local_x < 10. and candidates:
            roof = locomotion_context({**robot, 'x':candidates[0]['x']+candidates[0]['w']/2,
                                       'y':candidates[0]['y']+candidates[0]['h']/2}, obs, state)['roof']
        if roof and roof['wall']['id'] not in state.get('passed_passages', []):
            wall, axis, center = roof['wall'], roof['axis'], roof['center']
            a = (p[0]-center[0])*axis[0]+(p[1]-center[1])*axis[1]
            b = (goal[0]-center[0])*axis[0]+(goal[1]-center[1])*axis[1]
            t = -a/(b-a) if abs(b-a)>1e-8 else -1.
            crossing = (p[0]+t*(goal[0]-p[0]), p[1]+t*(goal[1]-p[1]))
            lateral = abs(-(crossing[0]-center[0])*axis[1]+(crossing[1]-center[1])*axis[0])
            if (central and local_x < 10. and wall['id'] == robot['team']+'_road_tunnel_roof'
                    or 0 <= t <= 1 and lateral < wall['h']/2-radius):
                sign = 1. if a < 0. else -1.
                axis = [axis[0]*sign, axis[1]*sign]
                passage = state['passage'] = dict(center=center, axis=axis,
                    half=wall['w']/2, width=wall['h'], since=now, id=wall['id'])
    auto_fold = not ctx['can_unfold']
    want_jump = False
    if passage:
        c, axis = passage['center'], passage['axis']
        along = (p[0]-c[0])*axis[0]+(p[1]-c[1])*axis[1]
        lateral = -(p[0]-c[0])*axis[1]+(p[1]-c[1])*axis[0]
        clearance = passage['half']+radius+.12
        heading = math.atan2(axis[1], axis[0])
        error = abs((heading-robot.get('heading', heading)+math.pi)%math.tau-math.pi)
        approach = (c[0]-axis[0]*(clearance+.4), c[1]-axis[1]*(clearance+.4))
        queue_peer = next((r for r in obs.get('robots',[]) if r.get('id')!=robot.get('id')
            and r.get('team')==robot.get('team') and r.get('slot',99)<robot.get('slot',0)
            and r.get('role')!='aerial' and 'x' in r and 'y' in r
            and math.dist((r['x'],r['y']),approach)<radius+r.get('radius',.28)+.3),None)
        auto_fold = (abs(lateral)<passage['width']/2+.3 and
                     along > -clearance-max(.5, speed*(d.get('transform_time_s', .8)+period)))
        if along > clearance and ctx['can_unfold']:
            state.setdefault('passed_passages', []).append(passage['id'])
            state.pop('passage', None)
            passage = None
            phase = 'exit'
            auto_fold = False
        elif queue_peer is not None and along < -clearance:
            # Give the lower-numbered peer room to align at a shared entrance.
            # This changes only the local staging waypoint, never the policy goal.
            holding=(approach[0]-axis[0]*1.2,approach[1]-axis[1]*1.2)
            fields=_trajectory_move(robot,holding,obs,state)
            if math.dist(p,holding)<.18:fields.update(move=(0.,0.),brake=True)
            state['wait_for_peer']=queue_peer['id']
            state['waiting_peer_active']=True
            passage['since']=now
            phase='approach'
        elif now-passage['since'] > 25.:
            state.pop('passage', None)
            fields = {'move': (0., 0.), 'brake': True}
            state['rejected'].append('passage_timeout')
            phase = 'recover'
        elif (along < -clearance-.45 or
              abs(lateral)>.12 and (along < -clearance-.3 or abs(lateral)>passage['width']/2-radius)):
            phase = 'approach'
            fields = _trajectory_move(robot, approach, obs, state)
            fields = _stationary_teammate_detour(robot, approach, fields, obs, state)
        else:
            phase = 'cross' if along >= -clearance else 'fold'
            fields['heading'] = heading
            folded = robot.get('transform_progress', 0.) >= .99
            if not folded or error > .12:
                fields.update(move=(0., 0.), brake=True)
            else:
                # Preserve sufficient tyre force on the 15-degree entrance;
                # the physical drive already halves speed while folded.
                desired = reference*.6
                fields['power'] = 120
                correction = max(-.3, min(.3, -lateral*1.8))
                fields['move'] = (axis[0]*desired/reference-axis[1]*correction,
                                  axis[1]*desired/reference+axis[0]*correction)
    route = state.get('jump_route')
    if not passage and phase == 'route' and ctx['can_jump'] and d.get('capabilities', {}).get('step_climb_m', 0)>0:
        if central and local_x < 10.3 and now >= state.get('retry_after', -1.):
            terrain = {t.get('id'):t for t in obs['arena'].get('terrain', [])}
            band = terrain.get('central_'+robot['team']+'_band')
            rim = terrain.get('central_'+robot['team']+'_rim')
            if band and rim and route is None:
                direction = -1. if flip else 1.
                edge = band['x']+band['w'] if flip else band['x']
                lane = band['y']+band['h']/2
                route = state['jump_route'] = dict(edge=edge, lane=lane, direction=direction,
                    end=(rim['x']-radius-.25 if flip else rim['x']+rim['w']+radius+.25),
                    second_edge=(rim['x']+rim['w'] if flip else rim['x']),stage=0,since=now, staged=False)
        if route:
            sign, lane = route['direction'], route['lane']
            if (d.get('arm_count',0)>0 and route.get('stage',0)==0 and ctx['stable']
                    and now-state.get('last_jump',-100.)>.7 and state.get('last_jump',-100.)>route['since']
                    and (p[0]-route['edge'])*sign>.35 and (p[0]-route['second_edge'])*sign<.15):
                route.update(edge=route['second_edge'],stage=1,staged=False,since=now)
                route.pop('retreat',None)
            heading = math.pi if sign<0 else 0.
            along = (p[0]-route['edge'])*sign
            error = abs((heading-robot.get('heading', heading)+math.pi)%math.tau-math.pi)
            phase = 'runup'
            fields.update(heading=heading, leg_height=d.get('leg_min_m', .12))
            runup_length = .65 if route.get('stage')==1 else 1.2
            staging = (route['edge']-sign*runup_length, lane)
            if (not route.get('staged') and math.dist(p,staging)<.2 and speed<.25):
                route.update(staged=True,since=now)
            if (state.get('last_jump', -100.)>route['since'] and now-state['last_jump']>2.
                    and (p[0]-route['end'])*sign<-.15):
                route['retreat']=True
            if (p[0]-route['end'])*sign >= -.15 and ctx['stable']:
                phase = 'settle'
                fields.update(move=(0., 0.), brake=True)
                if now-state.get('stable_since', now) >= .3:
                    state.pop('jump_route', None)
                    phase = 'route'
            elif route.get('retreat'):
                phase='recover'
                fields.update(move=(-sign*.35,0.),brake=False)
                if along < -runup_length:
                    route.update(retreat=False,staged=False,since=now)
            elif not route.get('staged'):
                phase='approach'
                fields=_trajectory_move(robot,staging,obs,state)
                if math.dist(p,staging)<.2:fields.update(move=(0.,0.),brake=True)
            elif not robot.get('grounded', True):
                phase = 'airborne'
                fields.update(move=(0., 0.), brake=False)
            elif now-state.get('last_jump', -100.) < d.get('jump_duration_s', .16)+d.get('jump_tuck_s', .35)+.1:
                phase = 'launch'
                fields.update(move=(0., 0.), brake=False)
            elif abs(p[1]-lane) > .15 or error > .12:
                fields = _trajectory_move(robot, staging, obs, state)
                if math.dist(p, staging)<.2:
                    if abs(p[1]-lane)>.1:
                        gap=max(.01,math.dist(p,staging))
                        fields.update(move=((staging[0]-p[0])*.2/(gap*reference),
                            (staging[1]-p[1])*.2/(gap*reference)),brake=False)
                        fields.pop('heading',None)
                    else:fields.update(move=(0., 0.), heading=heading, brake=True)
            else:
                forward = sign*robot.get('vx', 0.)
                window = max(.65,radius+max(0., forward)*(d.get('jump_duration_s', .16)+period)+.08)
                if route.get('stage')==1:
                    window=radius+.15+max(0.,forward)*max(0.,period-.1)
                if d.get('leg_type')=='serial':
                    window=radius+.25+max(0.,forward)*max(.1,period)
                want_jump = -along <= window and along < -.05 and forward > .5
                fields['move'] = (sign, 0.)
                fields['power'] = 180
                if (along >= -radius-.08 and intent is not None and not want_jump
                        and state.get('last_jump',-100.)<=route['since']):
                    fields.update(move=(0., 0.), brake=True)
            if now-route['since'] > 25.:
                state.pop('jump_route', None)
                state['retry_after'] = now+5.
                phase = 'recover'
                fields.update(move=(-sign*.3, 0.), jump=0.)
            # Establish the jump stance while aligning, before accelerating.
            # Delaying compression until run-up creates a vertical transient
            # precisely when the launch window is crossed.
            if route.get('staged') or d.get('arm_count',0)>0:
                fields.setdefault('leg_height',d.get('leg_min_m',.12))
    if phase in ('route', 'exit'):
        fields = _trajectory_move(robot, goal, obs, state)
        if math.dist(p, goal)<.2:
            fields.update(move=(0., 0.), brake=True)
        else:
            fields = _stationary_teammate_detour(robot, goal, fields, obs, state)
            # The shared predictive filter owns ordinary travel only.
            fields = predictive_navigation(robot, fields, {**obs, 'navigation_mode':'predictive','navigation_terrain_strict':False}, state)
    if intent is None:
        fold = auto_fold
        jump = want_jump
        if ctx['legged'] and (d.get('arm_count',0)>0 or d.get('leg_type')=='serial' and d.get('capabilities',{}).get('step_climb_m',0)>0):
            fields.setdefault('leg_height',d.get('leg_min_m',.12))
    else:
        fold = state.get('fold_intent', robot.get('transform_progress', 0.)>.5)
        requested_fold=fold
        if intent['transform'] > .3:
            requested_fold = True
        elif intent['transform'] < -.3:
            requested_fold = False
        if requested_fold!=fold and ctx['can_fold']:
            if now-state.get('fold_since',-100.)>=d.get('transform_time_s',.8):
                fold=requested_fold
                state['fold_since']=now
            else:
                state['rejected'].append('transform_dwell')
        state['fold_intent'] = fold
        level = intent['jump'] > .5
        jump = level and not state.get('jump_held', False)
        state['jump_held'] = level
        if ctx['legged'] and 'leg_height' in intent:
            fields['leg_height'] = d['leg_min_m']+(intent['leg_height']+1)*.5*(d['leg_max_m']-d['leg_min_m'])
    if ctx['can_fold']:
        if not fold and not ctx['can_unfold']:
            fold = True
            state['rejected'].append('ceiling')
        fields['transform'] = fold
    if jump:
        ready = (ctx['can_jump'] and ctx['stable'] and ctx['cooldown']<=0
                 and robot.get('powered', True) and ctx['can_unfold']
                 and ('last_jump' not in state or now-state.get('stable_since', now)>=.2-1e-8))
        if ready and phase not in ('cross', 'fold', 'settle', 'recover'):
            fields.update(jump=1., move=(0., 0.), brake=False)
            state['last_jump'] = now
            phase = 'launch'
        else:
            state['rejected'].append('jump_unavailable')
    fields.setdefault('jump', 0.)
    if not robot.get('powered', True) or robot.get('tipped', False):
        fields.update(move=(0., 0.), jump=0., brake=False)
    state['phase'] = phase
    state['executed_skills'] = {k:fields.get(k) for k in ('transform', 'leg_height', 'jump')}
    return fields


def timed_assembly_action(robot, command, observation, state):
    """Fixed heavy opening for the opt-in timed assembly simulation mode."""
    if (not observation.get('simulation_assumptions', {}).get('timed_assembly')
            or robot.get('role') != 'heavy' or not robot.get('alive') or robot.get('weak')):
        return command
    if robot.get('launcher') in ('unlocked', 'installing', 'fitted', 'installed'):
        return command
    team = robot['team']
    done = observation['teams'][team].get('completed_difficulties', [])
    difficulty = next((d for d in (1, 2, 3) if d not in done), None)
    if difficulty is None:
        return command
    # Timed assembly only requires standing on the card, not reaching the
    # physical arm docking point. That point can pin a chassis against the core.
    zone = next(z for z in observation['arena']['zones'] if z['id'] == team + '_assembly')
    goal = (zone['x'], zone['y'])
    if math.hypot(robot['x']-goal[0], robot['y']-goal[1]) > zone['radius'] * .7:
        return navigate_action(robot, goal, observation, state.setdefault('navigation', {}))
    return {'move': [0, 0], 'brake': True, 'assembly': difficulty}
