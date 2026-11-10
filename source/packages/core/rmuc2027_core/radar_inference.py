"""Causal radar estimates from decoded legal wire packets, never match objects.

These are observed motion and candidate-associated combat proxies, not hardware
maximum speeds or measured aiming accuracy. speed_lb is a historical observed
peak, NOT a lower bound on present capability when hardware performance drifts.
robust_speed is a current, slowly updated upper-quantile motion proxy.
The estimator accepts only command bytes, source timestamps and public geometry.
"""
from collections import defaultdict, deque
import copy
import math
import statistics
import struct

SLOTS = (1, 3, 4, 7)
ENEMY_SLOTS = (3, 4, 7)
CONTENT_OVERVIEW, CONTENT_ALLIES, CONTENT_ENEMIES, CONTENT_PREDICTION = range(0x201, 0x205)
HEADER = struct.Struct('<BHIB')  # version, sequence, generated milliseconds, page
SPEED_RECORD = struct.Struct('<BBHHBBHBBBB')
TEAM_RECORD = struct.Struct('<HHhBB BHB')
ROBOT_RECORD = struct.Struct('<BBHHbBBBBBBHHBBB')
PREDICTION_RECORD = struct.Struct('<BBBBBBBBBBHHBHHBHHB')
assert (HEADER.size, SPEED_RECORD.size, TEAM_RECORD.size, ROBOT_RECORD.size, PREDICTION_RECORD.size) == (8, 14, 12, 20, 25)
REASONS = ('missing_hp', 'missing_position', 'incomplete_candidates', 'multiple_candidates',
    'health_reset', 'status_ambiguous', 'flight_or_geometry', 'insufficient_support')


def _q(value, scale=1., maximum=65535):
    return max(0, min(maximum, round(float(value)*scale)))


def _at(history, time, max_age):
    for row in reversed(history):
        if row[0] <= time+1e-8:
            return row if time-row[0] <= max_age else None
    return None


def _clear(a, b, arena):
    """Conservative XY occlusion; overhead roofs are not opaque ground walls."""
    for wall in arena.get('obstacles', []):
        if wall.get('bottom', 0.) > .5 or 'roof' in wall.get('id', ''): continue
        angle = wall.get('rotation', 0.); ca, sa = math.cos(angle), math.sin(angle)
        cx, cy = wall['x']+wall['w']/2, wall['y']+wall['h']/2
        points = [(ca*(p[0]-cx)+sa*(p[1]-cy), -sa*(p[0]-cx)+ca*(p[1]-cy)) for p in (a, b)]
        low, high = 0., 1.
        for axis, half in ((0, wall['w']/2), (1, wall['h']/2)):
            p, delta = points[0][axis], points[1][axis]-points[0][axis]
            if abs(delta) < 1e-9:
                if abs(p) >= half: low = 2.; break
            else:
                x, y = (-half-p)/delta, (half-p)/delta
                low, high = max(low, min(x, y)), min(high, max(x, y))
        if low < high and high > 0 and low < 1: return False
    return True


class ChassisClassifier:
    """Diagonal Gaussian class model with an explicit calibration entry point.

Calibration rows contain allowed trajectory features and offline class labels.
Labels never form part of the runtime input. Uncalibrated models stay unknown.
"""
    labels = ('wheel', 'fold', 'omni')

    def __init__(self, model=None):
        self.model = copy.deepcopy(model or {})

    @classmethod
    def fit(cls, rows, validated=False):
        groups = {label: [] for label in cls.labels}
        for row in rows:
            x = [float(v) for v in row['features']]
            if len(x) != 4 or not all(math.isfinite(v) for v in x): raise ValueError('Four finite motion features required')
            groups[row['label']].append(x)
        if any(len(x) < 4 for x in groups.values()): raise ValueError('At least four calibration tracks per class required')
        model = {'validated': bool(validated), 'classes': {}}
        for label, samples in groups.items():
            means = [statistics.mean(v[i] for v in samples) for i in range(4)]
            variances = [max(.01, statistics.mean((v[i]-means[i])**2 for v in samples)) for i in range(4)]
            model['classes'][label] = {'mean': means, 'variance': variances}
        return cls(model)

    def predict(self, features):
        if not self.model.get('validated'): return [1/3]*3, False
        return self._likelihood(features)

    def _likelihood(self, features):
        """Raw research scores; runtime callers must use validated predict()."""
        logp = []
        for label in self.labels:
            item = self.model['classes'][label]
            distance = sum((v-m)**2/var for v, m, var in zip(features, item['mean'], item['variance']))
            logp.append(-.5*distance-.5*sum(math.log(v) for v in item['variance']))
        temperature = max(1., float(self.model.get('temperature', 1.)))
        largest = max(logp); weights = [math.exp((v-largest)/temperature) for v in logp]; total = sum(weights)
        # Reject trajectories far outside every fitted class distribution.
        supported = any(sum((v-m)**2/var for v, m, var in zip(features, item['mean'], item['variance'])) < 25
            for item in self.model['classes'].values())
        return [w/total for w in weights], supported


def _robot_default():
    return dict(pressure_rate=0., pressure_confidence=0., pressure_n=0, pressure_age=999., pressure_valid=False,
        contribution_rate=0., contribution_confidence=0., contribution_n=0, contribution_exposure=0.,
        contribution_age=999., contribution_valid=False, local_advantage=0., local_confidence=0., local_age=999.,
        local_valid=False, candidate_count=0, candidates_complete=False, geometry_unique=False,
        unknown={reason: True for reason in REASONS})


class RadarInference:
    def __init__(self, team, classifier=None, position_error_m=.03):
        if team not in ('red', 'blue'): raise ValueError('Unknown radar team')
        self.team = team; self.other = 'blue' if team == 'red' else 'red'
        self.classifier = classifier or ChassisClassifier()
        self.position_error = max(0., float(position_error_m))
        self.positions = defaultdict(lambda: deque(maxlen=800))
        self.health = defaultdict(lambda: deque(maxlen=800))
        self.status = defaultdict(lambda: deque(maxlen=400)); self.ammo = {}; self.warnings = {}; self.last_packet = {}
        self.short = defaultdict(lambda: deque(maxlen=30))
        self.long = defaultdict(lambda: dict(rate=0., n=0, evidence=0., exposure=0., last=-1., confidence=0.))
        self.motion = defaultdict(lambda: dict(peak=0., estimate=0., last=-1., p=[1/3]*3, class_last=-1.))
        self.last_bin = 0; self.last_update = -1.; self.sequence = 0
        self.result = dict(enemies={}, team={}, robots={'ally': {}, 'enemy': {}}, predictions={})

    def _position(self, key, source, x, y):
        if not (math.isfinite(x) and math.isfinite(y)) or (x == 0 and y == 0): return
        rows = self.positions[key]
        if rows and source <= rows[-1][0]: return
        # Public, chassis-independent envelope; never consult a hidden model.
        if rows and source-rows[-1][0] < 2 and math.hypot(x-rows[-1][1], y-rows[-1][2]) > 20*(source-rows[-1][0])+.25:
            return
        rows.append((source, x, y))

    def _health(self, key, source, hp):
        rows = self.health[key]
        if rows and source <= rows[-1][0]: return
        if rows and rows[-1][1] <= 0 < hp:
            # Same chassis retains its long-term evidence; clear only a new life's trajectory/window.
            self.positions[key].clear(); self.short[key].clear()
        rows.append((source, float(hp)))

    def consume(self, command, payload, source_time):
        source = float(source_time)
        if not math.isfinite(source): raise ValueError('Finite source timestamp required')
        if source <= self.last_packet.get(command, -math.inf): return
        self.last_packet[command] = source
        if command == 0xA01:
            values = struct.unpack('<12H', payload)
            for slot, i in ((1, 0), (3, 2), (4, 3), (7, 5)):
                self._position((self.other, slot), source, values[2*i]/100, values[2*i+1]/100)
        elif command == 0xF001:
            values = struct.unpack('<B8H', payload)
            for i, slot in enumerate(SLOTS):
                if values[0] & (1 << i): self._position((self.team, slot), source, values[1+2*i]/100, values[2+2*i]/100)
        elif command == 0xA02:
            values = struct.unpack('<6H', payload)
            for slot, i in ((1, 0), (3, 2), (4, 3), (7, 5)): self._health((self.other, slot), source, values[i])
        elif command == 3:
            values = struct.unpack('<4Hh7H', payload)
            for slot, i in ((1, 0), (3, 2), (4, 3), (7, 5)): self._health((self.team, slot), source, values[i])
        elif command == 0xA03:
            values = struct.unpack('<5H', payload)
            for slot, i in ((1, 0), (3, 1), (4, 2), (7, 4)): self.ammo[(self.other, slot)] = (source, values[i])
        elif command == 0xA05:
            if len(payload) != 47: raise ValueError('Incorrect 0A05 length')
            for slot, i in ((1, 0), (3, 2), (4, 3), (7, 4)):
                heal, cooling, defense, vulnerability, attack = struct.unpack_from('<BHBBH', payload, i*7)
                self.status[(self.other, slot)].append((source, bool(heal or defense or vulnerability or attack or payload[36+i])))
        elif command == 0x204:
            heal, cooling, defense, vulnerability, attack = struct.unpack('<BHBBH', payload)
            self.status[(self.team, 7)].append((source, bool(heal or defense or vulnerability or attack)))
        elif command == 0x104 and any(payload):
            if self.warnings.get('payload') != bytes(payload): self.warnings = {'payload': bytes(payload), 'time': source}

    def _speed(self, key, now):
        rows = self.positions[key]; memory = self.motion[key]
        valid = [p for p in rows if now-10 <= p[0] <= now]
        segments = []; anchor = valid[0] if valid else None
        for point in valid[1:]:
            if point[0]-anchor[0] < .5-1e-8: continue
            dt = point[0]-anchor[0]
            if dt <= .65:
                distance = math.hypot(point[1]-anchor[1], point[2]-anchor[2])
                segments.append((point[0], max(0., distance-2*self.position_error)/dt, distance/dt,
                    (point[1]-anchor[1])/dt, (point[2]-anchor[2])/dt))
            anchor = point
        age = now-rows[-1][0] if rows else 999.
        moving = [s for s in segments if s[2] > .2]
        enough = len(segments) >= 12 and sum(s[2]*.5 for s in segments) >= 1 and age <= 2
        if enough and moving and moving[-1][0] > memory['last']:
            observed = sorted(s[2] for s in moving)[min(len(moving)-1, math.floor(.9*len(moving)))]
            memory['peak'] = max(memory['peak'], max(s[1] for s in segments))
            if memory['last'] < 0:
                memory['estimate'] = observed
            elif observed >= memory['estimate']:
                memory['estimate'] = .95*memory['estimate']+.05*observed
            elif len(moving) >= 12:
                # Sustained slower motion may adjust a proxy, never proves a hard
                # maximum. Stationary or brief tactical braking is insufficient.
                memory['estimate'] = .98*memory['estimate']+.02*observed
            memory['last'] = moving[-1][0]
        history = [p for p in rows if now-20 <= p[0] <= now]
        class_supported = False
        if enough and len(history) >= 40 and history[-1][0]-history[0][0] >= 19 and len(moving) >= 4:
            speeds = [s[2] for s in segments]
            turns = [abs(math.atan2(a[3]*b[4]-a[4]*b[3], a[3]*b[3]+a[4]*b[4]))
                for a, b in zip(moving, moving[1:])]
            features = [statistics.median(speeds), max(speeds), statistics.mean(turns) if turns else 0.,
                sum(s <= .2 for s in speeds)/len(speeds)]
            # Research recorder may copy these legal features and attach labels
            # outside this module. No private label ever enters this estimator.
            memory['features'], memory['features_time'] = features, now
            p, class_supported = self.classifier.predict(features)
            if class_supported:
                memory['p'] = [.98*a+.02*b for a, b in zip(memory['p'], p)]; memory['class_last'] = now
        ranked = sorted(memory['p'], reverse=True)
        unknown = not (class_supported and ranked[0] >= .7 and ranked[0]-ranked[1] >= .2 and now-memory['class_last'] <= 5)
        evidence_age = now-memory['last'] if memory['last'] >= 0 else 999.
        confidence = min(1., len(moving)/12)*math.exp(-max(0., evidence_age)/10) if enough else 0.
        return dict(speed_lb=memory['peak'], robust_speed=memory['estimate'], speed_valid=enough,
            speed_confidence=confidence, n_valid=len(segments), age=evidence_age,
            p_wheel=memory['p'][0], p_fold=memory['p'][1], p_omni=memory['p'][2], class_unknown=unknown)

    def calibration_features(self, slot, now):
        """Latest runtime motion features, without updating the causal filter."""
        memory = self.motion[(self.other, int(slot))]
        if 0 <= now-memory.get('features_time', -100.) <= .55:
            return list(memory['features'])
        return None

    def _combat_bin(self, end, arena):
        keys = [(side, slot) for side in (self.team, self.other) for slot in SLOTS]
        position = {k: _at(self.positions[k], end, .6) for k in keys}
        current = {k: _at(self.health[k], end, .6) for k in keys}
        previous = {k: _at(self.health[k], end-1, .6) for k in keys}
        rows = {}; candidates = {}; loss = {}
        for key in keys:
            reasons = dict.fromkeys(REASONS, False)
            a, b, p = previous[key], current[key], position[key]
            reasons['missing_hp'] = a is None or b is None
            reasons['missing_position'] = p is None
            health_samples = ([a] if a else [])+[h for h in self.health[key] if end-1 < h[0] <= end]
            reasons['health_reset'] = bool(a and b and (a[1] <= 0 or b[1] > a[1])) or any(
                later[1] > earlier[1] for earlier, later in zip(health_samples, health_samples[1:]))
            statuses = self.status.get(key, ())
            initial_status = _at(statuses, end-1, 2.)
            status_samples = ([initial_status] if initial_status else [])+[s for s in statuses if end-1 < s[0] <= end]
            reasons['status_ambiguous'] = any(s[1] for s in status_samples) or (
                key[0] == self.team and 0 <= end-self.warnings.get('time', -100.) < 3)
            opponents = [(s, n) for s, n in keys if s != key[0] and (current[(s, n)] is None or current[(s, n)][1] > 0)]
            complete = p is not None and all(position[k] is not None and current[k] is not None for k in opponents)
            reasons['incomplete_candidates'] = not complete
            possible = []
            if p:
                for enemy in opponents:
                    # Include historical source locations, not just a current nearest enemy.
                    path = [q for q in self.positions[enemy] if end-4 <= q[0] <= end]
                    if any(math.dist(p[1:], q[1:]) <= 40 and _clear(p[1:], q[1:], arena) for q in path): possible.append(enemy)
            candidates[key] = possible
            reasons['multiple_candidates'] = len(possible) > 1
            reasons['flight_or_geometry'] = not possible or not complete
            clean = not any(reasons[name] for name in ('missing_hp', 'missing_position', 'health_reset', 'status_ambiguous'))
            loss[key] = max(0., a[1]-b[1]) if clean else 0.
            rows[key] = dict(time=end, clean=clean, loss=loss[key], outgoing=None, candidates=len(possible),
                complete=complete, unique=len(possible) == 1 and complete, reasons=reasons, local=None)
        for victim, possible in candidates.items():
            row = rows[victim]
            if row['clean'] and row['unique']:
                shooter = possible[0]
                if rows[shooter]['clean'] and rows[shooter]['complete']:
                    rows[shooter]['outgoing'] = (rows[shooter]['outgoing'] or 0.)+loss[victim]
            elif len(possible) > 1 and loss[victim] > 0:
                for shooter in possible:
                    rows[shooter]['reasons']['multiple_candidates'] = True
        for key in keys:
            p = position[key]
            if p:
                nearby = [k for k in keys if position[k] and math.dist(p[1:], position[k][1:]) <= 10
                    and _clear(p[1:], position[k][1:], arena) and rows[k]['clean']]
                own = [k for k in nearby if k[0] == key[0]]; enemy = [k for k in nearby if k[0] != key[0]]
                if own and enemy:
                    outgoing = sum(loss[k] for k in enemy); incoming = sum(loss[k] for k in own)
                    rows[key]['local'] = (outgoing-incoming)/max(20., outgoing+incoming)
            self.short[key].append(rows[key])
            if rows[key]['outgoing'] is not None:
                memory = self.long[key]; value = rows[key]['outgoing']
                memory['rate'] = value if memory['n'] == 0 else .95*memory['rate']+.05*value
                elapsed = end-memory['last'] if memory['last'] >= 0 else 999.
                memory['evidence'] = memory['evidence']*math.exp(-elapsed/30)+1.
                memory['n'] += 1; memory['exposure'] += 1; memory['last'] = end
                # Geometry-associated proxy, never a certain true-source attribution.
                memory['confidence'] = .75*min(1., memory['evidence']/20)

    def _robot(self, key, now):
        result = _robot_default(); rows = [r for r in self.short[key] if now-r['time'] <= 30]
        clean = [r for r in rows if r['clean']]; local = [r for r in clean if r['local'] is not None]
        if rows:
            last = rows[-1]
            result.update(candidate_count=last['candidates'], candidates_complete=last['complete'],
                geometry_unique=last['unique'], unknown=dict(last['reasons']))
        if clean:
            age = now-clean[-1]['time']; conf = min(1., len(clean)/20)*math.exp(-age/5)
            result.update(pressure_rate=statistics.mean(r['loss'] for r in clean), pressure_confidence=conf,
                pressure_n=len(clean), pressure_age=age, pressure_valid=len(clean) >= 5 and age <= 2)
        memory = self.long[key]; age = now-memory['last'] if memory['last'] >= 0 else 999.
        attribution_unknown = any(result['unknown'][name] for name in ('multiple_candidates', 'incomplete_candidates', 'status_ambiguous', 'flight_or_geometry'))
        result.update(contribution_rate=memory['rate'], contribution_confidence=memory['confidence']*math.exp(-age/30),
            contribution_n=memory['n'], contribution_exposure=memory['exposure'], contribution_age=age,
            contribution_valid=memory['evidence']*math.exp(-age/30) >= 7 and age <= 10 and not attribution_unknown)
        if local:
            age = now-local[-1]['time']
            result.update(local_advantage=statistics.mean(r['local'] for r in local), local_confidence=min(1., len(local)/20)*math.exp(-age/10),
                local_age=age, local_valid=len(local) >= 8 and age <= 10)
        result['unknown']['insufficient_support'] = not result['contribution_valid']
        return result

    def _prediction(self, key, estimate, now, arena):
        rows = self.positions[key]; latest = _at(rows, now, .6)
        past = _at(rows, now-.5, .3)
        if latest is None or past is None or latest[0] <= past[0]: return {}
        dt = latest[0]-past[0]; vx, vy = (latest[1]-past[1])/dt, (latest[2]-past[2])/dt
        x, y = latest[1:]; speed = math.hypot(vx, vy)
        def gap(w): return math.hypot(x-(w['x']+w['w']/2), y-(w['y']+w['h']/2))
        roofs = [w for w in arena.get('obstacles', []) if 'tunnel_roof' in w.get('id', '')]
        steps = [w for w in arena.get('terrain', []) if all(k in w for k in ('x', 'y', 'w', 'h'))
            and (w.get('kind') in ('step', 'platform') or any(s in w.get('id', '') for s in ('rim', 'band')))]
        roof = min(roofs, key=gap) if roofs else None; step = min(steps, key=gap) if steps else None
        p = [estimate.get('p_wheel', 1/3), estimate.get('p_fold', 1/3), estimate.get('p_omni', 1/3)]
        # Public nominal class envelopes, not queried opponent hardware. Smooth
        # probabilities retain uncertainty about build tolerances and 2-D tracks
        # being on top of, rather than underneath, an overhead structure.
        def logistic(margin): return 1/(1+math.exp(max(-30., min(30., -margin/.035))))
        tunnel_nominal, step_nominal = (.05, .8, .05), (.65, .05, .05)
        if roof:
            from .sdk import terrain_height
            center=(roof['x']+roof['w']/2,roof['y']+roof['h']/2)
            ground=terrain_height(dict(arena,terrain=arena.get('terrain',[]),height=arena.get('height',15)),*center)
            clearance=roof.get('bottom',0.)-ground
            width=max(roof['w'],roof['h'])
            tunnel_nominal=tuple(.02+.88*logistic(clearance-height)*logistic(width-body_width)
                for height,body_width in ((.48,.44),(.20,.46),(.45,.46)))
        if step:
            rise=max(0.,step.get('elevation',step.get('height',.2))-step.get('base_elevation',0.))
            step_nominal=tuple(.02+.83*logistic(ability-rise) for ability in (.22,.04,.04))
        tunnel_p = sum(a*b for a, b in zip(p, tunnel_nominal))
        step_p = sum(a*b for a, b in zip(p, step_nominal))
        capability_conf = 0. if estimate.get('class_unknown', True) else max(p)
        near_roof = bool(roof and gap(roof) < max(2., speed*3))
        near_step = bool(step and gap(step) < max(2., speed*3))
        route = [1., 0., 0., 0.]
        obstacle = roof if near_roof else step if near_step else None
        if obstacle:
            pass_p = tunnel_p if near_roof else step_p
            route = [.15, .85*(1-pass_p), .85*pass_p if near_roof else 0., .85*pass_p if near_step and not near_roof else 0.]
        confidence = min(.9, len(rows)/20)*math.exp(-(now-latest[0])/2)
        result = dict(tunnel_probability=tunnel_p, tunnel_valid=roof is not None, tunnel_confidence=capability_conf if roof else 0.,
            step_probability=step_p, step_valid=step is not None, step_confidence=capability_conf if step else 0.,
            route=dict(zip(('straight', 'detour', 'tunnel', 'step'), route)), future={})
        result['route'].update(valid=True, confidence=confidence*max(route))
        for horizon in (1, 2, 3):
            px, py = x+vx*horizon, y+vy*horizon
            # Retain multiple route probabilities; point forecasts describe the leading mode.
            if obstacle and route[1] == max(route) and speed > .1:
                angle = obstacle.get('rotation', 0.); tangent = (-math.sin(angle), math.cos(angle))
                sign = 1 if vx*tangent[0]+vy*tangent[1] >= 0 else -1
                travel = min(speed*horizon, max(.0, gap(obstacle)-.7))
                px, py = x+vx/speed*travel, y+vy/speed*travel
                residual = speed*horizon-travel
                px += tangent[0]*sign*residual; py += tangent[1]*sign*residual
            result['future'][str(horizon)] = dict(x=max(0., min(arena.get('width', 28), px)),
                y=max(0., min(arena.get('height', 15), py)), valid=True,
                confidence=confidence*max(route)/(1+.5*horizon))
        return result

    def update(self, now, arena):
        if now < self.last_update: raise ValueError('Create a fresh radar estimator for a new episode')
        if now-self.last_update < .5-1e-8: return self.result
        self.last_update = now
        while self.last_bin+1 <= math.floor(now+1e-8):
            self.last_bin += 1; self._combat_bin(self.last_bin, arena)
        result = dict(enemies={}, team={}, robots={'ally': {}, 'enemy': {}}, predictions={})
        for slot in ENEMY_SLOTS:
            key = (self.other, slot); result['enemies'][str(slot)] = self._speed(key, now)
            result['predictions'][str(slot)] = self._prediction(key, result['enemies'][str(slot)], now, arena)
        for label, team in (('ally', self.team), ('enemy', self.other)):
            for slot in SLOTS: result['robots'][label][str(slot)] = self._robot((team, slot), now)
        own = [r for r in result['robots']['ally'].values() if r['pressure_valid'] and r['candidate_count'] and r['candidates_complete']]
        enemy = [r for r in result['robots']['enemy'].values() if r['pressure_valid'] and r['candidate_count'] and r['candidates_complete']]
        incoming, outgoing = sum(r['pressure_rate'] for r in own), sum(r['pressure_rate'] for r in enemy)
        result['team'] = dict(own_effect=outgoing, enemy_effect=incoming,
            advantage=(outgoing-incoming)/max(20., outgoing+incoming),
            confidence=min([r['pressure_confidence'] for r in own+enemy], default=0.)*min(len(own),len(enemy))/4,
            n_own=min([r['pressure_n'] for r in own], default=0), n_enemy=min([r['pressure_n'] for r in enemy], default=0),
            age=max([r['pressure_age'] for r in own+enemy], default=999.), valid=bool(own and enemy))
        self.result = result; return result

    def packet(self, content_id, now):
        self.sequence = (self.sequence+1) & 65535
        body = HEADER.pack(1, self.sequence, _q(now, 1000, 2**32-1), content_id-0x201)
        if content_id == CONTENT_OVERVIEW:
            for slot in ENEMY_SLOTS:
                r = self.result['enemies'].get(str(slot), {})
                body += SPEED_RECORD.pack(slot, int(r.get('speed_valid', False)), _q(r.get('speed_lb', 0), 100),
                    _q(r.get('robust_speed', 0), 100), _q(r.get('speed_confidence', 0), 255, 255), _q(r.get('n_valid', 0), 1, 255),
                    _q(r.get('age', 999), 10), *[_q(r.get(k, 1/3), 255, 255) for k in ('p_wheel', 'p_fold', 'p_omni')], int(r.get('class_unknown', True)))
            r = self.result['team']
            body += TEAM_RECORD.pack(_q(r.get('own_effect', 0), 10), _q(r.get('enemy_effect', 0), 10),
                round(max(-1., min(1., r.get('advantage', 0)))*32767), _q(r.get('confidence', 0), 255, 255),
                _q(r.get('n_own', 0), 1, 255), _q(r.get('n_enemy', 0), 1, 255), _q(r.get('age', 999), 10), int(r.get('valid', False)))
        elif content_id in (CONTENT_ALLIES, CONTENT_ENEMIES):
            owner = 'ally' if content_id == CONTENT_ALLIES else 'enemy'
            for slot in SLOTS:
                r = self.result['robots'][owner].get(str(slot), _robot_default())
                flags = sum(int(r.get(k, False)) << i for i, k in enumerate(('pressure_valid', 'contribution_valid', 'local_valid', 'candidates_complete', 'geometry_unique')))
                reasons = sum(int(r.get('unknown', {}).get(k, True)) << i for i, k in enumerate(REASONS))
                body += ROBOT_RECORD.pack(slot, flags, _q(r['pressure_rate'], 10), _q(r['contribution_rate'], 10),
                    round(max(-1., min(1., r['local_advantage']))*127),
                    *[_q(r[k], 255, 255) for k in ('pressure_confidence', 'contribution_confidence', 'local_confidence')],
                    _q(r['contribution_exposure'], 1, 255), _q(r['pressure_n'], 1, 255), _q(r['contribution_n'], 1, 255),
                    _q(r['pressure_age'], 10), _q(r['contribution_age'], 10), _q(r['local_age'], 10, 255), _q(r['candidate_count'], 1, 255), reasons)
        elif content_id == CONTENT_PREDICTION:
            for slot in ENEMY_SLOTS:
                r = self.result['predictions'].get(str(slot), {}); route = r.get('route', {}); future = r.get('future', {})
                flags = int(r.get('tunnel_valid', False)) | int(r.get('step_valid', False)) << 1 | int(route.get('valid', False)) << 2
                flags |= sum(int(future.get(str(h), {}).get('valid', False)) << (h+2) for h in (1, 2, 3))
                values = [_q(r.get('tunnel_probability', .5), 255, 255), _q(r.get('step_probability', .5), 255, 255),
                    _q(r.get('tunnel_confidence', 0), 255, 255), _q(r.get('step_confidence', 0), 255, 255), flags,
                    *[_q(route.get(k, .25), 255, 255) for k in ('straight', 'detour', 'tunnel', 'step')], _q(route.get('confidence', 0), 255, 255)]
                for h in (1, 2, 3):
                    f = future.get(str(h), {}); values += [_q(f.get('x', 0), 100), _q(f.get('y', 0), 100), _q(f.get('confidence', 0), 255, 255)]
                body += PREDICTION_RECORD.pack(*values)
        else: raise ValueError('Unknown inference content ID')
        if len(body) > 112: raise AssertionError('Radar inference packet exceeds protocol payload')
        return body


class RadarInferenceReceiver:
    """Only successfully delivered application packets reach the actor cache."""
    def __init__(self): self.pages = {}

    def receive(self, content_id, payload, receipt_time):
        expected = {CONTENT_OVERVIEW: 62, CONTENT_ALLIES: 88, CONTENT_ENEMIES: 88, CONTENT_PREDICTION: 83}
        if content_id not in expected or len(payload) != expected[content_id]: raise ValueError('Invalid inference packet size')
        version, sequence, milliseconds, page = HEADER.unpack_from(payload)
        if version != 1 or page != content_id-0x201: raise ValueError('Invalid inference packet version/page')
        source = milliseconds/1000
        if source > receipt_time+1e-6: raise ValueError('Inference packet from the future')
        previous = self.pages.get(content_id)
        if previous and source <= previous[0]: return False
        self.pages[content_id] = (source, receipt_time, bytes(payload)); return True

    def observation(self, now):
        result = dict(enemies={}, team={}, robots={'ally': {}, 'enemy': {}}, predictions={})
        for content_id, (source, received, payload) in self.pages.items():
            delay = max(0., now-source); offset = HEADER.size
            if content_id == CONTENT_OVERVIEW:
                for _ in ENEMY_SLOTS:
                    slot, flags, lb, speed, confidence, n, age, pw, pf, po, unknown = SPEED_RECORD.unpack_from(payload, offset); offset += SPEED_RECORD.size
                    weights = [pw/255, pf/255, po/255]; total = sum(weights) or 1
                    result['enemies'][str(slot)] = dict(speed_lb=lb/100, robust_speed=speed/100,
                        speed_valid=bool(flags & 1) and age/10+delay <= 2, speed_confidence=confidence/255*math.exp(-delay/10),
                        n_valid=n, age=age/10+delay, p_wheel=weights[0]/total, p_fold=weights[1]/total,
                        p_omni=weights[2]/total, class_unknown=bool(unknown) or delay > 5)
                own, enemy, advantage, conf, no, ne, age, valid = TEAM_RECORD.unpack_from(payload, offset)
                result['team'] = dict(own_effect=own/10, enemy_effect=enemy/10, advantage=advantage/32767,
                    confidence=conf/255*math.exp(-delay/10), n_own=no, n_enemy=ne, age=age/10+delay, valid=bool(valid) and age/10+delay <= 10)
            elif content_id in (CONTENT_ALLIES, CONTENT_ENEMIES):
                owner = 'ally' if content_id == CONTENT_ALLIES else 'enemy'
                for _ in SLOTS:
                    (slot, flags, pressure, contribution, advantage, pc, cc, lc, exposure, pn, cn, pa, ca, la, candidates, reasons) = ROBOT_RECORD.unpack_from(payload, offset); offset += ROBOT_RECORD.size
                    result['robots'][owner][str(slot)] = dict(pressure_rate=pressure/10, contribution_rate=contribution/10,
                        local_advantage=advantage/127, pressure_confidence=pc/255*math.exp(-delay/5), contribution_confidence=cc/255*math.exp(-delay/30),
                        local_confidence=lc/255*math.exp(-delay/10), contribution_exposure=exposure, pressure_n=pn, contribution_n=cn,
                        pressure_age=pa/10+delay, contribution_age=ca/10+delay, local_age=la/10+delay,
                        pressure_valid=bool(flags & 1) and pa/10+delay <= 2, contribution_valid=bool(flags & 2) and ca/10+delay <= 10,
                        local_valid=bool(flags & 4) and la/10+delay <= 10, candidates_complete=bool(flags & 8) and delay <= 2,
                        geometry_unique=bool(flags & 16) and delay <= 2, candidate_count=candidates,
                        unknown={name: bool(reasons & (1 << i)) for i, name in enumerate(REASONS)})
            elif content_id == CONTENT_PREDICTION:
                for slot in ENEMY_SLOTS:
                    values = PREDICTION_RECORD.unpack_from(payload, offset); offset += PREDICTION_RECORD.size
                    tp, sp, tc, sc, flags, a, b, c, d, rc = values[:10]
                    r = dict(tunnel_probability=tp/255, step_probability=sp/255,
                        tunnel_valid=bool(flags & 1) and delay <= 2, step_valid=bool(flags & 2) and delay <= 2,
                        tunnel_confidence=tc/255*math.exp(-delay/2), step_confidence=sc/255*math.exp(-delay/2), future={},
                        route=dict(straight=a/255, detour=b/255, tunnel=c/255, step=d/255, valid=bool(flags & 4) and delay <= 2, confidence=rc/255*math.exp(-delay/2)))
                    for i, horizon in enumerate((1, 2, 3)):
                        x, y, confidence = values[10+i*3:13+i*3]
                        r['future'][str(horizon)] = dict(x=x/100, y=y/100, valid=bool(flags & (1 << (horizon+2))) and delay <= .75,
                            confidence=confidence/255*math.exp(-delay/2))
                    result['predictions'][str(slot)] = r
        return result
