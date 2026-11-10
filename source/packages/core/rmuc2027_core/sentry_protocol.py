"""2027 sentry wire boundary. Privileged simulation state stops in this module.

Reference: communication protocol V1.0.0 (20261029), tables 1-1..1-46.
Source events use rational clocks; transport delivery is quantized to the next
50 ms simulation frame. `scheduled_time` and actual `source_time` are separate.
The ideal localization and autoaim modules are explicit hardware abstractions.
"""
from collections import defaultdict, deque
import copy
import math
import struct

PROFILE = 'sentry_referee_2027_v1'
RATES = {0x0001:1, 0x0003:3, 0x0101:1, 0x0104:1, 0x0201:10,
         0x0202:10, 0x0203:1, 0x0204:3, 0x0208:10, 0x0209:3,
         0x020B:1, 0x020D:1, 0x020C:1, 0x020E:1, 0xF001:10, **{c:10 for c in range(0x0A01, 0x0A06)}}
LENGTHS = {1:11, 3:24, 0x101:8, 0x104:3, 0x201:17, 0x202:14,
           0x203:12, 0x204:7, 0x206:7, 0x207:7, 0x208:8, 0x209:6,
           0x20B:40, 0x20C:2, 0x20D:14, 0x20E:8, 0xA01:24,
           0xA02:12, 0xA03:10, 0xA04:12, 0xA05:47, 0xA06:6,
           0x303:12, 0x305:48, 0xF001:17}  # Application-defined 0x0200 relay record: ally xy and validity.
GROUND = (1, 3, 4, 7)
POSTURES = ('attack', 'defense', 'move')


def crc(data, initial, polynomial, bits):
    value = initial
    for byte in data:
        value ^= byte
        for _ in range(8):
            value = (value >> 1) ^ (polynomial if value & 1 else 0)
    return value & ((1 << bits)-1)


def frame(command, payload, sequence=0):
    payload = bytes(payload)
    if command in LENGTHS and len(payload) != LENGTHS[command]:
        raise ValueError('Incorrect referee payload length')
    header = struct.pack('<BHB', 0xA5, len(payload), sequence & 255)
    header += bytes([crc(header, 255, 0x8C, 8)])
    result = header + struct.pack('<H', command) + payload
    return result + struct.pack('<H', crc(result, 65535, 0x8408, 16))


def unframe(data):
    if len(data) < 9 or data[0] != 0xA5 or crc(data[:4], 255, 0x8C, 8) != data[4]:
        raise ValueError('Invalid frame header/CRC8')
    size, = struct.unpack_from('<H', data, 1)
    if len(data) != size+9 or crc(data[:-2], 65535, 0x8408, 16) != struct.unpack_from('<H', data, len(data)-2)[0]:
        raise ValueError('Invalid frame length/CRC16')
    command, = struct.unpack_from('<H', data, 5)
    if command in LENGTHS and size != LENGTHS[command]:
        raise ValueError('Incorrect referee payload length')
    return command, bytes(data[7:-2]), data[3]


class SourceClock:
    def __init__(self):
        self.count = defaultdict(int)

    def due(self, key, hz, now):
        """No rounded tick divisor: 3 Hz produces 0, 1/3, 2/3, 1, ... ."""
        result = []
        while self.count[key] <= math.floor(now*hz+1e-9):
            result.append(self.count[key]/hz)
            self.count[key] += 1
        return result


class InteractionLink:
    """Shared 0x0301 upload and rolling one-second recipient byte budgets."""
    def __init__(self):
        self.sent, self.received = defaultdict(deque), defaultdict(deque)
        self.trace = deque(maxlen=2048)

    def send(self, now, sender, recipient, content_id, payload, base_lost_hp=0):
        payload = bytes(payload)
        ss, rs = sender % 100, recipient % 100
        allied = (sender < 100) == (recipient < 100)
        legal = allied and sender != recipient and ss in (*GROUND, 9) and rs in (*GROUND, 9)
        legal &= (rs != 7 or ss == 9) and (rs != 9 or ss == 7)
        if content_id in (0x120, 0x121):
            legal = recipient == 0x8080 and ss == (7 if content_id == 0x120 else 9)
            legal &= len(payload) == (4 if content_id == 0x120 else 13)
        elif not 0x200 <= content_id <= 0x2FF:
            legal = False
        reason = None
        if not legal: reason = 'recipient_or_content'
        elif len(payload) > 112: reason = 'payload_size'
        tx, rx = self.sent[sender], self.received[recipient]
        for queue in (tx, rx):
            while queue and queue[0][0] <= now-1+1e-9:
                queue.popleft()
        wire = frame(0x301, struct.pack('<HHH', content_id, sender, recipient)+payload)
        capacity = math.floor((5120 if rs in (7,9) else 3720)*max(0.,1-base_lost_hp/20000)+.5)
        if not reason and len(tx) >= 30: reason = 'upload_30hz'
        if not reason and recipient != 0x8080 and sum(n for _,n in rx)+len(wire) > capacity:
            reason = 'receive_bandwidth'
        self.trace.append(dict(time=now,sender=sender,recipient=recipient,content_id=content_id,
                               bytes=len(wire),accepted=reason is None,reason=reason))
        if reason: return None
        tx.append((now,len(wire)))
        rx.append((now,len(wire)))
        return wire


def sentry_word(*, free=False, paid=False, local_ammo=0, remote_ammo=0, remote_heal=0, posture=3, rune=False):
    for value, limit in ((local_ammo,2047),(remote_ammo,15),(remote_heal,15),(posture,6)):
        if type(value) is not int or not 0 <= value <= limit: raise ValueError('Counter out of range')
    if posture == 0: raise ValueError('Invalid posture')
    return int(free) | int(paid)<<1 | local_ammo<<2 | remote_ammo<<13 | remote_heal<<17 | posture<<21 | int(rune)<<24


def integer(value, maximum=65535):
    return max(0, min(maximum, math.floor(float(value)+.5)))


def robot_id(team, slot):
    return slot + (100 if team == 'blue' else 0)


def rfid_names(team):
    other = 'blue' if team == 'red' else 'red'
    names = [team+'_supply', team+'_assembly', 'red_base', 'blue_base', 'red_fortress',
             'blue_fortress', 'red_central', 'blue_central', 'red_outpost', 'blue_outpost',team+'_high_point',None,None,None]
    names += [team+'_highland_low',team+'_central',other+'_highland_low',other+'_central']
    names += [f'{t}_road_{i}' for t in (team,other) for i in range(2)]
    names += [f'{team}_{kind}_tunnel_{i}' for kind in ('road','near','far') for i in range(3)]
    names += [None]
    names += [f'{other}_{kind}_tunnel_{i}' for kind in ('road','near','far') for i in range(3)]
    return names


def field_words(snapshot, team):
    """Only own full/temporary/not-controlled states; no capture progress."""
    positions = snapshot['positions']
    def control(key):
        p = positions[key]
        return (0 if p['control']=='full' else 1) if p['controller']==team else 2
    def active(zone):
        value = snapshot['zones'].get(zone,{})
        teams = [r['team'] for r in snapshot['robots'] if r['id'] in value.get('occupants',[])]
        # Zone occupiers are derived only inside the referee, never exposed raw.
        if not isinstance(teams, list): teams=[]
        own, enemy = team in teams, any(t!=team for t in teams)
        return int(own)+2*int(enemy)
    a = active(team+'_supply') & 1
    rune = snapshot['runes'][team]
    a |= (1 if rune.get('status')=='active' else 2 if rune.get('status')=='activating' else 0) << (1 if rune.get('mode')=='small' else 3)
    for owner, shift in (('red',5),('blue',8)):
        a |= (active(owner+'_fortress') if control(owner+'_outpost')<2 else 4) << shift
    for owner, shift in (('red',11),('blue',14)):
        a |= (active(owner+'_base') if control(owner+'_base')<2 else 4) << shift
    a |= (active(team+'_high_point')&1)<<17
    for owner, shift in (('red',18),('blue',20)):
        a |= (min(2,active(owner+'_central')) if control(owner+'_outpost')<2 else 3)<<shift
    for owner, shift in (('red',22),('blue',25)):
        a |= (active(owner+'_outpost') if control(owner+'_outpost')<2 else 4)<<shift
    a |= control('red_outpost')<<28 | control('blue_outpost')<<30
    b = control('red_base') | control('blue_base')<<2
    return a,b


def buffs(robot, telemetry):
    return (integer(telemetry.get('heal_percent',0),255),
            integer(max(robot.get('cooling_bonus',0),robot.get('cooling_rate',0)*(robot.get('cooling_multiplier',1)-1))),integer(robot.get('defense_buff',0)*100,255),
            integer(robot.get('vulnerability',0)*100,255),integer((robot.get('attack_multiplier',1)-1)*100))


def payloads(snapshot, team):
    """Referee encoder; the consumer only decodes the resulting wire bytes."""
    own = {r['slot']:r for r in snapshot['robots'] if r['team']==team}
    other = 'blue' if team=='red' else 'red'
    enemies = {r['slot']:r for r in snapshot['robots'] if r['team']==other}
    r = own[7]
    tele = snapshot['sentry_referee']
    t = tele[r['id']]
    hp = lambda slot: integer(own.get(slot,{}).get('hp',0))
    words = field_words(snapshot,team)
    p = {1:struct.pack('<BHQ',0x41,integer(snapshot['remaining']),0),
         3:struct.pack('<4Hh7H',hp(1),0,hp(3),hp(4),max(-32768,min(32767,round(snapshot['teams'][team].get('attack_damage',0)-snapshot['teams'][other].get('attack_damage',0)))),hp(7),
             integer(snapshot['outposts']['red']['hp']),integer(snapshot['bases'][team]['hp']),integer(snapshot['outposts']['blue']['hp']),
             integer(snapshot['bases'][other]['hp']),integer(snapshot['teams'][team]['control']),integer(snapshot['teams'][other]['control'])),
         0x101:struct.pack('<II',*words), 0x104:bytes(3),
         0x201:struct.pack('<BB5HfB',robot_id(team,7),t['level'],integer(r['hp']),integer(r['max_hp']),integer(r['cooling_rate']),integer(r['max_heat']),integer(r['power_limit']),r['muzzle_speed_mps'],t['power_outputs']),
         0x202:struct.pack('<8x3H',integer(r['buffer_energy_j']),integer(r['heat']),0),
         0x203:struct.pack('<3f',r['x'],r['y'],(90-math.degrees(r['aim']))%360),
         0x204:struct.pack('<BHBBH',*buffs(r,t)),
         0x208:struct.pack('<4H',r['ammo'],0,integer(snapshot['teams'][team]['resources']),integer(snapshot['teams'][team].get('fortress_ammo',0)))}
    flags = sum(1<<i for i,name in enumerate(rfid_names(team)) if name and name in r['zone_contacts'])
    p[0x209] = struct.pack('<IH',flags & 0xFFFFFFFF,flags>>32)
    p[0x20B] = struct.pack('<10f',*[v for slot in (1,2,3,4,6) for v in (own.get(slot,{}).get('x',0) if slot in (1,3,4) else 0,own.get(slot,{}).get('y',0) if slot in (1,3,4) else 0)])
    ledger = t['ledger']
    first = ledger['local_ammo'] | ledger['remote_ammo']<<11 | ledger['remote_heal']<<15 | int(t['free_revive'])<<19 | int(t['paid_revive'])<<20 | integer(t['paid_cost'],1023)<<21
    second = int(t['disengaged']) | integer(t['ammo_quota'],2047)<<1 | (POSTURES.index(r['posture'])+1)<<12 | int(t['rune_available'])<<14 | int(r.get('posture_enhanced',False))<<15
    timers = bytes([integer(max(0,180-v),255) for v in r.get('posture_time',[0]*3)]+[0]+[integer(max(0,15-v),255) for v in r.get('posture_enhanced_time',[0]*3)]+[0])
    p[0x20D] = struct.pack('<IH',first,second)+timers
    p[0xA01] = struct.pack('<12H',*[integer(enemies.get(s,{}).get(k,0)*100) if s in GROUND else 0 for s in (1,2,3,4,6,7) for k in ('x','y')])
    p[0xA02] = struct.pack('<6H',*[integer(enemies.get(s,{}).get('hp',0)) if s in GROUND else 0 for s in (1,2,3,4,6,7)])
    p[0xA03] = struct.pack('<5H',*[integer(enemies.get(s,{}).get('ammo',0)) if s in GROUND else 0 for s in (1,3,4,6,7)])
    a,b = field_words(snapshot,other)
    ea = (a&1) | ((a>>5)&0xFFF)<<1 | ((a>>17)&1)<<13 | ((a>>18)&15)<<14 | ((a>>22)&63)<<18
    eb = ((a>>28)&15)<<22 | (b&15)<<26
    # Enemy crossing activation is an information-wave aggregate, not identities.
    crossing = [other+'_central',team+'_central',other+'_road_1',team+'_road_1']
    crossing += [f'{side}_{kind}_tunnel_{i}' for side in (other,team) for kind in ('road','near','far') for i in range(3)]
    eb |= sum(1<<i for i,name in enumerate(crossing) if any(name in e.get('zone_contacts',{}) for e in enemies.values()))
    p[0xA04] = struct.pack('<HHII',integer(snapshot['teams'][other]['resources']),integer(snapshot['teams'][other].get('total_resources',snapshot['teams'][other]['resources'])),ea,eb)
    body = b''.join(struct.pack('<BHBBH',*buffs(enemies[s],tele[enemies[s]['id']])) if s in enemies else bytes(7) for s in (1,2,3,4,7))
    er = enemies[7]
    body += bytes([POSTURES.index(er['posture'])+1+3*er.get('posture_enhanced',False)])
    body += bytes([0 if s not in enemies else 1 if not enemies[s]['alive'] else 3 if enemies[s]['weak'] else 2 if enemies[s]['invulnerable'] else 0 for s in (1,2,3,4,7)])
    p[0xA05] = body+bytes(6)
    return p


class Commander:
    """0x0303 repetitions preserve an order id; consumers execute each order once."""
    def __init__(self,team=None,recipient=None):
        self.last_send = {}; self.order = None; self.serial = 0; self.sent = 0
        self.team=team;self.recipient=recipient;self.next_periodic=1.;self.pending=deque()

    def submit(self, now, x, y, *, key=0, target=0, source=None, semi=False):
        if source is None:raise ValueError('0x0303 requires the configured client source ID')
        clients=(0x101,0x103,0x104,0x105,0x106) if self.team=='red' else (0x165,0x167,0x168,0x169,0x16A) if self.team=='blue' else (0x101,0x103,0x104,0x105,0x106,0x165,0x167,0x168,0x169,0x16A)
        if source not in clients or semi and self.recipient==7:return False
        interval = 3. if semi else .5
        if now-self.last_send.get(source,-100.) < interval-1e-9: return False
        if not all(math.isfinite(v) for v in (x,y)): raise ValueError('Finite coordinates required')
        if target:x=y=0.
        wire = frame(0x303,struct.pack('<ffBBH',x,y,key,target,source))
        self.serial += 1; self.sent=0
        self.order=dict(id=self.serial,created=now,wire=wire)
        self.last_send[source]=now
        return True

    def poll(self, now):
        if self.order is None: return None
        while self.sent<5 and now+1e-9>=self.order['created']+self.sent*.1:
            self.pending.append((self.order, self.order['created']+self.sent*.1));self.sent+=1
        # This clock does not reset on a newly submitted command.
        while now+1e-9>=self.next_periodic:
            if self.next_periodic>=self.order['created']:self.pending.append((self.order, self.next_periodic))
            self.next_periodic+=1.
        if not self.pending:return None
        order,scheduled=self.pending.popleft()
        _,body,_ = unframe(order['wire'])
        x,y,key,target,source = struct.unpack('<ffBBH',body)
        return dict(id=order['id'],source_time=order['created'],scheduled_time=scheduled,receipt_time=now,x=x,y=y,key=key,target=target,source=source)


# Physical feedback explicitly agreed to be ideal. Referee fields are excluded.
LOCAL_FIELDS = ('id','slot','team','role','model_slot','x','y','z','vx','vy','vz',
    'heading','angular_velocity','quaternion','aim','pitch','grounded','upright','tipped',
    'legs','wheel_angular_velocity','transform_progress','transform_allowed',
    'radius','chassis_size_m','chassis_type','mass_kg','speed_reference_mps',
    'armor_center_z_m','muzzle_height_m','steering_angles','hip_angle','self_righting')


class SentryGateway:
    def __init__(self, snapshot):
        from .radar2027 import RadarReferee
        from .radar_inference import RadarInference, RadarInferenceReceiver
        self.radar=RadarReferee(snapshot['seed'])
        self.inference={t:RadarInference(t) for t in ('red','blue')}
        self.inference_received={t:RadarInferenceReceiver() for t in ('red','blue')}
        self.pending_inference=[]
        self.clock=SourceClock(); self.link=InteractionLink()
        self.cache={t:{} for t in ('red','blue')}
        self.history={t:{} for t in ('red','blue')}
        self.faults={}; self.pending=[]; self.trace=deque(maxlen=4096)
        self.commander={t:Commander(t,7) for t in ('red','blue')}
        self.orders={}; self.execution={}; self.snapshot=None
        self.event_seq=defaultdict(int);self.warning={t:bytes(3) for t in ('red','blue')}
        self.events={t:deque(maxlen=32) for t in ('red','blue')}
        self.update(snapshot)

    def _receive(self, team, cmd, body, scheduled, source, now, source_name):
        _,body,_ = unframe(frame(cmd,body))
        fault = self.faults.get((team,cmd),{})
        if fault.get('drop'): return
        if fault.get('corrupt'): return
        delay=float(fault.get('delay',0))
        sample=dict(payload=body,scheduled_time=scheduled,source_time=source,receipt_time=now+delay,source=source_name)
        if delay: self.pending.append((team,cmd,sample))
        else: self._install(team,cmd,sample)

    def _install(self,team,cmd,sample):
        previous = self.cache[team].get(cmd)
        if previous and previous['source_time'] > sample['source_time']: return
        if previous and previous['source_time'] == sample['source_time']:
            if cmd not in (0x206,0x207) or previous['payload']==sample['payload']:return
        if previous: self.history[team][cmd]=previous
        self.cache[team][cmd]=sample
        # Only decoded legal packets enter the estimator. In particular the
        # referee's private shot/hit ledgers never cross this boundary.
        self.inference[team].consume(cmd,sample['payload'],sample['source_time'])
        if cmd in (0x104,0x206,0x207):
            self.events[team].append(dict(command=cmd,payload=list(sample['payload']),source_time=sample['source_time'],receipt_time=sample['receipt_time']))
        self.trace.append(dict(team=team,command=cmd,**{k:v for k,v in sample.items() if k!='payload'}))

    def update(self,snapshot):
        self.snapshot=snapshot; now=snapshot['time']
        self.radar.update(snapshot)
        for team,cmd,sample in self.pending[:]:
            if sample['receipt_time']<=now+1e-9:
                self._install(team,cmd,sample); self.pending.remove((team,cmd,sample))
        for item in self.pending_inference[:]:
            team,content,body,delivery=item
            if delivery<=now+1e-9:
                self.inference_received[team].receive(content,body,now)
                self.pending_inference.remove(item)
        for team in ('red','blue'):
            p=payloads(snapshot,team)
            other='blue' if team=='red' else 'red'
            p[0xA05]=p[0xA05][:-6]+self.radar.feedback(other)[2:]
            feedback=self.execution.setdefault(team,{})
            feedback.update(execution_time=now,shots=0)
            for e in snapshot['events']:
                cmd=None;body=None
                if e.get('robot')==f'{team}-7' and e['type']=='shot':
                    robot=next(r for r in snapshot['robots'] if r['id']==e['robot'])
                    cmd=0x207;body=struct.pack('<BBBf',1,1,integer(1/robot['trigger_interval_s'],255),robot['muzzle_speed_mps'])
                if e.get('robot')==f'{team}-7' and e['type']=='armor_hit':
                    self.event_seq[team]+=1;cmd=0x206
                    body=struct.pack('<BHhh',e['armor_id'],self.event_seq[team]&65535,*e['armor_xy_mm'])
                if e['type'] in ('yellow_card','red_card','yellow_both','forfeit') and (e.get('team')==team or e['type']=='yellow_both'):
                    level={'yellow_card':2,'red_card':3,'yellow_both':1,'forfeit':4}[e['type']]
                    rid=robot_id(team,int(e['robot'].rsplit('-',1)[1])) if e.get('robot') else 0
                    self.warning[team]=bytes([level,rid,integer(e.get('card_count',1),255)])
                    cmd=0x104;body=self.warning[team]
                if cmd is not None:
                    self._receive(team,cmd,body,e['time'],e['time'],now,f'referee:0x{cmd:04X}')
            p[0x104]=self.warning[team]
            for cmd,hz in RATES.items():
                if cmd>=0xA00 or cmd in (0x20C,0x20E): continue
                for scheduled in self.clock.due((team,cmd),hz,now):
                    self._receive(team,cmd,p[cmd],scheduled,now,now,f'referee:0x{cmd:04X}')
            for scheduled in self.clock.due((team,'radar'),10,now):
                own={r['slot']:r for r in snapshot['robots'] if r['team']==team}
                valid=0;coords=[]
                for index,slot in enumerate(GROUND):
                    # Explicit ideal radar tracker, with injectable loss/error. Slot 6 has no ground packet.
                    fault=self.radar.faults.get((team,own[slot]['id']),{})
                    available=not fault.get('drop')
                    valid |= int(available)<<index
                    coords.extend(integer((own[slot][k]+(fault.get('error',0.) if k=='x' else 0))*100) for k in ('x','y'))
                p[0xF001]=struct.pack('<B8H',valid,*coords)
                marks=0
                for owner,threshold,base in ((('blue' if team=='red' else 'red'),100,0),(team,50,6)):
                    for slot,bit in ((1,0),(3,2),(4,3),(7,5)):
                        marks |= int(self.radar.marks.get((team,f'{owner}-{slot}'),{}).get('p',0)>=threshold)<<(base+bit)
                p[0x20C]=struct.pack('<H',marks);p[0x20E]=self.radar.feedback(team)
                wave=not self.faults.get((team,'wave'),{}).get('drop')
                emit_feedback=bool(self.clock.due((team,'radar_feedback'),1,now))
                # Two bounded application packets; all 0x0301 content shares one limiter.
                groups=[([0xA01,0xA02,0xA03] if wave else [])+[0xF001],([0xA04,0xA05] if wave else [])+([0x20C,0x20E] if emit_feedback else [])]
                for group in groups:
                    if not group:continue
                    body=b''.join(struct.pack('<H',c)+p[c] for c in group)
                    lost=snapshot['bases'][team]['max_hp']-snapshot['bases'][team]['hp']
                    packet=self.link.send(now,robot_id(team,9),robot_id(team,7),0x200,body,lost)
                    if packet:
                        _,envelope,_=unframe(packet); offset=6
                        while offset<len(envelope):
                            cmd,=struct.unpack_from('<H',envelope,offset);offset+=2
                            payload=envelope[offset:offset+LENGTHS[cmd]];offset+=LENGTHS[cmd]
                            self._receive(team,cmd,payload,scheduled,now,now,f'radar:0x{cmd:04X}/0x0301')
            # Original referee relays have priority. All application estimates
            # share the same 30 Hz sender and recipient byte limiter.
            self.inference[team].update(now,snapshot['arena'])
            for content,hz in ((0x201,2),(0x202,1),(0x203,1),(0x204,2)):
                if not self.clock.due((team,'inference',content),hz,now):continue
                body=self.inference[team].packet(content,now)
                lost=snapshot['bases'][team]['max_hp']-snapshot['bases'][team]['hp']
                packet=self.link.send(now,robot_id(team,9),robot_id(team,7),content,body,lost)
                fault=self.faults.get((team,'inference'),{})
                if not packet or fault.get('drop') or fault.get('corrupt'):continue
                _,envelope,_=unframe(packet)
                received_body=envelope[6:]
                delay=max(0.,float(fault.get('delay',0.)))
                if delay:self.pending_inference.append((team,content,received_body,now+delay))
                else:self.inference_received[team].receive(content,received_body,now)
            order=self.commander[team].poll(now)
            if order:self.orders[team]=order
            feedback['shots']=sum(e['command']==0x207 and now-.05+1e-9<e['receipt_time']<=now+1e-9 for e in self.events[team])

    def sample(self,team,cmd):
        value=self.cache[team].get(cmd)
        now=self.snapshot['time']
        if value is None:return None,dict(valid=False,age=None,source=None,source_time=None,receipt_time=None)
        age=now-value['source_time'];hz=RATES.get(cmd,1)
        # Validity is a receiver timeout, not a made-up referee frequency.
        meta={k:v for k,v in value.items() if k!='payload'}
        meta.update(valid=age<=2/hz+.05+1e-9,age=age)
        return value['payload'] if meta['valid'] else None,meta

    def command(self,team,fields):
        fields=dict(fields)
        for key in ('revive','buy_ammo','remote_heal','posture','posture_enhanced','rune_start'):
            fields.pop(key,None)
        word=fields.pop('sentry_cmd',None)
        accepted=False
        if type(word) is int and 0<=word<1<<25:
            accepted=self.link.send(self.snapshot['time'],robot_id(team,7),0x8080,0x120,struct.pack('<I',word)) is not None
            if accepted:fields['sentry_cmd']=word
        self.execution[team]=dict(time=self.snapshot['time'],requested=word,accepted=accepted,
                                 spin=bool(fields.get('yaw_rate')),fire=bool(fields.get('fire')),jump=bool(fields.get('jump')))
        return fields

    def radar_command(self,team,payload):
        """Host/radar integration entry point; sentry actions cannot impersonate radar."""
        wire=self.link.send(self.snapshot['time'],robot_id(team,9),0x8080,0x121,payload)
        return self.radar.command(team,payload) if wire else {'accepted':False}

    def observation(self,team,sensors):
        s=self.snapshot;now=s['time'];other='blue' if team=='red' else 'red'
        raw=next(r for r in s['robots'] if r['team']==team and r['slot']==7)
        me={k:copy.deepcopy(raw[k]) for k in LOCAL_FIELDS if k in raw}
        data={};sources={}
        for cmd in RATES:
            data[cmd],sources[f'0x{cmd:04X}']=self.sample(team,cmd)
        if data[0x201]:
            rid,level,hp,maximum,cooling,heat,power,speed,outputs=struct.unpack('<BB5HfB',data[0x201])
            me.update(level=level,hp=hp,max_hp=maximum,cooling_rate=cooling,max_heat=heat,power_limit=power,muzzle_speed_mps=speed,powered=bool(outputs&2),alive=hp>0,power_outputs=outputs)
        else:me.update(hp=0,max_hp=1,alive=False,powered=False,muzzle_speed_mps=0,max_heat=0,cooling_rate=0,power_limit=0)
        if data[0x202]:
            buffer,heat,_=struct.unpack('<8x3H',data[0x202]);me.update(buffer_energy_j=buffer,heat=heat)
        else:me.update(buffer_energy_j=0,heat=0)
        coins=None
        if data[0x208]:
            ammo,_,coins,reserve=struct.unpack('<4H',data[0x208]);me.update(ammo=ammo,fortress_ammo=reserve)
        else:me['ammo']=0
        if data[0x204]:me['buffs']=dict(zip(('heal','cooling','defense','vulnerability','attack'),struct.unpack('<BHBBH',data[0x204])))
        if data[0x209]:
            a,b=struct.unpack('<IH',data[0x209]);me['rfid']=[(a | b<<32)>>i & 1 for i in range(41)]
        if data[0x203]:
            x,y,angle=struct.unpack('<3f',data[0x203]);me['referee_pose']=dict(x=x,y=y,muzzle_angle_deg=angle)
        ack={}
        if data[0x20D]:
            a,b=struct.unpack_from('<IH',data[0x20D]);timers=data[0x20D][6:]
            ack=dict(local_ammo=a&2047,remote_ammo=(a>>11)&15,remote_heal=(a>>15)&15,free_revive=bool(a&(1<<19)),paid_revive=bool(a&(1<<20)),paid_cost=(a>>21)&1023,disengaged=bool(b&1),ammo_quota=(b>>1)&2047,rune_available=bool(b&(1<<14)))
            me.update(posture=POSTURES[max(0,((b>>12)&3)-1)],posture_enhanced=bool(b&(1<<15)),posture_remaining=list(timers[:3]),enhanced_remaining=list(timers[4:7]))
        friends=[]
        positions=struct.unpack('<10f',data[0x20B]) if data[0x20B] else None
        hp=struct.unpack('<4Hh7H',data[3]) if data[3] else None
        for slot,idx in ((1,0),(3,2),(4,3),(6,None)):
            f=dict(id=f'{team}-{slot}',team=team,slot=slot,position_valid=bool(positions and idx is not None),hp_valid=bool(hp and idx is not None))
            if f['position_valid']:f.update(x=positions[idx*2],y=positions[idx*2+1])
            if f['hp_valid']:f.update(hp=hp[idx],alive=hp[idx]>0)
            if data[0xF001] and slot in GROUND:
                values=struct.unpack('<B8H',data[0xF001]);j=GROUND.index(slot)
                if values[0] & (1<<j):f.update(x=values[1+j*2]/100,y=values[2+j*2]/100,position_valid=True,position_source='radar')
            if f['position_valid'] and 'position_source' not in f:f['position_source']='020B'
            friends.append(f)
        enemies=[]
        xy=struct.unpack('<12H',data[0xA01]) if data[0xA01] else None
        ehp=struct.unpack('<6H',data[0xA02]) if data[0xA02] else None
        ammo=struct.unpack('<5H',data[0xA03]) if data[0xA03] else None
        status=data[0xA05]
        for slot,index,bi,ai in ((1,0,0,0),(3,2,2,1),(4,3,3,2),(7,5,4,4)):
            e=dict(id=f'{other}-{slot}',slot=slot,team=other,position_valid=xy is not None,hp_valid=ehp is not None,status_valid=status is not None)
            if xy:
                e.update(x=xy[2*index]/100,y=xy[2*index+1]/100)
                previous=self.history[team].get(0xA01)
                if previous:
                    dt=self.cache[team][0xA01]['source_time']-previous['source_time']
                    old=struct.unpack('<12H',previous['payload'])
                    if dt>0:e.update(vx=(xy[2*index]-old[2*index])/100/dt,vy=(xy[2*index+1]-old[2*index+1])/100/dt,velocity_estimated=True)
            if ehp:e['hp']=ehp[index]
            if ammo:e['ammo']=ammo[ai]
            if status:
                e.update(buffs=dict(zip(('heal','cooling','defense','vulnerability','attack'),struct.unpack_from('<BHBBH',status,bi*7))),alive=status[36+bi]!=1,invulnerable=status[36+bi]>=2,weak=status[36+bi]==3)
                if slot==7:e['posture_code']=status[35]
            enemies.append(e)
        field={};control={};bases={};remaining=None
        if data[1]:remaining=struct.unpack_from('<H',data[1],1)[0]
        if data[0x101]:
            a,b=struct.unpack('<II',data[0x101]);field=dict(event_data=a,event_data_2=b)
            control=dict(red_outpost=(a>>28)&3,blue_outpost=(a>>30)&3,red_base=b&3,blue_base=(b>>2)&3)
        if hp:bases=dict(own=hp[7],enemy=hp[9],red_outpost=hp[6],blue_outpost=hp[8])
        radar={}
        if data[0xA04]:
            ec,tc,ea,eb=struct.unpack('<HHII',data[0xA04]);radar=dict(enemy_coins=ec,enemy_total_coins=tc,event_data=ea,event_data_2=eb)
        if data[0x20C]:radar['marker_flags']=struct.unpack('<H',data[0x20C])[0]
        if data[0x20E]:
            a,b,x,y,radius=struct.unpack('<BBHHH',data[0x20E])
            radar.update(double_chances=a&3,double_active=bool(a&4),encryption_level=(a>>3)&3,can_modify_key=bool(a&32),fortified=bool(b&1),fortify_chances=b>>1,fortify_x_mm=x,fortify_y_mm=y,fortify_radius_mm=radius)
        if status:
            x,y,radius=struct.unpack_from('<HHH',status,41)
            radar.update(enemy_fortify_x_mm=x,enemy_fortify_y_mm=y,enemy_fortify_radius_mm=radius)
        # Scripts only see valid friend positions. No accidental missing=origin obstacles.
        nav_friends=[{**f,'radius':.4,'vx':0.,'vy':0.,'alive':f.get('alive',True)} for f in friends if f['position_valid']]
        return dict(api_version=3,observation_profile=PROFILE,tick=s['tick'],time=now,dt=.05,policy_dt=.05,
            team=team,controlled_ids=['7'],arena=s['arena'],robots=[me,*nav_friends],allies=friends,enemies=enemies,
            remaining=remaining,referee=dict(sources=sources,ack=ack,coins=coins,field=field,control=control,bases=bases,
                own_control=hp[10] if hp else None,enemy_control=hp[11] if hp else None,radar=radar,
                events=list(self.events[team])),
            autoaim=autoaim(me,sensors),local_feedback=dict(source='ideal_local_module',source_time=now,valid=True),
            radar_inference=self.inference_received[team].observation(now),
            command_feedback=self.execution.get(team,{}),commander=self.orders.get(team),messages=[],message_inboxes={'7':[]})


def autoaim(robot,candidates):
    from . import sdk
    targets=sorted(candidates,key=lambda r:(math.hypot(r['x']-robot['x'],r['y']-robot['y']),r['id']))
    for target in targets:
        if not sdk.ballistic_reachable(robot,target):continue
        distance=math.hypot(target['x']-robot['x'],target['y']-robot['y'])
        flight=distance/max(1,robot.get('muzzle_speed_mps',25))
        target={**target,'x':target['x']+target.get('vx',0)*flight,'y':target['y']+target.get('vy',0)*flight}
        aim=sdk.aim_at(robot,target);pitch=sdk.aim_pitch(robot,target)
        error=math.atan2(math.sin(aim-robot['aim']),math.cos(aim-robot['aim']))
        pitch_error=pitch-robot['pitch']
        return dict(valid=True,target_id=target['id'],distance=distance,aim=aim,pitch=pitch,
                    aligned=abs(error)<.08 and abs(pitch_error)<.06,yaw_error=error,pitch_error=pitch_error)
    return dict(valid=False)
