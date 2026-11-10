"""Portable six-output sentry actor contract and independent 20 Hz executor."""
import hashlib
import json
import math
try:
    import sdk as _sdk
except ImportError:
    from rmuc2027_core import sdk as _sdk

PROFILE = 'sentry_referee_2027_v1'
ACTION_NAMES = ('goal_x','goal_y','fold','jump','spin','fire')
PACKETS = ('0001','0003','0101','0104','0201','0202','0203','0204','0208','0209','020B','020C','020D','020E','0A01','0A02','0A03','0A04','0A05','F001')
# (name, divisor, authoritative source). No clipping: distinct legal values stay distinct.
FEATURES = [(name,scale,'ideal_local_module') for name,scale in (
    ('self.x',28),('self.y',15),('self.z',3),('self.vx',10),('self.vy',10),('self.vz',5),
    ('self.heading_sin',1),('self.heading_cos',1),('self.yaw_rate',math.tau),('self.grounded',1),
    ('self.upright',1),('self.tipped',1),('self.leg_left',1),('self.leg_right',1),('self.fold_progress',1),
    ('self.can_fold',1),('self.can_jump',1),('self.legged',1))]
FEATURES += [(name,scale,source) for name,scale,source in (
    ('self.hp',1000,'0201'),('self.max_hp',1000,'0201'),('self.level',10,'0201'),('self.alive',1,'0201'),
    ('self.powered',1,'0201'),('self.power_limit',200,'0201'),('self.max_heat',500,'0201'),
    ('self.cooling_rate',100,'0201'),('self.speed_limit',30,'0201'),('self.heat',500,'0202'),
    ('self.buffer_j',60,'0202'),('self.ammo',500,'0208'),('self.fortress_ammo',500,'0208'),('coins',5000,'0208'))]
FEATURES += [(f'self.power_output.{i}',1,'0201') for i in range(3)]
FEATURES += [(f'self.referee_pose.{k}',scale,'0203') for k,scale in (('x',28),('y',15),('muzzle_angle_deg',360))]
for owner in ('self',*[f'enemy.{s}' for s in (1,3,4,7)]):
    FEATURES += [(f'{owner}.buff.{k}',100,'0204' if owner=='self' else '0A05') for k in ('heal','cooling','defense','vulnerability','attack')]
FEATURES += [(f'self.rfid.{i}',1,'0209') for i in range(41)]
FEATURES += [(f'self.posture.{i}',1,'020D') for i in ('attack','defense','move')]
FEATURES += [('self.enhanced',1,'020D')]
FEATURES += [(f'self.{kind}_remaining.{i}',180 if kind=='posture' else 15,'020D') for kind in ('posture','enhanced') for i in range(3)]
FEATURES += [(f'ack.{k}',scale,'020D') for k,scale in (('local_ammo',2047),('remote_ammo',15),('remote_heal',15),('free_revive',1),('paid_revive',1),('paid_cost',1023),('disengaged',1),('ammo_quota',2047),('rune_available',1))]
for slot in (1,3,4,6):
    FEATURES += [(f'ally.{slot}.{k}',scale,'F001/020B' if k in ('x','y','position_valid') else '0003') for k,scale in (('x',28),('y',15),('position_valid',1),('hp',1000),('hp_valid',1))]
for slot in (1,3,4,7):
    FEATURES += [(f'enemy.{slot}.{k}',scale,source) for k,scale,source in (
        ('x',28,'0A01'),('y',15,'0A01'),('vx',10,'position_difference'),('vy',10,'position_difference'),
        ('velocity_estimated',1,'position_difference'),('position_valid',1,'0A01'),('hp',1000,'0A02'),
        ('hp_valid',1,'0A02'),('ammo',500,'0A03'),('alive',1,'0A05'),('weak',1,'0A05'),('invulnerable',1,'0A05'),('status_valid',1,'0A05'))]
FEATURES += [(k,scale,source) for k,scale,source in (('remaining',420,'0001'),('own_control',1680,'0003'),('enemy_control',1680,'0003'),('base.own',7500,'0003'),('base.enemy',7500,'0003'),('base.red_outpost',1500,'0003'),('base.blue_outpost',1500,'0003'),('enemy_coins',5000,'0A04'),('enemy_total_coins',10000,'0A04'))]
FEATURES += [(f'control.{side}_{kind}.{state}',1,'0101') for side in ('red','blue') for kind in ('outpost','base') for state in range(3)]
FEATURES += [(f'field.bit.{i}',1,'0101') for i in range(28)]
FEATURES += [(f'radar.field.bit.{i}',1,'0A04') for i in range(24)]
FEATURES += [(f'radar.state.bit.{i}',1,'0A04') for i in range(30)]
FEATURES += [(f'radar.marker.{i}',1,'020C') for i in (0,2,3,5,6,8,9,11)]
FEATURES += [(f'radar.{k}',scale,'020E') for k,scale in (('double_chances',2),('double_active',1),('encryption_level',3),('can_modify_key',1),('fortified',1),('fortify_chances',7),('fortify_x_mm',1000),('fortify_y_mm',1000),('fortify_radius_mm',1000))]
FEATURES += [(f'radar.enemy_fortify_{k}_mm',1000,'0A05') for k in ('x','y','radius')]
FEATURES += [(f'event.{k}',scale,'referee_event') for k,scale in (('hurt',1),('shot',1),('warning_level',4),('age',10))]
FEATURES += [(f'packet.{cmd}.{key}',1 if key=='valid' else 5,'receiver') for cmd in PACKETS for key in ('valid','age')]
FEATURES += [(name,scale,'module_feedback') for name,scale in (
    ('aim.valid',1),('aim.distance',30),('aim.aligned',1),('aim.yaw_error',math.pi),('aim.pitch_error',math.pi),
    ('goal.x',28),('goal.y',15),('goal.age',5),('nav.spin_requested',1),('nav.spin_accepted',1),('nav.spin_guard',1),
    ('nav.fire_requested',1),('nav.fire_accepted',1),('nav.jump_requested',1),('nav.jump_accepted',1),('nav.jump_cooldown',2),
    ('nav.fold_requested',1),('nav.fold_accepted',1),('commander.valid',1),('commander.x',28),('commander.y',15),('commander.age',10))]
FEATURES += [('weapon.shots',1,'0207'),('command.transport_accepted',1,'local_transmitter')]
FEATURES += [('commander.key',255,'0303'),('commander.targets_robot',1,'0303')]
FEATURES += [(f'commander.target.{slot}',1,'0303') for slot in (1,3,4,6,7)]
FEATURE_INDEX={name:i for i,(name,_,_) in enumerate(FEATURES)}
SCHEMA_HASH=hashlib.sha256(json.dumps(FEATURES,separators=(',',':')).encode()).hexdigest()


def schema():
    return dict(version=3,obs_size=len(FEATURES),action_size=6,profile=PROFILE,schema_hash=SCHEMA_HASH)


def encode(obs,state):
    if obs.get('observation_profile')!=PROFILE:raise ValueError('Policy v3 requires referee-scoped observation')
    me=next(r for r in obs['robots'] if r['slot']==7)
    d=_sdk._design(me);f={};flip=obs['team']=='blue'
    def xy(x,y):return (28-x,15-y) if flip else (x,y)
    x,y=xy(me['x'],me['y']);heading=me['heading']+(math.pi if flip else 0)
    for key in ('z','vx','vy','vz','grounded','upright','tipped','hp','max_hp','level','alive','powered','power_limit','max_heat','cooling_rate','heat','ammo','fortress_ammo'):
        f['self.'+key]=me.get(key,0)
    for key in ('vx','vy'):f['self.'+key]*=-1 if flip else 1
    legs=me.get('legs',[0,0]);angular=me.get('angular_velocity',[0,0,0])
    f.update({'self.x':x,'self.y':y,'self.heading_sin':math.sin(heading),'self.heading_cos':math.cos(heading),
        'self.yaw_rate':angular[2],'self.leg_left':legs[0],'self.leg_right':legs[1],
        'self.fold_progress':me.get('transform_progress',0),'self.can_fold':bool(d.get('transform_allowed')),
        'self.can_jump':bool(d.get('capabilities',{}).get('jump')),'self.legged':d.get('chassis_type')=='wheel_legged',
        'self.buffer_j':me.get('buffer_energy_j',0),'self.speed_limit':me.get('muzzle_speed_mps',0)})
    for k,v in me.get('buffs',{}).items():f['self.buff.'+k]=v
    for i in range(3):f[f'self.power_output.{i}']=(me.get('power_outputs',0)>>i)&1
    for k,v in me.get('referee_pose',{}).items():f[f'self.referee_pose.{k}']=v
    for i,v in enumerate(me.get('rfid',[])):f[f'self.rfid.{i}']=v
    for p in ('attack','defense','move'):f['self.posture.'+p]=me.get('posture')==p
    f['self.enhanced']=me.get('posture_enhanced',False)
    for kind in ('posture','enhanced'):
        for i,v in enumerate(me.get(kind+'_remaining',[])):f[f'self.{kind}_remaining.{i}']=v
    ref=obs['referee']
    for k,v in ref['ack'].items():f['ack.'+k]=v
    for label,robots in (('ally',obs['allies']),('enemy',obs['enemies'])):
        for r in robots:
            for k,v in r.items():
                if isinstance(v,(int,float,bool)):f[f'{label}.{r["slot"]}.{k}']=v
            if r.get('position_valid'):
                x,y=xy(r['x'],r['y']);f[f'{label}.{r["slot"]}.x']=x;f[f'{label}.{r["slot"]}.y']=y
            for k in ('vx','vy'):
                name=f'{label}.{r["slot"]}.{k}'
                if name in f:f[name]*=-1 if flip else 1
            for k,v in r.get('buffs',{}).items():f[f'{label}.{r["slot"]}.buff.{k}']=v
    for k in ('coins','own_control','enemy_control'):f[k]=ref[k] or 0
    for k,v in ref['bases'].items():f['base.'+k]=v
    f['remaining']=obs['remaining'] or 0
    for k in ('enemy_coins','enemy_total_coins'):f[k]=ref['radar'].get(k,0)
    for i in range(28):f[f'field.bit.{i}']=(ref['field'].get('event_data',0)>>i)&1
    for prefix,key,count in (('radar.field','event_data',24),('radar.state','event_data_2',30),('radar.marker','marker_flags',12)):
        for i in range(count):f[f'{prefix}.bit.{i}' if prefix!='radar.marker' else f'{prefix}.{i}']=(ref['radar'].get(key,0)>>i)&1
    for k,v in ref['radar'].items():
        if isinstance(v,(int,float,bool)):f['radar.'+k]=v
    events=ref.get('events',[])
    if events:
        last=events[-1];f['event.age']=obs['time']-last['source_time']
        f['event.hurt']=last['command']==0x206;f['event.shot']=last['command']==0x207
    warnings=[e for e in events if e['command']==0x104]
    if warnings:f['event.warning_level']=warnings[-1]['payload'][0]
    for key,value in ref['control'].items():
        for i in range(3):f[f'control.{key}.{i}']=value==i
    for cmd in PACKETS:
        packet=ref['sources'].get('0x'+cmd,{})
        f[f'packet.{cmd}.valid']=packet.get('valid',False)
        f[f'packet.{cmd}.age']=packet.get('age') or 0
    for k in ('valid','distance','aligned','yaw_error','pitch_error'):f['aim.'+k]=obs['autoaim'].get(k,0)
    goal=state.get('goal')
    if goal:f['goal.x'],f['goal.y']=xy(*goal)
    f['goal.age']=obs['time']-state.get('goal_time',obs['time'])
    trace=state.get('execution_trace',{})
    for k in ('spin','fire','jump','fold'):
        f['nav.'+k+'_requested']=trace.get('requested',{}).get(k,False)
        f['nav.'+k+'_accepted']=trace.get('accepted',{}).get(k,False)
    f['nav.spin_guard']=trace.get('spin_guard',False)
    f['nav.jump_cooldown']=_sdk.locomotion_context(me,obs,state.get('navigation',{}))['cooldown']
    order=obs.get('commander')
    f['weapon.shots']=obs.get('command_feedback',{}).get('shots',0)
    f['command.transport_accepted']=obs.get('command_feedback',{}).get('accepted',False)
    if order:
        f['commander.valid']=True;f['commander.x'],f['commander.y']=xy(order['x'],order['y'])
        f['commander.age']=obs['time']-order['source_time']
        f['commander.key']=order['key'];f['commander.targets_robot']=bool(order['target'])
        for slot in (1,3,4,6,7):f[f'commander.target.{slot}']=order['target']%100==slot
    values=[float(f.get(name,0))/scale for name,scale,_ in FEATURES]
    if not all(math.isfinite(v) for v in values):raise ValueError('Non-finite v3 observation')
    return values


def spin_guard(robot,obs,nav,fields,requested):
    """Brake before the swept hull reaches a roof; keep heading alignment active."""
    d=_sdk._design(robot);size=d.get('chassis_size_m',robot.get('chassis_size_m',[.6,.5,.4]))
    hull=math.hypot(size[0],size[1])/2
    speed=math.hypot(robot.get('vx',0),robot.get('vy',0));omega=robot.get('angular_velocity',[0,0,0])[2]
    decel=max(.3,min(d.get('brake_force_max_n',100)/max(1,d.get('mass_kg',25)),d.get('tire_friction',.5)*9.81)*.6)
    inertia=max(.01,d.get('inertia_kgm2',[.5,.5,.5])[2]);yaw_acc=max(.1,d.get('yaw_torque_max_nm',10)/inertia)
    margin=hull+.2+speed*max(.05,abs(omega)/yaw_acc)+speed*speed/(2*decel)
    hazards=[]
    for w in obs['arena'].get('obstacles',[]):
        if 'tunnel' not in w.get('id','') or 'roof' not in w.get('id',''):continue
        c=(w['x']+w['w']/2,w['y']+w['h']/2);a=w.get('rotation',0);ca,sa=math.cos(a),math.sin(a)
        dx,dy=robot['x']-c[0],robot['y']-c[1];x=ca*dx+sa*dy;y=-sa*dx+ca*dy
        gap=math.hypot(max(0,abs(x)-w['w']/2),max(0,abs(y)-w['h']/2))
        if gap<margin or nav.get('spin_roof')==w['id'] and gap<hull+.25:
            hazards.append((gap,w,a))
    guarded=bool(hazards)
    if guarded:
        _,wall,angle=min(hazards,key=lambda h:(h[0],h[1]['id']));nav['spin_roof']=wall['id']
        fields.pop('yaw_rate',None)
        heading=fields.get('heading',angle)
        if 'heading' not in fields and abs(math.atan2(math.sin(heading+math.pi-robot['heading']),math.cos(heading+math.pi-robot['heading'])))<math.pi/2:heading+=math.pi
        fields['heading']=heading
        error=abs(math.atan2(math.sin(heading-robot['heading']),math.cos(heading-robot['heading'])))
        if abs(omega)>=.2 or error>=math.radians(5):fields.update(move=(0.,0.),brake=True)
    else:
        nav.pop('spin_roof',None)
        if requested and robot.get('alive') and robot.get('powered'):
            fields['yaw_rate']=math.tau
    return guarded


def resource_command(robot,obs,state,moving):
    ack=obs['referee']['ack'];now=obs['time']
    if not ack:return None
    counters=state.setdefault('referee_commands',dict(local_ammo=ack['local_ammo'],remote_ammo=ack['remote_ammo'],remote_heal=ack['remote_heal']))
    for k in ('local_ammo','remote_ammo','remote_heal'):counters[k]=max(counters[k],ack[k])
    pending=any(counters[k]!=ack[k] for k in counters)
    coins=obs['referee']['coins'] or 0
    if not pending and robot['alive'] and now>=state.get('next_exchange',0):
        cards=robot.get('rfid',[0]*41)
        local=bool(cards[0] or cards[2 if obs['team']=='red' else 3] or cards[10])
        if robot['ammo']<40 and ack['ammo_quota']>=100:
            if local and coins>=150 and counters['local_ammo']<=1947:
                counters['local_ammo']+=100;state['next_exchange']=now+7
            elif ack['disengaged'] and coins>=200 and counters['remote_ammo']<15:
                counters['remote_ammo']+=1;state['next_exchange']=now+7
        elif robot['hp']<robot['max_hp']*.4 and ack['disengaged'] and coins>=250 and counters['remote_heal']<15:
            counters['remote_heal']+=1;state['next_exchange']=now+7
    posture=3 if moving else 1
    # The referee owns cooldown and successful acknowledgements; a repeated word is idempotent.
    word=(int(ack['free_revive']) | counters['local_ammo']<<2 | counters['remote_ammo']<<13 | counters['remote_heal']<<17 | posture<<21)
    return word


def execute(obs,state,navigator,config,resource_builder=None):
    if obs.get('observation_profile')!=PROFILE:raise ValueError('Strict sentry observation required')
    me=next(r for r in obs['robots'] if r['slot']==7);nav=state.setdefault('navigation',{})
    intent=state.get('sentry_intent',dict(fold=-1,jump=-1,spin=-1,fire=-1))
    requested={k:intent[k]>.5 for k in ('fold','jump','spin','fire')}
    goal=state.get('goal');rejected=[]
    if me['alive'] and goal is not None:
        fields=_sdk.navigate_action(me,goal,obs,nav,locomotion_intent=dict(transform=intent['fold'],jump=intent['jump']))
    else:fields=dict(move=(0.,0.),jump=0.,brake=True)
    guarded=spin_guard(me,obs,nav,fields,requested['spin'])
    if guarded and requested['spin']:rejected.append('tunnel_spin_interlock')
    aim=obs['autoaim'];fire=requested['fire'] and aim['valid'] and aim.get('aligned',False)
    if aim['valid']:fields.update(aim=aim['aim'],pitch=aim['pitch'])
    fire=fire and me['alive'] and bool(me.get('power_outputs',0)&4) and me['ammo']>0 and me['heat']+10<=me['max_heat']
    fire=fire and not 0<me.get('transform_progress',0)<1
    fields['fire']=bool(fire)
    if requested['fire'] and not fire:rejected.append('fire_guard')
    command=(resource_builder or resource_command)(me,obs,state,math.hypot(*fields['move'])>.05)
    if command is not None:fields['sentry_cmd']=command
    accepted=dict(fold=bool(fields.get('transform')),jump=bool(fields.get('jump')),spin=bool(fields.get('yaw_rate')),fire=bool(fire))
    rejected+=nav.get('rejected',[])
    if requested['fold'] and not _sdk._design(me).get('transform_allowed'):rejected.append('fold_unsupported')
    state.update(mode='policy' if me['alive'] else 'dead',executed_goal=goal,
        execution_trace=dict(time=obs['time'],requested=requested,accepted=accepted,
            executed=dict(fold=me.get('transform_progress',0),yaw_rate=me.get('angular_velocity',[0,0,0])[2],grounded=me.get('grounded',True),
                shots=obs.get('command_feedback',{}).get('shots',0),feedback_time=obs.get('command_feedback',{}).get('execution_time')),
            spin_guard=guarded,rejections=rejected))
    action=_sdk.action(**fields);state['executed_action']=action
    return {'7':action}
