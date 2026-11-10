"""Training-only global state. Never imported by the portable actor encoder."""
import hashlib
import json
import math

ROBOT_FIELDS = (
    ('x',28),('y',15),('z',3),('vx',10),('vy',10),('vz',5),
    ('hp',1000),('max_hp',1000),('alive',1),('powered',1),('weak',1),
    ('grounded',1),('upright',1),('tipped',1),('transform_progress',1),
    ('ammo',500),('heat',500),('max_heat',500),('cooling_rate',100),
    ('power_limit',200),('buffer_energy_j',60),('level',10),
    ('respawn_progress',30),('respawn_required',30),('energy',250),
)
SLOTS=(1,3,4,6,7)
DESIGN_IDS=(2711,2731,2732,303,2761,2771,2772,2773)
ROBOT_SIZE=len(ROBOT_FIELDS)+6+len(DESIGN_IDS)+5
SIZE=10*ROBOT_SIZE+14
SCHEMA={'version':1,'size':SIZE,'robot_fields':ROBOT_FIELDS,'slots':SLOTS,
        'design_ids':DESIGN_IDS,'additional':['heading_sin','heading_cos','aim_sin','aim_cos','pitch','fold_allowed',
        'model_onehot','speed_multiplier','force_multiplier','efficiency_multiplier','aim_delay','aim_sigma'],
        'global':['time','remaining','own/enemy control/resources/base hp/outpost hp','own/enemy assembly level','projectile_count','finished']}
SCHEMA_HASH=hashlib.sha256(json.dumps(SCHEMA,sort_keys=True).encode()).hexdigest()


def encode(frame,team,capabilities=None):
    """Current global values only; no hidden future samples or reward history."""
    values=[];flip=team=='blue';other='blue' if team=='red' else 'red'
    def number(v):
        return float(v) if isinstance(v,(int,float,bool)) and math.isfinite(float(v)) else 0.
    for side in (team,other):
        for slot in SLOTS:
            r=next((r for r in frame['robots'] if r['team']==side and r['slot']==slot),{})
            for key,scale in ROBOT_FIELDS:
                value=number(r.get(key,0))
                if flip and key in ('x','y'):value=(28 if key=='x' else 15)-value if r else 0.
                if flip and key in ('vx','vy'):value=-value
                values.append(value/scale)
            heading=number(r.get('heading',0))+(math.pi if flip else 0)
            aim=number(r.get('aim',0))+(math.pi if flip else 0)
            values.extend((math.sin(heading),math.cos(heading),math.sin(aim),math.cos(aim),
                           number(r.get('pitch',0))/math.pi,float(bool(r.get('transform_allowed')))))
            model=r.get('model_slot',slot)
            values.extend(float(model==m) for m in DESIGN_IDS)
            profile=(capabilities or {}).get(f'{side}:{slot}',{})
            mobility=profile.get('mobility',{});aim_profile=profile.get('aim',{})
            values.extend(number(mobility.get(k,1)) for k in ('speed_reference_mps','motor_force_max_n','drive_efficiency'))
            values.extend((number(aim_profile.get('delay_s',0)),number(aim_profile.get('sigma_deg',0))/5))
    now=number(frame.get('time'));values.extend((now/420,(number(frame.get('duration',420))-now)/420))
    for side in (team,other):
        t=frame.get('teams',{}).get(side,{})
        values.extend((number(t.get('control'))/1680,number(t.get('resources'))/5000,
                       number(frame.get('bases',{}).get(side,{}).get('hp'))/7500,
                       number(frame.get('outposts',{}).get(side,{}).get('hp'))/1500))
    for side in (team,other):
        values.append(number(frame.get('assemblies',{}).get(side,{}).get('level',0))/4)
    values.extend((len(frame.get('projectiles',[]))/100,float(frame.get('status')=='finished')))
    if len(values)!=SIZE or not all(math.isfinite(v) for v in values):raise ValueError('Invalid critic state')
    return values
