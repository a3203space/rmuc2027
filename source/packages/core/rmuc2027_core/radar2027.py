"""2027 radar referee state: marking, double vulnerability and command counters."""
import math
import random
import statistics
import struct
from collections import defaultdict, deque

class RadarReferee:
    def __init__(self, seed=0):
        self.rng=random.Random(seed)
        self.teams={t:dict(level=1,key=self.key(),can_modify=True,verify_after=0.,double_used=0,
            vulnerability_seconds=0.,double_until=0.,fortify_used=0,center=None,hits=[],shots=0) for t in ('red','blue')}
        self.marks={};self.time=0.;self.faults={};self.last_mark=-.2
        self.map_uploads=defaultdict(deque)

    def submit_positions(self, team, now, payload, snapshot):
        """Receive the 48-byte 0x0305 payload; its 5 Hz cap is independent of 0x0301."""
        if len(payload)!=48:raise ValueError('0x0305 requires 48 bytes')
        queue=self.map_uploads[team]
        while queue and queue[0]<=now-1+1e-9:queue.popleft()
        if len(queue)>=5:return False
        queue.append(now)
        points=struct.unpack('<24H',payload)
        other='blue' if team=='red' else 'red'
        for owner,base in ((other,0),(team,12)):
            for slot,offset in ((1,0),(3,4),(4,6),(7,10)):
                x,y=points[base+offset:base+offset+2]
                if x==y==0:continue  # Protocol explicitly means no coordinate report.
                robot=next(r for r in snapshot['robots'] if r['id']==f'{owner}-{slot}')
                self.mark(team,robot['id'],now,math.hypot(x/100-robot['x'],y/100-robot['y']),robot['alive'])
        return True

    def key(self):
        return ''.join(self.rng.choice('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz') for _ in range(6))

    def mark(self,radar,target,now,error,alive=True):
        m=self.marks.setdefault((radar,target),dict(p=0.,x=0.,last=now,alive=alive))
        if alive and not m['alive']:m['x']=0.
        m['alive']=alive
        if error<1.6:
            delta=1. if error<.8 else .5
            m['x']=(max(0.,m['x'])+delta)
        else:m['x']=min(0.,m['x'])-.8
        m['p']=max(0.,min(150.,m['p']+m['x']));m['last']=now
        return m['p']

    def vulnerability(self,radar,target):
        other='blue' if radar=='red' else 'red'
        # A team with a higher interference level is protected from the opposing radar.
        if self.teams[other]['level']>self.teams[radar]['level']:return 0.
        p=self.marks.get((radar,target),{}).get('p',0.)
        return (.2 if p>=120 else .15 if p>=100 else 0.)*(2 if self.time<self.teams[radar]['double_until'] else 1)

    def update(self,snapshot):
        now=snapshot['time'];dt=max(0.,now-self.time)
        # Integrate the previous state's vulnerability over this simulation interval.
        for team,t in self.teams.items():
            if any(r['team']!=team and r['alive'] and self.vulnerability(team,r['id'])>0 for r in snapshot['robots']):
                t['vulnerability_seconds']+=dt
        self.time=now
        for (team,target),m in self.marks.items():
            robot=next((r for r in snapshot['robots'] if r['id']==target),None)
            if robot:
                if robot['alive'] and not m['alive']:m['x']=0.
                m['alive']=robot['alive']
        for event in snapshot['events']:
            team=event.get('team')
            if team not in self.teams:continue
            if event['type']=='shot':self.teams[team]['shots']+=1
            if event['type']=='armor_hit' and event.get('target_team')!=team and 'armor_xy_mm' in event:
                self.teams[team]['hits'].append(tuple(event['armor_xy_mm']))
        if now-self.last_mark>=.2-1e-9:
            self.last_mark=now
            for team in self.teams:
                points=[0]*24
                other='blue' if team=='red' else 'red'
                for owner,base in ((other,0),(team,12)):
                    for slot,offset in ((1,0),(3,4),(4,6),(7,10)):
                        r=next(r for r in snapshot['robots'] if r['id']==f'{owner}-{slot}')
                        fault=self.faults.get((team,r['id']),{})
                        if not fault.get('drop'):
                            points[base+offset:base+offset+2]=[max(0,min(65535,math.floor(v*100+.5))) for v in (r['x']+fault.get('error',0.),r['y'])]
                self.submit_positions(team,now,struct.pack('<24H',*points),snapshot)
        for (team,target),m in self.marks.items():
            while now-m['last']>=.5-1e-9:
                self.mark(team,target,m['last']+.5,math.inf,m['alive'])

    def command(self,team,payload):
        if len(payload)!=13:raise ValueError('0x0121 requires 13 bytes')
        double,mode,key,fortify,x,y=struct.unpack('<BB6sBHH',payload)
        t=self.teams[team];other=self.teams['blue' if team=='red' else 'red'];result={}
        opportunities=min(2,math.floor((t['vulnerability_seconds']+1e-8)/60))
        if double==t['double_used']+1 and double<=opportunities:
            t['double_used']=double;t['double_until']=max(self.time,t['double_until'])+30
            result['double']=True
        elif double!=t['double_used']:result['double']=False
        valid=all(chr(c) in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz' for c in key)
        if mode==1:
            result['key']=valid and t['can_modify']
            if result['key']:t['key']=key.decode();t['can_modify']=False
        elif mode==2:
            result['key']=False
            if valid and self.time>=t['verify_after']:
                t['verify_after']=self.time+10
                if key.decode()==other['key']:
                    result['key']=True
                    if other['level']<3:
                        other.update(level=other['level']+1,key=self.key(),can_modify=True)
        elif mode!=0:result['key']=False
        if (fortify==t['fortify_used']+1 and fortify<=int(self.time//60) and len(t['hits'])>=10
                and self.area_fraction(team) is not None):
            t['center']=(x,y);t['fortify_used']=fortify;result['fortify']=True
        elif fortify!=t['fortify_used']:result['fortify']=False
        return result

    def area_fraction(self,team):
        def accuracy(t):
            hits=t['hits']
            if not hits or not t['shots']:return None
            x=statistics.median(p[0] for p in hits);y=statistics.median(p[1] for p in hits)
            sigma=statistics.median(math.hypot(p[0]-x,p[1]-y) for p in hits)
            return len(hits)/t['shots']/sigma if sigma>0 else None
        a=accuracy(self.teams[team]);b=accuracy(self.teams['blue' if team=='red' else 'red'])
        # User-confirmed conservative treatment of the rulebook's undefined cases.
        if a is None or b is None or a==b==0:return None
        return a*a/(2*b*b+a*a)

    def feedback(self,team):
        t=self.teams[team];chances=max(0,min(2,int((t['vulnerability_seconds']+1e-8)//60))-t['double_used'])
        first=chances | int(self.time<t['double_until'])<<2 | t['level']<<3 | int(t['can_modify'])<<5
        fraction=self.area_fraction(team)
        second=int(t['center'] is not None and fraction is not None) | min(7,max(0,int(self.time//60)-t['fortify_used']))<<1
        center=t['center'] or (0,0)
        radius=math.floor(math.sqrt(135*125*fraction/math.pi)+.5) if t['center'] and fraction is not None else 0
        return struct.pack('<BBHHH',first,second,*center,radius)

    def effects(self,snapshot):
        vulnerability={r['id']:self.vulnerability('blue' if r['team']=='red' else 'red',r['id']) for r in snapshot['robots'] if r['role']!='aerial'}
        armor={team:dict(center=t['center'],fraction=self.area_fraction(team)) for team,t in self.teams.items() if t['center'] is not None and self.area_fraction(team) is not None}
        return dict(vulnerability=vulnerability,armor=armor)
