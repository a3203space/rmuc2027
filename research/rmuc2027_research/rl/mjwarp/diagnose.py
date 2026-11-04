"""Opt-in motion recording; runs scripted AI with the same settings as the native viewer."""
import argparse
from collections import deque
import json
import math
from pathlib import Path
import time


class MotionTrace:
    def __init__(self, directory, battle, config):
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=False)
        self.file = (self.path/'motion.jsonl').open('w')
        self.battle = battle
        battle.motion_trace = True
        self.started = time.monotonic()
        self.history = {}
        self.alerts = []
        self.active = set()
        self.tracks = {}
        self.substeps = deque(maxlen=240)
        self.count = 0
        from ..session import provenance
        metadata = {'config': config, 'battle': battle.metadata, 'provenance': provenance(config),
                    'record_order': 'result frame + commands held during the preceding step'}
        (self.path/'metadata.json').write_text(json.dumps(metadata, indent=2))

    def physics(self, request, result):
        # Bounded pre-event evidence; no disk writes on the ordinary physics path.
        self.substeps.append({'bodies': [dict(b) for b in request.get('bodies',[]) if b.get('dynamic')], 'result': result})

    def record(self, frame):
        now = frame['time']
        navigation = {f'{team}:{slots}':getattr(policy,'navigation_trace',{})
                      for team,slots,policy in self.battle.policies}
        rows = []
        keys = ('id','team','slot','model_slot','x','y','z','vx','vy','vz','heading','upright',
                'grounded','tipped','alive','hp','legs','transform_progress','powered',
                'battery_remaining_wh','motor_power_w','power_cutoff_until','quaternion','angular_velocity','pitch','wheel_angular_velocity','steering_angles','wheel_loads_n')
        for robot in frame['robots']:
            if robot['role']=='aerial':
                continue
            rid = robot['id']
            action = self.battle.held_commands[robot['team']].get(str(robot['slot']),{})
            row = {k:robot[k] for k in keys if k in robot}
            row['action'] = action
            rows.append(row)
            history = self.history.setdefault(rid,deque())
            history.append((now,robot['x'],robot['y'],math.hypot(*action.get('move',(0,0)))))
            while len(history)>1 and history[1][0] <= now-2:
                history.popleft()
            def find_navigation(value):
                if not isinstance(value, dict):
                    return None
                if 'navigation' in value:
                    candidate = value['navigation'].get(str(robot['slot']))
                    if candidate:
                        return candidate
                for child in value.values():
                    found = find_navigation(child)
                    if found:
                        return found
                return None
            nav = next((found for key,value in navigation.items()
                        if key.startswith(robot['team']+':')
                        for found in [find_navigation(value)] if found), None)
            kind = None
            if robot.get('alive') and robot.get('powered',True):
                if robot.get('tipped'):
                    kind='tipped'
                elif now-history[0][0]>=1.99 and all(p[3]>.2 for p in history):
                    if max(math.hypot(p[1]-history[0][1],p[2]-history[0][2]) for p in history)<.1:
                        kind='commanded_without_progress'
            if (kind is None and robot.get('alive') and robot.get('powered', True)
                    and nav and now-nav.get('time', -999) < .6 and nav.get('goal')
                    and math.hypot(robot['x']-nav['goal'][0],robot['y']-nav['goal'][1]) > .5
                    and now-history[0][0] >= 1.99
                    and all(p[3] < .01 for p in history)
                    and max(math.hypot(p[1]-history[0][1],p[2]-history[0][2]) for p in history)<.1):
                kind = 'zero_command_before_goal'
            event = (rid,kind)
            if kind and event not in self.active:
                alert = {'time':now,'robot':rid,'kind':kind,'review_required':True}
                self.alerts.append(alert)
                if self.substeps:
                    (self.path/f'physics-{len(self.alerts):04}.json').write_text(json.dumps(list(self.substeps)))
            self.active = {x for x in self.active if x[0]!=rid}
            if kind:
                self.active.add(event)
            self.tracks.setdefault(rid,[]).append((robot['x'],robot['y'],robot['z'],now))
        self.file.write(json.dumps({'tick':frame['tick'],'time':now,'robots':rows,
                                  'navigation':navigation,'events':frame.get('events',[])})+'\n')
        self.count += 1
        if self.count%100==0:
            self.file.flush()

    def close(self):
        self.file.close()
        self.battle.motion_trace = False
        report = {'frames':self.count,'wall_seconds':time.monotonic()-self.started,
                  'alerts':self.alerts,'note':'Flags require inspection; stationary zero-command units are not automatically stuck.'}
        (self.path/'summary.json').write_text(json.dumps(report,indent=2))
        svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="600" viewBox="0 0 28 15"><rect width="28" height="15" fill="#111"/>']
        for rid,points in self.tracks.items():
            color = '#ff6666' if rid.startswith('red') else '#66aaff'
            coordinates = ' '.join(f'{x:.3f},{15-y:.3f}' for x,y,_,_ in points)
            svg.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width=".035"><title>{rid}</title></polyline>')
            if points:
                x,y,_,_ = points[-1]
                svg.append(f'<text x="{x}" y="{15-y}" font-size=".3" fill="{color}">{rid}</text>')
        (self.path/'trajectories.svg').write_text(''.join(svg)+'</svg>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/rl/sentry-mjwarp-hour.json')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--backend',choices=('rapier','mjwarp'),default='mjwarp')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--duration',type=int,default=120)
    parser.add_argument('--trace-dir',required=True)
    args = parser.parse_args()
    if not 1 <= args.duration <=420:
        parser.error('duration must be in [1,420]')
    from ..session import load_config,resource
    from ...strategies.battle import Battle
    cfg = load_config(args.config)
    sources = (resource(cfg['friendly']),resource(cfg['opponent']))
    red,blue = sources if cfg['team']=='red' else sources[::-1]
    with Battle(red,blue,args.seed,args.duration,cfg['map'],sandbox='trusted-local', side_lineups=cfg.get('lineups'),
                timeout=cfg['policy_timeout_s'],record_replay=False,
                navigation_mode=cfg['navigation_mode'],battle_settings={'drones_enabled':cfg['drones_enabled'], 'timed_assembly':cfg['timed_assembly']},
                fixed_policy_hz=cfg['fixed_policy_hz'],physics_backend=args.backend,sim_device=args.device) as battle:
        trace = MotionTrace(args.trace_dir,battle,cfg)
        if battle.match.external_physics is not None:
            physics = battle.match.external_physics
            step = physics.step
            def recorded(request):
                result = step(request)
                trace.physics(request,result)
                return result
            physics.step = recorded
        try:
            while battle.match.status!='finished':
                frame = battle.step()
                trace.record(frame)
                if battle.diagnostics:
                    raise RuntimeError(str(battle.diagnostics[-1]))
                if frame['tick']%100==0:
                    print(f"sim={frame['time']:.1f}s alerts={len(trace.alerts)}",flush=True)
        finally:
            trace.close()
    print(args.trace_dir,flush=True)

if __name__=='__main__':
    main()
