"""Paired SDK ability trials at the actual control frequency, without training.

This diagnoses locomotion transfer; it does not establish trained-policy match
transfer or win rates. Controllers consume scoped public observations only.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
from rmuc2027_core.engine import Match,DT
from rmuc2027_core import sdk
from ..session import load_config,provenance
from .arena import ArenaMotion


def write_json(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def inside(robot,rectangle):
    return (rectangle['x']<=robot['x']<=rectangle['x']+rectangle['w'] and
            rectangle['y']<=robot['y']<=rectangle['y']+rectangle['h'])


def trial(model,seed,backend,control_hz,horizon,output,device='cuda:0'):
    design=sdk.DESIGNS[str(model)];slot=7 if design['role']=='sentry' else 1 if design['role']=='heavy' else 3
    ability='jump' if design.get('capabilities',{}).get('jump') else 'tunnel' if design.get('capabilities',{}).get('tunnel') else 'bumps'
    lineups={t:[dict(slot=s,model=model if s==slot else m) for s,m in [(1,2711),(3,2731),(4,2732),(6,2761),(7,2771)]] for t in ('red','blue')}
    # Nine nearby endpoints vary final approach without changing map geometry.
    dy=((seed%3)-1)*.1;dx=(((seed//3)%3)-1)*.1
    goals={'red':(12.+dx,7.5+dy),'blue':(16.-dx,7.5-dy)}
    states={t:{} for t in lineups};metrics={t:dict(reached_at=None,settled_at=None,settled_since=None,ability_enter=None,ability_exit=None,
            folded=False,jump_commands=0,tipped=False,success=False) for t in lineups}
    commands={};interval=round(1/(control_hz*DT));started=time.perf_counter()
    adapter=ArenaMotion(device) if backend=='warp_kinematic' else None
    with Match(seed=seed,lineups=lineups,physics_adapter=adapter,battle_settings={'drones_enabled':False,'timed_assembly':True}) as match:
        arena=match.snapshot()['arena']
        roofs={t:next(w for w in arena['obstacles'] if w['id']==t+'_road_tunnel_roof') for t in lineups}
        bumps={t:next(w for w in arena['terrain'] if w['id']==t+'_road_bumps') for t in lineups}
        frame=match.snapshot()
        with output.open('w') as stream:
            for tick in range(round(horizon/DT)+1):
                controlled={t:next(r for r in frame['robots'] if r['team']==t and r['slot']==slot) for t in lineups}
                for team,r in controlled.items():
                    m=metrics[team];m['tipped']|=bool(r.get('tipped'));m['folded']|=r.get('transform_progress',0)>=.99
                    if ability=='jump':
                        if m['ability_enter'] is None and m['jump_commands'] and not r.get('grounded',True):m['ability_enter']=frame['time']
                        x=r['x'] if team=='red' else 28-r['x']
                        if m['ability_enter'] is not None and m['ability_exit'] is None and x>=10.65 and r.get('grounded'):m['ability_exit']=frame['time']
                    else:
                        region=roofs[team] if ability=='tunnel' else bumps[team]
                        if inside(r,region) and m['ability_enter'] is None:m['ability_enter']=frame['time']
                        local_x=r['x'] if team=='red' else 28-r['x']
                        local_y=r['y'] if team=='red' else 15-r['y']
                        crossed=local_y>=13.235 if ability=='tunnel' else local_x>=7.65
                        if m['ability_enter'] is not None and m['ability_exit'] is None and crossed:m['ability_exit']=frame['time']
                    # Body origin is not a tyre/terrain contact point. Do not
                    # reject an otherwise settled physical chassis using an
                    # arbitrary origin-to-map-height tolerance.
                    at_goal=math.dist((r['x'],r['y']),goals[team])<.25 and r.get('grounded')
                    if m['reached_at'] is None and at_goal:m['reached_at']=frame['time']
                    if at_goal and math.hypot(r.get('vx',0.),r.get('vy',0.))<.2:
                        if m['settled_since'] is None:m['settled_since']=frame['time']
                        if m['settled_at'] is None and frame['time']-m['settled_since']>=.3-1e-12:m['settled_at']=frame['time']
                    else:m['settled_since']=None
                if tick%2==0:
                    stream.write(json.dumps(dict(time=frame['time'],robots=[dict(
                        {k:r.get(k) for k in ('id','x','y','z','vx','vy','vz','grounded','transform_progress','upright','tipped')},
                        public_ground_z=sdk.terrain_height(arena,r['x'],r['y'])) for r in controlled.values()],state=states),allow_nan=False)+'\n')
                if all(m['settled_at'] is not None for m in metrics.values()) or tick==round(horizon/DT):break
                if tick%interval==0:
                    commands={}
                    for team in lineups:
                        if metrics[team]['settled_at'] is not None:fields={'move':(0.,0.),'brake':True}
                        else:
                            obs=match.observation(team,[slot]);obs['policy_dt']=1/control_hz
                            robot=next(r for r in obs['robots'] if r['slot']==slot)
                            fields=sdk.navigate_action(robot,goals[team],obs,states[team])
                        metrics[team]['jump_commands']+=int(fields.get('jump',0)>0)
                        commands[team]={str(slot):sdk.action(**fields)}
                frame=match.step(commands)
        for team,m in metrics.items():
            m['success']=(m['settled_at'] is not None and m['ability_enter'] is not None and m['ability_exit'] is not None
                          and not m['tipped'] and (ability!='tunnel' or m['folded']))
            m['final']={k:controlled[team].get(k) for k in ('x','y','z','grounded','upright','tipped')}
            r=controlled[team]
            m['final']['public_ground_z']=sdk.terrain_height(arena,r['x'],r['y'])
            m['state']=states[team]
            m['ability_seconds']=None if m['ability_enter'] is None or m['ability_exit'] is None else m['ability_exit']-m['ability_enter']
    return dict(model=model,seed=seed,backend=backend,control_hz=control_hz,horizon_seconds=horizon,simulation_seconds=frame['time'],
                ability=ability,goals=goals,metrics=metrics,wall_seconds=time.perf_counter()-started,trace=str(output))


def compare(results):
    by_key=defaultdict(dict);pairs=[]
    for row in results:by_key[(row['model'],row['seed'])][row['backend']]=row
    for (model,seed),backends in by_key.items():
        if set(backends)!={'rapier','warp_kinematic'}:continue
        for team in ('red','blue'):
            ref=backends['rapier']['metrics'][team];fast=backends['warp_kinematic']['metrics'][team]
            errors={}
            for key in ('reached_at','settled_at','ability_seconds'):
                a,b=ref[key],fast[key]
                errors[key]=None if a is None or b is None or a<=0 else abs(b-a)/a
            pairs.append(dict(model=model,seed=seed,team=team,reference_success=ref['success'],simplified_success=fast['success'],relative_time_errors=errors,
                              final_origin_offset_difference_m=abs((fast['final']['z']-fast['final']['public_ground_z'])-(ref['final']['z']-ref['final']['public_ground_z'])),
                              pass_gate=ref['success'] and fast['success'] and all(e is not None and e<=.2+1e-12 for e in errors.values())))
    return pairs


def ability_summary(comparisons):
    groups=defaultdict(list)
    for pair in comparisons:groups[pair['model']].append(pair)
    summary={}
    for model,pairs in groups.items():
        reference=sum(p['reference_success'] for p in pairs)/len(pairs)
        simplified=sum(p['simplified_success'] for p in pairs)/len(pairs)
        successful=[p for p in pairs if p['reference_success'] and p['simplified_success']]
        times_match=bool(successful) and all(p['pass_gate'] for p in successful)
        summary[str(model)]=dict(paired_attempts=len(pairs),minimum_attempts=20,reference_success_rate=reference,
            simplified_success_rate=simplified,successful_pairs_meet_time_gate=times_match,
            ability_gate_pass=len(pairs)>=20 and reference>=.95 and simplified>=.95 and times_match)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',type=int,nargs='+',default=[2731,2732,2773,2714,303])
    parser.add_argument('--seeds',type=int,nargs='+',default=list(range(10)))
    parser.add_argument('--control-hz',type=int,choices=[2,5,10],default=2)
    parser.add_argument('--horizon',type=float,default=40.)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not math.isfinite(args.horizon) or not 0<args.horizon<=420:parser.error('horizon must be in (0, 420]')
    if any(str(m) not in sdk.DESIGNS or sdk.DESIGNS[str(m)].get('role') not in ('heavy','infantry','sentry') for m in args.models):
        parser.error('models must be existing ground heavy, infantry or sentry designs')
    args.output.mkdir(parents=True,exist_ok=False)
    sources={b:provenance(load_config(dict(sim_backend=b,navigation_precompute=False,rl_seconds=1/args.control_hz,learning_control_hz=args.control_hz))) for b in ('rapier','warp_kinematic')}
    write_json(args.output/'experiment.json',dict(kind='ability-calibration',ppo_included=False,control_hz=args.control_hz,sources=sources,
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Paired public-SDK navigation; seed and nearby goal variation; not trained-policy win-rate transfer'))
    results=[]
    for model in args.models:
        for seed in args.seeds:
            for backend in ('rapier','warp_kinematic'):
                result=trial(model,seed,backend,args.control_hz,args.horizon,args.output/f'{model}-{seed}-{backend}.jsonl',args.device)
                results.append(result)
                comparisons=compare(results)
                report=dict(completed=len(results),total=len(args.models)*len(args.seeds)*2,results=results,comparisons=comparisons,
                            ability_summary=ability_summary(comparisons),ppo_included=False)
                write_json(args.output/'progress.json',report)
                print(json.dumps(result,ensure_ascii=False),flush=True)
    write_json(args.output/'summary.json',report)


if __name__=='__main__':main()
