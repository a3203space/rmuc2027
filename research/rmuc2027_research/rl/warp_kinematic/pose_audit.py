"""Paired stationary ride-height measurements; not an RL training run."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from rmuc2027_core.engine import Match,DT,binary_path
from rmuc2027_core import sdk
from ..session import load_config,provenance
from .arena import ArenaMotion
from .calibration import write_json


def trial(model,backend,seconds,output,device):
    design=sdk.DESIGNS[str(model)];slot=1 if design['role']=='heavy' else 7 if design['role']=='sentry' else 3
    lineups={t:[dict(slot=s,model=model if s==slot else m) for s,m in
        [(1,2711),(3,2731),(4,2732),(6,2761),(7,2771)]] for t in ('red','blue')}
    legged=design.get('chassis_type')=='wheel_legged'
    phases=[('minimum',design['leg_min_m']),('nominal',None),('maximum',design['leg_max_m']),('nominal-return',None)] if legged else [('nominal',None)]
    results=[]
    adapter=ArenaMotion(device) if backend=='warp_kinematic' else None
    with Match(seed=42,lineups=lineups,physics_adapter=adapter,battle_settings={'drones_enabled':False,'timed_assembly':True}) as match,output.open('w') as stream:
        for label,target in phases:
            commands={t:{str(slot):dict(move=[0.,0.],brake=True,**({'leg_height':target} if target is not None else {}))} for t in lineups}
            measurements={t:[] for t in lineups}
            for tick in range(round(seconds/DT)):
                frame=match.step(commands)
                for robot in frame['robots']:
                    if robot['slot']!=slot:continue
                    q=robot['quaternion'];ground=sdk.terrain_height(frame['arena'],robot['x'],robot['y'])
                    row=dict(time=frame['time'],team=robot['team'],phase=label,x=robot['x'],y=robot['y'],
                        origin_z=robot['z'],public_ground_z=ground,offset_m=robot['z']-ground,
                        upright=robot['upright'],grounded=robot['grounded'],
                        muzzle_pivot_z=robot['z']+(1.-2.*(q[0]*q[0]+q[1]*q[1]))*robot['muzzle_height_m'],
                        legs=robot.get('legs'),battery_remaining_wh=robot['battery_remaining_wh'])
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
                    if tick>=round((seconds-.5)/DT):measurements[robot['team']].append(row)
            for team,rows in measurements.items():
                values=[r['offset_m'] for r in rows]
                results.append(dict(phase=label,target_leg_height_m=target,team=team,
                    median_origin_offset_m=statistics.median(values),origin_offset_range_m=max(values)-min(values),
                    all_grounded=all(r['grounded'] for r in rows),minimum_upright=min(r['upright'] for r in rows)))
    return dict(model=model,backend=backend,results=results,trace=str(output),trace_sha256=hashlib.sha256(output.read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',type=int,nargs='+',default=[2731,2714,2732,2773,303])
    parser.add_argument('--seconds',type=float,default=3.)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not math.isfinite(args.seconds) or not 1.<=args.seconds<=10.:parser.error('seconds must be in [1, 10]')
    if any(str(m) not in sdk.DESIGNS or sdk.DESIGNS[str(m)]['role'] not in ('heavy','infantry','sentry') for m in args.models):parser.error('ground models required')
    args.output.mkdir(parents=True,exist_ok=False)
    experiment=dict(kind='stationary-loaded-pose-audit',ppo_included=False,
        engine_binary_sha256=hashlib.sha256(binary_path().read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sources={b:provenance(load_config(dict(sim_backend=b))) for b in ('rapier','warp_kinematic')},
        scope='Constant held controls, referee poses for diagnostics only; no actor features or training samples.')
    write_json(args.output/'experiment.json',experiment)
    results=[];comparisons=[]
    for model in args.models:
        paired={}
        for backend in ('rapier','warp_kinematic'):
            row=trial(model,backend,args.seconds,args.output/f'{model}-{backend}.jsonl',args.device)
            results.append(row);paired[backend]=row
        native={(r['phase'],r['team']):r for r in paired['rapier']['results']}
        for r in paired['warp_kinematic']['results']:
            ref=native[(r['phase'],r['team'])]
            difference=abs(r['median_origin_offset_m']-ref['median_origin_offset_m'])
            comparisons.append(dict(model=model,phase=r['phase'],team=r['team'],native=ref,simplified=r,
                absolute_offset_difference_m=difference,
                pass_gate=difference<=.02 and ref['all_grounded'] and r['all_grounded'] and min(ref['minimum_upright'],r['minimum_upright'])>=.95))
        write_json(args.output/'progress.json',dict(results=results,comparisons=comparisons))
    write_json(args.output/'summary.json',dict(results=results,comparisons=comparisons,
        all_pass=all(r['pass_gate'] for r in comparisons),ppo_included=False,
        limitation='Stationary loaded height only. No validation of suspension bounce, moving terrain contact, airborne leg geometry or transfer win rate.'))


if __name__=='__main__':main()
