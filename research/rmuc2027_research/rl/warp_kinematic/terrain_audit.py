"""Read-only ray/analytic/CUDA height comparison at explicit map coordinates.

The static ray returns the highest fixed collider, including any roof/fixture.
Default central-platform samples avoid those overhead objects.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from rmuc2027_core.engine import Match,binary_path
from rmuc2027_core import sdk
from .terrain import Terrain


DEFAULT_POINTS=[(2.5,3.5),(3.2,6.15),(5.7,7.5),(12.,3.),(12.,6.5),(12.,7.4),(12.,7.5),(12.,8.5),(12.,12.)]


def audit(points,device):
    with Match(battle_settings={'drones_enabled':False,'timed_assembly':True}) as match:
        arena=match.snapshot()['arena']
        report=dict(samples=[])
        for start in range(0,len(points),4096):
            report['samples'].extend(match._request(dict(op='terrain_probe',points=points[start:start+4096]))['samples'])
        cuda=Terrain(arena,device).heights(points)
    for row,gpu in zip(report['samples'],cuda):
        analytic=sdk.terrain_height(arena,row['x'],row['y'])
        row.update(sdk_ground_z=analytic,cuda_ground_z=float(gpu),cuda_minus_analytic_m=float(gpu)-row['analytic_ground_z'])
    negative=[r for r in report['samples'] if r['static_top_minus_ground'] is not None and r['static_top_minus_ground']<-.002]
    report.update(kind='terrain-height-audit',device=device,ppo_included=False,
        sample_count=len(points),negative_static_difference_gt_2mm=len(negative),
        missing_static_surfaces=sum(r['physical_static_top_z'] is None for r in report['samples']),
        max_cuda_analytic_error_m=max(abs(r['cuda_minus_analytic_m']) for r in report['samples']),
        worst_cuda_samples=sorted(report['samples'],key=lambda r:-abs(r['cuda_minus_analytic_m']))[:10],
        worst_negative_static_samples=sorted(negative,key=lambda r:r['static_top_minus_ground'])[:20],
        engine_binary_sha256=hashlib.sha256(binary_path().read_bytes()).hexdigest(),
        sdk_sha256=hashlib.sha256(Path(sdk.__file__).read_bytes()).hexdigest(),
        cuda_terrain_sha256=hashlib.sha256(Path(__file__).with_name('terrain.py').read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        surface_definition='Uppermost fixed collider; selected points must be checked for roofs and fixtures before interpreting difference as ground error.')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--point',type=float,nargs=2,action='append')
    mode.add_argument('--grid',type=float,help='Sample cell centers at this spacing in metres')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('output must not already exist')
    points=args.point or DEFAULT_POINTS
    if args.grid is not None:
        g=args.grid;w=sdk._REFERENCE_ARENA['width'];h=sdk._REFERENCE_ARENA['height']
        if not math.isfinite(g) or not 0<g<=min(w,h) or math.ceil(w/g)*math.ceil(h/g)>50000:
            parser.error('grid must be positive, finite, fit the field and produce at most 50000 points')
        points=[(g*(i+.5),g*(j+.5)) for i in range(math.ceil(w/g)) for j in range(math.ceil(h/g))
                if g*(i+.5)<w and g*(j+.5)<h]
    report=audit(points,args.device)
    report['grid_spacing_m']=args.grid
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='samples'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
