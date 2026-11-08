"""Read-only ray/analytic/CUDA height comparison at explicit map coordinates.

The static ray returns the highest fixed collider, including any roof/fixture.
Default central-platform samples avoid those overhead objects.
"""
import argparse
import hashlib
import json
from pathlib import Path
from rmuc2027_core.engine import Match,binary_path
from rmuc2027_core import sdk
from .terrain import Terrain


DEFAULT_POINTS=[(2.5,3.5),(3.2,6.15),(5.7,7.5),(12.,3.),(12.,6.5),(12.,7.4),(12.,7.5),(12.,8.5),(12.,12.)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--point',type=float,nargs=2,action='append')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();points=args.point or DEFAULT_POINTS
    if args.output.exists():parser.error('output must not already exist')
    with Match(battle_settings={'drones_enabled':False,'timed_assembly':True}) as match:
        arena=match.snapshot()['arena']
        report=match._request(dict(op='terrain_probe',points=points))
        cuda=Terrain(arena,args.device).heights(points)
    for row,gpu in zip(report['samples'],cuda):
        analytic=sdk.terrain_height(arena,row['x'],row['y'])
        row.update(sdk_ground_z=analytic,cuda_ground_z=float(gpu),cuda_minus_analytic_m=float(gpu)-row['analytic_ground_z'])
    report.update(kind='terrain-height-audit',device=args.device,ppo_included=False,
        engine_binary_sha256=hashlib.sha256(binary_path().read_bytes()).hexdigest(),
        sdk_sha256=hashlib.sha256(Path(sdk.__file__).read_bytes()).hexdigest(),
        cuda_terrain_sha256=hashlib.sha256(Path(__file__).with_name('terrain.py').read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        surface_definition='Uppermost fixed collider; selected points must be checked for roofs and fixtures before interpreting difference as ground error.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
