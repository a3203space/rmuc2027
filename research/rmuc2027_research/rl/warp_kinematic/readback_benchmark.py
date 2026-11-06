"""Paired motion-only readback comparison; excludes AI, combat and PPO."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics
import time
import numpy as np
from rmuc2027_core.engine import Match
from .arena import ArenaMotion
from .protocol import MotionFrame


class Capture:
    protocol='kinematic_frame'

    def step(self,request):
        self.request=copy.deepcopy(request)
        robots=request['robots'];n=len(robots)
        return MotionFrame(tuple(r['id'] for r in robots),np.tile([r['pos'] for r in robots],(6,1,1)),
                           np.zeros((6,n,3)),np.tile([r['heading'] for r in robots],(6,1)),
                           np.tile([r['folded'] for r in robots],(6,1)),np.ones((6,n),dtype=bool)).response(0)

    def close(self):pass


def run(source,packed,frames):
    command=copy.deepcopy(source);adapter=ArenaMotion(packed_readback=packed)
    digest=hashlib.sha256();elapsed=0.
    try:
        for tick in range(frames+20):
            command['sequence']=tick
            if tick==20:started=time.perf_counter()
            frame=adapter.step(command)
            # Equal evolving movement workload, independent of match outcomes.
            for index,robot in enumerate(command['robots']):
                robot.update(pos=frame['positions'][-1][index],velocity=frame['velocities'][-1][index],
                             heading=frame['headings'][-1][index],folded=frame['folded'][-1][index])
            if tick>=20:digest.update(json.dumps(frame,separators=(',',':')).encode())
        elapsed=time.perf_counter()-started
    finally:adapter.close()
    return dict(packed=packed,wall_seconds=elapsed,digest=digest.hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames',type=int,default=200)
    args=parser.parse_args()
    if args.frames<1:parser.error('frames must be positive')
    capture=Capture()
    with Match(physics_adapter=capture,battle_settings={'drones_enabled':False}) as match:
        match.step({})
    source=capture.request
    for robot in source['robots']:
        if robot['role']!='aerial':robot['move']=[.2,.1]
    results=[]
    for order in ((False,True),(True,False),(False,True)):
        results.extend(run(source,packed,args.frames) for packed in order)
    if len({r['digest'] for r in results})!=1:
        raise RuntimeError('Readback variants produced different trajectories')
    medians={str(p):statistics.median(r['wall_seconds'] for r in results if r['packed']==p) for p in (False,True)}
    report=dict(kind='motion-only-readback',sim_seconds=args.frames*.05,results=results,median_seconds=medians,
                limitation='Excludes AI, combat and PPO; not a training throughput result')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':main()
