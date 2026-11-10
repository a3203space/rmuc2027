"""Flag coordinate anomalies in benchmark traces; standing still alone is valid."""
import argparse
import json
import math
from pathlib import Path


def audit(rows):
    previous = {}
    active = {}
    findings = []
    for frame in rows:
        now = frame['time']
        for robot in frame['robots']:
            identity = robot['id']
            old = previous.get(identity)
            values = [robot.get(k) for k in ('x','y','z','vx','vy')]
            if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in values):
                findings.append(dict(kind='nonfinite-motion',id=identity,time=now))
                continue
            bad = False
            if old is not None and robot.get('alive') and old[1].get('alive'):
                dt = now-old[0]
                distance = math.hypot(robot['x']-old[1]['x'],robot['y']-old[1]['y'])
                speed = math.hypot(robot['vx'],robot['vy'])
                bad = 0 < dt <= 1 and distance < .001 and speed > .2
            if bad:
                active.setdefault(identity, old[0])
            elif identity in active:
                since = active.pop(identity)
                if now-since >= 2:
                    findings.append(dict(kind='stationary-with-velocity',id=identity,start=since,end=now))
            previous[identity] = (now,robot)
    if previous:
        end=max(value[0] for value in previous.values())
        for identity,since in active.items():
            if end-since >= 2:
                findings.append(dict(kind='stationary-with-velocity',id=identity,start=since,end=end))
    return dict(samples=len(rows),findings=findings,
                limitation='Detects motion telemetry contradictions; does not prove navigation, visibility or combat correctness')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory',type=Path)
    args=parser.parse_args()
    rows=[json.loads(line) for line in args.trajectory.read_text().splitlines()]
    print(json.dumps(audit(rows),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
