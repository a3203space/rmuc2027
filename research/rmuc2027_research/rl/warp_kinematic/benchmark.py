"""Development full-policy benchmark; does not train or claim transfer parity."""
import argparse
from collections import Counter
import json
import hashlib
import cProfile
from pathlib import Path
import time

from rmuc2027_core.engine import Match
from ..session import load_config, resource
from ...strategies.battle import Battle
from .arena import ArenaMotion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/rl/sentry-mjwarp-hour.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--duration', type=float, default=420.)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', action='store_true', help='Record Python main-thread timing; affects throughput')
    args = parser.parse_args()
    if not 0 < args.duration <= 420:
        parser.error('duration must be in (0, 420]')
    config = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(Path(__file__).parent.glob('*.py'))}
    (args.output/'sources.json').write_text(json.dumps(sources, indent=2))
    settings = {key: config[key] for key in ('drones_enabled', 'timed_assembly')}
    # Compilation and initial CUDA context cost are reported separately.
    started = time.perf_counter()
    with Match(physics_adapter=ArenaMotion(args.device), battle_settings=settings) as warmup:
        for _ in range(20):
            warmup.step({})
    warmup_seconds = time.perf_counter()-started
    print(json.dumps(dict(phase='warmup-complete',wall_seconds=warmup_seconds)),flush=True)
    started = time.perf_counter()
    with Battle(resource(config['friendly']), resource(config['opponent']), seed=args.seed,
                duration=args.duration, map_name=config['map'], sandbox='trusted-local',
                timeout=config['policy_timeout_s'], record_replay=False,
                physics_backend='warp_kinematic', sim_device=args.device,
                fixed_policy_hz=config['fixed_policy_hz'], learning_control_hz=2,
                navigation_mode=config['navigation_mode'], side_lineups=config.get('lineups'),
                battle_settings=settings) as battle:
        setup_seconds = time.perf_counter()-started
        print(json.dumps(dict(phase='policies-ready',wall_seconds=setup_seconds)),flush=True)
        counts = Counter()
        profiler = cProfile.Profile() if args.profile else None
        if profiler is not None:
            profiler.enable()
        started = time.perf_counter()
        with (args.output/'trajectory.jsonl').open('w') as trace:
            while battle.match.status != 'finished':
                frame = battle.step()
                counts.update(e.get('type', 'unknown') for e in frame['events'])
                if battle.match.tick % 10 == 0:
                    robots = [{key: r.get(key) for key in ('id','x','y','z','vx','vy','hp','alive',
                              'grounded','transform_progress','ammo','launcher','battery_remaining_wh')}
                              for r in frame['robots']]
                    trace.write(json.dumps(dict(time=frame['time'],robots=robots), allow_nan=False)+'\n')
                if battle.match.tick % 600 == 0:
                    print(json.dumps(dict(sim_seconds=frame['time'], wall_seconds=time.perf_counter()-started)), flush=True)
        elapsed = time.perf_counter()-started
        if profiler is not None:
            profiler.disable()
            profiler.dump_stats(str(args.output/'profile.pstats'))
        report = dict(kind='fixed-policy-benchmark', backend='warp_kinematic', ppo_included=False,
                      profiled=args.profile,
                      warmup_seconds=warmup_seconds, setup_seconds=setup_seconds,
                      wall_seconds=elapsed, sim_seconds=battle.match.time,
                      realtime_factor=battle.match.time/elapsed, events=dict(counts),
                      source_sha256=sources, metadata=battle.metadata, result=battle.result(), diagnostics=battle.diagnostics)
    (args.output/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({key: report[key] for key in ('kind','ppo_included','wall_seconds','sim_seconds','realtime_factor','result')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
