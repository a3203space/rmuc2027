"""Run with python -m rmuc2027_research.rl --help (training dependencies are optional)."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

from .portable import OBS_SIZE
from .session import ArenaSession, DEFAULT_CONFIG, load_config


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def owned_pids(session):
    battle = session.battle
    return [battle.match.process.pid]+[p.process.pid for _, _, p in battle.policies if hasattr(p, 'process')]


def rss_kib():
    status = Path('/proc/self/status')
    if status.exists():
        return int(next(line.split()[1] for line in status.read_text().splitlines() if line.startswith('VmRSS:')))
    return None


def random_run(config, steps, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rng = random.Random(config['seed']+901)
    started = time.perf_counter()
    with ArenaSession(config, record_replay=True) as session:
        obs, info = session.reset(config['seed'])
        initialization = time.perf_counter()-started
        with (output/'actions.jsonl').open('w') as stream:
            for index in range(steps):
                action = [rng.uniform(-1, 1), rng.uniform(-1, 1)]
                obs, reward, terminated, truncated, info = session.step(action)
                assert len(obs) == OBS_SIZE and all(math.isfinite(v) and -1 <= v <= 1 for v in obs)
                assert math.isfinite(reward)
                stream.write(json.dumps({'step': index, 'reward': reward, **info}, allow_nan=False)+'\n')
                if terminated or truncated:
                    session.save_replay(output/f'episode-{index}.json')
                    if index+1 < steps:
                        session.reset()
        if not session.done:
            session.save_replay(output/'last.json')
        result = {'steps': steps, 'initialization_seconds': initialization, 'wall_seconds': time.perf_counter()-started,
            'observation_shape': [len(obs)], 'finite': True, 'final': info}
    write_json(output/'random.json', result)
    print(json.dumps(result), flush=True)


def check(config, cycles, steps, output, engine_only=False):
    started = time.perf_counter()
    cfg = copy.deepcopy(config)
    cfg['episode_seconds'] = max(cfg['episode_seconds'], steps*cfg['rl_seconds'])
    actions = [[random.Random(700+i).uniform(-1, 1), random.Random(900+i).uniform(-1, 1)] for i in range(steps)]
    traces, pids, memory = [], [], []
    session = ArenaSession(cfg)
    try:
        for cycle in range(max(2, cycles)):
            obs, _ = session.reset(42)
            pids.extend(owned_pids(session))
            trace = [obs]
            for action in actions:
                trace.append(session.step(action))
            trace.append(session.battle.match.digest())
            traces.append(trace)
            assert trace == traces[0], 'Same seed/action trace diverged'
            session.close()
            survivors = [pid for pid in pids if Path(f'/proc/{pid}').exists()]
            assert not survivors, f'Leaked processes: {survivors}'
            memory.append({'cycle': cycle, 'rss_kib': rss_kib(), 'fds': len(list(Path('/proc/self/fd').iterdir()))})
            print(json.dumps({'reset_cycle': cycle+1, 'reproducible': True, 'survivors': survivors}), flush=True)
    finally:
        session.close()
    # Exercise error cleanup against a real engine, not only a mocked close.
    try:
        session.reset(42)
        killed = owned_pids(session)
        session.battle.match.process.kill()
        session.battle.match.process.wait()
        try:
            session.step([0, 0])
            raise AssertionError('Dead engine was silently accepted')
        except (OSError, RuntimeError):
            pass
    finally:
        session.close()
    assert not [pid for pid in killed if Path(f'/proc/{pid}').exists()]
    report = {'reproducible': True, 'reset_cycles': max(2, cycles), 'steps_per_reset': steps,
        'processes_closed': True, 'engine_failure_cleanup': True, 'memory': memory}
    if not engine_only:
        from gymnasium.utils.env_checker import check_env as gym_check
        from stable_baselines3.common.env_checker import check_env as sb3_check
        from .env import SentryEnv
        env = SentryEnv(cfg)
        try:
            gym_check(env, skip_render_check=True)
            sb3_check(env)
            report.update(gymnasium_check=True, sb3_check=True)
        finally:
            env.close()
    report['wall_seconds'] = time.perf_counter()-started
    write_json(Path(output)/'checks.json', report)
    print(json.dumps(report), flush=True)
    return report


def benchmark(config, counts, steps, output):
    from .learning import vector_env
    import numpy as np
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for count in counts:
        started = time.perf_counter()
        vec = vector_env(config, count)
        try:
            vec.reset()
            initialization = time.perf_counter()-started
            begin = time.perf_counter()
            actions = np.zeros((count, 2), dtype=np.float32)
            for _ in range(steps):
                _, rewards, _, infos = vec.step(actions)
                assert np.isfinite(rewards).all()
            elapsed = time.perf_counter()-begin
            # Sum live workers' resident sets when procfs is available.
            descendants = []
            if hasattr(vec, 'processes'):
                descendants = [p.pid for p in vec.processes]
            record = {'num_envs': count, 'steps_per_env': steps,
                'initialization_seconds': initialization, 'sampling_seconds': elapsed,
                'rl_steps_per_second': steps*count/elapsed,
                'simulation_seconds_per_second': steps*count*config['rl_seconds']/elapsed,
                'parent_rss_kib': rss_kib(), 'worker_pids': descendants}
            results.append(record)
            print(json.dumps(record), flush=True)
        finally:
            vec.close()
    report = {'results': results, 'recommended_num_envs': max(results, key=lambda r: r['rl_steps_per_second'])['num_envs']}
    write_json(output/'benchmark.json', report)


def main():
    parser = argparse.ArgumentParser(description='RMUC 2027 sentry PPO and headless environment checks')
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('check', 'random', 'train', 'benchmark'):
        p = sub.add_parser(command)
        p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
        p.add_argument('--run-dir', type=Path, default=Path('data/research')/command)
        if command == 'check':
            p.add_argument('--reset-cycles', type=int, default=10)
            p.add_argument('--steps', type=int, default=4)
            p.add_argument('--engine-only', action='store_true')
        elif command == 'random':
            p.add_argument('--steps', type=int, default=64)
        elif command == 'train':
            p.add_argument('--total-steps', type=int, default=256)
            p.add_argument('--additional-steps', type=int)
            p.add_argument('--resume', type=Path)
            p.add_argument('--num-envs', type=int, default=1)
            p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
        else:
            p.add_argument('--num-envs', type=int, nargs='+', default=[1, 2, 4])
            p.add_argument('--steps', type=int, default=128)
    p = sub.add_parser('evaluate')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[1001])
    p.add_argument('--sides', choices=('red', 'blue'), nargs='+', default=['red', 'blue'])
    p.add_argument('--duration', type=int, default=420)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p = sub.add_parser('export')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        if args.command in ('check', 'random', 'train', 'benchmark'):
            config = load_config(args.config)
        if args.command == 'check':
            if args.steps < 1 or args.reset_cycles < 1:
                raise ValueError('steps and reset-cycles must be positive')
            check(config, args.reset_cycles, args.steps, args.run_dir, args.engine_only)
        elif args.command == 'random':
            if args.steps < 1:
                raise ValueError('steps must be positive')
            random_run(config, args.steps, args.run_dir)
        elif args.command == 'train':
            from .learning import train
            if args.additional_steps is not None and not args.resume:
                raise ValueError('--additional-steps requires --resume')
            train(config, args.run_dir, args.additional_steps if args.additional_steps is not None else args.total_steps,
                args.num_envs, args.device, args.resume)
        elif args.command == 'evaluate':
            from .learning import evaluate
            evaluate(args.model, args.run_dir, args.seeds, args.sides, args.duration, args.device)
        elif args.command == 'export':
            from .exporting import export_model
            print(json.dumps(export_model(args.model, args.output)), flush=True)
        elif args.command == 'benchmark':
            if args.steps < 1:
                raise ValueError('steps must be positive')
            benchmark(config, args.num_envs, args.steps, args.run_dir)
    except ImportError as exc:
        parser.exit(2, f'RL dependency unavailable: {exc}. Install with uv pip install --python research/.venv/bin/python --torch-backend cpu -r research/requirements-rl.txt -e packages/core -e research\n')
    except KeyboardInterrupt:
        parser.exit(130, 'Interrupted; environments closed (training checkpoint saved when available).\n')
    except (ValueError, RuntimeError) as exc:
        parser.exit(2, str(exc)+'\n')


if __name__ == '__main__':
    main()
