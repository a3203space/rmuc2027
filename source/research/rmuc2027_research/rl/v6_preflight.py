"""Deadline-bounded, frozen-policy load measurements inside the research budget.

Calling preflight performs real physics. Importing this module does not. Each
backend/count runs in an owned subprocess so a hung native/GPU operation cannot
consume the rest of the campaign indefinitely. No optimizer is ever stepped.
"""
import argparse
import copy
import hashlib
from importlib.metadata import distributions
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from .campaign import write_json


WINDOW_STEPS = 30
WINDOW_COUNT = 2
WARMUP_STEPS = 1


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_environment(root):
    installed, editable = [], []
    for distribution in distributions():
        name = distribution.metadata.get('Name')
        if not name:
            continue
        try:
            direct = json.loads(distribution.read_text('direct_url.json') or '{}')
        except (ValueError, TypeError):
            direct = {}
        if direct.get('dir_info', {}).get('editable'):
            editable.append(name)
        else:
            installed.append(f'{name}=={distribution.version}')
    (root / 'requirements-runtime.txt').write_text('\n'.join(sorted(set(installed), key=str.lower)) + '\n')
    write_json(root / 'runtime-environment.json', dict(python=platform.python_version(),
        platform=platform.platform(), editable_source_packages=sorted(editable), policy_device='cpu',
        physics_device='Warp CUDA required for warp_kinematic; native CPU for Rapier',
        freeze_file='requirements-runtime.txt', wheelhouse_embedded=False,
        benchmark_parallelism=[1, 2], target_5600x_5070ti_validation_claimed=False))


def benchmark_configurations(root, backend, count, deadline):
    """Create explicit workload configs, without leasing training scenarios."""
    from .scenarios import ScenarioSpec
    root = Path(root)
    base = json.loads((root / 'main/experiment.json').read_text())['config']
    if base.get('policy_version') != 6 or set(base.get('auxiliary_pool', {})) != {'infantry', 'heavy'}:
        raise ValueError('Preflight needs v6 main and both frozen auxiliary actors')
    if backend not in ('warp_kinematic', 'rapier') or count not in (1, 2) or (backend == 'rapier' and count != 1):
        raise ValueError('This campaign only measures Warp1/2 and Rapier1')
    result = []
    for index in range(count):
        cfg = copy.deepcopy(base)
        side = 'red' if index % 2 == 0 else 'blue'
        for key in ('_v6_sampler_path', '_v6_replay_dir', '_episode_log_dir', 'recovery_kind', 'curriculum_seconds'):
            cfg.pop(key, None)
        scenario = ScenarioSpec(learner_side=side)
        profile = root / f'capability-profiles/evaluation-{side}.json'
        if not profile.is_file():
            raise ValueError('Missing frozen capability profile: ' + str(profile))
        cfg.update(sim_backend=backend, sim_device='cuda:0' if backend == 'warp_kinematic' else 'cpu',
            seed=81001 + index, team=side, scenario=scenario.to_dict(), lineups=scenario.lineups(),
            scenario_sampling=False, capability_sampling=False, alternate_sides=False,
            training_capabilities=json.loads(profile.read_text()), evaluation=True,
            episode_seconds=420., decision_window_seconds=420., training_summary=False,
            action_trace=False, _training_deadline=deadline, _worker_id=index,
            controllers={f'{side}:7': {'kind': 'learning'}})
        result.append(cfg)
    return result


def measure_windows(vec, model, count, deadline, *, publish=lambda row: None):
    """Shared measurement loop, separately testable with tiny fake vector envs."""
    import numpy as np
    import torch
    row = dict(num_envs=count, warmup_steps=0, windows=[], completed=False,
               gradient_updates=0, full_frame_recording=True, measured_rl_steps=0)
    started = time.monotonic()
    if started >= deadline:
        row.update(censored=True, reason='deadline_before_reset')
        return row
    obs = vec.reset()
    row['initialization_seconds'] = time.monotonic() - started

    def step():
        nonlocal obs
        if time.monotonic() >= deadline:
            return None
        with torch.no_grad():
            actions, _ = model.predict(obs, deterministic=False)
        if not np.isfinite(actions).all():
            raise ValueError('Nonfinite main actor action')
        obs, rewards, dones, infos = vec.step(actions)
        if not all(np.isfinite(value).all() for value in obs.values()) or not np.isfinite(rewards).all():
            raise ValueError('Nonfinite preflight observations/rewards')
        if np.asarray(dones).any():
            raise ValueError('Unexpected termination/reset during the short opening benchmark')
        seconds = [float(info['sampled_simulation_seconds']) for info in infos]
        if len(seconds) != count or not all(math.isfinite(value) and value > 0 for value in seconds):
            raise ValueError('Actual per-environment sampled simulation seconds are required')
        if any(info.get('metrics', {}).get('policy_errors', 0) for info in infos):
            raise ValueError('Frozen policy execution failed during preflight')
        return sum(seconds)

    for _ in range(WARMUP_STEPS):
        if step() is None:
            row.update(censored=True, reason='deadline_during_warmup')
            publish(row)
            return row
        row['warmup_steps'] += 1
    publish(row)
    for index in range(WINDOW_COUNT):
        began = time.monotonic()
        steps, seconds = 0, 0.
        for _ in range(WINDOW_STEPS):
            sampled = step()
            if sampled is None:
                break
            steps += 1
            seconds += sampled
        wall = max(1e-9, time.monotonic() - began)
        completed = steps == WINDOW_STEPS and time.monotonic() <= deadline
        window = dict(index=index, rl_steps=steps, decisions=steps * count,
            sum_simulation_seconds=seconds, wall_seconds=wall, completed=completed,
            r=seconds / wall, single_world_rate=seconds / (count * wall))
        row['windows'].append(window)
        row['measured_rl_steps'] += steps
        if not completed:
            row.update(censored=True, reason='deadline_during_window')
            publish(row)
            return row
        publish(row)
    row.update(completed=True, censored=False,
        conservative_r=min(window['r'] for window in row['windows']),
        single_world_rate=min(window['single_world_rate'] for window in row['windows']))
    publish(row)
    return row


def benchmark_worker(root, backend, count, deadline_unix, output):
    """Actual physics worker. Used only when the campaign explicitly calls it."""
    import numpy as np
    import torch
    from functools import partial
    from stable_baselines3.common.vec_env import DummyVecEnv
    from .env import SentryEnv
    from .session import resource
    from .v6_ppo import HybridPPO
    root, output = Path(root), Path(output)
    deadline = time.monotonic() + max(0., deadline_unix - time.time())
    torch.set_num_threads(1)
    devices = dict(policy_device='cpu', torch_cuda_available=torch.cuda.is_available(),
                   warp_cuda_available=False, warp_cuda_devices=[], physics_device=None)
    row = dict(backend=backend, num_envs=count, windows=[], completed=False, gradient_updates=0)
    vec = None
    try:
        configs = benchmark_configurations(root, backend, count, deadline)
        checkpoint = root / 'main/initialized.zip'
        model = HybridPPO.load(checkpoint, device='cpu')
        torch.manual_seed(81001)
        np.random.seed(81001)
        before = (model.num_timesteps, model._n_updates)
        row['sources'] = dict(main=dict(path=str(checkpoint), sha256=_hash(checkpoint)),
            helpers={role: dict(path=str(resource(path)), sha256=_hash(resource(path)))
                     for role, path in configs[0]['auxiliary_pool'].items()},
            capability_profiles=[dict(path=str(root / f'capability-profiles/evaluation-{cfg["team"]}.json'),
                sha256=_hash(root / f'capability-profiles/evaluation-{cfg["team"]}.json')) for cfg in configs])
        if backend == 'warp_kinematic':
            import warp as wp
            wp.init()
            selected = wp.get_device('cuda:0')
            if not selected.is_cuda:
                raise RuntimeError('Warp physics requires real CUDA; no CPU fallback')
            devices.update(warp_cuda_available=True, physics_device=str(selected),
                warp_cuda_devices=[dict(device=str(selected), name=selected.name,
                    architecture=selected.arch, memory_bytes=int(selected.total_memory),
                    free_bytes=int(selected.free_memory))])
        else:
            devices['physics_device'] = 'native_rapier_cpu'
        row['devices'] = devices
        factories = [partial(SentryEnv, cfg, record_replay=True) for cfg in configs]
        if backend == 'warp_kinematic' and count > 1:
            # The exact same batched worker/adapter selected by learning.vector_env;
            # explicit factories are necessary to force full replay overhead.
            from .warp_kinematic.vector import WarpVecEnv
            vec = WarpVecEnv(factories)
        else:
            vec = DummyVecEnv(factories)
        vec.seed(81001)
        def publish(measured):
            write_json(output, {**row, **measured})
        row.update(measure_windows(vec, model, count, deadline, publish=publish))
        if (model.num_timesteps, model._n_updates) != before:
            raise RuntimeError('Frozen preflight unexpectedly changed training counters')
        if backend == 'warp_kinematic':
            row['gpu_free_bytes_after'] = int(selected.free_memory)
        row['training_counters_unchanged'] = True
    except Exception as error:
        if output.is_file():
            previous = json.loads(output.read_text())
            row = {**previous, **row, 'windows': previous.get('windows', row.get('windows', []))}
        row.update(completed=False, error=f'{type(error).__name__}: {error}', devices=devices)
    finally:
        if vec is not None:
            vec.close()
    write_json(output, row)
    return row


def summarize(rows):
    def complete(backend, count):
        return next((row for row in rows if row.get('backend') == backend and row.get('num_envs') == count
            and row.get('completed') and len(row.get('windows', [])) == WINDOW_COUNT
            and all(window.get('completed') and window.get('rl_steps') == WINDOW_STEPS
                    and math.isfinite(window.get('r', 0)) and window.get('r', 0) > 0
                    for window in row['windows'])), None)
    warp, rapier, two = complete('warp_kinematic', 1), complete('rapier', 1), complete('warp_kinematic', 2)
    ready = bool(warp and rapier and warp.get('devices', {}).get('warp_cuda_available'))
    selected = 2 if ready and two and two['conservative_r'] > warp['conservative_r'] * 1.05 else 1
    return dict(training_ready=ready, selected_num_envs=selected if ready else None,
        recommended_num_envs=selected if ready else None,
        evaluation_rates={name: row['single_world_rate'] for name, row in (('warp_kinematic', warp), ('rapier', rapier)) if row},
        parallelism_rule='prefer 1 unless both complete 2-env windows have >5% conservative aggregate gain',
        required_completed_windows_per_single_env_backend=WINDOW_COUNT,
        benchmark_includes_ppo_updates=False, gradient_updates=0)


def preflight(root, deadline_unix):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    freeze_environment(root)
    work = root / 'preflight-workers'
    work.mkdir(exist_ok=True)
    rows = []
    report = dict(kind='v6-frozen-policy-load', policy_device='cpu', full_frame_recording=True,
        warmup_rl_steps=WARMUP_STEPS, measured_windows=WINDOW_COUNT, rl_steps_per_window=WINDOW_STEPS,
        protocol_actor_hz=10, executor_hz=20, benchmark_includes_initialization=False,
        benchmark_scope='two short opening windows, not full-match/optimizer throughput',
        deadline_unix=deadline_unix, target_hardware_claimed=False)
    # Complete both required single-world backends before optional vectorization.
    jobs = [('warp_kinematic', 1), ('rapier', 1), ('warp_kinematic', 2)]
    for index, (backend, count) in enumerate(jobs):
        remaining = deadline_unix - time.time()
        if remaining <= 2:
            break
        # The first mandatory measurement may not consume the entire budget
        # before Rapier gets any opportunity. Every child also has a hard kill.
        allowance = (remaining - 1) / 2 if index == 0 else remaining - 1
        child_deadline = min(deadline_unix - 1, time.time() + allowance)
        output = work / f'{backend}-{count}.json'
        log = work / f'{backend}-{count}.log'
        command = [sys.executable, '-m', 'rmuc2027_research.rl.v6_preflight', 'worker',
            '--root', str(root), '--backend', backend, '--num-envs', str(count),
            '--deadline-unix', str(child_deadline), '--output', str(output)]
        with log.open('w') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            expired = False
            def stop_worker():
                from rmuc2027_core.platform_support import kill_process_tree
                try:
                    kill_process_tree(process)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=max(.01, min(.5, deadline_unix - time.time())))
                except subprocess.TimeoutExpired:
                    pass
            try:
                try:
                    process.wait(timeout=max(.01, child_deadline - time.time()))
                except subprocess.TimeoutExpired:
                    expired = True
                    stop_worker()
            finally:
                # Outer SIGINT/deadline/errors must never orphan the owned
                # inner worker's separate process group. No unrelated process
                # or competing workload is discovered or signaled here.
                if process.poll() is None:
                    stop_worker()
        row = json.loads(output.read_text()) if output.is_file() else dict(backend=backend, num_envs=count, windows=[])
        row.update(worker_exit_code=process.returncode, watchdog_timeout=expired)
        if expired or process.returncode != 0:
            row.update(completed=False, censored=True, error='worker_deadline' if expired else 'worker_failed',
                       interrupted_window_partial_counts_unknown=True)
        rows.append(row)
        devices = next((item['devices'] for item in rows if item.get('devices', {}).get('warp_cuda_available')), {})
        write_json(root / 'preflight.json', {**report, **devices, **summarize(rows), 'results': rows})
    devices = next((item['devices'] for item in rows if item.get('devices', {}).get('warp_cuda_available')), {})
    result = {**report, **devices, **summarize(rows), 'results': rows}
    if not result['training_ready']:
        result['error'] = 'Both complete single-env Warp CUDA and Rapier windows are required before training'
    write_json(root / 'preflight.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run')
    worker = commands.add_parser('worker')
    for child in (run, worker):
        child.add_argument('--root', type=Path, required=True)
        child.add_argument('--deadline-unix', type=float, required=True)
    worker.add_argument('--backend', choices=('warp_kinematic', 'rapier'), required=True)
    worker.add_argument('--num-envs', type=int, choices=(1, 2), required=True)
    worker.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'worker':
        result = benchmark_worker(args.root, args.backend, args.num_envs, args.deadline_unix, args.output)
    else:
        result = preflight(args.root, args.deadline_unix)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
