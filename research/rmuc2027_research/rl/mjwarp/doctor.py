"""Bounded CUDA admission checks. Does not modify drivers or Codex permissions."""
import argparse
import datetime
import glob
import json
from pathlib import Path
import subprocess
import sys
import time


def probe(stage):
    if stage == 'torch':
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(f'CUDA unavailable: torch={torch.__version__}, runtime={torch.version.cuda}')
        value = torch.ones((256, 256), device='cuda')
        result = float((value @ value)[0, 0].item())
        if result != 256:
            raise RuntimeError(f'Unexpected CUDA calculation: {result}')
        return {'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(0), 'calculation': result}
    import numpy as np
    import warp as wp
    import mujoco_warp as mjw
    from .runtime import BatchPhysics
    xml = '<mujoco><worldbody><geom type="plane" size="10 10 .1"/><body pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>'
    with BatchPhysics(xml, 2, 'cuda:0') as physics:
        # Compare native device steps against graph replay from identical state.
        for _ in range(20):
            with wp.ScopedDevice(physics.device):
                mjw.step(physics.model, physics.data)
        wp.synchronize_device(physics.device)
        direct = physics.state()
        physics.reset([0, 1])
        for _ in range(20):
            physics.step(np.zeros((2, 0)))
        graph = physics.state()
        for key in direct:
            np.testing.assert_allclose(direct[key], graph[key], rtol=1e-5, atol=1e-6)
            if not np.isfinite(graph[key]).all():
                raise RuntimeError(f'Nonfinite {key}')
        if not (graph['qpos'][:, 2] < 1).all():
            raise RuntimeError('Gravity did not advance both worlds')
        return {**physics.metadata(), 'graph_matches_direct': True}


def run_check(stage, timeout):
    started = time.monotonic()
    command = [sys.executable, '-m', 'rmuc2027_research.rl.mjwarp.doctor', '--probe', stage]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        output = result.stdout
        payload = next((json.loads(line[len('DOCTOR_RESULT '):]) for line in output.splitlines()
                        if line.startswith('DOCTOR_RESULT ')), None)
        return {'stage': stage, 'passed': result.returncode == 0 and payload is not None,
                'exit_code': result.returncode, 'seconds': time.monotonic()-started,
                'result': payload, 'stdout': output[-12000:], 'stderr': result.stderr[-12000:]}
    except subprocess.TimeoutExpired as exc:
        return {'stage': stage, 'passed': False, 'seconds': time.monotonic()-started,
                'error': f'timeout after {timeout} seconds',
                'stdout': (exc.stdout or b'').decode(errors='replace')[-12000:],
                'stderr': (exc.stderr or b'').decode(errors='replace')[-12000:]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--probe', choices=('torch', 'mjwarp'), help=argparse.SUPPRESS)
    parser.add_argument('--timeout', type=float, default=300, help='Maximum seconds for each CUDA probe (includes initial compilation)')
    args = parser.parse_args()
    if args.probe:
        print('DOCTOR_RESULT '+json.dumps(probe(args.probe)), flush=True)
        return
    if not 0 < args.timeout <= 1800:
        parser.error('timeout must be in (0, 1800]')
    report = {'kind': 'gpu-admission', 'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'python': sys.executable, 'device_nodes': sorted(glob.glob('/dev/nvidia*')),
              'checks': [], 'passed': False}
    try:
        driver = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                                capture_output=True, text=True, timeout=10)
        report['driver'] = {'exit_code': driver.returncode, 'stdout': driver.stdout, 'stderr': driver.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        report['driver'] = {'error': str(exc)}
    for stage in ('torch', 'mjwarp'):
        print(f'GPU admission: {stage}', flush=True)
        check = run_check(stage, args.timeout)
        report['checks'].append(check)
        if not check['passed']:
            report['blocked_stage'] = stage
            break
    report['passed'] = len(report['checks']) == 2 and all(c['passed'] for c in report['checks'])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report['passed'] else 2)


if __name__ == '__main__':
    main()
