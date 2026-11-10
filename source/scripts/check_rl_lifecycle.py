"""Real process tests: spawned samplers, SIGINT, checkpoint and child cleanup."""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def gone(pids):
    return [pid for pid in pids if Path(f'/proc/{pid}').exists()]


def descendants(pid):
    found = [pid]
    try:
        children = Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
    except FileNotFoundError:
        return found
    for child in children:
        found.extend(descendants(int(child)))
    return found


def check_vector():
    from rmuc2027_research.rl.learning import vector_env
    from rmuc2027_research.rl.session import load_config
    import numpy as np
    vec = vector_env(load_config(), 2)
    try:
        vec.reset()
        owned = [p.pid for p in vec.processes]
        owned += [pid for group in vec.env_method('process_ids') for pid in group]
        vec.step(np.zeros((2, 2), dtype=np.float32))
    finally:
        vec.close()
    time.sleep(.2)
    survivors = gone(owned)
    assert not survivors, survivors
    return {'num_envs': 2, 'owned_pids': owned, 'survivors': survivors}


def check_interrupt(output):
    training = output/'interrupted-training'
    with (output/'interruption.log').open('w') as log:
        proc = subprocess.Popen([sys.executable, '-m', 'rmuc2027_research.rl', 'train', '--total-steps', '256',
            '--run-dir', str(training)], cwd=ROOT, stdout=log, stderr=log)
        try:
            limit = time.monotonic()+60
            while time.monotonic() < limit:
                if proc.poll() is not None:
                    raise RuntimeError('Training exited before interruption test; see interruption.log')
                trace = training/'actions.jsonl'
                if trace.exists() and trace.stat().st_size > 0:
                    break
                time.sleep(.2)
            else:
                raise RuntimeError('No samples before interruption deadline')
            pids = descendants(proc.pid)
            proc.send_signal(signal.SIGINT)
            status = proc.wait(timeout=30)
            time.sleep(.2)
            result = {'exit_code': status, 'checkpoint_saved': (training/'interrupted.zip').is_file(),
                'owned_pids': pids, 'survivors': gone(pids)}
            (output/'interruption.json').write_text(json.dumps(result, indent=2))
            assert status == 130 and result['checkpoint_saved'] and not result['survivors'], result
            return result
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'data/research/lifecycle-check')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    results = {'vector': check_vector(), 'interrupt': check_interrupt(args.output)}
    results['wall_seconds'] = time.perf_counter()-started
    (args.output/'lifecycle.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results))


if __name__ == '__main__':
    main()
