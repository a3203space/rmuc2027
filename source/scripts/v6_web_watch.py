#!/usr/bin/env python3
"""Wait for a campaign's frozen web phase; never launch an extra training job.

Optional service construction consumes the supplied run's wall budget. Only the
explicit web_acceptance marker authorizes the one short frozen match. Existing
services are observed, never stopped; all processes created here are reaped.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen


REPO = Path(__file__).resolve().parents[1]
TERMINAL = {'completed', 'completed_partial', 'failed', 'deadline', 'results_frozen', 'packaging'}


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def budget_deadline(root, maximum=None):
    """Use the actual persisted budget; an argument can only shorten it."""
    values = []
    for name in ('budget.json', 'authorization.json'):
        path = Path(root) / name
        if path.is_file():
            value = float(read_json(path)['deadline_unix'])
            if not 0 < value < float('inf'):
                raise ValueError('Invalid persisted deadline: ' + name)
            values.append(value)
    if not values:
        raise ValueError('An existing budget.json or authorization.json is required')
    if maximum is not None:
        if not 0 < maximum < float('inf'):
            raise ValueError('Invalid requested deadline')
        values.append(maximum)
    return min(values)


def acceptance_deadline(root, stage, deadline, now):
    """Return None until the matching frozen marker, never extend a deadline."""
    marker = Path(root) / 'web-acceptance-request.json'
    if stage != 'web_acceptance' or not marker.is_file():
        return None
    request = read_json(marker)
    if request['run_id'] != Path(root).name:
        raise ValueError('Web acceptance marker belongs to another run')
    requested = float(request['deadline_unix'])
    if not 0 < requested < float('inf'):
        raise ValueError('Invalid web acceptance deadline')
    return min(requested, deadline, now + 480)


def stop_owned(process, deadline):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=max(.01, min(1, deadline - time.time())))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=1)


def bounded_command(command, log, deadline, env):
    if time.time() >= deadline - 3:
        raise TimeoutError('No time remains to start acceptance subprocess')
    with Path(log).open('a') as output:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                if time.time() >= deadline - 3:
                    raise TimeoutError('Acceptance subprocess exhausted its absolute budget')
                time.sleep(min(.5, max(.01, deadline - time.time() - 3)))
            return process.returncode
        finally:
            stop_owned(process, deadline)


def watch(args):
    root = args.run.resolve()
    deadline = budget_deadline(root, args.deadline_unix)
    if time.time() >= deadline - 15:
        raise TimeoutError('Run deadline leaves no web-service budget')
    url = 'http://127.0.0.1:' + str(args.port)
    environment = os.environ.copy()
    environment['PYTHONPATH'] = os.pathsep.join(str(REPO / entry) for entry in
        ('packages/core', 'research', 'apps/server')) + os.pathsep + environment.get('PYTHONPATH', '')
    service = service_log = None
    state_path = root / 'web-stage-watcher.json'
    previous = None

    def status(state, **fields):
        nonlocal previous
        row = dict(pid=os.getpid(), utc_unix=time.time(), budget_deadline_unix=deadline,
                   status=state, **fields)
        atomic_json(state_path, row)
        identity = (state, fields.get('campaign_stage'))
        if identity != previous:
            print(json.dumps(row, ensure_ascii=False), flush=True)
            previous = identity

    try:
        if args.start_service:
            with socket.socket() as probe:
                if probe.connect_ex(('127.0.0.1', args.port)) == 0:
                    raise RuntimeError('Requested service port is occupied; no existing service will be stopped')
            service_log = (root / 'web-server.log').open('a')
            service = subprocess.Popen([sys.executable, '-u', '-m', 'rmuc2027_research', 'serve', '--port', str(args.port)],
                cwd=REPO, env=environment, stdout=service_log, stderr=subprocess.STDOUT, start_new_session=True)
            atomic_json(root / 'web-service.json', dict(pid=service.pid, guard_pid=os.getpid(), port=args.port,
                url=url, started_unix=time.time(), deadline_unix=deadline, status='starting', ownership='v6_web_watch'))
        failures = 0
        startup_deadline = min(deadline - 15, time.time() + 90)
        while time.time() < deadline - 10:
            if service is not None and service.poll() is not None:
                raise RuntimeError('Owned web service exited: ' + str(service.returncode))
            stage = read_json(root / 'campaign.json').get('status') if (root / 'campaign.json').is_file() else 'preparing'
            web_deadline = acceptance_deadline(root, stage, deadline, time.time())
            if (root / 'web-acceptance.json').is_file():
                status('already_finished', campaign_stage=stage, result=read_json(root / 'web-acceptance.json'))
                return 0
            try:
                with urlopen(url + '/api/state', timeout=min(5, max(.1, deadline-time.time()-5))) as response:
                    state = json.load(response)
                failures = 0
            except (OSError, ValueError) as error:
                failures += 1
                status('waiting_for_service', campaign_stage=stage, error=str(error))
                if time.time() >= startup_deadline and failures >= 2:
                    raise RuntimeError('Web service is not healthy') from error
                time.sleep(min(5, max(.01, deadline-time.time()-10)))
                continue
            if web_deadline is not None:
                if state.get('busy'):
                    raise RuntimeError('Web service already has a match; watcher will not replace or stop it')
                if web_deadline-time.time() < 45:
                    raise TimeoutError('Insufficient frozen web acceptance budget')
                status('accepting', campaign_stage=stage, acceptance_deadline_unix=web_deadline)
                bounded_command([sys.executable, '-m', 'rmuc2027_research.rl.v6_replay_catalog', '--run', str(root)],
                    root/'web-replay-labels.log', min(web_deadline-30, time.time()+60), environment)
                command = ['node', str(REPO/'scripts/v6_web_acceptance.cjs'), '--run', str(root),
                    '--deadline-unix', str(web_deadline), '--url', url, '--start', '--seconds', '20']
                if args.playwright_module:
                    command.extend(['--playwright-module', args.playwright_module])
                code = bounded_command(command, root/'web-acceptance-browser.log', web_deadline, environment)
                result = read_json(root/'web-acceptance.json') if (root/'web-acceptance.json').is_file() else {'status':'missing'}
                status('finished', campaign_stage=stage, exit_code=code, result=result)
                return code
            if stage in TERMINAL:
                status('no_web_phase', campaign_stage=stage)
                return 1
            status('waiting', campaign_stage=stage, web_busy=state.get('busy'))
            time.sleep(min(30, max(.01, deadline-time.time()-10)))
        raise TimeoutError('Run absolute deadline reached')
    except BaseException as error:
        status('stopped', error=str(error))
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            return 130
        return 1
    finally:
        stop_owned(service, deadline)
        if service_log is not None:
            service_log.close()
        if service is not None:
            atomic_json(root/'web-service.json', dict(pid=service.pid, guard_pid=os.getpid(), port=args.port,
                url=url, deadline_unix=deadline, status='closed', exit_code=service.returncode, ownership='v6_web_watch'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--deadline-unix', type=float)
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--start-service', action='store_true')
    parser.add_argument('--playwright-module')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('port must be 1..65535')
    def interrupt(signum, _frame):
        raise KeyboardInterrupt('Signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    return watch(args)


if __name__ == '__main__':
    raise SystemExit(main())
