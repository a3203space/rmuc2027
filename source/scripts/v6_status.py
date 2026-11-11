#!/usr/bin/env python3
"""Read a v6 campaign without importing the simulator or touching its state.

Checkpoint counters come from the immutable generation, live coverage from
counts rather than a possibly inherited coverage cache. A running branch is
never treated as an accepted improvement. Cross-file observations are not atomic.
"""
import argparse
import datetime
import json
import math
from pathlib import Path
import time


FIELDS = ('started', 'valid_steps', 'short_completed', 'full_completed')


def read(path, issues, *, required=False):
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError('Expected JSON object')
        return value
    except (OSError, ValueError) as error:
        if required or path.exists():
            issues.append(str(path) + ': ' + str(error))
        return {}


def difference(current, baseline, field):
    values = (current.get(field), baseline.get(field))
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        return None
    result = values[0] - values[1]
    return result if result >= 0 else None


def coverage(current, baseline):
    if 'counts' not in current:
        return dict(cumulative_unique=None, run_unique_with_new_activity=None,
                    registered_scenarios_are_not_completed_coverage=True)
    counts = current.get('counts', {})
    old = baseline.get('counts', {})
    deltas = {key: {field: counts.get(key, {}).get(field, 0) - old.get(key, {}).get(field, 0)
                    for field in FIELDS} for key in counts.keys() | old.keys()}
    valid = all(value >= 0 for row in deltas.values() for value in row.values())
    return dict(
        cumulative_unique={field: sum(row.get(field, 0) > 0 for row in counts.values()) for field in FIELDS},
        run_unique_with_new_activity={field: sum(row[field] > 0 for row in deltas.values()) for field in FIELDS} if valid and baseline else None,
        registered_scenarios_are_not_completed_coverage=True)


def process(record, root):
    pid = record.get('pid')
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return dict(pid=None, matching_live_process=False)
    try:
        command = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
        matches = str(root).encode() in command
    except OSError:
        matches = False
    return dict(pid=pid, matching_live_process=matches)


def snapshot(root, now=None):
    root = Path(root).resolve()
    now = time.time() if now is None else now
    issues = []
    campaign = read(root / 'campaign.json', issues, required=True)
    budget = read(root / 'budget.json', issues) or campaign.get('budget', {})
    active = Path(campaign.get('active_run') or campaign.get('final_run') or root).resolve()
    stage = read(active / 'campaign.json', issues) if active != root else campaign
    ready = read(active / 'main/checkpoint-ready.json', issues)
    baseline = read(active / 'main/checkpoints/baseline/complete.json', issues)
    generation = (active / ready.get('directory', 'main/checkpoints/missing')).resolve()
    pointer_matches = False
    if active in generation.parents:
        complete = read(generation / 'complete.json', issues)
        pointer_matches = bool(ready) and complete == ready
    elif ready:
        issues.append('Checkpoint generation escapes active run')
    progress = read(active / 'main/progress.json', issues)
    training = read(active / 'main/training.json', issues)
    coordinator = read(active / 'sampler-coordinator.json', issues)
    prior_coordinator = read(active / 'main/checkpoints/baseline/coordinator-state.json', issues)
    receipt = read(active / 'github-backups/receipt.json', issues)
    backup = read(active / 'github-backups/status.json', issues)
    deadline = budget.get('deadline_unix')
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
        deadline = None
    rounds = campaign.get('rounds', [])
    accepted = [r for r in rounds if r.get('accepted_for_continuation') is True]
    rejected = [r for r in rounds if r.get('accepted_for_continuation') is False]
    progress_path = active / 'main/progress.json'
    try:
        progress_age = max(0, now - progress_path.stat().st_mtime)
    except OSError:
        progress_age = None
    selection = read(active / 'selection.json', issues)
    return dict(
        observed_unix=now, run=str(root), active_run=str(active),
        status=campaign.get('status', 'unknown'), active_status=stage.get('status', 'unknown'),
        deadline_unix=deadline, remaining_wall_seconds=max(0, deadline-now) if deadline is not None else None,
        orchestrator=process(read(root / 'orchestrator-launch.json', issues), root),
        worker=process(read(active / 'active-worker.json', issues), active),
        checkpoint=dict(directory=ready.get('directory'), total_samples=ready.get('samples'),
            update_id=ready.get('update_id'), pointer_matches_complete=pointer_matches,
            file_hashes_rechecked=False,
            run_delta={field: difference(ready, baseline, field) for field in ('samples', 'update_id', 'optimizer_steps')},
            declared_skill_unchanged=ready.get('skill_sha256') == baseline.get('skill_sha256') if ready.get('skill_sha256') else None),
        live_progress=dict(observed_samples=progress.get('actual_samples'), age_seconds=progress_age,
            simulation_seconds=progress.get('simulation_seconds'),
            note='Age can grow during real recovery preparation; not by itself a stalled worker'),
        training_report_status=training.get('status'),
        completed_rounds=dict(accepted=len(accepted), rejected=len(rejected),
            accepted_published_samples=sum(r.get('training', {}).get('published_samples', 0) for r in accepted),
            rejected_observed_samples=sum(r.get('training', {}).get('actual_samples', 0) for r in rejected)),
        coverage=coverage(coordinator, prior_coordinator),
        curriculum_actor_steps=coordinator.get('curriculum', {}).get('actor_steps', {}),
        active_leases=[{key: lease.get(key) for key in ('id', 'scenario_id', 'recovery_kind', 'valid_steps', 'seconds', 'prefix_outcome')}
                       for lease in coordinator.get('leases', {}).values()],
        backup=dict(phase=backup.get('phase'), last_verified_commit=receipt.get('commit') if receipt.get('remote_verified') else None,
            url=receipt.get('url'), checkpoint_samples=receipt.get('checkpoint', {}).get('samples'),
            verified_utc=receipt.get('verified_utc'), remote_rechecked_now=False),
        selection=selection.get('selected'), issues=issues,
        limitations=['Live files are sampled sequentially, not an atomic multi-file snapshot',
                     'Training activity and a canary pass do not establish deployment improvement'])


def render(row):
    remaining = row['remaining_wall_seconds']
    deadline = row['deadline_unix']
    wall = '未知' if remaining is None else f'{int(remaining)//3600}小时{int(remaining)%3600//60}分'
    end = datetime.datetime.fromtimestamp(deadline).astimezone().isoformat(timespec='seconds') if deadline else '未知'
    ck = row['checkpoint']; rounds = row['completed_rounds']; backup = row['backup']
    return '\n'.join([
        f"状态：{row['status']}；当前：{row['active_run']}",
        f'截止：{end}；剩余实际时间：{wall}',
        f"已发布样本：{ck['total_samples']}；本运行新增：{ck['run_delta']['samples']}；更新轮次：{ck['update_id']}",
        f"已结束分段：通过续训检查 {rounds['accepted']}，拒绝 {rounds['rejected']}；通过分段新增样本 {rounds['accepted_published_samples']}",
        f"本运行有新增活动的唯一场景：{json.dumps(row['coverage']['run_unique_with_new_activity'], ensure_ascii=False)}",
        f"恢复课程累计 actor 样本：{json.dumps(row['curriculum_actor_steps'], ensure_ascii=False)}",
        f"最近已核验备份：{backup['last_verified_commit']}；包含样本：{backup['checkpoint_samples']}",
        '训练表现与续训检查不代表已通过最终部署验收。',
        *(['读取问题：' + issue for issue in row['issues']]),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    row = snapshot(args.run)
    print(json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False) if args.json else render(row))
    return 1 if row['issues'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
