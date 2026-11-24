#!/usr/bin/env python3
"""Prove the outcome reward is actually paid, on one real, finished episode.

`training.json` cannot show this. The smoke run trains for one rollout of 16
envs x 64 steps x 0.5 s = 32 simulated seconds, which is far inside a 420 s
match, so every episode ends as `truncated`. `session.py` pays `outcome` only
when `terminated` - i.e. when the match itself finished - so the whole term
stayed at 0 and nothing on disk could tell "not wired up" from "never reached".

Rather than run 420 simulated seconds, this ends the match early with the
referee's own forfeit command, which is the same path a real win takes:
`Battle.finish('red', 'forfeit')` -> `winner='red'` -> the environment pays
`rewards['outcome'] * result_sign('red', 'red')` = +2.0.

Deliberately does not import `learning.py`, and does not inherit from SB3's
`BaseCallback` either: importing `stable_baselines3` executes its package
`__init__`, which imports torch, which dies on a 4 GiB address-space cap with
a message that reads like a broken install. Nothing here needs SB3 - the
harness only reads `infos`/`dones`/`rewards`, which are the env's own outputs.

    research/.venv/bin/python scripts/rl-episode-smoke.py \
        --config configs/rl/sentry-pooled.json \
        --run-dir data/research/pooled-v2-episode
"""

import argparse
import json
import math
from pathlib import Path
import sys
import time

from rmuc2027_research.rl.env import make_env
from rmuc2027_research.rl.preflight import check_steps, rollout_block
from rmuc2027_research.rl.session import load_config

# 32 simulated seconds at 0.5 s per step is exactly 64 steps, so with one env
# and n_steps 64 a single rollout is a single episode. Keeping them equal is
# what lets the harness treat "rollout ended" and "episode ended" as the same
# event, which is the whole point of running one env.
EPISODE_SECONDS = 32
FORFEIT_STEP = 20


class _Harness:
    """The subset of `learning.TraceCallback` that writes episodes.jsonl.

    Same field names and the same `dones`-driven write, minus checkpoints,
    backups, progress printing and the action trace. A divergence here would
    make this script prove nothing about the real run, so the row is built with
    the identical keys.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.num_timesteps = 0
        self.behavior = {'samples': 0, 'moving_commands': 0, 'modes': {}}

    def _on_step(self):
        for index, info in enumerate(self.locals['infos']):
            trace = info.get('action_trace') or {}
            move = (trace.get('executed_action') or {}).get('move', [0.0, 0.0])
            self.behavior['samples'] += 1
            self.behavior['moving_commands'] += int(math.hypot(*move) > 0.05)
            mode = trace.get('mode') or 'unknown'
            self.behavior['modes'][mode] = self.behavior['modes'].get(mode, 0) + 1
            if self.locals['dones'][index]:
                row = {'env': index, 'total_samples': self.num_timesteps,
                       **{key: info.get(key) for key in ('seed', 'time', 'winner', 'reason', 'metrics',
                                                         'reward_totals', 'control', 'bases', 'opponent',
                                                         'witness')}}
                with (self.path / 'episodes.jsonl').open('a') as episodes:
                    episodes.write(json.dumps(row, default=str) + '\n')
        return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--config', default='configs/rl/sentry-pooled.json')
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--episode-seconds', type=int, default=EPISODE_SECONDS)
    parser.add_argument('--forfeit-step', type=int, default=FORFEIT_STEP)
    args = parser.parse_args(argv)

    output = Path(args.run_dir)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise ValueError(f'run directory already exists: {output}; '
                         'pick another --run-dir or move the old one aside') from None

    config = dict(load_config(args.config))
    config.update(episode_seconds=args.episode_seconds, sim_backend='rapier')
    steps = int(round(args.episode_seconds / config['rl_seconds']))
    block = rollout_block(config, 1)
    if steps != block:
        raise ValueError(f'episode is {steps} steps but one rollout is {block}; '
                         'run this with --episode-seconds such that they match')
    check_steps(config, 1, steps)
    if not 0 <= args.forfeit_step < steps:
        raise ValueError('--forfeit-step must fall inside the episode')

    env = make_env(config)
    harness = _Harness(output)
    started = time.perf_counter()
    observation, _ = env.reset(seed=config['seed'])
    terminated = truncated = False
    info = {}
    try:
        for step in range(steps):
            # Queued, not applied: Battle.step() consumes pending referee
            # commands on the next tick, so the finish lands on step+1 and the
            # episode is ~(forfeit_step+2)*rl_seconds long, not the full 32 s.
            if step == args.forfeit_step:
                env.session.battle.queue_referee({'kind': 'forfeit', 'team': 'blue'})
            observation, reward, terminated, truncated, info = env.step([0.0, 0.0])
            harness.locals = {'infos': [info], 'dones': [terminated or truncated],
                              'rewards': [reward]}
            harness.num_timesteps = step + 1
            harness._on_step()
            if terminated or truncated:
                break
        seconds = time.perf_counter() - started
    finally:
        env.close()

    path = output / 'episodes.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []
    weights = config['rewards']['outcome']
    report = {'episode_seconds': args.episode_seconds, 'forfeit_step': args.forfeit_step,
              'steps_run': harness.behavior['samples'], 'episode_budget_steps': steps,
              'rollout_block': block,
              'terminated': terminated, 'truncated': truncated,
              'winner': info.get('winner'), 'reason': info.get('reason'),
              'simulated_seconds': info.get('time'), 'episodes': len(rows),
              'outcome_paid': rows[0]['reward_totals'].get('outcome') if rows else None,
              'outcome_weight': weights, 'wall_seconds': seconds, 'behavior': harness.behavior,
              'witness': info.get('witness'), 'policy_errors': (info.get('metrics') or {}).get('policy_errors')}

    # The gate. Any of these failing means the outcome term is not proven, and
    # a long run started on that assumption would spend 24 hours paying zero.
    problems = []
    if len(rows) != 1:
        problems.append(f'episodes.jsonl must hold exactly 1 episode, got {len(rows)}')
    if not terminated:
        problems.append('episode did not terminate; truncated episodes never pay outcome')
    if truncated:
        problems.append('episode was truncated as well as terminated')
    if info.get('winner') != 'red':
        problems.append(f'winner should be red after blue forfeits, got {info.get("winner")!r}')
    if info.get('reason') != 'forfeit':
        problems.append(f'reason should be forfeit, got {info.get("reason")!r}')
    if rows:
        paid = rows[0]['reward_totals'].get('outcome')
        if paid is None or abs(paid - weights) > 1e-9:
            problems.append(f'outcome paid {paid!r}, expected {weights} - the win term is not wired up')
    if harness.behavior['samples'] <= args.forfeit_step:
        problems.append(f'ran {harness.behavior["samples"]} steps, which is not past the forfeit '
                        f'at step {args.forfeit_step}; the episode ended for some other reason')
    if (info.get('metrics') or {}).get('policy_errors'):
        problems.append(f'{info["metrics"]["policy_errors"]} policy error(s) during the episode')
    # `snapshot()` reports only what this episode newly *witnessed*, not the
    # prior it falls back to. A 10 s episode of zero actions has nobody driving
    # into a tunnel or up a terrace, so empty lists are the correct answer here
    # - asserting non-empty would be asserting that the opponent moved, which
    # this script does not control. What is checkable is that the witness ran
    # and produced its documented shape for the opponent slot.
    witness = info.get('witness')
    if not isinstance(witness, dict) or set(witness) != {'tunnel', 'climb'}:
        problems.append(f'witness should be {{tunnel, climb}}, got {witness!r}')
    else:
        extra = [slot for slot in witness['tunnel'] + witness['climb'] if slot not in (1, 3, 4, 7)]
        if extra:
            problems.append(f'witness reported slots outside the opponent set: {extra}')

    (output / 'episode-smoke.json').write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, default=str), flush=True)
    if problems:
        for problem in problems:
            print(f'失败：{problem}', file=sys.stderr)
        return 1
    print(f'  outcome 已支付 {report["outcome_paid"]:+.1f} '
          f'(weights {weights:g})，{report["simulated_seconds"]:.1f}s 处由 forfeit 结束，'
          f'{harness.behavior["samples"]} 步')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
