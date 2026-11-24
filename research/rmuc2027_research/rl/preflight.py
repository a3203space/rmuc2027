"""Argument validation that runs before torch and the simulator exist.

Everything here is pure config arithmetic, which is exactly the point: these
checks have to be usable from a shell preflight that must not pay for `import
torch` - or fail because of it. On a 4 GiB address-space cap `import torch`
dies with "libtorch_cpu.so: failed to map segment from shared object", so a
preflight that imported `learning.py` to reuse a validator would report a
broken environment instead of a bad flag.

`learning.py` re-exports these names, so existing call sites are unchanged.
"""


def rollout_block(config, num_envs):
    """Steps per PPO update, honouring the config's own `n_steps`."""
    if num_envs < 1:
        raise ValueError('num-envs must be positive')
    n_steps = (config.get('ppo') or {}).get('n_steps', 64)
    return n_steps*num_envs


def check_steps(config, num_envs, steps):
    """Reject a step budget that would leave a partial rollout behind.

    PPO only updates on complete rollouts, so a step count that is not a whole
    multiple of `n_steps * num_envs` either silently overshoots or silently
    drops the remainder - both make the reported `total_steps` disagree with
    what was actually trained. Returns the block size for the message.
    """
    block = rollout_block(config, num_envs)
    n_steps = (config.get('ppo') or {}).get('n_steps', 64)
    if steps is not None and (steps <= 0 or steps % block):
        raise ValueError(f'Steps must be a positive multiple of rollout size {block} '
                         f'(n_steps {n_steps} x num-envs {num_envs}); '
                         'no hidden overshoot')
    return block


def check_action_std(config):
    """Refuse an `ppo.action_std` that cannot be honoured inside the action box.

    SB3 parameterises the policy as N(mean, std) * action_scale with bounds
    [-1, 1], so one exploration step spans the whole action range once std
    reaches 1.0. A larger value does not explore further - it only feeds the
    clip, and the gradient then points at actions the environment never ran.
    The env's own action space is the authority on the bound, and every RL
    action here is a normalised map coordinate in [-1, 1] on both axes.
    """
    import math
    std = float((config.get('ppo') or {}).get('action_std', 1.0))
    if not math.isfinite(std) or std <= 0:
        raise ValueError('ppo.action_std must be finite and positive')
    if std > 1.0:
        raise ValueError(f'ppo.action_std must be <= 1.0 (got {std}); the action box is [-1, 1], '
                         'so a wider exploration radius is entirely clipped')
    return std


def steady_throughput(steps, block, init_seconds, sampling_seconds):
    """Recover the steady-state steps/s from a run whose first rollout is init.

    `sampling_seconds` is defined as the wall clock from the first
    `_on_rollout_start` to the last `_on_rollout_end`, so on a one-rollout run
    it is nothing but start-up: 16 envs, warp's lazy compilation and the
    navigation mesh all land inside that window. Dividing by it reports 12
    steps/s for a run that actually sustains 25 - which is exactly the
    discrepancy the first smoke run produced. Subtracting the first rollout
    removes the constant without touching the legacy definition, so a long run
    stays comparable with the 22h baseline.

    Returns `(seconds, steps_per_second)`, or `(None, None)` when there is no
    second rollout to average over. Non-finite inputs raise rather than
    silently producing a nan or inf rate - a nan would compare `False` against
    every threshold and read as "within budget".
    """
    import math
    values = {'steps': steps, 'block': block,
              'init_seconds': init_seconds, 'sampling_seconds': sampling_seconds}
    for name, value in values.items():
        if not math.isfinite(float(value)):
            raise ValueError(f'{name} must be finite (got {value})')
    steps, block = float(steps), float(block)
    init_seconds, sampling_seconds = float(init_seconds), float(sampling_seconds)
    if init_seconds < 0 or sampling_seconds < 0:
        raise ValueError('elapsed seconds cannot be negative')
    if steps < 2*block:
        return None, None
    steady_seconds = sampling_seconds-init_seconds
    if steady_seconds <= 0:
        return None, None
    return steady_seconds, (steps-block)/steady_seconds
