#!/bin/sh
# First 131-dimension RL run: preflight, launch, and acceptance checklist.
#
# Run this yourself in a normal terminal. It is NOT meant to be launched from a
# sandbox whose address-space limit is 4 GiB: `import torch` fails there with
# "libtorch_cpu.so: failed to map segment from shared object", which looks like
# a broken install but is only RLIMIT_AS. If the preflight reports that exact
# string, raise the limit (ulimit -v unlimited) or run outside the sandbox.
#
# Layout this script assumes:
#   configs/rl/sentry-pooled.json  OBS v2 / 131, 12-opponent pool, warp_kinematic
#   data/research/pooled-v2/         fresh run directory (must not exist, 1 rollout)
#   data/research/pooled-v2-episode/ fresh episode directory (must not exist)
#
# Six steps. Steps 0-1 and 3 use no torch; step 2 plays the opponent pool;
# step 4 is the real PPO smoke; step 5 finishes one episode through the
# referee so the outcome reward is proven to be paid - step 4 cannot show that,
# because its episodes are all still running when the rollout ends.
#
# SMOKE_STEPS counts whole PPO rollouts (default 1). One rollout is
# n_steps x NUM_ENVS = 1024 steps at 16 envs, which is the minimum a real
# gradient update needs.
#
# Run it from a normal shell, not from a tool sandbox: `set -eu` means the
# preflight stops the script rather than launching a half-broken run.
#
# Nothing here is resumed: OBS went 116 -> 131, so every older checkpoint is
# rejected by design. Long-run commands and their reasoning live in
# docs/rl-randomized-opponents-plan.md section 9.
set -eu

RMUC_PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$RMUC_PROJECT_ROOT"
PY=research/.venv/bin/python
CONFIG=${CONFIG:-configs/rl/sentry-pooled.json}
RUN_DIR=${RUN_DIR:-data/research/pooled-v2}
NUM_ENVS=${NUM_ENVS:-16}
SMOKE_STEPS=${SMOKE_STEPS:-1}
WALL=${WALL:-}

fail() { echo "失败：$*" >&2; exit 1; }

echo "== 0/6 环境自检 =="
[ -x "$PY" ] || fail "缺少 $PY，先按 docs/environment-setup.md 配置 research 环境"
"$PY" - <<'PY' || fail "python 依赖不可用（见上）"
import sys
try:
    import torch
except ImportError as exc:
    print(f'  torch 导入失败：{exc}', file=sys.stderr)
    print('  若原文是 "failed to map segment from shared object"，那是地址空间限制 '
          '(RLIMIT_AS)，不是安装问题：ulimit -v unlimited 后重试。', file=sys.stderr)
    raise SystemExit(1)
print(f'  torch {torch.__version__}  cuda_available={torch.cuda.is_available()}  '
      f'devices={torch.cuda.device_count()}')
if not torch.cuda.is_available():
    print('  !! 无 CUDA：只能 --device cpu，16 环境会非常慢', file=sys.stderr)
PY

echo "== 1/6 配置与观测维度（不需要 torch）=="
"$PY" - "$CONFIG" <<'PY' || fail "配置校验未通过"
import json, sys
from rmuc2027_research.rl.session import load_config, REQUIRED_REWARD_TERMS
from rmuc2027_research.rl.portable import OBS_SIZE, OBS_VERSION, CONTROL_SCALE
cfg = load_config(sys.argv[1])
assert OBS_VERSION == 2 and OBS_SIZE == 131, (OBS_VERSION, OBS_SIZE)
missing = REQUIRED_REWARD_TERMS - set(cfg['rewards'])
assert not missing, f'奖励项缺 {sorted(missing)}'
pool = cfg.get('opponent_pool') or []
assert pool and cfg['opponent'] in pool, 'opponent_pool 缺失或不含基线对手'
assert cfg['control_scale'] > 0, 'control_scale 必须为正'
print(f'  OBS v{OBS_VERSION}/{OBS_SIZE}  control_scale={cfg["control_scale"]} '
      f'(default {CONTROL_SCALE:g})')
print(f'  对手池 {len(pool)} 份  基线 {cfg["opponent"].split("/")[-1]}')
print(f'  backend={cfg.get("sim_backend", "rapier")} device={cfg.get("sim_device")}')
print(f'  奖励 {json.dumps(cfg["rewards"])}')
PY

echo "== 2/6 对手池冒烟（不需要 torch）=="
sh scripts/smoke-rl-pooled.sh "$CONFIG" || fail "对手池冒烟未通过"

echo "== 3/6 步数倍数预检 =="
# SMOKE_STEPS counts whole rollouts, not samples. It must be a multiple of
# n_steps x num_envs, and with 16 envs one rollout is already 1024 steps - so
# the default is "one rollout" rather than a fixed sample count that would be
# rejected for most --num-envs values.
ROLLOUTS=${SMOKE_STEPS:-1}
SMOKE_STEPS=$("$PY" - "$CONFIG" "$NUM_ENVS" "$ROLLOUTS" <<'PY'
import sys
# `preflight` is torch-free on purpose: this block must run on a machine whose
# address space cannot map libtorch, which is exactly where a bad flag would
# otherwise be discovered after the run directory already exists.
from rmuc2027_research.rl.session import load_config
from rmuc2027_research.rl.preflight import rollout_block, check_steps, check_action_std
cfg, envs, rollouts = load_config(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
if rollouts < 1:
    print('SMOKE_STEPS (whole rollouts) must be >= 1', file=sys.stderr)
    raise SystemExit(1)
block = rollout_block(cfg, envs)
steps = block*rollouts
try:
    check_steps(cfg, envs, steps)
    std = check_action_std(cfg)
except ValueError as exc:
    print(str(exc), file=sys.stderr)
    raise SystemExit(1)
print(f'  n_steps {cfg["ppo"]["n_steps"]} x {envs} envs = {block}；'
      f'{steps} 步 = {rollouts} 个 rollout；action_std 上限 {std:g}', file=sys.stderr)
print(steps)
PY
) || fail "步数预检未通过"
echo "  冒烟用 $SMOKE_STEPS 步（$ROLLOUTS 个 rollout）"

echo "== 4/6 真实 PPO 冒烟 =="
[ -e "$RUN_DIR" ] && fail "$RUN_DIR 已存在。这是新 OBS 的首跑目录，请先改名或换 RUN_DIR"
START=$(date +%s)
"$PY" -m rmuc2027_research.rl train --config "$CONFIG" --run-dir "$RUN_DIR" \
    --total-steps "$SMOKE_STEPS" --num-envs "$NUM_ENVS" --device cuda \
    || fail "冒烟训练失败（上面是真实报错）"
echo "  冒烟耗时 $(( $(date +%s) - START )) 秒"

echo "  验收冒烟产物："
"$PY" - "$RUN_DIR" <<'PY' || fail "冒烟产物不符合预期"
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
training = json.loads((run/'training.json').read_text())
exp = json.loads((run/'experiment.json').read_text())
assert training['actor_updated'], 'actor 未更新'
assert training['reload_prediction_equal'], '重载后预测不一致'
assert exp['obs_size'] == 131 and exp['obs_version'] == 2, exp['obs_size']
assert exp['observation']['control_scale'] == 1000, exp['observation']
std = training['action_std']
print(f'  steps={training["total_steps"]} actor_l2={training["actor_parameter_l2_change"]:.3e} '
      f'grad_epochs={training["gradient_epochs"]}')
print(f'  action_std start={std["start"]:.4f} end={std["end"]:.4f} bound={std["bound"]} '
      f'clipping={std["clipping"]}')
assert std['end'] is not None and std['end'] <= 1.0, '探索宽度越过动作边界'
rows = [json.loads(line) for line in (run/'updates.jsonl').read_text().splitlines() if line]
assert rows and all('action_std' in row for row in rows), 'updates.jsonl 未记录 std'
print(f'  updates.jsonl {len(rows)} 行，均含 action_std')
# Timing denominators. The legacy rate divides by a clock that starts inside
# the first rollout, where 16 env spawns and warp's compile live - on a single
# rollout that is the entire denominator, which is how a 25 steps/s run
# reported 12. The new fields split it out; these three assertions keep both
# definitions from drifting back into each other.
assert training['rl_rollouts'] == 1, f'冒烟应是 1 个 rollout，实得 {training["rl_rollouts"]}'
assert training['rl_steps_per_second_excluding_first'] is None, \
    '单轮不能声称有稳态吞吐：没有第二个 rollout 可平均'
assert abs(training['sampling_seconds']-training['init_seconds']) < 1e-6, \
    '旧口径必须未变：仅一轮时 sampling_seconds 应当就等于 init_seconds'
print(f'  计时：setup={training["setup_seconds"]:.2f}s init={training["init_seconds"]:.2f}s '
      f'rollouts={training["rl_rollouts"]} '
      f'旧口径={training["rl_steps_per_second"]:.2f} 步/秒（含启动，不代表稳态）')
PY

echo "== 5/6 跑满一局（forfeit 驱动的自然终止）=="
# The step above cannot verify `outcome`. Its episodes never end - 16 envs x 64
# steps x 0.5 s is 32 s inside a 420 s match - so every one is `truncated`, and
# session.py pays outcome only on `terminated`. A whole reward term could be
# dead and training.json would look fine. This runs one real episode and ends
# it through the referee's own forfeit, the same path a genuine win takes.
EPISODE_DIR=${EPISODE_DIR:-data/research/pooled-v2-episode}
[ -e "$EPISODE_DIR" ] && fail "$EPISODE_DIR 已存在，请改名或换 EPISODE_DIR"
"$PY" scripts/rl-episode-smoke.py --config "$CONFIG" --run-dir "$EPISODE_DIR" \
    || fail "跑满一局冒烟未通过：outcome 未被支付，先别开长跑"

echo "== 6/6 正式训练命令（已为你打印，不自动执行）=="
if [ -n "$WALL" ]; then
    echo "  WALL=$WALL 已给出，但请先确认冒烟通过再手工执行长时间训练。"
fi
cat <<EOF

  冒烟目录 \`$RUN_DIR\` 里是真实权重，正式训练请换新目录：

    cd $RMUC_PROJECT_ROOT && \\
    $PY -m rmuc2027_research.rl train \\
      --config $CONFIG \\
      --run-dir data/research/pooled-131-24h \\
      --wall-seconds 86400 --num-envs $NUM_ENVS --device cuda

  不要传 --init-from / --resume：OBS v1/116 的旧权重会被目标函数维度直接拒绝。

  长跑期间看三处：
    data/research/pooled-131-24h/updates.jsonl     每轮 action_std、loss、behavior
    data/research/pooled-131-24h/episodes.jsonl    每局 reward_terms、outcome、opponent、witness
    data/research/pooled-131-24h/training-latest.json（若存在）随时可读的进度快照

  判断「训练是否在学东西」的最低标准（缺一条就不要延长）：
    1. episodes.jsonl 的 opponent 字段覆盖池里多份对手，不是只有一个名字
    2. reward_terms.outcome 有正有负；若全程为 0 说明判据没生效
    3. reward_terms.position_control 与 terrain_buff 不再恒为 0
    4. updates.jsonl 的 action_std 没有一路顶到 1.0（顶到即为裁剪主导）
    5. 每局 reward 均值随步数上升，而不是原地抖动

  验收（训练结束后，原 Rapier 引擎跑，别再走 warp）：
    $PY -m rmuc2027_research.rl evaluate \\
      --model data/research/pooled-131-24h/model.zip \\
      --seeds 7 42 91 --sides red blue --duration 420 \\
      --device cuda --backend rapier --run-dir data/research/pooled-131-eval

EOF
echo "冒烟全部通过（含 outcome 支付已实测）。下一步只有在上面五条判断标准成立后才值得开 24 小时长跑。"
