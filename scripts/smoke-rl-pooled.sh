#!/bin/sh
# Pooled-opponent smoke test: proves per-episode opponent rotation works
# before committing hours of GPU time. Needs no torch for the env checks.
set -eu
RMUC_PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$RMUC_PROJECT_ROOT"
PY=research/.venv/bin/python
CONFIG=${1:-configs/rl/sentry-pooled.json}

if [ ! -x "$PY" ]; then
    echo "缺少 $PY，先按 docs/environment-setup.md 配置 research 环境。" >&2
    exit 1
fi

echo "== 1/4 配置与策略池校验 =="
"$PY" - "$CONFIG" <<'PY'
import json, sys
sys.path.insert(0, 'packages/core')
from rmuc2027_core.submissions import load_submission
from rmuc2027_core.models import normalize_team_lineup, DEFAULT_TEAM_LINEUP
cfg = json.load(open(sys.argv[1]))
pool = cfg.get('opponent_pool')
assert pool, '缺少 opponent_pool'
assert cfg['opponent'] in pool, 'opponent_pool 必须包含基线对手'
assert len(set(pool)) == len(pool), 'opponent_pool 有重复项'
for path in pool:
    manifest, _ = load_submission('research/strategies/' + path)
    assert manifest['mode'] == 'team', path
    assert normalize_team_lineup(manifest.get('team_lineup'), '2027') == DEFAULT_TEAM_LINEUP, path
print(f'  池 {len(pool)} 份，全部 team 模式且编队 = 默认编队')
print(f'  基线 {cfg["opponent"]}')
PY

echo "== 2/4 provenance 指纹 =="
"$PY" - <<'PY'
import json, sys
sys.path.insert(0, 'packages/core'); sys.path.insert(0, 'research')
from rmuc2027_research.rl.session import load_config, provenance
from rmuc2027_research.rl.portable import OBS_SIZE, OBS_VERSION
new = json.load(open('configs/rl/sentry-pooled.json'))
old = json.load(open('data/research/checkpoints/rl-run-77167c5700617683/run/experiment.json'))
cur = provenance(load_config(new))
# Physics and fixed strategies must still match the 22 h run: that is what keeps
# evaluation comparable and what evaluate/compose compare on.
for key in ('sources', 'sdk_hash', 'navigation', 'rules_sha256',
            'engine_binary_sha256', 'simulation'):
    assert cur.get(key) == old.get(key), f'指纹不一致: {key}'
# runtime_hash hashes portable.py, which carries the observation encoder. Bumping
# OBS 116 -> 131 changes it by construction, so resume and export against the old
# run are refused - that refusal is the contract, not a regression.
assert (cur['runtime_hash'] != old['runtime_hash']) == (OBS_SIZE != old['obs_size']), \
    'runtime_hash 应与 portable.py 的 OBS 变更同步'
print(f'  6 项物理/固定策略指纹与 22h 旧实验一致')
print(f'  OBS v{old.get("obs_version")}/{old.get("obs_size")} -> v{OBS_VERSION}/{OBS_SIZE}，'
      f'runtime_hash 已随之变更')
PY

echo "== 3/4 真实对局：每局对手轮换 =="
"$PY" - <<'PY'
import json, sys
sys.path.insert(0, 'packages/core'); sys.path.insert(0, 'research')
from rmuc2027_research.rl.session import load_config, ArenaSession, opponent_for
cfg = load_config(json.load(open('configs/rl/sentry-pooled.json')))
cfg['episode_seconds'], cfg['rl_seconds'], cfg['sim_backend'] = 6.0, 1.0, 'rapier'
for seed in (48, 49, 50, 51):
    session = ArenaSession(cfg)
    try:
        _, info = session.reset(seed)
        expected = opponent_for(cfg, seed)
        assert session.active_opponent == expected, (seed, session.active_opponent, expected)
        assert info['opponent'] == expected, seed
        assert len(session.battle.metadata['rl']['opponent_pool']) == len(cfg['opponent_pool'])
        session.step([0.0, 0.0])
        print(f'  seed={seed} %{len(cfg["opponent_pool"])}={seed % len(cfg["opponent_pool"]):2d} -> '
              f'{expected.split("/")[-1]}')
    finally:
        session.close()
PY

echo "== 4/4 反向检查：不带池时行为不变 =="
"$PY" - <<'PY'
import json, sys
sys.path.insert(0, 'packages/core'); sys.path.insert(0, 'research')
from rmuc2027_research.rl.session import load_config, ArenaSession
raw = json.load(open('configs/rl/sentry-pooled.json'))
raw.pop('opponent_pool')
cfg = load_config(raw)
cfg['episode_seconds'], cfg['rl_seconds'], cfg['sim_backend'] = 6.0, 1.0, 'rapier'
session = ArenaSession(cfg)
try:
    session.reset(48)
    assert session.active_opponent == cfg['opponent']
    assert 'opponent_pool' not in session.battle.metadata['rl']
except Exception:
    raise
finally:
    session.close()
print('  无池配置固定打基线对手，metadata 不含池字段')
PY

echo
echo "冒烟通过。正式训练命令见 docs/rl-randomized-opponents-plan.md。"
echo "注意：OBS 已升到 v2/131，116 维的 pooled-24h 与 22h 权重都不能 --init-from，"
echo "必须从头训练（--run-dir 换新目录，勿复用旧 run）："
echo "  cd $RMUC_PROJECT_ROOT && \\"
echo "  $PY -m rmuc2027_research.rl train --config ${CONFIG#./} --run-dir data/research/pooled-v2 \\"
echo "    --wall-seconds 86400 --num-envs 16 --device cuda"
