# 随机对手 / 队友 / 兵种训练：可行性与实施方案

> 结论先行：**随机换对手和队友可行，只需改一处代码；随机换兵种有硬约束，因为红蓝编队必须一致。**

## 1. 检查点已落地

22 小时那次运行的完整目录已拉到本地，五个文件哈希与 `checksums.json` 逐一核对一致：

```
data/research/checkpoints/rl-run-77167c5700617683/
├── latest.zip            608,066 B（原始打包）
└── run/
    ├── model.zip         325,534 B   SB3 PPO 权重，2,043,616 步
    ├── experiment.json     5,324 B   完整来源与运行时指纹
    ├── progress.json         120 B   total_steps / elapsed_seconds
    ├── updates.jsonl     1,336,224 B  1,988 条更新记录
    └── episodes.jsonl    1,472,852 B  2,416 局对战记录
```

`experiment.json` 记录的运行时指纹与当前仓库**逐项相等**，这是能续训的前提：

| 指纹 | 训练时 | 当前仓库 | 匹配 |
|---|---|---|---|
| `sdk_hash` | `7a38b655…` | `7a38b655…` | ✅ |
| `runtime_hash` | `5a764c05…` | `c6d78fb8…` | ❌ 见下 |
| `rules_sha256` | `d9e6396d…` | `d9e6396d…` | ✅ |
| `engine_binary_sha256` | `d372c9a5…` | `d372c9a5…` | ✅ |
| `navigation.preparation_sha256` | `c9791016…` | `c9791016…` | ✅ |
| `obs_version` / `obs_size` | 1 / 116 | **2 / 131** | ❌ 见下 |

> **2026-11-22 更新**：`runtime_hash` 与观测契约已变更。敌方槽位从 8 维扩到 11 维（新增 `can_tunnel / can_climb / is_omni` 三个能力位），`OBS_VERSION` 升到 2、`OBS_SIZE` 扩到 131，`runtime_hash` 随之变化（它是 `sha256(portable.py)`）。
>
> 后果：22h 旧实验**不能 `--resume`、不能 `--init-from`、不能 `export`**（`learning.py:229`、`exporting.py:22`、`migrate.py:17` 三处都会拒绝）。其余 6 项物理与固定策略指纹仍逐项相等，评估仍可比。编码依据见 `docs/obs-dimension-chassis-audit.md`。

## 2. 现状：只有一个固定对手，没有轮换机制

`research/rmuc2027_research/rl/session.py` 的 `ArenaSession.reset()`：

```python
sources = (resource(cfg['friendly']), resource(cfg['opponent']))
red, blue = sources if cfg['team'] == 'red' else sources[::-1]
self.battle = Battle(red, blue, self.seed, ...)
```

每局重开都无条件加载配置里那两份提交，没有任何采样、轮换或课程。整个 `rl/` 目录搜不到 opponent pool / self-play / curriculum。

配置里 `friendly` 与 `opponent` 的角色语义：

- 学习方 `team: red` 时，`friendly` → 红方（我方 4 台队友），`opponent` → 蓝方（敌方 5 台）
- 学习方 `team: blue` 时两者互换
- 学习的那台哨兵由 `controllers` 里的 `{"red:7": {"kind": "learning"}}` 顶替，其余 9 台都由这两份脚本驱动

## 3. 三项需求的可行性

### 3.1 随机换对手：可行

策略库里 `mode=team` 的完整阵容可用者：

| 提交 | 兵种编队 |
|---|---|
| `claude_opus55_xhigh_2027_hold` | 2711 / 2731 / 2732 / 2761 / 2771 |
| `dsv41_1017_heavy` | 同上 |
| `dsv41_1017_pair` | 同上 |
| `dsv41_1017_sentry` | 同上 |
| `dsv41_2027_control` | 同上 |
| `gemini38_2027_blitz` | 同上 |
| `gemini38_2027_hunter` | 同上 |
| `gemini38_2027_tech_overload` | 同上 |
| `gpt61_2027_control` | 同上 |
| `gpt61_2027_pressure` | 同上 |
| `step5_2027_echo` | 同上 |
| `gpt61_2027_trident` | **不同**：2712 / 305 / 305 / 2763 / 2772 |
| `gpt61_2027_lineup_serial` | **不同**：2712 / 305 / 3 / 2763 / 2772 |

12 份同编队可直接进对手池，2 份异编队的要单独处理（见 3.3）。

### 3.2 随机换队友：可行，但语义要清楚

`friendly` 也是每局加载一份脚本，可以同样池化。注意它是**整队 4 台队友**，不是逐台替换：`squad` 或 `team` 模式下红方 1/3/4/6 号始终来自同一份提交。若想逐台混搭，需要改用 `controllers` 里的 `script` / `frozen` 逐槽位指定——但这样 `friendly` 的 `opening` 行为会退化，且 `baseline_module()` 只读 `friendly`，改动风险较大。建议先做整队池化。

### 3.3 随机换兵种：有硬约束

`research/rmuc2027_research/strategies/battle.py:56`：

```python
if side_lineups is None and not practice_lineup and (selected[0]!=selected[1] or any(s!=chosen for s in selected)):
    raise ValueError('Team matches require identical registered robot lineups; develop both strategies on the same lineup')
```

即**两队必须使用同一套兵种编队**，红蓝各自独立编队只在 `practice_lineup=True` 时允许，而该开关只在 `web/training.py:166` 的练习模式里用到，RL 路径没有暴露。

因此「随机兵种」只能是**在红蓝一致的前提下**随机整队编队，例如每局从下列编队里抽一套并同时应用到双方：

```python
LINEUP_POOL = [
    {"1":2711, "3":2731, "4":2732, "6":2761, "7":2771},
    {"1":2712, "3":2731, "4":2732, "6":2761, "7":2771},
    {"1":2711, "3":305,  "4":2732, "6":2762, "7":2772},
    {"1":2713, "3":3,    "4":4,    "6":2763, "7":2773},
]
```

可用的合法编队组合（`robot_models()` 共 20 个模型）：

| 角色 | 数量 | 型号 | 对应槽位 |
|---|---|---|---|
| 英雄 | 4 | 2711–2714 | 1 |
| 步兵 | 10 | 3, 4, 7, 301–305, 2731, 2732 | 3, 4 |
| 空中 | 3 | 2761–2763 | 6 |
| 哨兵 | 3 | 2771–2773 | 7 |

注意 `7` 号是哨兵槽位，`2771/2772/2773` 属哨兵；步兵型号 `7` 是另一回事，不能填进 7 号槽。`normalize_team_lineup()` 会校验角色与赛季规则，非法组合直接抛错。

## 4. 实施方案

### 4.1 代码改动（`session.py`，约 30 行）

在 `ArenaSession.reset()` 里，把固定的两份来源换成按种子采样的池：

```python
def _resolve_sources(self, cfg):
    """per-episode池采样；池为空时退化为固定来源"""
    pools = cfg.get('source_pools') or {}
    friendly = cfg['friendly']
    opponent = cfg['opponent']
    if pools.get('friendly'):
        friendly = pools['friendly'][self.seed % len(pools['friendly'])]
    if pools.get('opponent'):
        opponent = pools['opponent'][self.seed % len(pools['opponent'])]
    return resource(friendly), resource(opponent), friendly, opponent
```

要点：

1. **用 `self.seed` 取模而不是 `self.rng`**，这样采样可复现，且同一种子的红蓝两侧共用一个选择，便于事后按 `episodes.jsonl` 的 seed 复盘。
2. **必须把选中的版本写进 `provenance()`**，否则 `experiment.json` 只记一份来源，续训比对和事后归因都会失效。`provenance(config)` 接受 config，可在其中注入 `_active_sources` 供它读取。
3. **编队池同理**，在 `reset()` 里选出后再传给 `Battle(side_lineups=...)`，并同时写进 metadata。

### 4.2 配置扩展

```json
"source_pools": {
  "friendly": ["examples/claude_opus55_xhigh_2027_squad"],
  "opponent": [
    "examples/claude_opus55_xhigh_2027_hold",
    "examples/dsv41_1017_heavy",
    "examples/dsv41_1017_pair",
    "examples/dsv41_1017_sentry",
    "examples/dsv41_2027_control",
    "examples/gemini38_2027_blitz",
    "examples/gemini38_2027_hunter",
    "examples/gemini38_2027_tech_overload",
    "examples/gpt61_2027_control",
    "examples/gpt61_2027_pressure",
    "examples/step5_2027_echo"
  ]
},
"lineup_pool": [
  {"1":2711,"3":2731,"4":2732,"6":2761,"7":2771},
  {"1":2712,"3":2731,"4":2732,"6":2761,"7":2771}
]
```

### 4.3 续训的现实约束：换池子等于换实验

`learning.py:200` 的续训校验：

```python
for key in ('config', 'sources', 'sdk_hash', 'obs_version', 'obs_size', 'simulation'):
    if metadata[key] != previous[key]:
        raise ValueError(f'Resume incompatibility: {key}')
```

`config` 和 `sources` 都必须完全一致。一旦加入 `source_pools`，`config` 必然变化，`sources` 也会因 provenance 变化而不等，**`--resume` 会被直接拒绝**。

所以「在 22h 模型基础上换随机对手继续训练」有两条路：

| 方案 | 做法 | 代价 |
|---|---|---|
| A. 从权重出发新开实验 | 不传 `--resume`，改为从 `model.zip` 加载权重初始化新 PPO | 优化器状态、`num_timesteps`、RNG 全部重置；相当于在好权重上重新开始 |
| B. 放宽续训校验 | 给 `train()` 加 `allow_source_change=True`，跳过 `config`/`sources` 比较 | 需要接受「实验元数据不再连续」，后续无法把两段拼成一条曲线；也违背项目现有的严格校验设计 |

推荐 A：它符合 SB3 的常规做法，且不会破坏 `experiment.json` 的严格性。做法是给 `train()` 加一个 `init_from` 参数：

```python
if init_from:
    model = TimedPPO.load(init_from, env=vec, device=device)
    model.num_timesteps = 0
```

## 5. 前置阻塞：虚拟内存限制

当前 shell 的 `ulimit -v` 是 **4 GB**，且为硬限制，`ulimit -v unlimited` 报「不允许的操作」：

```
$ ulimit -v
4194304
```

`research/.venv` 里 torch 是 `2.11.0+cu128`，`libtorch_cuda.so` 单文件 912 MB，加载时映射的地址空间远超 4 GB：

```
ImportError: libtorch_cuda.so: failed to map segment from shared object
```

文件本身没坏（`ctypes.CDLL` 直接 dlopen 成功），纯粹是地址空间被限。**在这个限制解除前，任何 GPU 训练都启动不了**，与是否改代码无关。

解除方式（需要 root 或调整会话管理器配置）：

```bash
# 临时：新 shell 里执行
ulimit -v unlimited

# 永久：确认没有 systemd LimitAS 覆盖后写入
sudo sed -i 's/^DefaultLimitAS=.*/DefaultLimitAS=infinity/' /etc/systemd/system.conf
```

注意：硬限制只能由 root 降低或提高，普通用户无法自行放宽。若该限制由外层会话管理器（而非 systemd）施加，需要改那个管理器的启动参数。

## 6. 建议的执行顺序

1. 解除 `ulimit -v`，跑通 `research/.venv/bin/python -c "import torch; print(torch.cuda.is_available())"`
2. 加 `source_pools` / `lineup_pool` 支持与 `init_from` 参数
3. 先用 `--num-envs 2 --total-steps 256` 冒烟，确认 `experiment.json` 里能看到池采样记录
4. 用 `rl evaluate` 在固定种子集上对比「原 22h 模型」与「随机对手续训模型」，避免只看训练内胜率
5. 正式长跑，扩 `--num-envs`

## 7. 必须提醒的评估陷阱

原来 99% 的胜率是对 `claude_opus55_xhigh_2027_hold` 单一固定对手的拟合。换成随机对手池后，**训练内胜率一定会明显下降**，这是正常的（对手变难了），不能据此认为模型变差。判断标准应该是同一组冻结种子上的 `rl evaluate` 结果，且对手集合要固定。

## 8. 修订说明（已实施，取代第 4 节的 `source_pools` 草稿）

第 4 节写的 `source_pools` 是未实施的草稿，**实际落地的字段名是 `opponent_pool`**，并且本阶段只做"对手整队随机、队友保持固定"，不引入 `lineup_pool`。以本节为准。

### 8.1 最终实现的字段与语义

| 项 | 实现 |
|---|---|
| 配置字段 | `opponent_pool: list[str]`（与 `opponent` 同格式的 `examples/xxx` 路径） |
| 采样 | **每局一次**，`opponent_for(config, seed)` = `pool[seed % len(pool)]`，纯函数 |
| 队友 | 固定 `config['friendly']`，不池化 |
| 基线对手 | 必须出现在池内；`seed % len(pool) == 0` 那一档即原固定对手 |
| provenance | `sources['opponent']` **恒为基线对手**；本局对手与整池清单另存 `active_opponent` / `opponent_pool` |
| 续训 | 新增 `--init-from <checkpoint>`，新建实验目录、计数器归零；与 `--resume` 互斥 |

### 8.2 为什么 `sources['opponent']` 必须恒为基线，而不是"每局对手"

这是整个设计的关键不变量。`resume`（`learning.py:200-209`）、`evaluate`（`learning.py:307-315`）、`export`（`exporting.py:24-31`）三处都比较 `sources`。若让 `sources['opponent']` 跟随每局抽样：

- 同一实验每次运行得到的 `sources` 不同，`--resume` 立刻被拒；
- `evaluate`/`export` 重建 `provenance` 时无法复现训练时的抽样序列，必然 `incompatible`；
- "evaluate 用哪个对手"这一问题失去唯一答案。

反过来，保持恒为基线后，已实测：新配置的 `sources`/`sdk_hash`/`simulation` 与 22h 旧实验**逐项相等**，因此新模型仍可正常 `evaluate` 和 `export`。每局的真实对手记在 `battle.metadata['rl']` 与每步 `info['opponent']` 里，供人复盘。

### 8.3 为什么用 `seed % len(pool)` 而不是 `self.rng`

`SubprocVecEnv(start_method='spawn')` 与 `WarpVecEnv` 的每个 worker 各持一份 session，`random.Random` 状态互相独立，抽样结果只会碰巧一致。而 `seed` 由 SB3 的 `VecEnv.reset()` 显式下发到每个 worker，因此按 `seed` 取模可在事后用 `episodes.jsonl` 里的 `seed` 精确还原该局对手。

### 8.4 已知行为

- `--num-envs` 不同 → 首局种子集合（`config['seed']+idx`）不同 → 对手序列不同。episode 内容本来也会不同，不视为缺陷。
- 新配置的实验**不能** `--resume` 22h 旧实验（`config` 多了 `opponent_pool` 键），这是有意保留的强校验：换池即换实验。
- 池内任一策略被改动，下一局 `provenance` 即抛错，后续 `--resume` 被拒。这是期望行为。
- 16 环境时首批对手中只有 1 个（`idx=6`）打旧对手 `hold`，即旧对手占比 1/12。训练内胜率因此不可与 22h 的 99% 直接比较。

### 8.5 验证现状

- `provenance` 向后兼容：无 `opponent_pool` 的配置输出与改造前**逐字节相同**（已实测 7 项指纹全等）。
- `tests/python/test_rl_opponent_pool.py`：13 项，覆盖轮换纯函数、基线占位、池校验、`sources` 不变量、记录稳定性。
- `scripts/smoke-rl-pooled.sh`：四步冒烟（池校验 → 指纹比对 → **真实对局轮换** → 无池反向检查），**不需要 torch**，已跑通。真实对局用 rapier 后端跑 `seed=48/49/50/51`，分别命中 `hold/team/heavy/pair`，`active_opponent`、`info['opponent']`、`metadata['rl']` 三处一致。
- **`episodes.jsonl` 的字段白名单已补 `opponent`**（`learning.py:132`）。原先该行只挑 `seed/time/winner/reason/metrics/reward_totals/control/bases`，新加的 `info['opponent']` 不会落盘，会导致"按对手分列统计胜率"无法进行。`evaluate` 用 `**info` 全量展开，`evaluation.json` 原本就有。
- 尚待 GPU 环境下补跑：`--init-from` 的 `train()` 端到端冒烟（`test_rl_wall_time` 亦依赖 torch）。


配置文件：`configs/rl/sentry-pooled.json`（= 22h 运行配置 + 12 份同编队对手池）。

## 9. 24 小时训练操作手册

### 9.1 前置：确认 torch 能加载

`train` 需要 torch。若 `import torch` 报 `libtorch_cuda.so: failed to map segment from shared object`，说明所在进程的**虚拟地址空间上限太小**（不是磁盘或显存不足）：

```bash
cat /proc/self/limits | grep "address space"
research/.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

- **仅加载 `libtorch_cuda.so` 就需要约 3.99 GB 虚拟地址空间**（实测 `VmPeak` 4,186,888 kB），加上 CUDA 上下文后 4 GB 上限必然失败。
- 正常的桌面会话没有这个限制（`systemctl show user@1000.service -p LimitAS` → `infinity`）。若被限制，在**普通终端**（非受限容器/沙箱）里执行即可。
- 提高软限制：`ulimit -v unlimited`。硬限制只能由 root 放宽，普通用户无权修改。

### 9.2 冒烟（30 秒，先做）

```bash
sh /media/rm/E3E258F427E8E547/rmuc2027/scripts/smoke-rl-pooled.sh
```

脚本自己 `cd` 到项目根目录，任何位置执行都行，不需要设任何环境变量。

四步全过再进入正式训练。它不需要 torch，可随时重跑。

### 9.3 正式 24 小时训练

```bash
cd /media/rm/E3E258F427E8E547/rmuc2027 && \
research/.venv/bin/python -m rmuc2027_research.rl train \
  --config configs/rl/sentry-pooled.json \
  --run-dir data/research/pooled-24h \
  --init-from data/research/checkpoints/rl-run-77167c5700617683/run/model.zip \
  --wall-seconds 86400 \
  --num-envs 16 \
  --device cuda
```

两点说明：

- `ROOT` 由包路径自动推导（`packages/core/rmuc2027_core/paths.py:4`），**不需要任何 `RMUC_*` 环境变量**。实测：清空环境变量、甚至在 `/tmp` 下运行，`ROOT` 仍正确解析为项目根。
- 但配置与 `--run-dir` 是相对路径，只在项目根目录下有效，所以行首保留 `cd`（与后面的命令用 `&&` 连成一行，只影响这条命令，不改动当前 shell）。若不想 `cd`，把它们换成绝对路径即可。

参数说明：

| 参数 | 值 | 理由 |
|---|---|---|
| `--init-from` | 22h 的 `run/model.zip` | 只读权重，旧文件不会被改写；计数器归零，训练历史从 0 起算 |
| `--wall-seconds` | `86400`（24h） | 由时钟停止。步数上限被推导成 1.024e12，远大于 24h 能跑的约 218 万步，因此**不会被步数打断** |
| `--num-envs` | `16` | 与 22h 一致，统计口径可比 |
| `--device` | `cuda` | GPU 训练 |
| `--total-steps` | **不传** | 传了反而要在中断续训时额外算 `--additional-steps` |

不要传 `--resume`：它与 `--init-from` 互斥，且新配置含 `opponent_pool`，`--resume` 22h 旧实验本就会被拒（设计意图）。

### 9.4 容量与速率预期

> **更正（2026-11-23）：本节原基准的数据源错了，90,709 步/小时作废。**
>
> 原文写「`progress.json`：2,043,616 步 / 81,105.67 秒」。核对 `data/research/pooled-24h/`
> 的实际落盘后，22h 那轮的真实情况是：
>
> | 来源 | 读数 |
> |---|---|
> | `updates.jsonl` | **440 条**，末次 `total_samples = 448,448`，`gradient_epochs = 1,748` |
> | `episodes.jsonl` | **512 局** |
> | `training.json` | `total_steps = 61,376`，`requested_wall_seconds = null`，`stop_reason = step_limit` |
> | `backups/*.status.json` | 70 份（66 uploaded / 2 pending / 2 retry_pending） |
>
> 且 `updates.jsonl` 里有两处断点（`35840 → 37296`、`62896 → 61376`，后者是**回退**），
> 说明那轮**中途续跑过**。因此 22h 是「多段会话累计 448K 步」，不是「连续 200 万步」，
> 2,043,616 这个数字在任何文件里都找不到。
>
> 更要紧的是：那轮的 `train/std` 全程越界（见 §9.9），本来就不该被当作性能与学习效果的
> 基准。**下表全部作废，等 24h 跑完用新数据重建。**保留它只是为了记录当时是怎么想的。

按 22h 运行的实测值推算（`progress.json`：2,043,616 步 / 81,105.67 秒 = **90,709 步/小时**，每步 39.69 ms，每局采样 846 步）：

| 指标 | 24h 预期 | 22h 实际（对照） |
|---|---|---|
| 采样步数 | ≈ 2,177,000 | 2,043,616 |
| 完整对局 | ≈ 2,574 局（107 局/h） | 2,416 局 |
| rollout 数（每块 1024 步） | ≈ 2,118 | 1,988 |
| 梯度更新（每块 4 epoch） | ≈ 8,471 | 7,980 |

**注意**：这是假设 GPU 吞吐与 22h 相同。若池中某些对手（如 `tech_overload`、`pressure`）的单位策略耗时更长，实际步数会略低；并发仍是 16，瓶颈在物理步进，影响通常很小。

开跑 10 分钟后用 `updates.jsonl` 间隔采样反算真实速率（**不要用 `progress.json`**——它由 `backup.py` 生成，只在每 300 秒的 GitHub backup 时写入，不是实时；`updates.jsonl` 本身也没有墙钟时间戳，所以要用文件 mtime）：

```bash
cd /media/rm/E3E258F427E8E547/rmuc2027
f=data/research/pooled-24h/updates.jsonl
s1=$(wc -l < $f); t1=$(stat -c %Y $f)
sleep 120
s2=$(wc -l < $f); t2=$(stat -c %Y $f)
research/.venv/bin/python -c "
s1,s2,t1,t2=$s1,$s2,$t1,$t2
block=1024
steps=(s2-s1)*block; secs=t2-t1
rate=steps/secs*3600
print(f'{s2-s1} 个 rollout / {secs} 秒 = {rate:,.0f} 步/h')
print(f'24h 推算 {rate*24:,.0f} 步，约 {rate*24/846:,.0f} 局')
print(f'与 22h 基准 90,709 步/h 相比: {rate/90709*100:.0f}%')
"
```

#### 9.4.1 计时字段口径（`training.json`）

一次短跑曾报告 **12.57 步/秒**，看似比 22h 的 25.2 慢一倍，实为测量口径问题而非性能回归。原因：`TraceCallback._on_rollout_start` 在第一个 rollout 一开始就启动 `perf_counter`，而建 16 个环境、warp 惰性编译、导航网格准备全落在这段区间内。22h 跑 1988 次把它们摊薄到 1/1988；冒烟只跑 1 次，初始化占满整个分母。

因此 `training.json` 里的时间字段**不是一个口径**，读之前先看下表：

| 字段 | 起点 | 含首次启动开销 | 用途 |
|---|---|---|---|
| `setup_seconds` | `train()` 入口 | 是（仅此字段） | `vector_env` + callback + 策略构造的耗时，与采样无关 |
| `init_seconds` | 第一个 `_on_rollout_start` | 是（整段首轮） | 首轮的真实耗时，含编译与网格 |
| `sampling_seconds` | 第一个 `_on_rollout_start` | **是** | 旧口径，**勿改**：90,709 步/h 基准按它建立 |
| `sampling_seconds_excluding_first` | 第二个 `_on_rollout_start` | 否 | 稳态采样时长 |
| `rl_steps_per_second` | — | 是 | `总步数 / sampling_seconds`，短跑会被启动开销压低 |
| `rl_steps_per_second_excluding_first` | — | 否 | `(总步数 - block) / (…excluding_first)`；**不足两轮时为 `null`** |
| `rl_rollouts` | — | — | 整轮 rollout 个数，决定上面那格是不是 `null` |

三条读法：

1. **短跑只看 `_excluding_first` 与稳态推导**。只有一轮时它是 `null` —— 一轮没有稳态可报，编一个数正是当初 12.57 的来源。想看稳态至少跑两轮，或用 `preflight.steady_throughput(steps, block, init_seconds, sampling_seconds)` 反推（22h 的 `2,043,616 / 1024 / 40.55 / 81,105.67` 反推得 25.20 步/秒，与实测一致）。
2. **不要为了"更好看"换 `sampling_seconds` 的分母**。90,709 步/h、每步 39.69 ms 都是用含首轮的口径算的，换分母会让 24h 与 22h 混入第二个变量，无法对比。
3. **`progress.json` 的 `elapsed_seconds` 是本进程墙钟，不是跨会话累计**。续训后它从零重新计。

另外，22h 那一轮**只留下 `progress.json`，`training.json` 从未落盘**（当时 `--wall-seconds` 到点后被 `KeyboardInterrupt` 之外的路径中断）。所以与历史对比只能用它派生出的 `总步数 / elapsed_seconds`，而不是 `training.json` 里的字段——这也是本项目此后把 `--wall-seconds` 长跑跑到自然 `stop_reason: wall_time_limit` 的原因。

**`train/std` 是真实 std，不是 log 值。** SB3 记的是 `th.exp(self.policy.log_std).mean()`
（`stable_baselines3/ppo/ppo.py:295`），所以读 `updates.jsonl` 时可以直接与动作上界 `1.0` 比。
把 1.969 当成 log 值会算出 `exp(1.969)=7.16`，那是误读。这个区别很关键——见 §9.9。



### 9.5 开跑后 10 分钟内必须核对的三件事

```bash
RUN=data/research/pooled-24h

# 1) 计数器确实归零（关键：不是 2043616 起步）
research/.venv/bin/python -c "
import json; t=json.load(open('$RUN/training.json'))
print('total_steps', t['total_steps'], '| actual', t['actual_steps'], '| grad_epochs', t['gradient_epochs'])"

# 2) 池清单与基线对手已落盘
research/.venv/bin/python -c "
import json; e=json.load(open('$RUN/experiment.json'))
print('active_opponent', e['active_opponent'])
print('pool entries', len(e['opponent_pool']))
print('sources[opponent] version', e['sources']['opponent']['version'])"

# 3) 对手真的在轮换（应看到多个不同名字）
research/.venv/bin/python -c "
import json, collections
c=collections.Counter(json.loads(l)['opponent'].split('/')[-1] for l in open('$RUN/episodes.jsonl') if 'opponent' in json.loads(l))
for k,v in c.most_common(): print(f'{v:5d}  {k}')"
```

判定：第 1 项 `total_steps` 应是小数值（如 1024、2048），**不是** 2043616；第 2 项池为 12 项；第 3 项应逐渐覆盖 12 份对手，而不是只有一个名字。

### 9.6 训练期间监控

```bash
# 胜率与局数（每局一行）
tail -f data/research/pooled-24h/episodes.jsonl

# 每块更新后的 approx_kl / clip_fraction / explained_variance
tail -f data/research/pooled-24h/updates.jsonl
```

**预期现象**：训练内胜率会从 22h 的 98–100% 明显下降。原因是旧模型只对 `hold` 过拟合，而池中 `hold` 仅占 1/12。这不是退步。判断标准是固定种子集上的 `evaluate`（见 9.8）。

**异常信号与处置**：

| 现象 | 处置 |
|---|---|
| `approx_kl` 长期 > 0.05 且 value loss 暴涨 | 降价学习率：写 `data/research/pooled-24h/learning-rate.json`，内容 `{"learning_rate": 0.0001, "reason": "pool shock"}`，下一个 rollout 生效 |
| 熵（`train/entropy_loss`）趋近 0，动作退化 | 停训，改配置提高 `ent_coef` 后重开 |
| `episodes.jsonl` 长时间只有 1 个对手 | 池未生效，回看 9.5 第 3 项 |
| 出现 `Resume incompatibility` | 池内策略被改过，`git status research/strategies/examples` 排查 |

### 9.7 中断与续训

新实验跑起来后用 `--resume`（**不要**再用 `--init-from`，否则会重开实验目录）：

```bash
cd /media/rm/E3E258F427E8E547/rmuc2027 && \
research/.venv/bin/python -m rmuc2027_research.rl train \
  --config configs/rl/sentry-pooled.json \
  --run-dir data/research/pooled-24h \
  --resume data/research/pooled-24h/model.zip \
  --additional-steps <再跑多少步> \
  --wall-seconds <再跑多少秒> \
  --num-envs 16 --device cuda
```

前提：池内 12 份策略未被修改，且运行环境指纹未变。`--additional-steps` 必须是 1024（`64×16`）的倍数；只给 `--wall-seconds` 也可以。

### 9.8 训练后评估

```bash
cd /media/rm/E3E258F427E8E547/rmuc2027

# 新模型
research/.venv/bin/python -m rmuc2027_research.rl evaluate \
  --model data/research/pooled-24h/model.zip \
  --run-dir data/research/eval-pooled \
  --seeds 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 \
  --device cuda

# 旧 22h 模型作对照（其 experiment.json 无池，会一直打 hold）
research/.venv/bin/python -m rmuc2027_research.rl evaluate \
  --model data/research/checkpoints/rl-run-77167c5700617683/run/model.zip \
  --run-dir data/research/eval-22h \
  --seeds 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 1011 1012 \
  --device cuda
```

**分三列报告**，不要只看总数：

1. `seed % 12 == 0` 的种子（对 `hold`）——**唯一**可与 22h 旧数据直接比较的部分；
2. 其余单个对手各自的胜率——看有没有被某一族打崩；
3. 整池汇总——作为总体强度指标。

选 12 个连续种子（1001–1012）正好让 `%12` 覆盖 0–11，每个对手各出现一次，是最省算力的均衡方案。对应关系（已实测）：

| seed | 1001 | 1002 | 1003 | 1004 | 1005 | 1006 | 1007 | 1008 | 1009 | 1010 | 1011 | 1012 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `%12` | 5 | 6 | 7 | 8 | 9 | 10 | 11 | **0** | 1 | 2 | 3 | 4 |
| 对手 | dsv41_control | blitz | hunter | tech_overload | gpt61_control | pressure | echo | **hold** | team | heavy | pair | sentry |

其中 `seed=1008` 打基线 `hold`，即第 1 列；其余 11 个种子各对应一份新对手。


### 9.9 运行中夹 std（探索宽度越界的处置）

#### 为什么需要这个通道

`check_action_std`（§9.2 前置检查）只校验**配置里的值**。但 std 是**学出来的参数**，
配置写什么都没用。22h 那轮的实测：

| 读数 | 值 |
| --- | --- |
| `updates.jsonl` 记录数 | 440 |
| 首轮 `losses['train/std']`（step 1024） | **1.969** |
| 末轮（step 448448） | **2.219** |
| 全程范围 | 1.943 – 2.238 |
| 越界轮数（std > 1.0） | **440 / 440（100%）** |

动作空间是 `Box(-1, 1, shape=(2,))`。std=2.2 意味着每次采样的标准差是动作半宽的 2.2 倍，
**约一半采样落在 `[-1,1]` 外被 clip**。梯度因此指向环境从未执行的动作，而 loss 曲线反而
*更好看*——策略已无法区分「被裁剪」和「本来就偏好极端值」。

**这不是初始化问题。** 配置 `ppo` 段没有 `action_std` 键，SB3 `MlpPolicy` 默认
`log_std_init = 0.0` ⇒ 起步 std = 1.0。是**第 1 次更新就把它学到 1.969**，
`ent_coef = 0.01` 的熵奖励压不住。换初值治不了，因为问题在梯度方向。

#### 通道怎么用

与 `learning-rate.json` 同构：两次 rollout 之间读文件、改参数、写事件日志。
代码是 `TimedPPO.apply_action_std_control()`（`learning.py`）。

**判据**：每 30 分钟看 `updates.jsonl` 末行的 `losses['train/std']`，**超过 1.2 就夹一次**。

```bash
RUN=data/research/pooled-131-24h
research/.venv/bin/python -c "
import json
rows=[json.loads(l) for l in open('$RUN/updates.jsonl') if l.strip()]
last=rows[-1]
print('step', last['total_samples'], 'std', round(last['losses']['train/std'],4))
"
```

超了就写（下一次 rollout 边界生效）：

```bash
cat > data/research/pooled-131-24h/action-std.json <<'EOF'
{"max_std": 1.0, "reason": "train/std 越界，夹回动作上界"}
EOF
```

生效后 `action-std-events.jsonl` 会多一行：

```json
{"kind": "action-std-clamp", "total_samples": 51200, "before": 1.9734, "after": 1.0,
 "max_std": 1.0, "reason": "train/std 越界，夹回动作上界", "time_unix": 1795...}
```

`max_std` 必须在 `(0, 1]`；写成 0、1.5、nan、inf 会被拒绝并打一条
`action-std-control-error`，**不会让训练崩掉**（24h 中途不能因为一行坏配置就死）。

#### 为什么是手动而不是自动

自动夹紧会把一个正在学坏的策略硬掰回来，把「梯度在推大探索」这个信号抹掉。
手动通道留下两次干预之间的完整曲线，你能看见它是不是每次夹完又爬回去——
那才是最该知道的事。实现上必须用 `max=math.log(bound)`：`log_std` 是参数，
bound 在 std 空间，写成 `max=bound` 会把 ~0.0 夹到 1.0，**看起来成功但什么都没做**。

#### 什么时候该停

- 夹了 3 次、之后 200 轮仍在 1.0 —— 梯度在对抗夹紧，这个配置下学不出东西；
- 20 万步后 `train/explained_variance` 仍为负且无上升趋势；
- 连续 3 次吞吐低于新测基准的 70%。



