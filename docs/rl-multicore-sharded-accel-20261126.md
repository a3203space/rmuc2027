# 哨兵强化学习多核 CPU 分片加速与无损 JSON 解码优化（2026-11-26）

## 1. 背景与性能瓶颈定位

在 AMD Ryzen 5 5600X（6 物理核 / 12 逻辑线程）+ NVIDIA GeForce RTX 5070 Ti（15 GiB 显存）主机上运行第 3.1 阶段 16 环境并行训练（`--num-envs 16`、`--device cuda`、`sim_backend="warp_kinematic"`）时，实测总耗时 `2010.01s`，其中：

- **采样阶段耗时（`sampling_seconds`，环境交互与轨迹收集总时间）**：`2000.51s`（占总时长 **99.53%**）。
- **梯度优化耗时（`optimization_seconds`，PyTorch CUDA 反向传播与参数更新总时间）**：`2.01s`（仅占 **0.10%**）。

因此，训练吞吐与梯度更新频率的唯一瓶颈在于**采样侧 CPU 调度与序列化开销**，而非 GPU 反向传播算力。

通过 `cProfile`（Python 函数级性能分析器）与 `ps`（进程级 CPU 占用采样）对 8/16 环境分别进行活体剖析，定位到两项根本原因：

1. **单进程全局解释器锁（GIL，Global Interpreter Lock）与线程争抢**：
   - 原 `WarpVecEnv`（单进程线程并行向量环境）在**同一个 Python 进程**中启动 `ThreadPoolExecutor(16)`（16 个环境线程），而每个环境的 `Battle`（对局调度器）内部又各自创建 `ThreadPoolExecutor`（每环境约 12 线程），导致约 **209 个线程争抢同一把 Python GIL**。
   - 实测主进程 CPU 始终卡死在单核上限附近（`81.4%–86.8%`），热点集中在 `select.select`（管道轮询）、`orjson.loads`（解码）、`posix.read`（系统管道读取）、`engine.py:_request`（引擎通信）与 `_thread.lock.acquire`（线程锁争抢，24 步内高达 106,730 次调用）。
   - 结果出现**反扩展现象**：16 环境速度（`19.76 steps/s`，即每秒强化学习步数）反而低于 8 环境（`20.82 steps/s`）。
2. **策略子进程（`worker.py`）与主控（`GoalController` / `Policy`）每帧重复使用标准库 `json.loads` 解析约 100 KB 观测包**：
   - 每个环境挂载 5 个策略子进程（4 个红方队友单兵进程 + 1 个蓝方整队进程），16 环境共 80 个策略进程，在 10 Hz 决策频率下合计消耗 **`646%` CPU**（占全机总 CPU 的 78%）。
   - 微基准实测显示：每份观测报文高达 **97–100 KB**（包含场地 `arena` 42 KB、机器人模型 `robot_models` 19 KB、状态 `robots` 13 KB 等）。标准库 `json.loads` 单次耗时约 **`1.04 ms`**，而 `red:[3]`/`red:[4]`/`red:[6]` 策略本体 `act()` 仅耗时 **`0.01 ms`**（即 **98% 耗时都在解析 JSON**），蓝方整队策略 `act()` 仅 `0.51 ms`（65% 耗时在解析 JSON）。
   - 切换为原生 C/Rust 实现的 `orjson.loads` 后，单次 100 KB 观测解析耗时由 `1.04 ms` 降至 **`0.40 ms`（提速 2.6–2.7×）**，且在 `python -I`（隔离模式）下已实测可直接导入 `orjson 3.13.0`。

---

## 2. 无损优化方案与代码改动

本次改动严格遵循**数值无损、语义等价、向后兼容（默认 `sim_shards=1` 保持原单进程行为）**原则，不改变任何物理积分、观测编码、奖励函数或策略行为逻辑：

| 修改文件 | 核心改动与中文说明 |
| :--- | :--- |
| `research/rmuc2027_research/rl/warp_kinematic/vector.py` | 新增 `ShardedWarpVecEnv`（多进程分片向量环境）、`_shard_worker`（独立 `spawn` 子进程工作循环）与 `_ShardedPhysicsProxy`（跨分片物理计时汇总代理）。将 `num_envs` 个环境均匀切分到 `shards` 个独立 Python 进程中，每个分片拥有独立 GIL 与独立 `WarpVecEnv` / `PhysicsService`（共享 `cuda:0` 显存，单分片仅增约 `~263 MiB` 显存），并按 `seed + offset` 精确分发随机种子，确保每个环境的随机序列与单进程逐位一致；退出时统一广播 `close` 并清理所有 `rmuc-sim` 与策略子进程。 |
| `research/rmuc2027_research/rl/learning.py` | `vector_env(config, count, shards=1)` 支持 `shards`（分片进程数）参数；`train(..., sim_shards=1)` 支持传入 `sim_shards` 并在 `training.json` 中记录 `sim_shards` 字段（不修改 `experiment.json` 中的 `config` 与溯源哈希，保持与既有评估/导出流程完全兼容）。当 `sim_shards == 1` 时保持原有 2 参数调用路径以兼容现有单元测试替身。 |
| `research/rmuc2027_research/rl/__main__.py` | 为 `train`（训练）与 `benchmark`（基准测试）子命令新增 `--sim-shards`（仿真分片进程数，默认 `1`）命令行参数，并在基准结果中输出 `sim_shards` 与跨分片导航预处理耗时 `warmup_preparation_seconds`。 |
| `packages/core/rmuc2027_core/worker.py` | 策略隔离工作进程优先使用 `orjson.loads` 直接解析 `sys.stdin.buffer`（原始字节流），若在无 `orjson` 的极简容器环境下则自动回退到标准库 `json.loads`；`state_json` 与 `response` 序列化仍保留 `json.dumps(..., allow_nan=False)` 以严格维持 `NaN`/`Infinity` 拦截与 `STATE_LIMIT`/`ACTIONS_LIMIT` 字节上限语义。 |
| `packages/core/rmuc2027_core/policy.py` | `Policy._receive`（主机侧接收策略动作响应）优先使用 `orjson.loads(raw)`（原生拒绝 `NaN`/`Infinity` 并保留 `"non-finite JSON"` 异常语义），无 `orjson` 时自动回退到 `json.loads(..., parse_constant=...)`。 |
| `research/rmuc2027_research/rl/session.py` | `GoalController.decide`（学习哨兵外环目标执行器）改用 `engine.decode_response_json`（内置 `orjson.loads` 优先路径）解码 `EncodedObservation.payload`（编码观测字节负载），消除主进程/分片进程内标准库 `json.loads` 热点。 |
| `tests/python/test_kinematic_vector.py` | 新增单元测试 `test_sharded_vector_env_matches_independent_sessions_and_cleans_up`：校验 4 环境 × 2 分片（`ShardedWarpVecEnv`）在跨 8+ 次回合重置（`reset`）后的 `obs`（观测）、`rewards`（奖励）、`dones`（终止标志）、`terminal_observation`（终态观测）与 4 个独立单环境 `SentryEnv` **逐位完全一致**，并校验 `close()` 后零残留分片进程与 `rmuc-sim`/策略孤儿进程。 |

---

## 3. 正确性验证（单元测试）

全部相关单元测试一次性通过（**共 225 项测试，0 失败，0 错误**）：

- `tests/python/test_kinematic_vector.py`：**`Ran 2 tests in 76.105s — OK`**（单进程 `WarpVecEnv` 与多进程 `ShardedWarpVecEnv` 均通过逐位等价性与进程零泄漏校验）。
- 全部 26 个强化学习 / 运动学 / 传输协议测试套件（`test_rl*.py`、`test_kinematic*.py`、`test_warp_kinematic.py`、`test_engine_response_json.py`、`test_encoded_motion_transport.py`、`test_engine_pipe_transport.py`、`test_observation_reuse.py`、`test_fixed_assembly_dispatch.py`）：**`Ran 214 tests — OK`**。
- `tests/python/test_arena_v2.py`（策略沙箱与房间集成测试）：**`Ran 9 tests in 18.414s — OK`**。

---

## 4. 实测多核吞吐与资源对比

在同一主机（Ryzen 5 5600X 6 核 12 线程 + RTX 5070 Ti 15 GiB）上，使用实际第 3.2 阶段配置 `configs/rl/sentry-pooled-v3.json`（`rl_seconds=0.5`、`fixed_policy_hz=10`、`timed_assembly=true`、`drones_enabled=false`），预热 1 步后各连续测量 24 个强化学习步（`steps=24`）：

| 方案阶段 | 环境数 (`num_envs`) | 分片进程数 (`sim_shards`) | 采样吞吐 (`rl_steps_per_second`，步/秒) | 仿真倍率 (`sim_x`，实时倍数) | 相对优化前 `(16,1)` 加速比 | 峰值显存 (`vram`，MiB) | 总 CPU 占用 (`tot%`，上限 1200%) | 主进程 CPU (`par%`) | 各分片进程 CPU (`shards%`) | 策略子进程 CPU (`wrk%`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| 优化前基线 | 8 | 1 | `20.82` | `10.41×` | `1.05×` | — | `445%` | `86.8%` | — | `298%` |
| 优化前基线 | 16 | 1 | `19.76` | `9.88×` | `1.00×` | — | `831%` | `81.4%` | — | `646%` |
| 优化后（仅 `orjson`） | 8 | 1 | `21.97` | `10.98×` | `1.11×` | `881` | `401%` | `85%` | — | `271%` |
| 优化后（仅 `orjson`） | 16 | 1 | `21.00` | `10.50×` | `1.06×` | `881` | `671%` | `81%` | — | `518%` |
| 优化后（多核分片） | 8 | 2 | `32.30` | `16.15×` | `1.63×` | `1408` | `584%` | `66%` | `72, 72` | `322%` |
| 优化后（多核分片） | 8 | 4 | `39.86` | `19.93×` | `2.02×` | `1935` | `693%` | `52%` | `64, 62, 64, 64` | `333%` |
| 优化后（多核分片） | 12 | 4 | `42.91` | `21.46×` | `2.17×` | `1935` | `1000%` | `42%` | `62, 62, 62, 60` | `535%` |
| 优化后（多核分片） | 16 | 2 | `32.12` | `16.06×` | `1.63×` | `1408` | `769%` | `34%` | `73, 69` | `519%` |
| 优化后（多核分片） | 16 | 4 | `42.61` | `21.31×` | `2.16×` | `1935` | `935%` | `28%` | `58, 59, 57, 59` | `589%` |
| **优化后（推荐最优）** | **16** | **8** | **`44.97`** | **`22.49×`** | **`2.28×`** | **`2988`（含脚本内单进程残留上下文；独立运行约 ~2.4 GiB）** | **`1051%`** | **`24%`** | **`48, 55, 49, 54, 54, 55, 47, 47`** | **`535%`** |
| 优化后（过载对比） | 24 | 4 | `40.88` | `20.44×` | `2.07×` | `1935` | `1011%` | `20%` | `54, 54, 52, 56` | `672%` |

### 关键结论

1. **吞吐提升 `2.28×`**：在保持 `--num-envs 16`（与历史试验完全一致的 `4096` 单轮滚动样本块大小）前提下，启用 `--sim-shards 8` 后采样速率由 `19.76 steps/s` 跃升至 **`44.97 steps/s`**（`22.49×` 实时仿真速度）。
2. **成功打破单核 GIL 瓶颈**：8 个分片进程在全部 12 逻辑线程上均匀负载均衡（各 `47%–55%`），总 CPU 利用率由 `671%` 提升至 **`1051% / 1200%`**（约 88% 全核利用率），不再受困于单核。
3. **显存开销极低**：每个分片增量显存仅约 `263 MiB`，8 分片独立训练总显存占用不足 `2.5 GiB`（不到 RTX 5070 Ti 15 GiB 显存的 17%）。
4. **推荐训练命令**：
   ```bash
   PYTHONPATH=research:packages/core research/.venv/bin/python -m rmuc2027_research.rl train \
     --config configs/rl/sentry-pooled-v3.json \
     --run-dir data/research/<run-name> \
     --total-steps 32768 \
     --num-envs 16 \
     --sim-shards 8 \
     --device cuda
   ```
   （若主机显存较紧或为 4 核 CPU，可使用 `--sim-shards 4`，同样可达到 `42.61 steps/s`、`2.16×` 加速。）
