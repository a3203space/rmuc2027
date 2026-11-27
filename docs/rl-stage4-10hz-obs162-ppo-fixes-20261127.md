# Stage-4 哨兵强化学习重构：10 Hz 底层控制 + OBS v4（162 维）+ PPO 因果掩码与采样去偏

日期：2026-11-27  
基线提交：`72beae7`（Stage-3.5 100 次更新 `stage3-145-pooled-100upd`）

---

## 一、改前 Sim-to-Sim（仿真器间迁移）基线诊断

在改动任何代码之前，我们使用未修改的 Stage-3.5 模型 `data/research/stage3-145-pooled-100upd/model.zip`（145 维观测、`learning_control_hz = 2`（底层控制频率 2 Hz）、`rl_seconds = 0.5`（高层 RL 决策周期 0.5 s / 2 Hz）），在**确定性推理（`deterministic=True`，探索标准差 `std = 0`）**下，用完全相同的 8 个种子（`1008, 1010, 1018, 1005, 1026, 1030, 1000, 1002`，红方）分别对比 **Rapier（刚体物理仿真）** 与 **`warp_kinematic`（运动学训练仿真）**：

| 种子 | 对手策略 | Rapier 结果 | `warp_kinematic` 结果 | `warp_kinematic` 控制分（红 vs 蓝） | `warp_kinematic` 阵亡 / `recover` 时长 |
|---:|---|---|---|---|---|
| 1008 | `claude_opus55_xhigh_2027_hold` | 胜 | **胜** | 46.4 vs 0.0 | 0 死 / 78.5 s |
| 1010 | `claude_opus55_xhigh_2027_team` | 胜 | **负** | 83.2 vs 637.6 | 1 死 / 126.0 s |
| 1018 | `dsv41_1017_pair` | 负 | **负** | 5.3 vs 1086.0 | 0 死 / 44.0 s |
| 1005 | `gpt61_2027_pressure` | 胜 | **负** | 1.2 vs 27.5 | 0 死 / 88.0 s |
| 1026 | `gemini38_2027_blitz` | 胜 | **负** | 6.7 vs 81.8 | 0 死 / 63.5 s |
| 1030 | `gemini38_2027_hunter` | 胜 | **负** | 9.9 vs 138.1 | 1 死 / 57.5 s |
| 1000 | `gemini38_2027_tech_overload` | 胜 | **负** | 0.0 vs 67.1 | 1 死 / 63.0 s |
| 1002 | `gpt61_2027_control` | 负 | **胜** | 8.0 vs 7.1 | 1 死 / 45.5 s |
| **合计** | **8 局红方确定性评估** | **6 胜 2 负（75%）** | **2 胜 6 负（25%）** | — | **4 死 / 局均 70.8 s 撤退** |

核心结论：
1. 训练期 Q5（最后 20% 对局）在 6 个强对手上 0 胜并非仅由高噪声探索（`action_std = 0.48`）造成——即便关闭探索噪声，原 100upd 策略在 `warp_kinematic` 中确定性胜率也仅为 **2/8（25%）**，每局耗费 **44–126 s** 在 `recover`（低血/补弹撤退接管）且阵亡 4 次。
2. 底层 2 Hz 控制导致哨兵在 `held_fire_safe`（热量安全预算检查）中被截断在 `160/260` 热量（而 10 Hz 对手可打到 `240/260`），同时反应延迟是对手的 5 倍，交火吃亏后陷入长时间 `recover` 恶性循环。
3. 叠加姿态机缺陷、二次速度衰减爬行、非 `policy` 步污染 PPO Actor（策略网络）梯度、16 个并行环境同步在 420 s 结束引发 `value_loss`（价值损失）尖峰、以及 6 张非中央控制卡对纯 RL 完全不可见，导致训练后期出现伪收敛。

---

## 二、Stage-4 核心改动清单（遵循「底层 10 Hz + 决策 2 Hz + 高层纯 RL」约束）

### Tier 0：底层控制、物理与 PPO 训练闭环修复

1. **控制与决策双频解耦（`configs/rl/sentry-pooled-v3.json`）**
   - `learning_control_hz = 10`（哨兵底层导航、瞄准、开火与姿态控制升回 **10 Hz**，与 10 Hz 固定策略对手对齐，`held_fire_safe` 热量上限自动由 `160/260` 恢复至 `240/260`）。
   - `rl_seconds = 0.5`（高层 RL 目标点发点频率保持 **2 Hz**）。
   - `ppo` 参数收紧与扩容：`n_steps = 512`（16 并行环境单次更新 `8,192` 步，覆盖 1.2 个完整 420 s 赛局）、`batch_size = 256`、`n_epochs = 4`、`ent_coef = 0.002`、`action_std = 0.28`。
2. **修复交火期误切易伤 `move` 姿态（`research/rmuc2027_research/rl/portable.py`）**
   - 重排姿态优先级：`recovering`（撤退中）→ `combat_unsafe_attack`（低血/重甲近身/受围攻）→ `combat_ready`（持弹可射）→ `traverse_bumps and safe_for_move`（无敌情过减速带）→ `moving and gap > MOVE_POSTURE_DISTANCE_M and safe_for_move`（安全长距机动）→ `defense`（防御姿态）。
   - 严禁在 `threatened`（近 2 s 受击）、`live_duel`（10 m 内活体交火）、`combat_unsafe_attack` 或 `recovering` 遇敌/低血时切换至承受 `+25%` 易伤且锁死 5 s 的 `move` 姿态。
3. **消除近点二次衰减爬行（`research/rmuc2027_research/rl/portable.py`）**
   - `sdk._physical_toward` 内部已做线性减速，移除 `execute_goal` 的二次乘法；在 `gap > GOAL_STOP_RADIUS_M` 且非减速带原地转向（`mag > 0.05`）、非受限机动时将速度幅值下限钳位为 `max(0.25, min(mag, cap_mag))`，同时保证 `cmd_mag > 0.20` 触发超电（`power = 200`）辅助到位，消除持续 10 s 的 `move ≈ 0.038` 微速爬行。
4. **`recover` 目标锁定与 `goal_age`（目标保持时长）修复（`research/rmuc2027_research/rl/portable.py`）**
   - `select_goal` 将目标锁定范围由 `('dead', 'tipped')` 扩展至 `('dead', 'tipped', 'recover')`，并记录 `state['action_accepted'] = not locked`。
   - 仅当目标点真实变化（或首帧初始化）时更新 `goal_time`，`goal_age` 特征改为除以 `15.0` 秒，避免原先恒为 `1.0` 的死特征。
5. **PPO 因果 Actor 损失掩码（`session.py` + `learning.py`）**
   - `ArenaSession._step` 统计 50 ms 子步中的 `policy_ticks`，仅当 `action_accepted` 为真且 `policy_ticks > 0` 时输出 `info['policy_executed'] = True`（保证“在 `policy` 中驶入敌火并于步内阵亡”仍被追责，而整步处于 `dead`/`tipped`/`recover` 或步初被锁死的采样不误导策略梯度）。
   - 新增 `MaskedRolloutBuffer`（带 `policy_mask` 字段的经验缓冲区）、`_PolicyMaskCallback` 与 `TimedPPO._train_masked`：Actor 的优势归一化、`policy_loss`、`entropy_loss`、`clip_fraction` 与 `approx_kl` 仅在 `policy_mask == 1.0` 上计算，Critic 的 `value_loss` 仍在全量样本上训练，并记录 `train/policy_mask_fraction`。
6. **平滑加权对手轮换 + 分层洗牌袋（`session.py` + `env.py`）**
   - `opponent_schedule` 使用平滑加权轮询（Smooth Weighted Round-Robin）并旋转锚定基线对手至 0 号槽，彻底消除连续种子扎堆同一对手。
   - `SentryEnv.reset(seed=None)` 使用每个环境独立的分层洗牌袋（`base * period + slot`），保证长训 Q1–Q5 各对手比例严格恒定且完全可由 `episodes.jsonl` 种子复现。
7. **错峰首局截断（`learning.py` + `session.py`）**
   - `train()` 在多环境非英雄槽位注入 `_stagger_initial_horizons`（由 `vector_env` 拆分为各环境均匀分布的 `_initial_episode_seconds` 并在 `ArenaSession.__init__` 中弹出，不污染 `experiment.json` 与回放 `provenance`），将 16 个环境的 420 s 终局 `±10` 信号永久错峰分散至每次 PPO 更新，消除周期性 `value_loss` 爆炸。
8. **`--init-from`、非 Rapier `evaluate` 与尾帧序列化修复（`learning.py` + `exporting.py`）**
   - `--init-from` 通过 `custom_objects`（含 `FloatSchedule`）继承新配置 `ppo` 超参；若奖励配置相对源实验变化，则自动重置 Critic 权重与 Adam 优化器状态。
   - `evaluate()` 仅在 `sim_backend == 'rapier'` 时调用 `verify_replay`，修复 `warp_kinematic` 评估崩溃。
   - `TraceCallback._on_training_end` 补齐 `default=_json_default`，修复训练恰好落在回合结束步时序列化 `terminal_observation`（`ndarray`）报错的潜在缺陷。
   - `exporting.py` 为 `goal_tick` 补齐 tick 回绕保护。

### Tier 1：纯 RL 公开观测升级至 OBS v4（145 维 → 162 维）

所有新增特征严格来源于公开观测（零读取 `obs['referee']`），且满足红/蓝 180° 镜像严格对称：

- **修复死特征 #7**：优先读取 `me['buffer_energy_j'] / 60.0`（2027 赛季缓冲能量，修复原先读到恒为 0 的旧 `energy` 字段）。
- **修复死特征 #130（`goal_age`）**：`max(0, time - goal_time) / 15.0`。
- **新增 +17 维公开对称特征（索引 `145..161`）**：
  - `145`：自身超级电容比例 `supercap_energy_j / 1800.0`；
  - `146`：姿态切换冷却剩余比例 `max(0, posture_ready_at - time) / 5.0`；
  - `147..158`：其余 6 张控制卡（己方/敌方 `outpost` 前哨、`base` 基地、`fortress` 堡垒）的 `(队友占领, 敌方占领)` 公开 RFID 状态（共 12 维），使纯 RL 能够直接观测敌方偷点偷家并自主学习回防与反夺；
  - `159`：公开伤害差 `(own_attack_damage - enemy_attack_damage) / 4000.0`（平局第一判据）；
  - `160`：公开控制分差 `(own_control - enemy_control) / control_scale`；
  - `161`：己方存活地面机器人总血量比例 `own_ground_hp / 1800.0`。

---

## 三、验证与基准结果

1. **10 Hz 底层控制吞吐基准（`16 envs / 8 shards`，`uniform-random`，128 steps/env）**：
   - 采样速度：**`51.44 RL steps/s`**（`25.72 sim_s/s`，`39.81 s` 完成 `2,048` RL 步），较 Stage-3.5 的 2 Hz 基准（`39.02 RL steps/s`）**提升 +31.8%**。
2. **独立交叉代码审查（`gemini-4-hydrogen (high)`）**：
   - 阻断项（Blocking）：**0**。
   - 采纳的建议项：在 `select_goal` 增加 `action_accepted` 收紧跨步复活/撤退边界的 `policy_executed` 因果判定；`recovering` 姿态增加被围攻/中血交火防易伤护栏；`opponent_schedule` 锚定 `config['opponent']`；`get_schedule_fn` 升级为 `FloatSchedule`；`exporting.py` 补充 `goal_tick` 回绕检查。
3. **自动化测试**：21 个核心 RL 与运动学单元测试套件全部通过（含 OBS v4 162 维红蓝对称性、Actor 因果掩码与 CUDA 确定性、错峰首局截断、分层对手袋与 `--init-from` 回归测试）。
