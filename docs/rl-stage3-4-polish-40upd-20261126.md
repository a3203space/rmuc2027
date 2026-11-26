# 第 3.4 阶段哨兵 RL 收尾微修与 40 轮中程多核训练总结（2026-11-26）

本文件记录针对第 3.3 阶段残余 2 处 5 行级边界瑕疵（见 [docs/rl-stage3-3-survival-plinth-fix-20261126.md](file:///media/rm/E3E258F427E8E547/rmuc2027/docs/rl-stage3-3-survival-plinth-fix-20261126.md) 第 4 节）的收尾微修、全量单元测试回归结果，以及基于 8 分片多核并行（`--sim-shards 8`）的 **40 轮中程训练（163,840 步）**与 `Rapier`（高精度刚体物理引擎）四对手验收计划/结果。所有英文术语、配置键及代码符号均附中文说明。

---

## 1. 两处执行器收尾微修明细

改动文件：[research/rmuc2027_research/rl/portable.py](file:///media/rm/E3E258F427E8E547/rmuc2027/research/rmuc2027_research/rl/portable.py) 与 [tests/python/test_rl.py](file:///media/rm/E3E258F427E8E547/rmuc2027/tests/python/test_rl.py)：

1. **收敛 `traverse_bumps`（过减速带移动姿态强制）并无条件化低血保底 `bump_emergency_def`**：
   - 将 `traverse_bumps` 中的非减速带 `in_unstick`（普通卡滞脱困态）移除，仅保留 `on_bumps is not None or approaching_bumps is not None or bump_boost`（真实减速带及其助跑窗），杜绝在非减速带区域（如 `seed=1013` 阵亡 #3 的 `(17.84, 8.05)`）因脱困误触 `move` 弃防。
   - 将 `bump_emergency_def`（减速带紧急保命防御覆写）由 `hp < 0.35 and threatened` 改为无条件 **`hp < 0.35`**，防止 35% 血量以下因强化防御刚结束、2 秒受击窗 `threatened` 短暂归零而被切换回 0% 减伤的 `move` 姿态（根治 `seed=1013` 阵亡 #1）。
2. **新增 `HEAVY_MEMORY_S = 3.0`（近距敌方重型英雄 3 秒视野防抖记忆）**：
   - 在 `state['last_heavy_near_time']` 记录最近一次观测到 8m 内敌方重型英雄（`role == 'heavy'`）的时间戳（并在阵亡复活时自动清空），在 3.0 秒记忆窗内持续维持 `heavy_near = True`，防止敌方英雄因单帧墙角视野遮挡短暂离开 `obs['enemies']` 导致哨兵瞬间误切 `+25%` 易伤的 `attack` 姿态遭 42mm 炮弹秒杀（根治 `seed=1022` 阵亡 #2）。

---

## 2. 单元测试回归验证

- 在 [tests/python/test_rl.py](file:///media/rm/E3E258F427E8E547/rmuc2027/tests/python/test_rl.py) 的 `test_survival_posture_thresholds_and_low_ammo_attack_entry_guard` 中补充以下断言：
  - 看到敌方重型英雄后将其移出视野，`+2.0s` 内仍拦截 `attack`，`+4.0s` 记忆过期后恢复正常步兵对枪切 `attack`；
  - 减速带上 `hp = 0.18` 且无近期受击时无条件保持 `defense`，受击时叠加 `posture_enhanced`；
  - 非减速带 `(17.8, 8.0)` 处于 `unstick` 脱困态且 `hp = 0.10` 时绝不切回 `move`。
- 全量 27 组 Python RL 与运动学测试套件回归实测：**`Ran 197 tests in 167.781s — OK`**（197 项全部通过，0 失败，0 错误）。

---

## 3. 待运行：40 轮中程多核训练与 `Rapier` 四对手验收（`stage3-145-pooled-40upd`）

- 训练配置：基于 [configs/rl/sentry-pooled-v3.json](file:///media/rm/E3E258F427E8E547/rmuc2027/configs/rl/sentry-pooled-v3.json)（关闭 GitHub 远程备份），`--total-steps 163840`（`40` 轮 PPO 更新 × `4096` 步/轮，约 `160` 场完整 420 秒多对手池对抗）、`--num-envs 16 --sim-shards 8 --device cuda`，输出至 `data/research/stage3-145-pooled-40upd`。
- 验收流程：训练完成后导出便携策略 `exported-40upd`（核验 `actor_parity_max_error < 1e-5`），并运行 420 秒确定性 `Rapier` 四对手验收（种子 `1008 / 1009 / 1013 / 1022`，红方视角，输出至 `eval-rapier-40upd`），与 `v3.3`（8 轮）全面对比训练收敛曲线、站位迁移与对阵 `dsv41_1017_pair`（`1013`）的突破情况。
