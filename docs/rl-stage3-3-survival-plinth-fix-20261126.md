# 第 3.3 阶段哨兵 RL 紧急撤退、重型英雄防秒、跨障大滞回与备弹门槛优化总结（2026-11-26）

本文件记录针对第 3.2 阶段逐帧取证发现的 4 项残留根因（见 [docs/rl-stage3-2-hysteresis-survival-fix-20261126.md](file:///media/rm/E3E258F427E8E547/rmuc2027/docs/rl-stage3-2-hysteresis-survival-fix-20261126.md) 第 4 节）实施的第 3.3 阶段代码修复、197 项单元测试全量回归结果与路径 A 8 轮快速验证计划。所有英文术语、配置键及代码符号均附中文说明。

---

## 1. 修复内容与代码改动明细

本阶段所有逻辑改动集中于便携执行器 [research/rmuc2027_research/rl/portable.py](file:///media/rm/E3E258F427E8E547/rmuc2027/research/rmuc2027_research/rl/portable.py)（训练与导出共用的单源运行时）与单元测试 [tests/python/test_rl.py](file:///media/rm/E3E258F427E8E547/rmuc2027/tests/python/test_rl.py)：

### 1.1 修复项 1A（P0）：解除 `can_retreat`（撤退许可）对极低血/空弹哨兵的死锁
- **新增常量 `EMERGENCY_RECOVER_HP = 0.25`**（紧急无条件撤退血量比例，即 `100 / 400 HP`）。
- **更新 `execute_goal` 中的 `emergency_retreat` 判定**：
  - 当哨兵**不在中央得分区 `central` 1.2m 内**（`not holding_central`）**或** `defense` 强化防御 15 秒预算已耗尽（`not def_enh_available`）且血量跌破 `0.25` 时；**或者**血量低于 `recover_enter`（`0.35`）且已空弹（`ammo == 0`）时，直接置 `emergency_retreat = True`，绕过 `can_retreat` 立即进入 `recover`（回城恢复）模式。
  - 同时严格保留“当哨兵正稳守 `central` 且仍拥有 99% 减伤强化防御预算与弹药时，前哨站降级期间继续死守中心点”的战术语义（与原有回归用例 `test_uncovered_temporary_outpost_holds_ground_with_enhanced_defense` 完全兼容）。

### 1.2 修复项 1B 与 1C（P0）：封堵残血/重型英雄近距/贴脸多敌切入 `attack` 与减速带残血零减伤暴毙
- **新增生存阈值常量**：
  - `LOW_HP_NO_ATTACK = 0.45`（无条件低血禁攻下限：低于 45% 血量时，无论 2 秒受击窗口 `threatened` 是否清空，一律禁止切入附带 `+25%` 易伤且锁定 5 秒的 `attack` 进攻姿态）；
  - `MULTI_ENEMY_NO_ATTACK_HP = 0.55`（贴脸多敌禁攻血线：当 6m 内存在 `>= 2` 名存活且非无敌的非空中敌方单位时，血量 `<= 55%` 保持 `defense` 防御姿态）；
  - `HEAVY_THREAT_RADIUS_M = 8.0`（重型英雄近距防秒半径：8m 内存在存活且非无敌的敌方重型英雄 `role == 'heavy'` 时，即使满血也禁止切入 `attack`，避免在 5 秒姿态锁定期内吃两发 42mm 大弹丸各 `-200 HP` 直接被秒杀，改为在 `defense` 50% 减伤下继续保持 17mm 反击火力）。
- **减速带紧急防御覆写（`bump_emergency_def`）**：
  - 将 `if traverse_bumps:` 调整为 `if traverse_bumps and not (hp < 0.35 and threatened):`；且在 `recovering`（回城）分支增加 `or threatened` 保护（回城途中只要遭追击受击即切 `defense`），根治 `seed=1022` 阵亡 #2 中在减速带被敌方哨兵 1.6m 贴脸追杀却被迫全程保持 0% 减伤 `move` 姿态致死的缺陷。

### 1.3 修复项 2（P0）：跨障碍大跳变启用更宽的原始建议点滞回半径
- **新增常量 `OBSTACLE_SNAP_JUMP_M = 2.0` 与 `OBSTACLE_SNAP_PROPOSED_M = 1.8`**：
  - 在 `select_goal` 中，若导航器投影后的候选目标相对上一目标跳变超过 `2.0m`（典型跨中央高地底座翻边吸附），自动将原始建议点滞回门槛由普通场景的 `PROPOSED_HYSTERESIS_M = 1.0m` 放宽至 **`1.8m`**；普通非跨障移动仍保持 `1.0m` 灵敏响应，从而阻断中场底座左右两侧（如 `[13.25, 7.0] ↔ [16.0, 7.75]`）的渐进漂移反复横跳。

### 1.4 修复项 3（P1）：提高切入进攻姿态最低备弹量 `ATTACK_ENTRY_MIN_AMMO`
- 将 `ATTACK_ENTRY_MIN_AMMO` 从 `25` 提升至 **`45`**（匹配 `40` 发远程购弹阈值），避免在 20 Hz 射速下 1.25 秒即打空弹匣并在剩余 3–4 秒 `posture_ready_at` 冷却期内白扛 `+25%` 易伤惩罚。

---

## 2. 单元测试全量回归结果

在 [tests/python/test_rl.py](file:///media/rm/E3E258F427E8E547/rmuc2027/tests/python/test_rl.py) 中同步更新并扩充了对应回归断言：
- `test_goal_hysteresis_and_arrival_resume_deadband`：验证跨中央高地底座跳变 `> 2.0m` 时，原始建议点漂移 `<= 1.8m`（`[0.0, 0.0] → [0.095, 0.06]`）仍锁定原目标，超出 `1.8m` 才切换；
- `test_survival_posture_thresholds_and_low_ammo_attack_entry_guard`：验证 `ammo = 40 < 45` 拒绝切入 `attack`、`ammo = 50` 允许切入；验证 `hp = 0.40` 无近期受击仍禁止切 `attack`；验证 8m 内存在敌方重型英雄（`blue-1`）时即使满血也拒绝切入 `attack`；验证前哨站降级且不在中心点时 `hp = 0.20` 与 `hp = 0.31 & ammo = 0` 触发紧急回城（`hp = 0.31 & ammo = 100` 保持 `policy`）；验证减速带上 `hp = 0.18` 受击覆写 `move` 进入 `defense + posture_enhanced`。

运行全部 27 组 Python RL 与运动学测试套件（含多核分片 `ShardedWarpVecEnv` 逐位等价测试，排除预存坏例 `test_rl_drones.py`）：
- 实测结果：**`Ran 197 tests in 168.216s — OK`**（197 项全部通过，0 失败，0 错误）。

---

## 3. 待运行：路径 A — 第 3.3 阶段 8 轮快速验证与 `Rapier` 四对手验收（`v33`）

- 训练命令：使用 8 分片多核并行模式（`--num-envs 16 --sim-shards 8 --device cuda`，`total_steps = 32768`，输出至 `data/research/stage3-145-pooled-v33`）；
- 验收指标：重点验证 `v3.2 → v3.3` 在 `1009 / 1022` 的阵亡骤降效果（目标将 7 次阵亡大幅压低）、`1008–1022` 中场跨障大跳变 `jmp>1.0m` 降幅、`ammo0_att` 缩短情况与 `Rapier` 控分战绩，验收通过后即可转入 40–100 轮中长程训练。
