# 第 3.2 阶段哨兵 RL 双滞回、减速带航向锁、生存姿态与探索优化总结（2026-11-26）

本文件记录针对第 3.1 阶段 8 轮重训与 `Rapier`（高精度刚体物理引擎）评测残留不足（[docs/rl-stage3-1-bumps-recover-fix-20261126.md](file:///media/rm/E3E258F427E8E547/rmuc2027/docs/rl-stage3-1-bumps-recover-fix-20261126.md)）的第 3.2 阶段修复改动、全量单元测试回归结果与待跑的 `v32` 验收计划。所有英文术语、配置键、命令参数及代码符号均附中文解释。

---

## 1. 修复内容与代码改动明细

本阶段采用 **修复项 1 + 修复项 2 + 修复项 3 + 修复项 4（A1 + B1）**，改动位于 [research/rmuc2027_research/rl/portable.py](file:///media/rm/E3E258F427E8E547/rmuc2027/research/rmuc2027_research/rl/portable.py)、[configs/rl/sentry-pooled-v3.json](file:///media/rm/E3E258F427E8E547/rmuc2027/configs/rl/sentry-pooled-v3.json) 与 [tests/python/test_rl.py](file:///media/rm/E3E258F427E8E547/rmuc2027/tests/python/test_rl.py)：

### 1.1 修复项 1（P0）：双滞回锁目标 + 阵亡/侧翻冻结，消除中场跨障跳变
- **新增常量 `PROPOSED_HYSTERESIS_M = 1.0`**（原始连续建议点滞回半径）：保留原有网格目标滞回半径 `GOAL_HYSTERESIS_M = 1.15`（投影后合法网格点变动 `<= 1.15m` 保持锁定）的同时，在 `state` 中持久化锁定上一目标对应的原始连续建议点 `_goal_proposed`。
- **更新 `select_goal`**：
  - 当机器人处于 `dead`（阵亡）或 `tipped`（侧翻）模式且折叠状态未变（`same_fold`）时，直接冻结延续上一合法目标 `prev_goal` 与 `_goal_proposed`，不再调用 `navigator.resolve_goal` 产生无效切换。
  - 在有效状态下，只要满足以下任意一条（且未切换折叠态 `same_fold`）：
    1. 投影候选点变化 `<= GOAL_HYSTERESIS_M`（`1.15m`），**或**
    2. 神经网络原始连续输出点 `proposed` 相对 `_goal_proposed` 漂移 `<= PROPOSED_HYSTERESIS_M`（`1.0m`），
    即继续锁定原 `prev_goal`，从而彻底阻断“策略输出仅微移 `0.4m`，但跨过中央高地底座障碍中线后被吸附到 `2m+` 对侧”的跨障跳变；并移除原有的 `mode == 'policy'` 门槛，使 `recover`（回城恢复模式）期间同样受益于防抖滞回。

### 1.2 修复项 2（P0）：减速带全程锁航向、抑制横移并按主轴判定卡滞
- **新增 `_bump_axis_heading(robot, bump_zone)`**：根据减速带波纹主轴 `axis`（`'x'` 或 `'y'`）计算距机器人当前朝向最近的平行/反向轴线航向角（例如 `x` 轴减速带锁定 `0.0` 或 `math.pi`）。
- **改进 `_unstick_fields`（卡滞脱困状态机）**：
  - 位于减速带（`on_bumps is not None`）时，改用**沿波纹主轴的纯一维位移**（`abs(x - x0)` 或 `abs(y - y0)`）判定是否推进停滞，防止车体沿波谷横向侧滑 `y` 欺骗二维欧氏位移阈值，且不因车头扭转 `turned > 0.15` 而跳过脱困触发。
  - 将减速带倒车脱离时长从 `0.9s` 延长至 `1.1s`，并将过带助跑功率增强窗 `bump_boost_until` 从 `3.5s` 延长至 `3.8s`，统一调用 `_bump_axis_heading` 锁定倒退航向。
- **过减速带全程锁航向 + 横向分量限幅**：在 `execute_goal` 中，处于减速带内（`on_bumps`）或脱困后助跑窗内靠近减速带（`approaching_bumps if bump_boost`）的非机动步上，全程下发 `fields['heading'] = _bump_axis_heading(...)`；且处于 `on_bumps` 上且速度指令非零（`orig_mag > 0.2`）时，将横向驱动分量裁剪至主轴分量的 `±0.15` 并维持总推力幅值，避免像 `seed=1013` `t=344→377s` 那样斜身冲坡后横向扭转并来回拉扯 32 秒。

### 1.3 修复项 3（P1）：提早切防保命、提高撤退血线并限制残弹切进攻
- **新增常量 `ATTACK_ENTRY_MIN_AMMO = 25`（切入进攻姿态最低备弹量）**：仅当当前本就在 `'attack'` 或余弹 `ammo >= 25` 时才允许 `combat_ready` 触发进入 `'attack'`（进攻姿态，附带 `+25%` 易伤），避免剩余十几发弹药时刚切入 `attack` 就瞬间打光、随后白扛 5 秒 `posture_ready_at` 姿态冷却窗口内的高易伤惩罚。
- **收紧生存防御阈值**：
  - `policy`（主控）模式下受击切防御阈值由 `hp < 0.55 and threatened` 提前至 **`hp < 0.65 and threatened`**（血量低于 65% 且 2 秒内受击即切 `defense` 50% 减伤）。
  - 强化防御（`posture_enhanced`，99% 减伤）触发血线由 `hp < 0.50` 放宽至 **`hp < 0.55 and threatened`**。
  - `recovering`（回城恢复）模式下（非过减速带时）只要 **`hp < 0.40`**（无需再要求 2 秒内刚受击）或距补给区 `gap <= 3.0m`，即切并保持 `defense` 防止残血走位被秒杀。
- **提高哨兵回城撤退血线**：将 [configs/rl/sentry-pooled-v3.json](file:///media/rm/E3E258F427E8E547/rmuc2027/configs/rl/sentry-pooled-v3.json) 中的 `recover_enter`（触发回城血量比例）从 `0.25`（`100 HP`）提升至 **`0.35`（`140 HP`）**，留足撤退容错空间。

### 1.4 修复项 4（P2，已采纳选项 A1 + B1）：温和扩宽探索与轻度阵亡惩罚
- [configs/rl/sentry-pooled-v3.json](file:///media/rm/E3E258F427E8E547/rmuc2027/configs/rl/sentry-pooled-v3.json)：
  - `ppo.action_std`（策略高斯探索标准差）：从 `0.35` 调整至 **`0.48`**（把单步随机探索覆盖半径从 `±4.9m` 扩至约 `±6.7m`，兼顾稳定与覆盖己方高地/前哨/侧翼）。
  - `rewards.death`（每次阵亡奖励惩罚）：从 `0.0` 调整至 **`-0.5`**（仅为胜负终局奖 `±10.0` 的 `5%`，引导策略避开中场交叉火力站桩送头）。

---

## 2. 单元测试回归验证结果

在 [tests/python/test_rl.py](file:///media/rm/E3E258F427E8E547/rmuc2027/tests/python/test_rl.py) 中新增与更新了对应回归用例：
- `test_goal_hysteresis_and_arrival_resume_deadband`：新增中央高地跨障跳变（`[0.0, 0.0] → [0.055, 0.055]` 无约束吸附 `> 1.15m` 但原始点漂移 `<= 1.0m` 时仍锁定原目标）与 `dead`（阵亡）模式冻结目标断言；
- `test_bumps_lock_heading_dampen_lateral_and_detect_axis_stall`：验证减速带内锁定轴向航向 `math.pi`、抑制横向分量 `<= 0.16 * |mv_x|`，以及纵轴侧滑 + 车头扭动但主轴停滞时 `2 Hz` 精准触发向东倒车脱困并给 `power = 200`；
- `test_survival_posture_thresholds_and_low_ammo_attack_entry_guard`：验证非减速带低血未近期受击回城切 `defense`、`hp = 0.60` 受击切 `defense`、及 `ammo = 15 < 25` 时禁止从 `defense` 切入 `attack`（`ammo = 30` 时允许）；
- `test_v3_config_matches_requested_settings`：同步校验 `recover_enter = 0.35`、`rewards.death = -0.5`、`ppo.action_std = 0.48`。

运行全量 27 组 Python RL 与运动学单元测试（排除仓库预存失败的 `test_rl_drones.py`）：
```bash
FILES=$(ls tests/python/test_*rl*.py tests/python/test_*kinematic*.py | sort -u | grep -v 'test_rl_drones.py')
PYTHONPATH=research:packages/core:tests/python research/.venv/bin/python -m unittest $FILES
```
实测输出：**`Ran 196 tests in 121.468s — OK`**（`196` 项全部通过，零失败、零报错）。

---

## 3. 待运行：第 3.2 阶段 8 轮重训与 `Rapier` 四对手实测对比（`v32`）

- 训练配置：基于最新 [configs/rl/sentry-pooled-v3.json](file:///media/rm/E3E258F427E8E547/rmuc2027/configs/rl/sentry-pooled-v3.json)（关闭 GitHub 备份，写入 `data/research/stage3-145-pooled-v32`，`32,768` 步 / `8` 次更新）。
- 待对比指标：`v3.0` vs `v3.1` vs `v3.2` 的 `jmp>0.3` 跨障跳变次数、`1013` 减速带通过耗时、全图死卡、阵亡率（`deaths`）、`ammo0_att`、`recover` 占比与 `Rapier` 控分战绩。
