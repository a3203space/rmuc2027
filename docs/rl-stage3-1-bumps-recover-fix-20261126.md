# 第 3.1 阶段哨兵 RL 卡滞脱困、回城规则与目标滞回修复总结（2026-11-26）

本文件记录针对第 3 阶段 8 轮诊断基线（[docs/rl-stage3-obs145-baseline-20261126.md](file:///media/rm/E3E258F427E8E547/rmuc2027/docs/rl-stage3-obs145-baseline-20261126.md)）所暴露问题的第 3.1 阶段代码修复、对应单元测试回归结果，以及待补充的 8 轮重训与 `Rapier`（高精度刚体物理引擎）4 对手复测对照。所有英文技术名词、配置键与符号均附中文解释。

---

## 1. 修复内容与代码改动明细

本阶段采用 **方案 1 + 方案 2 + 方案 3 + 方案 4 选项 A**（核心修复集中在 [research/rmuc2027_research/rl/portable.py](file:///media/rm/E3E258F427E8E547/rmuc2027/research/rmuc2027_research/rl/portable.py) 与 [tests/python/test_rl.py](file:///media/rm/E3E258F427E8E547/rmuc2027/tests/python/test_rl.py)，暂保持 [configs/rl/sentry-pooled-v3.json](file:///media/rm/E3E258F427E8E547/rmuc2027/configs/rl/sentry-pooled-v3.json) 中的探索标准差 `ppo.action_std = 0.35` 与阵亡惩罚 `rewards.death = 0.0` 不变，优先彻底验证卡死与过度回城是否消除）：

### 1.1 方案 1（P0）：根治 `red_road_bumps` 减速带卡死与姿态误降功
- **新增 `_bump_zone_for(robot, arena, margin_x=0.0, margin_y=0.0)`**：识别 `arena['terrain']` 中所有 `kind == 'bumps'` 的减速带区域（支持横向/纵向扩展裕度 `margin_x` / `margin_y` 提前检测入坡）。
- **新增 `_unstick_fields(robot, fields, goal, obs, state, on_bumps)`（卡滞脱困状态机）**：
  - 使用状态键 `state['unstick']` 追踪近期位置轨迹 `trail`（窗口保留时长设为 `1.75s`，兼容 `2 Hz` 决策下 `1.5s` 浮点边界不误删首条采样）。
  - 当机器人持续收到行进指令但连续 `>= 1.35s` 实际位移 `<= 0.12m` 且朝向变化 `<= 0.15 rad` 时触发脱困：
    - **位于减速带上（`on_bumps is not None`）**：按减速带波纹轴向 `axis`（`'x'` 或 `'y'`）向来向反向退出波谷、同时朝中线微调横向偏移並锁定航向角 `hold_heading` 持续 `0.9s`，同时开启 `3.5s` 的冲坡辅助窗 `bump_boost_until`（脱出后二次高速冲坡）。
    - **位于普通平地/墙体障碍（`on_bumps is None`）**：沿目标反方向偏转 `±0.9 rad` 后退 `0.75s` 脱离卡滞。
- **超级电容 `power = 200`（`200W`）推力爆发**：在 `execute_goal` 中检测未受避障限速（`not nav_state.get('limited')`）且无跳跃/变形/刹车指令时，若低速卡滞（`speed < 0.35`）、处于脱困期间（`in_unstick`）或正在接近/冲刺减速带（`approaching_bumps` / `bump_boost` 且 `speed < 1.8`）且超级电容余量充足（`supercap_energy_j > 150.0` 或观测未含该字段），下发 `fields['power'] = 200`。
- **修复过坡与回城赶路姿态掉入 `50W` `defense`（防御姿态）的问题**：
  - 新增 `traverse_bumps` 判定：只要正在减速带上或接近减速带/处于减速带脱困助跑窗且尚未到点停稳，强制选择 `wanted = 'move'`（移动姿态，`150W/120W` 高功率输出），防止 `_square_to_bumps` 入坡对正瞬间的微小 `move` 向量把哨兵切成 `defense`/`attack`（`power_ratio = 0.5` 仅 `50W`）。
  - 在 `recovering`（恢复模式）下，仅当 **`hp < 0.35 and threatened`**（低血且近期受击）或 **`gap <= 3.0m`**（已接近补给区）时切 `defense`；长途返程赶路一律保持 `move` 姿态。

### 1.2 方案 2（P0）：`free_grant_due` 远程购弹优先 + 中途滞回防折返
- 新增常量：
  - `REMOTE_BUY_MIN_COINS = 200.0`（远程购弹所需最低队伍金币：`150` 元购弹费 + `50` 元保留金 `reserve`）
  - `TEAM_17MM_BUY_LIMIT = 900`（触发优先回城领免费弹前的团队 17mm 购弹接近上限阈值）
  - `FREE_GRANT_NEAR_M = 8.0`（距己方补给区 `<= 8.0m` 时才顺路回城领免费弹）
  - `RECOVER_COMMIT_M = 8.0`（回城承诺半径：已退到距补给区 `<= 8.0m` 后不再因 `can_retreat` 波动中途折返）
- 重构 `free_grant_due`：仅当存在到期未领免费弹（`due_grants > taken_grants`）、弹药为 0 且无在途购弹时，**必须在买不起/达团队购弹上限（`not can_afford_remote`）或本就离补给区很近（`dist_to_supply <= FREE_GRANT_NEAR_M`）时**才触发回城；其余情况保留在 `policy` 模式由 `baseline.ammo_fields` 在脱战后直接远程买弹。
- 取消 `mode == 'recover' and gap > 1.0` 对 `baseline.ammo_fields` 的一刀切屏蔽，改为仅在满足待领免费弹且距补给区 `<= 4.5m`（`skip_ammo_buy`）时才跳过买弹、专心进区白嫖。

### 1.3 方案 3（P1）：增大目标滞回阈值 `GOAL_HYSTERESIS_M`
- 将 `GOAL_HYSTERESIS_M`（策略建议目标点防抖滞回半径）由 `0.55m` 提升至 **`1.15m`**（显著大于 `0.25m` 离散网格的 `(2, 1)` 格斜距 `0.559m` 与 `(2, 2)` 格斜距 `0.707m`），同时维持 `GOAL_STOP_RADIUS_M = 0.30m`（到点停稳半径）与 `GOAL_RESUME_RADIUS_M = 0.48m`（停稳后重启导航滞回半径）。

### 1.4 方案 4（选项 A，保持配置不变）
- `ppo.action_std = 0.35`（策略初始/边界探索标准差）与 `rewards.death = 0.0`（阵亡惩罚权重）维持原值，以隔离前 3 项物理与状态机修复的纯收益。

---

## 2. 单元测试回归验证结果

更新 [tests/python/test_rl.py](file:///media/rm/E3E258F427E8E547/rmuc2027/tests/python/test_rl.py)：
- 在 `test_goal_hysteresis_and_arrival_resume_deadband` 新增开阔区 `(5.0, 5.0) → (5.75, 5.5)`（间距 `0.901m`，介于旧阈值 `0.55m` 与新阈值 `1.15m` 之间）锁定原目标断言；
- 更新 `test_structure_los_blocks_wall_fire_and_zero_ammo_avoids_attack_and_grants_retreat` 以校验长途返程领弹采用 `move` 姿态而非 `defense`；
- 新增 `test_bump_approach_keeps_move_posture_and_unstick_escapes_at_2hz`（入坡保持 `move` + 波谷卡滞 `2 Hz` 触发反向脱困与 `power = 200`）；
- 新增 `test_free_grant_prefers_remote_buy_far_from_supply_and_commits_halfway_retreat`（远端有钱优先远程购弹、触上限回城、`<= 8m` 半程防折返与低血受击切强化 `defense`）。

运行命令（排除仓库中因未改动文件 `sdk.py#L419` `KeyError: 'role'` 预存失败的 `test_rl_drones.py`）：
```bash
FILES=$(ls tests/python/test_*rl*.py tests/python/test_*kinematic*.py | sort -u | grep -v 'test_rl_drones.py')
PYTHONPATH=research:packages/core:tests/python research/.venv/bin/python -m unittest $FILES
```
实测结果：**`Ran 194 tests in 118.895s — OK`**（全部 `194` 项 Python RL 与 `warp_kinematic` CUDA 运动学单元测试零失败、零报错通过）。

---

## 3. 待补充：8 轮重训与 `Rapier` 四对手实测对比（下一步运行）

- 验证训练输出目录：`data/research/stage3-145-pooled-v31`（`32,768` 步、`8` 次 PPO 更新）
- 验证对比指标：训练 32 局 `avg recover` 耗时、`policy` 步占比、`Rapier` 4 局 `red_road_bumps` 卡死间隔（`>= 2s`）、目标跳变（`> 0.30m`）次数、`actor_parity_max_error`（导出一致性最大误差）及控分战绩。
