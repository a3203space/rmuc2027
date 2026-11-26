# 第 3 阶段哨兵 RL（145 维观测 + 有界策略）升级与 8 轮诊断基线总结（2026-11-26）

本文件总结本次提交中哨兵强化学习（`RL`，Reinforcement Learning）第 3 阶段架构升级、对应的 8 次 PPO（Proximal Policy Optimization，近端策略优化）更新诊断训练（`data/research/stage3-145-pooled`）以及 420 秒 `Rapier`（高精度刚体物理仿真引擎）4 对手评测中暴露出的 4 项根因问题。所有英文术语、配置键、命令参数与代码符号均附中文说明。

---

## 1. 第 3 阶段核心代码改动清单

1. **仿真引擎补全通道冷却状态与单元测试**
   - [packages/simulation/src/season2027.rs](file:///media/rm/E3E258F427E8E547/rmuc2027/packages/simulation/src/season2027.rs)：在触发 `tunnel_through`（穿洞冷却增益）时同步写入机器人字段 `crossing_cooling_until`（穿越冷却增益截止时间戳），使观测编码器与奖励统计不再漏算穿洞 1.5 倍冷却增益。
   - [packages/simulation/src/season2027_tests.rs](file:///media/rm/E3E258F427E8E547/rmuc2027/packages/simulation/src/season2027_tests.rs)：新增断言验证 `tunnel_through` 生效时 `crossing_cooling_until > m.time()`。

2. **诚实雷达地形目击器（`TerrainWitness`）消除了非爬阶底盘误报**
   - [research/rmuc2027_research/rl/terrain_witness.py](file:///media/rm/E3E258F427E8E547/rmuc2027/research/rmuc2027_research/rl/terrain_witness.py)：增加 `use_prior`（是否启用设计表先验开关），训练与导出默认使用 `use_prior=False`（只信赖雷达观测到的真实行为，避免“偷看设计表先验”）；新增 `FLOOR_SLOPE_TAN`（`1.5°` 场地纵坡正切补偿）与 `CLIMB_MAX_HORIZONTAL_M`（爬阶判定的最大水平跨距 `2.5m`），避免斜坡行进被误判为跨台阶。

3. **观测空间升至 `145` 维与动作执行护栏**
   - [research/rmuc2027_research/rl/portable.py](file:///media/rmuc2027_research/rl/portable.py)：
     - `OBS_VERSION = 3`、`OBS_SIZE = 145`（在原 131 维基础上新增 14 维战术态特征：哨兵应领未领免费补给次数的归一化余量 `grants_due`、3 种姿态的强化时长预算余量 `posture_budget`、穿越冷却/防御增益剩余时间 `cooling_rem`/`defense_rem`、双方前哨站血量比 `outpost_hp`、双方中央控制区占领标识、己方堡垒生效标识 `own_fortress_active` 与团队 17mm 购弹剩余额度比 `cap_left`）。
     - 目标滞回与到点停稳滞回：`GOAL_HYSTERESIS_M = 0.55`（建议目标点变化不超过 `0.55m` 时保持原目标 `prev_goal`）、`GOAL_RESUME_RADIUS_M = 0.48`（停稳后偏离目标 `<= 0.48m` 维持刹车）。
     - 三维建筑视线遮挡过滤：新增 `_segment_box_clear_3d`（3D 线段与旋转墙体包围盒求交检测）、`_structure_los_clear`（建筑视线遮挡判定）与 `_visible_structure_target`（仅在无墙体遮挡且在射程/视野内时才瞄准敌方前哨站或基地）。
     - 零弹药姿态保护：当 `ammo == 0` 时禁止进入或维持 `attack`（进攻姿态），强制切为 `defense`（防御姿态）。
     - 免费弹药回城判定：`free_grant_due` 在哨兵 `ammo == 0`、无在途购弹 `pending_ammo` 且 `due_grants > taken_grants`（存在到期未领的每分钟 100 发免费弹药）时触发回补给区 `recover`（恢复模式）。
     - 导出网络前向推理：`mlp_forward` 新增支持末端 `'tanh'`（双曲正切激活函数）层。

4. **有界策略网络（`TanhActorCriticPolicy`）与标准差受控**
   - [research/rmuc2027_research/rl/learning.py](file:///media/rmuc2027_research/rl/learning.py)：
     - 新增 `TanhActorCriticPolicy`（在 `action_net` 输出均值后接 `nn.Tanh()`，把策略输出严格限制在 `(-1, 1)^2` 场地边界内，消除动作分布外推越界与梯度饱和）。
     - 支持通过 `ppo.action_std` 初始化 `log_std_init = math.log(init_std)`；增强 `apply_action_std_control` 记录 `before_max` / `after_max`（截断前后的各维度最大动作标准差）。
   - [research/rmuc2027_research/rl/exporting.py](file:///media/rmuc2027_research/rl/exporting.py)：在导出策略脚本中内联 `TerrainWitness(use_prior=False)`，并支持导出 `nn.Sequential(nn.Linear, nn.Tanh)` 动作输出头。

5. **训练配置与加权对手池**
   - [research/rmuc2027_research/rl/session.py](file:///media/rmuc2027_research/rl/session.py)：支持 `opponent_pool_weights`（对手池加权确定性轮转权重）、训练期使用 `TerrainWitness(use_prior=False)`，并在 `terrain_buff_seconds` 统计中纳入 `crossing_cooling_until`。
   - [configs/rl/sentry-pooled-v3.json](file:///media/rm/E3E258F427E8E547/rmuc2027/configs/rl/sentry-pooled-v3.json)：关闭无人机（`drones_enabled: false`）、开启重装定时装配（`timed_assembly: true`）、给劣势对抗较强的 `claude_opus55_xhigh_2027_team` 与 `dsv41_1017_pair` 设置权重 `3.0`（其余 `1.0`）、保留 `rewards.death: 0.0`（阵亡不扣分）与 `rewards.terrain_buff: 0.0`（关闭单纯吃地形增益刷分），设定 `ppo.action_std: 0.35`、`ppo.ent_coef: 0.004`（熵正则系数）。
   - [tests/python/test_rl.py](file:///media/rm/E3E258F427E8E547/rmuc2027/tests/python/test_rl.py)与 [tests/python/test_rl_opponent_pool.py](file:///media/rm/E3E258F427E8E547/rmuc2027/tests/python/test_rl_opponent_pool.py)：补充 OBS v3 145 维特征、目标滞回、3D 墙体视线遮挡、零弹药切姿与免费弹药回城、有界 `TanhActorCriticPolicy` 导出一致性，以及 `opponent_pool_weights` 加权采样回归测试。

---

## 2. 8 次更新诊断训练与 `Rapier` 4 对手基线实测指标

在执行第 3.1 阶段修复前，使用上述代码对 `data/research/stage3-145-pooled` 运行了 `32,768` 步（`8` 次 rollout 更新、`16` 并行 `warp_kinematic` CUDA 环境、总耗时 `2310.67s`、采样速率 `14.254` 步/秒），随后导出模型（PyTorch 与导出脚本最大推理误差 `actor_parity_max_error = 2.26e-8`）并在 420 秒 `Rapier` 刚体物理引擎上对 `red`（红方哨兵）执行了 4 局确定性评测：

- **训练关键指标（对比第 2 阶段 `stage2-131-pooled` 同阶段前 8 轮 / 前 32 局）**：
  - **策略与探索稳定**：`action_std` 从 `0.3500` 保持至 `0.3490`（`clipping = false`），`approx_kl`（近似 KL 散度）在 `0.0019–0.0050`，`clip_fraction`（裁剪比例）`0.0045–0.0322`，`explained_variance`（价值网络解释方差）最高 `0.905`（第 8 轮 `0.815`），且动作越界 `oob = 0`、边界饱和 `sat(>0.98) = 0 / 1680`。
  - **地形目击器误报归零**：32 局训练中，非爬阶底盘槽位 `slot 1 / 4 / 7`（`2711 / 2732 / 2771`）的 `climb`（爬阶标记）误报为 **`0 / 32`**（第 2 阶段 528 局曾误报 `9` 次）。
  - **`recover`（回城恢复模式）耗时暴涨 > 3 倍**：32 局平均 `recover` 达 **`190.8s`**（最小 `41.0s`、最大 **`363.5s`**，第 2 阶段同期仅 `61.8s`）；单轮 `recover` 步占比高达 `22.9%–58.2%`，导致真正由神经网络主导的 `policy` 步占比被压缩至 `38.3%–70.1%`；`moving_commands`（非零移动指令占比）仍高达 `88.5%–98.8%`。

- **`Rapier` 4 局确定性评测明细（总战绩 1 胜 3 负，`policy_errors = 0`）**：

| 种子 (`seed`) | 对手策略 (`opponent`) | 胜负 | 控分比 (`red : blue`) | 死亡 (`deaths`) | 倒地耗时 (`dead`) | `recover` 耗时 | 减速带卡死 (`red_road_bumps >= 2s`) | `ammo0`（零弹占比 / 零弹进攻帧） |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `1008` | `claude_opus55_xhigh_2027_hold` | **胜** | `526.2 : 0.0` | `0` | `0.0s` | `25.5s` | **`187.8s`**（`t=13.9→201.7s` 单次死卡） | `6.1%` / `61` |
| `1009` | `claude_opus55_xhigh_2027_team` | **负** | `17.5 : 120.7` | `3` | `95.5s` | `119.0s` | **`106.9s`**（`t=313.1→420.0s` 卡至终局/阵亡） | `32.8%` / `3` |
| `1013` | `dsv41_1017_pair` | **负** | `158.1 : 785.4` | `1` | `20.5s` | `155.5s` | **`221.7s`**（`108.9s + 112.8s` 两次死卡） | `43.8%` / `53` |
| `1022` | `gpt61_2027_pressure` | **负** | `13.6 : 26.2` | `4` | `106.0s` | `118.0s` | `36.6s`（多段穿行未长期死卡） | `21.2%` / `42` |

---

## 3. 定位到的 4 项根因问题

1. **`2771` 哨兵在 `red_road_bumps`（红方公路减速带）高频死卡 `107–222` 秒**：
   - `red_road_bumps` 波高 `0.07m`、波距 `0.24m`（最大坡角约 `42.5°`），`2771` 质量 `27.15kg`、最大单推力 `105N`；一旦低速陷入波谷，在缺乏 `unstick`（卡滞倒车脱困）的情况下无法原地硬爬出波谷。
   - 同时 `sdk._square_to_bumps` 在入坡前原地转向会输出微小 `move` 向量触发 `moving == False`，再加上 `portable.py` 在 `recovering == True` 时无条件将姿态切为 `defense`（[configs/rules.json](file:///media/rm/E3E258F427E8E547/rmuc2027/configs/rules.json#L387-L420) 中 `defense` / `attack` 仅允许 `power_ratio = 0.5` 即 **`50W` 功率**，远低于 `move` 姿态的 `150W/120W` 及 `200W` 超级电容爆发），直接把哨兵送进波谷失速死锁。
2. **`free_grant_due`（免费弹药回城）与 `can_retreat`（允许撤退）耦合导致 `recover` 翻三倍**：
   - `free_grant_due` 无视团队当前金币与购弹上限，并在 `mode == 'recover' and gap > 1.0` 时直接禁用 `baseline.ammo_fields`（远程购弹），强制跨过全图 28m 跑回家；遇到 `can_retreat` 波动时更在中途产生“空弹乒乓折返”（单次浪费 ~55s），且全程 `50W` `defense` 龟速赶路。
3. **`GOAL_HYSTERESIS_M = 0.55` 小于 `0.25m` 网格斜步 `0.559m`**：
   - 在 `0.25m` 离散网格上 `(2, 1)` 格斜向距离为 `sqrt(0.50^2 + 0.25^2) = 0.559m > 0.55m`，导致中央高地边界附近仍有 `79–172` 次 `> 0.30m` 的目标跳变。
4. **早期中场探索偏窄与零死亡惩罚**：
   - 前 8 轮有效 `policy` 样本被严重挤压，确定性动作 `supplied_action` 尚集中在 `[-0.18, +0.13]`（地图中央火力交叉口），在按设定保留 `rewards.death = 0.0` 下高压局死亡偏多（后续将在第 3.1 阶段先保留设定解决前 3 项卡脖子阻塞问题后复测）。
