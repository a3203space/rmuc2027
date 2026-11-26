# Stage-3.5：西部斜坡回收防甩飞修复、无损静态观测缓存加速与 v3.5 训练配置

**日期**：2026-11-27  
**基线提交**：`da18f2c`  
**目标定位**：在严格保持高层纯强化学习（`Pure RL`，不写入硬编码守家逻辑）和仿真数值/策略等价（`lossless parity`）的前提下，解决 40 次更新诊断中暴露的西部减速带出口甩飞死亡问题、极端多体碰撞循环拖慢，以及引擎到策略子进程（`worker.py`）之间 50% 冗余静态 JSON 观测传输开销，并更新 `sentry-pooled-v3.json` 准备 100 次更新热启动训练。

---

## 一、核心改动拆解

### A1. 西部斜坡回收防甩飞局部走廊 (`_recover_nav_goal`)
- **问题根因**：在 40 次更新重放复盘中，红方哨兵处于 `recover`（低血量/零弹药撤退补给）模式跨越西部 `red_road_bumps`（红方公路减速带）后存在两处几何协同失配：
  1. `on_bumps` 的横向判区余量 `margin_x=0.35`（横向判定余量）向西延伸至 `x=4.90`，在 `x=5.07` 离开减速带后仍将纵向移动约束为 `0.15 * |vx|`，使哨兵继续冲向 `red_kerb_1`（红方西侧路缘石，停滞 2~3 秒）；
  2. 沿 `red_road_slope`（红方公路斜坡，`ramp_end=[5.996, 11.980]` 至 `ramp_start=[6.664, 11.647]`）下行时，`sdk._route` 在未过坡中段前缺少直连南侧路径，会将路径绕至中央开阔节点 `(7.6, 10.6)`，在 `2 Hz` 决策步长（`0.5s`）下惯性将哨兵甩至 `x=8.79` 暴露于敌方狙击火力网。
- **修复方案**（[portable.py](file:///media/rm/E3E258F427E8E547/rmuc2027/research/rmuc2027_research/rl/portable.py#L559-L586) 与 [execute_goal](file:///media/rm/E3E258F427E8E547/rmuc2027/research/rmuc2027_research/rl/portable.py#L667-L702)）：
  - 新增红蓝镜像对称（基于地图宽高 `28m x 15m` 转到红方局部坐标 `(lx, ly)`）的局部导航辅助函数 `_recover_nav_goal`（仅当 `mode == 'recover'` 且地图为 `chapter4-2027-v1` 且返程目标为己方补给区时激活）：
    - 贴近坡顶北拐角：`3.9 <= lx <= 5.25, ly >= 12.25` 指向 `(4.95, 12.05)`，并提前解除 `on_bumps` 抑制以便及时向南拐弯；
    - 对准斜坡中段：`3.9 <= lx < 6.50, 11.45 <= ly < 12.25` 指向坡脚 `(6.65, 11.50)`；
    - 越过可直通阈值 `lx >= 6.45` 后（`6.45 <= lx <= 7.80, 10.65 <= ly < 12.15`）立即转向西南方 `(6.15, 10.55)` 收紧回撤半径，避开 `red_kerb_2` 东端；
    - 过路缘南沿：`4.20 <= lx < 6.45, 10.10 <= ly <= 11.15` 指向 `(3.85, 10.70)`，随后交还 `navigator.resolve_goal` 驶入补给区 `[1.5, 13.0]`。
  - 在使用中间路径点 `nav_goal != goal` 时禁用近距速度叠加缩放 `step_travel`（`nav_goal == goal`），消除二次减速造成的局部爬行卡顿。
- **实机 Rapier 对比验证**（2 Hz 策略步长，种子 `1008`）：
  - 红方（`red`）：坡底最大东移坐标 `extreme_slope_exit_x` 从 **`8.79` 收窄至 `6.89`（减少 `1.90 m`）**，总回程耗时从 `18.0s` 降至 **`14.5s`**；
  - 蓝方（`blue`）：坡底最西极端坐标 `extreme_slope_exit_x` 从 **`19.44`（镜像 `8.56`）收窄至 `21.00`（镜像 `7.00`，减少 `1.56 m`）**；红蓝全程零卡死。

---

### A2. CUDA 物理接触求解器 Zeno 微震早停 (`resolve`)
- **修改文件**：[contacts.py](file:///media/rm/E3E258F427E8E547/rmuc2027/research/rmuc2027_research/rl/warp_kinematic/contacts.py#L87-L95) 与 [contacts.py#L179-L185](file:///media/rm/E3E258F427E8E547/rmuc2027/research/rmuc2027_research/rl/warp_kinematic/contacts.py#L179-L185)。
- **问题根因**：部分高频楔角挤夹场景会在 `4096` 次连续碰触迭代循环中产生零时间推进（`earliest <= 1e-7`，芝诺现象 `Zeno chatter`），单核单线程空转耗尽 `4096` 步。
- **修复设计**：通过实验测量十连体顶墙推挤链最大合法收敛连续零推进次数为 `2477` 次（自由十连体为 `619` 次）。据此采用与连通簇体数 `size` 平方自适应缩放的连续零推进阈值：
  $$\text{stalled\_limit} = \max(256,\ 32 \times \text{size}^2)$$
  当单次或双车卡死在墙角楔口（`size=1..2`）时，`stalled >= 256` 即提前退出（对比 `4096` 加速 **`16x`**），并保持原有的 `contact_exhausted = 1` 与速度归零语义不变；同时大尺寸多体链（如 `size=10` 时阈值为 `3200 > 2477`）仍可完整收敛，不破坏物理准确性。
- **新增回归测试**：[test_acute_wedge_zeno_exits_early_and_marks_exhausted](file:///media/rm/E3E258F427E8E547/rmuc2027/tests/python/test_kinematic_contacts.py#L80-L89)（`0.0176s` 内稳定退出并标记 `contact_exhausted=[1, 1]`）。

---

### A3. 引擎-策略进程无损静态观测缓存 (`compact_observations`)
- **跨层改动链路**：
  1. **Rust 仿真核心**：[lib.rs](file:///media/rm/E3E258F427E8E547/rmuc2027/packages/simulation/src/lib.rs#L1675-L1705) 新增 `static_observation()`（输出 4 个全场不变字段：`arena`, `rulebook_rules`, `simulation_assumptions`, `specs`）与 `observation_without_static()`（省略上述 4 项及重序列化克隆开销）；[main.rs](file:///media/rm/E3E258F427E8E547/rmuc2027/packages/simulation/src/main.rs#L98-L120) 新增 `"static_observation"` RPC 操作并在 `"observations"` 中支持 `"static": false` 开关。
  2. **核心引擎与协议层**：[engine.py](file:///media/rm/E3E258F427E8E547/rmuc2027/packages/core/rmuc2027_core/engine.py#L62-L64) 提取共享观测拉取辅助方法 `_fetch_observations` 与 `_static_obs_cache` 缓存，提供 `static_observation_data()` 和 `compact_observations()`；[policy.py](file:///media/rm/E3E258F427E8E547/rmuc2027/packages/core/rmuc2027_core/policy.py#L99-L235) 为 `EncodedObservation` 携带 `static_payload` 与 `static_data`，且仅针对受信任本地进程沙盒（`sandbox == "trusted-local"`）在首帧确认 `_receive` 成功后置位 `_static_seeded`，后续帧仅向管道发送精简动态 JSON；[worker.py](file:///media/rm/E3E258F427E8E547/rmuc2027/packages/core/rmuc2027_core/worker.py#L23-L60) 在读取 `"__cache_static__": true` 后于工作进程内存常驻该 4 字段并通过 `setdefault` 还原完整 `observation`。Docker 沙盒默认保持全量拼接，零兼容性风险。
  3. **对战与训练调度层**：[battle.py](file:///media/rm/E3E258F427E8E547/rmuc2027/research/rmuc2027_research/strategies/battle.py#L191-L246) 在 `Battle.step()` 中启用 `encoded_observations(..., compact=True)`；[session.py](file:///media/rm/E3E258F427E8E547/rmuc2027/research/rmuc2027_research/rl/session.py#L300-L307) 的 `GoalController.decide` 直接合并 `static_data`。
- **实测传输削减与无损等价验证**：
  - 单条策略观测荷载由 **`100,838 B` 降至 `51,052 B`（减少 `49.4%`）**；
  - 真实 `trusted-local` 对局同种子（`seed=1008`）对比全量路径与精简路径，终局 `match.digest()`（对局确定性哈希摘要）严格一致：**`d7f36db3b88dcad3 == d7f36db3b88dcad3`**。

---

### A4. Stage-3.5 纯 RL 奖励权重与 12 对手采样池更新
- 修改配置文件 [configs/rl/sentry-pooled-v3.json](file:///media/rm/E3E258F427E8E547/rmuc2027/configs/rl/sentry-pooled-v3.json#L28-L56)：
  - **纯 RL 奖励塑形调整**（不改变核心占领大权重）：
    - `rewards.position_control`：`0.01 -> 0.03`（增强占领要道点位的过程引导，引导哨兵积极夺取关键控制位置）；
    - `rewards.capture_seconds`：`0.005 -> 0.01`（鼓励持续压区与占点读秒）；
    - `rewards.control`（`0.01`）、`rewards.outcome`（`10.0`）、`rewards.death`（`-0.5`）保持稳定。
  - **对手采样权重（`opponent_pool_weights`）靶向倾斜**：
    `[1.0, 3.0, 1.0, 3.0, 0.5, 0.5, 2.0, 2.0, 2.0, 1.5, 1.0, 0.5]`，重点加强高压对抗及弱势对手配比（`team=3.0`, `pair=3.0`, `gemini38_{blitz,hunter,tech_overload}=2.0`, `gpt61_control=1.5`, `hold/heavy/pressure=1.0`, `sentry/dsv41_2027_control/echo=0.5`）。
  - 同步更新配置回归测试 [test_v3_config_matches_requested_settings](file:///media/rm/E3E258F427E8E547/rmuc2027/tests/python/test_rl.py#L1570-L1580)。

---

## 二、验证结果与吞吐基准

| 验证项目 | 命令/测试范围 | 实测结果 |
| :--- | :--- | :--- |
| Rust 仿真器全量单测 + Release 编译 | `cargo test --manifest-path packages/simulation/Cargo.toml && cargo build --release ...` | **67 passed, 0 failed**（14.87s）；Release 编译耗时 34.32s |
| Python 全量 RL / 运动学 / 观测复用 / 传输测试 | 32 个核心测试模块（包含 4 个新增/更新的 A1/A2/A3/A4 回归测试） | **231 tests OK**（196.554s） |
| `16 envs / 8 shards` 随机动作全对局吞吐压测 | `--num-envs 16 --sim-shards 8 --steps 2048 --action-pattern uniform-random` | 总步数 `32,768` 步（32 局完整 `420s` 仿真），采样耗时 `839.68s`，持续吞吐 **`39.02 RL steps/s`**（每环境 `1.22x` 实时倍速，全过程零芝诺卡死） |
