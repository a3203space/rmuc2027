# 24 小时哨兵训练结果（OBS v2/131，12 对手池）

`data/research/pooled-131-24h/`，2026-11-23 13:43 → 11-24 13:43，自然跑满
`stop_reason: wall_time_limit`。

**结论先行：工程链路全部正确，但策略学错了方向。**下文把「跑起来了」和「学对了」
分开记录，因为这两件事在本轮结论相反。

---

## 1. 训练元数据

```
actual_steps              1,138,944      gradient_epochs      4,448
rl_rollouts                   1,112      wall_seconds      86,401.7
stop_reason        wall_time_limit      num_envs                16
setup_seconds                 0.272      init_seconds         76.42
sampling_seconds         86,127.44       sampling_seconds_excluding_first  86,051.02
rl_steps_per_second       13.2239        rl_steps_per_second_excluding_first  13.2238
actor_parameter_l2_change     11.85      reload_prediction_equal   true
optimizer_reloaded             true      actor_updated             true
model_sha256   f4aea3d5ba8442854dd54ff1f39578efb20afc0b568534615bbb79c168cf629a
```

两个口径算出的速率几乎一致（13.2239 vs 13.2238），说明 `init_seconds` 在 86,000 秒
的总量里已经可以忽略 —— §9.4.1 那套拆分主要是为**短跑**准备的，长跑本来就会把
初始化摊薄。

---

## 2. 探索宽度：护栏守住，但末段在爬

| 阈值 | 轮数 | 占比 |
|---|---|---|
| `train/std` > 1.0 | 1,010 / 1,112 | 90.8% |
| > 1.1 | 220 / 1,112 | 19.8% |
| **> 1.2（判据线）** | **0 / 1,112** | **0.0%** |
| > 1.5 | 0 / 1,112 | 0.0% |

分段均值：1.012 → 1.000 → 1.033 → 1.061 → 1.047 → 1.048 → 1.109 → **1.131**。

对照 22h 那轮：首轮 **1.969**、末轮 **2.219**、**440/440 全部越界**。

所以 `action-std.json` 通道**一次都没被触发**，std 自己稳住了。但末段是单向上升的，
再跑 24 小时会碰到 1.2。下一轮建议把 `ent_coef` 从 0.01 降到 0.003~0.005，
而不是靠事后夹紧。

## 3. Critic 健康

`train/explained_variance` 分段均值 0.672 → 0.846 → 0.883 → 0.914 → 0.883 → 0.881
→ 0.899 → **0.904**，末值 **0.979**，中位 0.95 以上。比 22h（中位 0.885）更好。

---

## 4. 关键问题：学会了刷奖励，没学会赢

### 4.1 三个指标朝相反方向走

| 指标 | 起始 | 结束 | 变化 |
|---|---|---|---|
| `position_control` 奖励（每局均值） | 0.962 | 3.548 | **+269%** |
| 哨兵站在控制区的时间占比 | 22.9% | **89.2%** | **+66 pt** |
| 红方 `control` 得分 | 84.1 | 67.5 | **无上升** |
| 红方胜率 | 29.9% | **17.9%** | **下降** |

**红方 `control` 得分与「站控制区时长」的相关系数 = −0.077**，约等于零。

### 4.2 原因在奖励定义

`research/rmuc2027_research/rl/session.py:465-474`：

```python
on_control = robot['alive'] and any(
    robot['id'] in frame['zones'][zone].get('occupants', [])
    for zone in CONTROL_ZONES if zone in frame['zones'])
if on_control:
    self.metrics['position_control_seconds'] += dt
    terms['position_control'] += cfg['rewards']['position_control']*dt
```

判据只有「自己是否站在 `CONTROL_ZONES` 的占据者名单里」，**完全不看该站位有没有
让团队 `control` 上升**。于是最省力的策略不是争夺控制区，而是**找块控制区站着不动**
—— 这正是成绩单显示的：站桩时间涨到 89%，control 得分反而下降。

### 4.3 胜负由什么决定（已读引擎源码）

`packages/simulation/src/season2027.rs:2925-2943`，超时按三元组字典序比较：

```
(control×1000 取整, attack_damage, 存活非空中机器人总 HP)
```

1,344 局**全部 `timeout`**，双方基地 hp 从未低于 99.9%（`red` 均值 9,976 / 10,000，
`blue` 9,994）。也就是说**「摧毁基地」这条胜利路径在本轮从未出现过**。

只用 `control` 单项预测 winner：**1,160 / 1,344 = 86.3%** 正确；剩下 184 例是双方
control 都是 0.0，才落到 `attack_damage` / HP 比较。

⇒ 想赢就是在 420 秒内把 `control` 比值打上去，而 `position_control` 与它几乎无关。

---

## 5. 按对手胜率

```
dsv41_2027_control              95/96  = 99.0%
dsv41_1017_sentry               85/107 = 79.4%
dsv41_1017_heavy                47/113 = 41.6%
step5_2027_echo                 31/115 = 27.0%
claude_opus55_xhigh_2027_hold   30/120 = 25.0%
dsv41_1017_pair                 12/108 = 11.1%
gpt61_2027_control               5/134 =  3.7%
gpt61_2027_pressure              3/124 =  2.4%
gemini38_2027_blitz              2/106 =  1.9%
claude_opus55_xhigh_2027_team    1/99  =  1.0%
gemini38_2027_hunter             1/116 =  0.9%
gemini38_2027_tech_overload      1/106 =  0.9%
```

两端都压在极值上（99% 与 0.9%），说明学到的是对特定对手的固定反应，不是通用策略。
12 份池全部出现过，轮换机制本身正常。

## 6. 其他奖励项

| 项 | 非零局数 | 说明 |
|---|---|---|
| `outcome` | **1,344 / 1,344** | 符号与 winner **1,344/1,344 完全一致**，支付链路无误 |
| `position_control` | 1,344 / 1,344 | 均值 +2.60，见 §4 |
| `capture_seconds` | 219 / 1,344 | 均值 +0.0026 |
| `death` | 185 / 1,344 | 均值 −0.024 |
| `terrain_buff` | **0 / 1,344** | **从未触发**，隧道/攀爬路线一次没走 |

---

## 7. 下一轮该改什么（按影响排序，均未实施）

1. **重定义 `position_control`**：改成「站在区内 **且** 团队 control 因该站位上升」
   才给分，或直接让系数作用在 control 增量上。这是本轮 24 小时最大的浪费。
2. **`terrain_buff` 全零**：要么加探索引导让哨兵走隧道路线，要么先摘下这个项，
   避免它只占观测与奖励维度而不产生信号。
3. **`ent_coef` 0.01 → 0.003~0.005**：std 末段已爬到 1.13，再跑一轮会碰 1.2。
4. **注意胜利路径缺失**：1,344 局全 timeout 意味着权重里没有「打掉基地」的经验，
   评估时不要期待它能结束比赛。
5. 评估按 `docs/rl-randomized-opponents-plan.md` §9.8 的三列报告做，仍需
   `--backend rapier` 与 warp 训练环境对齐。

---

## 8. 产物位置

| 文件 | 说明 |
|---|---|
| `data/research/pooled-131-24h/model.zip` | 最终权重，sha256 `f4aea3d5…`（本地） |
| `data/research/pooled-131-24h/updates.jsonl` | 1,112 行，每轮 `action_std` / loss / behavior |
| `data/research/pooled-131-24h/episodes.jsonl` | 1,344 局，含 `reward_terms` / `control` / `witness` |
| `data/research/pooled-131-24h/training.json` | 上文 §1 的元数据 |
| `backup/rl-checkpoints/rl-run-8b55a3f7117730aa` | 训练期间 285 份检查点 + 最终权重（GitHub 分支） |

`data/` 被 `.gitignore` 排除，权重不进主分支历史；GitHub 上的副本走
`backup/rl-checkpoints/*` 分支，与既有 285 份检查点同一位置。
