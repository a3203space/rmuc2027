# 哨兵 v4：阵容、遥测准备、能力漂移与跨机恢复

本页对应 `rl/scenarios.py`、`offline2026.py`、`capabilities.py` 和 `v4_bundle.py`。
这些模块不会自行启动训练。SQLite 准备只做只读 ETL；模型拟合由训练入口另行执行。

## 阵容和覆盖口径

学习哨兵始终为 2773。步兵使用 2731（轮腿）、2732（折叠）、303（全向），敌哨兵使用 2772、2773、2771。
双方步兵 3/4 号席位都有顺序：`9 × 9 × 3 = 243` 阵容，换红蓝为 486 场景；合并步兵席位排列后是 108 数量组。

```python
from rmuc2027_research.rl.scenarios import ScenarioSampler
from rmuc2027_research.rl.capabilities import CapabilitySampler

sampler = ScenarioSampler(seed=42)
case = sampler.sample()
lineups = case.lineups()  # case.learner_side 的 7 号保证为 2773
profiles = CapabilitySampler(seed=43).sample(lineups, case.learner_side)
sampler.record(case, valid_steps=600, seconds=60, completed=True, failure=False)
saved = sampler.state_dict()  # 可直接 JSON 序列化
resumed = ScenarioSampler.from_state_dict(saved)
```

首批 108 场景每个数量组恰一次，红蓝各 54。然后轮换其余变体直到 486 个场景都被抽到。
此后 70% 按数量组均匀抽样，30% 重放训练失败；验证或测试结果不能反馈给此采样器。
`coverage()` 分别报告已启动、有有效样本、完成 60 秒、完成 420 秒的覆盖，不能把一次 reset 当作完整验收。
复现一局需要场景描述、比赛种子与实际能力配置；只有同一个 seed 不等于恢复完整抽样状态。

## 每局能力和缓慢漂移

初始机动低／中／高档概率为 `.25/.5/.25`，三项倍率依次为：

| 档位 | speed_reference_mps | motor_force_max_n | drive_efficiency |
| --- | ---: | ---: | ---: |
| 低 | .75 | .8 | .9 |
| 中 | 1 | 1 | 1 |
| 高 | 1.2 | 1.15 | 1.05 |

自瞄独立按相同概率抽强／中／弱档：50/150/300 ms 延迟，.25/1/2.5 度 OU 稳态标准差，相关时间常数 .15 秒。
`calibration_scale` 固定为 1；2026 速度／功率包络用于校验，不被反演成真实硬件极限。
主 2773 机动倍率固定 1，自瞄额外延迟与噪声固定 0；敌哨兵可随机，但仍使用严格哨兵观测。

```python
from rmuc2027_research.rl.capabilities import SlowCapabilityDrift, AimNoiseExecutor

drift = SlowCapabilityDrift(profiles, seed=44, learner_side=case.learner_side)
current = drift.update(time_s)
aim = AimNoiseExecutor(current["blue:3"]["aim"], seed=45)
aim.set_profile(current["blue:3"]["aim"])
reference = aim.step(time_s, aim_rad, pitch_rad)
```

只有敌方地面角色随时间漂移，队友保持初始抽样能力。每 15 秒独立选目标，以 20 秒时间常数指数逼近。
目标围绕初始档：速度 ±10%、力 ±8%、效率 ±3%、自瞄标准差 ±20%、延迟 ±50 ms。
倍率始终相对于名义硬件，不乘在上一帧结果上；真实效率限制为不超过 1。
摩擦、质量、制动、规则限额和弹丸散布不由此模块改变。

瞄准执行器保留最近 2 秒历史，延迟增长时仍可找到过去的参考；更新 profile 不清除 OU 状态。
启动时还没有足够历史则 `ready=False`，执行层不得据此接受开火。
漂移和 OU 均提供 `state_dict/from_state_dict`；跨机器新回合开始时重新初始化回合状态，保留抽样器祖先记录。

## 2026 数据准备和真实性边界

```bash
PYTHONPATH=packages/core:research research/.venv/bin/python \
  -m rmuc2027_research.rl.offline2026 \
  rmuc2026shuj/rmuc_2026_region_dataset.sqlite \
  --output data/research/sentry-v4-preparation
```

输出目录必须不存在，避免覆盖既有准备版本。读取使用 SQLite URI `mode=ro` 和 `query_only`；准备前后验证数据库 SHA-256 相同。
按 `赛区+场次号` 把 266 场系列赛分为 186/40/40，保持同一系列赛所有局、双方与机器人不跨集合，并按赛区分层。
训练包络只使用训练集合；另外保存验证／测试的独立审计统计，不用于拟合。

输出为 `train-trajectories.npz`、`validation-trajectories.npz`、`test-trajectories.npz`，均可 `np.load(..., allow_pickle=False)`。
manifest 保存数据库与表结构哈希、拆分哈希、源字段、准备源码快照及运行版本、清洗定义和每个轨迹在 NPZ 内的 offset。

`state` 列依次是：x、y、z、原始枪口朝向、功率、HP、最大HP、17mm热量/上限、42mm热量/上限、累计17mm/42mm发弹、队伍累计金币、剩余金币、雷达标记。
`state_valid` 为逐字段有效掩码。额外包含 game_id、robot_id、time_s、role_id、side_id、alive、原坐标与有效掩码。
`role_id` 为英雄0、工程1、步兵3为2、步兵4为3。

`future_source_xy[:, :, :]` 对应 1/3/5 秒的未来原坐标，并有完整连续链、存活和定位质量掩码。
工程仅保留第一次装配成功之前的弱前期区间；这并不证明到达装配区域的精确时刻。
`original_action_mask` 全为 false：没有真实摇杆、速度命令、瞄准俯仰或装配动作，不能把位置差伪装成原始动作。
没有确认地标／轴方向映射时，`field_position_valid` 和全部 `semantic_goal_mask` 为 false。
历史胜方单独作为标签保存，不是 2027 奖励，也不放进 state。

可在训练集合进行真实的辅助表征预训练：未来位移长度、停驻／运动持续性、累计发弹的粗时间窗变化。
这些量可不依赖绝对坐标原点；局部转向应使用连续运动方向建立参考，不能把未确认的枪口方向当作底盘朝向。
绝对 2027 路点与低层动作继续屏蔽；辅助预测头应与策略动作头区分，并由训练入口记录真实优化次数。

速度统计定义为 1 秒平面弦长，不是瞬时极速。报告原始异常最大值、条件场地边界分布、移动阈值 .2 和六点连续性候选；不使用速度硬截断。
位置有效清洗排除 0/-1 哨点，并标记超过局部位移中位数六倍且超过 1 米的孤立折返定位尖峰。
坐标单位米已由用户确认，轴方向／地标仍须独立确认。

本次准备的速度统计只包含步兵 3/4。训练集条件场地范围内的 595,175 个相邻位置样本，中位速度为 0.311439 m/s；其中运动样本 350,448 个，中位速度为 0.822707 m/s。六点连续性筛查后的 27,736 个样本，最大候选为 3.621895 m/s。全数据对应最大候选为 3.790741 m/s，来自验证集，不能用来拟合训练包络。原始训练集最大值 52.927531 m/s 是明显异常，不能作为硬件速度上限。

训练集相邻遥测平均功率在 0–50、50–80、80–100 W 的条件速度中位数分别为 0.180887、0.593196、0.710216 m/s。更高功率区间样本较少且关系不单调；遥测功率不等于规则功率限额，不能据此反推电机参数。

## 跨机器包

```bash
PYTHONPATH=packages/core:research research/.venv/bin/python \
  -m rmuc2027_research.rl.v4_bundle pack \
  --run-dir data/research/NEW_RUN --output data/research/NEW_RUN-resume.zip \
  --preparation data/research/sentry-v4-preparation
```

在 PPO 更新完成且停止修改检查点后打包。布局为 `source/`、`models/{main,infantry,heavy}/`、`preparation/`、`frozen-pool/`、`run-state/`。
源码包含 Python/Cargo/workspace manifests 和锁文件，排除 `.git`、构建产物、虚拟环境、数据目录。
数据库默认外置，manifest 记录其 SHA-256，不重复携带 1.2GB SQLite。
各 stage 全部保存模型、优化器 ZIP、actor.json、experiment、示范、resume-state.pt 与 samplers.json。
同时保存 `evaluation/`、`radar-calibration-next/`、`selected-export/` 和 run-state 下的 selection/preflight/initialization 报告，保留对照证据。
主模型使用 AsymmetricPPO 加载两个优化器，辅助模型使用 PPO；预检还可独立验证 portable actor JSON。

`run-state/requirements-runtime.txt` 与 `runtime-environment.json` 保存本轮实际安装的非 editable Python 版本和 Python 版本；源码中的旧 CPU 锁文件不代表本轮环境。包不包含 wheelhouse、GPU 驱动或系统 GUI 库。

目标机器先用已有 Python 运行安全解包，再依据源码锁定依赖安装环境：

```bash
python -m rmuc2027_research.rl.v4_bundle unpack NEW_RUN-resume.zip --destination NEW_BUNDLE
cd NEW_BUNDLE
cargo build --release --manifest-path source/packages/simulation/Cargo.toml
PYTHON_BIN=/path/to/venv/bin/python bash preflight-ubuntu24.sh
```

`load_runtime_config(bundle, stage)` 重定位 friendly、opponent、controllers、auxiliary_pool 和 radar_classifier_path。
`radar-calibration/` 的分类器、校准行与报告一并保存；配置非空但分类器缺失会拒绝打包，初始尚未校准时应省略此配置。
`relocate_model_runtime(model, bundle, stage, output_dir=...)` 重绑定学习率控制和日志路径，并清除旧示范缓存。主策略 v4 与辅助策略 v5 会显式清空运行时示范路径／数据集，PPO 阶段不启用额外 BC；示范文件仍保留供审计。旧版策略按配置保留示范重定位语义，绝不继续使用另一台机器的绝对路径。
`preflight` 核验文件哈希、Ubuntu版本、Python依赖、实际CUDA设备与显存及短推理，不远程连接、不启动训练；缺少推理依赖或 CUDA 不可用时输出失败报告并返回非零退出码。
`inference_ready` 表示三个阶段推理通过；`training_ready` 还要求 Rust 二进制能启动且规则哈希匹配、Warp 的 CUDA 设备可用，以及全部训练／仿真依赖可导入。策略推理／更新默认用 CPU，Torch CUDA 只单独报告，不是训练前提。`ready` 等于 `training_ready`，仅检查推理不能宣称可继续训练。

恢复保留 actor/critic、优化器、训练计数、抽样器／覆盖游标、可兼容的 RNG；归一化未使用时应显式记录 none。
机器／后端变化后丢弃半个 rollout 和未完成环境，开启新回合。GPU逐位一致并不承诺；并行数与吞吐在目标机器重新实测。

恢复时若 worker 数变化，或检测到旧检查点多个 worker 的场景／能力随机流被复制，`restore_state` 会进行显式 sampler 迁移。同拓扑且随机流独立时仍原样恢复。迁移为每个 worker 派生独立的场景、能力、会话和 Gym 随机流，并独立排列尚未开始的覆盖队列。旧队列已开始前缀取并集保留；这里的开始包含未完成回合，**不代表完成覆盖**，后续失败回放和全覆盖后的采样仍可回访。

旧场景计数逐项合计后只分配一次，防止扩容复制历史计数；失败回放权重按场景取最大值，完整原始权重仍保存在祖先记录中。历史本身若曾被复制，无法据此推断哪些记录应去重，因此原记录全部保留，迁移不会新增真实覆盖证据。每个 stage 的 `sampler-restore.json` 保存原文件 SHA、完整祖先状态、原游标、计数守恒、派生种子与迁移原因，并随恢复包携带。迁移明确不是逐位恢复；读取时不改写旧 `samplers.json` 或 `resume-state.pt`，新的活动状态只在后续正常检查点保存时写入。

验证环境后，从解压包继续三阶段训练（`--resume-from` 接受解压包，不接受原始 run 目录）：

```bash
cd NEW_BUNDLE/source
PYTHONPATH=packages/core:research .venv/bin/python -m rmuc2027_research.rl.v4_campaign run \
  --resume-from .. --output /path/to/NEW_RUN --wall-seconds 7200 --num-envs 0
```

计时包含预检、初始化或恢复、PPO、配对评估、导出打包和解包后推理验证。`num-envs=0` 实测选择并行数。v4/v5 的示范学习只在初始化阶段执行，恢复 PPO 不重新启用 BC；默认继续 `model.zip`，`selected.zip` 是通过配对选择的展示候选，二者不一定相同。

恢复训练的吞吐预检会加载包中的主策略与真实冻结辅助策略，逐步执行主策略预测，记录模型和 helper 哈希，报告为 `frozen_main_and_auxiliary_rollout`。尚未初始化辅助策略的新实验只能测脚本环境，明确标为 `scripted_environment_only`。两者目前均为一次热身加八步的开局短测，不含 PPO 更新、整局拥挤接触或后期战斗开销，不能把该速率当作完整训练吞吐；旧首轮约 16 samples/s 属于后者，实际主训练明显更慢。目标机器应使用新包的恢复入口重新测量，不按显卡型号直接推断并行数。
