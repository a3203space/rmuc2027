# 独立研究工具

研究源码位于 `research/`，与联机应用共享 2027 仿真和 SDK。联机主机不导入研究模块；独立网页服务提供策略上传、训练任务、实时对战与排行榜。

## 完整网页观战

在 rmuc2027 项目根目录准备并启动，所有资源均来自本项目：

```bash
uv sync --locked --group research
cargo build --locked --release -p rmuc-sim
bun install --frozen-lockfile
bun run build:arena-web
uv run --locked --group research rmuc2027-research serve --port 8767
```

打开 http://127.0.0.1:8767，选择兵种型号、双方策略或逐机器人策略，再点击开始对战。完整保留原版首页、正式/自由对战、训练、策略库、回放、排行榜、战场设置、外观上传和录像导出页面。数据独立保存在 `data/research/web`；无需 Qt，也无需在另一个仓库启动服务。

前端开发：先运行上述网页服务，再执行 `bun run dev:arena-web`，打开 http://127.0.0.1:5174。API 和 WebSocket 自动代理至 8767。

默认策略以可信本地模式执行。Docker 模式使用下节同一个策略镜像，启动命令添加 `--sandbox docker`。MP4 转码依赖研究环境中的 `imageio-ffmpeg`；原始 WebM 仍可单独保存。

## 策略开发与对战

```bash
uv sync --locked --group research
uv run --locked --group research rmuc2027-research new-strategy data/research/my-strategy
# 编辑生成的 strategy.py，实现 act(observation, state) -> (actions, new_state)
uv run --locked --group research rmuc2027-research bundle data/research/my-strategy/strategy.py data/research/my-bundle
uv run --locked --group research rmuc2027-research battle \
  --red data/research/my-bundle \
  --blue research/strategies/examples/claude_opus55_xhigh_2027_team \
  --output data/research/my-match.json
```

默认 `trusted-local` 只用于信任的本地代码。策略源通过独立子进程执行，并限制输出和响应时间。需要 Docker 隔离时，先构建 `docker build -f research/strategies/Dockerfile -t rmuc2027-policy:0.1 .`，对战加 `--sandbox docker`。

已有合法 2027 策略保留在 `research/strategies/examples/`。策略文件、manifest 与原哈希保持一致。

## 哨兵强化学习

神经网络依赖单独安装，不进入默认联机环境：

```bash
# 已有研究环境时跳过创建；研究依赖不安装到联机环境
uv venv --python 3.12 research/.venv
uv pip install --python research/.venv/bin/python -e packages/core -e research
uv pip install --python research/.venv/bin/python -r research/requirements-rl.txt --torch-backend cpu
research/.venv/bin/python -m rmuc2027_research rl check --config configs/rl/sentry.json --run-dir data/research/new-check
research/.venv/bin/python -m rmuc2027_research rl train --config configs/rl/sentry.json --total-steps 256 --num-envs 1 --device cpu --run-dir data/research/new-train
research/.venv/bin/python -m rmuc2027_research rl evaluate --model data/research/new-train/model.zip --seeds 1001 --sides red blue --duration 420 --run-dir data/research/new-eval
research/.venv/bin/python -m rmuc2027_research rl export --model data/research/new-train/model.zip --output data/research/new-policy
```

Windows 的解释器路径为 `research\.venv\Scripts\python.exe`。在已有独立研究环境中，也可直接使用 `python -m rmuc2027_research`，安装共享模块和研究包即可，无需客户端或联机服务。

已验证的 Linux/Python 3.10 CPU 依赖快照保存在 `research/requirements-rl-cpu.lock`，仅用于匹配的平台环境。

Actor 仍为 116 维合法局部观测，输出二维目标点。其余机器人由固定脚本控制，导航、瞄准、射击和回补由原执行模块负责。PPO 默认每 64 步更新，网络为两层 64 单元，默认回合在 60 个仿真秒截断；自然比赛持续 420 秒。完整配置位于 `configs/rl/sentry.json`。

旧模型需显式迁移，迁移保留原权重、优化器和原始实验元数据，记录运行时版本变化：

```bash
research/.venv/bin/python -m rmuc2027_research migrate-model \
  data/research/sentry-resume/model.zip data/research/sentry-resume-migrated
```

只接受已记录的原运行时与相同观测/动作契约，不通过修改原哈希绕过兼容检查。输出目录必须不存在。迁移后可恢复训练、评估和导出；不声称与旧引擎逐帧一致或胜率提升。

### 特殊底盘目标点导航

公共 SDK 的 `navigate_action(robot, point, observation, state)` 返回完整动作字段，调用方为每个机器人保存独立状态，再通过 `action(**fields)` 下发。原 `navigate()` 仍返回二维移动方向。默认策略及策略库的公共行驶入口已接入新接口，目标点训练执行层同样保留跳跃、腿高和折叠指令。

当前针对 RMUC 2027 地图验证了 2731 登阶轮足从己方进入中央区域的跳跃流程，以及 2732 折叠步兵经己方道路隧道进入中央区域的流程。洞口先对正并完成折叠，穿洞期间持续保持折叠，车身离开洞顶后展开。特殊动作有超时回退。其他目标仍使用原导航；尚未实现全地图按预计耗时比较所有特殊路线。

已有训练工作区需通过工作区参考文件刷新机制更新 SDK；用户自定义策略需自行接入新接口，刷新参考文件不会重写用户策略。

### MJWarp 完整 5v5 哨兵训练场景

Ubuntu 原生窗口实时观看 MJWarp 完整 AI 对战（不启动网页服务）：

```bash
research/.venv/bin/python -m rmuc2027_research.rl.mjwarp.viewer \
  --config configs/rl/sentry-mjwarp-hour.json --device cuda:0 --duration 420
```

在 Ubuntu 桌面终端运行。窗口使用 MuJoCo 原生查看器，呈现实际 MJWarp 场地和机器人碰撞几何，而非 Arena 网页的外观模型或操作手 HUD。双方使用配置中的固定 AI 策略，不加载 RL 检查点。空格暂停/继续，鼠标旋转、平移、缩放；关闭窗口停止，比赛结束窗口关闭。终端输出比赛仿真时间和双方基地血量。默认按不超过 1 倍实时节奏展示；`--speed 4` 提高播放目标速度，实际推进仍取决于仿真性能。可视化有额外开销，性能基准应使用无窗口 benchmark。此入口尚待桌面 GPU 实机窗口验证。


先确认实际执行训练的环境能访问 GPU：

```bash
research/.venv/bin/python -m rmuc2027_research.rl.mjwarp.doctor \
  --output "data/research/gpu-admission-$(date +%Y%m%d-%H%M%S).json"
```

此命令检查驱动、CUDA 张量计算、两世界 MJWarp 步进，以及 CUDA Graph 与普通 GPU 步进的结果一致性。每个计算探针默认限时 300 秒，包含首次编译；失败返回非零退出码并保留诊断。输出文件必须不存在。它不是完整比赛性能或训练验收。

如果用户终端能运行 CUDA，而代理执行环境没有 `/dev/nvidia*`，需要在执行环境管理层解决设备可见性。安装 CUDA 版 PyTorch 并不能解除执行环境的设备隔离。必须在代理执行工具中重新通过检查，才能声称已完成代理侧 GPU 接入。


MJWarp 后端已接入完整比赛与 PPO。保留原 116 维局部观测、二维目标点动作、固定队友与对手、射击、占点、资源、死亡复活和胜负规则。CPU 的 Rust 引擎计算驱动及规则，并维护光学/弹丸查询镜像；MJWarp 负责每个 120 Hz 子步的刚体运动与接触求解。规则与物理按原六子步顺序交替，Rapier 不再同时积分训练世界。

模型直接从 Rust 创建的碰撞几何与质量属性转换，覆盖现有 18 种型号。凹地形分解为凸棱柱，复合机器人保留各碰撞部件，折叠与机械臂变化在子步前同步。非同构碰撞掩码的世界分组运行，不会为并行而移除规则或机器人。

本轮性能修复：Rust 场景首次发送完整几何，随后只发送变化和删除的碰撞体；Python 缓存未变化几何；输入数组复用设备内存；碰撞结果批量回读，不再逐世界下载完整求解器。CUDA 路径增加单子步 Graph 捕获执行，CPU 调试仍直接步进。训练每个完成的采样步检查是否需要输出 15 秒进度，长采样步会推迟输出。

实际协议探针中，默认阵容首子步约 1.44 MB，后续子步约 38 KB（约减少 97%）。这仅是传输量测量，不能代表端到端加速比。当前工具环境无法访问 CUDA，CUDA Graph 与实际 GPU 吞吐尚待实机验证。

2026-11-03 实机分段基准（两环境各四个 RL 步）：采样 52.665 秒，其中几何展开与签名阶段 42.465 秒，模型准备 3.240 秒，步进及同步 0.318 秒。定位到不同世界复用碰撞体编号时，几何缓存互相覆盖；已改为按不可变描述对象身份保存有界 LRU 缓存，避免静态地形交替重新分解。局部 CPU 几何测试 20 次交替调用从约 1.064 秒降至 0.00071 秒；完整 GPU 采样提速尚待修复后实测，不能按局部加速倍数推算。

几何缓存修复后实机复测：两环境各四步的采样耗时降至 12.245 秒（每环境约 0.327 倍实时，较此前约快 4.3 倍），尚未达到两局实时目标。后续一批改动增加每槽位复用的 CPU 准备数据、模型参数缓存和几何变更检查，以及模型名称索引和有界拓扑签名缓存。切换槽位时恢复对应模型数组，防止不同世界几何相互污染；这一批改动的 GPU 吞吐仍待实测。物理热启动与固定世界调度尚未完成。

准备缓冲区优化后实机复测：两环境各四步采样 7.573 秒，每环境约 0.528 倍实时，聚合约 1.056 仿真秒/现实秒。聚合超过 1 不代表任一单环境快于现实，当前仍未达标。后续将 CPU 准备缓存从瞬时批次槽位绑定改为逻辑源世界绑定，避免到达顺序变化引起缓存失效；这一变化不等于已经完成固定世界 GPU 调度或热启动持久化。验收目标已提高为每环境快于现实时间。

训练专用摘要现通过 Rust `training_step` 直接生成，用于无回放采样；策略视野仍单独按原接口计算。开局测试状态传输从 47,313 字节降为 7,511 字节。录制回放时自动使用完整帧。`training_summary: false` 可用于同频率对照。

新增 `fixed_policy_hz`（5 或 10）与 `learning_control_hz`（2、5 或 10），分别控制固定队友/对手和学习哨兵底层执行频率；`rl_seconds` 单独控制 RL 目标点决策周期（2 Hz 对应 0.5）。所有频率均保持 20 Hz 比赛规则和 120 Hz 物理。固定策略降频会改变对手反应，需与原 10 Hz 环境复评。`configs/rl/sentry-mjwarp-fast.json` 当前仅固定 AI 改为 5 Hz，学习控制与目标点频率待确认后设置。

正式训练默认 `action_trace: false`，动作文件保存每 15 秒采样边界摘要及最终摘要；设为 true 恢复详细逐步动作。碰撞体没有变化的刚体停止重复重算质量属性。以上改动尚待本机 GPU 端到端实测；静态观测增量传输、同 tick 观测复用与进程内数组接口仍未实现。

后续仍需完成 PyO3 数组绑定、稳定世界槽位及确定性批次调度、GPU 接触压缩与完整阶段计时。以下命令可用于诊断；未达到实机性能验收前，不建议直接进行长时间训练。

并行训练使用 CPU 环境线程和共享物理批处理服务，合并同时到达的请求在 GPU 上推进。世界状态、策略状态与重置相互独立。模型缓存有容量上限；接触/约束缓冲区溢出会报错，不能静默丢掉碰撞。现阶段 Rust 与 Python 之间仍使用进程协议，CPU 规则和通信开销可能限制加速比；尚未实现计划中的 PyO3 数组绑定。

安装物理依赖（已有 CUDA 版 PyTorch 时无需重装）：

```bash
cd ~/rmuc2027
uv pip install --python research/.venv/bin/python -r research/requirements-mjwarp.txt
cargo build --release --locked -p rmuc-sim
```

先用两局并行运行小规模 GPU 训练（输出目录必须不存在）：

```bash
research/.venv/bin/python -m rmuc2027_research.rl train \
  --config configs/rl/sentry-mjwarp.json \
  --device cuda --num-envs 2 --total-steps 256 \
  --run-dir data/research/sentry-mjwarp-first
```

`--device cuda` 控制 PPO，`--sim-device cuda:0` 控制物理；配置已选择 `sim_backend=mjwarp`、`sim_device=cuda:0`。也可向现有训练/诊断命令显式传入 `--sim-backend mjwarp --sim-device cuda:0`。CPU 调试使用 `--device cpu --sim-device cpu`，请求 CUDA 时不会自动回退 CPU。

显存及吞吐量允许时扩大并行：

```bash
research/.venv/bin/python -m rmuc2027_research.rl train \
  --config configs/rl/sentry-mjwarp.json --device cuda \
  --num-envs 8 --total-steps 4096 \
  --run-dir data/research/sentry-mjwarp-8
```

端到端基准排除一次预热步，记录采样速度、物理批次数和最大合并世界数。预热耗时单独报告，物理计数和阶段计时在预热完成后清零，避免将预热计数除以正式采样时间。新增每环境实时倍率、每仿真秒现实耗时、批次大小分布，以及几何签名、编译、模型准备、上传提交、步进同步、状态与接触回读、结果编码的主机墙钟耗时。阶段计时不是 CUDA kernel 事件时间：异步上传的完成时间可能计入下一同步阶段。请使用不同输出目录分别比较 MJWarp 与 Rapier；CPU 调试结果不能证明 GPU 加速。

```bash
research/.venv/bin/python -m rmuc2027_research.rl benchmark \
  --config configs/rl/sentry-mjwarp.json --num-envs 1 2 4 8 --steps 16 \
  --run-dir data/research/bench-mjwarp
research/.venv/bin/python -m rmuc2027_research.rl benchmark \
  --config configs/rl/sentry.json --num-envs 1 2 4 8 --steps 16 \
  --run-dir data/research/bench-rapier
```

模型保存为 `model.zip`，支持 `--resume`；续训严格检查配置、规则、策略、仿真源码及依赖版本。更换后端需新实验，不能把旧优化器断点当成同一实验继续。

训练后在原 Rapier 比赛环境评估与导出：

```bash
research/.venv/bin/python -m rmuc2027_research.rl evaluate \
  --model data/research/sentry-mjwarp-first/model.zip \
  --seeds 7 42 91 --sides red blue --duration 420 \
  --device cuda --run-dir data/research/sentry-mjwarp-eval
research/.venv/bin/python -m rmuc2027_research.rl export \
  --model data/research/sentry-mjwarp-first/model.zip \
  --output data/research/sentry-mjwarp-policy
```

训练回放保留完整状态帧，可供原网页回放；标记 `physics_backend=mjwarp`，不能用 Rapier 的逐帧确定性验证器复算。物理引擎不同，轨迹不保证一致，实际部署能力必须以上述原引擎评估为准。

已完成的验证：18 型号场景编译、完整比赛步进、两世界采样/截断/重置、真实 PPO 参数更新、保存恢复，以及原有架构/RL/局域网回归。开发执行环境看不到 CUDA 设备，因此真实 GPU 长局稳定性、红蓝三种子完整评估与端到端加速比仍需实机验收；用户此前的 CUDA 冒烟测试只证明设备可用，不替代完整场景验收。

按墙钟时间训练一小时（完整 420 秒比赛回合）：

```bash
research/.venv/bin/python -m rmuc2027_research.rl train \
  --config configs/rl/sentry-mjwarp-hour.json \
  --device cuda --num-envs 2 --wall-seconds 3600 \
  --run-dir data/research/sentry-mjwarp-1h
```

不同时传入步数上限时，训练持续到时间上限，在下一个向量环境步完成后停止，因此结束时间可能略晚于一小时。计时包含模型及环境初始化，不包含命令启动前的依赖导入。每五分钟在采样边界保存 `latest.zip`，结束保存 `model.zip`、`training.json`；Ctrl+C 保存 `interrupted.zip`。未完成的末次 rollout 不参与梯度更新，报告中的 `actor_updated` 与 `gradient_epochs` 反映是否实际更新，不能把只采样未更新称为完成训练。

训练用固定策略的决策时限独立配置为 `policy_timeout_s`（默认 5 秒，MJWarp 配置中显式记录）。该值同时用于普通队友/对手和 script/frozen 覆盖控制器，避免并行采样时因原 0.2 秒墙钟预算误判失败。正式比赛的 `Battle` 默认时限仍为 0.2 秒。超出训练预算、策略异常或非法输出仍会使该 rollout 报错，不会静默忽略错误或以空动作继续训练。更改此预算会改变实验配置，旧检查点不能绕过兼容检查续训。

### MJWarp 学习哨兵统一 2 Hz

`sentry-mjwarp.json`、`sentry-mjwarp-hour.json` 和 `sentry-mjwarp-fast.json`
均显式设置 `rl_seconds: 0.5` 和 `learning_control_hz: 2`。
目标点、导航、瞄准、开火、补给与姿态每半秒仿真时间更新一次，中间保持指令；
导出的策略同样缓存控制指令，不因外部高频调用重新运行控制逻辑。
物理、碰撞、弹道和发射冷却保持原有频率，2 Hz 不代表每秒两发。
火控按整个指令保持周期的最大发弹数预留热量，不预支未来冷却。

一个 RL 步对应单环境 0.5 秒；两个环境累计 100 个样本，对应各环境
25 秒仿真。PPO 参数及其他兵种的固定策略频率保持各配置原值。
旧实验配置不自动迁移；更换频率后建议新建训练目录。
性能对比应采用相同仿真时长、种子和环境数，排除首次编译，查看单环境
`per_env_simulation_seconds_per_second`，而非仅比较总采样步数。

MJWarp RL 预设现在设置 `drones_enabled: false`，训练、评估和原生预览均保留机库但不生成无人机。
通用比赛及缺少该字段的旧实验默认仍启用无人机。需要重新开始实验，不能把旧模型的评估结果直接与新场景比较。
串联腿 305/2772 的中央登阶导航加入空中保持和落地稳定阶段；GPU 登阶成功率仍需实测。

### 导航轨迹实验

新导航出现持续制动回归，MJWarp RL 预设已恢复 `navigation_mode: legacy`。
`predictive` 仅保留为未验收的实验模式，不用于正常训练或预览。
该模式保留策略包与目标点，通过共享导航层恢复完整最短路、做可见性捷径处理、
按转角和决策保持时间限速，并采样局部运动候选避让观测中的机器人。
训练、评估、导出和原生预览记录并使用相同模式。学习哨兵仍为 2 Hz。
导航处于特殊跳跃、折叠或落地阶段时保留原专用控制。
当前预测使用简化加减速模型，尚未完成真实底盘制动标定、完整场景碰撞率及 GPU
吞吐对照；不能据此宣称达到碰撞减少 70%、抵达时间增幅 10% 的验收指标。

### 地面运动诊断

给原生查看器加 `--trace-dir data/research/motion-日期` 可记录该次比赛。
目录必须不存在。正常不传参数时不记录，战术策略不变。
无界面采集同一场景：

```bash
research/.venv/bin/python -m rmuc2027_research.rl.mjwarp.diagnose \
  --config configs/rl/sentry-mjwarp-hour.json --backend mjwarp --device cuda:0 \
  --seed 42 --duration 120 --trace-dir "data/research/motion-$(date +%Y%m%d-%H%M%S)"
```

输出 `metadata.json`、`motion.jsonl`、`summary.json`、`trajectories.svg`；
异常发生时另存有限长度的物理子步前置记录。`--backend rapier` 可进行参考对照。
停滞标记要求持续有移动指令且两秒位移不足 0.1 米；零指令驻守不自动判错。
所有告警仍需结合目标、导航阶段和接触数据复核。首次采集只能定位问题，
不能替代 GPU 修复验收，也不能将无异常程序退出当成全场运动正常。

### 分别训练后组合地面角色

每个训练运行只学习一个席位。可使用 `configs/rl/infantry3-mjwarp-hour.json`、
`infantry4-mjwarp-hour.json`、`heavy-mjwarp-hour.json` 和 `sentry-mjwarp-hour.json`。
重装的固定装配与武器安装前期不计入 PPO 样本，确认安装后才开始学习。

分别导出模型，再组合成一个标准 Arena squad：

```bash
research/.venv/bin/python -m rmuc2027_research.rl export \
  --model data/research/infantry-run/model.zip --output data/research/export-infantry
research/.venv/bin/python -m rmuc2027_research.rl export \
  --model data/research/sentry-run/model.zip --output data/research/export-sentry
research/.venv/bin/python -m rmuc2027_research.rl compose \
  --exports data/research/export-infantry data/research/export-sentry \
  --output data/research/export-combined
```

这些运行路径是示例，请替换成实际目录。组合包保留每个学习席位的源码与哈希；
其他席位沿用第一个包的固定策略。重复学习席位、不同阵容或不同对局设置会报错。
`rl-metadata.json` 中的 `deployment` 保存所需阵容和装配设置。网页和 CLI 对战会读取
所选导出包的要求，自动应用装配模式、概率、耗时及无人机开关；双方要求冲突或显式
参数与训练要求冲突时拒绝启动。网页后端短局已验证，浏览器画面与完整训练模型对局仍需验收。
