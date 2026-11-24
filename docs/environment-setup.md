# 环境配置指令（Arena 网页 + RL 训练）

本文只给出配置指令，不代替构建验收。默认 Ubuntu 22.04 / 24.04 x86_64。

## 0. 三套互相隔离的环境

| 环境 | 位置 | 用途 | 是否必装 torch |
| --- | --- | --- | --- |
| 主环境 | `<项目根>/.venv` | 联机主机、Arena 网页服务、策略对战、单元测试 | 否 |
| 研究环境 | `<项目根>/research/.venv` | PPO 训练（GPU）、MJWarp GPU 物理 | 是，CUDA 版 |
| 原生构建 | `<项目根>/build/client` | Qt 桌面客户端 | 否 |

三条硬规则：

1. **不要把 torch 装进主 `.venv`**。`uv sync --locked` 会按锁文件把它移除，联机环境也不该带神经网络依赖。
2. **不要用同一个解释器跑联机与训练**。文档中的 `research/.venv/bin/python` 一律不要替换成 `.venv/bin/python`。
3. **训练默认走 GPU**。CPU 只在排查观测/动作契约时临时用，见 5.9。

```bash
export RMUC_ROOT=/media/rm/E3E258F427E8E547/rmuc2027
cd "$RMUC_ROOT"
```

## 1. 系统依赖

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config curl unzip git python3.12 python3.12-venv
# Qt 客户端（用系统 Qt 时一并装；用官方 Qt SDK 可跳到第 6 节）
sudo apt install -y qt6-base-dev qt6-declarative-dev qt6-webengine-dev qt6-webengine-dev-tools \
  libqt6webenginecore6-bin libqt6webchannel6-dev libqt6websockets6-dev \
  libgl1-mesa-dev libglu1-mesa-dev
```

GPU 前置（RL 训练必需，Arena 网页与联机不需要 CUDA）：

```bash
nvidia-smi    # 记下 GPU 型号与右上角 CUDA Version，决定 torch 轮子版本
```

轮子选择规则：

| GPU 架构 | 代表型号 | 必须的 torch 轮子 |
| --- | --- | --- |
| Blackwell（sm_120） | RTX 50 系、RTX PRO 6000 | **cu128 及以上**，cu126 会报 no kernel image |
| Ada / Hopper（sm_89/90） | RTX 40 系、H100 | cu124 / cu126 / cu128 均可 |
| Ampere（sm_86） | RTX 30 系、A100 | cu118 及以上 |

轮子 CUDA 版本必须 ≤ 驱动支持的上限；驱动向下兼容，新版驱动装稍旧轮子没问题，反之不行。

## 2. 工具链

```bash
# Rust（cargo 必须在 PATH 中）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
. "$HOME/.cargo/env"

# uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Bun（package.json 要求 >= 1.3.14；CMake 用 find_program(bun REQUIRED) 找它）
curl -fsSL https://bun.sh/install | bash
export PATH="$HOME/.bun/bin:$PATH"

uv python install 3.12
rustc --version && cargo --version && uv --version && bun --version
```

## 3. 仿真引擎与前端（两套环境共用的基础）

RL 训练由 Rust 引擎计算规则与驱动、Python 负责物理与 PPO，所以**训练前必须先编译出 `target/release/rmuc-sim`**。Python 通过 `RMUC_SIM_BINARY` 或默认路径找它。

```bash
cd "$RMUC_ROOT"

uv sync --locked                     # 主环境：联机 + 网页 + 引擎绑定
cargo build --locked --release -p rmuc-sim
bun install --frozen-lockfile
bun run build                        # Qt 前端产物 -> build/web
```

检查点：

```bash
ls -l target/release/rmuc-sim
```

## 4. Arena 环境（网页对战 + 策略开发）

### 4.1 安装与构建

```bash
cd "$RMUC_ROOT"
uv sync --locked --group research    # 会补 fastapi / uvicorn / httpx / imageio-ffmpeg
bun run build:arena-web              # 产物 -> build/arena-web
```

### 4.2 启动

```bash
uv run --locked --group research rmuc2027-research serve --port 8767
# 浏览器打开 http://127.0.0.1:8767
```

`build/arena-web/index.html` 不存在时服务会直接报 `Build the web application first`，先跑 4.1。

### 4.3 可选：Docker 策略沙箱

默认 `trusted-local` 直接在本机子进程执行策略代码，只适合自己信任的代码。需要隔离时：

```bash
sudo apt install -y docker.io && sudo usermod -aG docker "$USER"   # 重新登录生效
docker build -f research/strategies/Dockerfile -t rmuc2027-policy:0.1 .
uv run --locked --group research rmuc2027-research serve --port 8767 --sandbox docker
```

### 4.4 策略开发与命令行对战

```bash
uv run --locked --group research rmuc2027-research new-strategy data/research/my-strategy
# 编辑 data/research/my-strategy/strategy.py：act(observation, state) -> (actions, new_state)
uv run --locked --group research rmuc2027-research bundle \
  data/research/my-strategy/strategy.py data/research/my-bundle
uv run --locked --group research rmuc2027-research battle \
  --red data/research/my-bundle \
  --blue research/strategies/examples/claude_opus55_xhigh_2027_team \
  --output data/research/my-match.json
```

### 4.5 可选：前端热更新

先保持 8767 服务运行，再开一个终端：

```bash
bun run dev:arena-web     # http://127.0.0.1:5174 ，/api 与 WebSocket 代理到 8767
```

Arena 相关数据目录与端口：

- 网页数据（比赛库、回放、训练任务、上传模型、录像）：`data/research/web`
- 前端产物：`build/arena-web`
- 端口：`8767` 网页服务、`5174` 前端开发、`8766` 联机主机（TCP）

## 5. RL 训练环境（GPU）

三个参数各管一段，不要混：

| 参数 | 控制什么 | 取值 |
| --- | --- | --- |
| `--device cuda` | **PPO 网络**在 GPU 上前向/反向 | 只接受 `cpu` 或 `cuda`，**不能写 `cuda:0`** |
| `--sim-device cuda:0` | **物理仿真**所在设备 | 要带序号，如 `cuda:0` |
| 配置 `sim_backend` | 物理后端 | `rapier`（CPU 原引擎）/ `mjwarp`（GPU） |

### 5.1 创建独立研究环境

```bash
cd "$RMUC_ROOT"
uv venv --python 3.12 research/.venv
uv pip install --python research/.venv/bin/python -e packages/core -e research
```

### 5.2 装 CUDA 版 PyTorch

```bash
# Blackwell（RTX 50 系）必须 cu128 起；Ada/Hopper 可用 cu126
uv pip install --python research/.venv/bin/python -r research/requirements-rl.txt --torch-backend cu128

# 不想手写版本时让 uv 按本机驱动自动选
# uv pip install --python research/.venv/bin/python -r research/requirements-rl.txt --torch-backend auto
```

校验（全部为真才继续）：

```bash
research/.venv/bin/python -c "
import torch
print('torch     ', torch.__version__)
print('cuda build', torch.version.cuda)
print('available ', torch.cuda.is_available())
print('device    ', torch.cuda.get_device_name(0))
print('capability', torch.cuda.get_device_capability(0))
x = torch.ones((256,256), device='cuda'); print('matmul    ', float((x@x).sum()))
"
```

`available` 为 False 时不要往下走：容器 / WSL 里先解决 `/dev/nvidia*` 设备可见性，装 CUDA 版 torch 不会解除设备隔离。

### 5.3 装 MJWarp GPU 物理

```bash
uv pip install --python research/.venv/bin/python -r research/requirements-mjwarp.txt
cargo build --release --locked -p rmuc-sim    # 若第 3 节已构建可跳过
```

`requirements-mjwarp.txt` 里的 `numpy==2.2.6` 与 `requirements-rl.txt` 一致，可直接叠加。首次运行含 JIT 编译，分钟级耗时正常。

### 5.4 GPU 准入检查

```bash
research/.venv/bin/python -m rmuc2027_research.rl.mjwarp.doctor \
  --output "data/research/gpu-admission-$(date +%Y%m%d-%H%M%S).json"
```

检查项：驱动、CUDA 张量计算、两世界 MJWarp 步进、CUDA Graph 与普通 GPU 步进结果一致性。每个探针默认限时 300 秒（含首次编译）。**退出码非 0 就不要开始训练**，先看输出文件里的失败阶段。

### 5.5 启动 GPU 训练

两环境最小验证（先跑这个，确认整条链路通）：

```bash
research/.venv/bin/python -m rmuc2027_research rl train \
  --config configs/rl/sentry-mjwarp.json \
  --device cuda --sim-device cuda:0 \
  --num-envs 2 --total-steps 256 \
  --run-dir data/research/sentry-gpu-smoke
```

按墙钟时间训练一小时（完整 420 秒回合）：

```bash
research/.venv/bin/python -m rmuc2027_research rl train \
  --config configs/rl/sentry-mjwarp-hour.json \
  --device cuda --sim-device cuda:0 \
  --num-envs 2 --wall-seconds 3600 \
  --run-dir data/research/sentry-gpu-1h
```

扩并行前先测吞吐，不要凭显存猜：

```bash
research/.venv/bin/python -m rmuc2027_research rl benchmark \
  --config configs/rl/sentry-mjwarp.json \
  --num-envs 1 2 4 8 --steps 16 \
  --run-dir data/research/bench-gpu
```

看 `per_env_simulation_seconds_per_second`：小于 1 表示单环境还没跑赢现实时间，加环境数只提升总吞吐、不改善单局速度。`recommended_num_envs` 为 `null` 说明没有达标组合。

显存参考（16 GB 级，如 RTX 5070 Ti）：先 `--num-envs 2`，稳定后试 4；`--num-envs 8` 需要更大显存。

### 5.6 训练产物与中途干预

运行目录内：

- `model.zip` / `latest.zip`：最终与每五分钟检查点
- `updates.jsonl`：每轮 PPO 的损失、KL、学习率、移动指令统计
- `episodes.jsonl`：完整回合胜负、占点、死亡
- `experiment.json`：配置、规则、源码与依赖哈希，续训严格校验
- `training.json`：结束汇总；`Ctrl+C` 额外保存 `interrupted.zip`

需要人工调学习率时，往运行目录写 `learning-rate.json`（先写临时文件再改名，避免读到半个 JSON）：

```bash
printf '{"learning_rate": 0.0001, "reason": "说明调整依据"}' > data/research/sentry-gpu-1h/.lr.tmp
mv data/research/sentry-gpu-1h/.lr.tmp data/research/sentry-gpu-1h/learning-rate.json
```

请求在完整 rollout 之后、下次梯度更新之前生效，实际变更记入 `learning-rate-events.jsonl`。

续训：

```bash
research/.venv/bin/python -m rmuc2027_research rl train \
  --config configs/rl/sentry-mjwarp-hour.json \
  --resume data/research/sentry-gpu-1h/latest.zip \
  --device cuda --sim-device cuda:0 --num-envs 2 \
  --run-dir data/research/sentry-gpu-1h-resume
```

续训会核对配置、规则、策略、仿真源码与依赖版本；换后端必须开新实验，不能把旧优化器断点当同一实验继续。

### 5.7 GPU 训练后在原引擎评估、导出、组合

训练用 MJWarp，部署与验收走原 Rapier 引擎，这一步不能省：

```bash
research/.venv/bin/python -m rmuc2027_research rl evaluate \
  --model data/research/sentry-gpu-1h/model.zip \
  --seeds 7 42 91 --sides red blue --duration 420 --device cuda \
  --backend rapier --run-dir data/research/sentry-gpu-1h-eval

research/.venv/bin/python -m rmuc2027_research rl export \
  --model data/research/sentry-gpu-1h/model.zip --output data/research/sentry-policy

research/.venv/bin/python -m rmuc2027_research rl compose \
  --exports data/research/export-infantry data/research/export-sentry \
  --output data/research/export-combined
```

`evaluate` 默认按训练配置里的 `sim_backend` 评估，所以 MJWarp 训练的模型默认就还在 MJWarp 里评；要回到原引擎验收必须显式写 `--backend rapier`，如上。跨后端对比时请分别 `--run-dir`，别把两个后端的胜率混在一张表里比。

各席位分开训练用 `configs/rl/infantry3-mjwarp-hour.json`、`infantry4-mjwarp-hour.json`、`heavy-mjwarp-hour.json`、`sentry-mjwarp-hour.json`，每次运行只学一个席位。

### 5.8 GPU 原生窗口观战

需要桌面会话（`DISPLAY` 或 `WAYLAND_DISPLAY`）：

```bash
research/.venv/bin/python -m rmuc2027_research.rl.mjwarp.viewer \
  --config configs/rl/sentry-mjwarp-hour.json --device cuda:0 --duration 420
```

带训练后的哨兵（只替换红方 7 号，其余沿用固定策略）：

```bash
research/.venv/bin/python -m rmuc2027_research.rl.mjwarp.viewer \
  --rl-model data/research/sentry-gpu-1h/model.zip \
  --rl-team red --blue examples/claude_opus55_xhigh_2027_hold \
  --device cuda:0 --seed 42 --duration 420
```

### 5.9 仅调试：CPU 路径

只在排查观测/动作契约时用，不代表训练性能：

```bash
research/.venv/bin/python -m rmuc2027_research rl check \
  --config configs/rl/sentry.json --run-dir data/research/new-check
research/.venv/bin/python -m rmuc2027_research rl train \
  --config configs/rl/sentry.json --device cpu --sim-device cpu \
  --total-steps 256 --num-envs 1 --run-dir data/research/new-train-cpu
```

请求 CUDA 时不会自动回退 CPU，反过来也一样。

## 6. Qt 桌面客户端（可选，只有联机玩法需要）

### 6.1 用系统 Qt（Ubuntu）

```bash
cmake --preset release -DCMAKE_PREFIX_PATH= -UQt6_DIR
cmake --build --preset release
ctest --preset release
./scripts/run-client.sh
```

### 6.2 用官方 Qt SDK

```bash
# 假设装在 $HOME/Qt/6.8.3/gcc_64
cmake --preset release -DCMAKE_PREFIX_PATH="$HOME/Qt/6.8.3/gcc_64"
cmake --build --preset release
./scripts/run-client.sh
```

CMake 需要 `Core Gui Widgets Network Quick WebEngineWidgets WebChannel WebSockets`（测试再加 `Test`），还会 `find_program(bun REQUIRED)`——bun 不在 PATH 时配置阶段就失败，而不是编译阶段。

## 7. 自检清单

```bash
uv run --locked --group research --group test python -m unittest discover -s tests/python -v
cargo test --locked --release -p rmuc-sim
bun test tests/web
ctest --preset release
```

| 目标 | 通过标准 |
| --- | --- |
| 引擎 | `target/release/rmuc-sim` 存在，`cargo test -p rmuc-sim` 全绿 |
| Arena 网页 | 8767 打开首页、能跑一局自由对战、`build/arena-web/index.html` 存在 |
| torch GPU | `torch.cuda.is_available()` 为 True，capability 与显卡架构匹配 |
| MJWarp | `doctor` 退出码 0 |
| GPU 训练 | 2 环境 256 步跑完，写出 `model.zip` 与 `updates.jsonl`，actor 权重有实际更新 |
| Qt | `build/client/bin/RMUC2027Client` 存在，`ctest --preset release` 通过 |

## 8. 环境变量

| 变量 | 作用 |
| --- | --- |
| `RMUC_SIM_BINARY` | 指到 `rmuc-sim` 可执行文件，默认 `<项目根>/target/release/rmuc-sim` |
| `RMUC_STRATEGY_UPDATES=0` | 关闭网页策略库的自动更新扫描 |
| `RMUC_DESKTOP_INSTANCE` | 网页服务上报的实例标识，用于区分桌面端 |
| `QT_QPA_PLATFORM=offscreen` / `QT_QUICK_BACKEND=software` | 无显示环境跑 Qt 测试（`ctest` 已自动设置） |
| `ARENA_RENDER_TIMING` | 打印操作手渲染缓冲统计，排查图传抖动 |
| `CUBLAS_WORKSPACE_CONFIG` | 训练代码自动设为 `:4096:8`，不要覆盖 |
| `CUDA_VISIBLE_DEVICES` | 多卡机器上限定训练用哪张卡 |

## 9. 常见问题

- **`no kernel image is available for execution on the device`**：torch 轮子不支持当前架构，RTX 50 系换 `--torch-backend cu128`（或更高）重装。
- **`CUDA driver version is insufficient for CUDA runtime version`**：轮子的 CUDA 版本高于驱动，降一档。
- **`doctor` 里 CUDA unavailable、宿主机却能跑**：执行环境没有 `/dev/nvidia*`，属设备隔离，不是依赖问题。
- **`--device cuda:0` 报 invalid choice**：`--device` 只接受 `cpu`/`cuda`；设备序号写在 `--sim-device`。
- **GPU 利用率低但训练慢**：先跑 `benchmark` 看 `per_env_simulation_seconds_per_second`；CPU 侧规则计算与 Python 通信仍是瓶颈，进程协议尚未换 PyO3 数组绑定。
- **`CUDA out of memory`**：降 `--num-envs`，或换更小的车型组合配置。
- **`Build the web application first: bun run build:arena-web`**：Arena 服务只托管静态产物，先构建。
- **CMake 找不到 Qt6**：`-DCMAKE_PREFIX_PATH` 指错，或从 SDK 切回系统 Qt 时残留缓存，加 `-UQt6_DIR` 重配。
- **`find_program(bun REQUIRED)` 失败**：`~/.bun/bin` 没进 PATH；CMake GUI/IDE 里要用带 PATH 的 shell 启动。
- **`uv sync` 后 torch 消失**：属正常，训练依赖只能待在 `research/.venv`。
- **训练启动即报目录已存在**：所有 `--run-dir` / `--output` 都要求新目录，换名而不是清空旧实验。
- **`bun test` 报 Chromium 权限错误**：多是无头 Chrome 的沙箱限制，不影响构建与联机，真实画面要到桌面环境验收。

## 10. 一次性串起来的脚本（GPU 版）

```bash
set -euo pipefail
cd "$RMUC_ROOT"
uv python install 3.12
uv sync --locked --group research --group test
cargo build --locked --release -p rmuc-sim
bun install --frozen-lockfile
bun run build
bun run build:arena-web

# --- RL：GPU ---
uv venv --python 3.12 research/.venv
uv pip install --python research/.venv/bin/python -e packages/core -e research
uv pip install --python research/.venv/bin/python -r research/requirements-rl.txt --torch-backend cu128
uv pip install --python research/.venv/bin/python -r research/requirements-mjwarp.txt

research/.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用'; print(torch.cuda.get_device_name(0))"
research/.venv/bin/python -m rmuc2027_research.rl.mjwarp.doctor \
  --output "data/research/gpu-admission-$(date +%Y%m%d-%H%M%S).json"

research/.venv/bin/python -m rmuc2027_research rl train \
  --config configs/rl/sentry-mjwarp.json \
  --device cuda --sim-device cuda:0 --num-envs 2 --total-steps 256 \
  --run-dir data/research/sentry-gpu-smoke
```

Qt 客户端构建不放进脚本，因为 `CMAKE_PREFIX_PATH` 必须按本机 Qt 安装位置填。
