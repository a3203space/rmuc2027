# RMUC 2027

RMUC 2027 多人局域网仿真。房主运行 Rust 仿真，每名玩家通过 Qt 客户端控制一台机器人；空位由所选 AI 策略补齐。默认阵容为重装、两台步兵、机库和哨兵，5 对 5，比赛 420 秒。

客户端采用 Arena 风格的大厅、三维机库、策略库、同步入场介绍、第一视角 HUD 和结算。18 种合法型号可按席位独立选择；内置 94 个策略包和通用基线，房主还可导入本地策略。

Qt 客户端提供局域网联机游玩；本项目另提供直接移植 Arena-1 的完整网页，可配置 AI 阵容与策略并实时观战。网页服务、策略研究与哨兵强化学习放在独立的 `research/`，不随联机运行包发布。

## 构建与启动

需要 Python 3.12、uv、Rust、Bun，以及包含 Quick、WebEngineWidgets、WebChannel、WebSockets 的 Qt 6 桌面 SDK。

在项目根目录执行：

```bash
uv python install 3.12
uv sync --locked
cargo build --locked --release -p rmuc-sim
bun install --frozen-lockfile
bun run build

# Qt 路径改为本机实际安装位置
cmake --preset release -DCMAKE_PREFIX_PATH="$HOME/Qt/6.8.3/gcc_64"
cmake --build --preset release
```

### Ubuntu 22.04 安装 Qt

上面的 Qt SDK 路径是示例，必须指向实际安装目录。Ubuntu 22.04 也可通过系统包安装 Qt 6 开发库与 WebEngine 运行组件：

```bash
sudo apt update
sudo apt install build-essential cmake \
  qt6-base-dev qt6-declarative-dev qt6-webengine-dev qt6-webengine-dev-tools \
  libqt6webenginecore6-bin \
  libqt6webchannel6-dev libqt6websockets6-dev

# 使用系统 Qt，不填写示例 SDK 路径；清除之前配置缓存中的路径
cmake --preset release -DCMAKE_PREFIX_PATH= -UQt6_DIR && \
cmake --build --preset release
```

CMake 配置成功后才能编译；编译成功后才能启动。前端构建时的 `Some chunks are larger than 500 kB` 是体积警告，不会导致 Qt 查找失败。

### 启动客户端

Linux 启动客户端：

```bash
./scripts/run-client.sh
```

Windows 客户端位于 `build/client/bin/Release/RMUC2027Client.exe`（Visual Studio）或 `build/client/bin/RMUC2027Client.exe`（Ninja）。Windows 部署脚本：

```powershell
powershell -File scripts/build-client.ps1 -QtPrefix C:\Qt\6.8.3\msvc2022_64 -Output build\client-dist
```

客户端使用直接移植的 Arena-1 首页、对战三栏、广播 HUD、机库和入场/结算组件。点击“准备正式对战”或顶部“RM 对战”，将“仿真工程或运行包目录”设为本项目根目录或 `build/runtime/app`，点击“创建房间”。其他玩家填写房主局域网 IPv4 地址，默认端口 `8766`。在对战页左栏选位、“机器人机库”选型、“策略库”选 AI。自己的 3／4 号步兵可在机库下方分别点击血量／功率、冷却／爆发按钮，等“已保存”后重新准备，再由房主开始。所有客户端完成模型与首帧加载后，同步播放约 38.3 秒原版入场介绍及 5 秒倒计时，再开放操作。房主可以跳过介绍，倒计时始终保留。

开局后中间视口使用云台第一视角，保留原版左右侧栏；点击视口右上“全屏”进入沉浸操控。画面显示真实枪管、左下血量、原版小地图、蜂窝比分条及模型卡片，四个控制位置保留简洁标记。图传与界面按 16:9 等比适配。地面兵种统一使用相对图传视角的 WASD，轮腿自动转向或倒车后行驶。左键点按单发、按住连发；**按住右键自瞄，松开取消**，左键独立开火。步兵与重装按规则零弹开局，按 **B** 手动购弹。按 **Esc** 打开角色操作菜单，按住 **Tab** 展开战术总览。战亡后保留实时灰度摄像头画面，HUD 保持彩色，复活后恢复；全场击杀公屏使用 Arena-1 原版首杀、连杀及团灭播报音与动画。原生构建会自动构建并复制前端资源。更新后需重启房主与客户端；已部署的主机包也需同步更新。本版使用联机协议 v2，并检查客户端与主机界面资源版本。

也可独立运行主机：

```bash
uv run --locked rmuc2027 host --host 0.0.0.0 --port 8766
```

独立主机输出的 `owner_key` 仅由房主填入客户端。放行主机 TCP 8766 即可；无需 MQTT、视频 UDP 或实机裁判系统。

## 完整网页：选择兵种与策略、实时观看 AI 对战

网页版直接移植 Arena-1 的原版页面和样式，在 **rmuc2027 项目内独立运行**，使用本项目当前的 Rust 引擎、2027 规则、18 种合法机器人和 94 个内置策略。包含首页、正式对战、自由对战、策略训练、策略库、比赛与回放、排行榜、战场设置、机器人外观上传及 MP4 / WebM 录像。

### 首次构建

需要 Python 3.12、uv、Rust / Cargo、Node.js 22+ 和 Bun 1.3.14+，无需 Qt。在 **rmuc2027 项目根目录**执行：

```bash
uv python install 3.12
uv sync --locked --group research
cargo build --locked --release -p rmuc-sim
bun install --frozen-lockfile
bun run build:arena-web
```

### 启动完整网页

```bash
cd ~/rmuc2027
uv run --locked --group research rmuc2027-research serve --port 8767
```

浏览器打开 **http://127.0.0.1:8767**。保持终端运行，按 `Ctrl+C` 停止。已经构建完成时只需执行启动命令；修改网页后重新运行 `bun run build:arena-web`，修改仿真后重新编译 Rust 并重启服务。

默认使用 `trusted-local` 运行内置或自己信任的策略。需要 Docker 隔离时先构建策略镜像，再添加 `--sandbox docker`，详见 [研究工具说明](docs/research.md)。服务仅监听本机；本入口不需要启动 Qt 客户端或另开联机主机。

### 网页操作

1. **RM 对战**：点击“准备正式对战”，在“调整共同阵容 / 开始策略开发”中选择兵种型号；选择红蓝双方策略，也可通过“浏览全部策略 / 并排比较与配对”载入匹配阵容。正式赛要求双方策略注册阵容一致。
2. **自由对战**：点击“配置机器人与策略”，分别添加红蓝双方机器人、选择型号和对应单机策略；支持独立策略或全队策略控制。
3. **开始对战**：实时播放原版入场介绍、倒计时和比赛。可切换自由、俯视、跟随、环绕镜头，选择跟随机器人、暂停、调整速度或停止比赛；双方血量、小地图和比赛数据随仿真实时更新。
4. **策略训练与策略库**：创建全队、单机、多机协同或自由适配任务，将任务路径交给开发者或 Agent；在生成的工作区使用 `arena.py` 练习和提交。也可通过策略上传界面保存代码及打法说明。
5. **比赛与回放 / 排行榜**：比赛自动保存记录和 JSON 回放；可重新播放、导出或删除。完成的有效正式对战按策略版本累计 Elo；自由、短局、停止、策略错误或自定义参数的测试赛不计排名。
6. **外观与录像**：上传 GLB 后在“机器人外观”中选择，仅替换显示模型。开赛前勾选“保存完整对局 MP4”，或手动录制 MP4 / WebM；自动录制需保持观战页面打开至保存完成。
7. **战场设置**：调整下一场比赛参数，或恢复默认、导出 CLI 参数文件。

网页构建输出位于 `build/arena-web`；比赛数据库、回放、训练任务、上传模型和录像保存在 `data/research/web`，不读取 Arena-1 原项目的数据。网页训练是策略开发与练习入口，PPO 强化学习仍使用独立研究命令。更多命令见 [AI 策略对战和 RL 训练](docs/research.md)。

## 文档

- [联机操作、主机打包与部署](docs/lan-play.md)
- [AI 策略对战和 RL 训练](docs/research.md)
- [目录、迁移与已有成果](docs/migration.md)
- [验证状态与待完成项](docs/verification.md)
- [来源和许可证](docs/legal/README.md)

## 目录

```text
apps/       Qt 客户端、完整 Arena 网页、联机主机、本地回放查看器
packages/   Rust 仿真、Python SDK/策略运行、共享三维渲染
research/   策略对战与 RL；依赖按需安装
assets/     2027 三维模型、Arena 音频和客户端准星
configs/    2027 规则与实验配置
scripts/    构建、启动、打包和检查工具
tests/      联机、研究边界和客户端用例
data/       已有回放、模型、实验记录；不进入发行包
```

Qt 6 原生构建已通过；实际桌面显示与真实双机验收仍需在目标环境完成，详见验证记录。

### 哨兵 GPU 仿真训练

MJWarp 完整 5v5 训练配置为 `configs/rl/sentry-mjwarp.json`，保留现有目标点策略接口。启动方法、CUDA 依赖、并行基准和原引擎评估见 [研究说明](docs/research.md#mjwarp-完整-5v5-哨兵训练场景)。
