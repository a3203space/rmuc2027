# RMUC 2027

RMUC 2027 多人局域网仿真。房主运行 Rust 仿真，每名玩家通过 Qt 客户端控制一台机器人；空位由所选 AI 策略补齐。默认阵容为重装、两台步兵、机库和哨兵，5 对 5，比赛 420 秒。

客户端采用 Arena 风格的大厅、三维机库、策略库、同步入场介绍、第一视角 HUD 和结算。18 种合法型号可按席位独立选择；内置 94 个策略包和通用基线，房主还可导入本地策略。

主应用只包含联机游玩。策略对战与哨兵强化学习放在独立的 `research/`，不随联机运行包发布。

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

开局后中间视口使用云台第一视角，保留原版左右侧栏；点击视口右上“全屏”进入沉浸操控。画面显示真实枪管、左下血量、原版小地图、蜂窝比分条及模型卡片，四个控制位置保留简洁标记。图传与界面按 16:9 等比适配。左键点按单发、按住连发；**按住右键自瞄，松开取消**，左键独立开火。步兵与重装按规则零弹开局，按 **B** 手动购弹。按 **Esc** 打开角色操作菜单，按住 **Tab** 展开战术总览。战亡后保留实时灰度摄像头画面，HUD 保持彩色，复活后恢复；全场击杀公屏使用 Arena-1 原版首杀、连杀及团灭播报音与动画。原生构建会自动构建并复制前端资源。更新后需重启房主与客户端；已部署的主机包也需同步更新。本版使用联机协议 v2，并检查客户端与主机界面资源版本。

也可独立运行主机：

```bash
uv run --locked rmuc2027 host --host 0.0.0.0 --port 8766
```

独立主机输出的 `owner_key` 仅由房主填入客户端。放行主机 TCP 8766 即可；无需 MQTT、视频 UDP 或实机裁判系统。

## 文档

- [联机操作、主机打包与部署](docs/lan-play.md)
- [AI 策略对战和 RL 训练](docs/research.md)
- [目录、迁移与已有成果](docs/migration.md)
- [验证状态与待完成项](docs/verification.md)
- [来源和许可证](docs/legal/README.md)

## 目录

```text
apps/       Qt 客户端、联机主机、本地回放查看器
packages/   Rust 仿真、Python SDK/策略运行、共享三维渲染
research/   策略对战与 RL；依赖按需安装
assets/     2027 三维模型、Arena 音频和客户端准星
configs/    2027 规则与实验配置
scripts/    构建、启动、打包和检查工具
tests/      联机、研究边界和客户端用例
data/       已有回放、模型、实验记录；不进入发行包
```

Qt 6 原生构建已通过；实际桌面显示与真实双机验收仍需在目标环境完成，详见验证记录。
