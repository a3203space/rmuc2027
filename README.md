# RMUC 2027

RMUC 2027 多人局域网仿真。房主运行 Rust 仿真，每名玩家通过 Qt 客户端控制一台机器人；空位由固定 AI 补齐。默认阵容为重装、两台步兵、机库和哨兵，5 对 5，比赛 420 秒。

主应用只包含联机游玩。策略对战与哨兵强化学习放在独立的 `research/`，不随联机运行包发布。

## 构建与启动

需要 Python 3.12、uv、Rust、Bun，以及包含 Quick/QuickControls2、WebEngineWidgets、WebChannel、WebSockets 的 Qt 6 桌面 SDK。

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

Linux 启动客户端：

```bash
./scripts/run-client.sh
```

Windows 客户端位于 `build/client/bin/Release/RMUC2027Client.exe`（Visual Studio）或 `build/client/bin/RMUC2027Client.exe`（Ninja）。Windows 部署脚本：

```powershell
powershell -File scripts/build-client.ps1 -QtPrefix C:\Qt\6.8.3\msvc2022_64 -Output build\client-dist
```

客户端中将“仿真目录”设为本项目根目录，点击“创建房间”。其他玩家填写房主局域网 IPv4 地址，默认端口 `8766`，选位并准备后由房主开始。

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
assets/     实际使用的三维模型
configs/    2027 规则与实验配置
scripts/    构建、启动、打包和检查工具
tests/      联机、研究边界和客户端用例
data/       已有回放、模型、实验记录；不进入发行包
```

当前代码仍需在具备 Qt 6 SDK 的环境完成原生构建与真实双机验收，详见验证记录。
