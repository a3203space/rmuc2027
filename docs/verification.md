# 迁移验证状态

日期：2026-10-22。以下区分代码与进程内验证、实际桌面与物理局域网验证。

| 项目 | 证据 |
| --- | --- |
| 2027 引擎 | 清理后 49 项 Rust 规则、机构、物理、伤害和控制点回归通过；release 引擎构建通过 |
| 架构边界 | 6 项检查通过：单赛季配置、合法素材、默认依赖、应用不依赖研究、客户端无旧通信栈、旧目录已移除 |
| Python 功能 | 33 项检查通过，默认环境跳过 1 项 Gymnasium 专属项；研究环境另行完整通过 8 项 RL 边界用例 |
| 回放分页 | 3 项检查通过，覆盖 UTF-8/转义/跨块偏移、只读接口、分页上限与错误输入；历史完整回放成功索引 8401 帧，读取首尾 tick 为 0/8400 |
| 新布局十人短局 | `data/replays/layout-check/report.json`：10 个输入源、4 秒/80 tick，回放逐帧校验通过；属于进程内负载，不是物理局域网 |
| 模型兼容 | 十组原始观测对应的 116 维输入完全一致；两个旧模型迁移后权重哈希不变，实际预测、短评估和策略导出通过 |
| 神经网络训练 | 64 步 CPU 训练与追加 64 步恢复训练通过，实际优化、权重更新及优化器恢复有记录；双采样进程与 SIGINT 中断检查均无残留子进程，检查点保存成功。见 `data/research/layout-train` 和 `layout-resume` |
| 前端 | TypeScript 检查与 Vite 生产构建通过 |
| QML | 当前 qmllint 语法检查通过；此工具不是 Qt 6 运行验收 |
| 主机运行包 | `build/runtime` 通过运行前检查、文件哈希及监听/父进程退出检查；不携带研究代码或训练依赖 |
| 原始数据 | 迁移前 72 个数据文件核对哈希，删除的旧平台状态另有备份记录 |
| Qt 原生客户端 | `cmake --preset release` 在查找 Qt6 时停止，缺少 Qt6Config.cmake；6 个客户端用例尚未编译执行 |
| Windows/真实双机/GPU | 尚未验证，不能声称完整交付验收完成 |

当前环境缺少 Qt 6 SDK。主机可以监听 TCP，但客户端连接和 Chromium IPC 受沙箱限制，尚不能在这里证明实际网络传输或画面显示。输入执行确认耗时不等于输入到可见响应。

## 在目标环境复核

```bash
uv sync --locked --group research --group test
uv run --locked --group research --group test python -m unittest discover -s tests/python -v
cargo test --locked --release -p rmuc-sim
bun install --frozen-lockfile
bun run build
cmake --preset release -DCMAKE_PREFIX_PATH=本机Qt桌面SDK目录
cmake --build --preset release
ctest --preset release
```

完成 Windows/Linux 原生构建后，实际运行两机建房、选位、命中、补给、复活、重装辅助、机库指令、断线接管和重连；检查回放显示、720p/1080p 及 Windows 125%/150% 缩放。完成这些检查前，原生与双机验收仍为待完成。
