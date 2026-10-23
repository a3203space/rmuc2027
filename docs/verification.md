# 验证状态

## 2026-10-23：参考视频的操作手第一视角

操作手 HUD 改为顶部紧凑双方比分条与横排机器人卡片、左下血量、右下地图、中央细线瞄准框及独立武器读数。图传和 HUD 使用同一个等比适配的 16:9 区域。相机安装点随云台俯仰、转向和车体姿态更新；本机只显示真实发射机构，重装相机外壳单独隐藏，未安装机构时不虚构枪管。自己的主机确认射击事件用于后坐与枪口效果，不透传其他机器人的全局事件。

| 本次检查 | 结果与证据 |
| --- | --- |
| 前端与模型 | `bun test tests/web` 46 项、2630 个断言通过。新增七种真实 GLB 地面模型在批处理后、多种转向/俯仰/车体倾斜、折叠或下蹲及重装安装前后的射线检查；验证准星中心无遮挡、已安装枪管在画面下部且保留深度测试。包含 720p、1080p、超宽屏、较方屏幕的同区域适配检查；不等同于 GPU 像素验收 |
| Rust | `cargo test --locked --release -p rmuc-sim --lib` 53 项通过；新增相机偏移与枪轴正交、随俯仰/车体姿态运动、折叠高度与视场角保持检查；武器读数与实际发射参数一致 |
| Python | 五组共 50 项通过：`test_operator_hud.py` 10、`test_combat_controls.py` 19、`test_performance.py` 6、`test_arena_v2.py` 9、`test_architecture.py` 6；包含自己的射击事件过滤、弹速/射频、信息边界及原有购弹/自瞄/性能选择回归 |
| 构建与 Qt 状态 | Release 仿真、TypeScript/Vite 和 `cmake --build --preset release` 成功；`ctest --preset release -R '^Arena_state$'` 通过。Vite 的公共 Three.js 包仍有大于 500 kB 的提示 |
| 运行包 | 已同步前端、主机代码与引擎，校验 `build/runtime` 691 个文件哈希，使用运行包自己的 Python/引擎完成短局快照检查。`build/web`、原生客户端和主机包的资源版本均为 `20ca705f6e7d8062ee7a` |
| 实际画面限制 | 尝试无头 Chrome 截图时仍在启动阶段因 Crashpad `setsockopt: 不允许的操作` 退出，未得到有效截图；没有将 DOM/模型射线检查记为 Qt 实际图传或 GPU 通过 |

桌面检查脚本现覆盖红蓝 3／4 号、14 种场景、720p/1080p、125%/150% 缩放、超宽屏和较方屏幕；检查地图/血量位置、横排卡片、准星居中和 Tab 面板边界。实际 Qt 桌面显示、与视频的像素观感对照、连续运行及物理双机仍需在目标环境验收。

## 2026-10-23：步兵性能选择按钮与保存反馈

修复自己的 3／4 号步兵开局前性能控件无法选择的问题：机库采用血量／功率、冷却／爆发两组页面内按钮，明确显示只读原因与返回本人席位入口；加入保存中、保存成功、失败恢复及超时处理。保存期间禁止准备、开始、换席位和修改配置。较旧的房间快照不会覆盖已经确认的选择；入场补齐性能名称与冷却速率，HUD 显示实际冷却速率。

| 本次检查 | 结果与证据 |
| --- | --- |
| 性能后端 | `test_performance.py` 6 项通过；包含 3／4 号各四种组合的 8 局真实引擎和回放首帧、四套入场参数且 tick/time=0、修改后取消准备、切型号／重连保留、其他玩家隔离、对局阶段锁定及非法参数拒绝 |
| 前端 | `bun test tests/web` 35 项、175 个断言通过；新增两组按钮及完整提交参数、双方席位权限／断线说明、保存与重复提交、匹配回执、失败恢复、旧快照防回退、入场模式名称和实际参数。属于逻辑与 DOM 检查，不代表真实点击／像素验收 |
| 既有回归 | HUD 后端 8 项、右键自瞄／射击控制 19 项、Arena 流程 9 项、架构 6 项通过 |
| 构建与 Qt 桥接 | TypeScript/Vite、Qt 6.2.4 release 构建通过；`Arena_state` 3 个业务用例通过（含初始化和清理共 5 项），新增无会话保存的失败回执。公共包仍有 500 kB 体积警告 |
| Qt HTTP 回执 | 新增成功与拒绝两个用例，因沙箱禁止 TCP 监听而明确跳过，不能计为业务通过；请求关联 ID 仅用于本地桥接，发送前移除，现有 HTTP Profile 接口不变 |
| 运行包 | 已同步 `build/runtime/app` 与客户端旁 `bin/web`；运行包真实引擎性能预检通过，693 项资源哈希匹配；三处前端版本一致：`c20a7816abe4ea213fe4` |
| 桌面／点击验收 | `scripts/check_operator_ui.cjs` 新增性能点击、保存时禁止准备、失败回退、型号切换保留和按钮遮挡检查，覆盖 720p／1080p 与 125%／150% 缩放；本次 Chromium 在启动时因 `setsockopt: Operation not permitted`／SIGTRAP 退出，报告为 `passed:false`、0 张截图，尚待真实 Qt 桌面与双机验收 |

复现性能检查：

```bash
.venv/bin/python -m unittest discover -s tests/python -p test_performance.py -v
bun test tests/web
cmake --build --preset release
ctest --test-dir build/client -R '^Arena_state$' --output-on-failure
```

在可运行桌面的机器上执行 `node scripts/check_operator_ui.cjs`；成功后仍需使用真实 Qt 客户端，加入自己的步兵席位、保存四种性能组合并实际开局核对。更新后应完全退出旧客户端与房主进程再启动。


## 2026-10-23：右键按住自瞄、左键射击与手动购弹

已完成右键按住跟踪/松开取消、左键短按事件保存和按住连发、B 购弹面板及主机武器受阻提示。参考来源与数据流见 [自瞄设计](aim-assist.md)，操作见 [联机说明](lan-play.md)。

| 本次检查 | 结果与证据 |
| --- | --- |
| 新增后端控制检查 | `test_combat_controls.py` 19 项通过：真实引擎短按一发/持续连发/停止、重复包与短按保留、8 次队列上限、双玩家输入隔离、失活与重连清空、右键跟踪及松开接续、角度跨界、15°/20° 范围、遮挡丢失/重获、仅使用本席位光学视野、目标提前量、辅助不自动开火、复活 RPC 保留、菜单恢复视角基准、武器原因与己方配送事件 |
| Rust 仿真回归 | `cargo test --locked --offline -p rmuc-sim --lib` 51 项通过；新增检查实际射击与 HUD 共用判定，以及区域购弹扣款、远程配送仅到账一次 |
| 既有后端 | `test_operator_hud.py` 8 项、`test_arena_v2.py` 9 项、`test_architecture.py` 6 项通过 |
| 前端输入与 HUD | `bun test tests/web` 23 项、115 个断言通过，涵盖左右键独立、短按计数、释放清理、购弹面板、受阻和配送提示、右键松开立即隐藏旧目标。属于逻辑和服务端 DOM 检查，不代表桌面输入或像素验收 |
| 构建和 Qt 桥接 | Rust release、TypeScript/Vite、`cmake --build --preset release` 成功；`Arena_state` 通过。公共 Three.js 包仍有 500 kB 体积警告 |
| 本地运行包 | `build/runtime/app` 已同步服务器、引擎及前端；运行包预检通过，693 项资源哈希匹配。客户端、工程和运行包界面版本一致：`c3d5ef7e6a180d9d31cf` |
| HTTP 回放回归 | `test_replay_archive.py` 未能导入：当前环境的 Starlette TestClient 要求 `httpx2`，该包未安装；不能计为通过 |
| 真实桌面与双机 | 本轮未完成真实 Qt 鼠标捕获、GPU 图传、两台电脑及 Windows 操作验收；此前环境限制见下文 |

复现新增检查：

```bash
.venv/bin/python -m unittest discover -s tests/python -p test_combat_controls.py -v
bun test tests/web
cargo test --locked --offline -p rmuc-sim --lib
```


## 2026-10-23：Arena 界面、选型、策略与入场流程

已实现本地 Arena 首页、对战大厅、原版动态模型预览、机库选型、策略库、入场介绍、5 秒倒计时、操作手 HUD、胜负/统计/MVP 页面以及回放入场。所有客户端须使用协议 v2；加入时检查界面资源指纹，指纹计算统一换行与排序以支持 Windows/Linux 互联。主机包与客户端旁的 `web/` 一同更新。

| 本次检查 | 结果与证据 |
| --- | --- |
| 席位、策略与入场后端 | `test_arena_v2.py` 9 项通过，覆盖配置权限、准备状态重置、非对称型号、兼容性拒绝、本地不可变导入、普通玩家禁止导入、加载期间 tick/time=0、断线退回大厅、房主跳过与保留倒计时、折叠指令实际进入引擎、单机/全队策略真实执行及版本写入回放 |
| 原操作手 HUD 后端 | `test_operator_hud.py` 8 项通过；奇偶帧 RPC、双席位短局、角色操作、投影与信息边界保留 |
| 前端行为 | `bun test tests/web` 14 项、84 个断言通过；包含离线首页、型号适配、十台逐一介绍、主机镜头进度、跳过权限与结算入口；这是服务端 DOM 检查，不代表像素验收 |
| 架构与素材 | 6 项架构检查通过；客户端与主机运行时不导入研究模块；18 种合法模型、94 个策略包及基线可用；Arena 音频记录来源和哈希 |
| 构建 | TypeScript/Vite 与 Qt 6.2.4 release 构建通过；原生构建自动构建前端并清理、复制 `bin/web`。FPV 增加离屏浮点帧缓冲/着色器失败时的直接渲染回退；该回退尚待真实显卡验证。Three.js 公共包仍有体积警告 |
| Qt 桥接 | 2 个业务用例通过（含初始化/清理共 4 项） |
| Qt 网络 | `Arena_connection` 3 个业务用例在 `listen()` 处因沙箱禁止 TCP 监听失败，不能计为通过 |
| 本地界面 HTTP | 新增页面/脚本与音频 Range 两个业务用例，因同一 TCP 限制明确跳过。客户端本地 HTTP 仅监听回环临时端口，供 Qt 6.2 标准 fetch/glTF 和音频加载，无需对局域网放行此端口 |
| 桌面与浏览器 | 当前进程无法连接 `DISPLAY=:1`；Qt 离屏启动和 Playwright Chromium 也因 IPC/`setsockopt` 权限失败。`build/operator-ui/report.json` 记录 `passed:false`、0 截图、未确认渲染帧 |
| 真实局域网、Windows、缩放 | 待目标环境验收；不能声称黑屏已在实际桌面消失或双机完整通过 |

图形检查脚本已扩展到首页、机库、加载、入场、倒计时、五席位、判罚/断线与结算，包含 720p、1080p、125% 和 150% 缩放。命令及桌面验收要求见本文末尾。完整旧 Python HTTP/WebSocket 套件仍因当前环境缺少 `httpx` 未执行；现有直接 Room/引擎检查不依赖该包。

日期：2026-10-22。以下区分历史迁移记录、本次操作手界面验证、实际桌面与物理局域网验证。

## 本次操作手界面

| 项目 | 本次结果 |
| --- | --- |
| 主机 HUD 与操作控制 | `test_operator_hud.py` 8 项通过，使用真实 Rust 引擎；覆盖四个控制位置、结构血量/重建、五席位操作、费用、信息边界、判罚/过热、双席位短局及奇偶帧指令只执行一次 |
| React HUD | `bun test tests/web` 8 项、34 个断言通过；覆盖血量、控制进度、角色面板、断线、判罚、结算。这是 DOM 服务端渲染检查，不能证明实际像素显示 |
| 架构 | `test_architecture.py` 6 项通过；旧 QML 叠层已移除，HUD 与三维画面共用 WebEngine 页面，原准星素材保留 |
| 前端构建 | TypeScript 与 Vite 生产构建通过；Three.js 公共包仍有大于 500 kB 的体积警告 |
| Qt 原生构建 | 系统 Qt 6.2.4 下 release 配置及编译通过，输出 `build/client/bin/RMUC2027Client` |
| Qt 页面桥接 | `test_arena_state` 2 个业务用例通过（含初始化/清理共 4 项） |
| Qt 网络检查 | `test_arena_connection` 的 3 个业务用例均被沙箱禁止 TCP 监听阻断，不能记为通过 |
| 浏览器画面检查 | 已提供 12 种场景、720p/1080p 与 125%/150% 缩放的检查脚本；当前 Chromium 因 `setsockopt: Operation not permitted` 无法启动，未生成有效截图、未验证 GPU 帧 |
| 完整 Python LAN 用例 | 当前虚拟环境缺少 `httpx`；离线缓存不足，未重跑完整套件。上面的 8 项检查不依赖该包 |
| Windows / 真实双机 | 待目标环境验收 |

一次性补给、复活和哨兵姿态指令现在保留到引擎实际处理的偶数帧；同类连续指令按序处理，避免在物理帧中被清空或互相覆盖。操作回执仍表示控制器接收，实际效果由后续主机状态确认。

## 历史迁移记录

以下保留此前迁移时的记录，不代表本次重新执行了所有检查。

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
| 主机运行包 | `build/runtime` 通过运行前检查、文件哈希及监听/父进程退出检查；不携带研究代码或训练依赖 |
| 原始数据 | 迁移前 72 个数据文件核对哈希，删除的旧平台状态另有备份记录 |
| Windows/真实双机/GPU | 尚未验证，不能声称完整交付验收完成 |

当前已安装 Qt 6 SDK，但此轮环境的 TCP 监听和 Chromium IPC 受沙箱限制，尚不能在这里证明实际网络传输或画面显示。输入执行确认耗时不等于输入到可见响应。

## 在目标环境复核

```bash
uv sync --locked --group research --group test
uv run --locked --group research --group test python -m unittest discover -s tests/python -v
cargo test --locked --release -p rmuc-sim
bun install --frozen-lockfile
bun run build
bun test tests/web
cmake --preset release -DCMAKE_PREFIX_PATH=本机Qt桌面SDK目录
cmake --build --preset release
ctest --preset release
```

### 桌面浏览器画面检查

在已安装 Google Chrome、Node.js 和 Python 环境的 Linux 桌面上执行；Playwright 仅为检查工具，可装在项目外：

```bash
npm install --prefix /tmp/rmuc2027-ui-tools playwright@1.64.1
NODE_PATH=/tmp/rmuc2027-ui-tools/node_modules node scripts/check_operator_ui.cjs
```

可用 `RMUC_CHROME` 指定兼容 Chromium 可执行文件，`RMUC_PYTHON` 指定 Python。报告与截图写入 `build/operator-ui/`。脚本使用真实引擎生成快照并模拟页面桥接，检查布局边界、操作菜单、过期状态与渲染帧；仍需另行在真实 Qt 客户端验证图传、指针捕获、网络传输与缩放。

完成 Windows/Linux 原生构建后，实际运行两机建房、选位、命中、补给、复活、重装辅助、机库指令、断线接管和重连；检查回放显示、720p/1080p 及 Windows 125%/150% 缩放。完成这些检查前，原生与双机验收仍为待完成。
