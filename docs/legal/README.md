# 来源与许可证

Rust 仿真、Python SDK 和三维渲染源自 Arena-1；原始许可证见根目录 LICENSE。
Qt 联机客户端及界面源自 RM26_Client；原始许可证见 apps/client/LICENSE。
项目现只使用 2027 配置所引用的三维素材，不携带旧赛事地图或 RM26 客户端音频包。源码中保留的三张准心/热量环图片来源与哈希见 client-assets.json，原素材清单将 resources/crosshair 标为可发布；图片不因客户端代码采用 MIT 而自动改为 MIT 许可。当前操作手准星改为代码绘制的细线矩形和中心标记，不再导入这三张图片，因此它们不进入当前 Vite 发行资源。
发行包通过 scripts/licenses.py 收集实际 JavaScript/Rust 依赖的许可证；Qt 部署还应随附所用 Qt 版本的许可证。
原始仓库来源记录保存在 .history，不进入发行包。

本次直接移植本机 Arena-1 的对战页框架、广播 HUD、模型缩略图、比赛数据面板、出战机库、首页、入场和结算组件。原样复制与适配文件的来源哈希见 [arena-ui-port.json](arena-ui-port.json)；原样复制文件有字节一致性检查。图标标记与广播 HUD 的属性检查、末项读取改用 Qt 6.2 / Chromium 87 兼容写法。导航使用同仓库的 `brand/arena-1.png`。源码与品牌素材继续遵循根目录 Arena-1 许可证。

新增 Lucide React 0.468.0 使用 ISC 许可证，发行包依赖许可证清单收录完整文本。策略来源标记的 SVG 路径与来源注释随 `agentLogos.tsx` 原样移植（仅属性检查做运行时兼容适配）；其 LobeHub MIT 许可说明见 [agent-logos-LICENSE.txt](agent-logos-LICENSE.txt)，标记仅用于标识策略来源，商标权归原权利人。

死亡视角与全场公屏版本另移植 Arena-1 的 `killAnnouncements.ts`、`KillAnnouncement.tsx`、`killAnnouncement.css` 和原版播报回归用例，从 `media.ts` 提取播报播放器；来源哈希记录在上述源码清单。首杀、双杀、三杀、四杀、五杀和团灭六段 MP3 原样复制，来源路径及逐文件哈希见 [arena-audio.json](arena-audio.json)，不使用替代提示音或合成语音。

Arena 风格界面适配了 Arena-1 的 RobotPreview、intro.css、broadcast.css、teamLineup.css 和入场镜头，原始参考提交为 `bbaa885b774e86582e8e537e5e01197f1c2bb1ea`。新增的首页、机库、入场、倒计时和结算音频来自同一提交；逐文件来源与哈希见 [arena-audio.json](arena-audio.json)。94 个内置 2027 策略包从本项目保留的策略示例复制到 `packages/core/bots/catalog`，原 manifest、版本哈希与作者信息保留。均遵循根目录 Arena-1 许可证。

右键自瞄参考 [chenjunnn/rm_auto_aim 的 tracker.cpp](https://github.com/chenjunnn/rm_auto_aim/blob/main/armor_tracker/src/tracker.cpp) 中准星中心优先、目标身份保持与连续角度处理的设计。该项目采用 MIT 许可证；本项目为仿真数据与现有 SDK 单独实现，没有复制其源代码或引入运行依赖，具体说明见 [自瞄设计](../aim-assist.md)。
