# 来源与许可证

Rust 仿真、Python SDK 和三维渲染源自 Arena-1；原始许可证见根目录 LICENSE。
Qt 联机客户端及界面源自 RM26_Client；原始许可证见 apps/client/LICENSE。
项目现只使用 2027 配置所引用的三维素材，仅保留客户端实际使用的三张准心/热量环图片，不携带旧赛事地图或 RM26 客户端音频包。原素材清单将 resources/crosshair 标为可发布；本次原样保留的文件来源与哈希见 client-assets.json，图片不因客户端代码采用 MIT 而自动改为 MIT 许可。
发行包通过 scripts/licenses.py 收集实际 JavaScript/Rust 依赖的许可证；Qt 部署还应随附所用 Qt 版本的许可证。
原始仓库来源记录保存在 .history，不进入发行包。

Arena 风格界面适配了 Arena-1 的 RobotPreview、intro.css、broadcast.css、teamLineup.css 和入场镜头，原始参考提交为 `bbaa885b774e86582e8e537e5e01197f1c2bb1ea`。新增的首页、机库、入场、倒计时和结算音频来自同一提交；逐文件来源与哈希见 [arena-audio.json](arena-audio.json)。94 个内置 2027 策略包从本项目保留的策略示例复制到 `packages/core/bots/catalog`，原 manifest、版本哈希与作者信息保留。均遵循根目录 Arena-1 许可证。
