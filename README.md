# 哨兵 v4 定时恢复快照

源码：[对应提交](https://github.com/a3203space/rmuc2027/commit/ab7e8a4e49291e8853d5e6d5f58a906173da7e0d)

main.zip, infantry.zip, heavy.zip 包含当前已完整发布的模型、初始化锚点、优化器ZIP、RNG、抽样/覆盖状态及阶段证据。
各模型和resume计数匹配，读取前后哈希稳定；三阶段分别捕获，并非全系统同一时刻事务。actor导出可能滞后，恢复训练以model.zip为准。
阶段：原两小时已完成：主哨兵10310 samples、317 optimizer_steps、20288 sample visits、206 gradient epochs，actor L2变化0.152359，PPO期间BC更新0；两场420秒训练对局结束。六组冻结对照全部完成：初始化锚点过洞5/6，PPO过洞6/6，但控制指标合计103.6降至71.0，故展示selected仍initialized.zip，续训model.zip。原412MiB可迁移包已通过异地解压/推理/Rust校验。另获授权的一小时是独立main-only追加运行，helpers冻结，绝对截止18:00:06UTC；不改原两小时历史，追加结果尚未验收。 未来preflight改为真实main+aux负载（5项测试）；future sampler拓扑与旧克隆独立迁移、祖先覆盖守恒（6项）；reward开关（4项）；evaluation diagnostics/reuse（13项）。这些修复面向后续启动/恢复，本次正在运行的进程未重置或迁移。新增review-notes/recovery-analysis.json由恢复包纳入。
不宣称收敛、真实对战优越性或486场景全部420秒验收。没有上传原始SQLite。最终完整跨机器包另行提供。
跨机器/后端恢复须丢弃partial rollout、重启episode，并重新测并行数；主2773保持严格actor输入。
