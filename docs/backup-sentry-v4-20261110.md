源码阶段备份：固定学习2773、严格actor／全知critic、辅助兵种、243/486阵容覆盖、2026只读轨迹、能力漂移和跨机器恢复。
新增几何穿越证据与7项测试：连续进出洞口、横向位置在通道内、洞内实际充分折叠，替代maxfold加finalx启发式统计。
包含此前授权研究变更。训练尚在进行，不能视为收敛或486场景全部420秒验收；未知原始动作保持掩码。
不覆盖main，不含原始SQLite和data运行缓存。检查点位于backup/rl-checkpoints/sentry-v4-20261110。

阶段说明：原两小时已完成：主哨兵10310 samples、317 optimizer_steps、20288 sample visits、206 gradient epochs，actor L2变化0.152359，PPO期间BC更新0；两场420秒训练对局结束。六组冻结对照全部完成：初始化锚点过洞5/6，PPO过洞6/6，但控制指标合计103.6降至71.0，故展示selected仍initialized.zip，续训model.zip。原412MiB可迁移包已通过异地解压/推理/Rust校验。另获授权的一小时是独立main-only追加运行，helpers冻结，绝对截止18:00:06UTC；不改原两小时历史，追加结果尚未验收。 未来preflight改为真实main+aux负载（5项测试）；future sampler拓扑与旧克隆独立迁移、祖先覆盖守恒（6项）；reward开关（4项）；evaluation diagnostics/reuse（13项）。这些修复面向后续启动/恢复，本次正在运行的进程未重置或迁移。新增review-notes/recovery-analysis.json由恢复包纳入。
