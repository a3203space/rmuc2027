# 哨兵 v4 定时恢复快照

源码：[对应提交](https://github.com/a3203space/rmuc2027/commit/ab58869a43acc30d8cd038ca0d7f18add6eda28f)

main.zip 包含当前已完整发布的模型、初始化锚点、优化器ZIP、RNG、抽样/覆盖状态及阶段证据。
本次是独立追加一小时的main-only续训，辅助策略冻结，不重复上传；可核验的原辅助ZIP提交和SHA见frozen-helpers.json。原两小时历史不变。
各模型和resume计数匹配，读取前后哈希稳定；三阶段分别捕获，并非全系统同一时刻事务。actor导出可能滞后，恢复训练以model.zip为准。
阶段：独立追加运行，从原主哨兵10310 samples/317 optimizer steps恢复；infantry与heavy助手冻结。检查点记录累计计数，PPO期间不启用隐藏BC；便携actor导出可能滞后，恢复以与resume-state计数一致的model.zip为准。每次迁移丢弃partial rollout并从新episode开始，不保证跨硬件逐位一致。
不宣称收敛、真实对战优越性或486场景全部420秒验收。没有上传原始SQLite。最终完整跨机器包另行提供。
跨机器/后端恢复须丢弃partial rollout、重启episode，并重新测并行数；主2773保持严格actor输入。
