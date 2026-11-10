# 哨兵 v4 定时恢复快照

源码：[对应提交](https://github.com/a3203space/rmuc2027/commit/2cf661411828757f7c94cccbfa5547d5fa1f9ab3)

main.zip、infantry.zip、heavy.zip分别包含当前已完整发布的模型、初始化锚点、优化器ZIP、RNG、抽样/覆盖状态及阶段证据。
各模型和resume计数匹配，读取前后哈希稳定；三阶段分别捕获，并非全系统同一时刻事务。actor导出可能滞后，恢复训练以model.zip为准。
当前训练尚在进行；不宣称收敛、真实对战优越性或486场景全部420秒验收。没有上传原始SQLite。最终完整跨机器包另行提供。
跨机器/后端恢复须丢弃partial rollout、重启episode，并重新测并行数；主2773保持严格actor输入。
