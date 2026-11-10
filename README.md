# 哨兵 v4 定时恢复快照

源码：[对应提交](https://github.com/a3203space/rmuc2027/commit/f649924bf981b562da93c91da1eb03ed7281efbd)

main.zip、infantry.zip、heavy.zip分别包含当前已完整发布的模型、初始化锚点、优化器ZIP、RNG、抽样/覆盖状态及阶段证据。
各模型和resume计数匹配，读取前后哈希稳定；三阶段分别捕获，并非全系统同一时刻事务。actor导出可能滞后，恢复训练以model.zip为准。
阶段：主哨兵PPO已结束：10,310个采集样本、317次optimizer steps、20,288次sample visits、206个gradient epochs，actor参数L2变化0.152359；PPO期间示范更新为0。三角色模型稳定且actor.json均已刷新。2场420秒训练对局已结束，不能替代冻结验收；6组冻结对照仍进行中。增量包含雷达标签分层拆分（21项测试），活动分类器保持unknown、较弱omni候选不自动晋级；恢复包synthetic mask修复（6项测试）与只读render调用。
不宣称收敛、真实对战优越性或486场景全部420秒验收。没有上传原始SQLite。最终完整跨机器包另行提供。
跨机器/后端恢复须丢弃partial rollout、重启episode，并重新测并行数；主2773保持严格actor输入。
