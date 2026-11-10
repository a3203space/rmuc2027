源码阶段备份：固定学习2773、严格actor／全知critic、辅助兵种、243/486阵容覆盖、2026只读轨迹、能力漂移和跨机器恢复。
新增几何穿越证据与7项测试：连续进出洞口、横向位置在通道内、洞内实际充分折叠，替代maxfold加finalx启发式统计。
包含此前授权研究变更。训练尚在进行，不能视为收敛或486场景全部420秒验收；未知原始动作保持掩码。
不覆盖main，不含原始SQLite和data运行缓存。检查点位于backup/rl-checkpoints/sentry-v4-20261110。

阶段说明：主哨兵PPO已结束：10,310个采集样本、317次optimizer steps、20,288次sample visits、206个gradient epochs，actor参数L2变化0.152359；PPO期间示范更新为0。三角色模型稳定且actor.json均已刷新。2场420秒训练对局已结束，不能替代冻结验收；6组冻结对照仍进行中。增量包含雷达标签分层拆分（21项测试），活动分类器保持unknown、较弱omni候选不自动晋级；恢复包synthetic mask修复（6项测试）与只读render调用。
