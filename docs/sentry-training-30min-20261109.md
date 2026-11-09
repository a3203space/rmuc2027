# 哨兵半小时训练记录（2026-11-09）

状态：训练中。开始时间 21:18:05，1800 现实秒预算，预计约 21:48 结束。

- 后端：`warp_kinematic`，`cuda:0`；PPO 网络使用 CUDA，16 并行。
- 仅红方 7 号哨兵学习目标点选择，控制 2 Hz；其余机器人固定策略。
- 从头训练，初始学习率 `3e-4`。允许按训练指标在完整 rollout 之间人工干预，实际变更单独记录。
- 运行目录：`data/research/sentry-kinematic-30min-16env-20261109-211805`。
- 日志：同级 `sentry-kinematic-30min-16env-20261109-211805.log`。
- 当前代码与干预功能已备份到主分支；对应启动代码提交 `e63fa9c`。

第一份约 5 分钟检查点已经上传并核对本地恢复包内各文件 SHA256。恢复包包含权重、实验元数据、每轮更新/回合记录及源码/仿真二进制快照。

[GitHub 检查点分支（latest.zip）](https://github.com/a3203space/rmuc2027/tree/backup/rl-checkpoints/rl-run-9c41286f32670af2)

当前过程记录见 [状态快照](validation/sentry-training-30min-20261109.json)。训练中指标不是独立评估成绩，尚不能据此认定战术学会或胜率改善。结束后追加最终统计和观看命令。
