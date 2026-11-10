# 哨兵 v4 定时恢复快照

源码：[对应提交](https://github.com/a3203space/rmuc2027/commit/ab7e8a4e49291e8853d5e6d5f58a906173da7e0d)

main.zip 包含当前已完整发布的模型、初始化锚点、优化器ZIP、RNG、抽样/覆盖状态及阶段证据。
本次是独立追加一小时的main-only续训，辅助策略冻结，不重复上传；可核验的原辅助ZIP提交和SHA见frozen-helpers.json。原两小时历史不变。
各模型和resume计数匹配，读取前后哈希稳定；三阶段分别捕获，并非全系统同一时刻事务。actor导出可能滞后，恢复训练以model.zip为准。
阶段：独立追加运行主哨兵训练已正常结束：新增4012 samples、累计14322，新增102 gradient epochs、177 optimizer steps（累计494），累计31616 sample visits，actor参数L2变化0.044858984；PPO期间BC更新0。actor.json已刷新。步兵与重装助手冻结；六组reuse-anchor冻结评估刚开始，尚未形成追加改进结论。跨机器恢复仍以model.zip与resume-state匹配计数为准，丢弃partial rollout并开启新episode。
不宣称收敛、真实对战优越性或486场景全部420秒验收。没有上传原始SQLite。最终完整跨机器包另行提供。
跨机器/后端恢复须丢弃partial rollout、重启episode，并重新测并行数；主2773保持严格actor输入。
