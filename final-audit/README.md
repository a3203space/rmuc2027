# 追加运行的最终审计补充

本目录保存170MiB恢复包完成后才写入的最终报告。ZIP本体及4个分片没有改写，整包 SHA256 为 `ab479934c9a902708decaff1dab8379a61f9b117db6808c7a1e2be58494df7cd`。

原包固定提交：[980b5da1f6f7486a39f4848914b7aa170c94d8f4](https://github.com/a3203space/rmuc2027/commit/980b5da1f6f7486a39f4848914b7aa170c94d8f4)。
最终源码：[dc27ce41f6f0aaf3c8d86c28fe7393c6c4211134](https://github.com/a3203space/rmuc2027/commit/dc27ce41f6f0aaf3c8d86c28fe7393c6c4211134)。

最终状态以本目录 campaign.json、review-notes 和 docs/validation 的报告为准。恢复用模型和优化器仍来自原ZIP，补充文件不改变权重。

追加主训练新增4012 samples、累计14322；新增177 optimizer steps、累计494，PPO期间BC更新0。评估受共享主机CPU争用：Warp两侧PPO仅16.8/14.5秒截断，Rapier红中断保留轨迹、其他未运行；MJ红完成60秒配对，初始化与PPO均未证实过洞或占领。最终completed_partial，selected保守保留initialized.zip，model.zip仍供纯PPO续训；没有验证追加综合改善。

含父运行历史的场景去重覆盖为4种started、2种short_completed、1种full_completed。两个相同队列worker的不同物理随机种子不计为两种阵容；未完成486阵容全部420秒验收。旧克隆sampler迁移修复只对后续恢复生效。

本机不同目录的迁移验证通过；未来5600X+5070Ti机器尚未实测，应独立执行preflight和吞吐测试。网络上传耗时不计入训练时长，没有继续训练或仿真。
