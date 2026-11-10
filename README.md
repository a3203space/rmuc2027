# 哨兵 v6 研究备份

源码位于 source/，保存包含未提交修改的工作树快照；主分支和共享索引保持不变。
checkpoints/latest.json 指向已原子发布的完整代：模型/优化器、RNG、采样状态、actor 和配置。helpers/ 是冻结辅助 actor。
若存在 final/latest.json，它指向最终恢复包的 48 MiB 分片。进入该 path 后重组：

```bash
cat bundle.part-* > bundle.zip
sha256sum -c SHA256SUMS
```

恢复时开启新 episode，丢弃半个 rollout；不承诺跨机器逐位一致。原始 2026 SQLite 未上传。
训练/验收范围与选用权重以 reports/ 中冻结报告为准；备份成功不等于策略改善、收敛或全矩阵通过。
本次阶段：training；完整检查点样本：15238。
