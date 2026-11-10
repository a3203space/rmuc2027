# 哨兵 v4 独立追加一小时的最终恢复包

用户追加授权从 2026-11-10 17:00:06 UTC 到 18:00:06 UTC；梯度截止 17:40:06 UTC。主哨兵从原 10310 samples / 317 optimizer steps 继续，步兵与重装辅助策略冻结。本包记录追加运行，不覆盖原两小时封存。

原两小时完整恢复包和审计证据：[固定提交 abc51594a43cad9127f003039f57e0040d0c9be1](https://github.com/a3203space/rmuc2027/commit/abc51594a43cad9127f003039f57e0040d0c9be1)。原始 SQLite 外置，只记录哈希；未知真实动作和未校准坐标语义保持掩码。

追加运行的选择结果（来自已完成 selection.json）：
```json
{
  "selected": "initialized.zip",
  "ppo_improvement_verified": false,
  "complete_pairs": 1,
  "resume_default": "model.zip",
  "reason": "keep_demonstration_anchor_pending_evidence"
}
```

完整评估、恢复诊断与真实性边界以包内 evaluation、review-notes、campaign 和 selection 为准。累计 samples 包含原运行；不将两个同场景 worker 计成两种阵容，不宣称 486 阵容全部 420 秒验收。

GitHub 按 48 MiB 分片保存，重组并校验：
```bash
cat bundle.part-* > bundle.zip
sha256sum -c SHA256SUMS
```

打包前迁移验证是在本机不同目录完成；尚未在用户的 5600X + 5070 Ti 目标机实测。校验后按包内 RESUME.txt 安装依赖、重建 Rust、执行 Ubuntu 24 preflight。Torch 推理设备与 Warp 物理设备分别检查；重新实测吞吐。恢复模型/优化器/RNG/采样状态，丢弃 partial rollout 并开启新 episode，不保证跨硬件逐位一致。PPO 运行不重新启用隐藏 BC。网络备份耗时独立于训练授权，不增加训练时长。
