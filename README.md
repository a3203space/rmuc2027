# 哨兵 v4 原两小时运行的最终恢复包

本包封存原两小时运行；之后授权的一小时续训是独立运行，不改写本包历史。为避免GitHub单文件大小限制，完整ZIP按48MiB分片；原始SQLite数据库外置，包内manifest记录其哈希。

合并并验证：
```bash
cat bundle.part-* > bundle.zip
sha256sum -c SHA256SUMS
```

通过校验后按bundle-manifest和包内RESUME.txt安全解包、安装锁定依赖、编译Rust并执行目标机preflight。
策略CPU Torch与物理Warp CUDA独立；重新实测并行数，恢复时丢弃partial rollout并开启新episode，不承诺跨硬件逐位一致。

六组冻结对照：初始化锚点过洞5/6、PPO过洞6/6；控制指标合计103.6降至71.0。因此展示selected仍为initialized.zip，继续训练使用model.zip，不宣称PPO整体优于锚点。详见selection.json和包内evaluation证据。

真实性边界：结果与已完成验收以包内campaign/selection/evaluation为准，上传成功不代表策略收敛或486阵容全部420秒验收。源码分支的后续修改属于追加工作，包内source快照和文件哈希才是本次封存依据。
