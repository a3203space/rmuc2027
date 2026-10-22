# 2027 工程迁移

本次将原 Arena-1 和 RM26_Client 整理为一个 RMUC 2027 工程。根目录统一提供 Python、Rust、前端和 Qt 构建入口。

## 保留与删除

| 内容 | 当前去向 |
| --- | --- |
| Qt 联机窗口与 HUD | `apps/client`，替换旧赛事状态模型，直接接收主机投影 |
| 联机主机 | `apps/server`，不导入研究代码 |
| 2027 Rust 规则与物理 | `packages/simulation`；只接受 2027 场地 |
| SDK、策略子进程、回放 | `packages/core`；固定补位策略在同目录的 bots 下 |
| 第一视角与回放渲染 | `packages/renderer` |
| 策略开发与对战 | `research/strategies` 和独立研究命令 |
| PPO 训练、评估、导出 | `research/rmuc2027_research/rl` |
| 旧赛事窗口、MQTT/UDP、Protobuf、视频解码、串口、AR、旧协议模拟器 | 删除 |
| 旧赛季地图、专属规则方法与非法型号 | 删除 |
| Electron、在线商店、自动更新、排行榜、Agent 平台与旧网站 | 删除 |

部分符合 2027 制作规范的通用步兵型号仍使用原来的几何素材文件名。这些型号经过 2027 合规过滤后保留，不代表仍支持旧赛季。模型目录仅保留 2027 合法型号，以及实际引用的地图、建筑和机构资源。

## 已有成果

迁移前的 72 个数据文件已逐个核对 SHA-256，清单位于 `data/migration.json`：

- 联机历史回放移至 `data/replays`。
- 已确认的 2027 训练模型、策略和评估回放移至 `data/research`。
- 无法从内容确认用途的诊断文件留在 `data/unclassified`，不进入发行包。
- 旧独立平台的数据库、凭据和运行日志单独备份后移除，清单记录删除状态与备份位置。

模型原件不修改。两个已有检查点的迁移副本位于 `sentry-smoke-migrated`、`sentry-resume-migrated`，权重和优化器文件与原件哈希一致；新实验记录明确保留原始运行时信息。迁移工具只接受登记过的源运行时、相同的 116 维输入与二维动作空间。

历史 2027 回放可按帧读取。新的确定性校验要求相同引擎版本，不把历史回放元数据改成当前哈希。

## 备份与历史

执行删除前已保存两个仓库的源码、未提交改动和 Git 元数据：

```text
/tmp/rmuc2027-cleanup-backup-20261022/
```

原 Git 元数据另保留于 `.history/arena` 和 `.history/client`，原提交历史可以独立读取；没有重写或合并历史。当前环境根目录的 `.git` 为只读保留目录，本次没有改写它。

```bash
git --git-dir=.history/arena log -1
git --git-dir=.history/client log -1
```

`.history`、备份、训练环境、数据和构建缓存不属于发行内容。旧的主机运行包已移除，新的包为 `build/runtime`。
