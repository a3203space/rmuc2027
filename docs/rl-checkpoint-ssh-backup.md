# RL 检查点 SSH 备份

设置 `github_backup_repo` 后，既有异步备份在没有 `gh` 时使用本地 SSH 上传至独立分支：

```text
backup/rl-checkpoints/rl-run-<运行目录哈希>
```

每次更新包含 `latest.zip` 和 `latest.json`。Git 历史保留旧版本；只做快进推送，冲突会拒绝并由后台重试。上传成功后验证远端提交，状态文件记录分支、提交和 SHA-256。SSH Git 通道限制单包不超过 95 MiB。

现有调度每 30 分钟及运行结束备份；本地原子检查点仍单独保存。没有 `gh` 时不需要 API 登录，SSH 访问权限沿用仓库推送配置。安装了 `gh` 的环境仍使用原 Release 通道。

恢复时从训练日志或 `.status.json` 找到备份分支，先取回到新的文件名：

```bash
RL_BACKUP_TAG='rl-run-替换为日志中的哈希'
git fetch origin "refs/heads/backup/rl-checkpoints/$RL_BACKUP_TAG"
git show FETCH_HEAD:latest.zip > checkpoint-backup.zip
git show FETCH_HEAD:latest.json > checkpoint-backup-metadata.json
```

校验整包及内部成员，再提取到新目录：

```bash
python3 - <<'PY'
import hashlib, json, zipfile
from pathlib import Path

archive = Path('checkpoint-backup.zip')
metadata = json.loads(Path('checkpoint-backup-metadata.json').read_text())
assert hashlib.sha256(archive.read_bytes()).hexdigest() == metadata['sha256']
target = Path('restored-checkpoint')
target.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as source:
    checks = json.loads(source.read('checksums.json'))
    for name in ('model.zip', 'experiment.json', 'progress.json'):
        data = source.read(name)
        assert hashlib.sha256(data).hexdigest() == checks[name]
        (target/name).write_bytes(data)
print(target)
PY
```

`model.zip` 是模型恢复入口；`experiment.json` 保留原配置及来源。续训仍须通过现有配置/规则/SDK/模拟后端兼容性检查，不能把下载成功当作兼容性证明。

当前已验证本地两代备份历史保留及 GitHub SSH 测试包上传、下载、哈希校验和临时分支清理。测试包不含训练模型；新后端真实模型的下载续训尚待验证。
