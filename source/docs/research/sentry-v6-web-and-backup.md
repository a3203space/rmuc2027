# v6 网页研究入口与受截止时间约束的备份

网页 `/` 的「研究」页读取 `data/research/*/run-manifest.json`。没有登记运行时显示空目录；不会查找随意路径、自动选取训练中的可变 `model.zip`，也不会启动训练。

启动 API 为 `POST /api/start`：

```json
{"mode":"research","research":{"run_id":"run-id","checkpoint_id":"baseline-0123456789ab","scenario_id":"f2731-2732_e2731-2732_s2773_red","capability_profile_id":"evaluation-red","backend":"rapier"},"seed":81001,"duration":60,"save_mp4":false}
```

checkpoint ID 必须以模型文件 SHA256 的前 12 位结尾。运行 manifest 的 format 为 `rmuc-research-run-v1`，资源统一使用相对运行目录的 `{path,sha256}`；必需字段是 `run_id`、`runtime_config`、`checkpoints`、`helpers`（infantry/heavy）、`scenarios` 与 `capability_profiles`。checkpoint 资源另含 label、policy_version；scenario 值是 `ScenarioSpec.to_dict()`；能力资源另含 learner_side、label。雷达分类器若使用，必须在 `radar_classifier` 中显式登记 SHA256。

能力配置的八个地面席位型号必须与所选红蓝阵容相符，只允许所选主哨兵有 protected_learner 标记；主哨兵机动倍率为 1，自瞄额外延迟和噪声为 0。能力档按 learner_side 筛选。网页重用 SentryEnv 和冻结模型的确定性推理，10 Hz 决策每步推进两个 20 Hz 物理帧，按 0.1 s 配速；不会发生画面节拍翻倍。训练专属采样器、恢复任务、自动训练录制路径和示范损失路径均从网页配置移除。研究模式始终 `test_match=true`，即使 420 秒完整结束也不进入 Elo；拒绝中途裁判干预。

目录接口：

- `GET /api/research/runs`
- `GET /api/research/runs/{run_id}/replays`
- `GET /api/research/runs/{run_id}/replays/{replay_id}`

`replays` 是 ID 到资源描述的映射；每条资源附 `kind`、`backend`、`checkpoint_id`（已知时）、`scenario_id`、`seconds`、`truncated`、`recording_end_reason`。服务端校验 SHA256、标准全帧结构和连续 20 Hz tick 后提供原始文件；浏览器只加载用户选择的一段。训练期间策略跨 PPO 更新时保留 `training_recording.policy_changes_during_episode` 和 `policy_updates`，不会把整段标成最终候选冻结权重。配额或时间截止的片段必须标明 truncated。录制策略由训练协调器负责，网页不新增训练直播。

## Git/SSH 备份

调用者负责授权并启动，导入模块不联网：

```bash
research/.venv/bin/python -m rmuc2027_research.rl.v6_backup \
  --run data/research/RUN_ID --deadline-unix ABSOLUTE_UTC_UNIX --loop
```

默认每 900 秒备份，所有子命令的 timeout 不超过剩余截止时间。工作进程和 Git/SSH 继承调用者进程组，便于协调器在同一截止时刻清理本任务进程；不延长上传预算。最终包使用同一 worker/分支：

```bash
research/.venv/bin/python -m rmuc2027_research.rl.v6_backup \
  --run data/research/RUN_ID --deadline-unix ABSOLUTE_UTC_UNIX --final /path/to/final.zip
```

Python API：`BackupWorker(run, deadline_unix, remote='origin', repo_root=ROOT).snapshot()`、`.upload_final(path)`、`.loop()`。同一运行用非阻塞文件锁避免两个上传器冲突；调度器应在最终上传前停止周期调用或等待其当前操作结束。

`main/checkpoint-ready.json` 仅在完整 generation 发布后原子写入：

```json
{"directory":"main/checkpoints/update-000081","samples":10438,"update_id":81,"optimizer_steps":321,"skill_sha256":"SHA256","sha256":{"model.zip":"SHA256","resume-state.pt":"SHA256","samplers.json":"SHA256","actor.json":"SHA256","experiment.json":"SHA256","coordinator-state.json":"SHA256"}}
```

备份核对五文件 SHA、模型 samples、读取前后稳定性，同时携带可选 coordinator-state.json。无 ready 标记时仅备份源码/现有证据，不把活动权重当完整检查点。源码以临时 Git index 保存整个工作树，排除 data、原始数据目录、数据库、node_modules、构建输出与缓存。共享 HEAD/index 不修改。分支为 `backup/sentry-v6-<run目录名>`，源码在 source/，检查点与最终包保留不可变哈希目录，latest.json 仅是目录指针。

Git push 不使用 force；每次以本地 receipt 所记录的远端 tip 为预期 parent，上传前后再验证。失败写 `github-backups/status.json` 并保留本地包。push 成功而验证超时的结果保持 pending；下次只在远端恰好为该 pending commit 时恢复确认，不声称未核验上传成功。

最终 ZIP 拆为 48 MiB 分片，包本体 SHA 与 package.json 核对。进入 final/latest.json 指定的目录后：

```bash
cat bundle.part-* > bundle.zip
sha256sum -c SHA256SUMS
```

备份不是模型晋升证据。训练/验证范围及 selected 权重以 reports/ 为准，恢复时开启新 episode、丢弃 partial rollout，不承诺跨硬件逐位一致。

## Ubuntu 24 目标机器续训

以下命令从已核验 ZIP SHA 并解压的包根目录运行。5600X／5070 Ti 是目标配置；本机的 CPU 单元测试、异路径恢复和负载测速不代表已在这台目标机器验证。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  -r run-state/requirements-runtime.txt
cargo build --release --manifest-path source/packages/simulation/Cargo.toml
export PYTHONPATH="$PWD/source/packages/core:$PWD/source/research:$PWD/source/apps/server"
export RMUC_RESOURCE_ROOT="$PWD/source"
PYTHON_BIN="$PWD/.venv/bin/python" bash preflight-ubuntu24.sh
```

依赖使用包内版本清单；不可为解决安装问题静默更换 Torch、SB3 或物理后端版本。策略和 PPO 使用 CPU Torch，Warp 物理需要真实 CUDA；`torch.cuda.is_available()==False` 本身不构成失败。这个包级检查核验文件、模型、三优化器、775／454／3 维输入、合法复活掩码、技能冻结和便携推理，并不测量目标机物理吞吐。

下面的 `run` 命令会开始**新一轮两小时**实验。输出目录须尚不存在；初始化、恢复前缀、真实负载测速、PPO、冻结评估、网页检查、封包和上传均计入该轮 `budget.json` 的绝对截止时间。

```bash
.venv/bin/python -m rmuc2027_research.rl.v6_campaign run \
  --resume-bundle "$PWD" --output /absolute/path/to/new-v6-round
```

该入口从包内当前 `model.zip` 继续，原样保留 tactical／revive／critic 三份 Adam 状态、训练计数及 RNG／抽样历史；新的配对基线是这个恢复模型，不能把原始祖先、历史 selected 或尚未发布的内存权重混为一谈。`main/coordinator-state.json` 优先于运行末尾协调器快照，未完成 lease 释放后重新开始回合，已记录的历史计数不清零。当前正式测速仅包含 Warp 的 1／2 环境和 Rapier 的 1 环境，各预热一次、测两个 30 步窗口；4／8 环境扩展尚未作为本轮证据。

仅检查材料化、暂不启动物理或训练时，可使用单独的新目录：

```bash
.venv/bin/python -m rmuc2027_research.rl.v6_campaign initialize-resume \
  --resume-bundle "$PWD" --output /absolute/path/to/materialization-check
```

不要把这个已存在的检查目录再传给 `run`。`run` 会自己创建目标目录和预算文件。恢复包包含既有数据准备结果，原始 2026 SQLite 保持外部数据；单纯继续 v6 PPO 不需要重新执行 2026 数据提取或辅助 BC。

在目标电脑上额外测试 1／2／4／8 环境时，使用上述独立材料化目录，并显式给出测速截止时间。例如下面为测速单独分配最多 20 分钟；这会执行真实物理和冻结策略推理，不会执行 PPO 更新：

```bash
target_deadline=$(.venv/bin/python -c 'import time; print(time.time()+1200)')
.venv/bin/python -m rmuc2027_research.rl.v6_preflight target \
  --root /absolute/path/to/materialization-check \
  --counts 1 2 4 8 --deadline-unix "$target_deadline"
```

结果写入该材料化目录的 `target-preflight/`，包括版本清单、每个进程日志、两窗实际仿真秒／墙秒、截断情况和建议环境数。再次测速须用新 `--output` 目录，不能覆盖旧证据。`requested_matrix_complete` 为 false 时不能声称完成 4／8 环境验证；只有实际完成的配置才能参与建议。这个可选入口不会扩大当前正式 `run` 的 1／2 环境范围，也不会自动把未来训练改为 4／8 环境。

ZIP 不含 `.git` 和 SSH 凭据。需要延续 GitHub 定时备份时，应事先让 `source/` 成为有 HEAD 的 Git 工作区、配置正确的 `origin`，并确认目标机能非交互 SSH 推送；训练脚本不会代建密钥。备份失败以 `github-backups/status.json` 为准，不等于本地检查点或 ZIP 已损坏。包外 `release-index.json` 才记录最终 ZIP SHA、异路径核验及远端备份结果，包内不会自引用自身 ZIP 哈希。
