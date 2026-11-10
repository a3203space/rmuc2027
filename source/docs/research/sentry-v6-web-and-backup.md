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

## 只读进度查看

从仓库根目录执行以下命令，只需 Python 标准库，不加载模型、不启动物理、不写训练文件：

```bash
python3 scripts/v6_status.py --run data/research/sentry-v6-extension-20261111-0457
python3 scripts/v6_status.py --run data/research/sentry-v6-extension-20261111-0457 --json
watch -n 30 python3 scripts/v6_status.py --run data/research/sentry-v6-extension-20261111-0457
```

发布计数来自 `checkpoint-ready.json` 指向的完整代，覆盖从 coordinator 的 `counts` 重算，并与该轮 baseline 分开；注册的 486 个场景不等于已经完成 486 局。运行中分支不计为已通过续训检查，短段完成不等于 420 秒整局完成。备份栏显示历史 receipt，不重新联网验证远端；各文件依次读取，也不构成原子全局快照。

## Ubuntu 24 目标机器续训

以下命令从已核验 ZIP SHA 并解压的包根目录运行。5600X／5070 Ti 是目标配置；本机的 CPU 单元测试、异路径恢复和负载测速不代表已在这台目标机器验证。

先准备 Python 3.12 及其 venv/pip、可编译工作区的 Rust/Cargo、NVIDIA 驱动与 `nvidia-smi`。包内没有虚拟环境、Rust 编译输出或 wheel 镜像；原机 Rust/Cargo 为 1.99.1，目标机应以 `cargo build --locked` 的实际结果为准。CUDA Python 包不替代内核驱动。系统工具安装、源码构建和下述包级预检应在主动启动新训练窗口前完成。

保留 `run-state/requirements-runtime.txt` 原件。旧冻结清单可能同时含可编辑包和同名普通 distribution；下面只派生第三方副本，避免从公共索引下载同名 `rmuc2027-*` 包：

```bash
python3 -m venv .venv
.venv/bin/python - <<'PY'
from pathlib import Path
import re
local = {'rmuc2027', 'rmuc2027-core', 'rmuc2027-research', 'rmuc2027-server'}
lines = Path('run-state/requirements-runtime.txt').read_text().splitlines()
def name(line):
    token = re.split(r'[\s<>=!~;\[]', line.strip(), maxsplit=1)[0]
    return re.sub(r'[-_.]+', '-', token).lower()
kept = [line for line in lines if name(line) not in local]
Path('.venv/requirements-third-party.txt').write_text('\n'.join(kept) + '\n')
PY
.venv/bin/python -m pip install --index-url https://pypi.org/simple \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r .venv/requirements-third-party.txt
.venv/bin/python -m pip install --no-index --no-deps --no-build-isolation \
  -e ./source/packages/core -e ./source/research -e ./source/apps/server
cargo build --locked --release --manifest-path source/packages/simulation/Cargo.toml
export PYTHONPATH="$PWD/source/packages/core:$PWD/source/research:$PWD/source/apps/server"
export RMUC_RESOURCE_ROOT="$PWD/source"
nvidia-smi
PYTHON_BIN="$PWD/.venv/bin/python" bash preflight-ubuntu24.sh
```

第三方版本来自包内清单，官方 PyPI 与 CPU Torch 索引只是下载来源，不保证未来持续可下载；下载或驱动不兼容时停止并保留错误，不静默更换 Torch、SB3 或物理后端版本。本仓三包从包内源码安装，`--no-deps --no-build-isolation` 使用刚装好的固定依赖与 setuptools。这个安装路径已在临时源码副本做过离线 metadata dry-run，未执行目标机安装或联网下载验证。

策略和 PPO 使用 CPU Torch，Warp 物理需要真实 CUDA；`torch.cuda.is_available()==False` 本身不构成失败。包级检查核验文件、模型、三优化器、775／454／3 维输入、合法复活掩码、技能冻结和便携推理，并检查原生后端初始化与 GPU 依赖；不进行 PPO 更新，也不测量目标机物理吞吐。`training_ready=false` 时不要直接开始训练。

网页构建和浏览器工具也不包含在 Python 锁中。先准备 `package.json` 指定的 Bun 1.3.14、Node、`/usr/bin/google-chrome`，从包内 `bun.lock` 构建：

```bash
(cd source && bun install --frozen-lockfile && bun run build:arena-web)
npm install --prefix "$PWD/.browser-tools" --save-exact playwright-core@1.65.0
node -e 'console.log(require(process.argv[1]).chromium ? "Playwright ready" : "missing")' \
  "$PWD/.browser-tools/node_modules/playwright-core"
/usr/bin/google-chrome --version
```

Playwright 1.65.0 是本轮实际使用版本，Chrome 与 Node 的目标机可用性仍须现场检查；`playwright-core` 不自动安装浏览器。当前 watcher 使用 `/usr/bin/google-chrome`，也不提供 `--chrome` 转发参数，不能仅下载浏览器到另一个路径就认为已就绪。网页静态产物必须是 `source/build/arena-web`。

下面的 `run` 命令会开始**新一轮两小时**实验。输出目录须尚不存在；初始化、恢复前缀、真实负载测速、PPO、冻结评估、网页检查、封包和上传均计入该轮 `budget.json` 的绝对截止时间。

```bash
new_run="$PWD/source/data/research/new-v6-round"
.venv/bin/python -m rmuc2027_research.rl.v6_campaign run \
  --resume-bundle "$PWD" --resume-generation baseline \
  --output "$new_run"
```

`new_run` 必须不存在。这里特意放在 `source/data/research/`：网页目录目前固定读取 `$RMUC_RESOURCE_ROOT/data/research`，`RMUC_USER_DATA` 不改变研究目录，任意其他输出路径不会自动出现在原网页。`run` 的 CLI 固定两小时，没有延用原八小时 deadline，也不会调用 `v6_longrun`。

单独的 `v6_campaign run` 不创建网页守护。启动上述命令后，在第二终端回到同一个包根目录，等该轮 `budget.json` 已存在，再运行：

```bash
new_run="$PWD/source/data/research/new-v6-round"
export PYTHONPATH="$PWD/source/packages/core:$PWD/source/research:$PWD/source/apps/server"
export RMUC_RESOURCE_ROOT="$PWD/source"
.venv/bin/python source/scripts/v6_web_watch.py --run "$new_run" \
  --start-service --port 8768 \
  --playwright-module "$PWD/.browser-tools/node_modules/playwright-core"
```

8768 必须空闲；不会停止占用该端口的其他服务。守护立即启动原网页服务，并仅在同一轮 `web_acceptance` 与 marker 都就绪后运行一场 20 秒冻结 Rapier 对局、读取一段已保存全帧回放，等待 3D 资产完成加载。所有动作受已保存的预算和至多八分钟网页子窗口约束，结束时清理自己创建的服务/浏览器/对局。没有 watcher 时 campaign 仍会推进，但网页验收会缺失，不能报告完整通过。若使用已有长轮调度器，它已经负责创建最终网页 watcher，不要重复启动。

推荐显式选择 `baseline`：从 `models/main/checkpoints/baseline` 的完整发布代恢复，校验 `complete.json` 及模型、RNG、samplers、actor、experiment、coordinator 六文件哈希，并核验配置引用的冻结辅助 actor、脚本和雷达资源。tactical／revive／critic 三份 Adam、训练计数、RNG 与抽样历史来自同一代；新的配对基线就是该完整恢复代。不能只替换 `model.zip` 或 `selected.zip`，同时沿用另一个候选的优化器或协调器状态。

这里推荐的是显式选择完整代，**不是保证 `baseline` 是最终部署策略或效果最优**。八小时调度器会把通过分段续训检查的代材料化为 final 的完整 baseline，同时把 `main/initialized.zip` 设为原八小时前推理锚点，供最终公平对照。若 `selection.json` 最终选回 initialized，`--resume-generation baseline` 仍恢复 final 的完整代，不自动恢复 selected；final 没有新增梯度时，baseline 与 current 还可能是同一代。续训前必须同时查看 `selection.json`、`run-results.json` 和完整代哈希。若目的是从原锚点的优化器/RNG状态继续，应使用原 `sentry-v6-20261111-0333` 包的 baseline 配合新 CLI，或该锚点原有的完整发布代；不能用最终 `selected.zip` 拼接另一代 Adam。

`--resume-generation current` 保留旧行为，默认值也仍是 `current`；它用于继续研究当前模型，可能恢复尚未晋升或已退化的候选，不代表推荐策略。该模式优先使用 `main/coordinator-state.json`；`baseline` 必须使用该代自己的 coordinator，不回退到运行末尾状态。两者都只释放未完成 lease、丢弃部分 rollout 并重新开始回合，历史计数不清零。当前正式测速仅包含 Warp 的 1／2 环境和 Rapier 的 1 环境，各预热一次、测两个 30 步窗口；4／8 环境扩展尚未作为本轮证据。

注意：已封存的 `sentry-v6-20261111-0333.zip`（SHA `50ad16014f51bac4be51ec656a5785be6b214e9b9e84b90b9d4162aafdf3d58a`）内置源码早于这个参数，只支持旧的 `current` 恢复行为。上述 `--resume-generation baseline` 命令要求当前已审阅源码或后续八小时运行最终包；原包保持不变，不能声称其内置 CLI 已支持新入口。

仅检查材料化、暂不启动物理或训练时，可使用单独的新目录：

```bash
.venv/bin/python -m rmuc2027_research.rl.v6_campaign initialize-resume \
  --resume-bundle "$PWD" --resume-generation baseline \
  --output /absolute/path/to/materialization-check
```

不要把这个已存在的检查目录再传给 `run`。`run` 会自己创建目标目录和预算文件，恢复接口重绑资源到新位置，不要求原机 `/home/lx/...` 路径存在。恢复包包含既有数据准备结果，原始 2026 SQLite 保持外部数据，其文件名、大小和 SHA 由 manifest 记录；单纯继续 v6 PPO 不需要原始 DB、重新执行数据提取或辅助 BC。若以后要重跑 ETL，应另行取得匹配哈希的 DB，不能用随机同名文件替代。

在目标电脑上额外测试 1／2／4／8 环境时，使用上述独立材料化目录，并显式给出测速截止时间。例如下面为测速单独分配最多 20 分钟；这会执行真实物理和冻结策略推理，不会执行 PPO 更新：

```bash
target_deadline=$(.venv/bin/python -c 'import time; print(time.time()+1200)')
.venv/bin/python -m rmuc2027_research.rl.v6_preflight target \
  --root /absolute/path/to/materialization-check \
  --counts 1 2 4 8 --deadline-unix "$target_deadline"
```

结果写入该材料化目录的 `target-preflight/`，包括版本清单、每个进程日志、两窗实际仿真秒／墙秒、截断情况和建议环境数。再次测速须用新 `--output` 目录，不能覆盖旧证据。`requested_matrix_complete` 为 false 时不能声称完成 4／8 环境验证；只有实际完成的配置才能参与建议。这个可选入口不会扩大当前正式 `run` 的 1／2 环境范围，也不会自动把未来训练改为 4／8 环境。

ZIP 不含 `.git` 和 SSH 凭据。需要延续 GitHub 定时备份时，应事先让 `source/` 成为有 HEAD 的 Git 工作区、配置正确的 `origin`，并确认目标机能非交互 SSH 推送；训练脚本不会代建密钥。备份失败以 `github-backups/status.json` 为准，不等于本地检查点或 ZIP 已损坏。包外 `release-index.json` 才记录最终 ZIP SHA、异路径核验及远端备份结果，包内不会自引用自身 ZIP 哈希。
