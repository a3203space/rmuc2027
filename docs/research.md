# 独立研究工具

研究源码位于 `research/`，与联机应用共享 2027 仿真和 SDK。主机不导入训练模块，也不提供策略上传、在线商店或排行榜。

## 策略开发与对战

```bash
uv sync --locked --group research
uv run --locked --group research rmuc2027-research new-strategy data/research/my-strategy
# 编辑生成的 strategy.py，实现 act(observation, state) -> (actions, new_state)
uv run --locked --group research rmuc2027-research bundle data/research/my-strategy/strategy.py data/research/my-bundle
uv run --locked --group research rmuc2027-research battle \
  --red data/research/my-bundle \
  --blue research/strategies/examples/claude_opus55_xhigh_2027_team \
  --output data/research/my-match.json
```

默认 `trusted-local` 只用于信任的本地代码。策略源通过独立子进程执行，并限制输出和响应时间。需要 Docker 隔离时，先构建 `docker build -f research/strategies/Dockerfile -t rmuc2027-policy:0.1 .`，对战加 `--sandbox docker`。

已有合法 2027 策略保留在 `research/strategies/examples/`。策略文件、manifest 与原哈希保持一致。

## 哨兵强化学习

神经网络依赖单独安装，不进入默认联机环境：

```bash
# 已有研究环境时跳过创建；研究依赖不安装到联机环境
uv venv --python 3.12 research/.venv
uv pip install --python research/.venv/bin/python -e packages/core -e research
uv pip install --python research/.venv/bin/python -r research/requirements-rl.txt --torch-backend cpu
research/.venv/bin/python -m rmuc2027_research rl check --config configs/rl/sentry.json --run-dir data/research/new-check
research/.venv/bin/python -m rmuc2027_research rl train --config configs/rl/sentry.json --total-steps 256 --num-envs 1 --device cpu --run-dir data/research/new-train
research/.venv/bin/python -m rmuc2027_research rl evaluate --model data/research/new-train/model.zip --seeds 1001 --sides red blue --duration 420 --run-dir data/research/new-eval
research/.venv/bin/python -m rmuc2027_research rl export --model data/research/new-train/model.zip --output data/research/new-policy
```

Windows 的解释器路径为 `research\.venv\Scripts\python.exe`。在已有独立研究环境中，也可直接使用 `python -m rmuc2027_research`，安装共享模块和研究包即可，无需客户端或联机服务。

已验证的 Linux/Python 3.10 CPU 依赖快照保存在 `research/requirements-rl-cpu.lock`，仅用于匹配的平台环境。

Actor 仍为 116 维合法局部观测，输出二维目标点。其余机器人由固定脚本控制，导航、瞄准、射击和回补由原执行模块负责。PPO 默认每 64 步更新，网络为两层 64 单元，默认回合在 60 个仿真秒截断；自然比赛持续 420 秒。完整配置位于 `configs/rl/sentry.json`。

旧模型需显式迁移，迁移保留原权重、优化器和原始实验元数据，记录运行时版本变化：

```bash
research/.venv/bin/python -m rmuc2027_research migrate-model \
  data/research/sentry-resume/model.zip data/research/sentry-resume-migrated
```

只接受已记录的原运行时与相同观测/动作契约，不通过修改原哈希绕过兼容检查。输出目录必须不存在。迁移后可恢复训练、评估和导出；不声称与旧引擎逐帧一致或胜率提升。
