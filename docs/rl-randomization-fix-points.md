# 整队策略随机化后，权重参数哪里会不对

> 面向的问题：把 `friendly`（我方 4 台队友）和 `opponent`（对方 5 台）改成**每局整队随机**之后，哪些地方会算出错误的权重、错误的哈希、或者让续训/评估/导出悄悄用错对象。

结论：有 **9 处**必须改，分三个严重等级。其中 3 处会**直接报错拒绝**，4 处会**静默产生错误的元数据**（最危险），2 处只影响工具链。

---

## A 级：会静默出错（没有异常，但记录和校验全错）

### A1 `session.py:229` — 每局重置仍用固定来源

```python
sources = (resource(cfg['friendly']), resource(cfg['opponent']))
red, blue = sources if cfg['team'] == 'red' else sources[::-1]
```

这是随机化的**唯一注入点**，必须改成按 `self.seed` 采样。改错的方式有两种，都会造成训练数据错位：

- 用 `self.rng.randrange()` 采样：不可复现，同一 seed 重跑得到不同对手，`episodes.jsonl` 里的 `seed` 就无法重现战局。
- 采样了但**没写进 provenance**：见 A2。

正确做法是 `pools[key][self.seed % len(pools[key])]`，纯函数式，给定 seed 必然得到同一个对手。

### A2 `session.py:110` `provenance()` — 只记了固定的那一份

```python
def provenance(config):
    result = {'config': config, 'sources': {}}
    for name in ('friendly', 'opponent'):
        manifest, entries = load_submission(resource(config[name]))   # ← 只读 config 里的单值
        result['sources'][name] = {'version': ..., 'hashes': ...}
```

随机化后 `config['friendly']` 只是池里的一个条目，甚至可能压根不在池里。`provenance()` 会记录**一个与实际训练无关的版本号**。后果：

- `experiment.json` 的 `sources` 是假的，事后看日志无法知道某局到底对谁打的
- 续训比对（`learning.py:202` 比较 `sources`）会拿假数据去比，可能**误判为兼容而放行**
- 导出校验（`exporting.py:27`）同样基于假数据

**必须改**：`provenance` 要能拿到「本局实际选用的来源」。建议 `reset()` 把选中项塞进一个私有字段（如 `cfg['_active_sources']`），`provenance` 优先读它；同时 `sources` 里要额外记录**整个池的版本清单**，否则无法说明训练覆盖了哪些对手。

### A3 `session.py:101` `baseline_module()` — 执行底座绑定 friendly

```python
def baseline_module(config):
    manifest, entries = load_submission(resource(config['friendly']))
    source = entries[str(config['slot'])].read_text()
```

`GoalController.__init__` 调用它，而 `GoalController` 每局在 `reset()` 里新建。`baseline` 提供 `Context`、`revive_fields`、`weapon`、`ammo_fields`（`portable.py:226/236/251/254`），是**动作执行层**的一部分。

整队随机队友后，`baseline` 也必须跟着换成本局选中那份，否则会出现「队友是 A 策略、我自己的执行细节却按 B 策略」的错配。这不是装饰性偏差——`revive_fields` 直接决定复活时的行为参数。

**注意**：`fixed_opening()`（`portable.py:210`）只对 `slot == 1` 生效，哨兵是 slot 7 直接 `return None`，所以开局那段不受影响。但 `execute_goal` 受影响，别只改 opening 就以为改完了。

### A4 `learning.py:307` `evaluate()` — 用训练配置原样重建

```python
metadata = json.loads((model_path.parent/'experiment.json').read_text())
current = experiment(metadata['config'])
...
config = copy.deepcopy(metadata['config'])
env = SentryEnv(config, record_replay=True)
```

`evaluate` 直接从 `experiment.json` 的 config 重建环境。加了池之后：

- `config` 里带 `source_pools`，`provenance` 按 A2 的方式处理时会算出与 metadata 不同的 `sources`，`learning.py:310` 的校验会报 `Evaluation incompatibility: sources`
- 即使放行，评估也会**每个 seed 都用同一套对手**（因为 evaluate 只传单个 seed 给 env），而非训练时的分布

**必须明确一个语义决策**：评估时对手是「固定一份」还是「按 seed 抽池」。前者可比性高（同一对手横向比），后者贴近训练分布。建议评估支持 `--opponent` 显式指定，默认取池的第一个，并在报告里写明。

---

## B 级：会直接报错（改对了才发现，改错了直接跑不动）

### B1 `learning.py:200-209` — 续训校验必然失败

```python
for key in ('config', 'sources', 'sdk_hash', 'obs_version', 'obs_size', 'simulation'):
    if metadata[key] != previous[key]:
        raise ValueError(f'Resume incompatibility: {key}')
```

加 `source_pools` 后 `config` 变了；按 A2 正确处理 provenance 后 `sources` 也会变。所以 **`--resume` 一定被拒**。需要加 `init_from` 走「载入权重、重置步数」的路，而不是放宽这个校验。

### B2 `exporting.py:27` — 导出校验同样卡住

```python
if current['sources'] != metadata['sources'] or current['sdk_hash'] != metadata['sdk_hash']:
    raise ValueError('Fixed strategy or SDK changed since training; export refused')
```

随机池训练的模型，导出时 `provenance(config)` 拿到的是「池」，与训练记录的 `sources` 结构不同，必然 `export refused`。

### B3 `exporting.py:37-40` — 导出时内嵌哪份 baseline

```python
base_manifest, entries = load_submission(resource(config['friendly']))
baseline = entries[str(config['slot'])].read_text()
code += '_baseline_namespace = {}\nexec(' + repr(baseline) + ', ...'
```

导出的 submission 会把 `friendly` 的哨兵源码**烧进**策略文件，作为动作执行的底座。随机队友训练时没有「唯一 friendly」，这里必须明确选一份（通常选最常用的、或与评估一致的），并在 `rl-metadata.json` 里写明选的是哪份，否则导出的策略行为与训练时的执行底座不一致。

---

## C 级：工具链会失配（不影响正确性，但会让你看不懂）

### C1 `mjwarp/viewer.py:131`、`mjwarp/diagnose.py:129`

```python
sources = {side: resource(cfg['friendly']), 'blue' if side == 'red' else 'red': resource(cfg['opponent'])}
```

可视化和诊断仍按固定两份组装。池化训练后，看回放会显示**错误的对阵双方**。需要读本局的 `_active_sources` 或 `experiment.json` 里记录的实际来源。

### C2 `warp_kinematic/benchmark.py:50`

```python
with Battle(resource(config['friendly']), resource(config['opponent']), seed=args.seed, ...)
```

基准测速同样固定两份。影响不大（只是测吞吐），但如果要对比「随机对手下每局耗时」，就得改成同一套采样逻辑。

---

## 汇总表

| 位置 | 问题 | 后果 | 必须改 |
|---|---|---|---|
| `session.py:229` | 重置用固定来源 | 随机化根本没生效 | ✅ |
| `session.py:110` | provenance 只记单值 | 元数据全假，续训误判 | ✅ |
| `session.py:101` | baseline 绑 friendly | 执行层与队友错配 | ✅ |
| `learning.py:307` | evaluate 用 config 重建 | 评估报错或用错对手 | ✅ |
| `learning.py:200` | 续训比对 config/sources | `--resume` 必失败 | ✅ 用 `init_from` |
| `exporting.py:27` | 导出比对 sources | `export refused` | ✅ |
| `exporting.py:37` | 导出内嵌 friendly | 烧入错误底座 | ✅ 需显式指定 |
| `mjwarp/viewer.py:131` | 回放显示固定对阵 | 看错对手 | 建议 |
| `warp_kinematic/benchmark.py:50` | 测速固定对阵 | 数据口径偏 | 建议 |

---

## 最容易被忽略的三点

**1. `sources` 的结构会变，不只是值变。**
现在 `sources = {'friendly': {version, manifest, hashes}, 'opponent': {...}}`。池化后至少要变成「池清单 + 本局选中项」两层。所有比对这个字段的代码（`learning.py:202/310`、`exporting.py:27`）都会因为**结构不同**而失败，哪怕语义是对的。

**2. 采样必须由 seed 决定，不能由 rng 状态决定。**
`SentryEnv.reset()` 把 `match_seed` 传给 `session.reset(match_seed)`，同一 seed 必须复现同一局。若用 `self.rng` 采样，`num_envs=16` 时各进程的 rng 起始状态不同（`SubprocVecEnv(start_method='spawn')`），同一 seed 在不同 env 里会抽到不同对手，回放对不上。

**3. 队友随机比对手随机更危险。**
对手只影响训练信号分布；队友会影响 `baseline_module` 提供的执行底座（A3），等于**同时改了策略的肌肉和对手**。如果队友池里各策略的 `Context`/`revive_fields` 签名不一致，会直接抛异常；如果一致但语义不同，会静默产生难以归因的行为漂移。建议第一阶段**队友池只放 1 份**（保持固定），先只随机对手，把变量隔离出来。
