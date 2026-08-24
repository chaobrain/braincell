# Network API

## Population

```python
net = braincell.Network("demo", seed=7)
stim = net.add_population("stim", braincell.NetStim(size=4))
post = net.add_population("post", cell, layer="demo")
```

`Network.add_population(name, model, **fields) -> Population` 接受 `Cell`、`NetStim`、
`EventSequence` 或返回这些对象的零参数 provider。Population 正式字段包括 `name`、`model`、
`kind`、`size`、`ids`、`sources` 和 `fields`；自定义字段不能覆盖正式字段。

Cell Population 直接转发：

```python
post.cell
post.synapses
post.connections
post.sources["spike"]
```

`model` 是统一 owner 字段；普通 Cell 操作可使用上述转发属性。

## Cell scope and mechanism views

Cell 与 CellView 使用同一套空间入口。选择顺序始终是：

```text
population members -> branch name/type or region -> CV -> mechanism rows
```

```python
cell[[0, 2]]
cell[[0, 2]].dendrite
cell[[0, 2]].dendrite.cv[1:]

cell.soma.channels
cell.dendrite.ions
cell[1:3].synapses
cell[1:3].connections
```

空间 View 只保存索引，不复制 Cell、morphology 或 runtime arrays。最后一级机制身份如下：

| Category | Type | Name | Extra identity | Logical row |
| --- | --- | --- | --- | --- |
| Channel | runtime model，例如 `Na_HH1952` | 用户声明的 owner，例如 `nav` | - | `(population, CV, type, name)` |
| Ion | implementation，例如 `SodiumFixed` | owner，例如 `na_pool` | species，例如 `na` | `(population, CV, type, name)` |
| Synapse | runtime model，例如 `ExpSyn` | group，例如 `fast_ampa` | stable logical ID | 一个独立 Synapse instance |
| Connection | source-to-synapse routing | connect call name | stable row ID | 一行 routing |

`type` 表示共享的动力学方程，`name` 表示用户声明的逻辑 owner/group。同一 type 可以有多个 name。
Channel/Ion View 的数字行没有独立公共身份，因此不支持数字索引；应先选择空间，再按 type/name 过滤。
Synapse 与 Connection 有 stable IDs，可以进行保序数字切片。

```python
cell.channels.by_type("IL")
cell.channels["leak_soma"]       # 等价于 by_name(...)

cell.ions.by_species("na")
cell.ions.by_type("SodiumFixed")
cell.ions["na_pool"]

cell.synapses.by_type("ExpSyn")
cell.synapses["fast_ampa"]
cell.synapses["fast_ampa"][[0, 2]]
```

Channel/Ion 的 `get(field)` 和 `set(**fields)` 要求 View 最终只包含一个 `(type, name)` owner。
Synapse `get/set` 要求同一 type，但可以跨同 type 的多个 name。View 在初始化前读取声明参数；初始化后
通过 logical-to-runtime mapping 读取 runtime parameter/state，不保存第二份数组。

## Direct connection

低层接口用于单 Cell 或 Network 组装前：

```python
braincell.connect(
    "drive",
    source=stim,
    synapse=cell.synapses["ampa"],
    weight=0.1 * u.uS,
    delay=0.5 * u.ms,
)
```

Network 入口要求端点已经注册。连接已有 Synapse：

```python
connection = net.connect(
    "stim_fast",
    source=stim.sources["spike"][0:2],
    synapse=post.synapses["fast"],
    weight=0.08 * u.uS,
)
```

快捷创建 Synapse 并连接：

```python
connection = net.connect(
    "stim_slow",
    source=stim.sources["spike"][2:4],
    target=post.cell[2:4],
    locations=at("dend_b", 0.7),
    synapse=braincell.mech.SynapseSpec(
        "Exp2Syn", name="slow", tau1=0.5 * u.ms, tau2=5 * u.ms
    ),
    weight=0.12 * u.uS,
)
```

快捷调用是原子的；place、参数广播或端点对齐失败时不会留下孤立 Synapse。返回值始终是
`ConnectionView`，新建目标可通过 `connection.synapse` 访问。

## Continuous location sampling

`sample` 返回延迟解析的 `LocsetExpr`，因此可直接交给 `Cell.place` 或 Network 的
`locations` 参数。样本保留连续 `branch_x`、生成顺序和重复行；直到放置阶段才映射到 CV/point。

```python
def proximal_density(ctx):
    distance = braincell.filter.metric.path_distance_from_soma(ctx)
    return u.math.exp(-distance / (100 * u.um))


locations = braincell.filter.sample(
    braincell.filter.branch_in("type", ["dendrite", "apical_dendrite"]),
    number=200,
    seed=7,
    measure="area",
    density=proximal_density,
)
cell.place(locations, ampa)
```

底层概率测度为 `density(context) * measure`。`normalized` 按每条 branch 的归一化
`dx`，`length` 按物理弧长，`lateral_area` 按正长度圆台侧面积；`area` 还包含半径跳变处
零长度圆台的离散环形面积。默认 `measure="length"`、`density=None`。自定义 density
返回非负、有限、无量纲的 scalar 或与 `context.branch_x` 同形数组；context 提供 branch
标识、局部半径、到 root/soma 的树路径距离，以及在完整 3-D morphology 上的 position。
`braincell.filter.metric` 为 Sampling、CV 和 Synapse context 提供统一的 `branch_x`、
`radius`、`path_distance_from_soma` 和 `position` 读取函数。soma-relative distance 是到全部
soma branches 所组成区域的最短树路径；若 morphology 没有 soma，则整条 root branch 是
零距离 reference region。

## Alignment and queries

等长 views 做 zip；任一侧长度为 1 时广播。其他形状必须显式构造重复索引：

```python
net.connect(
    "pairs",
    source=pre.sources["spike"][[0, 0, 2]],
    synapse=post.synapses["ampa"][[1, 3, 3]],
)
```

```python
post.connections["stim_fast"]
post.connections.by_source_type("NetStim")
post.connections.by_synapse_name("fast")
net.connections["post"]
net.connections["post", "stim_fast"]
```

连接名在目标 Cell 内唯一，不同目标 Population 可以同名。`len(ConnectionView)` 是 rows；
`len(net.connections)` 是 active named calls，`net.connections.n_rows` 是全网 active rows。

## Endpoint pairing

已有 SynapseView 可以通过 `pairing=` 生成稀疏连接行；pairing 只在调用期间生成原始 views 中的
局部索引，最终仍写入普通 Connection rows，不建立第二套 topology/storage。

固定总行数时，两端可以独立采样，或先采一端再按该端条件采另一端：

```python
net.connect(
    "random_drive",
    source=pre.sources["spike"],
    synapse=post.synapses["ampa"],
    pairing=braincell.connection.independent(
        500,
        source_score=source_probability,
        synapse_score=synapse_probability,
        seed=7,
    ),
)

net.connect(
    "distance_conditioned",
    source=pre.sources["spike"],
    synapse=post.synapses["ampa"],
    pairing=braincell.connection.source_first(
        500,
        synapse_score=lambda ctx: distance_kernel(
            ctx.source.get("position"),
            ctx.synapse.position,
        ),
        seed=8,
    ),
)
```

已知单侧 degree 时，由 degree 总和决定 Connection rows：

```python
pairing = braincell.connection.by_synapse(
    braincell.connection.degree.poisson(5.0),
    source_score=lambda ctx: source_preference(ctx.source),
    replace=False,
    seed=9,
)
```

`by_source` 对称地指定每个 source 的下游数。`match_degrees(source_degree, synapse_degree)` 为两端
展开 stub，要求总和严格相等，再随机配对；v1 不在该策略中混入 score 或额外约束。

默认所有候选端点属于一个全局池。`group_by="target_cell"` 将 SynapseView 按
`population_index` 升序分组，每组独立执行规则后拼接：

```python
pairing = braincell.connection.by_source(
    2,
    group_by="target_cell",
    seed=10,
)
```

此例表示每个 target cell 内，每个 source 各连接两个 Synapse。固定行数规则的 `number` 可为 scalar，
或长度等于实际 target-cell 分组数的一维整数数组。

score callable 接收一个 `ctx`。条件采样时固定端形状为 `(B, 1)`，候选端为 `(1, K)`，返回值需可
广播到 `(B, K)`；marginal score 使用 `B=1`。Synapse context 提供 logical/location/CV/branch 字段、
半径、树路径距离、3-D position 和 `get(parameter)`；Source context 提供 ID、type、name、owner、
可用时的 population index 和 `get(field)`。score 必须有限、非负、无量纲，且每个归一化行至少
一个正值。

degree 接受非负整数 scalar、一维整数数组，或 `(ctx, rng) -> counts` callable；callable 中端点字段
是一维。内置 `degree.poisson`、`binomial`、`negative_binomial` 和 `empirical` 使用
`brainstate.random.RandomState`。

候选 source/synapse views 不得包含重复 ID，生成结果允许重复。`replace=False` 对 marginal 两端分别
生效；条件采样仅保证同一个固定端点的 partner 不重复，不保证全局 pair 唯一。pairing 产生零行会
报错且不会修改 Connection store。

直接 `braincell.connect` 的隐式 pairing seed root 为 0；`Network.connect` 从 Network seed 与
`source population + target population + connection name` 派生，与 population 添加顺序无关。
显式 pairing seed 完全覆盖 Network seed。`pairing=` v1 只接受已经存在的 SynapseView，不与
place-and-connect shortcut 同时使用。

## Recording

`record(name, observable, ...)` 注册到调用它的 Cell/CellView 空间 scope；`observe.*` 再选择该范围内
的 state 或 mechanism owners。Recording 声明不会调用 `place()`，不会创建 point mechanism，也不会
改变 runtime layout。

```python
cell[[0, 2]].dendrite.cv[1:].record(
    "dend_v",
    braincell.observe.state("v"),
    period=0.1 * u.ms,
)
cell.soma.record(
    "nav_p",
    braincell.observe.channel(name="nav").state("p"),
    frequency=10 * u.kHz,
)
cell.soma.record(
    "sodium_current",
    braincell.observe.ion(species="na").current(),
)
cell.record(
    "selected_g",
    braincell.observe.synapse(ids=synapse_ids).state("g"),
)
cell.soma.record(
    "membrane_current",
    braincell.observe.membrane_current(),
    start=0.5 * u.ms,
)
```

Observable selector：

| Observable | Selectors | State/current row semantics |
| --- | --- | --- |
| `observe.state(field)` | spatial scope only | one row per selected `(population, CV)` |
| `observe.channel(...)` | one of `type`, `name`, or none | one row per matching Channel owner and `(population, CV)` |
| `observe.ion(...)` | one of `species`, `type`, `name`, or none | one row per matching Ion owner and `(population, CV)` |
| `observe.synapse(...)` | one of `type`, `name`, `ids`, or none | one row per matching stable Synapse ID |
| `observe.membrane_current()` | spatial scope only | total membrane-current density per `(population, CV)` |

`state(field)` 始终保留机制行。`current(reduce="none")` 同样保留每个 contributor；默认
`current(reduce="sum")` 将命中项按 `(population, CV)` 求和。Connection 的 weight/delay 是静态 routing
参数，通过 ConnectionView 查询，不属于 recording observable。

`period` 与 `frequency` 互斥；都省略时每个 `dt` 采样。`period` 与 `start` 在首次 run、dt 已知时解析，
必须是 dt 的整数倍。Recording name 在一个 Cell 内唯一，且只能在初始化前添加。

### Result model

```python
result = net.run(dt=0.025 * u.ms, duration=10 * u.ms)
block = result.samples["post"]["dend_v"]

block.time
block.values
block.schema.rows
```

`SampleBlock.values` 第一维是规则采样时间，最后一维与 `RecordingSchema.rows` 一一对应。每个
`RecordingRow` 保存 population/CV/point/branch、field/unit，以及可用时的 mechanism category/type/name、
Synapse ID。求和 current 的 `contributor_ids` 是该 recording 归约前 contributor positions。

EventSource 输出保存在稀疏结构中：

```python
events = result.events["stim"]["spike"]
events.time
events.source_id
events.count
```

`SampleBlock`、`EventSeries` 和 NetworkResult 的 mappings/arrays 是不可变结果。连续片段通过
`NetworkResult.concat((first, second))` 合并；它要求 dt、时间边界、recording names 和 schemas 一致。
`traces/spikes` 暂时保留给 legacy Probe 与 dense spike compatibility，新接口以 `samples/events` 为准。

## Lifecycle and run

```python
result = net.run(dt=0.025 * u.ms, duration=10 * u.ms)
net.reset_state()
```

第一次 `run` 隐式调用一次 `init_state`。初始化后不能添加 population、Synapse 或 Connection。
连续 `run` 延续时间、状态、detector history、delay queues、recording schedule 和 RNG；
`reset_state` 恢复初始化基线但不返回可编辑态。dt、delay quantization 和 backend 在首次运行后固定。

Network repr 同时报告命名连接和实际 rows：

```text
Network(name='demo', populations=2, connections=2, rows=4)
```
