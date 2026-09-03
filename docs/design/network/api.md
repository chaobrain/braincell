# Network API

本文是当前 Network、Synapse、Connection、连续位置采样和 Recording 接口的参考说明。最短的完整流程是：

```python
net = braincell.Network("demo", seed=7)
stim = net.add_population("stim", braincell.NetStim(size=4))
post = net.add_population("post", cell, layer="demo")

post.cell.place(at("dend_b", 0.7), braincell.mech.Synapse("ExpSyn", name="ampa"))
net.connect(
    "stim_to_post",
    source=stim.event_outputs["spike"],
    synapse=post.synapses["ampa"],
    weight=0.1 * u.uS,
)

post.cell.soma.record("v", braincell.observe.state("v"))
result = net.run(dt=0.025 * u.ms, duration=10.0 * u.ms)
```

## Network and Population

### `Network`

```python
braincell.Network(name=None, *, seed=0)
```

创建一个命名网络，统一管理 Population、Connection、时间、随机种子和运行时状态。

#### Parameters

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `str or None` | `None` | 可选的网络名称；非空字符串。 |
| `seed` | `int` | `0` | Network 级随机种子，用于派生未显式给定的局部随机流。 |

#### Main attributes

| Attribute | Meaning |
| --- | --- |
| `name` | Network 名称。 |
| `seed` | Network 级随机种子。 |
| `populations` | `population_name -> Population` 映射。 |
| `connections` | 全网 Connection 查询入口。 |

### `Network.add_population`

```python
Network.add_population(name, model, **metadata) -> Population
```

将一个已创建的模型或零参数 provider 注册为 Network Population。

#### Parameters

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `str` | required | Network 内唯一的非空 Population 名称。 |
| `model` | `Cell`, `NetStim`, `EventSequence`, or callable | required | 模型 owner，或返回其中一种模型的零参数 provider。 |
| `**metadata` | scalar or population-aligned array | - | 自定义 Population metadata；标量广播到 `size`，非标量首维必须等于 `size`。 |

#### Returns

| Type | Description |
| --- | --- |
| `Population` | 已解析并由当前 Network 管理的 Population。 |

#### Notes

- 同一个模型对象不能注册到多个 Population。
- metadata 不会转发给 `model`，也不会自动修改 Cell 参数。
- metadata 名称不能覆盖 Population 的保留属性或方法。
- Population 必须在 Network 初始化前添加。

```python
stim = net.add_population("stim", braincell.NetStim(size=4))
post = net.add_population(
    "post",
    cell,
    layer="molecular_layer",
    position=positions,
)
```

### `Population`

Population 是 Network 中一维模型集合的解析后句柄。正式属性、自定义 metadata 和 Cell 转发入口如下。

| Category | Name | Description |
| --- | --- | --- |
| identity | `name` | Network 内唯一名称。 |
| owner | `model` | 被管理的原始 `Cell`、`NetStim` 或 `EventSequence`。 |
| runtime dispatch | `kind` | Network 内部分派使用的只读类型。 |
| shape | `size` | Population 实例数。 |
| indexing | `ids` | 从 0 开始的 Population 局部索引。 |
| events | `event_outputs` | 该 Population 可提供给下游的命名事件输出。 |
| custom data | `metadata` | 自定义字段的只读映射。 |
| Cell forwarding | `cell` | Cell Population 的原始 Cell owner。 |
| Cell forwarding | `synapses` | Cell 拥有的逻辑 Synapse。 |
| Cell forwarding | `connections` | 以该 Cell 为目标的 routing rows。 |

`event_outputs` 表示 Population 向外提供什么事件，不表示它接收了哪些上游输入。指向 Cell Population
的上游连接通过 `post.connections` 查询。

```python
post.layer
post.metadata["layer"]

post.cell
post.synapses
post.connections
post.event_outputs["spike"]
```

### `Population.set`

```python
Population.set(**metadata) -> Population
```

设置经过 Population 维度校验的自定义 metadata。

#### Parameters

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `**metadata` | scalar or population-aligned array | - | 标量广播到 `size`；数组首维必须等于 `size`。 |

#### Returns

| Type | Description |
| --- | --- |
| `Population` | 当前 Population，支持链式调用。 |

### `Population.register_event_output`

```python
Population.register_event_output(source, *, name=None) -> EventSourceView
```

显式发布一个未参与 Connection、但需要出现在 `NetworkResult.events` 中的 Cell live event output。

#### Parameters

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `source` | `EventSource or EventSourceView` | required | 由该 Population 的 Cell 驱动的 live event source。 |
| `name` | `str or None` | `None` | Population 内唯一的 output 名称；省略时使用 source 自身名称。 |

#### Returns

| Type | Description |
| --- | --- |
| `EventSourceView` | 完整 source owner 的注册视图，即使传入的是 source 子集。 |

#### Notes

Cell Population 默认提供 `event_outputs["spike"]`，它检测 `RootLocation(0.5)` 所属 CV 的 canonical
threshold crossing。额外的具名 live EventSource 首次成功用于 `Network.connect()` 时会自动发布，通常
不需要手动调用本方法。同一个 source owner 只注册一次；同名不同 owner 会报错。

```python
monitor = braincell.VoltageCrossingSource(
    post.cell,
    location=at("dend_a", 0.4),
    threshold=-20.0 * u.mV,
    name="monitor",
)
post.register_event_output(monitor)
```

### `VoltageCrossingSource`

```python
VoltageCrossingSource(
    cells,
    *,
    location=None,
    threshold=<Cell.V_th>,
    direction="rising",
    name=None,
) -> VoltageCrossingSource
```

在 Cell 电压上声明一个或多个 live threshold detectors。它是可连接的 `EventSource`，也可以通过
`Population.register_event_output()` 只发布到结果中。

#### Parameters

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `cells` | `Cell or CellView` | required | 提供电压和 threshold state 的 Cell owner；CellView 可选择 Population members。 |
| `location` | locset expression or mask | root midpoint | 一个或多个连续 morphology 点；重复位置保留。 |
| `threshold` | voltage quantity | omitted | 省略时逐 endpoint 使用 Cell 自身的异质 `V_th`；显式值可为 scalar、每 Cell 的 `(P,)`、每位置的 `(1,L)`、`(P,L)` 或 flat endpoint rows。 |
| `direction` | `{"rising", "falling"}` | `"rising"` | rising 为 `v_prev < threshold <= v_next`；falling 为反向 crossing。 |
| `name` | `str or None` | `None` | 额外 event output 自动注册时所需的稳定名称。 |

#### Endpoint rows

若选择 `P` 个 Cell members，location 解析出 `L` 个点，则 source 有 `P * L` 行，顺序为
Population-major，再按 locset 原始顺序排列。以下只读数组把 source row 映射回模型：

| Attribute | Meaning |
| --- | --- |
| `population_index` | endpoint 所属 Cell member。 |
| `location_index` | endpoint 在已解析 locset 中的行号。 |
| `cv_id` | 连续位置最终所属的 CV。 |

省略 threshold 的 rising detector 与 Cell canonical spike 使用同一个 `cell.spike` 计算结果。显式 threshold
始终独立比较前后两步电压；省略 threshold 的 falling detector 也会独立使用 Cell `V_th` 比较。

```python
all_cv = braincell.VoltageCrossingSource(
    post.cell,
    location=post.cell.cv_midpoints,
    name="all_cv_spikes",
)
post.register_event_output(all_cv)

result = net.run(dt=0.025 * u.ms, duration=10.0 * u.ms)
events = result.events["post"]["all_cv_spikes"]
events.metadata["population_index"]
events.metadata["location_index"]
events.metadata["cv_id"]
```

不同模型的 canonical event output 如下。

| Population model | Canonical key | Output |
| --- | --- | --- |
| `Cell` | `"spike"` | root reference CV 的 threshold crossing。 |
| `NetStim` | `"spike"` | NetStim 生成的事件。 |
| `EventSequence` | `"event"` | 显式时间表中的事件。 |

## Cell Scope and Mechanism Views

Cell 与 CellView 使用同一套空间选择顺序：

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

空间 View 只保存索引，不复制 Cell、morphology 或 runtime arrays。机制的公共身份和最小逻辑行如下。

| Category | Type | Name | Extra identity | Logical row |
| --- | --- | --- | --- | --- |
| Channel | runtime model，例如 `Na_HH1952` | 用户声明的 owner，例如 `nav` | - | `(population, CV, type, name)` |
| Ion | implementation，例如 `SodiumFixed` | owner，例如 `na_pool` | species，例如 `na` | `(population, CV, type, name)` |
| Synapse | runtime model，例如 `ExpSyn` | group，例如 `fast_ampa` | stable logical ID | 一个独立 Synapse instance |
| Connection | source-to-synapse routing | connect call name | stable row ID | 一行 routing |

| View | Type selector | Name selector | Other selectors | Numeric slicing |
| --- | --- | --- | --- | --- |
| Channel | `by_type(type)` | `view[name]` | - | 不支持独立 logical row slicing |
| Ion | `by_type(type)` | `view[name]` | `by_species(species)` | 不支持独立 logical row slicing |
| Synapse | `by_type(type)` | `view[name]` | stable IDs | 支持，且保序 |
| Connection | `by_source_type(type)`, `by_synapse_type(type)` | connect/synapse name | stable row IDs | 支持，且保序 |

```python
cell.channels.by_type("IL")
cell.channels["leak_soma"]

cell.ions.by_species("na")
cell.ions.by_type("SodiumFixed")
cell.ions["na_pool"]

cell.synapses.by_type("ExpSyn")
cell.synapses["fast_ampa"]
cell.synapses["fast_ampa"][[0, 2]]
```

Channel/Ion 的 `get(field)` 和 `set(**fields)` 要求最终 View 只包含一个 `(type, name)` owner。Synapse
`get/set` 要求同一 type，但可以跨同 type 的多个 name。View 在初始化前读取声明参数；初始化后通过
logical-to-runtime mapping 读取 runtime parameter/state，不保存第二份数组。

### `Cell.place`

```python
Cell.place(locset, *mechanisms) -> Cell
CellView.place(locset, *mechanisms) -> CellView
```

在选定的 Population members 上放置独立 point mechanism instances。

#### Parameters

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `locset` | `LocsetExpr`, `LocsetMask`, `LocsetBatch`, or sequence | required | 共享位置、矩形批量位置，或每个 member 一个可不等长的位置集合。 |
| `*mechanisms` | point mechanism declarations | required | 要放置的 point mechanisms；异质 per-cell locset 当前用于 Synapse 声明。 |

#### Returns

| Type | Description |
| --- | --- |
| `Cell or CellView` | root 调用返回当前 Cell；Population view 调用返回当前 CellView。 |

#### Notes

`place` 保留输入位置顺序和重复位置。相同位置、相同 Synapse type/name 的多次放置仍是独立 logical
Synapse instances；runtime 按 Synapse type 组织 SoA storage。

## Synapse and Connection

### `braincell.connect`

```python
braincell.connect(
    name,
    *,
    source,
    synapse,
    pairing=None,
    weight=<synapse event default>,
    delay=0.0 * u.ms,
) -> ConnectionView
```

低层入口，用于单 Cell 或 Network 组装前，将 EventSource endpoints 绑定到已经存在的 Synapse rows。

#### Parameters

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `str` | required | 目标 Cell 内唯一的 Connection call 名称。 |
| `source` | `EventSource or EventSourceView` | required | 有序 source endpoints。 |
| `synapse` | `SynapseView` | required | 有序目标 Synapse；一次调用必须命中一个 synapse type 和一个 name。 |
| `pairing` | `PairingSpec or None` | `None` | endpoint 采样规则；省略时使用等长或 singleton 广播。 |
| `weight` | quantity or row-aligned quantity | synapse event default | 标量或每个生成 row 一个值；单位必须符合 Synapse event-input contract。 |
| `delay` | time quantity | `0.0 * u.ms` | 非负延迟；标量或每个生成 row 一个值。 |

#### Returns

| Type | Description |
| --- | --- |
| `ConnectionView` | 本次创建的具体 routing rows。 |

```python
braincell.connect(
    "drive",
    source=stim,
    synapse=cell.synapses["ampa"],
    weight=0.1 * u.uS,
    delay=0.5 * u.ms,
)
```

### `Network.connect`

```python
Network.connect(
    name,
    *,
    source,
    synapse,
    target=None,
    locations=None,
    pairing=None,
    weight=<synapse event default>,
    delay=0.0 * u.ms,
) -> ConnectionView
```

连接已注册的 source，并选择已有 Synapse 或在连接时快捷创建 Synapse。

#### Parameters

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `str` | required | 目标 Cell 内唯一的 Connection call 名称。 |
| `source` | `Population`, `EventSource`, or `EventSourceView` | required | 已注册 Population 提供的 source 或 source view。 |
| `synapse` | `SynapseView or Synapse` | required | 已有 Synapse rows，或要放置的新 Synapse 声明。 |
| `target` | `Population or CellView` | `None` | 使用 `Synapse` 时必需；已有 `SynapseView` 时禁止。 |
| `locations` | locset expression, mask, batch, or sequence | `None` | 使用 `Synapse` 时传给 `target.place` 的位置。 |
| `pairing` | `PairingSpec or None` | `None` | 仅支持已有 `SynapseView`。 |
| `weight` | quantity or row-aligned quantity | synapse event default | Connection event payload。 |
| `delay` | time quantity | `0.0 * u.ms` | 标量或 row-aligned 非负延迟。 |

#### Returns

| Type | Description |
| --- | --- |
| `ConnectionView` | 创建的 routing rows；快捷创建的目标可通过 `connection.synapse` 访问。 |

#### Notes

- source owner 与 target Cell 必须已经注册到同一个 Network。
- 额外具名 Cell EventSource 会在 Connection 成功后自动发布到 source Population。
- 快捷调用是原子的；place、广播、pairing 或 endpoint 对齐失败时不会留下孤立 Synapse、Connection
  或自动注册的 event output。

连接已有 Synapse：

```python
connection = net.connect(
    "stim_fast",
    source=stim.event_outputs["spike"][0:2],
    synapse=post.synapses["fast"],
    weight=0.08 * u.uS,
)
```

快捷创建 Synapse 并连接：

```python
connection = net.connect(
    "stim_slow",
    source=stim.event_outputs["spike"][2:4],
    target=post.cell[2:4],
    locations=at("dend_b", 0.7),
    synapse=braincell.mech.Synapse(
        "Exp2Syn",
        name="slow",
        tau1=0.5 * u.ms,
        tau2=5.0 * u.ms,
    ),
    weight=0.12 * u.uS,
)
```

### Endpoint alignment

省略 `pairing` 时，source 和 Synapse 使用以下对齐规则。

| Source length | Synapse length | Result |
| --- | --- | --- |
| `C` | `C` | 按输入顺序逐行 zip，产生 `C` rows。 |
| `1` | `C` | 同一个 source 广播到全部 Synapse。 |
| `C` | `1` | 全部 source 广播到同一个 Synapse。 |
| other unequal lengths | other unequal lengths | 报错；必须显式构造重复索引或使用 `pairing`。 |

```python
net.connect(
    "pairs",
    source=pre.event_outputs["spike"][[0, 0, 2]],
    synapse=post.synapses["ampa"][[1, 3, 3]],
)
```

### Connection queries

| Query | Meaning |
| --- | --- |
| `post.connections["stim_fast"]` | 目标 Cell 上一次具名 connect call。 |
| `post.connections.by_source_type("NetStim")` | 按 source type 筛选。 |
| `post.connections.by_synapse_type("ExpSyn")` | 按 Synapse type 筛选。 |
| `post.connections.by_synapse_name("fast")` | 按 Synapse name 筛选。 |
| `net.connections["post"]` | 目标 Population 的全部 active rows。 |
| `net.connections["post", "stim_fast"]` | 目标 Population 上一次具名 call。 |

连接名在目标 Cell 内唯一，不同目标 Population 可以同名。`len(ConnectionView)` 是 routing rows；
`len(net.connections)` 是 active named calls；`net.connections.n_rows` 是全网 active rows。

## Continuous Location Sampling

### `braincell.filter.sample`

```python
braincell.filter.sample(
    region,
    *,
    number,
    seed,
    measure="length",
    density=None,
    u_resolution=1e-10,
) -> SampleLocations
```

创建一个延迟解析的连续随机 `LocsetExpr`。表达式在获得具体 morphology 后才生成 `branch_id` 和连续
`branch_x`，因此可以直接传给 `Cell.place` 或 `Network.connect(locations=...)`。

#### Parameters

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `region` | `RegionExpr` | required | 连续 morphology 支持域。 |
| `number` | positive `int` | required | 样本数；保留抽样顺序和重复位置。 |
| `seed` | `int` | required | 该采样规则独立使用的显式随机种子。 |
| `measure` | `{"normalized", "length", "lateral_area", "area"}` | `"length"` | density 下方的基础几何测度。 |
| `density` | callable or `None` | `None` | 接收 `SamplingContext` 的非负、无量纲位置偏好。 |
| `u_resolution` | `float` | `1e-10` | 数值逆 CDF 的目标精度，范围为 `[1e-12, 1e-5]`。 |

#### Returns

| Type | Description |
| --- | --- |
| `SampleLocations` | 延迟到 morphology 已知时解析的 locset expression。 |

#### Probability measure

设所选区域为 \(R\)，用户 density 为 \(\rho\)，`measure` 指定的几何测度为 \(\mu_m\)。对任意
子区域 \(A\subseteq R\)，一个样本落入其中的概率为：

$$
P(X\in A)
=
\frac{\displaystyle\int_{A\cap R}\rho(x)\,\mathrm{d}\mu_m(x)}
     {\displaystyle\int_R\rho(x)\,\mathrm{d}\mu_m(x)}.
$$

因此 `density=None` 等价于 \(\rho(x)=1\)。“均匀”指相对于所选 `measure` 均匀，并不一定对
`branch_x`、物理长度或膜面积同时均匀。

在 branch \(b\) 的归一化坐标 \(x\in[0,1]\) 上，连续部分的未归一化概率密度是：

$$
q_b(x)=\rho\!\left(\mathrm{ctx}_b(x)\right)J_{m,b}(x),
$$

其中 \(J_{m,b}\) 是从 `branch_x` 到相应几何测度的 Jacobian。令 \(L_b\) 为整条 branch 的物理
长度，\(r_b(x)\) 为局部半径，则当前四种 measure 为：

| `measure` | Continuous Jacobian \(J_{m,b}(x)\) | Meaning |
| --- | --- | --- |
| `"normalized"` | \(1\) | 每个被选中的 `branch_x` 区间按归一化宽度贡献质量。 |
| `"length"` | \(L_b\) | 按物理弧长采样；长 branch 区间获得更大质量。 |
| `"lateral_area"` | \(2\pi r_b(x)\sqrt{L_b^2+(\mathrm{d}r_b/\mathrm{d}x)^2}\) | 按正长度圆台的侧面积采样。 |
| `"area"` | 与 `lateral_area` 相同 | 连续部分按侧面积，并额外包含零长度半径跳变的离散面积。 |

对于 `measure="area"`，零长度 segment 上从 \(r_0\) 跳变到 \(r_1\) 的环形面积作为位于该
`branch_x` 的离散 probability atom：

$$
A_k=\pi(r_0+r_1)|r_1-r_0|=\pi|r_1^2-r_0^2|.
$$

总归一化质量同时包含连续区间和离散 atoms：

$$
Z
=
\sum_c\int_{x_{c,0}}^{x_{c,1}}q_c(x)\,\mathrm{d}x
+
\sum_k\rho\!\left(\mathrm{ctx}_k\right)A_k.
$$

实现先依据每个连续 component 和 atom 的质量选择 component，再在连续 component 内通过局部 CDF

$$
F_c(x)
=
\frac{\displaystyle\int_{x_{c,0}}^x q_c(t)\,\mathrm{d}t}
     {\displaystyle\int_{x_{c,0}}^{x_{c,1}}q_c(t)\,\mathrm{d}t}
$$

反演 \(F_c(x)=U\), \(U\sim\mathrm{Uniform}(0,1)\)，得到连续 `branch_x`。atom 被选中时直接返回
其固定 `branch_x`。

#### SamplingContext

| Field | Type/shape | Meaning |
| --- | --- | --- |
| `branch_id` | scalar integer | 当前 morphology branch ID。 |
| `branch_name` | `str` | 当前 branch 名称。 |
| `branch_type` | `str` | 当前 branch morphology type。 |
| `branch_x` | scalar or inspected array | 当前连续 branch 坐标。 |
| `radius` | length quantity | `branch_x` 处的局部半径。 |
| `path_distance_to_root` | length quantity | 到 root reference 的树路径距离。 |
| `path_distance_from_soma` | length quantity | 到全部 soma branches 的最短树路径；没有 soma 时 root branch 为零距离区域。 |
| `local_position` | `(..., 3)` length quantity | morphology-local 3-D 坐标；需要完整 3-D geometry。 |
| `position` | `(..., 3)` length quantity | 当前等于 `local_position`，为后续 world transform 保留。 |

density 必须返回有限、非负、无量纲的 scalar，或与 `context.branch_x` 同形的数组。可通过
`braincell.filter.metric` 统一读取 `branch_x`、`radius`、`path_distance_from_soma` 和 `position`。

```python
def proximal_density(ctx):
    distance = braincell.filter.metric.path_distance_from_soma(ctx)
    return u.math.exp(-distance / (100.0 * u.um))


locations = braincell.filter.sample(
    braincell.filter.branch_in("type", ["dendrite", "apical_dendrite"]),
    number=200,
    seed=7,
    measure="area",
    density=proximal_density,
)
cell.place(locations, ampa)
```

## Endpoint Pairing

`pairing=` 从已有 source 和 Synapse candidate views 中生成临时局部索引，最终仍写入普通 Connection
rows，不建立第二套 topology 或 storage。它当前只接受已存在的 `SynapseView`。

### Strategy comparison

| Helper | Row count | Sampling order | Typical use |
| --- | --- | --- | --- |
| `independent(number, ...)` | 固定为 `number` | source 与 Synapse 独立采样 | 已知总 Connection 数。 |
| `source_first(number, ...)` | 固定为 `number` | 先 source，后条件采样 Synapse | Synapse 偏好依赖已选 source。 |
| `synapse_first(number, ...)` | 固定为 `number` | 先 Synapse，后条件采样 source | source 偏好依赖已选 Synapse。 |
| `by_source(degree, ...)` | source degrees 之和 | 每个 source 采样其 Synapse partners | 指定出度。 |
| `by_synapse(degree, ...)` | Synapse degrees 之和 | 每个 Synapse 采样其 source partners | 指定入度。 |
| `match_degrees(source_degree, synapse_degree, ...)` | 两侧 degree 和 | 展开两侧 stubs 后随机匹配 | 同时固定两侧 degree sequence。 |

### `braincell.network.connection.independent`

```python
braincell.network.connection.independent(
    number,
    *,
    source_score=None,
    synapse_score=None,
    source_replace=True,
    synapse_replace=True,
    group_by=None,
    seed=None,
) -> PairingSpec
```

固定总 row 数，分别从 source 和 Synapse candidate pools 独立采样。

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `number` | positive integer scalar or group-aligned array | required | 总 rows，或每个 `target_cell` group 的 rows。 |
| `source_score` | callable or `None` | `None` | source 边际非负权重。 |
| `synapse_score` | callable or `None` | `None` | Synapse 边际非负权重。 |
| `source_replace` | `bool` | `True` | source pool 是否放回采样。 |
| `synapse_replace` | `bool` | `True` | Synapse pool 是否放回采样。 |
| `group_by` | `None or "target_cell"` | `None` | 是否按 target cell 独立运行规则。 |
| `seed` | `int or None` | `None` | 显式局部 seed；给定后覆盖 Network seed 派生。 |

### `braincell.network.connection.source_first`

```python
braincell.network.connection.source_first(
    number,
    *,
    source_score=None,
    synapse_score=None,
    source_replace=True,
    replace=True,
    group_by=None,
    seed=None,
) -> PairingSpec
```

先采样 source，再让 `synapse_score(ctx)` 在已选 source 条件下为 Synapse candidates 赋权。

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `number` | positive integer scalar or group-aligned array | required | 生成 rows 数。 |
| `source_score` | callable or `None` | `None` | 第一阶段 source 边际权重。 |
| `synapse_score` | callable or `None` | `None` | 第二阶段条件 Synapse 权重。 |
| `source_replace` | `bool` | `True` | 第一阶段 source 是否放回。 |
| `replace` | `bool` | `True` | 同一固定 source 的 Synapse partners 是否可重复。 |
| `group_by` | `None or "target_cell"` | `None` | 可选 target-cell grouping。 |
| `seed` | `int or None` | `None` | 显式局部 seed。 |

### `braincell.network.connection.synapse_first`

```python
braincell.network.connection.synapse_first(
    number,
    *,
    source_score=None,
    synapse_score=None,
    synapse_replace=True,
    replace=True,
    group_by=None,
    seed=None,
) -> PairingSpec
```

先采样 Synapse，再让 `source_score(ctx)` 在已选 Synapse 条件下为 source candidates 赋权。

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `number` | positive integer scalar or group-aligned array | required | 生成 rows 数。 |
| `synapse_score` | callable or `None` | `None` | 第一阶段 Synapse 边际权重。 |
| `source_score` | callable or `None` | `None` | 第二阶段条件 source 权重。 |
| `synapse_replace` | `bool` | `True` | 第一阶段 Synapse 是否放回。 |
| `replace` | `bool` | `True` | 同一固定 Synapse 的 source partners 是否可重复。 |
| `group_by` | `None or "target_cell"` | `None` | 可选 target-cell grouping。 |
| `seed` | `int or None` | `None` | 显式局部 seed。 |

### `braincell.network.connection.by_source` and `by_synapse`

```python
braincell.network.connection.by_source(
    degree,
    *,
    synapse_score=None,
    replace=True,
    group_by=None,
    seed=None,
) -> PairingSpec

braincell.network.connection.by_synapse(
    degree,
    *,
    source_score=None,
    replace=True,
    group_by=None,
    seed=None,
) -> PairingSpec
```

分别为每个 source 指定下游 Synapse 数，或为每个 Synapse 指定上游 source 数。

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `degree` | non-negative integer scalar, array, or callable | required | 每个固定 endpoint 的 partner 数；callable 签名为 `(ctx, rng) -> counts`。 |
| `synapse_score` / `source_score` | callable or `None` | `None` | partner candidate 权重。 |
| `replace` | `bool` | `True` | 同一个固定 endpoint 内 partner 是否可重复。 |
| `group_by` | `None or "target_cell"` | `None` | 可选 target-cell grouping。 |
| `seed` | `int or None` | `None` | 显式局部 seed。 |

### `braincell.network.connection.match_degrees`

```python
braincell.network.connection.match_degrees(
    source_degree,
    synapse_degree,
    *,
    group_by=None,
    seed=None,
) -> PairingSpec
```

展开 source 与 Synapse stubs，然后随机一一配对。

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `source_degree` | scalar, array, or degree callable | required | 每个 source 的 stub 数。 |
| `synapse_degree` | scalar, array, or degree callable | required | 每个 Synapse 的 stub 数。 |
| `group_by` | `None or "target_cell"` | `None` | 可选 target-cell grouping。 |
| `seed` | `int or None` | `None` | 显式局部 seed。 |

两侧 degree 总和必须严格相等。该策略 v1 不接受 score 或额外约束。

### Degree helpers

| Signature | Distribution / meaning |
| --- | --- |
| `braincell.network.connection.degree.poisson(lam)` | Poisson degree callable。 |
| `braincell.network.connection.degree.binomial(n, p)` | Binomial degree callable。 |
| `braincell.network.connection.degree.negative_binomial(n, p)` | Negative-binomial degree callable，要求 \(p\in(0,1]\)。 |
| `braincell.network.connection.degree.empirical(values, probabilities)` | 从显式离散 PMF 采样 degree。 |

这些 callable 使用 `brainstate.random.RandomState`，返回非负整数 counts。

### Score and grouping contracts

score callable 接收 `ctx`，必须返回有限、非负、无量纲权重。权重 \(w_i\) 的归一化概率为：

$$
p_i=\frac{w_i}{\sum_j w_j}.
$$

条件采样时固定端形状为 `(B, 1)`，候选端为 `(1, K)`，score 应可广播到 `(B, K)`；边际 score
使用 `B=1`。每个被归一化的候选行至少需要一个正值。

| Context | Available information |
| --- | --- |
| Synapse | logical/location/CV/branch IDs、population index、radius、树路径距离、3-D position、`get(parameter)`。 |
| Source | source ID、type、name、owner、可用时的 population index、`get(field)`。 |

默认候选 endpoints 属于一个全局池。`group_by="target_cell"` 按 Synapse `population_index` 升序分组，
每组独立执行规则后拼接。固定行数规则的 `number` 可以是 scalar，或长度等于实际分组数的一维整数数组。

候选 source/Synapse views 不能含重复 ID，但生成结果允许重复。`replace=False` 在边际采样中分别作用于
对应池；条件采样只保证同一个固定 endpoint 的 partners 不重复，不保证全局 pair 唯一。生成零行会报错，
且不会修改 Connection store。

```python
net.connect(
    "distance_conditioned",
    source=pre.event_outputs["spike"],
    synapse=post.synapses["ampa"],
    pairing=braincell.network.connection.source_first(
        500,
        synapse_score=lambda ctx: distance_kernel(
            ctx.source.get("position"),
            ctx.synapse.position,
        ),
        seed=8,
    ),
)

pairing = braincell.network.connection.by_synapse(
    braincell.network.connection.degree.poisson(5.0),
    source_score=lambda ctx: source_preference(ctx.source),
    replace=False,
    seed=9,
)
```

直接 `braincell.connect` 的隐式 pairing seed root 为 0。`Network.connect` 从 Network seed 与 source
Population、target Population 和 connection name 派生，与 Population 添加顺序无关。显式 pairing
seed 完全覆盖 Network seed。

## Recording and Results

### `Cell.record`

```python
Cell.record(
    name,
    observable,
    *,
    period=None,
    frequency=None,
    start=0.0 * u.ms,
) -> RecordingSpec
```

在调用它的 Cell/CellView 空间 scope 上注册一个静态 observer。Recording 不调用 `place()`，不创建
point mechanism，也不改变 runtime layout。

#### Parameters

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `name` | `str` | required | Cell 内唯一 recording 名称。 |
| `observable` | observe descriptor | required | 由 `braincell.observe.*` 构建的观测声明。 |
| `period` | time quantity or `None` | `None` | 规则采样周期；与 `frequency` 互斥。 |
| `frequency` | frequency quantity or `None` | `None` | 规则采样频率；与 `period` 互斥。 |
| `start` | time quantity | `0.0 * u.ms` | 全局采样计划的开始时间。 |

#### Returns

| Type | Description |
| --- | --- |
| `RecordingSpec` | 由 root Cell 拥有的不可变 recording 声明。 |

`period` 和 `frequency` 都省略时每个 `dt` 采样。`period` 和 `start` 在首次 run、`dt` 已知时解析，
并必须是 `dt` 的整数倍。Recording 只能在初始化前添加。

### Observable constructors

| Signature | Selector | Result rows |
| --- | --- | --- |
| `observe.state(field)` | 当前空间 scope | 每个选定 `(population, CV)` 一行。 |
| `observe.channel(type=None, name=None)` | `type`、`name` 或全部 Channel owners | 每个匹配 Channel owner 和 `(population, CV)` 一行。 |
| `observe.ion(species=None, type=None, name=None)` | `species`、`type`、`name` 或全部 Ion owners | 每个匹配 Ion owner 和 `(population, CV)` 一行。 |
| `observe.synapse(type=None, name=None, ids=None)` | `type`、`name`、stable IDs 或全部 Synapse | 每个匹配 stable Synapse ID 一行。 |
| `observe.membrane_current()` | 当前空间 scope | 每个 `(population, CV)` 的总膜电流密度一行。 |
| `observe.clamp_current(reduce="sum")` | 当前空间 scope | 每个 `(population, CV)` 的外部 clamp 电流合计。 |
| `observe.clamp_current(reduce="none")` | 当前空间 scope | 每个匹配 clamp placement 一行。 |

Channel、Ion 和 Synapse builder 提供：

| Signature | Meaning |
| --- | --- |
| `.state(field)` | 保留每个匹配 mechanism row 的指定 state。 |
| `.current(reduce="sum")` | 将命中 current contributors 按 `(population, CV)` 求和。 |
| `.current(reduce="none")` | 保留每个 current contributor。 |

Connection 的 `weight` 和 `delay` 是静态 routing 参数，应通过 `ConnectionView` 查询，不属于 recording
observable。

### `ClampView`

`cell.clamps` 返回所有逻辑电流 clamp 的稳定 view。Clamp 没有 semantic name，字符串下标按类型选择；
类型筛选可继续使用普通位置索引：

```python
dc = cell.clamps["CurrentClamp"]
second_dc = dc[1]
cell.clamps.by_type(braincell.CurrentClamp).record("dc_inputs")
```

记录结果位于 `result.samples["dc_inputs"]`；每个 logical clamp 对应一列，不跨位置求和。该
`SampleBlock.time` 是实际刺激求值时间 `step_start + 0.5 * dt`。Solver 在整个主步内消费与 recording
相同的缓存值，包括 Runge-Kutta 的所有局部阶段。

```python
cell[[0, 2]].dendrite.cv[1:].record(
    "dend_v",
    braincell.observe.state("v"),
    period=0.1 * u.ms,
)
cell.soma.record(
    "nav_p",
    braincell.observe.channel(name="nav").state("p"),
    frequency=10.0 * u.kHz,
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

### `NetworkResult`

`Network.run` 返回不可变的 `NetworkResult`。

| Attribute | Structure | Meaning |
| --- | --- | --- |
| `time` | time quantity | 当前 segment 的 step times。 |
| `samples` | `population -> recording -> SampleBlock` | 新 Recording API 的规则样本。 |
| `events` | `population -> output -> EventSeries` | 稀疏 event outputs。 |
| `start_time`, `stop_time`, `dt` | time quantities | segment 边界和固定步长。 |
| `traces` | `population -> probe -> values` | legacy Probe compatibility。 |

```python
block = result.samples["post"]["dend_v"]
block.time
block.values
block.schema.rows

events = result.events["stim"]["spike"]
events.time
events.source_id
events.count
events.metadata
```

对于多位点 Cell event output，metadata 包含与 source endpoint 行对齐的只读 `population_index`、
`location_index` 和 `cv_id`。`source_id` 索引这些映射数组，而不是直接表示 Cell ID。

`SampleBlock.values` 第一维是规则采样时间，最后一维与 `RecordingSchema.rows` 一一对应。每个
`RecordingRow` 保存 population/CV/point/branch、field/unit，以及可用时的 mechanism category/type/name
和 Synapse ID。求和 current 的 `contributor_ids` 保存归约前 contributor positions。

### `NetworkResult.concat`

```python
NetworkResult.concat(parts) -> NetworkResult
```

合并时间连续且 schema 一致的多个运行结果。

#### Parameters

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `parts` | iterable of `NetworkResult` | required | 按时间排序的连续 segments。 |

#### Returns

| Type | Description |
| --- | --- |
| `NetworkResult` | 合并后的不可变结果。 |

所有 segments 必须具有相同 `dt`、相接的时间边界，以及相同的 sample/event Population、recording names
和 recording schemas。

## Lifecycle and Run

### `Network.run`

```python
Network.run(
    *,
    dt,
    duration,
    delay_quantization="nearest",
    event_backend="auto",
    brainevent_backend="jax_raw",
) -> NetworkResult
```

以固定 `dt` 推进 Network，并返回当前时间 segment 的结果。

#### Parameters

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `dt` | positive time quantity | required | 固定仿真步长。 |
| `duration` | positive time quantity | required | 当前调用推进的持续时间，必须是 `dt` 的整数倍。 |
| `delay_quantization` | `{"nearest", "ceil", "floor", "strict"}` | `"nearest"` | 将 Connection delay 映射到整数 steps 的规则。 |
| `event_backend` | `{"auto", "scatter", "brainevent"}` | `"auto"` | event delivery backend。 |
| `brainevent_backend` | `str or None` | `"jax_raw"` | 选择 BrainEvent 的具体 backend。 |

#### Returns

| Type | Description |
| --- | --- |
| `NetworkResult` | 当前半开时间区间 `[start_time, stop_time)` 的不可变结果。 |

#### Notes

- 第一次 `run` 隐式调用一次 `init_state`。初始化后不能添加 Population、Synapse、Connection 或 Recording。
- 首次运行后，`dt`、delay quantization 和 event backend 固定。
- 后续 `run` 从当前全局时间继续，并保留 Cell、Channel、Ion、Synapse 状态、threshold detector history、
  在途 delay events、recording schedule 和 RNG 状态。
- 因而在相同初始模型、seed、`dt` 和 runtime 配置下，连续 `run(5 ms)` 两次与一次 `run(10 ms)`
  产生相同的连续状态轨迹；区别是前者得到两个分段结果。
- 各 segment 使用相接的半开时间区间，因此边界时间不会被重复采样。

```python
first = net.run(dt=0.025 * u.ms, duration=5.0 * u.ms)
second = net.run(dt=0.025 * u.ms, duration=5.0 * u.ms)

assert first.stop_time == second.start_time
joined = braincell.NetworkResult.concat((first, second))
```

### `Network.reset_state`

```python
Network.reset_state(batch_size=None) -> Network
```

将已初始化 Network 的动态状态恢复到初始化基线，同时保留已经编译的 topology 和 runtime layout。

#### Parameters

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `batch_size` | `None` | `None` | Network batch execution 尚未实现；非 `None` 会报错。 |

#### Returns

| Type | Description |
| --- | --- |
| `Network` | 当前 Network，支持链式调用。 |

`reset_state` 将全局时间重置为 `0 ms`，恢复 Cell 和 Synapse 初始化状态，并清空 delay queues。它不会调用
`Cell.reset()`，不会返回可编辑声明阶段。

Network 的紧凑表示同时报告具名 connections 和实际 routing rows：

```text
Network(name='demo', populations=2, connections=2, rows=4)
```
