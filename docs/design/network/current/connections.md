# Network Events and Connections

事件源产生计数，Connection 把计数按 weight 和 delay 路由到 Cell 拥有的 Synapse。模型注册及运行见 [Network API](api.md)，突触自身状态见 [Synapse API](../../synapse/current/api.md)。下方 `text` 块用于签名与调用结构查询。

## Scheduled Event Sources

```text
NetStim(size=1, start=0*u.ms, number=1, interval=10*u.ms, noise=0.0, seed=None, name=None)
EventTable(source_index, time, event_id=None)
EventSequence(size, events, name=None)
EventSequence.from_times(times, *, name=None) -> EventSequence
```

NetStim 的 size 是正整数；start/interval 是标量或 `(size,)` 时间量，分别要求非负/正值。
number 是非负整数计数，noise 是 `[0,1]` 的无量纲比例，两者同样支持每源一值。
noise=0 产生周期事件；非零时每个间隔的一部分由指数等待时间决定。
seed=None 的独立源使用根 0，注册进 Network 时由 Network seed 和 population 名派生；
显式 seed 保持源自己的随机流。当前行为的替代设计见 [随机上下文](../proposals/random-context.md)。

EventTable 是扁平事件表，source_index 为 `(E,)` 非负整数，time 为同形的非负有限时间量，
event_id 可省略自动编号，显式 ID 需唯一。EventSequence.size 定义源总数，events 为 EventTable，
越界 source_index 抛出 IndexError。from_times 接受每源一个时间 Quantity 数组，可有不同事件数，返回新源。
这些对象在构造时形成事件日程，事件输出名称分别为 spike 和 event。

```python
import braincell as bc
import brainunit as u

source = bc.EventSequence.from_times(([0.1, 0.4] * u.ms, [0.2] * u.ms))
assert source.size == 2
assert len(source.events) == 3
assert source.events.source_index.tolist() == [0, 0, 1]
```

实现与校验见 [event.py](../../../../braincell/network/event.py)。

### `Population.register_event_output`

```text
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

```text
monitor = braincell.VoltageCrossingSource(
    post.cell,
    location=at("dend_a", 0.4),
    threshold=-20.0 * u.mV,
    name="monitor",
)
post.register_event_output(monitor)
```


### `VoltageCrossingSource`

```text
VoltageCrossingSource(
    cells,
    *,
    location=None,
    threshold=<Cell.V_th>,
    direction="rising",
    spk_fun=None,
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
| `spk_fun` | callable or `None` | `None` | 默认使用 Cell.spk_fun；自定义函数接收无量纲电压偏差，前向须为零点取 1 的硬阶跃，反向提供代理导数。 |
| `name` | `str or None` | `None` | 额外 event output 自动注册时所需的稳定名称。 |

#### Endpoint rows

若选择 `P` 个 Cell members，location 解析出 `L` 个点，则 source 有 `P * L` 行，顺序为
Population-major，再按 locset 原始顺序排列。以下只读数组把 source row 映射回模型：

| Attribute | Meaning |
| --- | --- |
| `population_index` | endpoint 所属 Cell member。 |
| `location_index` | endpoint 在已解析 locset 中的行号。 |
| `cv_id` | 连续位置最终所属的 CV。 |

同时省略 threshold 和 spk_fun 的 rising detector 复用 `cell.spike`。其余情况根据前后两步
电压计算事件；省略 threshold 时使用 Cell.V_th。到达阈值计一次，在阈值停留不重复发放。
检测器支持初始化前调用 `trainable(threshold=source)`，规则见
[训练接口](../../optim/current/api.md#synapse-connection-network)。

```text
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


## Synapse and Connection

### `braincell.connect`

```text
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

```text
braincell.connect(
    "drive",
    source=stim,
    synapse=cell.synapses["ampa"],
    weight=0.1 * u.uS,
    delay=0.5 * u.ms,
)
```

### `Network.connect`

```text
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

```text
connection = net.connect(
    "stim_fast",
    source=stim.event_outputs["spike"][0:2],
    synapse=post.synapses["fast"],
    weight=0.08 * u.uS,
)
```

快捷创建 Synapse 并连接：

```text
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

```text
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
