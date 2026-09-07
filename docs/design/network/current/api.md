# Network API

`braincell.Network` 注册模型、冻结拓扑并按固定步长推进。事件与连接、配对、记录分别由下列专题维护。

| 任务 | 文档 |
| --- | --- |
| 事件源与连接 | [Connections](connections.md) |
| 端点配对与随机规则 | [Pairing](pairing.md) |
| 观测声明与结果 | [Recording](recording.md) |
| Cell 空间与机制视图 | [Cell Views](../../cell/current/views.md) |
| 连续位置采样 | [Filter Sampling](../../filter/current/sampling.md) |
| 突触动力学 | [Synapse API](../../synapse/current/api.md) |

内部执行见 [架构](architecture.md)。当前 `seed` 行为见 [配对随机数](pairing.md)，替换方向见 [随机上下文提案](../proposals/random-context.md)。

## 最小用法

这个完整例子把两个事件源分别接到两个 Cell 的中点突触，并读取电压和稀疏事件。

```python
import braincell as bc
import brainunit as u
from braincell.filter import AllRegion, RootLocation

branch = bc.Branch.from_lengths(lengths=[20.0] * u.um, radii=[5.0, 5.0] * u.um)
cell = bc.Cell(bc.Morphology.from_root(branch), pop_size=2, cv_policy=bc.CVPerBranch(1))
cell.paint(AllRegion(), bc.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV))
cell.place(RootLocation(0.5), bc.mech.Synapse("ExpSyn", name="ampa", tau=2.0 * u.ms))
cell.loc(RootLocation(0.5)).record("v", bc.observe.state("v"))
net = bc.Network("demo", seed=7)
stim = net.add_population("stim", bc.NetStim(size=2, start=0.1 * u.ms))
post = net.add_population("post", cell)
connections = net.connect("input", source=stim, synapse=post.synapses["ampa"], weight=0.001 * u.uS)
result = net.run(dt=0.025 * u.ms, duration=0.5 * u.ms)
assert len(connections) == 2
assert result.samples["post"]["v"].values.shape == (20, 2)
assert result.events["stim"]["spike"].source_id.shape == (2,)
```

后续 text 块是签名和依赖已有模型的调用形式；可执行运行片段可复用上述 net。

## Network and Population

### `Network`

```text
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

```text
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

```text
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

```text
post.layer
post.metadata["layer"]

post.cell
post.synapses
post.connections
post.event_outputs["spike"]
```

### `Population.set`

```text
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


## Lifecycle and Run

### `Network.init_state`

```text
Network.init_state(batch_size=None) -> Network
```

验证事件源归属并初始化各 Cell，返回当前 Network。batch_size 当前只接受 None；
初始化后结构冻结。重复调用直接返回，不重置已经运行的 Cell；重新开始轨迹使用 reset_state。

### 单步与训练入口

```text
Network.prepare_run(*, dt, delay_quantization="nearest", event_backend="auto",
                    brainevent_backend="jax_raw") -> Network
Network.update() -> dict[str, spike array]
```

prepare_run 在 tracing 外初始化并准备固定路由、队列和 dt，返回当前 Network；
参数单位和选项与 run 相同，至少需要一个 Cell population。重复调用复用固定配置，不重置动态状态；
改变已固定的 dt 或 backend 会抛出 RuntimeError。
update 需先 prepare_run，否则抛出 RuntimeError；它物化可训参数、推进一个 dt 并更新各 Cell 时间，
返回按 Cell population 名组织的 `cell.spike.value` 浮点数组；电缆模型形状为
`cell.pop_size + (cell.n_cv,)`，最后一轴保留全部 CV 的检测结果。Reduction 模型沿用自身事件输出形状。
它不构造 host 侧的 recording/event 结果表，可放入 brainstate.transform.for_loop/scan。
训练上下文见 [Trainable API](../../optim/current/api.md#synapse-connection-network)。

### `Network.run`

```text
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

```text
first = net.run(dt=0.025 * u.ms, duration=5.0 * u.ms)
second = net.run(dt=0.025 * u.ms, duration=5.0 * u.ms)

assert first.stop_time == second.start_time
joined = braincell.NetworkResult.concat((first, second))
```

### `Network.reset_state`

```text
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
