# Network Recording and Results

Cell 声明观测及采样计划，Network 按 population 汇总 samples 和 events。独立 Cell 的运行结果见 [Cell API](../../cell/current/api.md#运行与结果)。

## 最小用法

独立 Cell 使用相同的 recording 声明；Network 结果在外层增加 population 名。

```python
import braincell as bc
import brainunit as u
from braincell.filter import RootLocation

branch = bc.Branch.from_lengths(lengths=[20.0] * u.um, radii=[3.0, 3.0] * u.um)
cell = bc.Cell(bc.Morphology.from_root(branch), cv_policy=bc.CVPerBranch(1))
cell.loc(RootLocation(0.5)).record("v", bc.observe.state("v"), period=0.05*u.ms)
net = bc.Network()
net.add_population("post", cell)
result = net.run(dt=0.025*u.ms, duration=0.1*u.ms)
block = result.samples["post"]["v"]
assert block.values.shape == (2, 1)
assert len(block.schema.rows) == 1
```

## Recording and Results

### `Cell.record`

```text
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

Clamp 选择与逐刺激记录见 [ClampView](../../cell/current/views.md#clampview)。

### Recording Scopes

下面是已有多 population、soma/dendrite、nav 通道和所选突触 ID 的模型中的调用形式：

```text
cell[[0, 2]].dendrite.cv[1:].record("dend_v", braincell.observe.state("v"), period=0.1*u.ms)
cell.soma.record("nav_p", braincell.observe.channel(name="nav").state("p"), frequency=10*u.kHz)
cell.soma.record("sodium_current", braincell.observe.ion(species="na").current())
cell.record("selected_g", braincell.observe.synapse(ids=synapse_ids).state("g"))
cell.soma.record("membrane_current", braincell.observe.membrane_current(), start=0.5*u.ms)
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

```text
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

```text
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
