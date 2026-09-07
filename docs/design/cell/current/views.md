# Cell Spatial and Mechanism Views

View 保存对同一 Cell 的选择，读写共享声明或运行时数组。Cell 构造及根对象 paint/place 见 [Cell API](api.md)，观测声明见 [Recording](../../network/current/recording.md)。下方 `text` 块表示依赖已有 Cell 的查询和调用形式。

## 最小用法

```python
import braincell as bc
import brainunit as u
from braincell.filter import AllRegion, RootLocation

branch = bc.Branch.from_lengths(lengths=[20.0] * u.um, radii=[3.0, 3.0] * u.um)
cell = bc.Cell(bc.Morphology.from_root(branch), pop_size=2, cv_policy=bc.CVPerBranch(1))
cell.paint(AllRegion(), bc.mech.Channel("IL", name="leak", g_max=0.1*u.mS/u.cm**2, E=-65*u.mV))
selected = cell[1:2].channels["leak"]
selected.set(g_max=0.2*u.mS/u.cm**2)
cell[1:2].loc(RootLocation(0.5)).record("selected_v", bc.observe.state("v"))
result = cell.run(dt=0.025*u.ms, duration=0.1*u.ms)
assert result.samples["selected_v"].values.shape == (4, 1)
assert u.math.allclose(selected.get("g_max"), 0.2*u.mS/u.cm**2)
```

## Cell Scope and Mechanism Views

Cell 与 CellView 使用同一套空间选择顺序：

```text
population members -> branch name/type or region -> CV -> mechanism rows
```

```text
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

```text
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

### `CellView.place`

```text
CellView.place(locset, *mechanisms) -> CellView
```

在选定的 Population members 上放置独立 point mechanism instances。根对象的完整调用条件见
[Cell.place](api.md#paint-与-place)。

#### Parameters

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `locset` | `LocsetExpr`, `LocsetMask`, `LocsetBatch`, or sequence | required | 共享位置、矩形批量位置，或每个 member 一个可不等长的位置集合。 |
| `*mechanisms` | point mechanism declarations | required | 要放置的 point mechanisms；异质 per-cell locset 当前用于 Synapse 声明。 |

#### Returns

| Type | Description |
| --- | --- |
| `CellView` | 返回当前 Population view。 |

#### Notes

`place` 保留输入位置顺序和重复位置。相同位置、相同 Synapse type/name 的多次放置仍是独立 logical
Synapse instances；runtime 按 Synapse type 组织 SoA storage。


### `ClampView`

`cell.clamps` 返回所有逻辑电流 clamp 的稳定 view。Clamp 没有 semantic name，字符串下标按类型选择；
类型筛选可继续使用普通位置索引：

```text
dc = cell.clamps["CurrentClamp"]
second_dc = dc[1]
cell.clamps.by_type(braincell.CurrentClamp).record("dc_inputs")
```

记录结果位于 `result.samples["dc_inputs"]`；每个 logical clamp 对应一列，不跨位置求和。该
`SampleBlock.time` 是实际刺激求值时间 `step_start + 0.5 * dt`。Solver 在整个主步内消费与 recording
相同的缓存值，包括 Runge-Kutta 的所有局部阶段。

机制观测的组合用法见 [Recording scopes](../../network/current/recording.md#recording-scopes)。
