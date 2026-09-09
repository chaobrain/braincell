# Trainable Parameter API

本文描述当前工作区的可训参数选择和映射接口。
设计依据和内部数据流见 [Architecture](architecture.md)，待实现方向见
[Roadmap](../proposals/roadmap.md)，更宽的模型优化能力边界见
[Design overview](../TODO.md)。

当前覆盖 multi-compartment Channel/Ion/Synapse 的构造签名参数、Connection weight、
电压事件检测器 threshold，以及 Network 参数聚合。机制不维护可训参数白名单。

## Quick Start

```python
import braincell
import brainstate
import braintools
import brainunit as u
from braincell.filter import AllRegion

branch = braincell.Branch.from_lengths(lengths=[20.0] * u.um, radii=[3.0, 3.0] * u.um)
cell = braincell.Cell(braincell.Morphology.from_root(branch), cv_policy=braincell.CVPerBranch(1))
cell.paint(AllRegion(), braincell.mech.Ion("SodiumFixed", E=50*u.mV))
cell.paint(AllRegion(), braincell.mech.Channel("Na_HH1952", name="na", g_max=1.0*u.mS/u.cm**2))
na = cell.channels["na"]

na.trainable(
    g_max=braincell.trainable.scale(
        group_by="all",
        transform=brainstate.nn.TanhT(0.5, 1.5),
        name="na.g_max.factor",
    )
)

cell.init_state()

parameters = cell.trainables.parameters()
states = parameters.states()

optimizer = braintools.optim.Adam(lr=1e-2)
optimizer.register_trainable_weights(states)
```

`braincell.trainable` 只提供参数 source 和管理类型。Adam、LBFGS、scheduler 等算法仍由
BrainTools 或用户代码提供。

## Public Namespace

当前 braincell.trainable.__all__ 导出：

```text
braincell.trainable.parameter
braincell.trainable.scale
braincell.trainable.parameterized
braincell.trainable.TrainableManager
braincell.trainable.ParameterSet
braincell.trainable.ParameterBinding
braincell.trainable.ParameterSource
```

Cell 公开：

```text
Cell.trainables       -> TrainableManager
```

View 公开：

```text
ChannelView.trainable(**fields) -> ChannelView
IonView.trainable(**fields) -> IonView
SynapseView.trainable(**fields) -> SynapseView
ConnectionView.trainable(weight=source) -> ConnectionView
```

Network 与检测器还提供以下入口（不同 owner 接受的字段见 [支持度](parameter-support.md)）：

```text
Network.trainables              -> Network aggregation facade
Network.prepare_run(...)        -> prepare fixed routing outside AD
Network.update()                -> one differentiable step
Cell.event_outputs["spike"].trainable(threshold=source)
VoltageCrossingSource.trainable(threshold=source)
```

Network 聚合门面不是新增公开导出类，具体类型名不加入 braincell.trainable.__all__。

## `View.trainable()`

```python
View.trainable(**fields) -> Self
```

为当前 View 选择的逻辑 rows 注册一个或多个 target field binding。

```python
na.trainable(
    g_max=braincell.trainable.parameter(),
    V_sh=braincell.trainable.parameter(group_by="all"),
)
```

### Parameters

| Name | Type | Description |
| --- | --- | --- |
| `**fields` | `field_name -> ParameterSource` | target 字段和对应的 direct、scale 或 parameterized source。 |

### Returns

返回原 View，支持 fluent declaration。

### Contract

- 只能在 `Cell.init_state()` 前调用；
- View 必须非空，并且只选择一个逻辑 mechanism owner；
- Channel/Ion/Synapse target 必须出现在其构造签名中；单位、形状和模型原有运算约束仍适用；
- 同一逻辑 row/field 只能有一个 binding；
- 多字段调用原子注册，失败时不留下部分 roots 或 bindings；
- 注册不会立即写 runtime；初始化时分配物理缓冲并进行 materialization；
- 普通 `set()` 已建立的当前值可以作为 direct initial 或 scale baseline。

### Channel 候选参数

候选项从 `__init__` 获取，包括继承和转发的已声明参数，而不是任意 `**kwargs`。

| Mechanism | Candidate fields |
| --- | --- |
| `IL` | `g_max`, `E` |
| `Na_HH1952` / `K_HH1952` | `g_max`, `V_sh`, `temp`, `q10`, `temp_ref` 等签名参数 |
| 其他 Channel | 该类构造签名中的参数，不需要另加白名单 |

表中为典型数值参数，并非额外白名单。整数默认值不代表不可微，例如浓度指数 `n`；
布尔比较可能产生零梯度，Python 控制流或字符串配置则可能在转换、初始化或求导时
自然报错。系统不承诺非零梯度，也不自动判断可辨识性。动态 gate state 和
不在构造签名中的内部常数不因本次扩展而成为候选项。

省略的数值参数可从签名读取默认值并在初始化前 `.set()`。必填参数没有虚构默认值；
若签名没有数值默认值，必须显式提供初值或覆盖值。通过 View 补值时，需覆盖该
运行时布局的所有有效行，不能猜测未选行的默认值。

温度派生的 sodium `phi` 在属性访问时重算；显式独立 `phi` 参数保持独立。

### Ion 候选参数与初始状态

Ion 使用同一参数 source、分组、区域选择和共享 root 机制，例如：

```python
import brainunit as u

cell.ions["pool"].trainable(
    Ci_initializer=braincell.trainable.parameter(0.001 * u.mM, group_by="all"),
    tau=braincell.trainable.scale(name="clearance"),
)
```

`Ci_initializer` 是初始参数，`Ci(t)` 是动态状态。loss 内先 `reset_state()` 再模拟；
reset 读取当前训练初值，不重置 optimizer root。固定 Ion 的 `Ci` 则是普通物理参数。
`Co/Ci/valence` 的 None 默认值按模型默认值解析。复杂 Ion 的默认初值在 reset 时
根据当前参数推导，显式初值覆盖仍然独立，未选区域继续使用模型默认关系。

固定 `E` 不因浓度改变而更新；InitNernst 在 init/reset/参数同步时刷新存储电位，
DynamicNernst/KineticIon 在读取时计算电位。`species_initializers` 的既有覆盖功能保留，
但不新增该字典内部字段的训练路径；已有具名 `BC_initializer` 等参数可直接选择。

示例见 [Ion learning](../../../../examples/optim/parameter_learning/ion_learning.ipynb)。

## `parameter()`

```python
braincell.trainable.parameter(
    initial=None,
    *,
    group_by="row",
    transform=brainstate.nn.IdentityT(),
    name=None,
) -> ParameterSource
```

创建直接物理参数 source：

```text
runtime[row] = q[group_index[row]]
```

### Parameters

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `initial` | physical value or `None` | `None` | root 初值；`None` 表示读取当前 target。 |
| `group_by` | group name | `"row"` | root 自由度分组。 |
| `transform` | BrainState Transform | `IdentityT()` | optimizer representation 到物理 root 的 transform。 |
| `name` | `str or None` | `None` | 稳定日志/checkpoint 名称。 |

### Initialization

top-level direct source 使用 `initial=None` 时，从 View 当前有效 target 值构造 root，因此
不改变此前的 `paint/set` 结果。

若一个 group 包含多个当前值，它们必须在单位归一后相等；否则报错，不取平均。需要
保留不同比例关系并共享一个自由度时使用 `scale()`。

显式 `initial` 必须与 target 单位兼容，并能广播到 root group shape。第一次 materialize
时，它会成为 target 值。

嵌套在 `parameterized()` 中的 `parameter()` 没有直接 target 可供读取，因此
`initial=None` 非法，必须显式提供初值。

### Example

```python
na.trainable(
    g_max=braincell.trainable.parameter(
        group_by="population",
        transform=brainstate.nn.SoftplusT(0.0 * u.mS / u.cm**2),
        name="na.g_max",
    )
)
```

## Grouping

当前接受以下 `group_by`：

| Value | Root identity | Meaning |
| --- | --- | --- |
| `"row"` | `(population, cv, owner-row)` | 每个 selected row 一个自由度。 |
| `"population"` | population member | 同一个 cell member 的 CV 共享。 |
| `"cv"` | CV identity | population 之间同一 CV 共享。 |
| `"all"` | one constant key | 全 selection 共享一个自由度。 |

若 population size 为 4、选择 10 个 CV，则 root DOF 为：

| Group | DOF | Runtime rows |
| --- | ---: | ---: |
| `row` | 40 | 40 |
| `population` | 4 | 40 |
| `cv` | 10 | 40 |
| `all` | 1 | 40 |

grouping 根据稳定 row metadata 建立，不要求 selection 可以 reshape 成规则矩阵。branch、
region、tuple keys 和 callable grouper 当前暂不支持。

## `scale()`

```python
braincell.trainable.scale(
    parameter=None,
    *,
    group_by="all",
    transform=brainstate.nn.IdentityT(),
    name=None,
) -> ParameterSource
```

保存当前 target 为 frozen row-aligned baseline，并创建无量纲 factor：

```text
runtime[row] = baseline[row] * theta[group_index[row]]
```

### Parameters

| Name | Type | Default | Description |
| --- | --- | --- | --- |
| `parameter` | `nn.Param or None` | `None` | 可选的现有共享 factor。 |
| `group_by` | group name | `"all"` | factor 的共享范围。 |
| `transform` | BrainState Transform | `IdentityT()` | 新建 factor 时使用的 transform。 |
| `name` | `str or None` | `None` | 稳定 root 名称。 |

### Contract

- `parameter=None` 时创建初值为 1 的 dimensionless `nn.Param`；
- 用户不需要传 `initial=1`；
- 第一次 materialize 仍得到原 target，不改变当前模型；
- transform bounds 约束 theta，不约束最终 runtime value；
- baseline 为零的 row 保持零，该乘法路径对 theta 的梯度也为零；
- frozen baseline 不进入 ParamState tree；
- 传入现有 `nn.Param` 时，该对象决定 transform，不能再传冲突 transform；
- 现有 parameter 的 physical shape 必须与 group root shape 兼容。

### Shared factor example

```python
theta = brainstate.nn.Param(
    1.0,
    t=brainstate.nn.TanhT(0.5, 1.5),
)

na.trainable(
    g_max=braincell.trainable.scale(theta, name="shared.g_factor")
)
k.trainable(
    g_max=braincell.trainable.scale(theta, name="shared.g_factor")
)
```

Na/K binding 各自保存 baseline，但同一个 factor 按对象身份去重，总自由度为 1。

## `parameterized()`

```python
braincell.trainable.parameterized(
    function,
    /,
    **arguments,
) -> ParameterSource
```

创建由一个普通函数生成 target 的 source。函数第一个参数为 `CVContext`，由 binding
提供；其余参数通过显式 keyword 与函数签名绑定。

```python
def conductance_profile(ctx, a, b, temperature):
    distance = metric.path_distance_from_soma(ctx)
    return temperature * (distance * a + b)

a = brainstate.nn.Param(a0, t=a_transform)
b = brainstate.nn.Param(b0, t=b_transform)

na.trainable(
    g_max=braincell.trainable.parameterized(
        conductance_profile,
        a=a,
        b=b,
        temperature=34.0,
    )
)
```

### Argument rules

| Argument value | Behavior |
| --- | --- |
| `nn.Param(fit=True)` | 注册为 root，调用函数时传入 `.value()`。 |
| `nn.Param(fit=False)` | 作为 fixed parameter 传值，不产生 ParamState。 |
| nested `parameter(...)` | 创建带显式 grouping 的 root，并按当前 row gather。 |
| Quantity/array/scalar | 作为 fixed argument。 |

当前要求稳定具名参数。positional-only 参数、`*args` 和不能稳定命名的匿名 varargs 被拒绝。
同一个 `nn.Param` 被多个 source 引用时按对象身份去重。

函数输出必须与 target physical unit 兼容，且 shape 在 JIT trace 内固定。函数及其参数
读取不得把 tracer 转为 NumPy array 或 Python scalar。

### Per-population coefficients

```python
na.trainable(
    g_max=braincell.trainable.parameterized(
        conductance_profile,
        a=braincell.trainable.parameter(
            initial=a0,
            group_by="population",
            transform=a_transform,
            name="profile.a",
        ),
        b=braincell.trainable.parameter(
            initial=b0,
            group_by="population",
            transform=b_transform,
            name="profile.b",
        ),
        temperature=34.0,
    )
)
```

scalar `a/b` 表示全 selection 共 2 DOF。对四个 population member 使用上面的 grouping
时共 8 DOF；每个 member 的所有 CV 复用同一组系数，空间变化来自 `ctx`。

系统不会根据旧 target 反求 latent 初值，也不预览或修正函数输出。用户负责提供合理的
`a/b` 初值。

## Transform、Bounds 与单位

所有约束直接写在 root `nn.Param.transform` 上：

```python
brainstate.nn.Param(
    initial,
    t=brainstate.nn.SigmoidT(lower, upper),
)
```

也可以使用 `TanhT`、`SoftplusT`、`SoftsignT` 或用户 transform。BrainCell 不固定
sigmoid，不维护另一份最终 target bounds，也不从 runtime bounds 反推 latent bounds。

lower/upper 按 root shape 使用标准广播：

- `group_by="all"`：scalar root；
- `group_by="population"` 且 `P=4`：root shape `(4,)`；
- `group_by="row"` 且 `P*C=40`：root shape `(40,)`。

direct physical root 和 runtime target 保留 `brainunit` 单位。scale factor 通常无量纲。
进入 autodiff 不要求整体去单位；loss 是否无量纲化属于调用方的 loss 合同。

## `TrainableManager`

```python
cell.trainables: braincell.trainable.TrainableManager
```

manager 隔离 root registry、binding 和 materialization。Cell 不另外重复公开
`trainable_parameters()` 等同义快捷方法。

### `parameters()`

```python
manager.parameters() -> ParameterSet
```

返回引用原始 roots 的稳定 ParameterSet，不复制 ParamState。

### `bindings()`

```python
manager.bindings() -> tuple[ParameterBinding, ...]
```

返回只读 binding inspection view。顺序按稳定 target identity 排列，不以声明调用顺序
作为 checkpoint identity。

### `materialize()`

```python
manager.materialize() -> None
```

读取当前 roots，计算所有 source，并原子更新对应 runtime parameter buffers。未初始化的
Cell 没有 runtime target，显式调用时报错；正常首次物化由 `init_state()` 完成。

## `ParameterSet`

### State and value access

```python
parameters = cell.trainables.parameters()

states = parameters.states()
physical = parameters.physical_values()
optimizer_values = parameters.optimizer_values()
```

| Method | Return |
| --- | --- |
| `states()` | stable name -> 原始 ParamState tree。 |
| `physical_values()` | transform 后、带单位的 root values。 |
| `optimizer_values()` | optimizer/raw representation tree。 |

### Atomic setters

```python
parameters.set_physical_values(candidate)
parameters.set_optimizer_values(candidate_z)
```

- 当前要求完整 tree；
- 写入前验证 keys、shape、dtype、单位和 finite 状态；
- 任一 leaf 失败时不写入任何 root；
- physical setter 使用 `Param.set_value()` 或等价正规路径；
- optimizer setter 写入现有 ParamState；
- 两者都保留 `nn.Param/ParamState` 对象身份。

ParameterSet 可用于多起点和黑盒搜索，但候选生成算法不属于本 API。

## `ParameterBinding`

binding inspection 至少公开只读 metadata：

```text
name
target_owner
target_field
row_keys
group_by
root_names
unit
baseline          optional
```

用户不直接构造 binding；`View.trainable()` 根据 source 创建。当前不提供 binding inverse、
reduce 或初始化后删除/替换 ownership。

## Lifecycle

自动 materialization 顺序为：

| Entry | Behavior |
| --- | --- |
| `Cell.init_state()` | runtime buffer 建立后、mechanism state 初始化前物化。 |
| `Cell.reset_state()` | 先物化，再重置 dynamic state。 |
| `Cell.run()` | rollout 入口保证当前 roots 已同步。 |

直接连续调用 `cell.update()` 时不会每一步自动物化。optimizer 更新 roots 后，低层调用方
在下一段 rollout 前显式调用：

```python
cell.trainables.materialize()
```

`reset_state()` 不回滚 roots 或 scale baseline。完整 `Cell.reset()` 清除 runtime；旧 runtime
buffer 引用随之失效。

## Synapse, Connection, Network

这些 owner 复用同一 source 和 manager，不创建新 namespace：

```python
synapses.trainable(
    tau=braincell.trainable.parameter(...)
)

connections.trainable(
    weight=braincell.trainable.scale(...)
)
```

`cell.event_outputs["spike"].trainable(threshold=...)` 绑定该位置的 Cell.V_th；显式
`VoltageCrossingSource(..., threshold=...)` 则有独立阈值。绑定仍须在初始化前声明。
Synapse 和 Connection 的 row 是逻辑突触/接触 ID，同一 CV 上的多个对象不会误合并。
`group_by="cv"` 则明确共享该 CV 的参数；其余 grouping 和 Channel/Ion 相同。

`network.trainables.parameters()` 聚合原始根，以 `population.local_name` 命名，
共享同一 nn.Param 时按对象身份去重。先在 host 调用 `network.prepare_run(dt=...)`，
再在 `brainstate.transform.for_loop` 中调用 `network.update()`，即可对完整网络做 BPTT。
`run()` 保留记录和事件表接口，其 host 结果转换不作为网络可微 rollout 接口。

Connection delay 显式拒绝训练；塑性规则及 weight_initial 尚未实现。
事件导数见 [架构](architecture.md#event-derivatives)，实验版 RTRL 见
[实验工作流](experimental-workflows.md)，未实现的塑性设计见 [proposal](../proposals/connection-plasticity.md)。

## Errors

当前在明确边界拒绝：

- 空 selection 或一个 View 跨多个逻辑 owners；
- 不在 Channel/Ion/Synapse 构造签名中的 target，或其他 owner 不支持的 target；
- 初始化后新增 binding；
- 重叠 row/field ownership；
- root name 冲突或共享对象使用冲突名称；
- direct grouped current values 不一致；
- initial、bounds 或 output 的 shape/单位不兼容；
- nested direct source 缺少 initial；
- parameterized signature 不稳定或 callable 非 JAX-traceable；
- ParameterSet tree 缺 key、多 key、shape/dtype/单位不匹配；
- 任意可能造成部分 root 或部分 target 写入的失败。

不额外拒绝整数或布尔候选，也不把零梯度视为错误。实际运算仍可能因静态控制流、
不合法类型等原因失败，这与签名候选发现是两件事。

## References

- [Design overview](../TODO.md)
- [Architecture](architecture.md)
- [Roadmap](../proposals/roadmap.md)
- [BrainState Parameter Model](https://brainstate.readthedocs.io/concepts/the_parameter_model.html)
- [BrainState Transformation Semantics](https://brainstate.readthedocs.io/concepts/transformation_semantics.html)
- [BrainState `transform.grad`](https://brainstate.readthedocs.io/apis/generated/brainstate.transform.grad.html)
