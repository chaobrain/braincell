# Trainable Parameter API

## 文档状态

本文定义 BrainCell 首版可训参数选择和映射公共 API。设计依据和内部数据流见
[Architecture](architecture.md)，实现阶段和验收项见
[Implementation plan](implementation-plan.md)，更宽的模型优化能力边界见
[Design overview](design-overview.md)。

当前 P0 范围为 multi-compartment `ChannelView` 上的 `IL`、`Na_HH1952` 和
`K_HH1952`。Ion、Synapse、Connection、Network 聚合、
数据、loss、搜索、history、checkpoint 和诊断接口不在本文预先占用公共名称。

## Quick Start

```python
import braincell
import brainstate
import braintools

na = cell.on(soma).channels["na"]

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

候选公开符号为：

```text
braincell.trainable.parameter
braincell.trainable.scale
braincell.trainable.parameterized
braincell.trainable.TrainableManager
braincell.trainable.ParameterSet
braincell.trainable.ParameterBinding
```

Cell 公开：

```text
Cell.trainables       -> TrainableManager
```

View 公开：

```text
ChannelView.trainable(**fields) -> ChannelView
```

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
- target 必须由统一 schema 标记为连续、shape-preserving、trainable parameter；
- 同一逻辑 row/field 只能有一个 binding；
- 多字段调用原子注册，失败时不留下部分 roots 或 bindings；
- 注册不会立即写 runtime，第一次写入发生在 `init_state()` 的 materialization 阶段；
- 普通 `set()` 已建立的当前值可以作为 direct initial 或 scale baseline。

### P0 supported targets

首条实现链覆盖：

| Mechanism | Candidate fields |
| --- | --- |
| `IL` | `g_max`, `E` |
| `Na_HH1952` | continuous declared physical parameters |
| `K_HH1952` | continuous declared physical parameters |

动态 concentration、gate state、derived Nernst potential、`valence`、solver、substeps 和
topology 不可作为 P0 target。

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

P0 接受以下 `group_by`：

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
region、tuple keys 和 callable grouper 不属于 P0。

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

P0 要求稳定具名参数。positional-only 参数、`*args` 和不能稳定命名的匿名 varargs 被拒绝。
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

- P0 要求完整 tree；
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

用户不直接构造 binding；`View.trainable()` 根据 source 创建。P0 不提供 binding inverse、
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

## Deferred Owners

后续 owner 复用同一 source 和 manager，不创建新 namespace：

```python
synapses.trainable(
    tau=braincell.trainable.parameter(...)
)

connections.trainable(
    weight=braincell.trainable.scale(...)
)
```

Ion/Synapse/Connection binding 和 Network 聚合不属于 P0，只有在对应 runtime buffer 能
保持固定 shape、单位和 JAX trace 后才进入正式 API。

## Errors

P0 必须在明确边界拒绝：

- 空 selection 或一个 View 跨多个逻辑 owners；
- 未知字段、state、derived、static、整数、布尔或 topology target；
- 初始化后新增 binding；
- 重叠 row/field ownership；
- root name 冲突或共享对象使用冲突名称；
- direct grouped current values 不一致；
- initial、bounds 或 output 的 shape/单位不兼容；
- nested direct source 缺少 initial；
- parameterized signature 不稳定或 callable 非 JAX-traceable；
- ParameterSet tree 缺 key、多 key、shape/dtype/单位不匹配；
- 任意可能造成部分 root 或部分 target 写入的失败。

## References

- [Design overview](design-overview.md)
- [Architecture](architecture.md)
- [Implementation plan](implementation-plan.md)
- [BrainState Parameter Model](https://brainstate.readthedocs.io/concepts/the_parameter_model.html)
- [BrainState Transformation Semantics](https://brainstate.readthedocs.io/concepts/transformation_semantics.html)
- [BrainState `transform.grad`](https://brainstate.readthedocs.io/apis/generated/brainstate.transform.grad.html)
