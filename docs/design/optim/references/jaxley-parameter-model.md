# Jaxley Parameter Model Reference

## Reference 状态

本文记录 Jaxley 参数选择和 simulation replacement 的实现调研，用于解释 BrainCell
Trainable Parameter Architecture 的取舍。它不是 BrainCell 公共 API 规范；规范性结论见
[Architecture](../architecture.md) 和 [API](../api.md)。

调研基于 Jaxley 仓库提交 `2638cca2665ec056c40c932dcee924192fc94da2`。后续 Jaxley
版本可能改变内部结构，BrainCell 不依赖这些私有实现。

## Parameter 与 State 声明

Jaxley mechanism 使用普通字典显式区分参数和动态状态。例如 HH channel 分别声明：

```text
channel_params
  HH_gNa
  HH_gK
  HH_gLeak
  HH_eNa
  HH_eK
  HH_eLeak

channel_states
  HH_m
  HH_h
  HH_n
```

这些字典提供参数名和默认值，但不是包含单位、validator、role metadata 的独立
`ParameterSpec` 类型。mechanism 插入 Module 后，参数和值被展开到 node/edge columns。

BrainCell 因此借鉴“显式区分 parameter/state”，但需要更丰富的 spec 保存 Quantity unit、
validator、field role 和 trainable flag。

## `make_trainable()` 数据结构

Jaxley 的 `make_trainable(key, init_val=None)` 在当前 View 上执行以下步骤：

1. 在 node 或 edge columns 中查找 `key`；
2. 过滤没有该参数的 rows；
3. 根据 `controlled_by_param` 分组 selected rows；
4. 为每组保存一行 indices；
5. 创建一个低维 trainable value array；
6. 将 value 和 indices 分别追加到 module-level collections。

核心状态可概括为：

```text
trainable_params
  [{key: values_per_group}, ...]

indices_set_by_trainables
  [rows_per_group, ...]
```

这不是把每个 channel object attribute 改成 trainable container，而是保存低维值及其要
覆盖的 dense parameter indices。

## Sharing 语义

View selection 决定 `controlled_by_param`：

- compartment view：每个 compartment 控制一个参数；
- branch view：每个 branch 控制一个参数；
- cell view：每个 cell 控制一个参数；
- edge view：每条 edge 控制一个参数；
- 更宽的 module view：可以让 selection 共享一个参数。

因此 sharing 不是 parameter object 自身的属性，而是 selection/grouping 产生的
`group -> indices` 映射。这一点与 BrainCell 的 logical row grouping 相同。

group 的 row 数量不同时，Jaxley 使用 `-1` padding 形成规则 index array。BrainCell
不需要复制 dataframe/padding 细节，但同样需要固定 shape 的 gather/scatter metadata。

## Initial Value

显式 `init_val` 可以是一个共享 scalar 或与创建参数数量相同的 list。`init_val=None` 时，
Jaxley 对每组当前 dense values 使用 `nanmean` 得到初值。

这对方便 parameter sharing 有利，但可能在声明 trainable 的瞬间改变原模型：同组原值
不一致时，它们下一次 simulation 都被均值覆盖。

BrainCell 不采用该默认行为：

- direct grouped source 要求当前值一致，否则报错；
- 保留不同空间比例时使用 frozen baseline + shared scale；
- arbitrary function 的 latent 初值由用户显式提供，不从 target 反求。

## Simulation Replacement

`get_parameters()` 返回低维 trainable value dictionaries。进入 simulation 前，Jaxley 的
`params_to_pstate()` 将它们与保存的 indices 组合为 parameter state：

```text
key
indices
val
```

`integrate(..., params=params)` 在初始化 simulation 参数时调用这条转换，并在
`get_all_parameters()` 中将低维 values 广播/覆盖到 dense node 或 edge parameter arrays。

因此用户可以把仿真视为显式函数：

```python
def simulate(params):
    return jx.integrate(cell, params=params)
```

JAX 对传入的 parameter PyTree 求导。更新后的 values 保留在独立 tree 中，并在下一次
integrate 时再次覆盖模型参数。

## `data_set()`

Jaxley 还提供 `data_set(key, value, param_state)`，用于在 JIT/vmap simulation 中构造一次性
parameter state，而不先把字段登记为长期 trainable。它同样采用 key + indices + value
结构，并传给 `integrate(param_state=...)`。

这说明参数选择和优化算法本身是解耦的：只要 simulation 能消费一个参数 replacement
tree，同一入口可用于 gradient descent、parameter sweep 或 black-box search。

BrainCell 的 ParameterSet 也应支持 gradient-based 和 gradient-free 写入，但长期 binding
由 owner-attached TrainableManager 管理，而不是每次显式传 param state。

## 与 BrainCell 的差异

| Topic | Jaxley | BrainCell decision |
| --- | --- | --- |
| parameter schema | params/states dictionaries | unit-aware `ParameterSpec` roles |
| root ownership | explicit parameter PyTree | owner-attached `nn.Param` graph state |
| simulation signature | `integrate(params=...)` | existing Cell run reads manager roots |
| sharing | View-controlled index groups | stable logical row groups |
| grouped initial | current group mean | require equality or use scale |
| units | simulator-specific numeric convention | preserve `brainunit.Quantity` |
| mapping | value replacement by indices | direct, scale, or latent callable binding |
| dense values | rebuilt during integrate | materialized into runtime buffers |
| optimizer | external | external BrainTools/user optimizer |

## BrainCell 采用的原则

- parameter 与 dynamic state 必须显式区分；
- selection 决定 sharing，自由度数量不等于 runtime row 数量；
- optimizer-facing roots 与 dense simulation values 应隔离；
- 参数 replacement 必须位于 differentiated simulation trace 内；
- 参数选择层不应绑定某一种 optimizer。

## BrainCell 不采用的行为

- 不要求用户把每个 rollout 写成显式 `simulate(params)`；
- 不将所有物理参数转成无单位数组作为公共边界；
- 不在 grouped current values 不一致时静默取平均；
- 不把映射限制为单字段 index replacement；
- 不让 View 或 ParameterSet 复制 owner 的 ParamState。

## Sources

- [Channel base](https://github.com/jaxleyverse/jaxley/blob/2638cca2665ec056c40c932dcee924192fc94da2/jaxley/channels/channel.py)
- [HH parameter/state declaration](https://github.com/jaxleyverse/jaxley/blob/2638cca2665ec056c40c932dcee924192fc94da2/jaxley/channels/hh.py)
- [`make_trainable`, `get_parameters`, and replacement](https://github.com/jaxleyverse/jaxley/blob/2638cca2665ec056c40c932dcee924192fc94da2/jaxley/modules/base.py)
- [`params_to_pstate`](https://github.com/jaxleyverse/jaxley/blob/2638cca2665ec056c40c932dcee924192fc94da2/jaxley/utils/cell_utils.py)
- [`integrate`](https://github.com/jaxleyverse/jaxley/blob/2638cca2665ec056c40c932dcee924192fc94da2/jaxley/integrate.py)
