# Jaxley 参数模型调研

## 文档定位

本文解释 BrainCell trainable architecture 对 Jaxley 参数选择与 replacement 的取舍，不定义
BrainCell API。调研固定在 Jaxley commit `2638cca2665ec056c40c932dcee924192fc94da2`；
后续版本可能不同，BrainCell 不依赖其私有实现。

## 核心数据流

Jaxley mechanism 以普通字典区分 `channel_params`（如 `HH_gNa/gK/gLeak/eNa/eK/eLeak`）和
`channel_states`（`HH_m/h/n`）。插入 Module 后，值被展开到 node/edge columns；字典本身
不是带 unit、validator 和 role 的 `ParameterSpec`。

`make_trainable(key, init_val=None)` 的行为可压缩为：

```text
View selection
  -> filter rows containing key
  -> group by controlled_by_param
  -> trainable_params[{key: values_per_group}]
  -> indices_set_by_trainables[rows_per_group]
```

它不改写每个 channel attribute，而是保存低维 trainable values 及其覆盖的 dense indices。
selection 决定 sharing：compartment/branch/cell/edge view 分别产生对应 group；宽 module view 可让
selection 共享一个值。不同 group row 数用 `-1` padding 形成固定 index shape。

`init_val` 可以是 shared scalar 或每组一值；为 `None` 时，对组内 dense values 取 `nanmean`。
这会在组内原值不一致时静默改变模型。BrainCell 的 direct group 因而要求值一致；保留比例用
frozen baseline + scale，latent 初值由用户显式提供。

进入 simulation 前，`get_parameters()` 的低维 tree 经 `params_to_pstate()` 变为：

```text
key
indices
val
```

`integrate(..., params=params)` 再广播/覆盖 dense node 或 edge arrays，所以 JAX 可直接对
`simulate(params)` 的参数 PyTree 求导。`data_set(key,value,param_state)` 使用同一结构进行一次性
replacement，说明参数选择与 gradient、sweep、black-box search 是解耦的。

BrainCell 保留这条“低维 root -> 固定映射 -> dense runtime”主线，但 root 由 owner-attached
`TrainableManager` 持有，仿真不要求每次显式传入 parameter state。

## 对照与决策

| Topic | Jaxley | BrainCell |
| --- | --- | --- |
| schema | params/states dictionaries | unit-aware `ParameterSpec` roles |
| root ownership | explicit parameter PyTree | owner-attached `nn.Param` graph state |
| simulation | `integrate(params=...)` | Cell run 读取 manager roots |
| sharing | View-controlled index groups | stable logical-row groups |
| grouped initial | 当前组均值 | direct 要求一致；否则 scale/latent |
| units | simulator numeric convention | 保留 `brainunit.Quantity` |
| mapping | index replacement | direct、scale、latent callable |
| dense values | integrate 时重建 | materialize 到 runtime buffers |
| optimizer | external | BrainTools 或用户 optimizer |

BrainCell 采用：parameter/state 显式区分；selection 决定自由度；optimizer roots 与 runtime values
隔离；replacement 位于 differentiated trace 内；参数层不绑定 optimizer。

BrainCell 不采用：强制 `simulate(params)`；公共边界去单位；不一致 group 静默取均值；只支持
单字段 index replacement；由 View/ParameterSet 复制 owner ParamState。

## Sources

- [Channel base](https://github.com/jaxleyverse/jaxley/blob/2638cca2665ec056c40c932dcee924192fc94da2/jaxley/channels/channel.py)
- [HH declaration](https://github.com/jaxleyverse/jaxley/blob/2638cca2665ec056c40c932dcee924192fc94da2/jaxley/channels/hh.py)
- [`make_trainable` and replacement](https://github.com/jaxleyverse/jaxley/blob/2638cca2665ec056c40c932dcee924192fc94da2/jaxley/modules/base.py)
- [`params_to_pstate`](https://github.com/jaxleyverse/jaxley/blob/2638cca2665ec056c40c932dcee924192fc94da2/jaxley/utils/cell_utils.py)
- [`integrate`](https://github.com/jaxleyverse/jaxley/blob/2638cca2665ec056c40c932dcee924192fc94da2/jaxley/integrate.py)
