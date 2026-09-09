# Cell API

`braincell.Cell` 用 morphology、离散策略和机制声明构建电缆模型，并持有运行时状态。
`braincell.MultiCompartment is braincell.Cell`；独立的 `SingleCompartment` 与 Cell 的兼容方向见
[统一提案](../proposals/single-multi-compartment-unification.md)。内部计算见 [Cell 架构](architecture.md)。

| 任务 | 入口 |
| --- | --- |
| 构造与分段 | [Cell 构造](#cell-构造)、[CVPolicy](#cvpolicy) |
| 声明膜属性、刺激与突触 | [Paint 与 Place](#paint-与-place) |
| 初始化、重置与运行 | [生命周期](#生命周期)、[运行与结果](#运行与结果) |
| 查询几何、布局与状态 | [静态查询](#静态查询)、[运行时查询](#运行时查询) |
| 接入积分器 | [积分协议](#积分协议) |
| 空间 View、记录、训练和约化模型 | [相关接口](#相关接口) |

## 最小用法

单 branch、三个 CV 的被动膜模型，在中间 CV 注入电流，并记录该处电压：

```python
import braincell as bc
import brainunit as u
from braincell.filter import AllRegion, RootLocation

branch = bc.Branch.from_lengths(
    lengths=[60.0] * u.um,
    radii=[2.0, 2.0] * u.um,
    type="dendrite",
)
cell = bc.Cell(
    bc.Morphology.from_root(branch),
    cv_policy=bc.CVPerBranch(3),
    V_init=-65.0 * u.mV,
)
cell.paint(
    AllRegion(),
    bc.mech.Channel("IL", g_max=0.1 * u.mS / u.cm**2, E=-65.0 * u.mV),
)
cell.place(
    RootLocation(0.5),
    bc.mech.CurrentClamp(
        delay=0.2 * u.ms, durations=0.5 * u.ms, amplitudes=0.01 * u.nA,
    ),
)
cell.loc(RootLocation(0.5)).record("v_mid", bc.observe.state("v"))
result = cell.run(dt=0.025 * u.ms, duration=1.0 * u.ms)
voltage = cell.V.value
trace = result.samples["v_mid"].values
assert voltage.shape == (1, 3)
assert result.time.shape == (40,)
assert trace.shape == (40, 1)
```

`voltage` 是末态电压，`trace` 的两轴分别为采样时刻和观测行；两者都保留电压单位。
后续示例复用这里已经初始化的独立 `cell`、`result` 和导入。

## Cell 构造

完整调用签名：

```text
Cell(
    morpho, *,
    pop_size=1, cv_policy=None,
    V_th=0 * u.mV, V_init=None,
    spk_fun=braintools.surrogate.ReluGrad(),
    solver="staggered", subsolver=None, substeps=None,
    cache_ion_total_current=True, ion_channel_update_order="family",
    membrane_linearizer="point", name=None,
) -> Cell
```

| 参数 | 类型与默认值 | 含义 |
| --- | --- | --- |
| `morpho` | `Morphology`，必填 | 声明期形态；初始化时复制为运行时快照 |
| `pop_size` | 正整数或非空正整数序列，`1` | 共享形态的 population 尺寸；整数转为单元素 tuple |
| `cv_policy` | `CVPolicy or None`，`None` | `None` 选择 `CVPerBranch()` |
| `V_th` | 电压 Quantity，`0 * u.mV` | spike 检测阈值，可广播到 `pop_size + (n_cv,)` |
| `V_init` | 电压 Quantity、callable 或 `None`，`None` | `None` 读取各 CV 静息电位；数值广播到电压形状 |
| `spk_fun` | callable，`ReluGrad()` | 将阈值归一化后的无量纲电压差映射成 spike，用于阈值跨越检测及替代梯度 |
| `solver` | 注册名称或 callable，`"staggered"` | 单步积分器，Cell 调用 `solver(cell)`，由其就地推进状态 |
| `subsolver`、`substeps` | 名称/callable 与正整数，同为 `None` | 默认 backward Euler、1 子步；显式设置必须成对提供，用于 Markov channel / kinetic ion |
| `cache_ion_total_current` | bool，`True` | 为需要总电流驱动的离子模型缓存旧状态电流 |
| `ion_channel_update_order` | `"family" / "integration"`，`"family"` | 电压求解后的机制更新顺序，见 [调度](architecture.md#离子电流快照与调度) |
| `membrane_linearizer` | `"point" / "generic"`，`"point"` | 保留的线性化选择参数；当前两值没有对应两套不同的电压线性化实现 |
| `name` | `str or None`，`None` | Cell 名称 |

构造返回处于声明阶段的 Cell，已有静态离散预览，动态 `V/spike` 在初始化时创建。
`V_init` 为 callable 时按 `initializer(shape)` 求值，`shape=pop_size + (n_cv,)`，
应返回该形状的电压量；初始化与 `reset_state()` 都会重新求值。空间电缆参数的
`fn(CVContext)` 是另一种回调，见 [空间参数](../../filter/current/spatial-callable-parameters.md)。

`V_init`、`V_th`、`cv_policy`、`solver`、`spk_fun` 和 `membrane_linearizer`
可在声明阶段赋值，初始化后赋值抛出 `RuntimeError`。错误的形态或 policy 类型抛出
`TypeError`；非正 population 尺寸、非法枚举或不完整的 subsolver 配置会报错。

## CVPolicy

策略从 `braincell` 导入，构造结果传给 `Cell(cv_policy=...)`。现有策略都按 branch 划分，
因此 `CVPerBranch(1)` 在分叉形态上产生多个 CV。

| 完整构造形式 | 参数与划分规则 |
| --- | --- |
| `CVPerBranch(cv_per_branch=1)` | 正整数；每个 branch 使用相同 CV 数量 |
| `CVPerBranchList(cv_per_branch)` | 正整数序列；按 `morpho.branches` 顺序指定数量，长度必须等于 branch 数 |
| `MaxCVLen(max_cv_len, keep_odd=True)` | 正的标量长度量；数量约为 `max(1, ceil(length / max_cv_len))`，考虑几何容差 |
| `DLambda(d_lambda, frequency=100 * u.Hz, keep_odd=True)` | 正的无量纲 `d_lambda` 和正的标量频率；根据 branch 电长度计算数量 |
| `CVPolicyByTypeRule(branch_types, policy)` | 非空 branch 类型字符串 tuple 与 `CVPolicy`，创建一条分派规则 |
| `CompositeByTypePolicy(rules, default_policy)` | 规则 tuple 与默认 `CVPolicy`；按 branch 类型选择策略，后匹配的规则覆盖先匹配者 |

`keep_odd=True` 将计算所得的偶数 CV 数量提升到下一个奇数。`DLambda` 从默认电缆属性及
`paint(CableProperty)` 读取 `Ra/cm`，要求同一 branch 内为一致标量，不同 branch 可以不同；
同 branch 内不一致或使用 callable 时会报错。静息电位与温度不参与这项检查。

组合策略对整个 morphology 求解各子 policy，再取对应 branch 的结果。因此子 policy
仍须适用于整个输入形态，例如 `CVPerBranchList` 仍需覆盖全部 branch。
策略参数类型错误通常抛出 `TypeError`，非正数、长度不匹配或不一致的电缆属性抛出 `ValueError`。

扩展入口为抽象方法 `CVPolicy.resolve_cv_bounds(morpho, *, paint_rules=None)`，
返回按 branch 排列的 `tuple[tuple[(prox, dist), ...], ...]`，坐标沿 branch 长度归一化。
`paint_rules` 是 Cell 传入的内部声明，供依赖电缆属性的策略使用。
实现与分段计算见 [policy.py](../../../../braincell/_discretization/policy.py)。

## Paint 与 Place

```text
Cell.paint(region, *mechanisms) -> Cell
Cell.place(locset, *mechanisms) -> Cell
```

两者在声明阶段累积规则、使离散缓存失效，均返回原 Cell，支持链式调用。
初始化后调用抛出 `RuntimeError`；密度声明传给 paint，点声明传给 place。

### 区域声明

`region` 接受 `RegionExpr`，例如 `AllRegion()`；直接传入 `RegionMask` 会抛出 `TypeError`。
`*mechanisms` 是
`braincell.mech.CableProperty`、`Channel` 或 `Ion` 等密度声明。

| 声明 | 覆盖与重复声明行为 |
| --- | --- |
| `CableProperty` | 初始有一条全区域默认规则；按 CV 中点覆盖关系和声明顺序解析，同一 region 保留最后一次声明 |
| density channel | 独立保留机制声明，离散后检查 identity 冲突；部分覆盖按实际膜面积比例缩放电流 |
| ion | 挂载机制并解析参数，机制本身不按覆盖面积或体积比例缩放 |

### 点声明

`*mechanisms` 接受 CurrentClamp、Synapse、Probe 等点机制声明。
`locset` 的三种表达为：

| 输入类型 | 位置如何应用 |
| --- | --- |
| `LocsetExpr / LocsetMask` | 所有 population 成员共享同一组位置 |
| `LocsetBatch` | 矩形、逐成员对齐的位置；要求一维 population，行数等于成员数 |
| 每个成员一个 locset 的序列 | 各成员位置数可不同；要求一维 population、序列长度匹配，当前用于 Synapse |

place 保留输入顺序和重复位置；同一位置多次声明的突触仍是独立实例。
placement 保留原始连续位置，分别解析所属 CV 和电气 point：

| 放置位置 | 所属 CV | 电气 point |
| --- | --- | --- |
| CV 内部 | 包含该位置的 CV | 该 CV 中点 |
| branch 内部 CV 分界 | 右侧 CV | 右侧 CV 中点 |
| branch 端点或连接处 | 声明位置所在 branch 的首/末 CV | 对应边界节点，连接处可共享 |

例如两个区间 `[0, 0.5]`、`[0.5, 1]` 的 `x=0.5` 归后者；
`(branch0, 1.0)` 始终归 branch0 的末端 CV，即使电气节点与子 branch 共享。

端点刺激和突触由 staggered/DHS 的边界行消费。当前显式 Euler/RK 使用的通用导数遗漏了
边界输入反馈，切换 solver 会影响结果；原因见 [边界输入提案](../proposals/explicit-solver-boundary-inputs.md)。

## 生命周期

```text
Cell.init_state(batch_size=None) -> None
Cell.reset_state(batch_size=None) -> None
Cell.reset() -> None
```

`batch_size` 为 `None` 或正整数：`None` 使用 population 与空间轴，整数增加前置 batch 轴。
重置已有批次时传回相同的 `batch_size`。

| 方法 | 前置状态 | 状态与结构变化 |
| --- | --- | --- |
| `init_state` | 声明阶段 | 复制形态、构建运行时、创建电压和机制状态；时间归零，声明冻结 |
| `reset_state` | 已初始化 | 按当前参数与初态重新设置动态状态，时间归零并清空本地待处理输入；保留声明和运行时结构 |
| `reset` | 已初始化 | 丢弃运行时与动态状态，恢复声明期形态引用；保留 paint/place 声明，返回可编辑阶段 |

`init_state()` 重复调用、初始化前调用两种 reset，以及 Network 管理的 Cell 独立调用
这些生命周期方法，均抛出 `RuntimeError`。由 Network 管理时使用其生命周期入口。

```python
cell.reset_state()
assert cell.current_time == 0.0 * u.ms
cell.reset()
cell.cv_policy = bc.CVPerBranch(5)
cell.init_state()
assert cell.V.value.shape == (1, 5)
```

后续示例继续使用这个五 CV、已初始化的 `cell`。

## 运行与结果

### 连续运行

```text
Cell.run(*, dt, duration) -> RunResult
```

| 参数 | 类型 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `dt` | 正的标量时间 Quantity | 必填 | 固定积分步长 |
| `duration` | 正的标量时间 Quantity | 必填 | 本次运行时长，必须为 `dt` 的整数倍 |

首次调用自动初始化，后续调用从当前状态和时间继续，返回一个新的结果对象。
调用会就地更新电压、机制状态、spike 和当前时间。Network 管理的 Cell 独立 run 会抛出
`RuntimeError`；不带时间单位抛出 `TypeError`，非正时间或非整数步数抛出 `ValueError`。

### RunResult

从 `braincell.RunResult` 导出，由 `run` 返回。结果字段与映射只读，数组值用于后续分析。

| 字段 | 类型与含义 |
| --- | --- |
| `time` | 时间 Quantity，形状 `(n_steps,)`，位于 `[start_time, stop_time)` |
| `traces` | `probe_name -> values`；旧 Probe 接口的兼容结果，首轴为时间，尾轴随 probe 选择而定 |
| `samples` | `recording_name -> SampleBlock`；通过 `Cell.record` 声明的记录 |
| `start_time / stop_time / dt` | 本次时间段起止与步长，均为时间 Quantity |

```text
RunResult.concat(parts) -> RunResult
```

`parts` 为非空、按时间排列的 RunResult 序列。要求时间段连续、步长相同、recording 名称和
schema 一致，违反时抛出 `ValueError`。返回新的合并结果，输入对象保持原值；
`traces` 仅合并所有时间段共有的 probe 名称。

```python
first = cell.run(dt=0.025 * u.ms, duration=1.0 * u.ms)
second = cell.run(dt=0.025 * u.ms, duration=1.0 * u.ms)
joined = bc.RunResult.concat((first, second))
assert first.stop_time == second.start_time
assert joined.time.shape == (80,)
```

Recording 的采样规则与 SampleBlock 见 [记录接口](../../network/current/recording.md#recording-and-results)；
结果实现见 [run.py](../../../../braincell/_multi_compartment/run.py)。

### 单步推进

```text
Cell.update() -> spike_value
```

要求已初始化并设置环境 `dt`；缺少步长时抛出 `ValueError`。
调用就地推进状态并返回无量纲 spike，形状与当前 `Cell.V` 相同。
`t` 优先取环境值，未设置时读取 `cell.current_time`。

`update()` 不推进 `current_time`；循环驱动者负责时间。独立连续运行使用 `run()`。
单步调用示例：

```python
import brainstate

@brainstate.transform.jit
def one_step():
    with brainstate.environ.context(t=cell.current_time, dt=0.025 * u.ms):
        return cell.update()

spike = one_step()
assert spike.shape == cell.V.value.shape
```

## 静态查询

下面的属性在初始化前后均可读取。离散结果按需构建，声明或形态 revision 改变后重建。

| 只读属性 | 返回内容 |
| --- | --- |
| `morpho` | 当前 Morphology 引用；声明期为输入对象，初始化后为快照 |
| `paint_rules / place_rules` | 标准化声明 tuple |
| `pop_size / varshape` | population shape / `pop_size + (n_cv,)`，后者不包含可选 batch 轴 |
| `n_cv` | CV 数量，整数 |
| `cvs` | 按 CV 顺序排列的不可变记录 tuple |
| `cv_midpoints` | 每个 CV 一个连续中点位置的 `LocsetMask` |
| `node_tree` | NodeTree，`node_tree.n_point` 包含中点及边界节点 |
| `point_placements` | 保留原始位点和空间索引的点机制 placement 序列 |

CV 是单 branch 区间的静态记录，动态膜电压通过 `Cell.V` 读取：

| CV 字段 | 含义与单位 |
| --- | --- |
| `branch_id, prox, dist, midpoint` | branch 索引及沿长度归一化的位置 |
| `region, parent_cv, children_cv` | RegionMask 及相邻 CV 索引 |
| `cm, ra, v, temp` | 比膜电容、轴向电阻率、静息电位、温度，均带相应物理单位 |
| `length, area, radius_mid, diam_mid` | 长度、膜面积、中点半径与直径 |
| `r_axial_prox, r_axial_dist, r_axial` | 两个半 CV 及整个 CV 的轴向电阻 |
| `density_mech, point_mech, point_mech_roles` | 密度与点机制声明；roles 保存原始位置和局部几何角色 |

CV 与 point 的关系及几何计算见 [离散表示](architecture.md#cv-与-point)。

## 运行时查询

本节针对已初始化的详细电缆模型，初始化前调用抛出 `RuntimeError`。
约化模式的状态入口见 [相关接口](#相关接口)。

### 电压、布局与机制对象

| 属性或完整调用形式 | 返回值 |
| --- | --- |
| `V / spike` | 状态对象；用 `.value` 读取当前电压 Quantity / 无量纲 spike |
| `n_point` | 运行时电气点数量 |
| `voltage_shape` | 布局中的 `pop_size + (n_cv,)`；实际批次形状读取 `V.value.shape` |
| `layouts` | `MechanismLayout` tuple，各项的 `id` 用于布局查询 |
| `get_point_layouts(point_id)` | 覆盖该电气 point 的 layout tuple |
| `get_cv_layouts(cv_id)` | 归属于该 CV 的 layout tuple |
| `get_runtime_node(layout_id)` | 原运行时机制对象，非副本 |
| `get_ion(name)` | 原离子容器；按唯一名称或无歧义的类/家族别名解析 |

`cv_id`、`point_id` 为从 0 开始的整数，越界抛出 `IndexError`。
未注册的 runtime node 或 ion 抛出 `KeyError`；离子别名匹配多个容器时抛出 `ValueError`。
`get_runtime_node` 返回对象中的隐藏状态由对应机制定义，例如门控变量的 `.value`。

### 布局 Buffer 读写

```text
Cell.expected_state_shape(layout_id, var_name) -> tuple
Cell.get_state(layout_id, var_name) -> buffer_value
Cell.set_state(layout_id, var_name, value) -> None
Cell.get_point_state(point_id) -> dict
Cell.get_cv_state(cv_id) -> dict
```

`layout_id` 为布局 ID，`var_name` 为已注册 buffer 的字段名字符串。这里读取的是
运行时布局 buffer，包括机制参数和 clamp 序列；任意门控状态不一定注册在其中。
膜电压读写用 `Cell.V.value`，机制隐藏状态通过运行时机制对象访问。

| 操作 | 返回与修改行为 |
| --- | --- |
| `expected_state_shape` | 指定 buffer 的已登记形状 |
| `get_state` | buffer 的当前值，保留单位与布局形状 |
| `set_state` | 更新已有 buffer 并同步对应机制参数，返回 `None`；不创建新字段 |
| `get_point_state` | 新字典 `layout_id -> {var_name: point_value}` |
| `get_cv_state` | 新字典，包含该 CV 中点的点机制 buffer 与密度机制的 CV 切片 |

未知 buffer 键抛出 `KeyError`。赋值使用与现有 buffer 相容的单位和形状；
标量可填充整个 buffer，形状不匹配抛出 `ValueError`。优先先读后写，避免猜测布局尺寸：

```python
leak = next(layout for layout in cell.layouts if layout.target == "density")
g_max = cell.get_state(leak.id, "g_max")
assert g_max.shape == cell.expected_state_shape(leak.id, "g_max")
cell.set_state(leak.id, "g_max", 0.5 * g_max)
```

clamp 的 ragged 序列 buffer 还支持逐位点的时长、幅值序列，写入会更新 padding 和 mask；
常用操作通过 [ClampView](views.md#clampview) 完成。
布局读取、赋值和离子名称解析的实现见 [state.py](../../../../braincell/_compute/state.py)。

## 积分协议

以下接口供积分器在已初始化 Cell 上调用。`update()` 和 `compute_derivative()`
均不接收 `I_ext`；外部输入通过 clamp、突触和电流输入路径汇总。

| 完整调用形式 | 结果与状态影响 |
| --- | --- |
| `pre_integral() -> None` | 调用非独立积分机制的预处理钩子 |
| `compute_derivative() -> None` | 写入 `V.derivative` 与非独立积分机制的导数，供积分器组合 stage |
| `post_integral() -> None` | 将 delta 输入应用到 V，并调用非独立机制的后处理钩子 |
| `compute_membrane_derivative(V)` | 当前膜电流密度除以比膜电容 |
| `compute_axial_derivative(V)` | CV 轴向算子产生的电压导数，首次调用可能构建缓存 |
| `compute_voltage_derivative(V)` | 上述两个导数之和 |

三个电压导数方法接收与 `Cell.V.value` 形状相容的电压 Quantity，
返回相同电压空间的电压/时间 Quantity，不把结果写回膜电压。
机制状态从当前 Cell 读取，膜输入时间取环境 `t` 或 `current_time`。
通用导数的边界输入限制见 [电压与电流路径](architecture.md#电压与电流路径)。

## 相关接口

| 能力 | 权威说明 |
| --- | --- |
| population 与空间选择，channel/ion/synapse views | [Cell Scope and Mechanism Views](views.md#cell-scope-and-mechanism-views) |
| 事件连接 | [Synapse and Connection](../../network/current/connections.md#synapse-and-connection) |
| `Cell.record`、观测选择与采样 | [Recording and Results](../../network/current/recording.md#recording-and-results) |
| `View.trainable` 与可训练参数 | [Trainable API](../../optim/current/api.md) |
| `add_reduction(name, model)`、`use_model(name="detailed")` | [约化模型接入指南](../../reduction/current/model-integration-guide.md) |

源码与回归入口：[Cell](../../../../braincell/_multi_compartment/cell.py)、
[Cell 测试](../../../../braincell/_multi_compartment/cell_test.py)。
