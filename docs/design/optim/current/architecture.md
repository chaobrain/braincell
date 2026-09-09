# Trainable Parameter Architecture

## 目标

参数系统需要让 BrainCell 模型暴露可训练自由度，同时不改变现有 mechanism 的动力学
实现和 runtime owner。它必须同时支持：

- 直接训练一个物理字段；
- 多个 runtime row 共享一个自由度；
- 保持当前空间分布、只训练比例；
- runtime field 由一个或多个 latent 参数和 `CVContext` 生成；
- BrainState graph 自动发现 roots；
- gradient-based 和 gradient-free 方法写入同一个 ParameterSet。

参数映射由 `braincell.trainable` 提供；完整 RTRL 等实验梯度引擎可从 `braincell.experimental.optim` 导入。
公共调用合同见 [API](api.md)，落地顺序见 [Roadmap](../proposals/roadmap.md)。

## 三层参数模型

```text
optimizer raw variable z
        | brainstate.nn.Param transform
        v
physical root q / latent theta
        | ParameterBinding
        v
runtime physical field, for example g_max
```

BrainState 管理第一层 transform、ParamState 和 autodiff。BrainCell 只管理 root 的空间
含义以及 root 到 runtime field 的映射。

root 不一定是原模型参数：

- direct：`g_max[row] = q[group(row)]`；
- scale：`g_max[row] = baseline[row] * theta[group(row)]`；
- function：`g_max[row] = f(ctx[row], a[group(row)], b[group(row)], fixed)`。

只有 root 是 ParamState。runtime field 是 materialized physical buffer，不进入 optimizer
tree。`brainstate.graph.states(cell, ParamState)` 因而看到 `q/theta/a/b`，不会看到每个 CV
已经计算出的 `g_max`。

## 所有权与隔离

实现集中在独立的 `braincell.trainable` 模块。由 Cell 持有 `trainables` 门面：

```text
Cell.trainables -> TrainableManager
  root registry
  ParameterBinding collection
  ParameterSet construction
  materialization

Network.trainables -> aggregation facade over Cell managers
```

Cell manager 是实际 owner，并作为 Cell graph 的子 module 保存 `nn.Param`。Network
manager 只形成跨 Cell 聚合 view，不复制 roots。

这使核心 Cell 的侵入保持在两个边界：

1. 构造并暴露一个 `trainables` manager；
2. 在 init/reset/run 的既定位置调用 manager materialization。

View 只把 target selection 和 source 交给目标 Cell manager，不保存 optimizer state。

## 不同 View 的归属

当前仓库的物理 ownership 已经适合统一参数系统：

| View | 实际 storage owner | Binding owner |
| --- | --- | --- |
| ChannelView | target Cell density/runtime layout | target Cell |
| IonView | target Cell density/runtime layout | target Cell |
| SynapseView | target Cell SynapseStore/runtime node | target Cell |
| ConnectionView | target Cell ConnectionStore/runtime delivery | target Cell |

`NetworkConnections` 只聚合查询 Cell-owned ConnectionView，没有第二份 connection columns。
因此训练 connection weight 时，`ConnectionView.trainable(weight=...)` 仍注册到目标
Cell manager。全网优化聚合 Cell managers，不在 Connection 或 Population 上增加
独立 manager。

delay 影响离散调度和 queue schema，显式拒绝训练。weight 使用独立物理 State，
delivery block 在运行时读取，不在 lowering 或编译外冻结。

## Parameter Schema

Channel、Ion 和 Synapse 的候选字段来自实际 `__init__` 签名，包括转发的父类签名，不再维护
`ParameterSpec` 科学分类白名单。内部仍生成默认值、单位验证所需的 metadata；数值
缓冲分配与训练资格是不同问题，非数值配置不因出现在签名中就被强制数组化。

只有签名中的参数可选择；内部 gate state、硬编码常数不会自动暴露。选择后可能
获得非零梯度、合法零梯度，或者原有数值转换、形状和静态控制流错误。dtype 不被
当作可学习性的判据。Synapse 的状态 schema、事件输入和模型有效性验证仍与参数发现分开。

## Runtime 参数列

Channel 的数值物理参数由非可训 `RuntimeParameterState(LongTermState)` 保存。状态内部按
`uniform`、`population`、`cv` 或 `row` 保存最小值，Channel 读取时才广播为执行矩形；旧的
`get_state()` 和 buffer inspection 仍获得矩形兼容视图。point mask 在读取 conductance 时
应用，因此 scalar `g_max` 不必为未 paint 点分配完整数组。

首次 materialization 冻结参数列的轴语义。optimizer 之后只改变同 shape 的 state value，
不会因数值从相同变为不同而触发第二步 JIT 重编译。
压缩轴同时考虑 binding 的分组和区域所有权，不能仅因初始值相同而合并独立区域。
浮点训练值写入整数默认缓冲时提升 dtype，避免截断数值并切断梯度。

Ion 复用相同的紧凑布局参数列；同名 Ion 跨布局合并到持久的完整矩形参数状态，
同步使用 JAX scatter，不把 tracer 转回 NumPy。自动 root 名包含 category，避免
同名 Channel/Ion 字段冲突。参数精度服从 JAX/BrainState 的配置，不再隐式依赖
旧 Ion NumPy 合并路径的 float64。

初始参数和动态物种分别存储。初值覆盖使用可跟踪的区域 mask，未覆盖点在 reset
时求模型默认值；完全显式的数值初值仍可作为 Quantity 读取。缓冲物平衡初值和
caiBase/caliBase 默认关系保存计算方式，不保存构造时的结果。
InitNernst 的存储电位使用 LongTermState，保持原有刷新时机并避免 JIT 缓存旧值。
Nannuli 派生的 vrat/dsqvol 动态求值，反应图与单次导数求值内的 factor 复用保留。

## TrainableManager

Cell manager 内部维护：

```text
TrainableManager
  roots             stable name -> nn.Param
  bindings          ordered ParameterBinding collection
  target ownership  logical row/field -> binding
  parameters()      ParameterSet facade
  materialize()     evaluate and scatter all bindings
```

注册是事务性的：source 解析、schema、单位、group、root name 和 target overlap 全部验证
成功后才提交。一个逻辑 row/field 只能由一个 binding 消费；同一个 `nn.Param` 可以被多个
binding 引用，并按对象身份去重。

自动生成的 root name 来自稳定 owner、field、group 或 function argument。显式名称优先，
重复名称指向不同对象时失败。共享对象的多个显式名称不一致时同样失败。

## ParameterBinding

binding 是 source 与 runtime target 之间的持续关系，至少保存：

```text
target category / owner / field
stable selected row keys
root references
group and gather indices
expected output unit and shape
source evaluator
optional frozen baseline
```

binding 只需要 forward materialization，不要求 inverse/reduce。direct 的物理 setter 通过
root transform inverse 完成；任意 latent function 不存在通用的 target-to-root 逆映射。

普通 `View.set()` 与 binding 的区别是：

- `set()` 立即修改声明 override 或 runtime buffer；
- `trainable()` 注册持续关系；
- optimizer 只更新 root；
- `materialize()` 根据当前 root 刷新 target。

已由 binding 占有的 target 不允许普通 set 静默替换关系，当前实现明确拒绝。
当前没有初始化后删除/替换 ownership 的公共接口。

## Grouping 与广播

View selection 展开为稳定 logical rows。当前 group key 为：

| Group | Root identity |
| --- | --- |
| `row` | `(population, cv, owner-row)` |
| `population` | population member |
| `cv` | CV identity |
| `all` | one constant key |

group engine 根据 row metadata 创建 `row -> root index` gather，不依赖矩形 reshape，因而
可以扩展到 ragged selection。

direct source 从当前 target 初始化 grouped root 时，同组物理值必须相等；否则不平均。
scale source 将当前不同值保存为 row-aligned baseline，只共享 factor。

对 `P=4, C=10` 的 selection：

```text
row         40 DOF
population   4 DOF
cv          10 DOF
all          1 DOF
runtime     40 rows in every case
```

branch、region、组合 key 和用户 grouper 延后，直到稳定 row metadata 可以为 checkpoint
提供可靠 fingerprint。

## Parameterized Function

custom source 是一次 deferred normal function call。binding 按函数签名绑定 trainable
和 fixed arguments，并为每个 logical row 提供 `CVContext`。

- `nn.Param(fit=True)` 是 root，调用时传 `.value()`；
- `fit=False` 或普通 Quantity/array 是 fixed argument；
- grouped nested source 在调用前按当前 row gather；
- raw vector Param 作为完整函数参数，不靠 shape 猜测 population/CV 轴；
- callable output 必须与 target unit 兼容且 shape 固定；
- 热路径不得转换为 NumPy 或 Python scalar。

scalar `a/b` 表示全 selection 共 2 DOF。若 `a/b` 按 population 分组，4 个 population
member 共 8 DOF；CV 变化来自 `ctx`，不是自动增加的 latent 轴。

任意函数没有通用 inverse。用户负责 root 初值；系统不会根据旧 `g_max` 反求 `a/b`，
也不会把最终 target bounds 反推到 latent bounds。

## Transform、Bounds 与单位

root 的约束完全由 `brainstate.nn.Param.transform` 管理。BrainCell 不把 bounds 固定为
sigmoid，也不再维护第二份 target physical bounds。

lower/upper 按 root shape 广播，而不是 runtime row shape。例如 population group 的 root
shape 是 `(P,)`，可以使用 scalar 或 `(P,)` bounds。

direct root 保留 target 的物理单位；scale factor 通常无量纲；custom root 由函数合同决定。
进入训练不需要整体去单位。只有 loss 在选择 canonical unit 或观测尺度后形成无量纲
scalar。

多个 runtime values 依赖同一 latent 时，逐 target bounds 会形成耦合约束。当前不自动求
约束交集；用户直接约束 latent，或构造始终满足条件的 parameterized function。

## Materialization 生命周期

现有 density callable 在 lowering 时求值一次，不能承载 optimizer 持续更新的 latent。
TrainableManager 提供 JAX-traceable materialization 路径：

1. `init_state()`：runtime buffer 建立后、mechanism state 初始化前；
2. `reset_state()`：先物化，再重置 gates 和 ion dynamic state；
3. `Cell.run()`：rollout 入口保证当前 roots 已同步；
4. 直接连续调用 `update()` 时不在每一步求值，用户在 root 更新后显式 materialize 一次。

materialization 必须位于 differentiated trace 内。若参数影响 reset 初值，仅在 run 开始后
刷新已经太晚，因此 reset 入口也必须接入。

`reset_state()` 不改变 roots 或 frozen baseline。完整 `Cell.reset()` 清除 runtime；重新
初始化必须从仍有效的声明和 manager metadata 重建 binding target，旧 runtime 引用不可复用。

## Ownership and Initialization

Synapse constructors declare their physical parameters. `parameter_info()` derives
default and unit metadata from those signatures; it is not a handwritten
trainability whitelist. The state schema and event-input contract remain separate.
Custom classes migrate their former `parameters` dictionary to an explicit
`__init__`, calling `super().__init__(size, name)` and `_init_parameters(...)`.
Legacy nonempty dictionaries raise a migration error instead of silently losing fields.

Synapse rows are logical synapse IDs; Connection rows are contact IDs. Colocated
objects remain distinct under row grouping. CV grouping deliberately ties them.
The common manager owns parameter roots, transformations, grouping, registration
rollback, and runtime materialization. Runtime physical parameter States are not
additional optimizer roots. Shared roots in Network are deduplicated by identity.

Units and source shape are checked during registration. Synapse registration also
validates jointly proposed values (`tau > 0`, `0 < tau1 < tau2`). Numerical constraints
are not Python checks inside the differentiated step: users must select parameter
transforms that preserve valid domains during optimization. Exp2Syn's normalization
factor is computed from the current time constants on every event application.

Reset clears dynamic states and event queues, then uses the current trainable
values. It does not restore the optimizer roots to their original values.
Ordinary setters reject binding-owned fields. Delay remains static and trainable
delay raises `NotImplementedError`. Plasticity and `weight_initial` are not added.

## Event Derivatives

Rising: `last < threshold <= next`. Falling: `last > threshold >= next`.
Arriving at equality emits once; staying at or leaving equality does not emit.
Both use `S(sign * (next-threshold)/20mV) * (1-S(sign * (last-threshold)/20mV))`,
where sign is +1 for rising and -1 for falling. `S` has a hard Heaviside forward
value (one at zero) and a surrogate backward derivative. A custom `spk_fun` must
preserve that forward contract. The default is the owning Cell's `spk_fun`.

An omitted detector threshold uses Cell.V_th; an explicit threshold has its own
State. Default spike and inherited-threshold views detect overlapping ownership.
Floating event values must survive weight multiplication, sparse delivery and
queues. Converting them to bool/integer would sever surrogate derivatives.

The brainevent delivery adapter keeps `coomv` for the forward result and supplies
its exact bilinear JVP with JAX scatter: `d(W z) = dW z + W dz`. This also preserves
units and repeated destinations. It avoids a dependency batching failure when
full RTRL vmaps weight tangents; it does not introduce another surrogate or
change the detector's event semantics. Reverse mode transposes the same JVP.

Fixed NetStim input does not require an event-time surrogate to learn synapse tau
or connection weight. Gradients into a presynaptic Cell or detector threshold do
require the voltage surrogate. A zero gradient is valid: no input, an insensitive
loss, or a surrogate with no support at the visited voltages can cause it. Hard
event timing finite differences are not a reference for surrogate derivatives.

### NEURON Boundary Reference

NEURON 9.0.1 [APCount](https://github.com/neuronsimulator/nrn/blob/9.0.1/src/nrnoc/apcount.mod)
uses `v >= thresh` and rearms below threshold. Its initialization can count an
initially suprathreshold value, unlike BrainCell's reset-to-zero spike state.
[Fixed-step NetCon detection](https://github.com/neuronsimulator/nrn/blob/9.0.1/src/nrncvode/netcvode.cpp)
uses a strictly positive threshold condition. CVode may interpolate event time.
Thus the arrival-equality convention is not a claim of complete NEURON equivalence.

## Differentiable Network Execution

Call `Network.prepare_run(dt=..., event_backend=...)` outside AD to fix routing,
delay quantization and queue shapes. `Network.update()` reuses the existing run
loop's single-step ordering and returns floating spikes keyed by Cell population.
Use `brainstate.transform.for_loop` or `scan`; read continuous observations from
Cell states. `Network.run()` remains the host recording/EventSeries interface.

The existing experimental full-state RTRL accepts the same prepared target. Its
carry includes every traced Cell, synapse and queue state, including cross-Cell
sensitivities. Do not replace it by independent per-Cell sensitivities. Comparisons
with BPTT use the identical surrogate and parameters fixed throughout a rollout.
The sensitivity carry has fixed shape as duration grows, but cost still scales
with state size times parameter count. Per-time-step optimizer updates and a new
online-learning algorithm are outside this change.

## 与 Jaxley 的对比

完整实现调研见 [Jaxley Parameter Model](../references/jaxley-parameter-model.md)。本架构只冻结
直接影响 BrainCell 的结论：

- 保留显式 gate/Markov 状态声明；Channel 参数由构造签名发现，不用 dtype 判断可微性；
- View selection 和稳定 row metadata 决定 sharing；
- 低维 root 通过 gather/scatter 映射到 dense runtime values；
- BrainCell root 由 `nn.Param` graph state 持有，不要求显式 `simulate(params)`；
- 保留物理单位、支持 arbitrary latent function，且不平均不一致的 grouped direct 初值。

## 架构不变量

- root 才是 ParamState，runtime target 始终是物理 materialized value。
- manager 是参数系统唯一 owner，View 和 ParameterSet 不复制 state。
- direct、scale 和 function source 共享一条 binding 主链。
- 参数 ownership、group、row fingerprint 和 output shape 在 JIT 内固定。
- reset dynamic state 不得回滚 roots。
- 任意 callable 不作为 checkpoint 数据序列化。
- 不在 BrainCell 内实现 optimizer 或通用局部最优证明。

## 梯度方法的实测依据

[多 CV scaling](../../../../benchmarks/performance/optim_gradient_scaling/results/bptt-rtrl-scaling.md) 与
[独立网络计时](../../../../benchmarks/performance/optim_gradient_scaling/results/synapse-network-cpu.md)
显示方法取舍依赖全网状态、独立参数数和轨迹长度：RTRL 的工作内存可以较小，但参数数增加后运行可能更慢。
这些证据支持保留显式方法选择，不能导出任意模型的自动选择阈值。
[参数拟合吞吐实验](../../../../benchmarks/performance/parameter_fitting/results/batch-size-and-throughput.md)
还表明更宽的 batch 提高设备吞吐时，训练质量不一定同步改善；容量和拟合效果需要分别评估。
