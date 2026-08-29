# 电压轨迹与 Spike-Aware 参数训练

## 文档状态

本文是 BrainCell 参数学习的非规范性研究 reference，不是已经稳定的公共 API 规范。
它回答两个近期问题：没有动作电位的电压数据怎样尽可能拟合好，以及数据包含动作
电位时怎样避免尖峰时间偏移主导训练。本文保留的历史候选设计不能覆盖当前 API 和
Architecture 合同。

候选公共接口、依赖复用边界和分阶段实现范围见
[Trainable Parameter API](../api.md)。

非凸训练恢复的后续设计已拆分为三个相互约束的专题文档：

- [非凸搜索、学习率重启与局部盆地恢复](nonconvex-search-and-restarts.md)：定义
  checkpoint、plateau 检测、SGDR、局部扰动与全局筛选的控制流程；
- [Spike-Count 区域控制与可行 Checkpoint](spike-count-region-control.md)：定义
  spike-count region、双 archive、区域感知的候选接受规则和 curriculum；
- [非凸优化恢复的消融与测试协议](optimization-ablation-protocol.md)：冻结当前
  8 个初值基线、方法矩阵、预算、晋级条件、回归测试和必需产物。

数据集规模、mini-batch 梯度噪声和 GPU 并行宽度的实验结论见
[Batch Size, Dataset Scale, and GPU Throughput](batch-size-and-gpu-throughput.md)。

为避免术语混淆，本文约定：

- **subthreshold trace**：目标窗口中没有动作电位的连续膜电位轨迹；
- **spiking trace**：连续膜电位轨迹中包含动作电位，不只是 spike timestamp；
- **event trace**：经过阈值检测得到的离散或平滑 spike train；
- **physical parameter** `theta`：带物理单位、实际进入 mechanism 的参数；
- **optimizer parameter** `z`：无单位或规范化后的无约束优化变量。

## 结论先行

推荐采用一套公共的参数与仿真基础设施，但将损失和训练 curriculum 分成两条
路线。

| 数据类型 | 第一阶段目标 | 后期目标 | 不应作为唯一损失 |
| --- | --- | --- | --- |
| 无 spike | resting、稳态、时间常数和整体波形 | 多尺度 trace、导数、跨 protocol 泛化 | 单一 raw MSE |
| 含 spike | 进入正确的兴奋性区域、spike 数和大致时间 | 对齐、spike shape、ISI、AHP 和 subthreshold | 未对齐的 raw MSE 或 hard clipping |

具体建议如下：

1. 所有待学习的物理参数先从无约束空间映射，并保留单位、范围、共享范围和
   mask 元数据；不要直接优化裸的 `g_max` 数组。
2. 无 spike 数据用 robust voltage、多尺度、导数和生理特征的组合损失；含
   spike 数据增加对齐和 event loss，并按阶段逐步引入动作电位形状。
3. spike 附近可以用由**目标数据固定生成**的 mask 暂时降低权重，但不能根据
   当前预测动态决定 mask，也不应永久丢掉尖峰，因为 `gNa`、`gK` 等参数正是由
   spike shape 约束。
4. 采用多 protocol、train/validation sweep 拆分和 multi-start。低 trace loss
   不能证明恢复了真实参数，生物物理模型经常存在补偿和不可辨识性 [6]。
5. 长轨迹先使用 checkpoint/rematerialization；只有在确有显存或梯度问题时才
   使用 truncated BPTT，因为截断会改变长期 credit assignment。

## 当前 BrainCell 证据与缺口

### 已经证明的梯度路径

仓库历史快照 `55871ea` 中的
`examples/experimental/parameter_learning/heterogeneous_nine_parameter_training.py`
构造一个 soma 和两个 dendrite 的模型，使用 144 条多位置 DC、paired-pulse 和
sine protocol，从八组初值拟合三个 compartment 各自的 leak、Na 和 K `g_max`。
该实验使用：

- `brainstate.nn.Param(..., t=SigmoidT(lower, upper), fit=True)` 保存九个有界标量；
- `fit=False` 构造 target cell，使 target conductance 不注册为 trainable
  `ParamState`；
- `brainstate.transform.grad` 对整段仿真求梯度；
- `braintools.optim.Adam` 更新显式选择的参数；
- BrainState transform 编译时间 rollout、batch 和重复 optimizer update；
- 每次 loss 计算前 reset 动态状态。

其具体实验约束记录在同一历史快照的
`docs/specs/2026-08-17-heterogeneous-nine-parameter-training.md`。
这个实验证明梯度能够经过 cable solve、HH gating state、probe 和时间循环，到达
example-local channel parameters。它没有证明 BrainCell 已经拥有通用的参数发现、
group/shared parameter、density-function parameter、checkpointed training 或在线学习
接口。

仓库还包含较早的
[single-compartment HH fitting example](../../../../examples/single_compartment/SC01_fitting_a_hh_neuron.py)：
它使用 `braintools.optim.NevergradOptimizer` 在 `gL`、`gNa`、`gK` 和 `C` 的物理
范围内做 gradient-free search，并以完整 voltage trace 的 squared error 为目标。
该例可以作为优化预算和最终 trace 的 baseline，但没有 parameter transform 的
反向传播，也没有处理 spike timing shift 的专用 loss。

### `fit=False` 的语义

`fit` 描述的是“该值是否作为优化器可发现和可更新的参数状态”，不是“该机制
是否参与仿真”。例如 target cell 的 leak channel 仍然在每一个 time step 计算
电流，但它的 `g_max` 保持固定：

```python
target_g = brainstate.nn.Param(
    0.6 * u.mS / u.cm**2,
    t=brainstate.nn.SigmoidT(0.01 * u.mS / u.cm**2,
                            1.0 * u.mS / u.cm**2),
    fit=False,
)

fitted_g = brainstate.nn.Param(
    0.15 * u.mS / u.cm**2,
    t=brainstate.nn.SigmoidT(0.01 * u.mS / u.cm**2,
                            1.0 * u.mS / u.cm**2),
    fit=True,
)
```

两者都能通过 `.value()` 取得物理量参与电流计算；只有 `fitted_g` 应进入
optimizer parameter tree。未来 Parameter System 应把 trainable/frozen 作为参数
元数据，而不是依赖两种不同 mechanism registry entry。当前实验为绕开 spatial
lowering 的限制，仍分别注册了 `ExplorationTrainable*` 和
`ExplorationFrozen*` 两组 Leak、Na、K channel adapter。

### 当前手动参数适配的实际边界

当前 example-local core 不是在 lowering 完成后临时替换 runtime node。它为 Leak、
HH Na 和 HH K 分别继承原始 channel 类，并以新名字注册：

```text
ExplorationTrainableLeak -> IL
ExplorationTrainableNa   -> Na_HH1952
ExplorationTrainableK    -> K_HH1952
```

`mech.Channel("ExplorationTrainableLeak", ...)` 只是保存 class name 和声明参数；
`Cell.init_state()` lowering 时通过 mechanism registry 找到上述子类，再把按 CV 展开
后的参数传给子类构造函数。因此当前实现已经覆盖首次 lowering 边界：runtime node
从创建开始就持有 `brainstate.nn.Param`，而不是先创建普通 channel 再对实例做
`setattr`。

以跨两个 active CV 共享的 leak conductance 为例，lowering 传入：

```text
dense g_max = [0.1, 0.1, 0.0] * unit
```

自定义构造函数将其压缩为一个共享物理标量和静态空间 mask：

```text
node.g_max       = Param(0.1 * unit)
node._g_max_mask = [True, True, False]
```

`current()` 使用 `node.g_max.value() * node._g_max_mask` 恢复完整 CV 参数。训练
rollout 之间只调用 `reset_state()`，它重置电压、门控状态、spike 和时间，不重建
runtime channel，也不替换 `Param`。优化器直接更新 `node.g_max.val` 对应的
`ParamState`，所以当前实验的求导和更新链路是完整的。

完整的 `Cell.reset()` 会主动销毁 runtime，并不属于当前训练 rollout 的生命周期。
销毁后重新 `init_state()` 会再次创建正确类型的 trainable channel，但旧 runtime
对象、训练值和 optimizer 引用也随之失效；这属于模型重建/checkpoint 恢复问题，
不应描述为当前 core 的 lowering 缺陷。

真正尚未统一的是普通 runtime 参数更新接口。当前 `Cell.set_state()` 先更新
`state_buffers`，随后通过近似如下的赋值同步 runtime node：

```python
node.g_max = new_value
```

Python 属性赋值会让 `node.g_max` 改为引用新的 `Quantity`，而不是修改原有
`Param`。优化器即使仍持有旧 `ParamState`，channel 也不再读取它。对于已包装字段，
正确的物理值更新语义是：

```python
node.g_max.value()          # 读取带单位的物理参数
node.g_max.set_value(theta) # 保留 Param/ParamState 身份并设置物理参数
node.g_max = theta          # 替换整个容器，不适用于 trainable 字段
```

这里还必须区分两层映射：

```text
optimizer variable z
        | bounded/log/softplus transform
        v
physical scalar theta with units
        | sharing index / mask expansion
        v
dense runtime parameter per CV
```

`Param.value()` 和 `Param.set_value()` 负责 `z` 与 `theta` 的约束变换；parameter
binding 负责共享物理标量与 dense CV 参数之间的空间映射。如果 `set_state()` 接收
的是 `[0.2, 0.2, 0.0] * unit`，而 runtime field 是共享标量 `Param`，binding 必须先
根据 mask 验证 active values 一致并缩减为 `0.2 * unit`，再调用 `set_value()`。

未来公共参数系统需要提供容器保持的赋值协议和显式 spatial binding：普通字段仍可
直接 `setattr`，`Param`、`State` 或后续参数容器则更新其内部值；shared、per-region、
per-CV 和 generated-field 参数各自由 binding 定义展开与缩减规则。公共逻辑不能
硬编码 `g_max`。当前 core 已经手工证明这条路径可行，但每种 channel 和字段仍要
编写子类，尚未实现设计中的通用 `View.trainable()` API。

### 当前 spike 梯度

BrainCell 的 [`get_spike`](../../../../braincell/_base.py) 用前后两个膜电位相对
`V_th` 的 surrogate Heaviside 乘积检测向上越阈。multi-compartment `Cell` 默认
`V_th = 0 mV`，默认 `spk_fun` 是 `braintools.surrogate.ReluGrad()`。
`ReluGrad` 的 forward 是 hard step，但 backward 是有限支撑的三角形：超出
surrogate width 后梯度为零 [7]。因此：

- 已经靠近阈值时，event loss 可以给出 surrogate gradient；
- 模型离阈值很远、完全静默时，event loss 可能没有足够梯度把它推入放电区；
- spike 出现或消失的临界参数附近仍可能高度不光滑；
- 连续电压损失可以提供比单独 event indicator 更稠密的训练信号。

这决定了第一版 spike-aware fitting 应采用连续电压、平滑事件和特征的组合，
而不是只对 `Cell.spike` 做 MSE。surrogate-gradient 方法是处理离散 spike 的实用
近似，但其训练稳定性依赖 surrogate 的形状和支撑 [7]。

## 总体训练数据流

建议将完整拟合过程固定为以下数据流：

```text
physical bounds / priors / sharing
                |
                v
       unconstrained parameters z
                |
         physical transform
                v
  theta with units + spatial expansion
                |
      reset state -> simulate protocols
                |
       voltage / state / event traces
                |
 preprocessing fixed by target metadata
                |
 normalized component losses + regularization
                |
        gradient -> optimizer update
```

关键不变量：

- transform 后再附加或保持 `brainunit` 单位；mechanism 不接收无单位物理参数；
- parameter 与 dynamic state 分离，reset 只能重置 state，不能覆盖 optimizer
  parameter；
- target preprocessing、mask 和归一化常数在训练开始前确定，并停止梯度；
- 多个 loss component 先无量纲化，再加权求和；
- simulation 内重复 time step 不使用裸 Python `for`/`while`，使用
  `brainstate.transform.for_loop`、`scan` 或 checkpointed variants；
- 所有随机初始化或 protocol sampling 使用 `brainstate.random`。

## Parameter System

### 参数描述应包含什么

一个参数不能只有数值。未来参数描述至少需要以下信息：

| 字段 | 作用 | 例子 |
| --- | --- | --- |
| `name` | 稳定标识和日志名称 | `dend.Na.g_max` |
| `value` | 当前物理量 | `0.12 * u.S / u.cm**2` |
| `trainable` | 是否进入 optimizer tree | `True` |
| `transform` | 从 `z` 到物理域 | bounded sigmoid |
| `bounds` / `prior` | 生理约束 | `[0.0, 0.5] * u.S/u.cm**2` |
| `selector` / `mask` | 作用位置 | apical dendrite CVs |
| `sharing` | 一个值映射到哪些实例 | per-region / per-cell |
| `group` | optimizer 与 regularizer 分组 | `conductance`, `kinetics` |
| `generator` | 从少量系数生成空间场 | `g(d; alpha)` |

参数层级至少应支持：

- **per-compartment**：每个 CV 一个值，表达力最高但最易过拟合；
- **per-region**：soma、axon、basal、apical 各共享一个值；
- **per-cell**：整细胞共享，例如 temperature correction；
- **population/shared**：同类型 cell 共享 population parameter，同时保留
  cell-level random effect；
- **hierarchical**：`theta_cell = theta_population + delta_cell`；
- **generated field**：少量系数通过 morphology context 生成每个 CV 的密度。

选择 parameter 的 API 应与 spatial selector 组合，而 parameter sharing 需要显式，
不能通过“数组里恰好出现相同数值”推断。mask 只决定 parameter 展开到哪些 runtime
位置；inactive 位置应得到零密度，但不能产生多余 optimizer variables。

### 推荐的参数映射

| 物理量 | 推荐 transform | 原因 |
| --- | --- | --- |
| `g_max`, `Cm`, `Ra` 有可信上下界 | bounded sigmoid | 始终在生理范围内，尺度统一 |
| 只有正值约束的 rate / time constant | log 或 softplus | 不产生负值 |
| reversal potential、voltage shift | bounded sigmoid 或 affine | 通常有可信有限区间 |
| time-constant multiplier | log 或窄范围 sigmoid | 乘法尺度更自然 |
| density-function amplitude | softplus | 确保最终密度非负 |
| density length scale | log/softplus | 距离尺度必须为正 |

有界映射为：

```text
theta(z) = lower + (upper - lower) * sigmoid(z)
```

它便于约束和跨参数规范化，但在上下界附近会饱和。若最优值长期贴边，应检查
bound 是否不合理，而不是无限提高 learning rate。对跨越多个数量级的 conductance，
log-space 初始化通常比 linear-space 初始化更合理。

### 密度函数参数化

学习每个 compartment 的 `g_max` 会快速增加自由度，而且容易拟合噪声。推荐优先
学习低维、可解释的 density function，例如按 soma path distance `d`：

```text
g(d; alpha) = softplus(alpha_0)
              + softplus(alpha_1) * sigmoid((d - alpha_2) / softplus(alpha_3))
```

其中 amplitude 和 width 保持为正，transition distance 可设置有界范围。另一种
平滑表示是正基函数展开：

```text
g(d; alpha) = softplus(sum_k alpha_k * phi_k(d))
```

`phi_k` 可以是固定 B-spline 或 RBF。需要同时保存 coefficient、basis 定义和
morphology distance 语义；不能只保存展开后的 CV array，否则模型换 morphology
后无法复现。空间正则可以是：

```text
L_smooth = mean_edges(((g_i - g_j) / distance_scale)^2)
L_prior  = mean_k(((alpha_k - prior_k) / prior_scale_k)^2)
```

密度函数与当前的
[spatial callable parameter proposal](../../filter-spatial-callable-parameters.md)
有关，但训练版本还需要可微的 morphology context 和 coefficient ownership。

## Trainable Mechanism

channel、synapse 和 plasticity 要进入同一计算图，核心不是给每个类增加 `fit=True`，
而是统一 parameter/state 分离：

| 内容 | 生命周期 | 是否优化 | 例子 |
| --- | --- | --- | --- |
| parameter | 跨 rollout 保留 | 可 trainable/frozen | `g_max`, `E`, `tau_scale` |
| dynamic state | 每次 rollout 初始化或携带 | 通常不由 optimizer 更新 | gate `m/h/n`, `V`, ion concentration |
| online trace | 跨 time step 携带 | 由 learning rule 更新 | eligibility trace, STDP trace |
| static metadata | build/lowering 阶段 | 否 | mask, units, sharing index |

统一 mechanism contract 至少需要：

```text
declare parameters -> lower shared values to runtime layout
initialize dynamic state -> step(state, parameters, inputs)
read observables -> reset/detach selected state
```

channel current、gating update 和 synapse dynamics 中不得把 trainable parameter 转成
NumPy/Python scalar，否则会切断 tracing。需要静态构建的数据可以用 NumPy，但进入
仿真热路径的 parameter expansion 和数学运算必须保持 JAX/brainunit compatibility。

## Memory & Differentiation

一段 `T` 个 time step 的 reverse-mode BPTT 默认要保留足够的中间状态，内存近似随
`T * state_size` 增长。优先级建议如下：

1. **plain `for_loop` / `scan`**：短轨迹的默认方案，最少重计算；
2. **checkpointed loop / scan**：长 rollout 的首选，通过 backward 时重算换内存；
3. **multilevel checkpointing**：将长轨迹分层 rematerialize，Jaxley 也采用此策略
   [1, 2]；
4. **truncated BPTT**：仅在梯度爆炸/消失或内存仍不可接受时使用，必须记录 truncation
   window；Jaxley 的长 retina example 曾每 50 ms 截断 [1]；
5. **custom gradient / adjoint**：只有 profiler 证明 solver 是瓶颈、且数学推导与
   数值验证充分后再引入。

checkpoint 不改变目标函数，只增加重算；truncation 会阻断跨窗口梯度。因此训练
报告必须同时记录 `dt`、rollout length、checkpoint base/levels 和 truncation window。
对于跨秒行为，先做短窗口 curriculum，再扩大 horizon，通常比直接在完整时程上
训练稳定。Jaxley 的工作也指出毫秒级机制与秒级行为之间的 credit assignment 是
主要挑战 [1]。

## 无 Spike 数据的训练路线

### 为什么不只用 raw MSE

MSE 对少量 outlier 很敏感，而且长稳态区会在 sample count 上压过短暂的 onset。
不同 protocol、probe 和时间窗口的幅度也不同。推荐的无量纲组合为：

```text
L_no_spike = w_v    * L_voltage
           + w_dv   * L_derivative
           + w_ms   * L_multiscale
           + w_feat * L_features
           + w_reg  * L_regularization
```

候选分量：

- `L_voltage`：以 mV 表示误差后的 Huber 或 MAE；
- `L_derivative`：对平滑后的 `dV/dt` 使用 Huber，强调 onset/offset dynamics；
- `L_multiscale`：原分辨率和多个低通/降采样尺度的 trace loss；
- `L_features`：resting voltage、steady-state deflection、input resistance、膜时间
  常数、sag ratio、rebound amplitude；
- `L_regularization`：parameter prior、空间平滑和 hierarchical shrinkage。

为了避免某一项仅因数值尺度大而支配训练，先以初始模型或固定数据尺度归一化：

```text
L_hat_k = L_k / stop_gradient(max(L_k_at_initialization, epsilon))
L_total = sum_k w_k * L_hat_k
```

如果已有实验噪声估计，更优选择是用每项观测标准差归一化。所有归一化常数必须
固定，不能随当前 batch 波动到改变目标含义。

### 推荐 curriculum

1. 先拟合 resting/steady-state 特征，确认 leak、`Cm`、`Ra` 进入合理区域；
2. 加入 onset/offset 和 `dV/dt`，拟合时间常数；
3. 加入多尺度完整 trace；
4. 从短 protocol 扩大到完整时程；
5. 最后降低 learning rate，并在 held-out current sweep 上选模型。

初始探索超参数而非默认值：Adam learning rate 在无约束空间 sweep
`{1e-3, 3e-3, 1e-2, 3e-2}`，global gradient norm clip 从 `1.0` 开始，Huber
`delta` 从 `1--3 mV` 开始，归一化后 regularization weight 从
`1e-3--1e-2` 开始。快速实验至少 16 个 multi-start；正式 identifiability 分析
建议 32--64 个或更多。

### 成功标准

不能只报告 training loss。至少报告：

- train 和 held-out sweep 的 voltage MAE/Huber；
- resting、steady state、tau、sag 等 feature error；
- 参数是否贴 bound、梯度 norm、不同 seed 的终点分布；
- target 为 synthetic 时的 parameter recovery；
- 多组参数产生近似轨迹时的等价解集合，而不是只选一个“真实值”。

## 含 Spike 数据的训练路线

### raw pointwise loss 的问题

这不应归类为普通的“异常值序列”问题。spike 是样本少但生理信息密度高的
**稀疏关键事件**（sparse salient event），背景和事件在时间轴上存在严重的
样本不平衡。把 spike 当成 outlier 并用 Huber、clipping 或低权重永久压制，会正好
丢掉 `gNa`、`gK` 和 channel kinetics 最重要的监督信号。

全轨迹 pointwise MSE 有两个独立的失败机制：

1. **时间稀释**：例如 `100 ms` 轨迹中只有 `5 ms` spike 窗口，全平预测可以用
   剩余 `95 ms` 的低误差稀释 spike 误差。延长静息段还会在生理目标不变的情况下
   改变 loss 的相对权重。
2. **相位敏感**：假设 target spike 在 `t`，prediction 只晚 `delta t`。两条 spike
   shape 可能几乎相同，逐点相减却会产生两个大误差峰。Neurofitter 的实验甚至
   展示了相移的两条放电轨迹可能比“放电对静息”获得更高的 least-squares
   error [9]。

参数稍微移动还会让 spike 产生或消失，因此 loss landscape 在 spike-count
boundary 附近出现狭窄谷、台阶和梯度骤变。这三个问题需要分别处理，单独把
MSE 换成 Huber 不会解决。

Jaxley 在拟合实验电生理 spike trace 时使用 sliding-window maximum reduction、
归一化和 soft-DTW；其论文设置为 50 time-step window、30 time-step stride，原始
`dt = 0.025 ms`，并将观测与仿真按相同因子缩放到 unit interval [1]。这些具体数值
是一个经过验证的参考配置，但不能直接当作 BrainCell 对所有数据的默认值。

### 推荐组合损失

```text
L_spike = w_bg     * L_background
        + w_win    * L_spike_window
        + w_event  * L_filtered_events
        + w_count  * L_soft_count
        + w_shape  * L_AP_shape
        + w_align  * L_local_alignment
        + w_reg    * L_regularization
```

各项定义：

- `L_background`：由 target spike time 预先生成固定窗口，窗口外计算 Huber；
- `L_spike_window`：在每个 target spike 的局部窗口内计算正权重 waveform loss，
  首轮可测试 `[-1, +4] ms`；
- `L_filtered_events`：将 target 和 predicted event train 与指数或 Gaussian kernel
  卷积，再计算距离。同时使用宽和窄两个时间尺度，例如 `tau=5 ms`
  与 `tau=2 ms`，先容忍 timing jitter，再约束精确时间。该思路对应 van Rossum
  spike distance [4]，SuperSpike 也使用 surrogate gradient 优化这类目标 [12]；
- `L_soft_count`：对 smooth threshold crossing 求和后比较，在 count 错误时提供连续近似梯度；
- `L_AP_shape`：对齐后的 amplitude、half-width、max `dV/dt`、repolarization 和 AHP；
- `L_local_alignment`：只对 spike 局部窗口或 sliding smooth-max/envelope 使用
  soft-DTW [3]；
- `L_regularization`：同无 spike 路线。

`L_background` 和 `L_spike_window` 必须按各自的有效样本数分别取平均，不能一起
除以完整轨迹的 time-step 数。之后再用显式的 `w_bg` 和 `w_win` 决定两者
相对重要性。这使 loss 不会因为静息段变长而改变生理含义。target-fixed
window 仅负责定义区域；全轨迹的 event 和 count loss 仍必须捕获 target
window 外多出的 predicted spike。

soft-DTW 可微并允许时间轴上的局部对齐 [3]，但计算和存储成本随两条序列长度
近似二次增长。对当前 `100 ms` 全轨迹直接做 full soft-DTW 不仅昂贵，还可能
用过度 warping 隐藏错误放电模式。因此它只应用于局部窗口、降采样 envelope，
或限制 Sakoe-Chiba band，并显式惩罚时间偏移。Victor--Purpura edit distance
也能表达 spike 的删除、插入和时移 [11]，但其 hard minimum/edit path 不适合直接
作为当前 gradient-based 训练的主目标，更适合做评估指标或构建 smooth approximation。

Neurofitter 的 phase-plane trajectory density 在 `(V, dV/dt)` 空间比较轨迹密度，
对小相移不敏感，同时保留 spike shape 和 firing-rate 信息 [9]。它值得作为
BrainCell 后续消融的另一个 shape objective，但 histogram/binning 的可微实现和
分辨率需要单独验证。

### 当前 BrainCell 实验的对应关系

`heterogeneous_nine_parameter_composite_ablation.py` 当前有七个可微分量：

| 现有分量 | 实际定义 | 与推荐路线的关系 |
| --- | --- | --- |
| `voltage` | 全 compartment masked Huber | 近似 `L_background`，target spike `[-1, +3] ms` 权重为 `0.1` |
| `derivative` | masked `dV` Huber | `L_AP_shape` 的一部分，但 spike 窗口也被降权 |
| `multiscale` | 20 time-step block mean 后的 Huber | 低频 trace/envelope 约束 |
| `event` | soma smooth crossing 后 `tau=2 ms` 指数滤波 MSE | 单时间尺度 `L_filtered_events` |
| `count` | soma smooth crossing 求和差的平方 | `L_soft_count` |
| `peak` | `20--100 ms` soma smooth maximum 差的平方 | 粗粒度 spike-birth/peak 约束 |
| `mse` | 全 protocol、全 time-step、全 compartment MSE | 存在时间稀释和相位敏感 |

因此当前 Composite 已经有 van-Rossum-like event 和 soft count，但还没有：

- 与 background 独立归一化、且正权重监督的 spike-window waveform loss；
- 宽窄两个 event filter 时间尺度；
- target 应放电但 prediction 静息时的显式 threshold-margin/spike-birth loss。

这些项是后续实验候选，不应在文档中写成已实现的 API 能力。

### clipping、masking 和 envelope 的选择

| 方法 | 优点 | 主要风险 | 推荐用途 |
| --- | --- | --- | --- |
| hard voltage clipping | 简单，防止 peak 主导 | clipped 区域梯度为零，丢失 amplitude/width | 仅早期 subthreshold stage |
| target-fixed mask | 稳定定义 background/spike 区域 | 只降权而没有独立 spike loss 会失去 active-channel 约束 | 作为 `L_background` 和 `L_spike_window` 的共享预处理 |
| prediction-derived mask | 看似适应当前 spike | 目标随预测改变，可规避错误 spike | 不使用 |
| soft clipping / smooth-max envelope | 保持较平滑梯度 | 多一个温度/窗口超参数 | alignment preprocessing |
| raw AP shape loss | 约束 `gNa/gK` | 对 timing shift 极敏感 | 对齐后或局部 feature loss |

因此“特别尖的就不学”只能是早期 curriculum 的策略，不能是最终目标。当只学习
leak、`Cm` 或 subthreshold conductance 时，可以长期降低 AP peak 权重；当学习
`gNa`、`gK`、channel kinetics 时，必须在后期恢复 amplitude、width、upstroke 和
repolarization 的监督。

### 从静默到放电

只用 filtered event loss 时，静默模型可能拿不到有效 surrogate gradient。增加
连续的 threshold-margin objective：

```text
V_peak = smooth_max(V_in_stimulus_window)
L_margin = softplus((V_threshold + margin - V_peak) / temperature)
```

当 target 应该放电而 prediction 静默时使用该项；当 target 不应放电时反转 margin
方向。它提供把电压推近阈值的连续信号。一旦 spike count 正确，逐渐降低
`L_margin`，提高 alignment 和 AP shape 权重。

margin 应在每个 target spike 的局部窗口内计算，而不是只对整个 stimulus 取一次
maximum，否则多 spike protocol 中的一个正确 peak 会掩盖其他缺失 spike。对于 target
静息的 protocol，不启用 spike-window 和 birth 项，但仍在全时程使用 event、count 和
upper-threshold margin 惩罚额外放电。

### 推荐 curriculum

1. 对 background 和 spike window 分别归一化，先用 resting、steady-state、smooth peak
   和较宽温度的 threshold margin 把模型带入正确兴奋性区域；
2. 加入 soft count 和宽时间尺度 filtered event loss，先修正缺失/额外 spike；
3. 加入窄时间尺度 filtered event，逐渐降低 surrogate temperature，缩小 timing
   tolerance；
4. spike 数稳定后加入局部 waveform、amplitude、half-width、`dV/dt` 和 AHP；
5. 若 timing jitter 仍主导梯度，再加局部 envelope soft-DTW，而不是默认对全轨迹
   执行；
6. 最后在完整 trace 与全部 protocol 上微调，并按 held-out protocol 选模型。

不建议一次性打开所有分量后手工猜权重。每项应使用固定 data-derived 或
initial-loss normalizer，并记录其对九维参数的 gradient norm 和 gradient cosine。若某些
分量长期压制其他分量，可在实验层对比 GradNorm 类自适应平衡 [13]；这只是
候选实验，不应在没有 ablation 前变成默认训练策略。

### Checkpoint 选择与评估

对 spike protocol，最低 composite validation loss 不一定是生理上最好的 checkpoint。
建议在 held-out protocol 上用如下字典序选择：

1. 最大化 hard spike-count exact protocol fraction；
2. count 相同时，最小化 unmatched spike 和 ordered timing/event distance；
3. 前两项相同时，最小化 spike-window waveform 和 background voltage error。

hard crossing count 和 Victor--Purpura distance 可用于 checkpoint 选择与报告，无需强行对它们
求梯度。对 synthetic target 同时报告 parameter recovery；对实验数据则优先报告 held-out
生理 feature，不把单一参数距离当作真值。

Jaxley 的复杂 L5PC 拟合建议先 random search，再从较优候选启动梯度下降，并监控
gradient norm [2]。其论文还使用有界 inverse-sigmoid 参数规范化和 Polyak-style
update：

```text
step = gamma * grad(L) / ||grad(L)||^beta,  beta in [0.8, 0.99]
```

部分任务再乘当前 loss，使训练后期自动缩小步长 [1]。BrainCell 第一轮实验应同时
保留 Adam baseline；只有对同一 seed/预算的对比证明 Polyak variant 更稳定时，才
把它提升为推荐 optimizer。

### EventProp 的位置

EventProp 使用 continuous-time event、root finding、adjoint 和 spike 时刻的 jump
condition，为 hard-threshold LIF network 计算精确梯度 [8]。它不是 BrainCell
spike fitting 第一版的直接方案：BrainCell 当前是固定 `dt` 的 conductance-based
multi-compartment simulation，并使用 surrogate crossing；引入 EventProp 需要明确
事件根求解、mechanism jump、cable state adjoint 和 event queue 语义。短期应先验证
step-based composite loss，中长期再评估 hybrid event adjoint。

如果观测只包含 spike time 而没有可信的连续电压，可以将放电建模为 conditional
intensity 并优化 point-process likelihood [14]。当前 BrainCell synthetic dataset 提供完整
soma/dendrite voltage，因此它不是首选目标；否则会主动丢掉 AP shape 和
subthreshold dynamics 中的参数信息。

## Loss Landscape 与可辨识性

loss landscape 不只是美化训练结果，而是判断“梯度为什么失败”和“参数能否由数据
确定”的必要诊断。

### 必须输出的图

1. **一维 physical profile**：逐个参数扫描，其余参数固定在 target、initial 或
   fitted point；
2. **二维关键参数对**：优先 `gNa-gK`、`gLeak-Cm`、density amplitude-length scale；
3. **分量 landscape**：分别画 subthreshold、alignment、event、AP shape 和 total；
4. **spike-count heatmap**：覆盖在连续 loss contour 上，显示 spike 出现/消失边界；
5. **trajectory overlay**：标出 target、initial、每个 optimizer step 和 final；
6. **multi-start endpoints**：展示不同 seed 是否收敛到同一 basin 或补偿解流形。

扫描应在规范化的无约束 `z` 空间均匀取点，但坐标标签转换回带单位的 physical
parameter。否则线性扫描跨数量级 conductance 会误导曲率判断。每个 grid point
必须 reset state，并对相同 protocol 集求 loss。二维图还应显示 non-finite
simulation、bound saturation 和 spike count，而不是把失败点默认为最大 loss 后
隐藏。

局部 Hessian eigenvalue 或 Gauss-Newton/Fisher approximation 可以描述 final
附近的 flat direction，但不能代替大范围扫描。多个 conductance 组合产生相似输出
是生物模型的常见现象 [6]；因此最终应报告可接受参数集合、相关性和 profile
likelihood，而不只是一个 point estimate。

## Worked Examples

以下例子是后续实现应满足的实验合同，不表示这些 API 已存在。

### Example A：无 spike 的单个 dendritic leak `g_max`

**模型。** 沿用现有三 compartment demo，target 为
`0.6 mS/cm^2`，initial 为 `0.15 mS/cm^2`，范围
`[0.01, 1.0] mS/cm^2`。对 soma 和 `dend_a` 同时记录多个 subthreshold current
sweep。

**训练。** 只选择 `dend_a.leak.g_max`，target cell 使用 `fit=False`。第一阶段
优化 resting、steady-state 和 tau；第二阶段增加原分辨率 Huber 与 4x/16x
low-pass trace loss。Adam learning rate 从 `{0.003, 0.01, 0.03}` sweep，clip norm
从 `1.0` 开始。

**验证。** 保留一个未训练的 current amplitude，报告 target/initial/fitted trace、
loss components、parameter trajectory 和一维 `g_max` landscape。synthetic 数据
除 held-out trace 外，还要求 parameter relative error。

### Example B：含 spike 的 soma `gNa/gK` 联合拟合

**模型。** 固定 morphology、leak 和 reversal potential，学习 soma/axon 的共享
`gNa`、`gK`。protocol 至少包含一个静默 sweep、一个单 spike sweep 和一个重复
放电 sweep。

**训练。** 对 conductance 使用 bounded sigmoid。阶段 1 用 subthreshold Huber、
smooth peak 和 threshold margin；阶段 2 加 filtered event、count 和 first latency；
阶段 3 加 envelope soft-DTW；阶段 4 加 amplitude、half-width、max `dV/dt`、AHP。

**验证。** 输出 `gNa-gK` total landscape 及 spike-count heatmap。额外检查错误但
补偿的 `gNa/gK` 是否在 train trace 上低 loss、却在 held-out amplitude 上失败。

### Example C：树突 `g_max` 密度函数系数

**模型。** 以 soma path distance 为输入，用三个或四个系数生成所有 apical CV 的
channel density，而不是每 CV 一个自由参数。多个 dendritic probe 提供局部响应。

**训练。** amplitude/width 使用 positive transform，transition distance 使用
morphology 范围内 bounded transform。loss 包括所有 probe 的多尺度 voltage、系数
prior 和相邻 CV 空间平滑。先学习低频/稳态分量，再加入局部 transient。

**验证。** 除 trace 外画 target/fitted density profile、coefficient landscape 和
不同 morphology discretization 下的 profile。关键测试是改变 CV 数后，物理位置上
的密度函数保持一致，而不是按 array index 复用。

### Example D：混合五个 current sweep

**数据。** 五列 recording 中可能同时有无 spike、单 spike 和多 spike trace。每个
sweep 保存 stimulus、`dt`、recording location、target spike windows 和噪声尺度。

**训练。** batch 内先按 sweep 类型计算适用分量：所有 sweep 都计算 voltage
baseline；静默 sweep 计算 subthreshold features；spiking sweep 计算 event、alignment
和 AP feature。每个 component 按固定尺度归一化，再跨 sweep 求均值，避免 spike
较多的 sweep 因 sample/event 数多而自动占更大权重。

**验证。** 每轮 cross-validation 留出一个 amplitude，比较 raw MSE baseline、
masked-only、soft-DTW-only 和 composite loss。最终选择依据是 held-out 综合指标，
不是训练集最低 total loss。

## Online Learning

offline parameter fitting 的 optimizer update 发生在完整 rollout 或 batch 之后；
online learning 则在 simulation 过程中持续改变 parameter。两者必须明确区分：

- **online gradient / streaming BPTT**：分 chunk 计算梯度并更新，但 update 后过去的
  trajectory 已来自旧参数；
- **exact forward sensitivity / RTRL**：rollout 内保持参数不变，正向 carry 完整
  $\partial x_t/\partial\theta$；每个 prefix gradient 都精确，最终与 full BPTT 相同，但
  memory 为 $O(N_xN_\theta)$，计算随 parameter direction 数增长；
- **eligibility trace**：在 forward 中累积局部敏感度，与稍后的 error/reward 结合；
- **STDP/local rule**：机制本身定义 pre/post trace 和权重更新，不一定等价于全局
  objective 的梯度；
- **reward-modulated plasticity**：eligibility 与延迟 reward 相乘，需要清晰的 episode
  boundary 和 trace reset 规则。

online parameter update 会改变积分方程本身，并且容易造成 optimizer state、model
state 和 plasticity trace 的所有权混乱。建议分两阶段：

1. 先实现离线 differentiable fitting，稳定 parameter/state/mechanism contract；
2. 再定义 `learning_state` 和显式 update boundary，分别验证 stop-gradient、跨 chunk
   carry、reset 和 checkpoint semantics。

第一版 online API 不应通过在 mechanism `current()` 内偷偷赋值 parameter 实现。
update 应是显式 transform step，便于 JIT、复现、暂停/恢复和日志记录。

单 Cell、固定外源 delay、不使用 spike feedback 时的精确边界和原型见
[`single-cell-online-forward-sensitivity.md`](./single-cell-online-forward-sensitivity.md)。

## 实验矩阵与记录要求

每个新训练例子至少记录：

| 类别 | 必要字段 |
| --- | --- |
| data | source、protocol split、`dt`、duration、units、noise preprocessing |
| model | morphology、CV policy、solver、mechanisms、initial state |
| parameters | names、units、bounds、transform、sharing、initial seed |
| differentiation | loop primitive、checkpoint levels/base、truncation window |
| optimizer | algorithm、learning rate、clip、epochs、multi-start budget |
| loss | component definitions、normalizers、weights、mask windows |
| diagnostics | train/validation metrics、gradient norm、bound saturation |
| artifacts | trace plot、loss components、parameter trajectory、landscape |

建议按如下顺序建立证据：

1. 在现有 leak demo 上加入 subthreshold composite loss 对照；
2. 构造已知 `gNa/gK` 的 synthetic spike recovery；
3. 在相同 seed、batch schedule 和 optimizer budget 下依次消融：
   `background-only -> separately-normalized spike window -> broad+narrow event ->`
   `spike-birth margin -> soft count -> AP shape/local alignment`；
4. 每次消融都加入 spike timing offset、静息目标、缺失/额外 spike 和噪声的受控
   扰动，分别报告 loss 分量、gradient norm/cosine、hard count 和 timing；
5. 再使用实验 recording，并用 feature-based 或 gradient-free result 做 baseline；
   BluePyOpt 是这类多目标生物物理参数优化的成熟参考实现 [5]，
   Druckmann 等人的框架则说明了为什么应将 spike rate、width 等 feature 按实验
   variability 归一化后作为多目标，而不是强求单条 trace 的逐点完美匹配 [10]；
6. 最后扩展到 density function 和在线学习。

## 失败模式与边界情况

- **silent-to-spike zero gradient**：event surrogate 支撑不足；使用 voltage margin 和
  curriculum。
- **extra/missing spike**：soft-DTW 可能强行对齐；显式加入 count 和 unmatched-event
  penalty。
- **burst**：只比较 first spike 不足；加入 ISI distribution、burst duration 和
  adaptation。
- **target spike 在窗口边界**：mask 需要 clip 到合法索引，不能 wrap-around。
- **不同 `dt`**：先用单位明确的时间轴重采样；不能直接按 sample index 比较。
- **recording dropout / NaN**：用 target-fixed validity mask，并按有效样本重新归一化。
- **parameter 贴界**：检查 bound、transform saturation 和模型缺失机制。
- **non-finite simulation**：记录物理参数和 protocol，optimizer step 不得污染最后一个
  finite checkpoint。
- **loss component 为零**：initial-loss normalization 必须有 `epsilon`，并考虑直接
  禁用无信息项。
- **多 probe 数量不平衡**：先 per-probe 归一化再聚合，避免 CV/probe 多的 region
  支配训练。
- **机制参数补偿**：用多 protocol、先验、landscape 和 held-out feature 判断，不以
  单一参数 recovery 作为真实实验数据的唯一标准。

## 待验证的设计问题

以下问题必须由实验决定，本文不预设答案：

- BrainCell 的通用 parameter tree 是由 Cell 集中拥有，还是由 runtime mechanism
  声明后统一收集；
- parameter sharing 与 selector 的稳定 public API；
- density function 在 discretization 前还是 runtime lowering 时求值；
- soft-DTW 是引入依赖、实现自定义 JAX kernel，还是只作为研究例子；
- `ReluGrad` width/alpha 是否需要针对 biophysical voltage crossing 暴露；
- composite loss 的默认归一化和权重是否足够跨 cell type 复用；
- checkpoint base 和 truncation window 如何根据 state size 自动建议；
- online learning 的 parameter、optimizer state 和 eligibility trace checkpoint 格式。

## References

### 本地实现依据

以下两个训练/数据脚本只存在于所列历史提交，不存在于当前 worktree；当前目录状态和仍活跃的
diagnostics helper 见 [Parameter Learning README](../../../../examples/experimental/parameter_learning/README.md)。

- `examples/experimental/parameter_learning/heterogeneous_nine_parameter_training.py`
  及其规格 `docs/specs/2026-08-17-heterogeneous-nine-parameter-training.md`，位于提交
  `55871ea`。
- `examples/experimental/parameter_learning/heterogeneous_protocol_dataset.py`
  及其规格 `docs/specs/2026-08-17-heterogeneous-protocol-dataset.md`，位于提交 `55871ea`。
- [Single-compartment HH gradient-free fitting example](../../../../examples/single_compartment/SC01_fitting_a_hh_neuron.py)。
- BrainCell [`get_spike` surrogate crossing](../../../../braincell/_base.py) 与
  [spatial callable parameter proposal](../../filter-spatial-callable-parameters.md)。

### 外部资料

1. Deistler, M. et al. *Jaxley: differentiable simulation enables large-scale
   training of detailed biophysical models of neural dynamics*. Nature Methods
   22 (2025). [doi:10.1038/s41592-025-02895-w](https://doi.org/10.1038/s41592-025-02895-w).
2. Jaxley documentation. *Training biophysical models* and
   *Fitting an L5PC with gradient descent*.
   [Training tutorial](https://jaxley.readthedocs.io/en/latest/tutorials/07_gradient_descent.html),
   [L5PC example](https://jaxley.readthedocs.io/en/latest/examples/00_l5pc_gradient_descent.html).
3. Cuturi, M. & Blondel, M. *Soft-DTW: a Differentiable Loss Function for
   Time-Series*. Proceedings of Machine Learning Research 70, 894--903 (2017).
   [PMLR](https://proceedings.mlr.press/v70/cuturi17a.html).
4. van Rossum, M. C. W. *A Novel Spike Distance*. Neural Computation 13,
   751--763 (2001).
   [doi:10.1162/089976601300014321](https://doi.org/10.1162/089976601300014321).
5. Van Geit, W. et al. *BluePyOpt: Leveraging Open Source Software and Cloud
   Infrastructure to Optimise Model Parameters in Neuroscience*. Frontiers in
   Neuroinformatics 10, 17 (2016).
   [doi:10.3389/fninf.2016.00017](https://doi.org/10.3389/fninf.2016.00017).
6. Marder, E. & Taylor, A. L. *Multiple models to capture the variability in
   biological neurons and networks*. Nature Neuroscience 14, 133--138 (2011).
   [doi:10.1038/nn.2735](https://doi.org/10.1038/nn.2735).
7. Neftci, E. O., Mostafa, H. & Zenke, F. *Surrogate Gradient Learning in
   Spiking Neural Networks: Bringing the Power of Gradient-Based Optimization
   to Spiking Neural Networks*. IEEE Signal Processing Magazine 36, 51--63
   (2019).
   [doi:10.1109/MSP.2019.2931595](https://doi.org/10.1109/MSP.2019.2931595).
8. Wunderlich, T. C. & Pehle, C. *Event-based backpropagation can compute exact
   gradients for spiking neural networks*. Scientific Reports 11, 12829
   (2021).
   [doi:10.1038/s41598-021-91786-z](https://doi.org/10.1038/s41598-021-91786-z).
9. Van Geit, W., Achard, P. & De Schutter, E. *Neurofitter: a parameter tuning
   package for a wide range of electrophysiological neuron models*. Frontiers
   in Neuroinformatics 1 (2007).
   [doi:10.3389/neuro.11.001.2007](https://doi.org/10.3389/neuro.11.001.2007).
10. Druckmann, S. et al. *A novel multiple objective optimization framework for
    constraining conductance-based neuron models by experimental data*.
    Frontiers in Neuroscience 1, 7--18 (2007).
    [doi:10.3389/neuro.01.1.1.001.2007](https://doi.org/10.3389/neuro.01.1.1.001.2007).
11. Victor, J. D. & Purpura, K. P. *Nature and precision of temporal coding in
    visual cortex: a metric-space analysis*. Journal of Neurophysiology 76,
    1310--1326 (1996).
    [doi:10.1152/jn.1996.76.2.1310](https://doi.org/10.1152/jn.1996.76.2.1310).
12. Zenke, F. & Ganguli, S. *SuperSpike: supervised learning in multilayer
    spiking neural networks*. Neural Computation 30, 1514--1541 (2018).
    [doi:10.1162/neco_a_01086](https://doi.org/10.1162/neco_a_01086).
13. Chen, Z. et al. *GradNorm: gradient normalization for adaptive loss
    balancing in deep multitask networks*. Proceedings of Machine Learning
    Research 80, 794--803 (2018).
    [PMLR](https://proceedings.mlr.press/v80/chen18a.html).
14. Paninski, L. *Maximum likelihood estimation of cascade point-process neural
    encoding models*. Network: Computation in Neural Systems 15, 243--262
    (2004).
    [doi:10.1088/0954-898X/15/4/002](https://doi.org/10.1088/0954-898X/15/4/002).
