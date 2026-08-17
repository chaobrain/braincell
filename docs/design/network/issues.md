# Network Builder Issue Register

本文只管理尚未冻结或需要保留决策记录的设计点。正式行为必须写回
[api.md](./api.md) 或 [architecture.md](./architecture.md)；issue 本身不能成为第二份规范。

状态定义：

- `OPEN`：实现前仍需选择方案；
- `LOCKED`：行为已经确定，保留记录并等待实现/验收；
- `DEFERRED`：明确不属于 v1，不阻塞当前主线。

## Issue overview

当前共 14 项：`4 OPEN`、`8 LOCKED`、`2 DEFERRED`。

| Issue | Topic | Status |
| --- | --- | --- |
| [I-01](#i-01-naming-and-api-vocabulary) | public naming 与 API vocabulary | `OPEN` (`P0`) |
| [I-02](#i-02-network-lifecycle-and-initialization-boundary) | Network lifecycle 与初始化边界 | `LOCKED` |
| [I-03](#i-03-weight-ownership-and-defaults) | event weight ownership、fallback 与单位 | `LOCKED` |
| [I-04](#i-04-pairrule-scalability) | PairRule 保存协议与规模边界 | `LOCKED` |
| [I-05](#i-05-weighted-continuous-region-sampling) | weighted continuous Region sampling | `LOCKED` |
| [I-06](#i-06-batch-placement-lowering) | contact batch placement lowering | `OPEN` |
| [I-07](#i-07-delay-runtime-semantics) | physical delay 与 runtime timing | `LOCKED` |
| [I-08](#i-08-trainable-versus-non-trainable-fields) | trainable parameters 的 v1 边界 | `LOCKED` |
| [I-09](#i-09-generated-instance-identity) | generated mechanism instance identity | `LOCKED` |
| [I-10](#i-10-materialization-time-rng-contract) | RNG 语义已锁定，sampling backend 待选 | `OPEN` |
| [I-11](#i-11-heterogeneous-delay-buffer-layout) | heterogeneous delay buffer memory | `OPEN` |
| [I-12](#i-12-model-mutation-ownership-and-cache-invalidation) | mutation ownership 与 cache invalidation | `LOCKED` |
| [I-13](#i-13-kinetic-and-continuous-synapse-input-protocols) | kinetic/continuous synapse input protocols | `DEFERRED` |
| [I-14](#i-14-runtime-state-mutation-api) | initialized runtime state 的原地修改接口 | `DEFERRED` |

## I-01 Naming and API vocabulary

**Status:** `OPEN` (`P0`)

### Problem description

Network 同时存在用户持有的可编辑对象、不可变声明、物化结果和 runtime state。若这些层级
共用 `Population`、`Projection`、`Table` 或 `Resolved*` 等含混名称，后续 API、context 和
错误信息会形成两套词汇。命名必须先于其余 public schema 冻结。

### Working vocabulary

| Layer | Working name | Meaning |
| --- | --- | --- |
| mutable handle | `Population`, `Projection` | Network-owned registration and lifecycle handle |
| immutable declaration | `PopulationSpec`, `ProjectionSpec`, `SynapseSpec` | validated/default-filled declaration; callables are not executed |
| population materialization | `PopulationInstances` | cell IDs, position, rotation and properties |
| pair/contact materialization | `PairTable`, `ContactTable` | normalized sparse pair rows and expanded contact rows |
| runtime | `*Block`, `*State` | lowered static arrays and dynamic state |

`Population` 暴露 `.spec`、`.instances` 和 `.cell`；`Projection` 暴露 `.spec`、`.pairs`、
`.contacts` 和 `.parameters`。不提供 `pc[id]`、public `ResolvedProjection` 或额外 result handle。

仍需在完整 examples 中复核的候选词：

| Concept | Working name | Alternatives retained for review |
| --- | --- | --- |
| population rows | `PopulationInstances` | `PopulationData`, `PopulationTable` |
| cell-pair rows | `PairTable` | `CellPairTable`, `EdgeTable` |
| contact rows | `ContactTable` | `SynapticContactTable`, `ConnectionTable`, `SynapseTable` |
| endpoint location | `source_loc`, `target_loc` | full `source_location`, `target_location` |
| placement grouping | `sampling_unit` | `group_by` |
| typed rotation | `Rotation` | `Orientation` |

### Locked grammar

- `Spec` 只表示 canonical effective declaration，不保存某次 seed 生成的 arrays 或 runtime state。
- selector 声明使用 `PopulationSelector`；解析后的 endpoint subset 使用 `PopulationView`。
- raw custom rules 返回标准 arrays/LocsetMask，不引入 `PairBatch`、`PairRows`、`LocationBatch`
  或 `LocationRows`。
- `NetworkContext` 的 resolved views 不使用 `Resolved*` public class names。

### Decision deadline

所有新 public classes 开始实现前关闭。关闭前 working names 可以出现在规范中，但必须由本
issue 集中管理，其他模块不得自行创造同义词。

## I-02 Network lifecycle and initialization boundary

**Status:** `LOCKED`

### Problem description

Network 在初始化前既保存 population/projection declarations，也能按需生成 Cell、CV、Pair、
Contact、Location 和 parameter tables。若查看这些静态结果就冻结 Network，用户无法在检查
连接后继续修改；若初始化后仍允许改变 contact 数量、位置或 mechanism type，JAX runtime
的 shape、scatter indices、delay queues 和 compiled functions 会与声明不一致。

### Failing example

```python
proj = net.add_projection(...)
print(proj.contacts)          # 用户只是检查实际生成的 contacts
proj.set(target_loc=new_loc)  # 不应因为上一次查看而被禁止

net.init_state()
proj.set(pair_rule=new_rule)  # 会改变 runtime shape，必须禁止
```

公开生命周期需要同时回答：何时冻结结构、如何返回编辑态、reset 是否保留参数和结构，以及
连续 `run()` 如何处理尚未到期的 events。

### Locked solution

- 只有 `EDITABLE` 和 `INITIALIZED` 两个 public phases；不公开 `BUILT` phase、`build()` 或
  `resolve_*()`。
- static tables 通过 inspection properties 按需物化和缓存；查看结果不冻结 Network。
- `init_state()` 是唯一显式提交边界，完成所有 stale static materialization，冻结结构并
  统一初始化 Network-owned Cells、detectors 和 delivery state。
- `run()` 不自动初始化；未调用 `init_state()` 时明确报错。
- Network-owned Population Cell 不能单独 `init_state()`。
- `reset_state()` 保持 `INITIALIZED`，重置 dynamic state、time、detector history 和 event
  queues；保留结构、模型参数和 compiled caches。
- `deinit_state()` 销毁 runtime、event queues 和 compiled caches，保留 specs、static tables、
  managed placements 和模型参数，并返回 `EDITABLE`。
- 一个连续运行 episode 固定 `dt` 和 delay quantization mode，并保留 pending events；
  `reset_state()` 后可以重新选择二者。
- `init_state()` 原子提交；失败时撤销部分 runtime 并回到 `EDITABLE`。

### Resolved example

```python
print(net.projections["E_to_I"].contacts)
net.projections["E_to_I"].set(target_loc=new_loc)

net.init_state()
net.run(dt=0.025 * u.ms, duration=50.0 * u.ms)
net.run(dt=0.025 * u.ms, duration=50.0 * u.ms)  # continues state and events

net.reset_state()   # same structure/parameters, dynamic state returns to t=0
net.deinit_state()  # runtime removed; declarations are editable again
net.projections["E_to_I"].set(pair_rule=new_rule)
```

### Gate

Phase 7 lifecycle、atomic initialization、continued-run、reset 和 deinit tests 全部通过。

## I-03 Weight ownership and defaults

**Status:** `LOCKED`

### Problem description

`weight` 是 contact 在事件到达时投递给 postsynaptic mechanism 的 payload，不是所有
mechanisms 共享的电导参数。它的单位、shape 以及是否存在都由 target mechanism 决定，因而
不能由 Network 提供一个跨机制的 `default_weight`，也不能把它混入 `tau/e/gmax/...` 等
mechanism parameters。

NEURON 的内置 `ExpSyn` 和 `Exp2Syn` 都声明 `weight (uS)`：前者在事件到达时执行
`g <- g + weight`，后者执行 `A/B <- A/B + weight * factor`，并将孤立事件的峰值电导
归一化为 weight。其他 target 可以把 weight 解释为电流、无量纲强度或递质输入；纯触发
target 甚至不消费 payload。参考 NEURON 的
[ExpSyn](https://github.com/neuronsimulator/nrn/blob/master/src/nrnoc/expsyn.mod)、
[Exp2Syn](https://github.com/neuronsimulator/nrn/blob/master/src/nrnoc/exp2syn.mod) 和
[NetCon weight 说明](https://www.neuron.yale.edu/phpBB/viewtopic.php?t=1187)。

| Target semantics | Event action example | Contact weight contract |
| --- | --- | --- |
| single exponential conductance | `g <- g + weight` | scalar conductance, canonical `uS` |
| double exponential conductance | `A/B <- A/B + weight * factor` | scalar conductance, canonical `uS` |
| current-based event | `i <- i + weight` | scalar current, for example `nA` |
| kinetic/release model | open or scale an internal transmitter state | mechanism-defined scalar |
| trigger/reset model | `on <- 1` or reset state | no payload; `weight=None` |

事件输入不要求 target 必须执行离散加法。覆盖、状态切换、随机释放、饱和和随后连续演化的
kinetic ODE 都属于 mechanism 内部语义，Projection 只负责按时投递 target 声明的 payload。

### Failing examples

```python
# Exp2Syn consumes conductance, not current.
net.add_projection(..., synapse=exp2, weight=0.1 * u.nA)

# Physical payloads must carry units; no implicit uS is attached to bare values.
net.add_projection(..., synapse=exp2, weight=0.1)

# gmax is a mechanism parameter; weight is a contact value.
SynapseSpec(model="KineticAMPA", parameters={"weight": 0.1, "gmax": 0.5 * u.uS})
```

### Locked solution

- mechanism registry 为每个 event-capable synapse 声明 event input contract。v1 支持
  scalar weighted event 和 payload-free trigger event；没有 event port 的 mechanism 不能
  用作 event Projection target。
- `Projection.weight` 是 per-contact declaration。materialization 后的 effective values 存入
  `ContactTable.weight`，runtime 只消费该 canonical contact column。
- resolution precedence 固定为：显式 `Projection.weight`，否则直接使用
  `SynapseSpec.default_weight`，再没有则报错。fallback 不属于 target placement，只减少声明
  重复；物化后 owner 仍是 contact。
- `SynapseSpec.parameters/parameter_rule` 只生成 postsynaptic mechanism parameters；
  `default_weight` 是独立 fallback 字段，不进入 parameters mapping。
- physical weight 必须是显式 `Quantity`。量纲兼容的单位允许换算，例如 `100 nS` 可转换为
  `0.1 uS`；错误量纲、裸数、NaN、Inf、错误 shape 都拒绝。
- `ExpSyn/Exp2Syn` 接受任意有限 signed conductance，包括负值。Network 不把负值自动解释
  为抑制，也不负责训练期间保持符号；常规抑制性电导由正 weight 配合适当 reversal
  potential 表达。
- trigger-only target 要求 effective weight 为 `None`；提供任何值都报错。无 event port 的
  target 与 trigger-only target 必须保持可区分。
- v1 public weight 是每个 contact 一个 scalar，不支持任意向量或多字段 payload。AMPA/NMDA
  比例、release probability 和 plasticity state 属于 mechanism parameters 或内部
  per-contact state。
- initial/materialized weight 由本 issue 校验；optimizer、STDP 和其他 runtime mutation 的
  约束与 checkpoint ownership 由
  [I-08](#i-08-trainable-versus-non-trainable-fields) 管理。

### Resolved examples

```python
exp2 = SynapseSpec(
    model="Exp2Syn",
    parameters={"tau1": 0.5 * u.ms, "tau2": 5.0 * u.ms, "e": 0.0 * u.mV},
    default_weight=0.1 * u.uS,
)

net.add_projection(..., synapse=exp2)                       # uses 0.1 uS fallback
net.add_projection(..., synapse=exp2, weight=250 * u.nS)   # stores 0.25 uS
net.add_projection(..., synapse=exp2, weight=-0.05 * u.uS) # valid signed payload
```

### Gate

Phase 4 registry、`SynapseSpec` 和 ContactTable schema 实现前保持本 contract；unit/shape、
fallback precedence、signed value、trigger-only 和 unsupported-port tests 必须全部通过。

## I-04 PairRule scalability

**Status:** `LOCKED`

### Problem description

PairTable 最终只保存实际存在的 `P` 个 cell pairs，但 naive generator 可能先构造 resolved
source/target views 的完整 `S*T` Cartesian candidates。`S/T` 是 selector 之后的 endpoint
sizes，不一定等于完整 population sizes。

例如 `S=T=100,000`、`p=0.001` 的 independent probability rule 期望只产生 `P=10,000,000`
rows，最终 `(P,3)` `int32` array 约 120 MB。若先生成 `float64 (S,T)` random matrix 和 bool
mask，临时内存约为 80 GB + 10 GB。fixed-degree 的最终 `P` 更可直接计算，更没有理由创建
Cartesian candidates。

另一个不可消除的边界是最终结果本身：resolved subsets 上的 `all_to_all()` 必须产生
`P=S*T` 个真实 rows。若这 `P` 行本身无法驻留 host memory，改变生成算法不能解决问题，
只能引入 out-of-core/streaming architecture。

### Locked solution

- v1 public protocol 固定为 `PairRule(ctx) -> integer array (P,3)`，columns 为
  `source_id/target_id/nsyn`。不接受 generator、iterable、dense adjacency、CSR/CSC 或隐式
  `(p, seed)` object 作为 custom rule 返回值。
- materialized PairTable 的 source of truth 是显式 sparse COO-style rows。物理实现可以保存
  三个 column arrays；CSR/CSC、source pointers、sorted permutations 或 runtime sparse
  structures 只能是 derived caches。
- source/target selectors 先解析，所有 built-in rule 只在 resolved `S` 和 `T` 上工作。
  subset-to-subset `all_to_all()` 明确物化 `S*T` rows；该最终分配不可避免，但不得再创建
  无必要的二维 meshgrid copies。
- `fixed_indegree/fixed_outdegree` 直接为每个 endpoint 抽样，generation memory 随最终
  `P` 增长，不扫描或保存完整 Cartesian product。
- `probability(p)` 保持每个合法候选 pair 独立 Bernoulli 的精确定义。v1 baseline 使用逐
  source 或 bounded chunk generation，只保留一块候选工作区和累计 sparse outputs；不得
  无条件分配完整 `(S,T)` random/mask arrays。geometric skip 等 `O(P)` fast path 可以后续加入，
  但不能改变公开协议或统计语义。
- distance-dependent rules 可以后续使用 bounded chunking 或 spatial index；其 materialized
  result 仍进入相同 PairTable validator。
- custom PairRule 负责 callable 内部的生成时间与临时内存；Network 负责最终 array 的
  dtype、shape、membership、bounds、duplicates、`nsyn`、sorting 和 immutable normalization。
- 所有生成路径在返回后执行相同 canonical source/target sorting，并据此分配当前
  snapshot 的 dense `pair_id` 和 contact row order；durable `contact_id` 由 [I-09](#i-09-generated-instance-identity)
  的 allocator 分配。seed determinism 和未来算法替换必须遵守
  [I-10](#i-10-materialization-time-rng-contract)；无法保持 topology 时必须显式版本化。
- v1 不支持 incremental validation、external merge、lazy PairTable 或 out-of-core lowering。
  只有最终 `P` 本身成为实际规模瓶颈后，才重新评估 streaming protocol；不得恢复公开
  candidate/filter/score PairPipeline。

### Resolved examples

```python
source = braincell.filter.cells(population="grc", where=select_1k)
target = braincell.filter.cells(population="pc", where=select_2k)

# Materializes exactly 1,000 * 2,000 explicit pair rows.
net.add_projection(..., source=source, target=target, pair_rule=all_to_all())

# Logically tests independent Bernoulli candidates, but does not retain an S*T mask.
net.add_projection(..., source=source, target=target, pair_rule=probability(p=0.01))
```

### Gate

Phase 3 PairRule implementation must pass subset cardinality、exact degree、Bernoulli statistics、
seed determinism、canonical identity 和 peak host-memory tests。最终 PairTable、ContactTable 和
runtime 数值语义不得依赖 built-in generator 使用 direct、chunked 或 future skip algorithm。

## I-05 Weighted continuous Region sampling

**Status:** `LOCKED`

### Problem description

对 continuous Region 按权重采样时，概率质量同时取决于位置密度和几何测度：

```text
p(d location) proportional to density(location) * d measure
```

即使先按整个 branch 的总权重选中 branch，当 density 在 branch 内变化时也不能再
均匀抽取 `branch_x`。任意用户 callable 还会引入积分精度、尖峰/跳变检测、JAX 可追踪性和
可微语义，v1 不对这些行为做过早承诺。

### Locked decision

- continuous Region sampler 在 morphology-native branch/segment 几何上建立累积概率，反演产生
  continuous `branch/x`，然后才 lowering 到 CV。更改 CV policy 不得改变该采样分布。
- `measure="length"` 使用 `density(x) * ds`；`measure="area"` 使用
  `density(x) * dA`。density 和 base measure 正交，不由 density 使用的距离属性自动选择。
- `density=None` 是 uniform density；显式 `uniform_density()` 与之等价。
- v1 公开两个 tree-distance profiles：
  `exponential_tree_distance(origin, length_constant)` 和
  `gaussian_tree_distance(origin, center, sigma)`。`origin` 必须解析为唯一 continuous location，
  例如 `at("soma", 0.5)`。
- density 是无量纲、非负的相对权重；distance profile 参数必须携带 length unit。
  负值、非有限值、无法解析的 origin 或请求 contacts 时零总质量均报错。
- density 只控制已确定 contact 的位置分布，不控制 `nsyn` 或 pair/contact 数量。
- 距离范围使用 `RegionExpr` 与目标 Region 求交，不再增加重复的 density-window API。

profiles 定义为：

```text
exponential: exp(-tree_distance(origin, x) / length_constant)
gaussian:    exp(-0.5 * ((tree_distance(origin, x) - center) / sigma) ** 2)
```

`length_constant > 0`、`sigma > 0`；`center >= 0`。这些分布是 morphology path distance，不是 world-space
Euclidean distance。`delay` 不用作 density 参数名，仍专指传导延迟。

### Deferred follow-ups

- 任意 Python callable 和其 vectorization/JAX contract；
- adaptive quadrature、用户可见 tolerance/order 和尖锐/不连续 density 的精度保证；
- 基于任意 morphology properties 的自定义 location context；
- 明确依赖 CV state/attributes 的 discrete candidate sampler；
- density 参数可训练以及对采样位置求梯度。

这些 follow-ups 不阻塞 v1 内置 profiles；需要 CV-specific 值时必须选择未来的显式 discrete
sampler，不得让 continuous Region 隐式依赖 CV resolution。

## I-06 Batch placement lowering

**Status:** `OPEN`

### Problem

ContactTable 是 paired C rows；`LocsetMask` 已能保序表达重复 locations，但现有 PlaceRule 和
CellSelection 仍不能一次携带任意 C 行 population/contact ownership 与异质 parameter columns。
逐 contact 调用 `place()` 会产生高 Python overhead。

### Options

- A. 新增 columnar `InstancePlaceRule`，统一进入 discretization；
- B. network lowering 直接创建 discretization point-placement records；
- C. 临时对每个 contact 调用现有 Cell API。

### Current leaning

A。它保留 declaration/discretization ownership boundary，并能与手动 PlaceRules 合并。
C 只允许作为 correctness prototype，不能通过性能验收。

### Decision deadline

Phase 5 batch lowering 开始前。

## I-07 Delay runtime semantics

**Status:** `LOCKED`

### Decision

- ContactTable 保存 canonical `(C,)` physical time Quantity；run setup 根据 `dt` 量化。
- 默认使用 `ceil`，事件不能早于用户请求的 delay。
- `0 ms` 仍在 next solver step delivery。
- public `delay_quantization` 固定支持 `ceil`、`strict` 和 `floor`；`strict` 要求整数 grid，
  `floor` 只有显式选择时才允许提前到最近网格。
- v1 使用 fixed-step ring buffer。
- 连续 `run()` 保留未到期 events；`reset_state()` 清空 delivery queue。
- 第一次 `run()` 将 episode 绑定到 `dt` 和 quantization mode；time 已推进后不能改变任一值，
  `reset_state()` 解除两者绑定。

### Implementation gap

当前 runtime 已实现 `ceil/floor/strict` 和 next-step zero delay，但每次 `run()` 都清空
delivery state。实现阶段必须修复跨 run queue ownership，并增加等价性测试。buffer
layout 的性能问题单列为 [I-11](#i-11-heterogeneous-delay-buffer-layout)。

### Gate

Phase 6 event delivery 完成前通过三种 quantization mode、zero delay、continued-run、episode
binding 和 reset tests。

## I-08 Trainable versus non-trainable fields

**Status:** `LOCKED`

### Decision

v1 generated topology 和 locations 是 static materialized structure，不可训练。weights 和
mechanism parameters 是模型参数：普通 ContactTable column 默认不自动成为 trainable leaf，
但可通过显式 schema/selection 进入 Network parameter PyTree，并在 shape 不变时更新而不改变
结构。

模型参数与 dynamic state 分离；`reset_state()` 和 `deinit_state()` 不回滚当前参数值。普通
`set(...)` 与结构失效后的映射由
[I-12](#i-12-model-mutation-ownership-and-cache-invalidation) 冻结。optimizer assignment、
checkpoint ownership、STDP 与训练期间的符号/约束策略属于 I-08 后续训练接口，不由 I-12
的静态 mutation contract 定义。

## I-09 Generated instance identity

**Status:** `LOCKED`

### Problem description

同一 postsynaptic cell 上的多个 contacts 可以共享 location、CV、electrical point 和
mechanism model，却仍有独立 parameters、event input 和 runtime state。删除一个 contact
或重新排序 packed layouts 时，不能让旧 ID 静默指向另一个实例。

例如初始 IDs 为 `[0, 1, 2]`，删除 contact 1 后应为 `[0, 2]`，而不是将原 contact 2
重命名为 1。之后新增 contact 获得 ID 3。与此同时，JAX arrays 只应保存三个存活
rows，不应按 `max(contact_id) + 1` padding。

Network 必须稳定追踪：

```text
(projection owner, contact_id)
  -> current ContactTable row
  -> target population/cell/location
  -> placement_id
  -> runtime layout/local state index
  -> probes/inspection
```

自动生成 names 还可能与 model factory 中的手动 mechanisms 冲突，因此 display name
不能兼任 identity key。

### Locked decision

- public generated-contact identity 是 `(Projection handle, contact_id)`。可读展示可以使用
  `(projection_name, contact_id)`，但内部 owner 是 Projection 生命周期内唯一的 immutable token，
  不依赖可复用的 name 或 Python object address。
- `contact_id` 是 projection-local host `int64`，从 0 单调分配。删除留空洞，已退役 ID
  在该 Projection 生命周期内永不复用。
- ContactTable 只保存存活 contacts 的紧凑 `C` rows；`contact_id` 是 `(C,)` column，
  row index 不是 identity。`pair_id` 仍是指向当前紧凑 PairTable 的 foreign key。
- rematerialization 以 `(source_id, target_id, synapse_index)` 作为 active contact key。仍然存在的
  key 保留 ID；消失的 key 退役 ID；之后重新出现的相同 key 是新 contact，获得新 ID。
- 只有成功提交的 ContactTable snapshot 会分配或退役 IDs。在下一次 inspection/
  `init_state()` 之前被连续覆盖的 declaration 修改不产生中间 topology，也不消耗 IDs。
- weight、delay、target location 或 mechanism spec 变化不改变仍存活 contact 的 ID。
  `nsyn` 缩减会退役消失的 slots，以后增大不复用旧 IDs。
- 删除整个 Projection 时，owner token、allocator state、managed placement layer
  和 mappings 一起销毁。之后同名新 Projection 是新对象，可从 0 开始；旧 handle 报
  `ReferenceError`。
- lowering 显式保存 `contact_id -> contact_row -> placement_id -> layout/runtime row`
  mapping。`placement_id`、`point_id` 和 runtime row 可在重新物化时变化，不是 durable IDs。
- `contacts[...]` 保留当前 row/mask selection 语义；`contacts.by_id(id_or_ids)` 按稳定 ID
  查找。退役或未分配 ID 报 `KeyError`，不按 array offset 猜测；不同 Projection 的
  local ID namespaces 相互独立。
- exact colocated contacts 保持独立 placement/state，即使共享 `point_id`。display name 可包含
  projection/contact 信息并使用冲突后缀，但只用于 debug 和 inspection。

I-09 不新增逐 contact 删除 API；上述行为首先适用于 PairRule、`nsyn` 或 Projection
mutation 导致的 rematerialization。

### Gate

Phase 5 batch lowering 必须通过 middle deletion/new allocation、rematerialization preservation、
`nsyn` shrink/grow、colocated independent state、dense storage、stale handle、name collision 和
ID-to-runtime mapping tests。

## I-10 Materialization-time RNG contract

**Status:** `OPEN`

### Problem

`Network.seed=0` 是所有 framework-managed materialization randomness 的 root。现已
锁定 semantic stream allocation、explicit rule seed 的层级语义和 `ctx.rng`
callable interface；尚未选定的只是 NumPy、JAX 或 BrainState 的底层 sampling
adapter。I-10 因此保持 `OPEN`，但实现不得再改变下述 observable contract。

### Locked stream contract

默认 rule stream 从稳定 semantic path 派生：

```text
derive(Network.seed, domain="auto", object kind/name, materialization stage, rule slot)
```

`ctx.rng.with_seed(stream_id)` 中的 `stream_id` 是用户显式选择的 rule-local stream
ID，不是替换 `Network.seed` 的绝对 seed：

```text
derive(Network.seed, domain="user", stream_id)
```

`auto` 与 `user` 使用不同 domain tag，不会因为编码巧合落入同一 stream。
同一 `stream_id` 可以故意在不同 rules 中表示 matched randomness：每个 rule evaluation
持有独立的可变游标，但它们从相同初始 stream 开始。只有当两个 rules 的
sampling methods、arguments、shapes 和调用顺序都相同时，才承诺逐元素相同样本。

同一 evaluation 内重复请求同一 `stream_id` 返回同一 evaluation-local handle，
后续调用继续消费已确定的序列，不从头重放：

```python
def rule(ctx, seed=7):
    rng = ctx.rng.with_seed(seed)
    a = rng.uniform(size=2)
    b = ctx.rng.with_seed(seed).uniform(size=2)  # the next two draws
    return a, b
```

下一次 evaluation 创建新的 local handles，并从同一派生状态重放。因此相同
Network/config/inputs 可复现，同时不在 rules 之间共享会受 evaluation order 影响的
mutable generator。

### Locked observable behavior

- 改变 `Network.seed` 会改变所有 automatic 和 explicit framework-managed streams；
- 改变一个 `stream_id` 只影响显式使用该 ID 的 rules；
- 调换互不依赖 objects 的 add order、插入无关 object、inspection order 和
  concurrent evaluation order 都不改变已有 streams；
- cache refresh 或依赖 context 变化不进入 stream key；它们会触发重新计算，
  并可以通过新的分布、shape 或 mapping 改变最终结果；
- semantic identity 使用有版本的 canonical encoding，不使用 Python randomized
  `hash()`；
- Network 不读取、重设或推进 NumPy、Python 或 BrainState process-global RNG state。

例如连续 location sampling 可先从稳定 stream 取得分位数 `u`，再根据当前
context 中的密度 CDF `F` 计算 `x = F^-1(u)`。context 改变时 `F` 可变，但不需要
为此改变基础 stream。

精确逐元素复现只承诺相同 BrainCell、RNG libraries、precision、backend 和执行环境。跨版本
长期复现必须保存 materialized tables，不能只保存 seed。

### Locked callable boundary

- `ctx.rng` 是 backend-neutral stateful facade，提供 `uniform/normal/choice/...` 等经锁定
  sampling protocol；它不向 callable 暴露 raw JAX key；
- 直接使用 `ctx.rng` 表示当前 rule 的 automatic semantic stream；使用
  `ctx.rng.with_seed(stream_id)` 表示 Network-rooted explicit stream；
- v1 不增加 `seeded()` wrapper，custom callable 直接在函数体内选择 `ctx.rng` 或
  `ctx.rng.with_seed(...)`；
- built-in stochastic rules 的 `seed=None` 选择 automatic stream，显式 integer 选择
  对应 user stream ID；
- custom callable 只有使用 `ctx.rng` 才享有 I-10 guarantee；自行创建 `np.random`、
  Python `random`、global `brainstate.random`、closure RNG 或第三方 RNG 时由用户负责
  reproducibility 和 cache consistency；
- rule-local facade/handle 只在当次 evaluation 内有效，不能保存后跨 refresh
  继续消费；
- Network 不通过临时重设 global seeds 模拟控制，因为该方法会破坏 nested/
  concurrent materialization。

Spec/repr 对 built-in rules 显示 automatic 或 explicit stream ID。custom callable 内部的
`with_seed()` 调用不从 Python 签名静态猜测；resolved materialization diagnostics 记录
实际请求过的 semantic path、root 来源和不可逆 key fingerprint，不显示可重建的
raw key。

### Open implementation axis

唯一未锁定的设计轴是 sampling backend。它可以影响 host build cost、device dispatch、
array conversion 和 ecosystem consistency，但不得改变上述 facade 和 stream semantics。

### Candidate designs

#### A. NumPy semantic streams

将 canonical derived identity 转为 `SeedSequence` entropy，每次 evaluation 构造局部
`Generator` adapters。它最贴合当前 host materialization，且不触发 JAX
compilation/device transfer；代价是 materialization RNG 与 BrainState/JAX runtime 分成两套
体系。不能按 allocation order 调用 `SeedSequence.spawn()`。

#### B. JAX-backed stateful facade

以 `jax.random.key(Network.seed)` 为 root，将稳定 semantic identity 编码为 integer words 并
用 `jax.random.fold_in()` 派生 rule key；BrainCell facade 内部显式 split 并推进局部 key。
该方案最符合 JAX functional model 和未来 `jit/vmap`，但所有 host samplers 都必须
处理 JAX arrays、CPU placement 和 host conversion，并且 facade 必须防止 raw-key reuse。

#### C. JAX semantic keys with BrainState RandomState

root/semantic derivation 与 B 相同，但每次 rule evaluation 用派生 key 新建局部
`brainstate.random.RandomState` adapter。相同派生 key 重建时序列复现。BrainState
只管理 rule-local sequential sampling，不提供 BrainCell 所需的 semantic identity
derivation。

上述 candidates 都必须通过同一 BrainCell facade；backend-neutral semantic manager
不再是独立候选，而是已锁定 architecture。evaluation-order root splitting 因违反
order independence 而正式排除，只作为 benchmark 的 negative baseline。

### Exploration evidence

2026-08-17 在 `braincell_311` 环境验证了 Python 3.11.15、JAX/JAXLIB 0.10.1、
BrainState 0.3.0 和
BrainTools 0.1.9 可以共同运行。JAX typed key、`fold_in`、CPU device placement，以及用同一
typed key 重建 `brainstate.random.RandomState` 后复现 `uniform()` 结果均已通过交互检查。
BrainState 的 global `DEFAULT/split_key()` 会按调用顺序推进，因此不能单独满足当前 semantic
isolation requirements；BrainTools 也没有现成的 named/semantic stream manager。

JAX key/fold-in 设计参考
[JAX random](https://docs.jax.dev/en/latest/jax.random.html) 和
[typed keys](https://docs.jax.dev/en/latest/jep/9263-typed-keys.html)；NumPy candidate 参考
[parallel RNG/SeedSequence](https://numpy.org/doc/stable/reference/random/parallel.html)。

### Evidence required before selection

对 A/B/C 使用相同的 `probability`、fixed-degree、Region/Locset sampling 和 per-contact
parameter callable，比较 cold/warm build time、peak memory、host conversion、precision、
facade overhead 和 key-reuse failure modes。所有候选必须先通过已锁定的 Network/
stream-ID composition、matched-stream、order independence、cache rematerialization 和 global RNG
non-interference tests；这些是 acceptance gates，不是可根据 benchmark 改变的语义。

### Decision deadline

Phase 2 context/RNG implementation 提交前完成 backend benchmark，选择 sampling adapter，
并在不改变已锁定 stream contract 的前提下关闭 I-10。

## I-11 Heterogeneous delay buffer layout

**Status:** `OPEN`

### Problem

当前 `delivery.py` 为一个 lowered block 中每个 unique `delay_steps` 建立 `DeliveryBlock`，
并为每个 block 分配 depth 为 `delay_steps + 1`、宽度为完整 target layout 的 ring buffer。
当 contacts 的 delay 高度异质时，内存可能近似随 unique delay group 数、最大 delay 和
target layout size 的乘积增长。

这不改变 I-07 已锁定的 physical delay、quantization modes 和 event phase 语义，但可能违反
大规模异质 network 的性能目标。

### Options

- A. v1 保持 per-delay-group dense target ring buffers，明确规模边界；
- B. 一个 shared time ring，每个 slot 保存完整 target-layout arrivals；
- C. contact/event sparse queue，在到期 slot 才 scatter 到 target layout；
- D. 根据 delay diversity 和 density 在 B/C 之间选择内部 backend。

### Current leaning

先保留 A 作为 correctness baseline，同时建立 memory benchmark。若 unique delays 增长时
buffer memory 明显超过 contact/state storage，则优先评估 B；只有 full-layout slots 本身
仍不可接受时再评估 C/D。内部优化不得改变 API 或 I-07 event timing。

### Decision deadline

Phase 6 runtime gate 前完成 baseline benchmark，并关闭本 issue 或明确记录 v1 支持的 delay
diversity/queue-memory 规模上限。

## I-12 Model mutation, ownership, and cache invalidation

**Status:** `LOCKED`

### Problem description

统一的 `set(...)` 必须区分 declaration、当前 materialized table snapshot、模型参数和
dynamic state，并在修改上游对象后只使正确的下游结果失效。Network-generated placements
还必须保留 Projection owner，才能在删除或修改连接时撤销，而不影响手动
`Cell.place(...)`。

### Example

```python
proj.set(weight=weight_rule)                     # 修改整列生成声明
proj.contacts.by_id([2, 7]).set(weight=0.2 * u.uS)  # 修改当前 snapshot rows

net.remove_population("pc", cascade=True)       # 同时删除依赖 Projections
```

例如用户直接把 contact 7 的 weight 改为 `0.2 uS` 后，若 PairRule 的上游输入改变，weight
stage 必须在新 contacts 上重新求值；这次 table edit 不得静默成为永久 override。反过来，
仅修改一个没有任何 rule 读取的 population property，也不应让所有 Projections 无条件重建。

### Locked decision

- Population、Projection 和 materialized row views 使用统一 `set(...)` vocabulary。handle 上
  的 `set(...)` 修改 declaration；row view 上的 `set(...)` 修改当前 table snapshot。
- PopulationInstances 可写值为 position、rotation 和 properties。contact-aligned 可写值为
  target location、weight、delay，以及由 `Projection.parameters` 暴露的 mechanism parameter
  columns。cell ID、source/target ID、
  `pair_id`、`contact_id`、`synapse_index` 和 `nsyn` 是只读 identity/topology；pair topology
  只能通过 PairRule 修改。
- ContactTable 是 contact-aligned mechanism parameter columns 的唯一 backing storage；
  `Projection.parameters` 是共享 current-row order、storage 和 `by_id(...)` selection 的 typed
  view，不建立第二套 contact ordering。Network-owned Cell 的静态 channel/mechanism parameter
  edit 也作为 dependency producer 进入相同 transaction；dynamic state 仍排除在外。
- table edit 不是持久 sparse override。它增加对应 stage/field 的 producer version 并使实际
  消费该值的下游 stale；若该 stage 的任一已记录上游依赖变化，它从 declaration/rule 完整
  重建，当前 snapshot edit 消失。希望跨刷新保留的行为必须写入 declaration/rule；v1 不维护
  override registry 或 `clear_override()`。
- Network config 只供 rule 显式读取，不自动缩放 number、weight 或 delay。
- Projection rule 可以读取更早完成的 Projections；add order 是依赖方向，反向和循环依赖报错。
- Context 只能读取同一 Network 的 resolved results，不允许跨 Network 自动依赖。
- static cache record 保存直接输入版本；Context read 记录 producer-to-consumer dependency。
  失效采用 stage/field-level conservative granularity，不追踪逐 row 依赖。刷新在下一次相关
  属性访问或 `init_state()` 时惰性执行。
- rule evaluation 中的 `ctx.current` 是 transaction-local progressive candidate view，只允许
  读取已完成的上游 fields；未来 field 访问报 forward-dependency error。factory 对 candidate
  PopulationInstances 的实际 field reads 进入同一 dependency graph，失败 candidate 不发布。
- manual placements 与按 Projection owner 管理的 placement layers 合并进入
  `Cell.point_placements`；删除/替换 Projection 只撤销对应 layer。
- 删除仍被引用的 Population 默认报错；`cascade=True` 显式删除相关 Projections 和 layers。
- 每个 Projection 使用生命周期内唯一 owner token。删除整个 Projection 时一并销毁 managed
  layer、contact identity registry 和映射；同名新 Projection 不继承旧 ownership。
- `INITIALIZED` 时，mutation 必须先验证完整传递依赖闭包。仅当 shape、dtype、runtime layout、
  mechanism schema 和 runtime signature 全部保持不变时才允许原子提交。weight 和普通
  mechanism parameter 通常满足；若有下游 rule 读取它并改变结构或 layout，同样拒绝。
- delay update 只有在 delay quantization groups、event-buffer layout 和全部下游 signatures
  均不变时允许。target location、PairRule、`nsyn` 和 mechanism type 默认是结构修改，必须先
  `deinit_state()`。
- mutation 与 lazy refresh 都先在临时结果中完成验证再提交。mutation 验证失败不提交输入
  version，原模型仍有效；已经 stale 的 refresh 失败则保留旧 snapshot 并保持 stale。两者都
  不发布新旧混合状态。
- `reset_state()`、`deinit_state()` 都保留模型参数。voltage、gating variable、synaptic state
  等 runtime dynamic state 不由 `.set(...)` 修改，后续接口见
  [I-14](#i-14-runtime-state-mutation-api)。
- `proj.spec.weight` 检查生成声明，`proj.contacts.weight` 检查 resolved canonical contact column。

### Resolved behavior

```python
proj.contacts.by_id(7).set(weight=0.2 * u.uS)
assert proj.contacts.by_id(7).weight == 0.2 * u.uS

grc.instances[0].set(position=new_position)
# If weight_rule read grc.position, the next access rebuilds the weight stage.
print(proj.contacts.weight)   # the snapshot-only edit may now be replaced

net.init_state()
proj.set(weight=0.3 * u.uS)   # accepted only if the transitive runtime signature is unchanged

net.deinit_state()
proj.set(pair_rule=new_rule)  # structural mutation is editable again
print(proj.contacts)          # lazily and atomically rematerializes the affected chain
```

### Gate

Phase 2/5/7 必须通过 dependency-selective invalidation、snapshot edit replacement、stable
contact identity、owner-layer cleanup、initialized signature gate、conditional delay update 和
atomic refresh rollback tests。

## I-13 Kinetic and continuous synapse input protocols

**Status:** `DEFERRED`

### Problem description

当前 `AMPA`、`GABAa` 和 `NMDA` 在 ODE 中连续读取 `pre_drive()`，例如
`g' = alpha * pre_drive * T * (1-g) - beta*g`。它们尚未定义如何把一次 spike event 转换为
具有明确幅值和持续时间的 transmitter pulse。把 spike 仅在一个 solver step 写成 `1` 会使
效果依赖 `dt`；把 conductance weight 直接写入又会造成量纲错误。

这类 receptor kinetics 仍是正常化学突触，不等同于 gap junction。完整 event-driven kinetic
model 应在内部将离散 event 转换为 transmitter/release state，再让 receptor states 连续演化。
gap junction 或 graded synapse 则连续读取 presynaptic voltage，根本不经过 spike detector、
event delay 或 event weight。

### Locked v1 boundary

- v1 先纵向打通 registry 已声明 scalar/trigger event port 的 Projection；初始标准目标是
  `ExpSyn/Exp2Syn`。
- 当前连续读取 `pre_drive()` 且没有 event bridge 的 models 不得自动登记成 event target。
- `net.add_projection(...)` 保持稳定入口。未来 kinetic event target 通过新增 registry port
  contract 和内部 lowering 横向扩展，不改变 pair/contact/location 主流程。
- continuous/graded coupling 若以后复用 `add_projection(...)`，必须使用独立 input-port
  schema，并拒绝 `source_threshold/delay/event weight` 等无意义字段，不能伪装成事件投递。

### Follow-up design

kinetic 扩展需要单独冻结 transmitter waveform、pulse duration、重复事件叠加/饱和、
per-contact release state 和 reset semantics；continuous coupling 需要单独冻结 source
variable、solver coupling 和参数 ownership。两者均不阻塞 event Projection v1。

## I-14 Runtime state mutation API

**Status:** `DEFERRED`

### Problem description

训练、干预或交互式实验可能需要在 `INITIALIZED` Network 中修改 membrane voltage、channel
gating variables、synaptic conductance/release state 等 dynamic state。这些值属于当前运行轨迹，
既不是 declaration，也不是 ContactTable/PopulationInstances 的静态模型参数，不能复用 I-12
的 `.set(...)` 而混淆 reset、event queue 和 JAX state ownership。

### Deferred boundary

- v1 不承诺运行中 dynamic state 的 public mutation API；I-12 的 static/model mutation 不因此阻塞。
- 后续接口必须显式选择 runtime state，使用 JAX-compatible immutable replacement，校验 shape、
  dtype 和单位，并定义 JIT/compiled-loop cache 是否可复用。
- 必须分别定义修改 voltage、mechanism state、detector previous voltage 与 pending event queues
  后的同步语义，不能假设它们彼此独立。
- 必须明确 `reset_state()` 是回到初始化基线还是用户修改后的新基线，以及 checkpoint restore
  如何处理 time、RNG 和 event queues。
- optimizer/checkpoint 的参数 ownership 仍由 I-08 训练接口负责；本 issue 只负责 dynamic state。
