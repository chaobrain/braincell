# Network Builder 内部架构规范

本文定义 [api.md](./api.md) 背后的对象边界、物化表、materialization/lowering/runtime
数据流，以及
与当前 `braincell/network` 实现的融合方式。本文与 API 规范具有同等约束力；内部重构
不能静默改变 API 行为。

## 1. 分层与不变量

### 1.1 Specs 是声明，tables 是结果

`Population`、`Projection` 是 Network-owned handles；其 `.spec` 保存不可变 effective
declaration。`PopulationSpec`、`PopulationSelector`、`ProjectionSpec` 和 `SynapseSpec`
不保存某次 seed 生成的 arrays，也不持有 runtime state。解析结果进入受 lifecycle 和 schema
约束的 `PopulationInstances`、`PopulationView`、`PairTable`、`ContactTable` 和 resolved
runtime metadata。PairTable 及 identity/topology columns 只读；API 明确列出的 model-value
columns 只能通过受控 `set(...)` transaction 修改。

```text
Declaration
  "fixed indegree=20，按 dendritic length 放置"

Materialized result
  "seed=1 实际产生的 pair、contact、location 和 parameters"
```

### 1.2 Pair 与 contact 是不同的 rows

一个 `(source cell, target cell)` pair 可以生成 `nsyn` 个 contacts。每个 contact 都是新的
postsynaptic point-mechanism instance。即使 target cell、branch/x 和 parameters 完全相同，
contacts 仍具有独立 identity 和 state，不允许合并。

### 1.3 稀疏是唯一 pair representation

PairRule 直接返回 sparse pair rows。declaration、context、table、placement 和 lowering
都不接受或保存 cell-pair adjacency matrix。built-in rule 可以使用内部 chunking，但不能
将 dense matrix 暴露为主数据模型，也不能把 `(rule, seed)` 当作已经物化的 topology。
大规模执行 contract 见
[I-04](./issues.md#i-04-pairrule-scalability)。

### 1.4 连续位置是权威信息

`(branch_id, branch_x)` 是 contact placement 的权威位置。`cv_id`、`point_id`、
`layout_id` 和 runtime state index 都是 lowering 后的派生结果。electrical node 共享不能
抹掉 contact identity 或 morphology provenance。

### 1.5 Materialization 与 runtime 使用不同 backend

Network generation 是 host-side static process，使用 host morphology/geometry、BrainUnit
Quantity 和 canonical table validators。materialized tables 最终保存为 host arrays，并在
initialization/lowering 时转换为 JAX/runtime arrays。materialization RNG 分为两层：
backend-neutral semantic stream manager 实现已锁定的 path derivation 和 stateful facade，
sampling adapter 负责 NumPy、JAX 或 BrainState 的实际 draws。I-10 仅仍在比较后一层。
若选择 JAX/BrainState，rule-local key/sampling 可以固定在 CPU，输出再规范化为
host tables，而不把
topology generation 变成 runtime JIT contract。v1 不要求 topology、location 或 parameter
generation 可微；trainable generated values 见
[I-08](./issues.md#i-08-trainable-versus-non-trainable-fields)。

## 2. 对象和表

### 2.1 Declaration objects

```text
Network
  ├── Network configuration
  ├── Population[] handles -> PopulationSpec + PopulationInstances + Cell
  └── Projection[] handles -> ProjectionSpec
        ├── PopulationSelector source/target -> PopulationView
        ├── PairRule -> PairTable
        ├── target_loc -> ContactTable location columns
        ├── SynapseSpec + resolved parameters
        └── event source / weight / delay rules
```

specs immutable；callables 在 spec 中保持未执行。`Network` 负责注册顺序和 lifecycle，
不允许 rule 通过 context 修改当前 registry。handles 暴露 inspection 和受 lifecycle 约束的
mutation，不额外发布 `ResolvedProjection`。

### 2.2 PopulationInstances 与 PopulationView

每个 population 的列式 table：

```text
cell_id              int32, (N,), values 0..N-1
position             length Quantity, (N,3), or None
rotation_quaternion  float, (N,4), normalized (w,x,y,z)
properties           frozen mapping, each leading dimension N
```

Cell、morphology、spatial anchor 和 spec 是 owner metadata，不在每一行重复 Python
objects。`PopulationView` 保存 base instances identity 和筛选后的稳定 `cell_id`；其余 columns
是按 IDs 提供的 read-only gathered views。

public `.rotation` 是由 `rotation_quaternion` 支撑的 canonical `Rotation` view。默认 spatial
anchor 是 `RootLocation(0.5)`；`position[i]` 精确定义为该 anchor 的 world coordinate。Cell
只保存共享 morphology-local coordinates，不保存或应用 population position。对 local point：

```text
world_point = position[i] + rotation[i] @ (local_point - anchor_point)
```

`position=None` 时 world transform 未定义，依赖 world distance 的 rule 必须在 materialization
时报错。

`Population.cell` 是实际 cached uninitialized batched Cell；`Population.cell[id]` 返回
CellSelection。`Population.instances[id]` 返回网络 metadata row，两者不能混为一层。

### 2.3 PairRule array 与 PairTable

PairRule 原始返回 strict integer array `(P,3)`，columns 为 source ID、target ID 和 nsyn。
validator 拒绝 bool/float/object、out-of-view IDs、`nsyn<=0` 和 duplicate pairs，然后排序并
分配 projection-local `pair_id`：

| pair_id | source_id | target_id | nsyn |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 3 |
| 1 | 1 | 0 | 2 |

在当前 materialized topology snapshot 内，pair reference 是 `(projection owner, pair_id)`。`pair_id`
是 canonical dense row，topology mutation 后可以改变；contact 的 durable identity 不依赖它。

PairTable 的 logical source of truth 是 COO-style rows。内部可以保存独立的
`source_id/target_id/nsyn` column arrays，而不要求长期保留一个二维 ndarray。CSR/CSC、
source pointers、target indices、sorted permutations 和 JAX sparse structures 都是可重建的
derived caches；删除 cache 不能丢失 topology 或改变 canonical rows。

built-in PairRule generation 与 table storage 分层：selectors 先产生 sizes `S/T`，generator
可以 direct、bounded chunk 或 future skip sampling，之后统一进入 array validator 和
canonical sort。`all_to_all` 的 `S*T` output 是必要分配；fixed-degree 和 sparse probability
不得用额外 `S*T` intermediates 掩盖 PairTable 本身的 `O(P)` storage。

### 2.4 ContactTable

PairTable 按 `nsyn` 展开成 normalized contact rows：

```text
contact_id       projection-local host int64, (C,), monotonic and non-reused
pair_id          foreign key into PairTable, (C,)
synapse_index    0..nsyn-1 within pair, (C,)
target_branch_id int32, (C,)
target_branch_x  float, (C,)
weight           canonical scalar Quantity, (C,), or absent for trigger-only contacts
delay            canonical time Quantity, (C,)
parameters       mapping; every mechanism parameter has leading dimension C
```

`source_id` 和 `target_id` 通过 `pair_id` gather，不在 C rows 中重复。source location、
threshold 和 mechanism model 是 Projection metadata。declaration 中的 scalar weight、delay
和 parameters 在 ContactTable 发布前广播并规范化为 C rows。ContactTable 是 contact-aligned
mechanism parameter columns 的唯一 backing storage；`Projection.parameters` 只是同一 columns
和 contact-ID mapping 上的 typed view/alias，不复制数据或维护第二套 row order。

ContactTable 的物理 row axis 是紧凑 `0..C-1`，不是 identity namespace。每个活跃
Projection 持有：

```text
owner_token          opaque Network-assigned Projection lifetime identity
next_contact_id      next monotonic host int64
active_contact_ids   (source_id, target_id, synapse_index) -> contact_id
```

owner token 不使用 Projection name、Python `id()` 或 object address。rematerialization 在验证完整
PairTable/contact keys 后事务性更新 registry：对 key 交集复用原 ID，按当前 canonical
contact row order 为新 keys 分配 IDs，并从 active map 删除消失 keys。删除的 ID 即退役；
无需保留 tombstone rows 或长度为 `max_id + 1` 的 arrays，`next_contact_id` 已保证它们
不会复用。验证/生成失败不得提交 registry 或消耗 IDs；int64 溢出前明确报错。
因为 materialization 是 lazy 的，只有被 inspection 或 `init_state()` 触发并成功提交的
ContactTable snapshot 参与这次 diff。一系列未物化 declaration edits 被合并为最终一次
diff，不为中间状态分配或退役 IDs。

contact identity 是 mutation-history aware：一个 key 连续存活时，weight、delay、location 或
mechanism spec 变化不改 ID；key 在任何一次成功 rematerialization 中消失后，以后
重新出现是新 contact。因此稳定性范围是当前 Projection 对象的生命周期，不是跨
Projection 删除/重建或跨独立 session 的 persistent identifier。

### 2.5 Contact locations 与 ResolvedEventSource

`LocsetExpr` 在 target Population morphology 上延迟求值为 ordered、duplicate-preserving
`LocsetMask`。一行 location 广播；C 行按 canonical contacts 一一对应；sampling 必须由
显式 `sample(...)` 声明。物化后 target branch/x 直接进入 ContactTable columns，不保留额外
public LocationBatch。降低后必须额外保存稳定 mapping：

```text
(projection_owner_token, contact_id)
  -> current ContactTable row
  -> target population/cell/branch/x
  -> placement_id
  -> cv_id/point_id
  -> layout_id/runtime state index
```

public inspection 用 Projection handle 和可读 name 展示该 key。host-side `contact_id -> row`
lookup 使用 active map 或 sorted index，不创建按最大 ID padding 的 lookup vector。lowering 将当前
dense rows 和 `int32` runtime indices 传给 JAX；重新 packing 可以改变 placement/layout rows，但必须
原子替换上述 mapping，使 probes、parameter views 和 state inspection 仍按 contact ID 查找。

删除 Projection 会先撤销 owner token 对应的 managed placement layer，再销毁 registry、
allocator 和 mappings，并将已分发的 Projection/contact views 标记为 deleted。其后任何访问
报 `ReferenceError`。同名新 Projection 使用新 owner token，不与旧 mappings 发生关联。

continuous Region placement 必须在上述 CV lowering 之前完成。placement sampler 先将
`RegionExpr` 解析为 morphology-native branch intervals，并按原始 morphology segment 几何
建立跨全部 intervals 的 cumulative mass：

```text
mass(A) = integral_A density(location) * d measure
```

`measure="length"` 使用 arc-length `ds`；`measure="area"` 使用 tapered frustum lateral
area `dA`。sampler 生成一个 `u in [0, 1)`，在 global CDF 中找到对应 branch interval，
再反演该 interval 的 conditional CDF 得到 continuous `branch_x`。“按 branch 总权重选 branch，
再在 branch 内采样”是该过程的等价分解；density 在 branch 内变化时，第二步不得
均匀抽取 `branch_x`。

v1 density layer 是内置、immutable 的 expressions：uniform、基于 morphology path distance 的
exponential decay 和 Gaussian profile。它们产生无量纲非负相对权重，并由 placement
module 统一验证 units、finite values 和 positive total mass。数值积分/反演可对内置
profile 使用解析或确定性 morphology-native 近似，但该过程不消耗 RNG；只有最终
CDF sampling 消耗 materialization RNG。相同 morphology/Region/density 的几何质量和 CDF
可缓存。

CV IDs、CV measure 和 runtime state 不得进入 continuous density context，因此改变 CV
policy 只会改变最终 lowering，不会改变 branch/x 分布。任意 callable、adaptive
quadrature 和 CV-specific discrete sampling 保留为 [I-05](./issues.md#i-05-weighted-continuous-region-sampling)
follow-ups，不扩大 v1 `DensityExpr` contract。

每个 Projection 的 event source record 至少包含：

```text
source_population
source_cell_id[]
source_branch_id/source_branch_x/source_cv_id
source_threshold
source_variable = "v"
```

LocsetMask 的 host representation 使用只读 branch ID/x columns；兼容 `.points` 只用于行访问。
内部可以建立 membership/branch/sorted permutation indices，但原始顺序是语义，任何计算重排
必须映射回 canonical contact IDs。

### 2.6 Mechanism event input contract

registry entry 除 mechanism class 外还必须描述 input port；实际 contact value 不保存在
registry：

```text
no event port
  mechanism cannot be a spike-event Projection target

trigger event port
  payload schema = none
  effective weight must be None

scalar event port
  payload schema = canonical unit + scalar shape + finite requirement
  effective weight is stored in ContactTable.weight
```

`ExpSyn/Exp2Syn` 的 scalar port 使用 conductance canonical unit `uS`。validator 接受兼容
Quantity 并转换到 canonical unit，拒绝裸 physical values、错误量纲、非 scalar rows 和
non-finite values；不施加 sign bounds。`Projection.weight`、直接
`SynapseSpec.default_weight` fallback 和 callable 结果必须经过同一 validator。

registry contract 只描述 target 能消费什么，不描述 contact 的实际 strength，也不规定
`on_event` 必须执行加法。mechanism 可以执行 jump、assignment、trigger、release 或 kinetic
state transition。v1 runtime protocol 保持一个 contact 一个 scalar payload；扩展边界见
[I-03](./issues.md#i-03-weight-ownership-and-defaults) 和
[I-13](./issues.md#i-13-kinetic-and-continuous-synapse-input-protocols)。

### 2.7 NetworkContext result views

公开 `net.context` 与 rule context 使用同一个 `NetworkContext`。普通 inspection context 的
`current/rng` 为 None；rule evaluation 填充当前 result view 和该 rule 的 RNG facade。
facade 的 public sampling protocol、semantic derivation 和 local stream composition 已锁定；内部
adapter/device 仍由 [I-10](./issues.md#i-10-materialization-time-rng-contract) 选择。

stream manager 使用有版本的 canonical component encoding，不使用 Python `hash()`：

```text
automatic: derive(network seed, "auto", object kind/name, stage, rule slot)
explicit:  derive(network seed, "user", stream ID)
```

`auto/user` 是独立 domain。每次 rule evaluation 都建立新的 handle registry；同一
evaluation 内重复 `with_seed(id)` 返回 registry 中的同一 handle 以连续消费序列。
不同 rule evaluations 不共享 mutable handle；相同 explicit ID 只让它们从相同派生
状态开始，因此不会因并发或 inspection order 互相推进。

```text
PopulationResultView: name, size, instances
ProjectionResultView: name, pairs, contacts, parameters, source, target
```

Context 不暴露 specs、Cells、runtime 或 mutation methods。Contact logical view 通过 pair ID
gather source/target IDs、positions 和 properties；物理表无需在每个 contact 重复这些 columns。
rule evaluation 时，`current` 指向 transaction-local progressive candidate view，只允许读取
当前 stage 已完成的上游 fields；读取未来 field 报 forward-dependency error。population 的
factory 可以读取完成了 position/rotation/properties 的 candidate instances。其他 objects 始终
只暴露原子发布的结果，失败 transaction 的 partial candidates 不进入 registry views。
cache version、context fingerprint 和 evaluation count 不进入 stream key。dependency invalidation
使 rule 重新 evaluation，但是从相同 stream 初始状态重放；变化的 context 可以通过
新的 distribution、shape 或 mapping 得到不同最终结果。custom callable 绕过 `ctx.rng`
使用 global/external RNG 时，其 reproducibility、side effects 和 cache consistency 由用户
负责；Network 不修改 process-global RNG state。

built-in rule spec/repr 保存 automatic/explicit mode 和 stream ID。custom callable 不做 Python
signature introspection；materialization diagnostics 从当次 handle registry 记录实际使用的
semantic paths、root source 和不可逆 key fingerprints，不暴露 raw keys。

## 3. Network lifecycle

[I-02](./issues.md#i-02-network-lifecycle-and-initialization-boundary) 已冻结两个 public phases：

```text
EDITABLE
  declarations mutable
  static tables lazily materialized and cached
        |
        | init_state() -- atomic commit
        v
INITIALIZED
  structure/shapes frozen
  Cell, detector and delivery runtime allocated
        |
        +-- reset_state() --> INITIALIZED
        |
        +-- deinit_state() --> EDITABLE
```

不存在 public `BUILT` phase。inspection properties 是 static cache entry points，不是 phase
transitions。

### 3.1 Static cache graph

Population 的 materialization graph 为：

```text
number -> position -> rotation -> properties
       -> candidate PopulationInstances
       -> cell_factory/cell validation + spatial_anchor resolution
       -> publish PopulationInstances + Cell
```

candidate 只在当前 transaction 的 progressive `ctx.current` 中可见。factory 实际读取的
candidate fields 会记录为 field-level dependencies；未读取字段改变时不因此重建 Cell。只有
Cell、`pop_size`、morphology 和 anchor 全部验证成功，才原子发布完整
`PopulationInstances + Cell`。population 按 add order 物化；只有已发布结果才能进入后续
objects 的 contexts。后一个 population 可以读取更早 populations。Projection 可以读取所有
完成 populations 和更早
Projections；反向、循环和跨 Network 读取报错。Projection
按下列 stage graph 物化：

```text
resolve source/target PopulationView
  -> resolve source event template
  -> PairRule -> validate/sort PairTable
  -> expand nsyn -> ContactTable identities
  -> target placement -> ContactTable location columns
  -> spatial derived columns
  -> weight rules -> resolve fallback -> validate target event input
  -> delay rules
  -> mechanism parameter rules
  -> publish resolved projection cache
```

每个 cache record 至少保存 stage/field identity、当前 producer version、直接 input versions、
recorded Context dependencies、materialized value 和 stale flag。rule 通过 Context 读取 resolved
producer 时，内部 graph 记录 producer-to-consumer edge；Context 自身不公开 graph。v1 追踪到
stage/field，不追踪逐 row dependencies。

declaration mutation 增加声明输入版本；materialized table edit 增加当前 stage/field 的 producer
version。二者沿相同 dependency graph 使真实 consumers 及其下游 stale，不在 `set(...)` 中同步
执行昂贵 callbacks。table edit 是当前 snapshot 的新 canonical value，但不进入 declaration：
它自己的任一 recorded upstream version 改变时，该 stage 从 rule 完整重建并替换 table edit。
`Projection.parameters` 复用 ContactTable row/contact-ID mapping；Network-owned Cell 的静态
parameter edit 同样注册为 producer version，不能在 cache graph 外原地改 array。

Network config mutation 保守失效所有 Population 和 Projection derived caches；Population
mutation 失效该 population、按 add order 可能依赖它的后续 populations，以及实际依赖它的
Projections；Projection-local mutation 只从对应 stage/field 向下失效。下一次读取 property 或
`init_state()` 才按需刷新；inspection 顺序不得改变 semantic RNG 结果。完整 public contract 见
[I-12](./issues.md#i-12-model-mutation-ownership-and-cache-invalidation)。

### 3.2 Mutation transaction 与 runtime signature

每次 mutation/refresh 先在临时 transaction 中计算 affected dependency closure、候选 tables、
managed placements 和 lowering metadata，然后统一验证并提交 versions。mutation 候选验证
失败时不提交新的 input versions，原 graph 仍有效；已经 stale 的 refresh 若失败，则保留旧
snapshot 并保持 affected records stale。两者都不更新 published placement layer、identity
registry 或 runtime array，并保存可诊断的失败原因。

`EDITABLE` 允许候选 closure 改变结构。`INITIALIZED` 则比较 mutation 前后的 transitive runtime
signature，至少覆盖 shape、dtype、contact/placement/layout mappings、mechanism schema、delay
quantization groups 和 event-buffer layout。全部相同才可原子替换模型参数及受影响 runtime
values；任一项变化都报 phase error 并要求 `deinit_state()`。因此：

- weight 和普通 mechanism parameter 通常可更新，但 downstream rule 仍参与 closure 检查；
- delay 仅在 quantized groups、buffers 和全部 downstream signatures 不变时可更新；
- target location、PairRule、`nsyn` 和 mechanism type 默认是 structural mutation；
- voltage、gating、synaptic state 和 event queues 不进入此协议，留给
  [I-14](./issues.md#i-14-runtime-state-mutation-api) runtime-state API。

### 3.3 Atomic initialization

`init_state()` 按以下顺序提交：

```text
materialize every stale Population/Projection stage
  -> validate effective manual + managed placements
  -> freeze registry and static versions
  -> initialize all Population Cells
  -> lower projection event detectors and delivery blocks
  -> allocate detector/delivery dynamic state
  -> publish INITIALIZED phase
```

初始化期间不能发布 partial `INITIALIZED` state。任一步失败时，所有已初始化 Cells 必须销毁
本次创建的 runtime，detector/delivery allocations 必须丢弃，Network 回到 `EDITABLE`；已
验证的 static caches 可以保留。Network 给 resolved Population Cells 安装 owner guard，用户
直接调用 `cell.init_state()` 必须报错并指向 `net.init_state()`。

### 3.4 Run、reset 与 deinit

- `run(dt, duration, delay_quantization="nearest")` 只接受 `INITIALIZED` Network，不隐式调用
  `init_state()`。
- run 创建或复用 dt/quantization-specific lowering、delivery operators 和 compiled loop。
- 连续 `run()` 延续 current time、Cell states、detector previous voltage 和未到期 delayed
  events。
- 第一次 `run()` 将当前 episode 绑定到 `dt` 和 delay quantization mode；time 已推进后任一值
  改变都报错。
- `reset_state()` 重置 Cell dynamic state、current time、detector state 和 delivery queues，
  解除 episode 的 `dt` 和 delay quantization 绑定，但不改变 static tables、placements、model
  parameters 或 compiled caches。
- `deinit_state()` 销毁 Cell/detector/delivery runtime 和 compiled caches，保留 declarations、
  static caches、managed placements 和 model parameters，并回到 `EDITABLE`。
- repeated `init_state()`、在 `EDITABLE` 调用 reset/deinit，以及在 `INITIALIZED` 修改结构都
  明确报 phase error。

## 4. Point placement 与 Cell runtime

### 4.1 可直接复用的能力

当前 Cell/discretization/runtime 已支持：

- 独立 `PointPlacement` identity 和原始 branch/x provenance；
- 多个 placements 映射到同一 electrical point；
- population-specific point placements；
- packed `MechanismLayout.population_index/point_index`；
- packed state shape `(n_active, ...)`；
- point current scatter-add 回 population/point membrane arrays。

因此 network-generated contacts 不需要 `N_cell * max_synapses_per_cell` padding，也不需要
每个 contact 创建一个 Python `Synapse` object。

### 4.2 缺失的 owned batch declaration bridge

现有 `cell[indices].place(locset, mechanism)` 已能通过 ordered locset 表达重复 locations，但
不能无损承接任意 C 行 contact ownership 和异质 parameter columns。需要一个内部 columnar
placement contract：

```text
population_index   int32, (C,)
branch_id          int32, (C,)
branch_x           float, (C,)
contact_id         host int64, (C,)
mechanism spec     one shared declaration
parameter columns  leading dimension C
owner token        immutable Projection-lifetime metadata
```

Cell declaration 将 placements 分为 manual layer 和按稳定 Projection owner key 管理的
layers。Projection materialization 原子替换自己的 layer；删除 Projection 只撤销该 layer。
`Cell.point_placements`、discretization 和 initialization 始终读取所有 layers 的合并结果，
并在 inspection rows 中保留 owner/contact provenance。多个 generated rows 即使映射到相同
branch/x、CV 或 electrical `point_id`，也必须建立独立 point-mechanism instances 和 packed
state rows。mechanism/display names 可用 contact ID 作 debug suffix，但 name collision resolution 不参与
owner/contact lookup。

该 bridge 不允许逐 contact Python `place()` 成为正式实现。具体类型由
[I-06](./issues.md#i-06-batch-placement-lowering) 决定，identity mapping 必须遵守已锁定的
[I-09](./issues.md#i-09-generated-instance-identity) contract。

## 5. Event detection 与 delay delivery

### 5.1 Projection-specific detector

当前 runtime 从 `Cell.spike` 读取 source CV；目标行为是每个 Projection 独立维护
`previous_v`，按 `previous_v < threshold <= current_v` 生成 events。不同 source location
或 threshold 必须能够同时存在。detector deduplication 只能是保持表语义不变的优化。

### 5.2 Delay lowering

```text
ContactTable.delay (physical Quantity)
  -> lower with run dt
  -> integer delay_steps
  -> split/group DeliveryBlock
  -> enqueue into fixed-depth ring buffer
  -> write due arrivals into target packed layout
```

public mode 固定为 `nearest`、`ceil`、`strict` 和 `floor`。v1 默认 `nearest`，与 NEURON
fixed-step 的最近网格语义一致，恰好半步时进入较晚边界；`ceil` 保证事件不早于用户请求的
physical delay；`strict` 要求 delay 是 dt 的整数倍；`floor` 显式允许向下量化。四种模式下
`0 ms` 都在下一 solver step 到达，不在 detector step 内重入。

delivery operator 保留两个等价 backend：JAX `scatter_add` 与可用时的
`brainevent.coomv`。backend 只影响计算路径，不改变 contact、weight、delay 或加和语义。

event lowering 只能为声明 event port 的 target 建立 delivery block。scalar port 将
`ContactTable.weight` 作为 payload data；trigger port 只传递 event occurrence，不创建虚假的
`weight=1` column。没有 event port 的 target 在 static materialization 阶段报错，而不是等到
JIT/runtime 才失败。kinetic mechanism 可以在未来通过自己的 event bridge 消费同一 scalar
payload；连续耦合必须走不同 input-port lowering。

### 5.3 当前实现差距

- 当前 `Network.run()` 在每次调用前清空 `DeliveryState`，会丢弃跨 run 边界的未到期事件；
  目标实现必须让 queue state 随 INITIALIZED Network 延续，只有 reset 清空。
- 当前按每个 unique `delay_steps` 创建完整 target-layout ring buffer。在 delay 高度异质时，
  内存可能按 delay group 数和最大 delay 放大。v1 可先复用该实现，但不能在 benchmark 前
  宣称 delay queue 始终严格 `O(C)`；优化方向由 [I-11](./issues.md#i-11-heterogeneous-delay-buffer-layout)
  决定。

## 6. 现有 `braincell/network` 融合矩阵

| Current module | Existing objects | Target treatment |
| --- | --- | --- |
| `core.py` | `Population`, `Connection`, `NetworkRunResult` | 保留 `NetworkRunResult`；旧 wrapper/table 由 specs 和 normalized tables 替换，迁移期可作为 lowering adapter |
| `edges.py` | `EdgeSet`, `EdgeMethod`, `pairs/dense/all_pairs/probability` | common sparse algorithms 迁入 PairRule helpers；删除 dense public path；迁移完成后删除 module 或仅保留 private compatibility adapter |
| `projections.py` | `Projection`, contexts, old `ContactTable`, `per_edge/by_post` | Projection 概念保留并升级为完整 source-to-target spec；旧 pool target sampling 不作为新主线；ContactTable 替换为 normalized schema |
| `pools.py` | `SynapsePool`, `RandomLocations`, `SynapseInstanceTable` | 连续采样和 instance provenance 可复用；“先建 pool 再抽 target”转为 legacy adapter，最终由 direct contact placement 取代 |
| `lowering.py` | layout/source CV resolution, weight/delay lowering, `ConnectionBlock` | 重点保留；输入改为 resolved Projection/ContactTable 和 instance mapping，继续生成 runtime blocks |
| `delivery.py` | `DeliveryBlock/State`, ring buffers, scatter/brainevent ops | 重点保留；加入 projection detector input，修复跨 run queue ownership，并按 I-11 优化布局 |
| `engine.py` | `Network`, build/init/run/reset, caches, JIT loop | 删除 public build phase；重构 lifecycle/cache ownership，run loop、setup cache 和 Cell stepping 尽量复用 |
| `__init__.py` | legacy public exports | 开发期并存；examples/tests 迁移后一次性切换 exports，不长期维护两套 API |
| `runtime_test.py` | lowering/delay/backend/run regressions | 保留为数值基线，并补 detector、continued-run 和 heterogeneous delay tests |
| `topology_test.py` | EdgeSet/pool/contact-method tests | 将有效 invariant 迁移为 PairTable/placement tests，删除仅验证旧 API 的 cases |
| package `README.md` | 当前公开接口说明 | API 实现落地后按 `docs/design/network/api.md` 重写 |

删除旧 public class 不等于丢弃其算法。迁移必须先建立新 table 到现有 lowering/delivery
的 adapter，并用数值回归证明一致，再移除旧出口。

## 7. 目标模块责任

| Module | Responsibility |
| --- | --- |
| `braincell/filter/cell.py` | population filter protocol、selector、bool-mask validation |
| `braincell/network/specs.py` | population/projection/synapse specs、Rotation、rule protocols |
| `braincell/network/contexts.py` | NetworkContext、typed result views、RNG derivation |
| `braincell/network/tables.py` | PopulationInstances、Pair/Contact tables 和 logical views |
| `braincell/network/connectivity.py` | PairRule validation 和 direct common sparse rules |
| `braincell/network/placement.py` | Region/Locset samplers、measure integration、location validation |
| `braincell/network/builder.py` | registries、versioned lazy materialization 和 dependency invalidation |
| `braincell/network/lowering.py` | contacts 到 point declarations、detectors 和 runtime blocks |
| `braincell/network/delivery.py` | detector event delivery、delay queues 和 sparse backend ops |
| `braincell/network/engine.py` | init/run/reset/deinit、dt-specific setup、compiled loop 和 results |

Cell-side 只增加 batch declaration/lowering 所需能力；packed layout、state storage 和 current
scatter 应继续复用现有实现。

## 8. 性能不变量

- materialized contact metadata、mechanism parameters 和 synapse states 按 C 稀疏保存；
- 不分配 `N_cell * max_synapses_per_cell` padding；
- 不按 unique parameter rows 复制 layouts；
- common PairRules 避免不必要的完整 Cartesian intermediates；
- `all_to_all` generation peak memory 为 `O(P)`；fixed-degree generation 为 `O(P)`；
- probability baseline 的 candidate workspace 有界，不随完整 `S*T` 同时驻留；
- PairRule peak-memory benchmark 分别报告 candidate count `S*T`、materialized `P` 和
  generation workspace，不能只报告最终 table bytes；
- batch placement 必须显著快于 C 次 Python `place()`；
- packed current scatter 在异质 indegree 下保持稳定；
- delay queue memory 单独测量，不用 contact storage 的 `O(C)` 结论掩盖 queue amplification。
