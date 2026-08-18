# Network Builder 用户 API 规范

本文定义用户如何声明、检查、初始化和运行 BrainCell network。working names 由
[I-01](./issues.md#i-01-naming-and-api-vocabulary) 集中管理；本文中的行为、shape 和表语义
是正式规范。

## 1. Network

```python
net = braincell.network.Network(
    name="microcircuit",
    propagation_velocity=500.0 * u.um / u.ms,
    size_scale=1.0,
    weight_scale=1.0,
    delay_scale=1.0,
    seed=0,
)
```

| Field | Contract | Default | Behavior |
| --- | --- | --- | --- |
| `name` | non-empty `str` | required | network identity and diagnostics |
| `propagation_velocity` | positive velocity Quantity or `None` | `500 um/ms` | read-only rule context data |
| `size_scale` | finite scalar | `1.0` | only applies when a rule reads it |
| `weight_scale` | finite scalar | `1.0` | only applies when a rule reads it |
| `delay_scale` | finite scalar | `1.0` | only applies when a rule reads it |
| `seed` | integer | `0` | root of every framework-managed materialization stream |

Network 不自动缩放 number、weight 或 delay。例如：

```python
number=lambda ctx: round(base_number * ctx.network.size_scale)
weight=lambda ctx: base_weight * ctx.network.weight_scale
delay=lambda ctx: ctx.current.contacts.soma_distance / ctx.network.propagation_velocity
```

如果 rule 不读取对应字段，该字段不会产生隐藏行为。Network 也不提供 threshold、weight、
delay 或 placement sampler 的全局 fallback；这些默认值由其实际 owner 定义。

## 2. Population

### 2.1 声明

```python
def make_pc(*, pop_size, context):
    return make_multicompartment_pc(pop_size=pop_size)

pc = net.add_population(
    name="pc",
    number=4,
    cell_factory=make_pc,
    position=grid_xyz(4) * u.um,
    rotation=braincell.network.Rotation.from_axis_angle(
        axis=(0.0, 0.0, 1.0),
        angle=15.0 * u.degree,
    ),
    properties={
        "layer": "purkinje_layer",
        "cell_class": "inhibitory",
    },
)
```

Conceptual contract：

```python
@dataclass(frozen=True)
class PopulationSpec:
    name: str
    number: int | PopulationNumberRule | None = None
    cell: Cell | None = None
    cell_factory: CellFactory | None = None
    position: PositionValue | PopulationRule | None = None
    rotation: Rotation | PopulationRule | None = None
    properties: Mapping[str, PropertyValue | PopulationRule] = field(default_factory=dict)
    spatial_anchor: SingleLocationExpr = RootLocation(0.5)
```

`cell` 与 `cell_factory` 恰好提供一个：

- `cell=` 接管一个未初始化、`pop_size` 为一维的 batched Cell。`number` 省略时由
  `cell.pop_size[0]` 推导；显式提供时只能用于相等性验证。一个 Cell 不能注册到两个
  populations，也不能脱离其 `pop_size` 单独 resize。
- `cell_factory=` 要求 `number`，factory contract 固定为：

```python
cell_factory(*, pop_size=(number,), context=ctx) -> uninitialized Cell
```

factory 可以通过 `context.current.instances` 显式读取本 population 已解析的
position、rotation 和 properties，例如用 world position 生成异质 channel parameters。
只有实际读取的 fields 记录为 factory dependencies；它们改变时才重建缓存
Cell。返回 Cell 必须满足 `cell.pop_size == (number,)` 且尚未执行
`init_state()`。`cell=` 路径没有 creation-time context。

Population position/rotation 是 Network 拥有的 world metadata，不会自动传入 Cell 或改写
Cell morphology coordinates。factory 即使读取这些 metadata，也必须返回使用共享
morphology-local coordinates 的 Cell，避免 Network 在后续 world transform 时重复平移/
旋转。

一个 population 的 morphology、CV policy/count、paint/channel layout、solver 和 mechanism
tree 必须结构同构。parameter values、runtime state 和 point-mechanism placements 可以逐 cell
异质。需要在部分 cells 上“关闭”同一个结构机制时使用零效应参数或 mask，不能逐 cell 改变
paint/morphology 结构。

### 2.2 Position、rotation 与 properties

- `position=None` 表示没有 world coordinates；任何 world-distance rule 必须明确报错。
- `(3,)` length Quantity 广播为 `(N, 3)`；逐 cell 输入必须严格为 `(N, 3)`。
- `Rotation(quaternion)` 接受 `(4,)` 或 `(N,4)`，省略 rotation 时内部使用
  `Rotation([1, 0, 0, 0])`。
- axis-angle 输入使用 `Rotation.from_axis_angle(axis, angle)`。
- axis 必须 finite、non-zero；angle 必须是 angle Quantity。
- quaternion 顺序固定为 `(w, x, y, z)`，每行必须 finite、non-zero；解析时归一化并将
  `q/-q` 规范为唯一符号。
- `spatial_anchor` 默认为 `RootLocation(0.5)`，必须在 Cell morphology 上解析为唯一
  continuous location。
- `position[i]` 是第 i 个 cell 的 anchor world coordinate；rotation 表示 morphology
  local 到 world 的主动旋转。对任意 local point：

```text
world_point = position[i] + rotation[i] @ (local_point - anchor_point)
```

因此 anchor 本身精确映射到 `position[i]`。`position=None` 时不定义该 world transform，
任何 world-distance rule 都必须明确报错。

- `PopulationInstances.rotation` 是 canonical `Rotation`，底层 quaternion shape 为 `(N,4)`。
- property scalar 广播到 N；非标量值的 leading dimension 必须是 N。
- 常量 vector 与 per-cell axis 有歧义时使用显式 `broadcast(value)`，不依赖 shape 猜测。

### 2.3 解析顺序

```text
number -> position -> rotation -> properties
       -> candidate PopulationInstances
       -> cell_factory/cell validation + spatial_anchor resolution
       -> publish PopulationInstances + Cell
```

candidate table 只在当次 materialization transaction 中可见。factory 返回的 Cell、
`pop_size`、morphology 和 anchor 全部验证成功后，Network 才原子发布
`PopulationInstances + Cell`；失败不会让其他 objects 看到 partial result。

populations 按 `add_population()` 顺序解析，后一个 rule 可以读取前面已发布的 results，
但不允许反向或循环依赖。首次读取 `population.instances`、其他对象需要该 result，或调用
`init_state()` 时按需物化；inspection 不初始化 Cell，也不冻结 Network。lifecycle contract
见 [I-02](./issues.md#i-02-network-lifecycle-and-initialization-boundary)。

`Population` 是 Network-owned handle：

```python
pc.spec                 # immutable effective declaration
pc.instances[1]         # network metadata row
pc.cell[1]              # CellSelection over the batched Cell
```

不提供 `pc[1]`。`pc.instances` 的 canonical columns 为 `cell_id`、`position`、`rotation` 和
`properties`。

## 3. Rule 与 context

除 cell factory 外，用户规则统一为：

```python
rule(ctx) -> value
```

| Cardinality | Accepted result |
| --- | --- |
| Network/resolved scalar | scalar |
| Population filter | exact bool array `(N,)` |
| Per-population | scalar/intrinsic row broadcast or leading dimension `N` |
| Per-pair | scalar broadcast or leading dimension `P` |
| Per-contact | scalar broadcast or leading dimension `C` |
| Exact contact locations | `LocsetMask` with one or exactly `C` rows |

`net.context` 与传给 callable 的对象都是 `NetworkContext`：

```python
ctx.network
ctx.current
ctx.populations
ctx.projections
ctx.rng
```

普通 `net.context` 的 `current/rng` 为 `None`；rule evaluation 时填充。rule 内的
`ctx.current` 是 transaction-local progressive candidate view，只公开当前 stage 已经解析的
上游字段。population stages 依次开放 size/cell IDs、position、rotation 和 properties；
cell factory 可以读取完整 candidate `instances`。读取当前或未来 stage 尚未完成的字段报
forward-dependency error，不返回 placeholder。projection stages 也按其 materialization graph
逐步开放 pair/contact/location/parameter fields；其他 populations/projections 始终只暴露已
原子发布的结果。

`ctx.rng` 是 backend-neutral stateful facade，它的 sampling protocol 和 stream semantics
已锁定；I-10 仅保留底层 NumPy/JAX/BrainState adapter 选择。直接调用
`ctx.rng.uniform()/normal()/choice()` 消费当前 rule 的 automatic semantic stream。
对当前对象使用：

```python
ctx.current.size
ctx.current.contacts.size
ctx.current.source
ctx.current.target
ctx.current.contacts.source.position
ctx.current.contacts.target["layer"]
```

`PopulationResultView` 只暴露 `name/size/instances`；`ProjectionResultView` 只暴露
`name/pairs/contacts/parameters/source/target`。Context 不暴露 spec、Cell、runtime、
`set/add/remove/init/run`，也不允许跨 Network 自动依赖。population rules 可读取更早
populations；projection rules 可读取所有完成的 populations 和更早 projections。add order
是依赖方向，反向或循环读取报错。

Contact view 是逻辑完整视图：即使物理 C rows 只保存 `pair_id`，也提供 gathered
`source_id/target_id`、endpoint positions 和 properties。`net.context` 的 repr 只显示
ready/stale/unmaterialized 摘要，不因打印而全量物化。

automatic stream 按
`(Network.seed, "auto", object kind/name, materialization stage, rule slot)` 稳定派生。需要单独
控制某个 rule 或让多个 rules 共用相同原始随机序列时，使用：

```python
def position_rule(ctx, seed=7):
    rng = ctx.rng.with_seed(seed)
    return rng.uniform(
        -100.0,
        100.0,
        size=(ctx.current.size, 3),
    ) * u.um
```

`with_seed(stream_id)` 中的 integer 是 Network-rooted stream ID，不是绕过
`Network.seed` 的绝对 seed。改变 `Network.seed` 会改变全部 managed streams；只改变
`stream_id` 则只影响使用该 ID 的 rules。built-in stochastic rules 的
`seed=None` 选择 automatic stream，显式 integer 选择对应 user stream。

不同 rules 使用同一 stream ID 时，各自创建独立 handle 并从同一初始状态
开始；只有 sampling methods、arguments、shapes 和调用顺序都一致时，才产生
逐元素相同样本。同一 evaluation 内重复调用 `with_seed(7)` 取得同一局部
handle，后续 sampling 继续消费该序列，不从头重放。

Spec/repr 对 built-in rules 显示 automatic 或 explicit stream ID。resolved diagnostics 记录
semantic path、root 来源和不可逆 key fingerprint；custom callable 内实际调用过的
`with_seed()` 在物化后记录，Network 不从 Python 签名静态猜测。custom callable
自行创建 external RNG 时由用户负责 reproducibility 和 cache consistency，Network
不临时重设全局 seeds。完整 contract 见
[I-10](./issues.md#i-10-materialization-time-rng-contract)。

例如下面的 callable 故意脱离 Network random hierarchy：

```python
def externally_seeded_property(ctx):
    rng = brainstate.random.RandomState(7)
    return rng.uniform(size=ctx.current.size)
```

改变 `Network.seed` 不会改变它，Network 也不记录或恢复该 RNG 的消费状态。

## 4. Population endpoint filtering

```python
def select_excitatory(ctx):
    return ctx.current.instances["cell_class"] == "excitatory"

source = braincell.filter.cells(
    population="grc",
    where=select_excitatory,
)
target = "pc"
```

`source="grc"` 与 `cells(population="grc")` 等价。general protocol 是：

```python
PopulationFilterRule(ctx) -> bool_array  # exact shape (N,)
```

filter 返回值不接受 scalar、integer IDs 或整数 `0/1`。解析结果是 immutable
`PopulationView`，保留原始 population-local `cell_id`、稳定升序且不重新编号。它只用于
network endpoint inspection/connectivity，不提供 paint/place/runtime。空 selection
合法，并产生 size 0 的 view。

## 5. Projection 与 PairRule

### 5.1 Projection 声明

```python
net.add_projection(
    name="grc_to_pc",
    source=source,
    target=target,
    pair_rule=braincell.network.fixed_indegree(
        number=2,
        nsyn=2,
        replace=False,
        seed=1,
    ),
    source_loc=braincell.filter.RootLocation(0.5),
    source_threshold=-10.0 * u.mV,
    target_loc=braincell.network.sample(
        dendrite_region,
        measure="length",
        method="random",
        sampling_unit="cell_pair",
        seed=2,
    ),
    weight=0.1 * u.uS,
    delay=0.0 * u.ms,
    synapse=braincell.network.SynapseSpec(
        model="Exp2Syn",
        parameters={
            "tau1": 0.5 * u.ms,
            "tau2": 5.0 * u.ms,
            "e": 0.0 * u.mV,
        },
    ),
)
```

Conceptual contract：

```python
@dataclass(frozen=True)
class ProjectionSpec:
    name: str
    source: str | Population | PopulationSelector
    target: str | Population | PopulationSelector
    pair_rule: PairRule
    synapse: SynapseSpec
    source_loc: SingleLocationExpr = RootLocation(0.5)
    source_threshold: ScalarOrRule = -10.0 * u.mV
    target_loc: TargetLocationSpec | PlacementRule = RootLocation(0.5)
    weight: ContactValueOrRule | None = None
    delay: ContactValueOrRule = 0.0 * u.ms
```

一个 Projection 固定一种 postsynaptic mechanism model。需要不同 mechanism classes 时
声明不同 Projections。`net.add_projection(...)` 返回 Network-owned `Projection` handle；
声明、物化结果和参数分别通过 `.spec/.pairs/.contacts/.parameters` 查询，不再创建
`ResolvedProjection`。

### 5.2 PairRule protocol

```python
PairRule(ctx) -> integer_array  # shape (P, 3)
# columns: source_id, target_id, nsyn
```

这里的 source/target universe 是 selector 已解析的 `PopulationView`，其 sizes 记为 `S/T`。
返回 IDs 仍是原 population-local IDs，不因 selector 重新编号。

共享 validator 必须：

- 要求严格二维 `(P,3)`，空结果为 `(0,3)`；
- dtype 必须是 integer，拒绝 bool、float 和 object；
- 验证 source/target IDs 属于 resolved endpoint views；
- 要求 `nsyn >= 1`，zero/negative rows 直接报错；
- 拒绝重复 `(source_id, target_id)`，不自动求和；
- 按 source/target lexicographic order 排序并分配确定性、snapshot-local dense `pair_id`。
- 规范化结果使用 immutable canonical dtype。

常用 `all_to_all()`、`probability()`、`fixed_indegree()`、
`fixed_outdegree()` 和 `explicit_pairs()` 都直接实现相同 protocol，不公开额外的
candidate/filter/score pipeline。custom rule 也必须一次返回标准 array；dense adjacency、
CSR/CSC、generator、iterable 和 lazy probability declaration 都不是 v1 PairRule outputs。
`probability/fixed_indegree/fixed_outdegree` 的 `seed: int | None = None`；`None` 使用该
rule slot 的 automatic stream，integer 选择对应 Network-rooted user stream。

### 5.3 PairRule scalability 与 materialization

PairTable 是实际 topology，不是 rule 的压缩描述：

- subset-to-subset `all_to_all()` 物化 `S*T` 个 rows；该最终行数不可避免；
- `probability(p)` 物化实际成功的 independent Bernoulli rows，不保存 probability matrix；
- fixed-degree helpers 直接生成准确 degree，不先生成 Cartesian candidates；
- custom callable 可以内部 chunk 或使用 spatial index，但最终返回协议不变。

v1 PairRule 不提供 public chunk-size、streaming 或 out-of-core controls。built-in helpers 必须
限制无必要的 generation workspace；最终 `P` 行本身无法驻留 host memory 时属于 v1 明确
规模边界。未来 generation fast path 不得改变 shared validation、canonical ordering 或
stable identity。完整 contract 见 [I-04](./issues.md#i-04-pairrule-scalability)。

例如原始 rows：

```text
(source=1, target=0, nsyn=2)
(source=0, target=0, nsyn=3)
```

规范化后按 pair 顺序展开为 5 个 contacts，`synapse_index` 分别为 `0,1,2` 和 `0,1`。
所有 per-contact values 都按这个当前 canonical row 顺序对齐；持久 identity 由下文
`contact_id` 单独表达。

### 5.4 Contact identity 与 lookup

ContactTable 的 row axis 始终是当前存活 contacts 的紧凑顺序，但每行同时包含
projection-local、只读 `int64 contact_id`。`contact_id` 在 Projection 生命周期内单调
分配，删除留空洞且不复用。

canonical stored columns 是 `contact_id`、`pair_id`、`synapse_index`、`target_branch_id`、
`target_branch_x`、`weight`（weighted target）和 `delay`；mechanism parameter columns 由
同一 ContactTable backing storage 管理。`source_id/target_id` 通过 `pair_id` 作为 logical
gathered columns 暴露，不在每个 contact row 重复保存。

例如：

```text
initial contact_id column:       [0, 1, 2]
after retiring contact 1:        [0, 2]
after creating one new contact:  [0, 2, 3]
current row index:               [0, 1, 2]
```

`contacts[...]` 始终按当前 row index、slice 或 mask 选择；稳定 ID 查询使用
`by_id(...)`：

```python
contacts = net.projections["grc_to_pc"].contacts

contacts[1]           # current row 1; its contact_id may be 2
contacts.by_id(2)     # the contact whose stable ID is 2
contacts.by_id([3, 0])  # ordered lookup; result order is [3, 0]
```

scalar lookup 返回一行 view；integer sequence 返回按请求 ID 顺序排列的 view，并像普通
row selection 一样保留重复请求。bool 或 non-integer ID 报输入错误；退役、未分配或
不存在的 ID 报 `KeyError`。`by_id(...)` 由当前 Projection 的 ContactTable 定位 namespace，
不同 Projections 中的相同整数 ID 没有关联。

rematerialization 按 `(source_id, target_id, synapse_index)` 识别仍存活 contact。修改 weight、
delay、target location 或 mechanism spec 不改变其 ID。若 key 消失，其 ID 永久退役；
以后相同 key 再次出现也获得新 ID。例如 `nsyn: 3 -> 1 -> 3` 不会复活已删除
slots 的 contact IDs，前提是 `nsyn=1` 已经因 inspection 或 `init_state()` 成功物化为
一个 ContactTable snapshot。lazy materialization 前连续覆盖的声明不生成中间 contacts，因此
不分配或退役 IDs。

删除整个 Projection 会销毁它的 contact ID namespace、managed placements 和 mappings。
持有的旧 Projection/contact view 之后访问时报 `ReferenceError`。同名新 Projection 是新对象，
其 contact IDs 可以重新从 0 开始。`placement_id`、`point_id` 和 runtime layout row 只是
当前物化的 inspection fields，不得作为持久 contact identity。

## 6. Target placement

### 6.1 LocsetExpr 与 LocsetMask

`LocsetExpr` 是尚未绑定 morphology 的可复用声明。用户直接保存表达式：

```python
locations = braincell.filter.at("dend", 0.2)
```

不要求用户调用 `.evaluate(morpho)`。Network 在 target Population Cell 已可用后使用其共享
morphology 求值为 `LocsetMask`。LocsetMask 是 ordered multiset：默认保留行序和重复位置，
不排序、不去重；相同 branch/x rows 仍创建独立 mechanism instances。

表达式代数固定为：

```python
a + b        # ordered concatenation, duplicates retained
a | b        # stable unique union
a & b        # stable unique intersection
a - b        # stable unique difference
a.unique()   # stable explicit deduplication
```

`LocsetMask` 只公开 immediate `.unique()`。ordinary unique 只比较 canonical
`(branch_id, branch_x)`，不能按 XYZ、CV、electrical point 或 topology junction 合并。

### 6.2 target_loc 输入

| Input | Behavior |
| --- | --- |
| one-row `LocsetExpr` / `LocsetMask` | 广播到所有 contacts |
| C-row `LocsetExpr` / `LocsetMask` | 按 canonical contact order 精确一一对应 |
| other exact row count | cardinality error |
| `sample(RegionExpr, ...)` | 显式 continuous sampling |
| `sample(LocsetExpr, ...)` | 显式 finite candidate sampling |
| callable | 返回 `LocsetExpr` 或 `LocsetMask`，再应用同一 cardinality contract |

默认 `target_loc=RootLocation(0.5)`，因此每个 contact 都在 root midpoint 创建独立
instance。direct 多行 locset 永远不隐式解释为 candidate pool。

### 6.3 Region sampling

```python
braincell.network.sample(
    dendrite_region,
    measure="length",       # or "area"
    density=None,           # uniform; or a built-in DensityExpr
    method="random",        # or "stratified"
    sampling_unit="cell_pair",  # or "target_cell"
    seed=None,               # automatic stream; or an explicit stream ID
)
```

默认始终是 `length + uniform density + random + cell_pair`，不会根据 `nsyn` 自动
改变。`density=None` 与显式 `braincell.network.uniform_density()` 等价。area sampling 对
tapered cable 使用 frustum lateral area，不包含 segment end caps。采样先产生连续
branch/x，再 lowering 到 CV，避免 morphology boundary provenance 被 CV 划分覆盖。

v1 的 weighted continuous sampling 只公开内置 density expressions。例如按到 soma midpoint 的
morphology path distance 指数衰减：

```python
braincell.network.sample(
    dendrite_region,
    measure="length",
    density=braincell.network.exponential_tree_distance(
        origin=braincell.filter.at("soma", 0.5),
        length_constant=200.0 * u.um,
    ),
)
```

或者按膜面积采样，同时偏好距 soma `300 um` 附近的位置：

```python
braincell.network.sample(
    dendrite_region,
    measure="area",
    density=braincell.network.gaussian_tree_distance(
        origin=braincell.filter.at("soma", 0.5),
        center=300.0 * u.um,
        sigma=50.0 * u.um,
    ),
)
```

两个 profile 的定义为：

```text
exponential: exp(-tree_distance(origin, x) / length_constant)
gaussian:    exp(-0.5 * ((tree_distance(origin, x) - center) / sigma) ** 2)
```

`origin` 必须解析为唯一 continuous location。`length_constant` 和 `sigma` 必须大于零，
`center` 必须非负，且所有 distance arguments 必须携带 length unit。density 是无量纲、
非负的相对权重；负值、非有限值或请求 contacts 时零总质量均报错。

权重的完整语义由 density 和 measure 共同决定：

```text
measure="length": p(d location) proportional to density(location) * ds
measure="area":   p(d location) proportional to density(location) * dA
```

因此 distance-based density 并不自动意味着按 length 采样。距离范围继续通过 `RegionExpr`
缩小 `dendrite_region`，而不是 density 的额外 window 参数。density 只分配已确定 contacts
的位置，不改变 PairRule 产生的 pair、`nsyn` 或 contact 数量。

`sampling_unit="cell_pair"` 对每个 cell pair 独立采样；`"target_cell"` 汇总同一 target
cell 的 incoming contacts。二者对 i.i.d. continuous sampling 分布相同，但 finite locset
without replacement、stratified 和联合避让行为不同。

finite Locset sampling 默认在每个 sampling unit 内 random without replacement；候选不足时报错。
候选身份是 row identity：候选 Locset 自身包含两个同坐标 rows 时，`replace=False` 仍可分别
抽中两行。显式 `replace=True` 还允许重复抽取同一候选 row；contacts 始终是独立实例。

任意 Python callable、自适应积分/公开精度参数、CV-specific density 和可微采样均不属于
v1，详见 [I-05](./issues.md#i-05-weighted-continuous-region-sampling)。`delay` 仍专指传导
延迟，不用作 density 的指数尺度参数名。

## 7. Synapse parameters

```python
def correlated_taus(ctx):
    tau1 = ctx.rng.uniform(0.2, 0.8, size=ctx.current.contacts.size) * u.ms
    return {"tau1": tau1, "tau2": 8.0 * tau1 + 1.0 * u.ms}

synapse = braincell.network.SynapseSpec(
    model="Exp2Syn",
    parameters={"e": 0.0 * u.mV},
    parameter_rule=correlated_taus,
    default_weight=0.1 * u.uS,
)
```

`parameters` 与 `parameter_rule` 产生同名 key 时直接报错。每个参数必须是 scalar 或
leading dimension C，并通过 mechanism registry 的 name、unit 和 shape validation。scalar
声明在发布前广播为 leading dimension C 的 canonical columns。contact-aligned mechanism
parameters 由 ContactTable 作为唯一 backing storage；`Projection.parameters` 是这些列的
typed view/alias，沿用相同 current-row order、selection 和 `by_id(...)`，不复制数据或维护
第二套 identity。

v1 中 pair topology、target locations 和 contact count 属于 static、不可训练结构。weight
和 mechanism parameter 也不会仅因是 array 就自动成为 trainable；只有被显式 trainable
schema 标记并由训练接口选择的参数才进入 optimizer tree。具体 optimizer/checkpoint API 不由
本规范定义。该边界见
[I-08](./issues.md#i-08-trainable-versus-non-trainable-fields)。

以下概念分别拥有独立 owner：

- `source_threshold`：presynaptic detector property；
- `delay`：Projection event-delivery property，默认 `0 ms`；
- `weight`：per-contact event payload；
- `tau1/tau2/e/...`：postsynaptic mechanism parameters。

### 7.1 Weight 与 event input

`weight` 的实际 owner 是 contact。`SynapseSpec.default_weight` 是 Projection 未声明
`weight` 时的直接 fallback，不属于 target placement，也不属于 `parameters`；materialization
后每条 weighted contact 都有 canonical effective weight：

```text
Projection.weight
  -> otherwise SynapseSpec.default_weight
  -> otherwise error
  -> validate against target event input contract
  -> ContactTable.weight
```

标准 `ExpSyn/Exp2Syn` 消费 scalar conductance，canonical unit 为 `uS`。physical weight 必须
显式携带单位，不对裸数隐式补 `uS`；兼容量纲允许换算。所有 scalar、C-row array 和 callable
返回值使用同一 validator：

```python
weight=0.1 * u.uS       # valid
weight=100.0 * u.nS     # valid; canonicalized to 0.1 uS
weight=-0.1 * u.uS      # valid; sign semantics belong to the model/training rule
weight=0.1 * u.nA       # error: Exp2Syn does not consume current
weight=0.1              # error: physical unit is missing
```

负 conductance 不等同于常规抑制性突触；API 保持 NEURON-like signed freedom，只作语义提醒，
不增加 sign validation。常规抑制使用正 conductance weight 和适当的 reversal potential。

target registry contract 还区分以下情况：

- scalar weighted event：Projection 提供或解析出一个有限、unit-compatible scalar；
- trigger-only event：`weight` 必须保持 `None`；
- no event port：不能作为本版 spike-event Projection target。

事件到达后的加法、覆盖、reset、随机释放、饱和或 kinetic state transition 都由 mechanism
实现；Projection 不推断公式。v1 不接受 vector/multi-field event payload。当前连续读取
`pre_drive()` 的 `AMPA/GABAa/NMDA` 需要后续 event bridge，边界见
[I-13](./issues.md#i-13-kinetic-and-continuous-synapse-input-protocols)。完整 weight contract 见
[I-03](./issues.md#i-03-weight-ownership-and-defaults)。

## 8. Presynaptic event source

`source_loc` 与 `source_threshold` 共同定义 NetCon-like event detector：

- source location 默认 `RootLocation(0.5)`，并且必须解析为恰好一个 continuous location；
- source variable 首版只支持 membrane voltage `v`；
- threshold 是 voltage scalar 或返回 scalar 的 callable，Projection 默认 `-10 mV`；
- event 是 `previous_v < threshold <= current_v` 的 positive crossing；
- detector threshold 与 `Cell.V_th` 的 spike/reporting threshold 相互独立；
- 不同 Projections 首版保存独立 detector，即使 location/threshold 相同也不合并。

## 9. Inspection、initialization 与 run

### 9.1 EDITABLE phase

新 Network 从 `EDITABLE` 开始。Population、Projection 和 Cell 的静态派生结果通过属性
按需物化并缓存，不需要公开 `build()` 或 `resolve_*()`：

```python
print(net.populations["pc"].instances)
print(net.populations["pc"].cell.cvs)
print(net.projections["grc_to_pc"].pairs)
print(net.projections["grc_to_pc"].contacts)
print(net.populations["pc"].cell.point_placements)
```

inspection 不冻结 Network。`set(...)`、add/remove 或 Cell declaration mutation 会按
[I-12](./issues.md#i-12-model-mutation-ownership-and-cache-invalidation) 标记受影响的静态
结果为 stale；下一次属性访问或 `init_state()` 时刷新。

### 9.2 Declaration 与 materialized table mutation

handle 与 row view 都使用 `set(...)`，但修改层级不同：

```python
proj.set(weight=weight_rule)                         # declaration; survives rematerialization
proj.contacts.by_id([2, 7]).set(weight=0.2 * u.uS)  # current table snapshot only
pc.instances[mask].set(position=new_positions)       # selected population rows
```

PopulationInstances 可修改 position、rotation 和 properties；contact-aligned writable values
包括 target location、weight、delay，以及由 `Projection.parameters` 暴露的 mechanism
parameter columns。cell/source/target ID、`pair_id`、
`contact_id`、`synapse_index` 和 `nsyn` 是只读 identity/topology fields；pair topology 只能通过
Projection 的 PairRule declaration 修改。

`Projection.parameters` 与 ContactTable 使用相同 current-row selection，并提供相同的
`by_id(...)` stable-contact lookup；参数列不能维护另一套 row order。通过现有 Cell API 修改
Network-owned Cell 的静态 channel/mechanism parameter 时，该值同样作为 dependency producer
进入本节 transaction，而不是绕过 Network cache graph。

manual placement 与每个 Projection 拥有的 managed placement layer 同时存在。替换或删除
Projection 只撤销其 owner layer；删除仍被引用的 Population 默认报错，只有显式
`cascade=True` 才同时删除 dependent Projections。删除后旧 Projection/contact views 报
`ReferenceError`，同名新 Projection 不继承旧 owner 或 contact IDs。

table edit 会立即成为该 field 的当前 canonical value，并使实际读取它的下游 cache stale；它
不是永久 row override。如果这个 stage 的已记录上游依赖随后改变，下一次 inspection 会从
declaration/rule 重建整个 stage，table edit 随旧 snapshot 消失。需要跨刷新保留的修改必须
写入 declaration/rule；v1 不提供 override registry 或 `clear_override()`。

在 `EDITABLE` 中，刷新保持 lazy。在 `INITIALIZED` 中，`set(...)` 先验证修改的完整传递依赖
闭包，只有 shape、dtype、runtime layout、mechanism schema 和 runtime signature 全部不变才会
原子提交。普通 weight/mechanism parameter 更新通常允许；delay 还必须保持量化 delay groups
和 event-buffer layout 不变。target location、PairRule、`nsyn`、mechanism type 或任何会经
下游 rule 改变结构/layout 的修改报 phase error，并要求先 `deinit_state()`。

mutation 验证失败时不提交输入版本，原模型仍有效；已经 stale 的 lazy refresh 若失败，则保留
旧 snapshot 但继续标记 stale，不能据此 `init_state()`。两者都不会发布部分更新。
`.set(...)` 不修改 voltage、gating variable、synaptic state 或 pending events；未来
runtime-state 接口边界见
[I-14](./issues.md#i-14-runtime-state-mutation-api)。

### 9.3 Initialization boundary

```python
net.init_state()
```

`init_state()` 补齐所有 stale materialization，冻结 contact 数量、locations、mechanism
types、placements、runtime shapes 和 indices，然后统一初始化 Population Cells、event
detectors 和 delivery state。它是显式且一次性的：

- 已 `INITIALIZED` 时再次调用 `init_state()` 报错；
- Network-owned Population Cell 不能单独调用 `cell.init_state()`；
- 任一 Population 初始化失败时撤销部分 runtime，Network 返回 `EDITABLE`；
- 未初始化时调用 `run()`、`reset_state()` 或 `deinit_state()` 报错。

### 9.4 Run episodes

```python
first = net.run(
    dt=0.025 * u.ms,
    duration=50.0 * u.ms,
    delay_quantization="nearest",
)
second = net.run(
    dt=0.025 * u.ms,
    duration=50.0 * u.ms,
    delay_quantization="nearest",
)
```

连续 `run()` 延续 current time、Cell/mechanism state、detector previous voltage 和尚未到期的
events。`delay_quantization` 公开 `"nearest"`、`"ceil"`、`"strict"` 和 `"floor"`，默认
`"nearest"`，与 NEURON fixed-step 在 `t + dt / 2` 前交付事件的规则一致；量化误差不超过
半个 `dt`。`"ceil"` 保证 delivery 不早于请求 delay，`"strict"` 要求 delay 是 `dt` 的
整数倍，`"floor"` 显式向下量化。四种模式下 zero delay 都在 detector crossing 后的下一
solver step delivery，不在当前 step 重入。

一个 episode 在第一次 `run()` 时同时绑定 `dt` 和 `delay_quantization`；time 已推进后使用不同
值报错。
`duration` 可以变化，并可能选择或创建不同 step-count 的 compiled loop cache。固定步长量化
与 zero-delay phase 见 [I-07](./issues.md#i-07-delay-runtime-semantics)。

### 9.5 Reset 与返回编辑态

```python
net.reset_state()
```

`reset_state()` 保持 `INITIALIZED`，把 voltage、mechanism dynamic state、current time、
detector history 和 event queues 恢复到刚初始化时的状态。它保留 Pair/Contact tables、
placements、当前模型参数和 compiled caches，并解除当前 episode 的 `dt` 与
`delay_quantization` 绑定；下一次 `run()` 可以选择新的 `dt` 和 `delay_quantization`。

```python
net.deinit_state()
net.projections["grc_to_pc"].set(target_loc=new_target_loc)
net.init_state()
```

`deinit_state()` 销毁 runtime objects、dynamic state、event queues 和 compiled caches，保留
specs、static tables、managed placements 和当前模型参数，然后返回 `EDITABLE`。结构修改
必须先进入该 phase；`deinit_state()` 不回滚训练或手动设置的 weight/conductance。

内部 tables 是 source of truth。joined contact view 可以提供 records 或 optional DataFrame
adapter，但核心 API 不依赖 pandas，也不在每行保存 Python `Synapse` object。
