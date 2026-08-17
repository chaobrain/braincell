# BrainCell Network Builder 实施计划

本文是工程进度和验收表。正式 API 与内部行为分别以 [api.md](./api.md) 和
[architecture.md](./architecture.md) 为准；实现过程中发现规范缺口时，先更新
[issues.md](./issues.md)，不能在本文件中私自定义语义。

## 1. 状态规则

- `[ ]` 未开始或缺少验收证据；
- `[~]` 正在实施，尚未通过 gate；
- `[x]` 产物和对应 tests/benchmarks 均已完成；
- 每个阶段只有在所有 gate 都有可重复证据时才能标记完成。

当前总体状态：设计已收敛为 population-first、sparse Pair/Contact、direct placement 和
existing packed runtime integration；I-01/I-06/I-10/I-11 仍需按各自
deadline 关闭。

## 2. 现有实现迁移表

| Area | Current source | Action | Completion evidence |
| --- | --- | --- | --- |
| network declarations/results | `core.py` | 落地 Population/Projection handles、specs 和 normalized results；保留 run result type | 新 API tests 不依赖旧 wrappers |
| topology | `edges.py` | sparse algorithms 迁入 PairRule；移除 dense public path | PairRule unit/statistical tests |
| projection/contact | `projections.py` | 升级为完整 ProjectionSpec；old pool sampling 仅作迁移 adapter | normalized Pair/Contact table tests |
| placement pool | `pools.py` | 复用 continuous sampler/provenance；迁移到 direct contact placement | manual/network placement equivalence |
| runtime lowering | `lowering.py` | 接受 resolved contacts 和 instance map；保留 CV/layout/delay lowering | lowering and identity tests |
| event delivery | `delivery.py` | 保留 ring/scatter/brainevent；接入 Projection detector；修复 continued-run queue | phase/backend/continued-run tests |
| orchestration | `engine.py` | 实现 EDITABLE/INITIALIZED 与 lazy materialization；复用 setup cache、JIT loop、Cell stepping | lifecycle and end-to-end tests |
| exports | `__init__.py` | 开发期兼容，迁移完成后切换并删除 legacy exports | import surface test and repository search |
| docs/examples | package README/notebooks | 按新 API 迁移，旧示例不长期保留 | notebooks execute successfully |

迁移期间允许 private adapter，但必须满足：

1. 新 tables 是唯一 source of truth；
2. adapter 不改变 row identity/order；
3. 新 runtime tests 覆盖后才能删除旧 path；
4. 不新增第二套长期公开 DSL。

## 3. 分阶段实施

### Phase 0 - Platform 和当前实现 audit

- [x] 将 `LocsetMask` 改为 ordered duplicate-preserving rows，确认 deferred `LocsetExpr`
  algebra、continuous branch/x provenance、ForkPoints topology junction 和 branch boundary 行为。
- [x] 确认 independent point placement、packed population-specific layouts 和 current scatter。
- [x] 确认现有 lowering、ring buffer、scatter/brainevent backend 和 run-loop cache。
- [x] 记录当前跨 `run()` queue reset 与 heterogeneous-delay buffer amplification。

Gate：`architecture.md` 明确列出可复用能力、差距和现有文件迁移方向。

Evidence (2026-08-15)：`filter_locset_test.py`、`filter_vis_test.py` 和
`_discretization/build_test.py` 覆盖 ordered duplicates、column storage、ForkPoints 与
independent placement；完整 `pytest braincell/` 为 2173 passed、23 skipped。

### Phase 1 - 设计规范整理

- [x] 将用户 API 与内部 architecture 拆成同等正式规范。
- [x] 建立独立 Issue Register 和工程推进表。
- [x] 删除互相冲突的旧规范，只保留标记为 historical 的平台调研。
- [x] 关闭 I-02，冻结 explicit initialization、reset/deinit 和 continued-run lifecycle。
- [x] 关闭 I-03，冻结 weight ownership、fallback 和 target event-input validation。
- [x] 关闭 I-04，冻结 explicit sparse PairTable 与可替换的 bounded generation contract。
- [x] 关闭 I-05/I-09，冻结 weighted Region sampling 与 generated contact identity。
- [x] 关闭 I-12，冻结 declaration/table mutation、ownership、cache invalidation 和 runtime-signature gate。
- [ ] 关闭 I-01，冻结 public class/parameter names。

Gate：API signatures、table schemas 和 vocabulary 不再使用未标注的 working names。

### Phase 2 - Specs、contexts 和 population tables

- [ ] 实现 `Network` config、quaternion `Rotation`、Population handle 和 immutable spec。
- [ ] 实现 `cell=`/`cell_factory=` ownership、transaction-local candidate
  PopulationInstances、factory validation 和 atomic publish graph。
- [ ] 实现 position 是 spatial anchor world coordinate 的变换契约、默认
  `RootLocation(0.5)`、identity rotation 和 missing-world-position diagnostics；Cell 保持
  morphology-local coordinates。
- [ ] 实现 NetworkContext/result-view whitelist、progressive current view、future-field access
  rejection 和 rule cardinality；factory 实际 metadata field reads 进入 dependency graph。
- [ ] 按 I-10 已锁定协议实现
  stateful `ctx.rng` facade、automatic semantic path 和 `with_seed(stream_id)`。
- [ ] 实现 evaluation-local handle registry：同一 evaluation 内重复取得同一显式流时
  连续消费，不同 evaluations 从相同派生状态重建；记录 semantic path、root
  source 和不可逆 key fingerprint diagnostics。
- [ ] 用相同 workloads 比较 I-10 NumPy semantic adapter、JAX-backed stateful facade 和
  JAX key + BrainState RandomState adapter，记录 cold/warm time、peak memory、host
  conversion、precision 和 facade overhead。
- [ ] 在全部候选通过同一 stream-contract tests 后选择 sampling adapter；benchmark 不得
  改变 semantic derivation、`ctx.rng` interface 或 Network/stream-ID composition。
- [ ] 实现 `PopulationView`、`PopulationSelector` 和 `cells()` exact bool-mask protocol。
- [ ] 实现 ordered population lazy materialization、versioned cache 和 conservative invalidation。

Gate：I-10 backend comparison evidence 可重复，选定 adapter 通过已锁定 stream
contract 并关闭 issue；
declaration/context/table tests 在不修改 Cell runtime 的情况下通过。

### Phase 3 - Pair generation

- [ ] 实现 `(P,3)` integer PairRule protocol、PairTable validator、sorting 和 stable pair IDs。
- [ ] 实现 `explicit_pairs/all_to_all/probability/fixed_indegree/fixed_outdegree`。
- [ ] `all_to_all` 直接生成必要的 `O(P)` rows；fixed-degree 使用 direct sampling。
- [ ] `probability` 保持 exact Bernoulli，并使用逐 source 或 bounded chunk workspace；fast
  geometric skip 可以在不改变 RNG contract 的前提下后续加入。
- [ ] 实现 `nsyn` expansion 和 stable contact IDs。

Gate：empty selections、duplicates、degree constraints、determinism、subset cardinality 和
I-04 peak-memory tests 通过。

### Phase 4 - Placement 与 SynapseSpec

- [x] 实现 columnar ordered duplicate-preserving `LocsetMask` 和 deferred LocsetExpr algebra。
- [ ] 实现 exact one/C-row location、explicit Region/Locset sampling 和 custom placement inputs。
- [ ] 实现 length/area random、stratified、sampling unit 和 replacement constraints。
- [ ] 实现 uniform、exponential tree-distance 和 Gaussian tree-distance continuous density。
- [ ] 实现 fixed model、heterogeneous parameter columns 和 joint parameter rules。
- [ ] 实现 canonical ContactTable weight/delay/parameter columns，以及
  `Projection.parameters` 对同一 backing storage 的 typed view。
- [ ] 实现 `Projection.weight -> SynapseSpec.default_weight -> error`、delay 和
  contact-derived spatial rule resolution。

Gate：location/density statistical tests、CV-policy independence、branch boundary provenance、
parameter correlation 和 invalid shape/unit diagnostics 通过。

### Phase 5 - Batch point lowering

- [ ] 关闭 I-06，固定 columnar placement bridge，并实施 I-09 identity mapping。
- [ ] 将 C rows 按 Projection owner 批量附加到 uninitialized target Cells。
- [ ] 合并 manual placements 与 managed Projection placement layers。
- [ ] 支持 Projection layer 的原子替换和撤销。
- [ ] 实现 projection-local monotonic `contact_id` allocator、active-key registry 和 dense ContactTable rows。
- [ ] 建立 contact ID -> current row -> placement -> CV/point -> layout/state index mapping。
- [ ] 实现 `contacts.by_id(...)`、retired-ID diagnostics 和 deleted-handle `ReferenceError`。
- [ ] 绑定 vectorized mechanism parameter columns 到 packed state layouts。

Gate：middle deletion/new allocation、committed `nsyn` shrink/grow、canonical reorder/unrelated
Projection stability、value/location/model update preservation、colocated independent state、
manual/network voltage equivalence、dense storage without ID padding、stale handle、display-name
collision 和 ID-to-runtime lookup tests 全部通过。

### Phase 6 - Event detector 和 delay runtime

- [ ] 实现 Projection-specific source location/threshold positive crossing detector。
- [ ] 将 resolved contacts 接入现有 `ConnectionBlock/DeliveryBlock` lowering。
- [ ] 保留 scatter 与 brainevent backend numerical equivalence。
- [ ] 让 delayed events 跨连续 `run()` 保留，并在 `reset_state()` 清空。
- [ ] 实现公开 `ceil/strict/floor` quantization、默认 `ceil` 和三种模式一致的 next-step
  zero-delay phase。
- [ ] benchmark I-11 的 unique-delay memory scaling，并按结论调整 buffer layout。

Gate：I-07 三种 quantization/zero-delay phase semantics、multiple detector thresholds、
continued-run equivalence、reset 和 heterogeneous delay tests 全部通过；关闭 I-11 或明确 v1
规模限制。

### Phase 7 - Network lifecycle 与 inspection

- [ ] 按 I-02 实现 EDITABLE/init/INITIALIZED/reset/deinit state transitions。
- [ ] 实现 PairTable、ContactTable、instance mapping 和 lazy inspection properties。
- [ ] 实现 atomic initialization rollback 和 Network-owned Cell init guard。
- [ ] 按 I-12 实现 stage/field-level cache versions、dependency-selective invalidation 和 atomic refresh。
- [ ] 实现 table snapshot mutation，以及 `INITIALIZED` transitive runtime-signature validation。
- [ ] 定义 repeated init、pre-init run/reset/deinit 和 structural set phase errors。
- [ ] 绑定连续 episode 的 `dt` 和 delay quantization mode，reset 后允许重新选择二者。

Gate：lifecycle state tests、table inspection 和 continued-run tests 通过。

### Phase 8 - Migration 和公开切换

- [ ] 迁移 network package README、notebooks、examples 和 downstream tests。
- [ ] 比较 legacy 与新 runtime 在等价 topology/placement 下的数值结果。
- [ ] 切换 `braincell.network` exports，并删除旧 public EdgeSet/pool/contact-method path。
- [ ] 删除不再使用的 compatibility adapters 和旧 API 文档。

Gate：repository test suite、代表性 notebooks 和 performance suite 通过；repository search
不再存在未授权 legacy public usage。

## 4. 验证矩阵

### Declaration 和 context

- number scalar/callable、positive constraint、factory signature 和 uninitialized Cell；
- position Quantity、broadcast、missing-world-position diagnostics；
- axis-angle/quaternion conversion、normalization、unit 和 invalid input；
- default `RootLocation(0.5)` anchor、identity rotation，以及
  `position + rotation @ (local - anchor)` world transform；
- property leading dimension、ambiguous constant vector；
- progressive candidate 的逐 stage 可见性和 future-field access error；
- factory 显式读取 position/properties 时选择性失效，未读取字段变化不重建 Cell；
- factory/anchor validation 失败不发布 partial PopulationInstances 或 Cell；
- filter exact bool mask、empty selection、stable IDs 和 no reindexing；
- add-order dependencies 与 independent semantic RNG streams；
- `propagation_velocity=500 um/ms` default 以及“不自动改变 delay”。

### Materialization RNG

- 相同 Network/config/inputs 在固定软件、precision 和 backend 环境逐元素复现；
- 改变 Network seed 会改变全部 managed streams，改变一个 stream ID 只影响使用它的
  rules；
- 不同 rules 使用同一 stream ID 和相同 sampling call trace 时产生相同样本；
- 同一 evaluation 内重复 `with_seed(id)` 共享局部游标并连续消费，不从头
  replay；
- 插入无关对象、调换无依赖 add order、改变 inspection order 和并发 evaluation order；
- cache rematerialization 使用新建 rule-local handles 重现相同基础 stream；context 改变
  不进入 key，但可通过新 distribution、shape 或 mapping 改变结果；
- `probability`、fixed degree、Region/Locset sampling 和 per-contact parameter rule 的统一 workloads；
- NumPy semantic、JAX-backed facade 和 JAX + BrainState 的 cold/warm time、peak memory、
  host conversion 与 per-rule facade setup cost；
- float32/float64 precision 对 key/cache inputs、topology 和 continuous locations 的影响；
- custom callable 使用 `ctx.rng` 的确定性，以及 external/global RNG 不在保证范围内的诊断/文档；
- Network 不修改 NumPy、Python 或 BrainState global RNG state；
- semantic identity encoding 不使用 Python `hash()`，区分 component boundaries 和
  `auto/user` domains；
- built-in spec/repr 的 stream mode/ID 与 resolved diagnostics 的 semantic path、root source、
  non-reversible fingerprint；raw key 不得被诊断暴露；
- evaluation-order root splitting 只作为 order-sensitive baseline，不得静默通过 semantic gate。

### Pair 和 contact

- explicit/all-to-all/probability/fixed degree；
- endpoint membership、bounds、strict integer dtype、`nsyn<=0` 和 duplicate rejection；
- canonical sorting、pair/contact/synapse indices；
- empty `(0,3)` pair array/PairTable/ContactTable；
- scalar weight/delay/parameters 广播为 canonical leading-C columns，delay 严格为 `(C,)`；
- `Projection.parameters` 与 ContactTable parameter columns 共享 row order、storage 和
  `by_id(...)` selection；
- subset `all_to_all` 精确产生 `S*T` rows；
- fixed-degree memory 随 `P` 增长，probability 不保留完整 `S*T` random/mask arrays；
- Python generator/iterable、CSR/CSC 和 dense adjacency PairRule outputs 明确拒绝。

### Placement 和 synapse

- duplicate exact locations 保持独立 rows；
- Locset `+ | & - unique()` 的稳定顺序、sampling cardinality 和 ForkPoints topology junction；
- Region length/area random 和 stratified statistical checks；
- `cell_pair/target_cell` grouping differences；
- Locset replacement、candidate shortage 和 explicit probabilities；
- branch boundary provenance、world-coordinate failures；
- heterogeneous parameter columns、joint correlation 和 mechanism validation。
- `Projection.weight -> SynapseSpec.default_weight -> error` precedence、unit conversion、
  signed values、trigger-only/no-port validation；

### Runtime

- same point contacts 具有独立 states，currents 加到同一 membrane node；
- manual/network equivalent placement 的 voltage trace 一致；
- source threshold 与 `Cell.V_th` 独立；
- multiple Projections 使用不同 locations/thresholds；
- `ceil/strict/floor` 对 non-grid/heterogeneous delays 的量化，以及三种模式下 next-step
  zero-delay event phase；
- one long run 与连续 short runs 等价，包括跨边界 pending events；
- 未 init 的 run 和 Network-owned Cell direct init 明确报错；
- init failure 原子回到 EDITABLE，不保留 partial runtime；
- reset 后重现初始 dynamic state，同时保留 model parameters 和 compiled cache；
- deinit 后可编辑并重建，static tables/managed placements/model parameters 保留；
- episode 内改变 `dt` 或 delay quantization mode 报错，reset 后新值合法；
- scatter 与 brainevent backend 数值等价。

### Mutation、ownership 与 cache

- declaration mutation 跨 rematerialization 保留；table edit 只属于当前 snapshot；
- 上游 dependency 改变后 table edit 被重建结果替换，无关 dependency 不触发刷新；
- table edit 作为 producer version 正确失效读取它的 downstream rules；
- failed refresh 不发布 partial tables、placement layers、identity mappings 或 runtime arrays；
- Projection replacement/deletion 只撤销自己的 managed layer，manual/other layers 保持不变；
- population delete 默认检查引用，`cascade=True` 清理 dependent Projections；
- initialized weight/parameter same-signature update 成功，transitive structural change 拒绝；
- delay update 在 quantized groups/buffer layout 不变时成功，变化时要求 `deinit_state()`；
- reset/deinit 保留 model parameters；dynamic-state mutation 明确不经静态 `.set(...)`。

## 5. 性能验收

在逐步增大的 `N_cell`、`P_pair`、`C_contact`、unique delays 和 maximum delay 上记录：

- Population/Pair/Contact table host memory；
- packed mechanism parameters 和 dynamic state memory；
- batch placement wall time 与 C 次 Python `place()` baseline；
- PairRule candidate count `S*T`、materialized `P`、generation workspace 和 peak host memory；
- materialization RNG 的 cold/warm dispatch、sampling、host conversion 和 per-rule setup cost；
- JIT compile time、run step time 和 cache reuse；
- scatter 与 brainevent delivery throughput；
- ring-buffer bytes 随 unique delays 和 delay depth 的增长曲线。

验收要求：contact metadata 和 synapse state 按 C 保存；不出现
`N_cell * max_synapses_per_cell` padding；不得用 contact storage 的线性结果掩盖 delay queue
的额外增长。

## 6. 进度维护

每次实施更新只修改对应 task 状态，并附上测试或 benchmark 证据。规范决策先关闭 issue，
再更新 API/architecture，最后实施；不得先由代码事实反向生成未评审的公开契约。
