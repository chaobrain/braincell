# Network Implementation Plan

## Completed foundation

- [x] Population 统一注册 Cell、NetStim 和 EventSequence，提供 canonical event-output ports，并自动公开
  首次用于 Network Connection 的具名 Cell EventSource。
- [x] Cell-owned `_SynapseStore`、`SynapseView` 和按 type 合并的 runtime SoA。
- [x] Cell-owned `_ConnectionStore`、named batched connect calls 和 `ConnectionView`。
- [x] scheduled/live source、weight contract、异质 delay、split run 和 reset semantics。
- [x] spatial Channel/Ion views、CV overlap validation 和 layout-free recording。
- [x] Channel type/name、Ion species/type/name、Synapse type/name/ids 的统一 observable identity。
- [x] current contributor/sum reduction、静态 RecordingSchema、不可变 SampleBlock 与稀疏 EventSeries。

## Direct Network convergence

- [x] 删除旧 cell-pair/table/pool public path 和显式 build phase。
- [x] `Network.connect` 支持已有 SynapseView 与原子 place+connect。
- [x] Population 转发 `synapses`、`connections`。
- [x] Network 按 target 聚合 connections，并分别报告 named calls 与 rows。
- [x] direct runtime tests 覆盖 owner、lifecycle、delay、backend、split run 和 cache reuse。
- [x] recording notebook 覆盖空间 scope、机制 identity、state/current reduction、result schema、注册 source
  和 continued-run comparison。

## Endpoint pairing

- [x] direct/Network `connect(..., pairing=...)` 共用临时 endpoint materialization。
- [x] 固定行数 independent、source-first 和 synapse-first sampling。
- [x] by-source、by-synapse degree expansion 与双侧 exact stub matching。
- [x] target-cell grouping、conditional `(B, K)` score context 和 morphology geometry fields。
- [x] BrainState RNG、Network order-independent seed path 和显式 seed override。
- [x] duplicate-candidate、无放回 positive support、零行与 shape/unit validation。
- [x] focused correctness tests 与非阻断 profiling benchmark。

## Verification gates

- [x] 全部 `braincell/` tests 通过（2261 passed，30 skipped）。
- [x] `synapse.ipynb`、`connection.ipynb`、`network.ipynb`、`recording.ipynb` 无错误执行。
- [x] recording 示例验证 raw contributor rows、按 `(population, CV)` 聚合 rows 和 schema metadata。
- [x] NEURON comparison notebook 不引用已删除 API。
- [x] docs/examples 中无旧 public topology symbol。
- [x] profiling probability workload 使用显式 source/synapse row views 并通过最小规模测试。

## Deferred

- [ ] 大 N endpoint pairing 的 chunked score evaluation 和 specialized sparse generators。
- [ ] sparse/dense delay queue 自动选择和性能基准。
- [ ] 初始化后结构 mutation、trainable topology 和 Network batch runtime。
