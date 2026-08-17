# BrainCell Network 设计文档

本目录定义 BrainCell 新一代多室细胞网络构建器。目标是把用户声明、构建结果和
JAX runtime 分层，同时让异质 point synapse 按实际 contact 稀疏保存。

## 文档地位

- [api.md](./api.md) 与 [architecture.md](./architecture.md) 是同等正式的规范。公开行为或
  内部数据模型变化时，必须检查并同步更新另一份。
- [issues.md](./issues.md) 保存开放问题以及已锁定/延期的决策记录。实现者不能绕过其中仍
  开放的决策点自行选择新语义；正式行为仍以 API/architecture 为准。
- [implementation-plan.md](./implementation-plan.md) 是可持续更新的工程进度表，不定义
  新的 API 或 runtime 语义。
- [references/](./references/) 只保存历史调研，属于非规范材料。

## 核心结论

- v1 面向 vectorized multicompartment `Cell`，population 是构建和索引的基本单位。
- Network-owned Population/Projection handles 管理 lifecycle；immutable specs 只负责声明；
  PopulationInstances、PairTable 和 ContactTable 是按需缓存的 static materialization。
- cell pair 与 synaptic contact 是不同层级；一个 pair 可以展开为多个独立 contacts。
- cell-pair topology 只使用稀疏 rows，不保存 dense adjacency matrix。
- 每个 contact 对应一个独立 point-mechanism state，即使多个 contacts 位于同一
  branch/x 或同一 electrical point 也不合并。
- `(branch_id, branch_x)` 是权威位置；CV、point 和 runtime layout index 是 lowering 后的
  派生信息。
- `LocsetExpr` 是延迟求值的声明和集合代数；求值后的 `LocsetMask` 是 ordered、
  duplicate-preserving rows，默认不排序、不去重，sampling 必须显式声明。
- static materialization 使用 host-side arrays；runtime 使用现有 packed point layout 和
  JAX arrays。
- physical delay 在 runtime setup 时按 `dt` 量化，v1 继续使用 ring-buffer delivery。

## 当前状态

已锁定：Network defaults、quaternion Rotation 与 spatial-anchor world transform、population
cell/factory ownership、progressive candidate context、endpoint view、array PairRule protocol、
稀疏 Pair/Contact tables、连续 morphology location、ordered `LocsetMask`、add-order dependencies、
packed runtime、Projection-local event source、target-defined event weight、三种 delay quantization、
EDITABLE/INITIALIZED lifecycle、bounded generation、weighted Region sampling、monotonic contact ID、
dependency-selective invalidation，以及 Network-rooted semantic RNG streams。

仍开放：I-01 public vocabulary 最终复核、I-06 batch placement bridge、I-10 materialization RNG
sampling adapter，以及 I-11 heterogeneous-delay buffer layout。

已延期：I-13 kinetic/continuous synapse input protocols 和 I-14 initialized dynamic-state
mutation API。两者不阻塞 v1。

## 阅读顺序

1. 使用或评审公开接口：阅读 [api.md](./api.md)。
2. 实现 builder、lowering 或 runtime：阅读 [architecture.md](./architecture.md)。
3. 处理尚未冻结的设计点：阅读 [issues.md](./issues.md)。
4. 开始工程任务或检查验收 gate：阅读 [implementation-plan.md](./implementation-plan.md)。

## 首版范围

首版包括静态多室网络、稀疏 cell pairs、独立 chemical synapse contacts、连续位置放置、
异质 mechanism parameters、projection-specific voltage event detection 和固定步长 delay。

首版不包括 point-neuron adapter、gap junction、运行中 structural rewiring、topology/placement
可微生成、跨 Projection detector 自动合并，以及长期维护两套公开 network API。
