# BrainCell Network 设计总览

本文档集用于整理 BrainCell 后续 `network` 层的设计方向，聚焦以下问题：

- network construction 的前端接口应该长什么样
- 同构 `population` 在 JAX 下应如何组织内部 runtime
- synapse / spike / delay 的更新语义该如何选型
- 哪条路线更兼容后续 gradient-based training

这是一组内部设计文档，服务于架构讨论和实现拆解；它不替代 `README.md`、`TODO.md` 或 `docs/` 下的对外说明。

## 当前结论

基于本轮对 `NEURON/CoreNEURON`、`Arbor`、`Jaxley`、`brainevent` 的调研，以及 BrainCell 当前声明层 / runtime 层结构，当前建议如下：

- 外部 API 不强制用户先创建 `population` 对象；应允许 cell-centric network declaration。
- 内部 runtime 对同构 `population` 默认做 batch-style 聚合，而不是逐 cell Python 对象更新。
- 第一阶段默认推荐 `JAX-friendly static-shape path`：
  - batched neuron state
  - edge table or structured connectivity
  - per-step synapse state update
  - `scatter_add`-style postsynaptic aggregation
- delay 第一阶段优先使用 fixed-shape `ring buffer`，不先实现动态长度 `event queue`。
- training 主线优先兼容 surrogate spike / soft release；后续可基于 `brainevent` 探索 JAX-friendly event-sparse delivery，但不把完整动态 `event queue` 作为第一阶段前提。

一句话总结：**API 层不必 population-first，但 runtime 层应对同构群体默认 population-like。**

## 文档地图

- [platform-survey.md](./platform-survey.md)
  - 平台调研
  - 对 `NEURON/CoreNEURON`、`Arbor`、`Jaxley`、`brainevent` 的 network / synapse / delay / parallelization 做横向比较
- [braincell-network-design.md](./braincell-network-design.md)
  - BrainCell 候选架构
  - 明确推荐路线、数据结构、step 顺序、复杂度与训练兼容性
- [implementation-plan.md](./implementation-plan.md)
  - 分阶段实施计划
  - 以同构 `population` 的最小可运行版本为起点

## 建议阅读顺序

如果是第一次参与这项工作，建议按下面顺序读：

1. 本文，先拿到结论和术语范围
2. [platform-survey.md](./platform-survey.md)，理解已有平台的实现差异
3. [braincell-network-design.md](./braincell-network-design.md)，看 BrainCell 推荐方案
4. [implementation-plan.md](./implementation-plan.md)，进入具体落地阶段

## 范围与非目标

本文档集当前主要覆盖：

- 同构 `population`
- JAX / XLA friendly runtime organization
- chemical synapse 的 spike / delay / current aggregation
- gradient-compatible training path

当前明确不覆盖：

- 异构 morphology 的自动 regroup compiler
- 分布式多机 spike transport 协议
- gap junction / plasticity / structural rewiring 的完整方案
- 最终稳定对外 API 文案
