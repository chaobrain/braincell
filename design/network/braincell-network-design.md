# BrainCell Network 候选架构与推荐路线

本文给出面向 BrainCell 的推荐设计，不做“方案并列而不决”。  
默认前提是：

- 第一阶段聚焦 **同构 `population`**
- 运行后端基于 **JAX / XLA**
- 后续需要兼容 **gradient-based training**

## 1. 设计目标

当前 network 层设计应优先满足：

- **static-shape friendly**
  - 能稳定 `jit`
  - 能进入 `lax.scan`
- **batch-friendly**
  - 同构群体能直接走 batched tensors
- **training-friendly**
  - 对 surrogate spike / soft release 保持兼容
- **incrementally extensible**
  - 后续可以增量接入更 event-sparse 的 sparse path

## 2. 问题重述

本轮讨论实际涉及四个判断：

### 2.1 是否需要 population-first API

结论：**不需要**。

原因：

- BrainCell 当前声明层已经偏 cell / mechanism-centric
- `NEURON` 和 `Arbor` 证明了前端可以不是 population-first
- 真正决定效率的是 runtime grouping，不是用户 API 外形

### 2.2 JAX 下是否需要“内部聚合”

结论：**需要，而且对同构群体应默认强聚合**。

对于同构 `population`：

- morphology 一致
- state layout 一致
- 只有参数值 / 初值不同

那么最自然的 runtime 形态就是 batched arrays，而不是一组独立 Python cell 对象。

### 2.3 Synapse 更新应选哪条主路线

结论：**第一阶段默认选择 per-step synapse update，而不是动态 event queue**。

原因：

- JAX 更擅长规则张量计算
- `scan` / `jit` 更喜欢固定 shape
- event queue 的动态长度和稀疏随机写入不适合作为第一阶段主路径
- 若后续需要更 sparse 的 presynaptic delivery，优先考虑 `brainevent` 一类固定 shape 稀疏算子后端

### 2.4 Delay 与训练是否冲突

结论：**不冲突，但 delay 的表达方式要 static-shape**。

推荐方式：

- integer-step delay
- fixed-shape `ring buffer`

不推荐第一阶段直接上：

- 动态 event list
- Python-side pending event containers

## 3. 推荐路线

### 3.1 总体判断

BrainCell 第一阶段推荐采用：

- **外部**：cell-centric declaration，可额外提供 population helper
- **内部**：batch-style runtime lowering
- **synapse**：edge-wise state + per-step update
- **delay**：fixed-shape ring buffer
- **training**：surrogate / soft-spike compatible

这条路线更接近：

- execution shape: `Jaxley`
- layering philosophy: `Arbor`

同时为后续接入 `brainevent` 风格的 event-sparse kernel 预留空间，而不是直接复制 `NEURON` 的完整动态事件系统。

### 3.2 推荐接口层次

建议把 network 层分成三层：

1. **Declaration layer**
   - cells
   - connectivity
   - synapse specs
2. **Lowering layer**
   - 将同构 cell 群体编译成 batched state layout
   - 将连接编译成 edge-major 或 grouped connectivity layout
3. **Execution layer**
   - 单步推进
   - synapse update
   - delay delivery
   - current aggregation

## 4. 数据结构建议

以下是同构 `population` 的默认推荐形态。

### 4.1 Neuron states

对于多隔室同构群体，优先采用：

- `V.shape = [n_cell, n_comp]`
- gate / ion / auxiliary states 也按 batch 维展开

例如：

- `m.shape = [n_cell, n_comp]`
- `Ca_i.shape = [n_cell, n_comp]`

如果某些机制只存在于固定 subset compartments，也应尽量在 lowering 后统一成规则张量或带固定 index 的 gather/scatter 路径。

### 4.2 Connectivity

第一阶段建议同时支持两种 layout，但默认先落地 edge table：

- **dense / structured path**
  - `W.shape = [n_pre, n_post]`
  - 适合同构 point-neuron 或小规模 dense network
- **sparse edge path**
  - `src[e]`
  - `dst[e]`
  - `weight[e]`
  - `delay[e]`
  - 可选 `synapse_type[e]` 或按 type 分桶

默认更推荐 sparse edge path，因为它更接近后续多机制 / delay / 稀疏网络需求。

### 4.3 Synapse states

推荐 edge-wise：

- `g[e]`
- `x[e]`
- `u[e]`

不同 synapse 类型可采用：

- 按 type 分开存放 edge blocks
- 或一个统一 edge table + type-specific views

第一阶段建议优先：

- `type-grouped edge blocks`

原因是：

- 更好 `jit`
- 更少 branchy logic
- 更容易写 type-specific update kernel

### 4.4 Delay buffer

推荐 fixed-shape ring buffer：

- `buffer[delay_slot, post]`
  - 适合先聚合后投递
- 或 `buffer[delay_slot, edge]`
  - 适合 edge-wise exact delivery

默认建议先从 `buffer[delay_slot, edge]` 起步，再视性能评估是否转成 post-major 聚合。

原因：

- 语义最直观
- 与 edge-wise synapse state 对齐
- 更容易验证 correctness

## 5. 单步执行顺序

推荐把 network step 写成固定顺序的纯函数，放进 `lax.scan`。

### 5.1 Training-friendly 默认顺序

每个时间步执行：

1. 读取当前 delay slot 的 arrivals
2. 用 arrivals 或 soft release 更新 synapse states
3. 计算每条 synapse 的 current term
4. 将 synaptic current 用 `scatter_add` 聚合到 postsynaptic compartments
5. 推进 neuron intrinsic dynamics
6. 产生当前时刻 `spike` 或 `release`
7. 将其按 `delay` 写入未来 ring buffer slot
8. 滚动 ring pointer

这个顺序的核心优点是：

- 每步都是固定 shape
- 方便把 synapse current 合并进现有 membrane current path
- 兼容 surrogate spike 和 soft release

### 5.2 为什么不先用动态 event queue

动态 queue 的主要问题不是数学语义，而是执行形态：

- queue 长度随时间变化
- event append / pop 是数据依赖的
- `jit` 下不自然
- 容易把实现推回 Python 控制流

因此对 BrainCell 第一阶段不合适。

## 6. 两条语义路径

### 6.1 Path A: training-friendly path

这是默认主线。

特点：

- batched neuron states
- edge-wise synapse states
- per-step update
- fixed-shape ring buffer
- surrogate spike 或 soft presynaptic release

优点：

- 最适合 JAX / autodiff / BPTT
- 实现路径短
- shape 稳定

缺点：

- 不完全等价于 `NEURON` 的离散事件语义
- 在超低 firing 稀疏网络里，可能不是最省算的

### 6.2 Path B: event-driven compatibility path

这是后续兼容路线，不是第一阶段默认。

特点：

- 更接近 `NEURON` / `Arbor` 的稀疏事件投递
- 可以只在 spike / release 有效时触发稀疏 accumulation
- 优先基于 `brainevent` 这类 JAX-compatible sparse/event primitives 实现
- 保持 runtime shape 静态；不默认引入动态长度 queue

优点：

- 低 firing 稀疏网络下有潜在性能优势
- 更接近传统神经模拟器语义
- 比完整动态 queue 更有机会兼容 `jit` / `scan`

缺点：

- 实现复杂
- 仍然比纯 per-step path 更难调试和验证
- 如果要完全复刻 `NEURON` 式 queue 语义，仍然不利于 JAX-friendly static-shape design

## 7. 复杂度与适用场景

### 7.1 Dense matrix path

复杂度接近：

- `O(n_pre * n_post)` 每步

适用：

- 小型 dense network
- 原型验证

不适用：

- 大稀疏网络

### 7.2 Sparse edge path

复杂度接近：

- `O(E)` 每步

适用：

- 同构群体
- 中大规模稀疏连接
- training-first 路线

### 7.3 Ring buffer path

复杂度接近：

- `O(E_active_update + E_delivery)` 或 `O(E)`，取决于具体更新策略

适用：

- 存在 delay
- 需要保持 static shape

### 7.4 Dynamic event queue path

复杂度与 firing activity 更相关，低 firing 时潜在更优，但：

- 实现和编译复杂度高
- 不适合作为第一阶段主路径

### 7.5 brainevent-backed event-sparse path

复杂度更接近：

- `O(E_active)` 到 `O(E)`，取决于 active event 比例和所选 sparse layout

适用：

- 低 firing 稀疏网络
- 仍希望保持 JAX `jit` / `scan` 友好
- 已经能把 connectivity lower 成 `CSR` / fixed-conn / edge-group layout

限制：

- 不能自动替代 delay scheduler
- 仍需 BrainCell 自己定义 network-level step 语义

## 8. 与 BrainCell 当前结构的对接

从当前仓库设计看，已有以下基础：

- `HHTypedNeuron.pop_size`
- `batch_size`-aware state initialization
- `_multi_compartment.bridge` 中已有 scatter/gather helper
- multi-compartment runtime 已有 `V` / `spike` / layout lowering 语义

因此 network 层应尽量：

- 复用现有 neuron / channel / ion runtime 协议
- 将 network 视为“在 neuron runtime 外再加 connectivity / synapse / delay 层”
- 不重造单细胞机制接口

## 9. 当前明确不做的事

第一阶段不建议实现：

- 动态长度事件队列
- 异构 morphology 自动 regroup compiler
- 完整 MPI-style spike exchange
- 完全仿 `NEURON` 的事件语义对齐

## 10. 待后续单独定稿的决策项

后面需要单独下结论，但不阻塞第一阶段：

- edge-major vs post-major synapse layout
- `delay` 的单位是否只允许整数步长
- `scatter_add` baseline 与 `brainevent` sparse kernel 的切换条件
- surrogate spike 与 soft release 是否同时暴露
- 多种 synapse type 是否统一存储还是分 block lowering
