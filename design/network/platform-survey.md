# Network / Synapse 平台调研

本文聚焦四条相关路线：

- `NEURON / CoreNEURON`
- `Arbor`
- `Jaxley`
- `brainevent`

重点不是只记 API 名字，而是同时看：

- 用户如何写网络
- runtime 内部按什么粒度组织计算
- presynaptic spike / synapse / delay 是如何传递的
- 哪条路线更适合 BrainCell 当前的 JAX 架构

## 1. 判断维度

为避免“接口形式”和“执行形式”混在一起，本文按下面几个维度比较：

- **Front-end API**
  - 用户是按 `cell`、`population`、`recipe` 还是 `edge table` 建网
- **Internal execution unit**
  - 运行时按单 cell、cell group、population block 还是 synapse edge 组织
- **Synapse delivery semantics**
  - 是 event-driven 还是 per-step continuous update
- **Delay handling**
  - delay 是通过 queue、ring buffer、或连续状态来表达
- **Parallelization strategy**
  - 并行工作是围绕 cells、groups、events 还是 edges 展开
- **Training compatibility**
  - 与 JAX / autodiff / BPTT 的天然兼容程度如何

## 2. NEURON / CoreNEURON

### 2.1 Front-end API

`NEURON` 的典型 network API 不是 population-first，而是基于 `gid` 和 `NetCon`：

- `ParallelContext.set_gid2node(gid, rank)`
- `ParallelContext.cell(gid, netcon_source)`
- `ParallelContext.gid_connect(src_gid, target_synapse)`
- `NetCon.weight[0]`
- `NetCon.delay`

这意味着：

- 用户可以任意创建很多 cell
- 不要求这些 cell 先包成统一 `population`
- 网络层的核心身份是 `gid -> spike source`

### 2.2 Internal execution

`NEURON` 的接口虽然是 cell-centric，但 `CoreNEURON` 后端会显式做性能重排：

- data layout 优化
- `Structure of Arrays (SoA)`
- cell permutation / node reorder
- 负载均衡

因此，“不要求用户先建 population”并不等于“运行时逐对象低效地算”。  
关键不在前端有没有 `population`，而在后端是否做 regroup / repack / reorder。

### 2.3 Synapse / spike delivery

`NEURON` 主路径是典型 **event-driven**：

- presynaptic source 发生 threshold crossing
- 该 spike 通过 `NetCon` 形成 future delivery event
- event 按 `delay` 放进 event queue
- 到达投递时刻后，target point process 的 `NET_RECEIVE` 被触发

所以它不是“每一步都算一遍全连接稀疏矩阵乘法”，而是：

- 只在有 spike 时生成 event
- 只在 event 到达时投递 synapse effect

### 2.4 Delay / parallelization

`NEURON/CoreNEURON` 的 delay 表达天然来自 event queue 语义。  
并行时，跨 rank 交换的是 spike，然后本地再按 `gid` / `NetCon` 分发为 target events。

### 2.5 对 BrainCell 的启示

可借鉴点：

- 外部不强制 population-first
- network identity 可围绕 source id / connection table 组织
- event-driven 路线在低 firing 稀疏网络中很有效

不建议第一阶段直接照搬的点：

- 动态事件系统复杂
- JAX / static-shape / autodiff 不友好
- 对当前 BrainCell 的 training-first 路线不是最短路径

## 3. Arbor

### 3.1 Front-end API

`Arbor` 的 network 前端更偏 `recipe`：

- `num_cells()`
- `cell_description(gid)`
- `cell_kind(gid)`
- `connections_on(gid)`
- `event_generators(gid)`

也就是说：

- 网络仍然是按 `gid` 描述
- `connections_on(gid)` 返回一个 cell 的 incoming connections
- 用户并不需要手工组织“巨大的 population tensor”

### 3.2 Internal execution

`Arbor` 的关键设计点是 **cell groups**：

- external declaration 仍是 cell-centric
- internal execution 会按 compatible cell kind / domain decomposition 建 `cell_group`
- cell group 在后端 lockstep 推进

这说明 Arbor 本质上采用的是：

- front-end: cell-centric
- runtime: group-centric

### 3.3 Synapse / spike delivery

`Arbor` 使用离散 `spike {source, time}` 作为通信单位：

- spikes 在 cell groups 之间传播
- local targets 再根据 connection table 变成 post events
- 事件分发不是每步全扫所有连接

因此 Arbor 的突触传播语义仍是 **event-driven sparse communication**。

### 3.4 对 BrainCell 的启示

最值得借鉴的是结构分层：

- 用户可以保留 cell-centric declaration
- 编译后 runtime 可以按“可共用 kernel 的签名”自动 regroup

对 BrainCell 的实际含义是：

- 不必要求用户先写死 population API
- 但内部必须允许 batch/group lowering

## 4. Jaxley

### 4.1 Front-end API

`Jaxley` 的 network 构造更直接暴露 edge-centric helper：

- `Network([...])`
- `connect(pre, post, synapse)`
- `fully_connect(pre_cells, post_cells, synapse)`
- `sparse_connect(pre_cells, post_cells, synapse, p=...)`

其连接创建本质上是在构造一张 `edges` 表。

### 4.2 Internal execution

从源码看，`Jaxley` 的主路径更接近：

- batched / vectorized neuron states
- edge-wise synapse states
- per-step synapse update
- `scatter_add` 聚合到 postsynaptic compartments

典型逻辑是：

- 根据 `pre_voltage` / `post_voltage` 更新每条 synapse 的状态
- 计算每条 synapse 的 current term
- 用 `scatter_add` 把这些项累加到 post side

### 4.3 Synapse semantics

当前主线并不是 `NEURON` 式 event queue，而是更适合 JAX 的 **per-step continuous update**。

这带来两个直接后果：

- 优点：shape 更静态，容易 `jit`、`scan`、自动微分
- 缺点：生物事件语义不如 queue-based path 直接

### 4.4 对 BrainCell 的启示

对 BrainCell 当前阶段最有价值的借鉴是：

- 同构 `population` 可直接采用 batched tensors
- synapse 先走 edge table + per-step update
- postsynaptic current 聚合采用 `scatter_add`
- delay 先设计成 fixed-shape buffer，而不是动态 queue

## 5. brainevent

### 5.1 定位

`brainevent` 不是一个完整的 network simulator，而更接近一层面向 JAX 的
**event-sparse / sparse-connectivity 算子库**。

从 `braincell_311` 环境中的 `brainevent 0.0.7` 包结构与公开 API 看，它主要提供：

- `EventArray`、`BinaryArray`、`MaskedFloat`
- `CSR`、`COO`、`FixedPostNumConn`、`FixedPreNumConn`
- `update_csr_on_binary_pre`、`update_csr_on_binary_post`
- `update_coo_on_binary_pre`、`update_coo_on_binary_post`
- 面向 XLA 的 custom kernel / pallas / numba / warp 后端

这意味着它擅长解决的是：

- presynaptic event 到 postsynaptic accumulation 的稀疏算子表达
- 固定 shape 稀疏连接在 JAX `jit` / `scan` 下的执行效率
- 稀疏 event matvec / matmat 的 kernel specialization

### 5.2 它没有直接替 BrainCell 解决什么

至少按当前可见 API，`brainevent` 并没有直接提供：

- `NEURON` 式 network-level 动态 event queue
- 带调度语义的 spike delivery runtime
- 多 delay 事件容器的完整 simulator 抽象

因此它不能直接替代 BrainCell 自己的：

- network declaration
- delay slot / ring buffer 语义
- neuron / synapse step scheduling

### 5.3 对 BrainCell 的实际价值

对 BrainCell 来说，`brainevent` 最值得利用的不是“把 network 直接改成
event queue simulator”，而是：

- 在保持 static shape 的前提下表达更稀疏的 presynaptic event delivery
- 将 edge table / fixed-conn layout lower 到更高效的稀疏 kernel
- 为低 firing 稀疏网络提供一个比纯 `scatter_add` 更强的后端选项

因此它更像：

- **execution primitive layer**

而不是：

- **complete network semantics layer**

## 6. 横向比较

| 平台 | 用户接口 | 内部执行单元 | Synapse 更新主语义 | Delay 表达 | 对训练友好度 | 对 BrainCell 的借鉴 |
| --- | --- | --- | --- | --- | --- | --- |
| `NEURON/CoreNEURON` | `gid` + `NetCon` + cell-centric | reordered cell/node runtime | event-driven | event queue | 低到中 | 前端不必 population-first；后端必须重排 |
| `Arbor` | `recipe` + `connections_on(gid)` | `cell_group` | sparse spikes + local event delivery | event-based | 中 | cell-centric declaration + grouped runtime |
| `Jaxley` | `Network` + edge helpers | batched tensors + edge table | per-step edge update | 可做 fixed buffer | 高 | JAX-friendly 第一阶段基线 |
| `brainevent` | sparse/event operators | sparse matrix or fixed-conn kernel | event-sparse accumulation | 不自带 network queue | 中到高 | 适合作为 JAX event-sparse execution backend |

## 7. 面向 BrainCell 的总结判断

对于 BrainCell 当前问题，最重要的不是复制哪一个平台的 API，而是明确：

- **同构 population 的第一阶段实现目标** 应该优先服务 JAX / training / static shape
- **异构 cell 的长期架构** 可以借鉴 Arbor / CoreNEURON 的 regroup 思路

因此当前推荐路线是：

1. 第一阶段更靠近 `Jaxley` 的计算形态
2. 第二阶段引入 `Arbor` 风格的 grouped runtime 思想
3. 在需要更 sparse 的 delivery path 时，优先评估 `brainevent` 支撑的 JAX event-sparse kernel
4. 只在确有性能或语义收益时，再评估 `NEURON` 风格动态 event system

## 8. 参考资料

### NEURON / CoreNEURON

- `ParallelContext`
  - https://nrn.readthedocs.io/en/latest/progref/modelspec/programmatic/network/parcon.html
- `NetCon`
  - https://nrn.readthedocs.io/en/latest/progref/modelspec/programmatic/network/netcon.html
- parallel ring tutorial
  - https://nrn.readthedocs.io/en/latest/tutorials/ball-and-stick-4.html
- CoreNEURON overview
  - https://nrn.readthedocs.io/en/8.2.2/coreneuron/index.html
- CoreNEURON inputs / execution notes
  - https://nrn.readthedocs.io/en/latest/coreneuron/inputs.html
- NEURON data structures
  - https://nrn.readthedocs.io/en/latest/dev/data-structures.html
- `netcvode.cpp`
  - https://github.com/neuronsimulator/nrn/blob/master/src/nrncvode/netcvode.cpp
- CoreNEURON paper
  - https://arxiv.org/abs/1901.10975

### Arbor

- ring network tutorial
  - https://docs.arbor-sim.org/en/latest/tutorial/network_ring.html
- interconnectivity
  - https://docs.arbor-sim.org/en/latest/concepts/interconnectivity.html
- cell groups
  - https://docs.arbor-sim.org/en/latest/dev/cell_groups.html
- communication
  - https://docs.arbor-sim.org/en/latest/dev/communication.html
- `cable_cell_group.cpp`
  - https://github.com/arbor-sim/arbor/blob/master/arbor/cable_cell_group.cpp

### Jaxley

- `connect.py`
  - https://github.com/jaxleyverse/jaxley/blob/main/jaxley/connect.py
- `network.py`
  - https://github.com/jaxleyverse/jaxley/blob/main/jaxley/modules/network.py
- `syn_utils.py`
  - https://github.com/jaxleyverse/jaxley/blob/main/jaxley/utils/syn_utils.py
- `ionotropic.py`
  - https://github.com/jaxleyverse/jaxley/blob/main/jaxley/synapses/ionotropic.py

### brainevent

- PyPI
  - https://pypi.org/project/brainevent/
- repository
  - https://github.com/chaobrain/brainevent
