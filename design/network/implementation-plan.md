# BrainCell Network 分阶段实施计划

本文给出 network 层的逐步落地路线。  
目标不是一次做完所有语义，而是先拿到一条可运行、可训练、可扩展的主线。

## 1. 总体原则

实施顺序按下面优先级展开：

1. 先做 **同构 `population`**
2. 先做 **JAX-friendly static-shape path**
3. 先跑通 **chemical synapse + delay**
4. 再评估是否需要 `brainevent` 支撑的 event-sparse mode

## 2. Phase 0: 文档与接口定稿

### 目标

把高层语义定清楚，避免实现阶段反复返工。

### 主要工作

- 明确 network 层的输入对象
- 明确 state tensor shape
- 明确 synapse update 顺序
- 明确 delay 默认采用 `ring buffer`
- 明确 training 主线采用 surrogate / soft-compatible 方案

### 产出

- `design/network/README.md`
- `design/network/platform-survey.md`
- `design/network/braincell-network-design.md`
- `design/network/implementation-plan.md`

### 完成标准

- 实现者无需再追问高层路线
- “为什么不先上 event queue” 有明确记录
- “为什么同构 population 默认 batch 化” 有明确结论

## 3. Phase 1: 同构 population 最小可运行版

### 目标

支持一个最小可运行的 batched network forward path。

### 建议范围

- 先从 `SingleCompartment` 同构群体起步
- connectivity 固定
- synapse 先支持一到两种常见 type
- 暂不支持 delay 或只支持 `delay = 0`

### 主要工作

- neuron states batched 化
- connectivity lowering 为 edge table
- edge-wise synapse state 更新
- postsynaptic current `scatter_add` 聚合
- network step 进入 `lax.scan`

### 测试项

- 小型 E/I toy network 可前向运行
- CPU / GPU 下 `jit` 成功
- batched forward 结果与逐个 cell 计算一致

### 完成标准

- 有一个稳定可运行的 forward baseline
- 代码路径不依赖 Python-side per-cell loop

## 4. Phase 2: integer-step delay ring buffer

### 目标

在保持 static shape 的前提下支持 delayed spike / release transmission。

### 主要工作

- 引入 fixed-size `ring buffer`
- 设计 `delay -> slot offset` 的映射
- 增加未来 slot 写入
- 增加当前 slot 读取与消费

### 推荐默认约束

- delay 先限制为 integer number of steps
- 不支持动态变化的 queue length

### 测试项

- 单条连接 delay correctness
- 多条不同 delay 的连接在同一步混合投递
- `delay = 0` 与无 delay path 等价

### 完成标准

- delayed network 能稳定进入 `jit` / `scan`
- shape 与内存布局不随仿真过程变化

## 5. Phase 3: training path

### 目标

让 network 路径能稳定参与 gradient-based optimization。

### 主要工作

- surrogate spike 或 soft release 路径接入
- 检查 BPTT 下 state/grad 是否稳定
- 明确哪些参数在第一阶段允许训练

### 建议先开放的可训练参数

- synaptic weights
- synaptic conductance / time constants
- 部分 neuron intrinsic parameters

### 测试项

- toy loss 对 synaptic weights 可反传
- gradient finite 且非全零
- 多步 `scan` 下不出现明显 shape/trace 问题

### 完成标准

- 有最小训练脚本或测试证明 network path 可反传

## 6. Phase 4: multi-compartment 同构 population

### 目标

将 Phase 1-3 的单隔室同构路线推广到多隔室同构群体。

### 主要工作

- batched `V[n_cell, n_comp]`
- runtime layout 与现有 `_multi_compartment` lowering 对齐
- synapse target 从 post cell 扩展到 post compartment / point target
- current aggregation 接入现有 membrane current path

### 风险

- state tensor 变大
- scatter/gather 路径更重
- compartment-level indexing 复杂度上升

### 测试项

- 同构 morphology 的小型多隔室 network
- 与单细胞前向语义一致
- 多 compartment target 的 delay correctness

### 完成标准

- 同构 multi-compartment population 具备最小可运行 network path

## 7. Phase 5: brainevent-backed event-sparse exploration

### 目标

只在需要时评估是否引入 `brainevent` 支撑的 JAX-friendly sparse event path。

### 触发条件

只有在以下任一情况成立时才值得推进：

- 低 firing 稀疏网络下 per-step path 明显浪费
- 需要更贴近传统 simulator 的事件语义
- 需要和已有 event-driven reference 做高精度对照

### 主要工作

- 比较 ring-buffer + `scatter_add` path 与 `brainevent` sparse kernel path 的真实性能差异
- 确认是否需要独立 execution mode 或只是在 aggregation kernel 上切换后端
- 评估 `CSR` / fixed-conn / edge-group 哪种 lowering 最适合 BrainCell
- 评估与 training path 的 API 关系

### 完成标准

- 有 benchmark 证明额外复杂度值得引入
- 若无明显收益，则保持 ring-buffer 主线不变

## 8. 贯穿各阶段的测试与验收

每个阶段都应保留以下检查：

- **shape stability**
  - `jit` 前后 shape 不漂移
- **correctness**
  - 小网络手算可验证
- **consistency**
  - batch path 与 reference path 一致
- **delay correctness**
  - 不同步延迟不会串位
- **gradient sanity**
  - 训练主线路径梯度存在且有限

## 9. 当前主要风险

### 9.1 hard spike 带来的梯度不稳定

根源通常不在 delay buffer，而在离散 threshold 本身。  
因此训练主线必须提前定义 surrogate / soft strategy。

### 9.2 scatter-heavy path 在部分硬件上的性能波动

edge-wise `scatter_add` 很灵活，但在不同 accelerator 上性能可能差异较大。  
因此需要尽早保留 benchmark 位点。

### 9.3 layout 决策会影响后续扩展

例如：

- edge-major
- post-major
- 按 synapse type 分 block

这些都会影响未来性能与代码复杂度，因此第一阶段实现要尽量保持可替换。

### 9.4 多隔室 population 的内存增长

同构多隔室群体一旦 batch 化，状态量会迅速放大。  
因此应尽量先用小型 network 验证 correctness，再上更大规模 benchmark。

## 10. 推荐推进顺序

建议实际开工顺序如下：

1. 文档定稿
2. `SingleCompartment` 同构 network forward
3. delay ring buffer
4. surrogate/soft training path
5. multi-compartment 同构 network
6. 再评估 `brainevent` event-sparse compatibility
