# Cerebellum ion calcium dynamics import plan

## 背景

当前 `braincell/channel/` 和 `braincell/ion/` 已经有一套 template 化实现：

- `channel/_base.py` 已支持 HH 与单 conserved pool 的 Markov channel。
- `ion/_base.py` 已支持固定离子、初始化 Nernst 离子、以及单个动态 `Ci` 的 `DynamicNernstIon`。

但 `examples/neuron_compare/Cerebellum_mod` 里的钙 ion 机制不只是“再写一个 `CalciumDetailed` 变体”。它们分成两类：

- 简单 `DERIVATIVE` 型：如 DCN 的 `CdpHVA_SU15_DCN`、`CdpLVA_SU15_DCN`
- 复杂 `KINETIC` 型：如 BC/SC/GoC 的 `CdpStC`、GrC 的 `CdpCR`、PC 的 `CdpCAM`

后者包含 `KINETIC`、`COMPARTMENT`、`CONSERVE`、`<<`、`f_flux/b_flux`、buffer/pump 反应网络，已经超出现有 ion template 覆盖范围。

## 当前状态判断

### 1. channel.Markov 的能力边界

`channel/_base.py` 中的 `Markov` 目前只覆盖：

- 一个 conserved pool
- 一个 dependent/redundant state
- 概率态之间的转移图

它适合 channel 概率动力学，不适合 ion 侧的浓度-缓冲-pump 反应网络。

### 2. ion.DynamicNernstIon 的能力边界

`ion/_base.py` 中的 `DynamicNernstIon` 目前只覆盖：

- 一个动态 `Ci`
- 从 `Ci` 动态计算 `E`
- 可选读取 `total_current`
- 导数接口本质上还是 `dCi/dt = f(Ci, V, total_current)`

它可以覆盖单 ODE 型钙池，但不能直接表达多状态 `KINETIC state { ... }` 机制。

### 3. multi-compartment runtime 的现状

当前 `Cell` runtime 侧的 lowering 重点仍是 `channel`：

- `mech.Channel(...)` 已可真正安装成 runtime channel
- `mech.Ion(...)` 目前仍主要停留在 declaration 层
- multi-compartment runtime 默认只建固定的 `na/k/ca` 容器

这意味着 Cerebellum 的 `Ion_dyn` 机制即使写成了 ion class，也还缺少完整的 runtime 安装路径。

## 目标机制分类

### A. 简单钙池：先走单 ODE 路径

这类机制的特点是：

- `DERIVATIVE states`
- 只有一个主动态变量，例如 `cai`
- 通过 `ica` 或 `ical` 驱动
- 没有显式 reaction network

典型机制：

- `DCN/ion/CdpHVA_SU15_DCN.mod`
- `DCN/ion/CdpLVA_SU15_DCN.mod`

这类机制可以先基于现有 `DynamicNernstIon` 思路扩展导入。

### B. 复杂钙池：需要新的 reaction-network ion template

这类机制的特点是：

- `KINETIC state`
- 多个 buffer / pump / 中间状态
- 使用 `COMPARTMENT`
- 使用 `CONSERVE`
- 使用 `<<`
- 需要 `f_flux / b_flux`
- `INITIAL` 中存在一组稳态初始化和辅助函数

典型机制：

- `BC/ion/CdpStC_MA25_BC.mod`
- `SC/ion/CdpStC_RI21_SC.mod`
- `GoC/ion/CdpStC_MA20_GoC.mod`
- `GrC/ion/CdpCR_MA20_GrC.mod`
- `PC/ion/CdpCAM_MA24_PC.mod`

这类机制不是现有 `CalciumDetailed` 的小改，必须单独设计模板能力。

## 主要缺口

下面按“真正会阻塞导入”的顺序列出当前还需要克服的问题。

### 1. `mech.Ion(...) -> runtime ion` 还没有真正打通

当前 multi-compartment runtime 主要会安装 `category == "channel"` 的 density 机制。  
后续如果要把 Cerebellum 的 ion 机制用于 `Cell`，必须先补上：

- `mech.Ion(...)` 的 runtime lowering
- ion runtime 实例化
- ion 与 channel 的实际绑定

否则 ion class 只能停留在 standalone 层，不能进入真正的 multi-compartment 执行路径。

### 2. ion 需要读取局部 CV 几何信息

这是当前最明确的基础缺口之一。

这些 mod 中实际用到的不是“只要半径或只要体积”这么简单，而是：

- `diam`
- `parea = PI * diam`
- `vrat`
- `dsq = diam * diam`
- `dsqvol = dsq * vrat`

所以 ion 侧需要一个明确的局部几何 contract，至少应允许读取：

- 当前局部计算点对应 CV 的 `diam` 或 `radius`
- `area`
- 以及可由其派生的量

建议不要把这个问题做成“半径/体积二选一”，而是定义一个最小但稳定的 geometry contract。

### 3. 现有 ion template 只够支撑单 ODE，不够支撑 reaction network

对于 DCN 这类：

- `cai' = D - C`

可以继续沿 `DynamicNernstIon` 路线实现。

但对于 `CdpStC/CdpCR/CdpCAM` 这类：

- 多状态
- 多反应
- 多 compartment
- buffer / pump / source term 混合

需要新的 ion-side reaction-network template，而不是继续在 `DynamicNernstIon` 上硬堆。

### 4. `COMPARTMENT` 语义目前没有落点

这些 ion mod 不是普通 Markov channel。  
不同状态属于不同的几何尺度：

- `ca/mg/buffer` 在一个体积尺度里
- `pump/pumpca` 在膜面积尺度里

所以 `COMPARTMENT` 不只是装饰信息，而是：

- 反应速率缩放的一部分
- 单位换算的一部分
- 守恒方程的一部分

这必须进入新的 reaction-network template 语义层。

### 5. `<<` 需要单独定义语义

当前讨论里的判断需要修正成更精确的版本：

- `<<` 的确是单向的
- 但它不是现有 `channel.Markov` 单向边的直接特化

在这些 ion mod 中：

- `~ ca << expr`

表达的是对某个 species 的单向 source/sink 注入或抽走，而不是“从一个状态跳到另一个状态”的概率转移。

因此它应在新模板中作为单独的 reaction/source 语义处理。

### 6. `CONSERVE` 需要从 channel 概率守恒推广到 reaction-network 守恒

当前 `channel.Markov` 中，守恒的处理方式是：

- 一个守恒池
- 通过消掉一个 dependent state 来闭合系统

这个思路本身仍然可用，但需要推广到 reaction-network 模板中。  
建议规则固定为：

- 每个独立 `CONSERVE` 组只消掉一个方程
- 不引入通用 DAE 解法

需要注意的是：

- 从 Cerebellum 当前这批 `Ion_dyn` 机制看，显式 `CONSERVE` 最主要的是 `pump + pumpca = const`
- 所以“多守恒”是模板能力上要支持的扩展点
- 但它并不是当前第一阻塞项

真正先卡住导入的仍然是 runtime、geometry、`COMPARTMENT`、`<<` 与 reaction-network 模板本身。

### 7. `f_flux / b_flux` 还没有对应实现

例如：

- `ica_pmp = 2*FARADAY*(f_flux - b_flux)/parea`

这里依赖 reaction statement 的正反向通量别名。  
这不是现有 HH/Markov template 中已有的概念，需要在新的 reaction-network template 中明确定义。

### 8. 需要支持 alias / assigned 写回

这些 mod 不只是求状态导数，还会在 `KINETIC` 末尾做写回：

- `cai = ca`
- `mgi = mg`
- `icazz = nrvci`

所以模板还需要支持：

- 状态变量到 ion public field 的 alias 写回
- 诊断量 / assigned 变量的同步更新

### 9. 需要支持多个独立 calcium pool / 命名空间

这是 DCN 特别关键的结构问题。

DCN 并不是只有一个钙池，而是同时使用了：

- `ca`
- `cal`
- `call`

其中不同 channel / ion 机制分别使用不同的 `USEION` 名字。  
但当前 BrainCell 的 runtime 默认只有一个 `ca` 容器，这会导致：

- 多个 pool 无法独立绑定
- channel 可能读错 ion payload

因此后续不能只做“单一 calcium pool”，而要支持独立命名空间。

### 10. 额外只读输入变量需要单独判定

例如 GoC 的：

- `USEION nrvc READ nrvci`
- `icazz = nrvci`

目前看它更像诊断量透传，而不是主钙动力学的一部分。  
因此默认策略应是：

- 保留这类额外 read-only 输入
- 首轮按诊断量处理
- 不让其阻塞主 reaction-network 模板设计

## 对当前三个判断的澄清

### 1. “需要一个通道从 Ion 去读取当前局部计算点对应的 CV 尺寸信息”

结论：是的，而且这是必需项。  
但建议不要只盯住“半径”或“体积”中的某一个，而是定义一组最小几何上下文，使 ion 可自行派生所需量。

### 2. “多个守恒，之前 channel 里面 markov 都是单个守恒”

结论：方向是对的。  
后续 reaction-network template 也应采用“一个守恒里消掉一个方程”的方式处理。  
但对当前这批 Cerebellum `Ion_dyn` 来说，它不是最先要解决的问题。

### 3. “`<<` 貌似只是单向的，之前 markov 的一个特化单向实施而已”

结论：前半句对，后半句不够准确。

- `<<` 是单向的
- 但它更像 species source/sink
- 不是现有 `channel.Markov` 单向 state transition 的直接复用

## 建议的实施顺序

### 第一步：补 `mech.Ion -> runtime ion` 骨架

先打通：

- `mech.Ion(...)` 的 runtime lowering
- ion runtime 安装
- ion 与 channel 的绑定路径

这是 multi-compartment 场景下的基础前置条件。

### 第二步：定义 ion 可读取的局部 geometry contract

至少让 ion 在当前局部点能读取：

- `diam` / `radius`
- `area`
- 必要的局部几何上下文

并明确这些量的来源是当前 CV，而不是全 cell 全局值。

### 第三步：先导入 DCN 的简单钙池

优先做：

- `CdpHVA_SU15_DCN`
- `CdpLVA_SU15_DCN`

原因：

- 它们是单 ODE
- 不依赖复杂 reaction-network
- 可以先把 ion runtime + geometry + current coupling 路径跑通

### 第四步：设计并实现 ion-side reaction-network template

该模板至少需要支持：

- `KINETIC`
- `<->`
- `<<`
- `COMPARTMENT`
- `CONSERVE`
- `f_flux / b_flux`
- `INITIAL`
- alias write-back

### 第五步：先落地最小复杂版 `CdpStC`

先从较小反应图开始：

- `BC/SC/GoC` 的 `CdpStC`

因为它们比 `CdpCR/CdpCAM` 更适合作为 reaction-network template 的第一批验证对象。

### 第六步：扩展到 `CdpCR` 与 `CdpCAM`

在 `CdpStC` 路径稳定后，再扩到：

- `GrC` 的 `CdpCR`
- `PC` 的 `CdpCAM`

这一步主要验证更大反应图与更多 buffer / intermediate states 的表达能力。

### 第七步：最后补 GoC 的附加读入与整体验证

包括：

- `nrvci -> icazz` 这种附加诊断输入
- 全套 `Ion_dyn` 机制的回归验证

## 默认假设

为避免后续计划反复改写，这里固定几个默认假设：

- 后续工作目标包含 multi-compartment `Cell` runtime，不只做 standalone ion 类
- 多 calcium pool 命名空间现在就设计，不延后
- `CONSERVE` 采用“每组消掉一个 dependent state”的实现策略
- GoC 中的 `nrvci` 首轮按诊断量处理

## 总结

当前 Cerebellum 钙 ion 导入的真正难点，不是单独某一个 `mod` 文件，而是缺少一整层 ion-side runtime 与 reaction-network 语义支持。

因此后续实现应按下面主线推进：

1. 先打通 ion runtime lowering
2. 再补 geometry contract
3. 先导入 DCN 的简单钙池
4. 再实现 reaction-network ion template
5. 再逐步导入 `CdpStC`、`CdpCR`、`CdpCAM`

这个顺序可以尽量减少返工，并把“简单可验证路径”和“复杂模板扩展路径”拆开推进。
