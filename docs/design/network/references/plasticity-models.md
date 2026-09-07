# Plasticity Models Reference

本文比较可塑性模型需要的信号、动态状态与修改对象，为
[Connection 可塑性提案](../proposals/connection-plasticity.md)的公共接口和架构提供依据。
模型清单、横向延伸和模型数值例子集中在本页；BrainCell 的职责决定及共享语义以提案为准。

采用状态：参考与候选延伸，尚未移植为 BrainCell 可塑性模型。下文“对 BrainCell 的启示”和
“建议归属”是本项目的设计推论，不是外部库对对象边界的规定。

## BrainPy 权重规则

参考基线为 2026-09-07 核查的本地 BrainPy 2.7.8 与 brainpy.state 0.0.4 源码。
下表以 BP 表示 BrainPy，BS 表示 brainpy.state；BS 条目来自其 NEST 兼容模型。
链接指向官方在线文档，在线版本可能继续更新。它们是方程与模型组织的参考，尚未在 BrainCell
采用；外部类名中的 `synapse` 不决定本项目的职责归属，也不表示其运行时可以直接接入 BrainCell。

| 规则家族 | 参考实现 | 输入与核心状态 | 对 BrainCell 的启示 |
| --- | --- | --- | --- |
| 成对 STDP | BP [STDP_Song2000][song]；BS [stdp_synapse][pair] | pre/post spike；双侧 trace、weight | 基础时序窗口；区分加性与权重依赖更新 |
| 非线性权重依赖 | BS [stdp_pl_synapse_hom][power]、[jonke_synapse][jonke] | pre/post spike；trace、weight | 分别参考幂律与指数形式，不能仅用一个学习率代表所有规则 |
| 最近邻配对 | BS [stdp_nn_pre_centered_synapse][nn-pre]、[stdp_nn_restr_synapse][nn-restr]、[stdp_nn_symm_synapse][nn-symm] | pre/post spike；最近事件及配对历史 | 配对策略是模型语义，同一 spike 序列可以得到不同更新 |
| Triplet STDP | BS [stdp_triplet_synapse][triplet] | pre/post spike；双侧快慢 trace | 表达三脉冲相互作用及频率依赖 |
| 电压依赖 | BS [clopath_synapse][clopath] | pre 活动、post 电压及滤波量；pre trace、weight | 多区室模型必须明确电压位置；外部模型还依赖目标神经元提供电压历史 |
| 多巴胺调制 STDP | BS [stdp_dopamine_synapse][dopamine] | pre/post spike、多巴胺信号；eligibility 与调制 trace | 局部相关性先形成资格迹，第三因子决定其如何改变 weight |
| 抑制性稳态规则 | BS [vogels_sprekeler_synapse][vogels] | pre/post spike；活动 trace、抑制性 weight | 学习目标可为活动稳态，不能把所有 STDP 都视为兴奋性因果增强 |
| 树突预测学习 | BS [urbanczik_synapse][urbanczik] | pre 活动、树突预测误差；pre trace、滤波学习信号 | 需要 Cell 提供明确的预测误差信号，单个 post spike 端口不够 |

[stdp_synapse_hom][pair-hom] 将同质可塑性参数与逐连接状态区分，可用于讨论参数共享，
不另列为生物学规则家族。[stdp_facetshw_synapse_hom][hardware] 则提供硬件约束下的离散权重、
累积量与周期更新参考，作为扩展条目，不作为通用模型默认值。

## BrainPy 释放模型

版本基线同上。BS 表中的 Tsodyks 等模型在原库按连接模型实现；这里建议借鉴其内部方程，
将独立释放位点的状态放入 BrainCell Syn，而非照搬外部对象边界。

| 模型家族 | 参考实现 | 输入与核心状态 | 对 BrainCell 的启示 |
| --- | --- | --- | --- |
| 短时抑制 | BP [STD][bp-std]；BS [STD][bs-std] | pre 事件；资源比例 x | 资源消耗与恢复改变后续释放量 |
| 易化与抑制 | BP [STP][bp-stp]；BS [STP][bs-stp] | pre 事件；释放变量 u、资源比例 x | 易化和抑制可以同时存在，并有不同恢复时间 |
| Tsodyks 系列 | BS [tsodyks_synapse][tsodyks]、[tsodyks2_synapse][tsodyks2]、[tsodyks_synapse_hom][tsodyks-hom] | pre 事件；利用率、资源状态及事件历史，具体变量依版本而异 | 比较资源方程、释放与消耗顺序；hom 变体另用于讨论参数共享 |
| 随机量子释放 | BS [quantal_stp_synapse][quantal] | pre 事件、随机抽样；可用释放位点与释放概率 | 有限位点的随机释放与恢复，不能用确定性的幅度缩放完全替代 |
| 囊泡池抑制 | BS [ht_synapse][ht] | pre 事件；池可用比例 P | Hill–Tononi 的恢复、释放、消耗顺序提供另一种简化模型 |

模型需要分别定义基线参数和动态状态，例如固定 `U` 与动态释放变量 `u`，以及固定最大电导
与动态受体数量。长期变化也可以由内部状态表达，不需要把所有响应变化都回写到 Connection weight。

## 延伸方向

以下是候选研究方向，不是上述版本已有 API 的清单，也不承诺首批全部实现。

| 方向 | 输入、状态及修改对象 | 建议归属与新增需求 |
| --- | --- | --- |
| Hebb / [Oja][oja] / [BCM][bcm] | pre/post 活动；相关性、活动平均或滑动阈值；修改 weight | Connection；需要定义 rate 或 spike 到活动估计的方式。本次源码核查未找到独立同名实现 |
| [突触缩放][scaling]与跨连接归一化 | post 活动或所选输入组统计；慢尺度状态；缩放各行 weight | 抽象规则放 Connection；跨行统计必须显式定义分组、归约和信号 owner，不能由某行任意修改其他行 |
| 一般三因子规则 | pre/post 活动形成 eligibility，奖励或教学信号调制更新 | Connection；将多巴胺实例延伸到显式调制信号，明确广播范围 |
| [钙驱动 LTP/LTD][calcium] | 活动引起的局部钙变化、阈值及效能状态 | 抽象钙 trace 只驱动 weight 时放 Connection；局部钙与释放、受体等耦合时由 Syn 表达 |
| 受体插入、移除与长期释放变化 | 局部活动或生化信号；受体数量、可释放资源或利用率的慢变量 | Syn 的模型延伸；给出具体方程后再确定依赖的 Ion/Cell 信号 |
| [BTSP][btsp] | pre 活动历史、树突平台电位相关教学信号；秒级历史或资格迹 | 更新 weight 的抽象规则放 Connection；平台电位由 Cell 产生，需指定树突位置和信号定义 |

这些方向说明信号不能只抽象成两个 spike 布尔值。与此同时，读取钙或电压并不自动要求
Syn 拥有完整的钙或电压模型：要区分规则自己维护的现象学 trace、Cell/Ion 提供的物理观测量，
以及 Syn 内部生化状态。若显式模拟受体变化来实现突触缩放，其表达也应归入第二类。

## 模型数值例子

以下配置用于手工核对模型含义，不定义 BrainCell 的默认规则、公共 API 或更新时序。

### 成对 STDP

以成对 STDP 为最小例子：设 pre 在 0 ms 发放，post 在 10 ms 发放，无其他事件，初始 trace 为零，
本例忽略传输延迟，使用示意性的加性增强分支：

```text
Delta w = A_plus * exp(-(t_post - t_pre) / tau_plus)
```

取 `A_plus = 0.01 uS`、`tau_plus = 20 ms`，则 `Delta w ≈ 0.00607 uS`。
这些数值和配对条件仅用于解释模型，不是默认规则；权重尚未触及边界。Synapse 的电导衰减方程
不因此改变，后续事件的输入幅度由变化后的 weight 决定。

### 短时抑制

以资源消耗解释短时抑制，示意模型中的 `x` 是可用资源比例，`U` 是每次事件利用的比例，
二者无量纲：

```text
between events: dx/dt = (1 - x) / tau_rec
on one event:   Delta g = w * U * x_before
                x_after = x_before * (1 - U)
```

`tau_rec` 是恢复时间，`w` 和 `Delta g` 的单位为电导。固定 `w = 1 nS`、`U = 0.5`，
初始 `x = 1`；忽略两次紧邻但先后发生的事件之间的恢复，电导增量依次为 `0.5 nS` 与
`0.25 nS`。变化来自 Syn 的资源消耗，Connection weight 保持不变；长时间休息后资源恢复。
这是逐事件“先释放、后消耗”的说明模型，同一步事件的表示方式仍需另外确定。

同样设每次事件利用一半资源、两次事件间忽略恢复：A、B 各自拥有初始 `x = 1` 的资源池时，
先后激活的释放因子都是 0.5；若共用一个池，A 消耗后 B 的释放因子只有 0.25。
该对比用于核对[提案中的状态共享语义](../proposals/connection-plasticity.md#两类的组合与共享)。

[song]: https://brainpy.readthedocs.io/apis/generated/brainpy.dyn.STDP_Song2000.html
[pair]: https://brainx.chaobrain.com/brainpy-state/apis/generated/brainpy.state.stdp_synapse.html
[power]: https://brainx.chaobrain.com/brainpy-state/apis/generated/brainpy.state.stdp_pl_synapse_hom.html
[jonke]: https://brainx.chaobrain.com/brainpy-state/apis/generated/brainpy.state.jonke_synapse.html
[nn-pre]: https://brainx.chaobrain.com/brainpy-state/apis/generated/brainpy.state.stdp_nn_pre_centered_synapse.html
[nn-restr]: https://brainx.chaobrain.com/brainpy-state/apis/generated/brainpy.state.stdp_nn_restr_synapse.html
[nn-symm]: https://brainx.chaobrain.com/brainpy-state/apis/generated/brainpy.state.stdp_nn_symm_synapse.html
[triplet]: https://brainx.chaobrain.com/brainpy-state/apis/generated/brainpy.state.stdp_triplet_synapse.html
[clopath]: https://brainx.chaobrain.com/brainpy-state/apis/generated/brainpy.state.clopath_synapse.html
[dopamine]: https://brainx.chaobrain.com/brainpy-state/apis/generated/brainpy.state.stdp_dopamine_synapse.html
[vogels]: https://brainx.chaobrain.com/brainpy-state/apis/generated/brainpy.state.vogels_sprekeler_synapse.html
[urbanczik]: https://brainx.chaobrain.com/brainpy-state/apis/generated/brainpy.state.urbanczik_synapse.html
[pair-hom]: https://brainx.chaobrain.com/brainpy-state/apis/generated/brainpy.state.stdp_synapse_hom.html
[hardware]: https://brainx.chaobrain.com/brainpy-state/apis/generated/brainpy.state.stdp_facetshw_synapse_hom.html
[bp-std]: https://brainpy.readthedocs.io/apis/generated/brainpy.dyn.STD.html
[bs-std]: https://brainx.chaobrain.com/brainpy-state/apis/generated/brainpy.state.STD.html
[bp-stp]: https://brainpy.readthedocs.io/apis/generated/brainpy.dyn.STP.html
[bs-stp]: https://brainx.chaobrain.com/brainpy-state/apis/generated/brainpy.state.STP.html
[tsodyks]: https://brainx.chaobrain.com/brainpy-state/apis/generated/brainpy.state.tsodyks_synapse.html
[tsodyks2]: https://brainx.chaobrain.com/brainpy-state/apis/generated/brainpy.state.tsodyks2_synapse.html
[tsodyks-hom]: https://brainx.chaobrain.com/brainpy-state/apis/generated/brainpy.state.tsodyks_synapse_hom.html
[quantal]: https://brainx.chaobrain.com/brainpy-state/apis/generated/brainpy.state.quantal_stp_synapse.html
[ht]: https://brainx.chaobrain.com/brainpy-state/apis/generated/brainpy.state.ht_synapse.html
[oja]: https://pubmed.ncbi.nlm.nih.gov/7153672/
[bcm]: https://pubmed.ncbi.nlm.nih.gov/7054394/
[scaling]: https://pubmed.ncbi.nlm.nih.gov/9495341/
[calcium]: https://pubmed.ncbi.nlm.nih.gov/22357758/
[btsp]: https://pubmed.ncbi.nlm.nih.gov/28883072/
