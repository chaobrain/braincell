# BrainCell Project TODO

BrainCell 用带物理单位的形态、机制和事件声明构建可微分的细胞与网络模型。
本页按模块列出已有能力、后续目标和主要缺口；任务拆分、下一步与验收细节由各模块 TODO 维护。
系统职责与数据流见 [系统总览](architecture/current/system-overview.md)。

## 进度口径

- `[x]` **已完成**：沿用 shipped 口径，实现、相关测试、设计及适用示例已随提交验收。
- `[~]` **部分完成 / 实施中 / 待提交验收**：已有能力或正在实现，具体缺项及实施、提交阶段随条目说明；待提交验收表示工作区实现与验证已完成。
- `[ ]` **待讨论 / 讨论中 / 已确认待实施**：问题待分析、方案正在讨论，或方案和验收边界已确定、等待实现。

Current 描述工作区行为，条目状态按其所列范围判断。详细规则见
[进度维护规范](AGENTS.md#全局-todo-规格)；历史条目及原始验证口径见
[整理前快照](../specs/2026-09-07-design-todo-snapshot.md)。

## 模块里程碑

### Architecture：系统分工与公共约定

协作入口：[Architecture TODO](architecture/TODO.md)。

- [x] **数据与状态分工**：形态和机制声明经 Cell 构建为运行时状态，Network 协调事件与推进，Trainable 管理参数映射。 [系统总览](architecture/current/system-overview.md)
- [~] **Python 支持覆盖**：现有 CI 主要测试 3.13，classifiers 声明 3.11 至 3.14；需要扩展测试矩阵或调整支持声明。 [版本覆盖](architecture/TODO.md#当前需要推进的事项)
- [ ] **公共接口命名**：统一声明类型与运行时基类的区分方式，并明确公共导出、内部路径及兼容策略。状态：**待讨论**。 [接口一致性](architecture/proposals/interface-consistency.md)

### Morph：形态构造与编辑

协作入口：[Morph TODO](morph/TODO.md)。

- [x] **分支几何**：通过长度、半径或三维采样点构造 Branch，计算长度、面积和体积。 [Branch 几何](morph/current/api.md#branch-几何)
- [x] **树构建与连接**：构造 Morphology，在指定位置连接分支，保留父子关系和连接方向。 [树连接](morph/current/api.md#morphology-与连接)
- [x] **查询、统计与复制**：提供树遍历、分支视图、路径及几何指标；通过独立复制创建可修改的形态副本。 [查询与指标](morph/current/api.md#查询视图和指标)
- [ ] **子树编辑**：支持删除、拼接和替换子树，需确定分支身份、引用及连接方向的保持规则。状态：**待讨论**。 [编辑事项](morph/TODO.md#当前需要推进的事项)
- [ ] **几何变换**：支持平移、旋转、缩放和主轴对齐，并更新依赖几何的指标及派生缓存。状态：**待讨论**。 [变换事项](morph/TODO.md#当前需要推进的事项)
- [ ] **公共导出**：评估在顶层入口之外，同时从 morph 子包导出 Branch 和 Morphology 的收益与依赖影响。状态：**待讨论**。 [导出事项](morph/TODO.md#当前需要推进的事项)

### IO：形态读写与外部数据

协作入口：[IO TODO](io/TODO.md)。

- [x] **SWC 读写与结构往返**：导入形态及诊断报告，处理 soma 和分支连接，并将树写回 SWC；已有共享端点、反向分支等回归。 [SWC API](io/current/api.md#swc-读取与检查)
- [x] **形态 checkpoint**：以自包含的 .bcm 格式保存和恢复形态，配有调用示例。 [Checkpoint](io/current/api.md#checkpoint)
- [x] **NeuroMorpho 检索与下载**：提供便捷加载、客户端检索、下载及缓存，返回形态和元数据。 [NeuroMorpho](io/current/neuromorpho.md)
- [~] **ASC 几何与标记覆盖**：已有树和元数据导入；spine、轮廓 soma 及多树案例仍需补齐处理与验证。 [ASC 覆盖](io/TODO.md#当前需要推进的事项)
- [~] **自动几何对照**：已有 NEURON 形态差异工具，下一步将 NeuroMorpho 指标比较整理为固定数据集、单位和容差的回归。 [对照事项](io/TODO.md#当前需要推进的事项)
- [ ] **NeuroML2 导入**：将 cell 和 segment-group 映射为 Morphology；当前 reader 仍是 stub，最小映射与验收样本待确定。状态：**待讨论**。 [导入事项](io/TODO.md#当前需要推进的事项)

### Filter：区域、位点与空间参数

协作入口：[Filter TODO](filter/TODO.md)。

- [x] **区域与集合运算**：按分支及形态标签选择区间，组合交、并、差，并缓存解析结果供 Cell 使用。 [区域 API](filter/current/api.md#区域)
- [x] **离散位点与采样批次**：选择根、末端、分叉或显式位置，支持区域内均匀及随机取点和批次组合。 [位点 API](filter/current/api.md#位点和批次)
- [x] **连续采样与空间参数**：按长度、面积、体积测度和 density 采样，空间 callable 可读取形态上下文与指标。 [连续采样](filter/current/sampling.md)、[空间参数](filter/current/spatial-callable-parameters.md)
- [ ] **半径、距离与子树区域**：扩展阈值切分、路径距离、欧氏距离及子树选择；需确定单位、边界和树修改后的缓存规则。状态：**待讨论**。 [区域扩展](filter/TODO.md#当前需要推进的事项)
- [ ] **区域锚点与固定步长取点**：实现 RegionAnchors 和 StepSamples 的区域相对位置、端点及重复值规则；两者当前均为预留表达式。状态：**待讨论**。 [位点扩展](filter/TODO.md#当前需要推进的事项)
- [ ] **随机流一致性**：让旧 RandomSamples 的 NumPy 局部流与 BrainState 随机上下文方案衔接。状态：**待讨论**。 [随机上下文](network/proposals/random-context.md)

### Mech：机制声明与运行时契约

协作入口：[Mech TODO](mech/TODO.md)。

- [x] **机制声明与注册**：将 Channel、Ion、Synapse 等模型注册为可解析声明，供 paint/place 构建实际机制。 [声明 API](mech/current/api.md)
- [x] **刺激与事件输入**：提供电流钳、电压钳和事件输入契约，将声明交给 Cell 和 Network 绑定执行。 [声明到运行时](mech/current/architecture.md)
- [~] **Junction 电耦合**：已有占位声明，仍缺 partner 身份、对称配对及电压方程中的电流贡献。 [接线缺口](mech/proposals/runtime-extensions.md)
- [ ] **参数单位诊断**：在构造或 paint 阶段定位错误参数、物理维度和声明位置，所需元数据与 Channel/Ion 共同设计。状态：**待讨论**。 [参数诊断](mech/TODO.md)
- [ ] **旧 Probe 迁移**：核对旧字段声明与现行 observe 的对应关系，确定错误校验和迁移方式。状态：**待讨论**。 [迁移事项](mech/TODO.md)
- [ ] **模型验证框架**：从现有 NEURON 比较例子提取可复用的电压钳、电流钳及误差验收流程。状态：**待讨论**。 [机制验证](mech/proposals/runtime-extensions.md#机制生成与验证)
- [ ] **NMODL 生成器**：研究以 registry 为目标的机制代码生成，先确定最小语法集和标准模板的映射。状态：**待讨论**。 [生成方向](mech/proposals/runtime-extensions.md#机制生成与验证)

### Channel：通道模板与模型目录

协作入口：[Channel TODO](channel/TODO.md)。

- [x] **HH 与 Markov 模板**：统一门状态、转移、生命周期和速率单位，在类定义时检查名称、速率形式及依赖；Markov 必须显式指定 dependent_state。 [模板约束](channel/current/template-invariants.md)
- [x] **电流与裁剪策略**：提供 ohmic/GHK 驱动力、Q10 辅助函数及显式裁剪配置，复用各模型共有的计算。 [模板 API](channel/current/api.md)
- [x] **通道目录与注册**：提供钠、钾、钙、HCN、混合离子和漏通道家族，包含 PC MA2024 等导入模型及相邻测试。 [模型入口](channel/current/api.md)
- [~] **GHK 与温度审计**：模板和部分模型已采用共享驱动力及温度路径，仍需逐家族核对原模型要求与参数来源。 [审计事项](channel/TODO.md#当前需要推进的事项)
- [~] **逐模型精度与刚性验收**：在已有定向测试和模型比较基础上，补全目录级 MOD 对照与 dt/solver 收敛矩阵。 [验证事项](channel/TODO.md#当前需要推进的事项)
- [ ] **参数单位元数据**：确定参数维度的维护位置，与 Mech 的声明校验及错误定位衔接。状态：**待讨论**。 [元数据事项](channel/TODO.md#当前需要推进的事项)
- [ ] **门变量命名**：为 p/q 与模型自定义名称设计兼容路径，保留状态读写的可追溯关系。状态：**待讨论**。 [命名事项](channel/TODO.md#当前需要推进的事项)
- [ ] **氯通道**：在 Chloride 离子家族确定后，补齐相应电流与反转电位的模型。状态：**待讨论**。 [氯通道事项](channel/TODO.md#当前需要推进的事项)

### Ion：离子状态与浓度动力学

协作入口：[Ion TODO](ion/TODO.md)。

- [x] **固定值与 Nernst 模型**：提供钠、钾、钙等固定反转电位或浓度驱动的反转电位，以及共享的子通道生命周期。 [固定值与 Nernst](ion/current/api.md#固定值与-nernst-模型)
- [x] **KineticIon 反应模型**：声明物种、反应、source、factor 和守恒关系，接入 Cell 的浓度状态及电流输入。 [KineticIon](ion/current/kinetic-ion-api.md)
- [~] **动态钙浓度**：已有 Detailed、FirstOrder 和小脑动力学模型；CalciumFirstOrder 的默认 alpha/beta 缺少正确单位转换，仍不能完成对应导数路径。 [具体缺项](ion/current/api.md#动态钙浓度)
- [ ] **动态钠、钾浓度**：扩展内外浓度、泵和电流驱动模型，为活动依赖的离子积累提供状态方程。状态：**待讨论**。 [家族扩展](ion/TODO.md#当前需要推进的事项)
- [ ] **Chloride 家族**：确定固定与动态氯反转电位，以及 GABA 模型所需的浓度方程。状态：**待讨论**。 [氯离子事项](ion/TODO.md#当前需要推进的事项)
- [ ] **外部电流一致性**：逐模型核对 include_external、总电流缓存及电流到浓度导数的转换。状态：**待讨论**。 [电流审计](ion/TODO.md#当前需要推进的事项)
- [ ] **模型来源补全**：补齐共享文献表中未核实的归因和模型版本，为通道及离子对照提供依据。状态：**待讨论**。 [来源事项](ion/TODO.md#当前需要推进的事项)

### Cell：声明、离散与细胞运行

协作入口：[Cell TODO](cell/TODO.md)。

- [x] **声明与空间离散**：通过 morphology、CVPolicy 和 paint/place 构建电缆、密度机制与点机制布局。 [构造与 CVPolicy](cell/current/api.md#cell-构造)
- [x] **CV 与 point 状态**：密度机制使用 CV 空间，点机制使用 point 空间；population 轴支持多维形状，并保留末尾空间轴。 [状态布局](cell/current/architecture.md#状态布局)
- [x] **生命周期与直接运行**：Cell 直接初始化、重置和按固定步长推进，支持连续运行及带时间范围的结果。 [生命周期](cell/current/api.md#生命周期)、[运行与结果](cell/current/api.md#运行与结果)
- [x] **视图、观测与状态读写**：按 population、区域和机制选择状态，读取轨迹或 buffer，并记录固定步长刺激结果。 [Views](cell/current/views.md)、[状态查询](cell/current/api.md#运行时查询)
- [x] **离子电流快照与调度**：staggered 可读取步首总离子电流，并选择 family 或 integration 的机制更新顺序。 [调度契约](cell/current/architecture.md#离子电流快照与调度)
- [~] **显式 solver 的边界输入**：显式路径已推进 CV 电压，消元时仍遗漏端点刺激和突触的等效贡献；修复方案正在讨论。 [边界输入](cell/proposals/explicit-solver-boundary-inputs.md)
- [ ] **Single 与多室统一**：从单 branch、特殊单 CV policy 和位点语义开始，使单方程 ODE 模型兼容 Cell；形态等效与积分路径仍需比较。状态：**讨论中**。 [统一提案](cell/proposals/single-multi-compartment-unification.md)
- [ ] **生命周期与查询风格**：明确 reset/reset_state 的命名，并统一缓存查询与触发构建操作的表达。状态：**待讨论**。 [接口事项](cell/TODO.md#当前需要推进的事项)

### Quad：积分方法与电压求解

协作入口：[Quad TODO](quad/TODO.md)。

- [x] **积分注册与通用 ODE**：提供显式 RK、隐式和指数方法，通过统一步函数及阶段协议推进模型状态。 [积分 API](quad/current/api.md)
- [x] **Staggered 与 DHS**：按机制和电压阶段推进，使用树结构求解包含边界与分叉约束的电缆系统。 [求解架构](quad/current/architecture.md)
- [~] **精度与性能对照**：已有算法回归、收敛测试及电缆数值对照，仍缺 Mainen/Hay/L5PC 等标准模型上的统一精度、编译和计时比较。 [对照事项](quad/TODO.md)
- [~] **显式边界输入**：已有 CV 导数推进，仍需与 Cell 共同补入边界刺激和突触反馈；修复方案正在讨论。 [边界方案](cell/proposals/explicit-solver-boundary-inputs.md)
- [ ] **Single 积分路径**：在 single 统一中比较共享边界装配与独立 ODE 积分，核对等价性和 solver 一致性。状态：**讨论中**。 [路径方案](cell/proposals/single-multi-compartment-unification.md)
- [ ] **自适应步长**：基于 embedded RK 误差估计推进，同时处理事件时间和记录采样对齐。状态：**待讨论**。 [步长事项](quad/TODO.md)

### Synapse：突触动力学与状态

协作入口：[Synapse TODO](synapse/TODO.md)。

- [x] **指数突触模型**：ExpSyn 和 Exp2Syn 提供事件驱动的电导动力学、电流计算及状态重置。 [模型 API](synapse/current/api.md)
- [x] **Cell-owned 突触状态**：突触状态由目标 Cell 持有，连接投递事件并乘权，空间布局与生命周期由 Cell 管理。 [状态与事件](synapse/current/architecture.md)
- [~] **预定事件执行优化**：两轮算法实验与完整模型四层对照已验证 CPU/GPU 查询、直接投递、基线开销及代表性真实 Cell/Network 与权重/tau 梯度；生产窗口生命周期和自动选型仍缺合同。 [算法实测](../../benchmarks/performance/synapse_events/results/scheduled-event-delivery.md)、[四层对照](../../benchmarks/performance/synapse_events/results/scheduled-event-controls.md)、[方案](synapse/proposals/event-delivery-optimization.md)
- [ ] **内部动力学可塑性**：通过新的 Synapse 模型表达释放或内部状态的可塑变化，与只更新 Connection weight 的规则区分。状态：**讨论中**。 [可塑性方案](network/proposals/connection-plasticity.md)
- [ ] **模型验证覆盖**：在已有初始化、衰减和事件测试之外，扩展事件序列、时间常数及电压驱动力的参照对照。状态：**待讨论**。 [验证事项](synapse/TODO.md)

### Network：组网、事件与结果

协作入口：[Network TODO](network/TODO.md)。

- [x] **Population 与网络生命周期**：注册 Cell population 和事件源，协调初始化、重置与连续运行。 [Network API](network/current/api.md)
- [x] **端点配对与连接**：通过显式索引或采样配对建立具名连接，管理连接权重及异构延迟。 [配对](network/current/pairing.md)、[连接](network/current/connections.md)
- [x] **事件路由与记录**：调度延迟投递，聚合规则采样和稀疏事件，保持目标 Cell 的状态归属。 [事件架构](network/current/architecture.md)、[Recording](network/current/recording.md)
- [ ] **预定事件执行计划接入**：依据 Synapse 的 CPU/GPU 查询实验选型，定义 Network 准备、续跑和参数失效合同。状态：**讨论中**。 [实验与阶段](synapse/proposals/event-delivery-optimization.md)
- [ ] **统一随机上下文**：用 BrainState 随机区域管理网络、source 和 pairing 的默认随机流，确定局部子流与迁移语义。状态：**讨论中**。 [随机方案](network/proposals/random-context.md)
- [ ] **Connection 权重可塑性**：根据上下游 spike 或电压维护规则状态并更新 weight，确定信号绑定及更新顺序。状态：**讨论中**。 [可塑性方案](network/proposals/connection-plasticity.md)
- [ ] **大规模事件与配对**：研究稀疏 delay slots 和分块 endpoint generators，控制静态 shape、调度开销及内存。状态：**待讨论**。 [运行时扩展](network/proposals/runtime-extensions.md)
- [ ] **可学习拓扑**：定义结构变化的状态、梯度及重编译协议。状态：**待讨论**。 [拓扑方案](network/proposals/runtime-extensions.md#i-10-trainable-topology)
- [ ] **网络 batch**：确定网络 batch 的连接、事件和状态轴语义。状态：**待讨论**。 [Batch 方案](network/proposals/runtime-extensions.md#network-batch-runtime)

### Optim：参数映射与训练验证

协作入口：[Optim TODO](optim/TODO.md)。

- [x] **参数 source 与映射**：提供 direct、共享 scale、parameterized 映射及分组，保留物理单位、共享关系和 reset/materialization 语义。 [参数 API](optim/current/api.md)
- [x] **Channel 与 Ion 参数训练**：按构造签名发现候选参数，支持 Cell 内通道和离子参数绑定；已有单参数教学拟合及相关回归。 [支持度](optim/current/parameter-support.md)、[结果](optim/current/results/parameter-learning.md)
- [x] **训练与诊断实验**：已有固定参数 rollout 的 BPTT/RTRL、分阶段拟合、搜索、敏感度诊断及刺激设计实验，梯度核心位于 `braincell.experimental.optim`，使用流程在 `examples/optim/`。 [实验工作流](optim/current/experimental-workflows.md)
- [x] **Synapse 与 Network 参数训练**：支持突触参数、静态连接 weight、检测阈值和网络 roots 聚合，已通过 CPU 梯度及拟合验收并提交。 [接口范围](optim/current/api.md#synapse-connection-network)、[验证记录](optim/current/results/synapse-network-learning.md#提交验收)
- [~] **组合精度与性能验证**：已有多 CV、population、CPU/GPU 等单项结果；新事件网络的 GPU、多 CV 与多 population 组合及 checkpoint 比较仍缺证据。 [验证缺口](optim/proposals/roadmap.md#验证缺口)
- [ ] **可塑性参数训练**：明确动态 weight 初值、规则参数与运行中状态的关系，连接可塑性调度和梯度验收。状态：**讨论中**。 [训练提案](optim/proposals/connection-plasticity.md)
- [ ] **训练自动恢复**：在已有历史 archive 和诊断基础上，设计 plateau、SGDR、perturb 控制器及恢复协议。状态：**讨论中**。 [恢复方案](optim/proposals/training-recovery.md)
- [ ] **参数范围与公共训练协议**：确定 Cell 初值和 cable 参数的 owner，以及可复用训练协议、稳定 grouping 和持久化边界。状态：**待讨论**。 [后续方向](optim/proposals/roadmap.md)
- [ ] **Rollout 内参数更新**：定义逐步更新参数时的参数历史、状态演化及梯度含义。状态：**待讨论**。 [研究方向](optim/proposals/roadmap.md#rollout-内更新参数)

### Reduction：约化模型接入

协作入口：[Reduction TODO](reduction/TODO.md)。

- [x] **可替换的 Cell 约化运行时**：通过 ReductionModel 接入 Cell 的生命周期、输入输出和网络运行，已有示例及回归。 [接入指南](reduction/current/model-integration-guide.md)
- [ ] **DBNN 数据、训练与部署**：沿用现有挂载契约，确定数据布局、训练规模、模型资产与分阶段验收。状态：**讨论中**。 [DBNN 提案](reduction/proposals/DBNN-plan.md)

### Vis：形态、拓扑与结果展示

协作入口：[Vis TODO](vis/TODO.md)。

- [x] **形态与空间数据绘图**：提供 2D/3D 几何和树形布局，按 branch、segment 或采样点着色，并高亮区域与位点。 [绘图 API](vis/current/api.md)
- [x] **Cell 拓扑与动态结果**：展示 branch、CV、point 拓扑、时间轨迹、多模型比较和动画。 [展示能力](vis/current/visualization.md#当前支持什么)
- [x] **交互与导出**：按后端提供拾取、图片、动画及 HTML 导出；后端之间的交互能力有明确差异。 [后端支持](vis/current/visualization.md#后端与数据)
- [~] **渲染回归**：已有布局、场景和图元断言，仍缺代表性像素基线及执行图像比较的 CI。 [验证现状](vis/current/visualization.md#验证现状)
- [ ] **迁入 BrainTools**：讨论公共可视化模块归属、数据入口、简单绘图与 GUI，以及旧调用的兼容方案。状态：**讨论中**。 [迁移提案](vis/proposals/braintools-migration.md)

## 跨模块依赖与阻塞

- [ ] **单方程 ODE 兼容到 Cell**：Cell 牵头确定特殊 policy 与 place 语义，Filter 配合位点表达，Quad 比较积分路径；先确定单 branch 情形，再讨论分叉形态等效。状态：**讨论中**。 [Cell 统一方案](cell/proposals/single-multi-compartment-unification.md)
- [~] **显式积分完整处理边界输入**：Cell 牵头将 point 刺激和 Synapse 电流传入消元后的 RHS，Quad 配合边界约束及完整五行系统对照；现有完整装配位于 staggered 路径。 [边界输入问题](cell/proposals/explicit-solver-boundary-inputs.md)
- [ ] **统一随机区域**：Network 牵头，source、pairing 与 Filter 配合默认随机流和局部子流设计，使区域内自定义随机调用也能统一管理。状态：**讨论中**。 [随机上下文](network/proposals/random-context.md)
- [ ] **可塑性与参数训练**：Network 牵头明确 Connection weight 更新，Synapse 管理内部动力学，Optim 绑定规则参数及初值；需确定信号读取和更新顺序。状态：**讨论中**。 [可塑性](network/proposals/connection-plasticity.md)、[训练提案](optim/proposals/connection-plasticity.md)
- [ ] **空间编辑后的缓存一致性**：Morph 牵头定义结构和几何修订，Filter、Cell 配合选择结果与离散缓存的失效，保证编辑后重新构建使用新形态。状态：**待讨论**。 [Morph](morph/TODO.md)、[Filter](filter/TODO.md)
- [ ] **单位诊断与模型验证**：Mech 牵头建立声明诊断及验证框架，Channel/Ion 提供参数维度、模型来源和参考机制；已有模型比较需整理成可复用验收流程。状态：**待讨论**。 [Mech](mech/TODO.md)、[Channel](channel/TODO.md)、[Ion](ion/TODO.md)
- [ ] **量化 GABA 与氯反转电位**：Ion 牵头建立 Chloride 状态和反转电位，Channel、Synapse 配合电流及事件模型；当前仍缺氯离子家族。状态：**待讨论**。 [Ion](ion/TODO.md)、[Channel](channel/TODO.md)
- [~] **大规模精度与性能结论**：Optim 牵头训练组合验证，Quad 提供 solver/dt 对照，Cell、Network 配合多 CV 与多 population 场景；GPU 和 checkpoint 的组合证据尚不完整。 [Optim 验证缺口](optim/proposals/roadmap.md#验证缺口)、[Quad](quad/TODO.md)
- [~] **Python 支持范围一致**：Architecture 牵头，CI 配置与依赖维护配合；在现有 3.13 测试基础上扩展矩阵或收窄 3.11 至 3.14 的声明。 [版本覆盖](architecture/TODO.md)

## 专题与维护入口

- [小脑示例进度](../../validation/neuron/cerebellum-import-progress.md)：具体模型导入、PC 装配与数值比较。
- [共享 Ion/Channel 文献表](ion/references/ion-channel-bibliography.md)：模型来源、版本和归因证据。
- [Design 规范](AGENTS.md)：文档职责及进度维护；提交前检查遵循 [仓库约定](../../AGENTS.md#design-code-and-examples)。
