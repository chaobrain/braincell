# DBNN 约化模型设计大纲

## 1. 目标

DBNN 是 BrainCell `reduction` 模块中的一种细胞约化模型。它以一个已经完成形态、
CV 离散、膜机制和突触机制配置的 detailed `Cell` 为教师模型，通过自动生成刺激数据并
拟合其胞体响应，得到可以低成本运行的简化模型。

第一版需要覆盖一条完整的约化链路：

1. 从 detailed Cell 的 NodeTree 确定 electrical points 和可训练的突触输入布局。
2. 自动或按用户策略生成突触刺激和训练数据。
3. 训练、验证并评估 DBNN。
4. 保存训练状态，导出可独立部署的 BrainCell 模型资产。
5. 将导出的模型作为 population 使用，并允许它与 detailed Cell 或其他 population 混合建网。

本文只确定上述能力的层次、职责和完成边界，不提前规定具体类名、函数签名、参数默认值、
checkpoint 格式或 Network 接口形式。

## 2. 模型定位与适用范围

### 2.1 DBNN 的基本结构

DBNN 保持两层、可解释的基本结构：

- 第一层为每个输入通道学习双指数时域响应核，用来表示该位置、该突触机制在胞体产生的 PSP。
- 第二层对所有通道响应进行线性求和，并加入不同通道之间的双线性交叉项，用来拟合树突
  对多突触输入的非线性整合。
- 模型输出一个胞体亚阈值电压；spike 由电压阈值上穿读出。

双线性层只描述不同通道之间的交互，不需要把单通道平方项作为第一版主模型的一部分。
具体的时域计算方式、参数存储形式和在线递推算法留到实现设计阶段确定。

参考：J. Ma, S. Li, D. Zhou, [Mapping Biological Neuron Dynamics into an
Interpretable Two-layer Artificial Neural Network](https://arxiv.org/abs/2305.12471)。

### 2.2 “通用”的边界

一次训练得到的 DBNN 对以下组合有效：

- 一个确定的 detailed Cell；
- 一次固定的 CV 离散结果；
- 由该离散结果生成的一次固定 NodeTree point 布局；
- 一组固定的突触原型；
- 训练数据覆盖的刺激强度、时间尺度和活动分布。

这里的“通用”是指：对这个 Cell 当前 NodeTree 中全部已物化 electrical points 建立稳定输入
映射，训练完成后可以把连接解析到对应 `point_id` 和突触原型，而不是只支持训练时临时抽取的
少量位置。

第一版不承诺以下能力：

- 在不同 morphology、Cell 类型、重新离散后的 CV 或不同 NodeTree point 布局之间直接复用参数；
- 在部署时任意改变突触动力学而不重新训练；
- 预测树突各位置电压、离子浓度或其他 detailed Cell 内部状态；
- 用同一个位置条件模型泛化到任意连续树突位置。

这些需求可以由 `reduction` 下后续的其他约化模型承担，不应混入第一版 DBNN。

## 3. `reduction` 模块的层次

`reduction` 应作为多种约化模型的共同归属，而不是把 DBNN 直接做成独立的顶层模块。整体上
需要区分以下职责：

```text
reduction
├── 公共约化工作流
│   ├── 教师 Cell 快照与一致性检查
│   ├── 数据集生成和数据契约
│   ├── 训练、验证与评估编排
│   └── 模型资产和来源追踪
├── DBNN
│   ├── DBNN 数学模型
│   ├── DBNN 训练逻辑
│   └── DBNN 部署运行时
└── 其他约化模型
```

这个划分表达的是所有权边界，不是最终文件或公开 API 清单。未来其他约化模型应能复用教师
Cell、数据生成、元数据和导出等公共能力，同时保留自己的模型结构与训练方法。

## 4. 端到端约化流程

### 4.1 固化教师 Cell

数据生成开始前，需要把 detailed Cell 的有效配置固化为本次约化任务的来源：

- morphology、CV 离散结果和 NodeTree point 布局；
- 膜、离子和通道机制及参数；
- 初始状态、仿真步长和温度等运行条件；
- 胞体输出位置和 spike 阈值约定；
- 影响结果的随机种子与外部配置。

数据集和模型资产都必须能够追溯到同一个来源配置。训练完成后，如果来源 Cell、CV 布局或
NodeTree points 发生变化，原模型不能被静默当作仍然兼容。

### 4.2 建立输入通道布局

第一版默认覆盖 detailed Cell 当前 NodeTree 中全部已物化的 electrical points。输入通道由以下
二元关系确定：

```text
输入通道 = electrical point_id × 突触原型
```

这里的 point 是 `NodeTree.nodes` 中具有稳定 `point_id` 的 electrical node，不只是 CV 中点。
当前 NodeTree 实际物化：

- 每个 CV 的 midpoint node；
- root 和 branch endpoint 对应的 boundary node；
- branch junction 处由多个 CV/branch role 共享的 node。

多个 CV 或 branch role 折叠到同一个 electrical node 时，按 `point_id` 去重。一个 point 需要保留
全部 `NodeRole` 和连续 branch 坐标别名，但同一突触原型只产生一个 DBNN 通道。

当前实现不会为 branch 内部的每个 CV 分界单独物化 boundary node。突触若恰好放在内部 CV
边界，仍按现有半开区间规则归属一个 CV，并落到该 CV 的 midpoint `point_id`。第一阶段复用此
规则，不为 DBNN 单独创造底层不存在的 point。

每个 point 的位置清单至少保存：

- `point_id` 和 node kind；
- point 对应的全部 CV-local roles；
- 一个用于自动放置的 canonical branch location；
- 所有等价 branch location aliases；
- 对应的 source CV ids。

canonical location 从该 point 的全部连续位置别名中按稳定顺序选取；mid role 使用所属 CV 中点，
prox/dist role 使用对应 CV 边界。自动放置后必须检查 SynapseView 返回的实际 `point_id` 与清单
一致，不能只假设 branch 坐标解析顺序永远不变。

突触原型不能只保存一个模糊的 E/I 标签，还需要描述会影响教师 Cell 响应的固定配置，至少包括
polarity、机制类型、动力学参数、反转电位以及参考输入强度或训练强度范围。

第一阶段默认在每个 point 上建立 E 和 I 两个原型。最简单配置下，两者使用相同的突触机制和
动力学，只使用分别显式配置的反转电位区分；不写死适用于所有细胞的通用反转电位。用户可以
覆盖两者动力学或增加更多固定原型，但改变原型配置后需要重新训练。

突触事件表示非负的事件计数或电导幅度，兴奋和抑制由 detailed Cell 中的反转电位产生。DBNN
第一层据此学习 EPSP/IPSP 的符号和幅度。不能既使用抑制性反转电位，又在输入侧对 I 通道额外
乘负号，否则会造成双重符号处理。

每个通道必须有稳定、可保存的身份和顺序。通道使用 point-major 排列：

```text
channel_id = point_id * n_prototype + prototype_index
```

默认 prototype 顺序为 E、I。同一 `point_id` 上的 E/I 属于不同通道；同一 point、同一 prototype
上的多个连接可以汇聚到一个通道，但其事件幅度和统计分布应落在训练覆盖范围内。

改变 CV 离散、NodeTree point 布局、prototype 顺序、反转电位或其他原型动力学意味着通道语义
发生变化，需要重新生成数据并训练。经典 DBNN 不通过隐藏的近邻匹配来支持未训练 point 或
未训练机制。

### 4.3 生成刺激与数据集

数据生成层负责在输入通道上安排事件，运行 detailed Cell，并同步记录：

- 各通道的事件序列或刺激幅度；
- detailed Cell 的胞体电压；
- spike 时间，仅用于屏蔽动作电位区间、校准阈值和评估放电；
- 通道布局、突触原型、仿真条件、刺激协议和随机种子等元数据。

训练 DBNN 的双线性系数至少需要两个通道共同激活。默认训练协议应以双通道和多通道刺激为
核心，而不是把单突触响应当作主要训练数据：

- 双通道刺激用于直接约束通道对之间的双线性交互；
- 随机多通道时空刺激用于覆盖模型在实际网络中的联合输入分布；
- 单通道刺激是可选的校准与诊断数据，可用于检查第一层 PSP 核和通道位置映射，但不负责
  学习双线性系数。

自动生成不能被收缩成一个不可调整的固定协议。数据层需要同时支持：

- 一个开箱即用、覆盖感知的默认采样策略；
- 用户指定通道组合、事件时间、活动率、幅度分布、重复次数和随机化方式；
- 用户提供自定义采样策略或分阶段组合多种协议；
- 保存中间结果并复用已有数据，避免每次训练都重新运行 detailed Cell。

默认策略需要持续统计每个通道和每个通道对的有效激活覆盖度，并根据欠覆盖情况补充样本。
仅生成大量随机轨迹而不报告哪些通道对实际得到训练，不能视为自动数据生成已经完成。

通道对还必须按 polarity 分为 EE、EI 和 II 三类分别统计与补样。总体 pair coverage 达标但其中
任意一类明显欠覆盖时，不能认为双线性训练数据已经充分。

### 4.4 数据划分与质量控制

训练、验证和测试数据应按独立仿真轨迹或独立刺激实例划分，避免同一响应片段进入多个集合。
数据质量检查至少覆盖：

- 教师 Cell、CV/NodeTree point 布局和突触原型是否与任务配置一致；
- 输入事件和输出电压的时间对齐、步长和长度是否一致；
- 数据是否包含非有限值、异常初始状态或不完整仿真；
- 各通道、通道对及 EE/EI/II 分类覆盖是否达到本次任务声明的目标；
- 训练、验证和测试刺激是否分别可追溯且不存在泄漏。

覆盖目标不应在本大纲中写死为一个数字。用户可以根据 point 数量、突触原型数量、仿真成本
和模型用途调整，但最终训练报告必须给出实际覆盖情况。

### 4.5 训练与评估

训练层负责拟合第一层双指数时域核、第二层线性项和双线性交叉项，使 DBNN 预测的胞体亚阈值电压
逼近 detailed Cell。动作电位波形本身不作为第一版 DBNN 的主要拟合目标；训练时需要明确其
屏蔽或处理规则，避免 spike 波形主导亚阈值损失。

评估需要分别覆盖亚阈值电压、spike 时间和分布外推能力。

#### 亚阈值电压

亚阈值电压以可解释方差（variance explained，VE）作为核心指标：

$$
VE = 1 -
\frac{\sum_{t \in M}(y_t - \hat{y}_t)^2}
     {\sum_{t \in M}(y_t - \bar{y}_M)^2}.
$$

其中，$y_t$ 是 detailed Cell 的胞体电压，$\hat{y}_t$ 是 DBNN 预测电压，$M$ 是屏蔽动作
电位邻域后保留的亚阈值样本集合，$\bar{y}_M$ 是目标电压在同一集合上的均值。训练、验证和
测试必须使用一致且被记录的亚阈值 mask 语义，不能在预测结果出来后再选择性改变 mask。

VE 越接近 1，表示模型解释的亚阈值电压方差越多。测试目标在有效样本上的方差为零时，VE
没有定义，应标记为不可计算并同时报告有效样本数，不能用零或其他有限值掩盖该情况。

#### Spike 时间

DBNN spike 由预测电压的阈值上穿产生，真实 spike 来自同一刺激下的 detailed Cell。评估与
论文保持一致，默认把绝对时间差不超过 $10\,\mathrm{ms}$ 的预测和真实 spike 视为匹配，并采用
一对一匹配，避免一个 spike 被重复计数。匹配后统计：

- TP：成功匹配的预测 spike；
- FP：没有匹配真实 spike 的预测 spike；
- FN：没有被任何预测 spike 匹配的真实 spike。

spike 指标至少包括：

$$
\mathrm{Precision} = \frac{TP}{TP + FP}, \qquad
\mathrm{Recall} = \frac{TP}{TP + FN}.
$$

这里的 Precision 是 spike 的查准率，即本文所说的 spike“准确率”，不是按每个时间 bin
计算的分类 Accuracy；后者会因绝大多数时间 bin 没有 spike 而产生误导。报告必须同时给出
Precision、Recall、TP、FP、FN 和实际使用的匹配窗口。对应分母为零时，指标标记为不可计算，
并保留原始计数说明原因。

$10\,\mathrm{ms}$ 是 spike 评估的默认匹配窗口，不是训练电压时屏蔽动作电位邻域的窗口，两者
必须分别配置和记录。

#### 报告范围

验证集用于模型选择和阈值校准，独立测试集用于最终报告 VE、Precision 和 Recall。除总体指标
外，报告还应按刺激协议、突触原型、point kind、CV/branch 区域、EE/EI/II、活动率、输入强度
或通道对覆盖情况分组，避免总体指标掩盖局部位置、特定交互或训练分布边缘的明显失效。

完整双线性参数量随通道数平方增长，其中 `n_channel = n_point * n_prototype`。覆盖全部已物化
points 时，训练前必须估算通道数、通道对数量、数据量和计算成本。若规模不可接受，应由用户
减少候选 points、减少突触原型，或在未来选择低秩、位置条件化等其他约化模型；第一版不应
悄悄把完整双线性项改成另一种模型。

## 5. 保存、恢复与导出

需要区分两类用途：

- 训练状态用于中断恢复和继续优化，包含模型参数之外的训练进度与优化状态；
- 部署模型资产用于推理和网络运行，只包含复现模型语义所需的稳定内容。

部署资产至少需要携带：

- 模型类型、结构版本和训练得到的参数；
- 来源 detailed Cell、CV 离散和 NodeTree point 布局的可核对标识；
- 完整的 point manifest、位置 aliases、通道顺序以及 point/prototype 映射；
- 突触原型的 polarity、机制、动力学和反转电位配置；
- 时间步、输入时间对齐和单位约定；
- 训练覆盖的事件幅度、活动率和其他有效范围；
- spike 读取阈值；
- 数据集、训练配置和主要验证结果的来源信息。

第一版导出目标是可由 BrainCell 独立加载的模型资产，不要求同时支持 ONNX 等通用模型格式。
具体容器和序列化格式在实现阶段确定，但加载时必须进行版本、布局和运行条件校验。

## 6. 作为 population 运行

训练后的 DBNN 应表现为没有显式 morphology、但具有固定输入通道和胞体输出的简化细胞模型。
同一个模型资产可以实例化为多个同构单元组成 population，每个单元维护独立的在线状态。

Network 接入需要覆盖以下语义：

- 连接目标的连续 branch 位置使用与教师 Cell 相同的规则解析为 `point_id`，再结合突触原型
  稳定映射到对应输入通道；
- 同一时间步到达同一通道的多个连接输入能够正确汇聚；
- DBNN 的电压状态和阈值 spike 能被网络推进与记录；
- DBNN 可以作为突触前或突触后 population；
- `Cell -> DBNN`、`DBNN -> Cell` 和 `DBNN -> DBNN` 均能使用统一的事件延迟语义；
- DBNN population 与 detailed Cell population 可以出现在同一个 Network 中；
- DBNN 接入不能改变仅包含传统 detailed Cell 的现有网络结果。

当前 Network 运行时对多室 `Cell` 内部结构存在直接依赖，后续实现需要把网络节点运行能力与
具体 morphology runtime 解耦。但是本文不规定协议方法名、Connection 参数或事件 buffer 的
具体形状，这些应在 Network 与 reduction 联合实现设计中决定。

网络侧还必须校验模型资产声明的时间步、point manifest、通道语义、突触原型和输入有效范围。
不能把任意真实突触机制直接连接到一个仅按 E/I 标签匹配的 DBNN 通道，并假定其响应仍然等价。

## 7. 最小改动的第一阶段实现方案

第一阶段以“完成一个可独立验证的 DBNN 约化闭环”为目标，只新增 `reduction` 模块并包装调用
现有 Cell、事件源、连接和 Network 能力，不修改多室求解器、突触 runtime、连接 lowering 或
Network 时间循环。

```text
detailed Cell factory
    -> 全部已物化 electrical points 通道布局
    -> 刺激协议
    -> 现有 EventSequence + Cell.place + Network.run
    -> DBNN 数据集
    -> 训练与评估
    -> BrainCell 模型资产
    -> 独立 DBNN population runner
```

这一阶段能够完成数据自动生成、模型训练、评估、保存、加载、批量推理和有状态单步运行；
DBNN 暂不直接注册到现有 `Network`，混合网络接入在模型本身稳定后单独处理。

### 7.1 独立模块组织

DBNN 首先作为 `braincell.reduction` 下的独立子模块实现。建议按以下职责拆分，但具体文件名和
公开导出仍可在实现时收敛：

- 通道布局：管理 NodeTree points、位置 aliases、突触原型和稳定通道编号；
- 刺激协议：生成双通道、多通道和用户自定义刺激计划；
- 教师运行与数据集：包装 detailed Cell 仿真，保存输入、响应和元数据；
- DBNN 模型：实现序列前向、单步递推和阈值 spike 读取；
- 训练与指标：负责优化、验证、VE、Precision 和 Recall；
- 模型资产：负责训练状态、部署参数、通道清单和来源信息的保存与加载。

第一阶段不急于为所有未来 reduction 方法定义统一基类。只有教师运行、数据集或模型资产在
第二种约化模型中出现真实复用需求后，再把对应能力提升到公共层。

除可选的顶层导出外，第一阶段代码和测试均应位于 `braincell/reduction/`。训练优先复用项目
已有的 JAX、BrainState 和 Braintools 能力，不为 DBNN 新增 PyTorch 或 Optax 运行依赖。

### 7.2 教师 Cell 的输入形式

数据生成器应接收一个能够按指定 `pop_size` 创建未初始化 detailed Cell 的 factory，而不是
直接持有一个已经初始化的 Cell 实例。原因包括：

- 训练突触必须在 Cell 初始化前放置；
- Network 初始化后不能更改 population、连接或事件计划；
- 不同数据批次需要不同 EventSequence 和独立初始状态；
- 使用 factory 可以明确复现 morphology、paint、CV policy 和膜机制配置。

数据生成器可以先创建单细胞模板，从 NodeTree 建立去重后的 point manifest 和通道清单；随后
为每个数据 batch 创建大小为 `B` 的同构 detailed Cell population。factory 每次返回的 CV、
point_id、NodeRole、机制配置和初始状态必须一致，否则拒绝合并为同一个数据集。

### 7.3 用现有能力生成一个数据 batch

设通道数为 `C = n_point * n_prototype`，一个 batch 包含 `B` 条独立刺激轨迹。教师运行包装按
以下顺序工作：

1. 创建大小为 `B` 的 detailed Cell population。
2. 从模板 NodeTree 构建按 `point_id` 去重的 canonical locations 和位置 aliases。
3. 对每个突触原型，使用现有 `Cell.place` 将其广播放置到全部 canonical point locations。
4. 检查每个 placement 的实际 `point_id`、prototype 和预期通道完全一致。
5. 将每个教师输入源稳定编号为 `source_id = trace_id * C + channel_id`。
6. 用一个扁平 `EventSequence` 保存本批次全部通道的事件时间。
7. 按突触原型选择对应 source view 和 SynapseView，通过现有连接接口做等长对齐。
8. 在指定胞体位置注册电压 recording，把 EventSequence 和 detailed Cell 加入临时 Network。
9. 调用现有 `Network.run`，提取每条 trace 的胞体电压和 detailed Cell spike。
10. 将实际输入、输出、point manifest、单位、时间轴和随机种子写入数据 batch。

连接时不能假设 SynapseView 的内部行顺序。包装层应读取每行的 population index、point id、
CV/branch provenance 和突触原型身份，再计算对应 source id，以防以后底层布局顺序调整。

现有 EventSequence 只描述事件时间，connection weight 对一条路由中的所有事件固定。因此
第一阶段允许为每个 `(trace, channel)` 选择一个固定刺激强度，但不支持同一通道内逐事件改变
幅度。若以后确实需要逐事件幅度，应扩展事件 payload，而不是在 DBNN 包装层伪造语义。

### 7.4 数据生成和训练执行

数据集内部使用稀疏事件表保存 `trace_id`、`channel_id` 和事件时间，训练时再按 batch 栅格化为
固定时间步输入，避免长期保存巨大的稠密事件张量。每条 trace 还需要保存通道权重、胞体电压、
真实 spike 和 split 身份。

默认生成策略分为两部分：

- 覆盖感知的双通道刺激，优先补充缺少有效共同激活的通道对；
- 随机多通道刺激，覆盖部署时预期的活动率、时间关系和输入强度。

通道对覆盖计数使用与双线性参数相同的 packed 上三角顺序，并附加 EE、EI、II 分类统计。
训练集、验证集和测试集直接使用不同随机种子和刺激实例生成，不在仿真完成后把同一长轨迹
切分到不同集合。

DBNN 离线训练只保留一套权威数学实现：事件栅格化、双指数卷积、线性求和和严格上三角
双线性交互。训练使用屏蔽动作电位邻域的亚阈值损失，验证集用于模型选择和 spike 阈值校准，
测试集使用第 4.5 节定义的 VE、Precision 和 Recall。

重复执行的模型前向、训练 step 和时间循环必须使用 JIT 及 BrainState 循环变换，不使用 Python
逐时间步驱动模型。优化器优先使用项目已经依赖的 Braintools。

### 7.5 保存和独立 population 验证

第一阶段将训练状态与部署模型分开保存。部署资产采用 BrainCell 自有格式，至少由参数数组和
一份可读 manifest 组成，不依赖教师 Cell 或训练框架即可加载。

DBNN 模块需要提供两个数值一致的运行方式：

- 对完整事件序列进行批量推理，用于训练、评估和离线预测；
- 持有每个 population member 独立状态的单步递推，用于验证未来网络运行语义。

独立 population runner 显式接收当前时间步的 `(population, channel)` 事件，返回每个 member
的胞体电压和 threshold spike。它不冒充具有 morphology 和真实 SynapseView 的 detailed Cell。

第一阶段至少进行以下模块级验证：

- 通道布局在不同 batch 和保存加载后保持稳定；
- midpoint 和 branch boundary points 均生成 E/I 通道，共享 junction 按 `point_id` 去重；
- 内部 CV boundary 按当前底层规则映射到所属 midpoint，不产生虚假的独立通道；
- 同一 point 的 E/I 通道保持独立，非负输入通过不同反转电位产生 EPSP/IPSP；
- 相同 seed 生成相同刺激，不同 split 不共享轨迹；
- 一个小型 detailed Cell 能在全部已物化 points 接收 E/I 刺激，并得到对齐的输入、电压和 spike；
- EE、EI、II 三类通道对均有独立覆盖统计；
- 人工参数生成的数据能验证 DBNN 前向公式和双线性项顺序；
- 训练 smoke test 能降低验证损失并提高 VE；
- VE、Precision、Recall 和 spike 一对一匹配具有手工可核对结果；
- 完整序列与逐步递推在容差内一致，population member 之间无状态串扰；
- 模型资产 round trip 后 point roles、位置 aliases、prototype 顺序、通道编号、参数、预测和
  阈值 spike 不变。

### 7.6 第一阶段明确不做

现有 Network 会把具有 `pop_size` 的普通模型按多室 Cell 处理，连接目标也必须是 Cell 拥有的
真实 SynapseView。因此仅给 DBNN 增加 `pop_size` 或包装几个同名方法，并不能正确实现突触后
事件接收。

第一阶段不伪造 morphology、CV runtime 或 SynapseView，也不让 DBNN 继承 detailed Cell 来绕过
检查。以下能力推迟到独立模型验证完成之后：

- 将 DBNN 直接传给 `Network.add_population`；
- `Cell -> DBNN`、`DBNN -> Cell` 和 `DBNN -> DBNN` 的在线连接；
- 递归混合网络和统一事件延迟队列。

后续接入 Network 时，应单独引入最小网络节点协议和显式事件输入端口，使 Network 面向运行
能力而不是多室 Cell 私有实现。该改造不属于第一阶段“只包装调用”的范围。

## 8. 完成范围

### 8.1 第一阶段验收

“只包装调用”的第一阶段完成时，应覆盖以下端到端场景：

- 从一个可运行的 detailed Cell 建立按 `point_id` 去重的全 point E/I 通道布局；
- 使用默认覆盖感知策略生成可复现的数据集；
- 允许用户替换或组合刺激策略，而无需修改 DBNN 数学模型；
- 给出通道、通道对及 EE/EI/II 分类覆盖报告，并识别欠覆盖的双线性系数；
- 完成训练、验证和独立测试；测试报告至少包含亚阈值 VE、spike Precision、Recall、TP、
  FP、FN、有效亚阈值样本数和 spike 匹配窗口；
- 保存训练状态，并导出可以重新加载的 BrainCell 模型资产；
- 资产重新加载后保持通道语义、模型输出和适用范围一致；
- 将 DBNN 作为独立 population runner 完成批量和单步运行验证；
- 对 Cell、CV/point 布局、突触原型、反转电位、时间步或输入范围不兼容给出明确错误或警告。

第一阶段验收不要求 DBNN 出现在现有 Network 中，也不包含与 detailed Cell 的在线混合连接。

### 8.2 完整目标验收

后续完成 Network 节点协议和事件输入端口后，再增加以下系统级验收：

- DBNN 可以作为正式 population 加入现有 Network；
- `Cell -> DBNN`、`DBNN -> Cell` 和 `DBNN -> DBNN` 的事件、weight 和 delay 语义正确；
- DBNN 与 detailed Cell 可以在同一网络中同时运行和记录；
- 混合网络中的结果与 DBNN 独立单步运行结果一致；
- 接入 DBNN 后，只有传统 detailed Cell 的现有网络行为和测试结果保持不变。

## 9. 后续实现前需要继续讨论的问题

以下问题不阻塞本大纲，但在进入具体 API 和实现规格前需要分别确定：

- 全部已物化 points、多个突触原型下可接受的最大通道数和完整双线性参数规模；
- 默认覆盖感知采样的停止条件，以及不同通道对需要的最低有效样本量；
- 刺激幅度与未来 Network connection weight 之间的归一化和单位语义；
- 动作电位邻域在电压训练中的屏蔽范围，以及 spike 阈值的校准方式；
- 教师 Cell、CV 离散、NodeTree point manifest 和位置 aliases 的稳定指纹如何定义；
- 数据集、训练状态和部署资产的具体存储格式与版本迁移策略；
- Network 为 morphology Cell 和 reduction Cell 提供的最小统一运行协议。
