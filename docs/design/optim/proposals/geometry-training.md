# Nonlinear Pattern Separation Validation Design

状态：分阶段方案已确认；运行时数组、声明/runtime 生命周期迁移和固定网格 Geometry View 已实现第一版。
长度/半径比例、Ra、cm 已接入 runtime 派生量和统一 trainable registry，cable/operator 支持
population 前导轴；严格 policy 一致性与复杂 coverage 的完整验收仍属后续工作。能力边界见 [参数支持度](../current/parameter-support.md)。
已实现的数组与缓存行为见 [Runtime Cable Arrays](../current/runtime-cable-arrays.md)。

本方案最初解决从静态 morphology/CV 数据到电压损失的几何梯度被 NumPy 转换及缓存
截断的问题。当前数值层、固定网格 View 和训练验证已进入工作区；本文保留选型依据、
阶段验收条件及尚未完成的 policy 约束与任务对照。实际调用契约以 Current 为准。

## 一个分支、四个 CV

长 40 μm 的分支经 `CVPerBranch(4)` 得到四个 10 μm 的 CV。
学习后长度若为 `[5, 10, 15, 20] μm`，CV 的身份和连接可以不变，
但重新对长 50 μm 的分支应用相同 policy 会得到四个 12.5 μm 的 CV。
CV 数相同不等于重新离散化结果相同。

圆柱 CV 的膜面积、总膜电容和单侧半段电阻为：

\[
A=2\pi rL,\qquad C=c_m A,\qquad R_{half}=R_a\frac{L/2}{\pi r^2}.
\]

这里 L、r 用 cm，A 用 cm²，cm 用 μF/cm²，Ra 用 Ω·cm，得到 C 的单位为 μF、
R 的单位为 Ω。相邻 CV 的半段电阻按节点图上的串并联关系组合，不能将所有半段
电导简单相加。点电流还需要通过当前面积转换为密度；只更新轴向矩阵会产生不一致。
锥形和多片段 CV 必须使用实际片段公式，不能隐式替换成等半径圆柱。

## 迁移前的数值归属与改造目标

| 位置 | 迁移前行为 | 改造目标 |
| --- | --- | --- |
| `braincell/_discretization/geometry.py` | 根据区间裁剪片段，计算面积和半段轴向因子 | 将固定片段归属与连续数值求值分离 |
| `braincell/_compute/state.py`、`bridge.py` | 创建面积数组、clamp 面积副本和 ion 几何 | 所有消费者读取同一份当前几何，防止旧副本 |
| `braincell/quad/_staggered.py` | DHS 静态源同时保存索引、电容、轴向系数 | 保留树索引及消元顺序，分离可微数值系数 |
| `Cell._get_axial_operator` | NumPy 构建并缓存导数路径的轴向算子 | 与 DHS 使用一致的几何数值来源 |
| `braincell/trainable/_manager.py` | 求值 bindings，再提交运行时参数 | 几何、context、cable、派生量之间需要显式依赖顺序 |
| `braincell/experimental/optim/gradients.py` | 在被求导的 initializer/step 中 materialize | 保持完整梯度链，不能把数值缓存移到求导范围外 |

整数索引、拓扑、片段数量和消元调度不需要变成浮点可微量。普通形态构建也不必
全部改用 JAX。已采用的数值层在 host 固定映射后，用成批数组计算连续物理量；
与直接将所有构造步骤替换为 eager JAX 运算相比，这能减少逐标量 dispatch，
但初始化实际成本必须通过基线测量确认。

## 两种 policy 语义

| 比较项 | 固定离散结构、自由几何（首版） | 保持 policy 一致（后续约束模式） |
| --- | --- | --- |
| policy 的职责 | 产生初始 CV 和映射 | 初始离散化，并约束允许的参数变化 |
| MaxCVLen 后学习每个 CV 的长度 | 允许，可以超过初始 max length | 必须满足所选一致性契约 |
| DLambda 后学习半径、Ra、cm | 允许，电气节点结构固定 | 需要限制影响电紧长度和划分的变化 |
| 重新应用 policy | 不承诺重建同一个离散模型 | 必须定义并验证何谓“同一个” |
| 训练中的重新划分 | 不进行 | 同样不在梯度循环内自动进行 |
| 代价 | 需要明确区分参考形态和当前电缆几何 | 增加依赖协议、参数约束及重建一致性验证 |

已决定先采用自由几何模式，**不设置基于 policy 类型的训练白名单**。
初始 policy 即使是 MaxCVLen 或 DLambda，也不妨碍之后每 CV 的连续参数训练。
“自由”不包括改变 CV 数、节点连接、机制布局、刺激/突触归属，或使用非物理的负长度、
非正半径、Ra、cm。正值/有界变换与 policy 约束是两回事。

源 Morphology 保留为参考声明，训练更新运行时电缆几何。参考归一化位置是稳定身份，
不自动解释成当前形态的物理弧长比例。当前长度、半径轮廓、面积及路径距离应有明确
读取入口；原始 xyz 不自动代表优化后的三维嵌入。结果导出需要标记参考/当前几何，
不能将运行时覆盖值静默写回可能被其他 Cell 共享的 morphology。

后续约束模式建立在同一可微数值层上。依赖字段可作为保守检查：MaxCVLen 依赖长度，
DLambda 依赖长度、半径、Ra、cm，自定义策略依赖未知时不能假定安全。
但这还不足以保证完整重建一致性：CVPerBranch 独立长度训练也会破坏等分；
片段划分、机制 coverage、位置映射也可能改变。因此必须先选择一致性强度，
再提供如分支整体长度缩放、共享半径约束等参数化。组合 policy 必须复用真实 dispatch
与依赖范围，包括跨分支依赖，不能只按类名施加限制。

## 阶段一：可微数值层与初始化测量

历史数组原型测量将声明构造、离散化、`init_state`、首次 JIT 执行分开计时。
当时的 `init_state` 会 clone morphology 并重新离散化，所以其耗时包含这一工作；
该历史协议中，事先访问 `cell.cvs` 不等于完成 init 内部的离散化。当前生命周期已改变，
历史 timing 不能直接当作当前版本的性能；当前契约见下节和 Cell API。
报告 12/128/512/1024 CV 分叉模型、CPU/GPU float64 的原始样本与初始化相对变化。
首次 JIT 时间包含 trace、编译、执行以及求解器懒构建，不称为稳态仿真耗时。
每种规模的首次预热也单独报告：可微数组的 eager kernel 编译可能出现在 init_state，
不能只比较预热后的初始化中位数而遗漏首次成本。

测量入口为 [geometry initialization benchmark](../../../../benchmarks/performance/geometry_initialization/README.md)。
正式测量按 benchmark 规范事先确认次数；源码和单元测试检查不替代性能证据。

补充的 [128-CV population 对照](../../../../benchmarks/performance/geometry_initialization/results/population-128.md)
以单一 CV 规模扫描 1/10/100/1000，避免扩展完整矩阵。固定共享几何下未见一致稳态
退化，新增成本主要体现在首次初始化，population 本身仍影响状态规模和设备吞吐。
该证据支持继续推进统一数值路径的集成，但不代替动态几何或 RTRL 的成本验证。

阶段验收顺序（当前实现见 Runtime Cable Arrays）：

1. 固定参考片段和节点映射，定义单位清楚的连续几何数组与派生数组。
2. 分离 DHS 拓扑和数值；导数路径、总电容、点电流换算、ion 几何同步接入。
3. 保留没有几何训练时的静态快速路径，或在测量证明成本可接受后采用统一路径。
   是否默认 eager JAX、整体 JIT 或按需构建由测量决定，不预先声称初始化无回归。
4. 验证无参数变化时旧/新前向一致，变化后与显式构建的参考模型一致；用有限差分
   验证长度、半径、Ra、cm 的梯度，并覆盖重复 JIT、reset 和缓存刷新。

## 声明、离散结果与 runtime 生命周期（已确认契约）

生命周期以 [Cell API](../../cell/current/api.md#生命周期) 和
[Trainable API](../current/api.md#lifecycle) 为维护入口；以下说明本方案依赖的边界。

`morphology`、`cv_policy`、`paint` 和 `place` 是 init 前的声明。Cell 创建后，
以及这些声明发生变化后，系统自动用最新声明生成离散结果；用户不需要显式调用
`cell.discretize()`。`init_state()` 只消费当前离散快照，不重新执行 policy。

离散得到的 CV、CVTree 和 NodeTree 是声明的派生结果。init 前它们只能读取，不能
通过 CV View 写入参数、绑定 record 或注册 trainable。连续位置的 record、stimulus
和 connection 保存位置声明，init 时再按照当前网格解析；CV midpoint 也必须按
branch 上的连续位置处理，不能保存为永久 CV 身份。

`init_state()` 后网格、拓扑、机制附着和位置映射全部固定。此时 `length`、`radius`、
`Ra`、`cm` 以及 channel/synapse 数值都属于同一个 runtime 参数层，可以直接修改或
注册 trainable；这些修改只影响固定 CV 上的数值，绝不触发 cv_policy，也不反写
morphology、paint 或 place 声明。

`reset_state()` 保留 runtime 参数和训练注册，只重建动态状态；`reset()` 清除所有
init 后覆盖、绑定、旧 View 和梯度引擎，恢复最近一次 init 前的声明快照。比如声明
`g_max=120` 后在 init 前改为 `80`，训练得到 `60`，则 `reset_state()` 后仍为 `60`，
而 `reset()` 后恢复为 `80`；paint rule 的原始 `120` 仍可作为声明记录读取。

固定网格上的 CV runtime 参数可以在训练中改变，但不自动反写 morphology。CV View
直接暴露 `length`、`radius_scale`、`Ra`、`cm`；旧的 `cell.geometry` 入口保留兼容。

`cell.cvs` 对用户表现为位置集合 View：它保留 tuple 的迭代和整数索引兼容性，
同时提供 `cell.cvs.length`、`cell.cvs[2:5].Ra` 等内容字段，并可继续展开
`channels`、`ions`、`synapses` 和 `connections`。底层求解器仍使用不可变的
离散 CV 记录；runtime 几何 View 只承载固定网格上的当前数值。

当前 cable/operator 已支持在 CV 轴前携带 population/batch 维度，因此不同 population
的 runtime geometry 可以进入同一批量求解；拓扑和消元调度仍然共享。该能力不改变
row/cv/population/all 的注册语义。
CV 到 morphology 的转换只能作为显式导出和模型重建操作，因为重新离散可能改变 CV 数量、
边界、拓扑和机制映射。直接优化 morphology points 也不属于当前可微训练路径；
连续函数到逐 CV 参数的映射（例如 `gmax=f(distance, theta)`）则可以保留。

阶段验收覆盖圆柱、taper、多片段、分叉和零长度半径跳变。部分区域 paint 的面积
coverage 不能在几何变化时无意保留过期权重：需要保留参考区域归属并重算其面积，
或在支持前明确拒绝该组合。不得仅凭“数组是 JAX array”宣布整体已可微。
已开放字段及尚未支持的组合见 [参数支持度](../current/parameter-support.md)。

## 阶段二：固定离散结构的自由几何

以已生成的 CV 图为训练对象，保留参考映射；policy 不再参与后续连续更新。
用 MaxCVLen 和 DLambda 构造模型后，更新长度/半径/Ra/cm，验证图结构和 root shape
保持不变、数值响应确实变化、源 morphology 不变。测试应包含重新离散化确实不同的
例子，以证明实现没有暗中重新划分或错误宣称重建一致。

当前参数化为每 CV 的 `length` 和 `radius_scale`，沿参考片段缩放长度及完整半径
轮廓，保留内部比例和 taper。圆柱上可对应绝对长度和半径；一般形态的独立 radius
profile、相邻 CV 边界与覆盖约束仍需单独定义，不把单一 radius 值冒充完整轮廓。

约束模式是这一阶段之后的可选扩展，不阻塞自由模式、View 或实验。禁止用未来
严格模式的限制反向收窄已经确认的首版范围。

## 阶段三：Trainable View

沿用现有 View 的选择、get/set、parameter/scale/parameterized、单位、共享 root、
group_by；注册统一在 init 后进行。geometry 与 cable 是不同 owner，聚合到同一个
`cell.trainables`。当前入口为 CV selection 的 `length`、`radius_scale`、`Ra`、`cm`；
`cell.geometry` 为兼容入口，完整字段及限制见 [参数支持度](../current/parameter-support.md)。

逐项核对 row/population/cv/all 的含义及运行时数组轴，不能因为初值相同就折叠独立
根。若第一版仅支持 population 共享几何，必须明确说明受限的 grouping 及原因，
不能表面接受 population 分组、内部却忽略它。branch/region/custom grouping
与已有公共分组议题共同讨论，不为单个实验建立特殊路径。

materialize 的顺序应从几何根、参考片段生成当前 context，再处理依赖它的 cable/
机制参数和派生量。几何自身的参数化读取参考 context，以避免隐式自依赖；
普通声明 callable 与运行时 parameterized callable 的 context 语义分别定义。
初始验证失败不应留下部分写入；reset_state 不回滚优化器根，完整 reset 清除训练注册。

## 阶段四：可训练实验

先在小模型验证 RTRL、BPTT 与有限差分，再回到 Nonlinear Pattern Separation 实验。
第一项对照只增加几何/Ra 自由度，保留原数据、刺激、3 ms loss、batch/update 语义、
CPU float64 和随机种子；spike loss 是另一个独立变量。
保存初末参数、训练曲线、完整电压轨迹、3 ms 分类及全窗 spike 诊断。
成功拟合并不证明几何是必要条件，也不证明网格误差足够小；训练后单独进行更细网格
的响应检查，重新划分不进入训练循环。

### Jaxley 严格对照

严格对照必须同时开放每个 CV 的 `gNa`、`gK`、`gLeak`、`length`、`radius` 和
`axial_resistivity`，即 `12 × 6 = 72` 个优化坐标。BrainCell 中对应为
`na.g_max`、`k.g_max`、`leak.g_max`、`length`、`radius_scale` 和 `Ra`。
该对照固定 `cm`，只使用 `voltage_at_3ms` loss，并保持原 nonlinear-pattern 的
数据、刺激、RTRL、学习率、batch size、seed 和 200 epoch 设置。36 个通道参数的
固定几何实验只是控制组，不能作为 Jaxley 的完整复现。

实验协议和运行入口由 [Nonlinear Pattern Separation validation README](../../../../validation/optim/nonlinear_pattern_separation/README.md)
维护；个别运行与多 seed 的有限性、收敛和对照限制见
[实验结果](../../../../validation/optim/nonlinear_pattern_separation/results/README.md)。
现有记录包含非有限训练失败，不能仅凭 72 个参数宣称成功复现。

## 当前验收：几何梯度对齐验证

在进入 nonlinear-pattern 之前，先在
[`validation/optim/nonlinear_pattern_separation`](../../../../validation/optim/nonlinear_pattern_separation/README.md)
完成单 population 的梯度对齐验证。验证对象按由小到大的顺序分为三层：

1. `CableArrays` 纯数值层：对 `length`、`radius_scale`、`Ra`、`cm` 的 JAX 梯度与中心有限差分比较；
2. runtime geometry 层：修改参数后面积、电容、轴向电阻和 axial operator 必须变化，且不改变 CV 数量、拓扑或机制布局；
3. rollout 层：同一个小模型分别用 BPTT、RTRL 和有限差分比较 loss 梯度，并检查一次小步优化降低目标误差。

当前三层小规模验证已经覆盖纯数值、runtime 和短 rollout 梯度方向，并对四类参数分别
完成 cable 观测恢复和电压轨迹下降检查；nonlinear-pattern 的单次与多 seed 训练结果
在实验目录记录，已有非有限失败，仍需进一步验收任务一致性。梯度小模型使用
`pop_size=(1,)`；现有 solver 的 population-specific cable/operator 批量路径已经实现；测试要求逐 population 结果与
独立 Cell 求解一致，不能只使用第一个 population 的几何。

## 尚待证据决定的细节

- 可微数值准备对小模型和较大模型初始化的影响；是否需要按需初始化或独立静态路径。
- taper、区域覆盖和 ion 几何的完整运行时表达，及其对缓存和梯度状态规模的影响。
- 参考/当前 context、独立 population 几何与复杂覆盖组合的完整验收；既有字段和
  row/population/cv/all 接口按 Current 维护，不重新作为待定接口。
- 后续严格模式的一致性强度：只保持 CV 数、保持参考映射，还是重建完整电气模型一致。

任务状态和下一步统一在 [Optimization TODO](../TODO.md) 维护。
