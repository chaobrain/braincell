# Runtime Cable Arrays

工作区已接入连续电缆数组、可微节点系数计算和固定网格几何训练接口。
当前从静态 CV 记录初始化；固定网格 runtime 字段为 `length`、`radius_scale`、`Ra`、`cm`。
`length` 支持直接值或 `trainable.scale()`，半径只支持对完整 profile 的整体 `radius_scale`。
后续工作见 [Nonlinear Pattern Separation validation design](../proposals/geometry-training.md)。

## 数值与拓扑

`braincell._compute.cable.CableArrays` 是四个带单位的 JAX 数组组成的 NamedTuple：
面积（cm²）、比膜电容（μF/cm²）、近端半段电阻（Ω）、远端半段电阻（Ω）。
每个数组的 CV 轴位于最后一维，前导维度可表示 population/batch。共享几何时为
`(n_cv,)`；独立 population 几何时为 `(..., n_cv)`。初始数据在 host 按完整向量
收集，统一传到设备，再交给运行时，避免每 CV 单独传输设备标量。

`CableTopology` 保留节点行映射、半段所属边、串并联分类等 NumPy 整数/布尔数组。
`axial_coefficients` 计算总膜电容及两侧有向耦合系数，`axial_matrix` 装配混合节点算子。
同分支内部相邻的半段串联，其他多角色边合并并联电导；代数边界行仍使用 1 μF 的
辅助归一化。该辅助量不表示边界拥有真实膜电容。

DHS 的数值源与通用导数路径的 Schur 消元都使用 JAX 数组运算，不再把中间系数用
`float` 或 `np.asarray` 转回 host。拓扑索引和消元调度保持静态。DHS 数值源和导数算子
只有在结果不含 tracer 时才写入 host 缓存，防止首次 JIT 留下失效 tracer。
原有 dense mixed-node matrix 装配仍在；稀疏化/线性复杂度改造不属于这次比较。

## 初始化、精度与数据副本

运行时 CV 面积直接引用 `CableArrays.area`，point 面积按代表 CV 映射取得。
Cell 的比膜电容来自同一 cable 数组；ion 几何和 clamp 面积在初始化期间转成设备向量。
这些派生量通过 `refresh_geometry()` 统一刷新；geometry View 的写入会同步面积、电容、
半段电阻、point area 和相关算子缓存。几何训练仍保持 CV 数量、拓扑和机制布局不变。

存储的 cable 数组遵循初始化时的 BrainState 精度。以 32 位初始化，再切换环境为
64 位，只能提升已存储数值的 dtype，不能恢复此前舍入的信息；要得到从头到尾的
64 位数值，应在 64 位环境下初始化。DHS 的每步操作数继续匹配已有电压状态的 dtype。
整数拓扑不随浮点精度切换。

普通 `quantity_vector` 也以完整 host 向量转换为设备数组，初始化期间原来的 NumPy
类型并不是新的公开兼容性保证。形态与离散化声明仍为 host 数据。

## 验证范围

[cable_test.py](../../../../braincell/_compute/cable_test.py) 使用分叉、taper、多片段和同分支
串联场景，与独立的旧式标量物理装配逐项对照；通过一个非均匀初始电压的线性求解，
比较四类数值输入的自动微分与有限差分，并检查重复 JIT 对输入变化有响应。
现有 Cell/DHS、面积映射、clamp 和 ion bridge 回归覆盖生产求解路径。

这验证的是面积、cm、两侧电阻到数值求解的梯度，以及 population 前导轴的批量一致性。
它不验证任意 morphology 的重划分梯度；重划分仍属于 init 前声明阶段，训练期间不会发生。
nonlinear-pattern/Jaxley 的训练记录见
[Nonlinear Pattern Separation validation 中的几何梯度证据](../../../../validation/optim/nonlinear_pattern_separation/results/README.md)。
初始化成本的测量协议、原始样本与性能结论由
[benchmark](../../../../benchmarks/performance/geometry_initialization/README.md) 维护。
首轮 [CPU/GPU 实测](../../../../benchmarks/performance/geometry_initialization/results/arrays-cpu-gpu.md)
显示 CPU/GPU 预热后都有毫秒级额外成本，first-use 增量更明显；CPU 基线修复重跑后，
两后端的对照均已完成。该证据用于下一步数组准备方案选择，不代表完整几何训练的性能。

## Population 与稳态成本

固定形态下，四个 cable 数组可为 `(n_cv,)` 或 `(..., n_cv)`；电压和 DHS
diags/solves 携带相同的 population 状态轴。
节点图和求解调度共享，但状态分配、部分 population/CV 元数据及每步求解工作会随
population 增大。因此共享几何不能推导出总初始化成本与 pop_size 无关。

当前 cable 数组不是 trainable State，编译时可能作为常量进入程序，连续系数运算
可能被常量折叠或移出时间循环；是否完全消除差异仍取决于实际编译结果。
稳态计时接近不能证明编译程序完全相同，也不能保证任何模型的后续仿真耗时不变。
未来几何成为动态参数后还需刷新派生系数；该场景与固定几何 forward 的性能不同。

[128-CV population 实测](../../../../benchmarks/performance/geometry_initialization/results/population-128.md)
覆盖 pop_size=1/10/100/1000、CPU/GPU 与编译后重复执行。GPU 新旧稳态差异约在 3%
以内，CPU 未观察到一致的额外退化；CPU 部分规模更快的现象不作为稳定加速结论。
共享几何时四个 cable 数组共享；独立几何时它们携带 population 轴。逻辑 State 大小随 population 线性增长，CPU/GPU 的
吞吐随规模变化明显。首次初始化仍有可测额外成本。这仅覆盖固定共享几何的被动
forward，不覆盖独立几何参数、主动通道或 RTRL。
