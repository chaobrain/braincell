# Arbor CV Discretization

## 用途与关联

Arbor v0.12.2 的跨 branch CV 如何表示、如何汇总电学属性，是 Cell 形态兼容方案的参考。

| 关联内容 | 用途与采用状态 |
| --- | --- |
| [SingleCompartment 与 MultiCompartment 的统一](../proposals/single-multi-compartment-unification.md) | 比较等效圆柱体预处理与原形态跨 branch 单 CV；后续形态表示待选择 |
| [Cell 架构](../current/architecture.md#cv-与-point) | 当前实现的对照：BrainCell 的 CV 仍是单 branch 区间 |
| [Cell TODO](../TODO.md) | 查看后续问题和模块协作进度 |

## 跨 branch CV 的表示

Arbor 的一个 CV 可以由原始 morphology 上多个连通的 cable 区间组成。每个区间记录
branch 及其起止位置，原始几何与拓扑仍然保留。例如下图约定子 branch 接在 branch 0 远端：

```text
branch 0 --------+-------- branch 1
                 |
                 +-------- branch 2
```

可以用以下两种方式理解划分，区间坐标均按各自 branch 的长度归一化：

| 划分 | CV 所覆盖的区间 |
| --- | --- |
| 整树一个 CV | CV 0: branch 0、1、2 的 `[0, 1]` |
| 中央分叉 CV | CV 0: branch 0 的 `[0, 1]`，branch 1、2 的 `[0, 0.5]` |
| 上述划分的两个远端 CV | CV 1: branch 1 的 `[0.5, 1]`；CV 2: branch 2 的 `[0.5, 1]` |

后两行合起来是一种三个 CV 的划分，CV 0 分别连接 CV 1、CV 2；不是把不连通的片段任意
合并。Arbor 的 `cv_policy_single` 对整个 cell 生成一个 CV，对区域则每个连通分量一个 CV。
显式边界策略以及允许分叉位于 CV 内部的策略支持跨 branch 的划分。参见
[Arbor CV policy 与表示](https://docs.arbor-sim.org/en/v0.12.2/cpp/morphology.html)。

## 电学属性如何汇总

按 Arbor v0.12.2 的 `fvm_cv_discretize` 和 `apply_parameters_on_cv` 实现：

- 面积由覆盖区间的膜面积求和，总膜电容由膜电容密度对面积积分。
- 初始电压与温度按膜面积加权平均。
- 密度机制参数在其覆盖区域内按面积汇总，并记录覆盖区域占整个 CV 的面积比例。
- 相邻 CV 的轴向电阻按连接路径积分。无分叉 CV 使用中点作为参考位置；分叉 CV
  使用靠近相应接口的分叉点。一个 CV 电压不要求在形态上只有一个参考位置。
- 实现也由总面积和总长度计算等效直径 `d = A / (pi L)`，并派生 `volume = A d / 4`。
  这是离散后的派生几何量，不是先把整棵树变成圆柱再划分 CV。

源码见 [Arbor v0.12.2 fvm_layout.cpp](https://github.com/arbor-sim/arbor/blob/v0.12.2/arbor/fvm_layout.cpp)。

## 近似与使用边界

整树一个 CV 把空间膜电压自由度合为一个；在通常的封闭边界、无额外电耦合条件下，
没有相邻 CV 的轴向耦合项。这种等电位近似舍弃了空间传播与局部电压差异；
非线性机制的参数平均一般会改变动力学。参见
[Arbor 离散电压方程](https://docs.arbor-sim.org/en/v0.12.2/dev/matrix_solver.html)。

整树一个 CV 与恰好 N 个 CV 是不同的划分问题；后者还需确定边界选择准则，
上述 Arbor 策略没有直接提供这种通用接口。BrainCell 的候选方向见
[形态兼容讨论](../proposals/single-multi-compartment-unification.md#形态与旧接口的兼容)。
