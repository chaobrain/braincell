# SingleCompartment 与 MultiCompartment 的统一

状态：讨论中。

## 两个类如何统一

目标是让 `Cell` 同时表达多房室电缆模型和现有 `SingleCompartment` 的集中参数模型。
当前 `MultiCompartment` 是 `Cell` 的别名，`SingleCompartment` 仍是独立类，
见 [Cell API](../current/api.md)。首阶段兼容现有 SingleCompartment 的集中参数 ODE 建模方式。

主要差异有三个：single 使用唯一膜电压位点；Cell 需要 morphology；Cell 的部分 solver
还会装配边界约束。下面先说明方程何时等价，再讨论位点、构造和积分路径的兼容。
进度见 [Cell TODO](../TODO.md)。

## 单 CV 的三行方程

单 branch、单 CV 的电气节点为：

```text
proximal endpoint ---- midpoint ---- distal endpoint
       V_p       g_p     V_m    g_d       V_d
```

中点承载总膜电容 $C_{\mathrm{tot}}$，两端是无膜电容的代数约束点。取电流向内为正，
$g_p,g_d$ 为两段轴向电导，$I_p,I_d$ 包含端点刺激和突触电流，$I_m$ 为中点输入，
$I_{\mathrm{mem}}(V_m,z)$ 为总膜电流，$z$ 表示机制状态。完整电压系统是：

$$
\begin{aligned}
0 &= I_p+g_p(V_m-V_p),\\
C_{\mathrm{tot}}\dot V_m
  &= I_{\mathrm{mem}}(V_m,z)+I_m
     +g_p(V_p-V_m)+g_d(V_d-V_m),\\
0 &= I_d+g_d(V_m-V_d).
\end{aligned}
$$

这是一行膜电压 ODE 加两行边界约束。封闭边界且所有点机制和输入位于中点时，
$I_p=I_d=0$。在有限正轴向电导下，两端电压满足 $V_p=V_m=V_d$，轴向项消失：

$$
C_{\mathrm{tot}}\dot V_m=I_{\mathrm{mem}}(V_m,z)+I_m.
$$

匹配总电容、膜电流、初态及机制更新方式后，这就是 single 的膜电压方程。
门控变量、离子浓度和突触仍有各自的状态。普通单 CV cable 可以接受端点输入，
其额外反馈见 [边界输入提案](explicit-solver-boundary-inputs.md)；集中参数模式通过位点约束
保证使用这里的单方程形式。

## 特殊 Policy 与位点表达

候选方案是用特殊 CV policy 指定 single 模式，同时表达“只有一个 CV”和“只有一个输入位点”。
首阶段用于单 branch，保留 branch 的几何信息及其中的多个 segment。
普通 `CVPerBranch(1)` 表示每个 branch 一个 CV；特殊 policy 还携带集中参数模式的放置规则。

| 操作 | Single 模式下的候选行为 |
| --- | --- |
| 显式传入 locset | 只接受唯一 CV 中点，边界或其他位置报错 |
| 省略 locset | 默认放到该 CV 中点 |
| 刺激、突触及其他输入 | 统一作用于唯一膜电压位点 |

以下是候选写法，`single_policy` 表示待设计的策略对象：

```python
cell = Cell(morpho, cv_policy=single_policy)

# 两种等价的放置写法，任选其一。
cell.place(midpoint, mechanism)
cell.place(mechanism)
```

省略的是 `place(locset, mechanism)` 中的 locset。显式传入的位置仍需校验，
快捷方式和其他输入入口共同遵守中点语义。
需要决定：由 policy 自身负责校验，还是由 Cell 读取 policy 的模式后执行校验？

## 形态与旧接口的兼容

Cell 构造时需要 morphology。为了接收已有的分叉形态，可以先生成一个集中参数表示，
再复用单 branch 的 single policy：

```text
branched morphology -> lumped properties -> equivalent cylinder -> single policy
```

等效圆柱体至少需要匹配总膜面积和总膜电容。对原形态各段 k：

$$
A_{\mathrm{eq}}=\sum_k A_k,\qquad
C_{\mathrm{eq}}=\sum_k c_{m,k}A_k,\qquad
c_{m,\mathrm{eq}}=C_{\mathrm{eq}}/A_{\mathrm{eq}}.
$$

膜面积采用圆柱侧面积约定时，$A_{\mathrm{eq}}=\pi dL$。面积只约束直径与长度的乘积，
还需选择 d、L，以及处理离子池所需的体积。机制汇总需要保持目标膜电流；不同动力学参数
是否可以合并，应按机制判断。原 branch 上的 paint 区域、place 位置和初始状态也需要映射。
这一步将空间电压差异合并为一个电压自由度，是等电位近似；单方程与 single 的等价性针对
汇总后的模型，而原多房室模型的空间传播与局部响应需要另行评价。

可比较两种形态表示：

| 方案 | 表示方式 | 待解决问题 |
| --- | --- | --- |
| 等效圆柱体预处理 | 分叉 morphology 转成一个圆柱 branch，再使用现有单 branch CV 表示 | 几何与机制汇总、原位置映射、转换入口 |
| 原 morphology 跨 branch 单 CV | 保留原始形态，由一个 CV 覆盖多个连通区间 | 扩展当前 CV 表示及离散汇总，参见 [Arbor 参考](../references/arbor-cv-discretization.md) |

先完成单 branch 的兼容，再确定分叉形态采用哪种表示。构造接口还需衔接旧
SingleCompartment 的参数：由用户显式调用形态转换，还是由 Cell 的初始化入口生成所需 morphology？

### 状态形状与参数

现有 single 没有空间尾轴，Cell 使用 `pop_size + (1,)`。需要对应旧类的状态读写、
population 和 batch 用法，并明确总电流与电流密度、总电容与比膜电容的转换。
形态入口的简化和这些参数、状态兼容一起决定两个类如何统一。

## 积分路径是否需要单独实现

特殊 policy 已经让模型满足单方程条件，接下来比较是否值得为它省去边界装配：

| 方案 | 计算方式 | 需要验证 |
| --- | --- | --- |
| 复用现有 solver | staggered 保留边界装配；通用导数使用已有轴向消元 | 无边界输入时与 single 的导数、积分顺序和轨迹是否一致，以及额外开销 |
| 专用 single 分支 | 直接推进单行膜电压 ODE，复用膜电流与机制状态更新，省去边界装配和求解 | 性能收益、状态更新一致性及新增维护成本 |

核心问题是：能否在现有 solver 协议内部选择 single 分支，保持相同的调用和机制更新语义？
若需要独立积分实现，应明确它与通用路径共享哪些计算，以及如何维持求解器之间的一致性。

先用被动膜、主动通道和中点输入验证复用路径，匹配参数、初态、积分器、步长与机制更新顺序，
比较导数和轨迹，并检查初始化、reset 和批量状态。再根据边界装配的实际开销决定是否增加专用路径。
