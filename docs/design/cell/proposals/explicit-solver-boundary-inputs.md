# Explicit Solver Boundary Inputs Proposal

## 目前的问题

状态：讨论中。当前 staggered/DHS 装配了包含边界刺激和突触的边界条件；显式 Euler/RK
使用的通用导数路径把边界 point 消去后，只保留无边界输入时的 CV 轴向算子，
没有把边界电流和突触的作用同步带入剩下的电压方程。

因此，模型可以在边界成功 place，但切换到当前显式路径后，这些输入对膜电压的作用会遗漏。
下面用一个 branch、3 CV、5 个节点把完整方程和缺失项列出来。问题是当前导数缺项，
提高显式方法的阶数不能补回；显式积分方法本身并不要求丢弃边界条件。

运行路径见 [Cell 架构](../current/architecture.md#电压与电流路径)，进度见 [Cell TODO](../TODO.md)。
下文依据源码分析和方程推导；显式 Euler/RK 的端点输入仍需数值复现与独立参考解对照。

## 例子：一个 branch、3 CV、5 个节点

取单个 branch，沿归一化位置均匀划分为 3 CV。当前 node tree 是：

```text
node       0          1          2          3          4
x          0         1/6        1/2        5/6         1
role     boundary   CV1 mid   CV2 mid   CV3 mid    boundary
           o---a0-----o---a1-----o---a2-----o---a3-----o
```

节点 1、2、3 承载膜电容和动态膜电压，节点 0、4 是无膜电容的边界约束点。
内部 CV 分界 x=1/3、2/3 不另建节点。这里的五行是 **3 行膜电压 ODE 加 2 行边界代数约束**，
不把突触、通道等机制自身的状态方程计入这五行。

| 符号 | 含义 |
| --- | --- |
| $V_0,\ldots,V_4$ | 五个节点的电压 |
| $C_1,C_2,C_3$ | 三个 CV 的总膜电容，即膜面积乘比膜电容 |
| $a_0,\ldots,a_3$ | 相邻节点间的轴向电导；中点之间包含相邻半 CV 的串联电阻 |
| $F_i$ | CV i 的总膜电流及中点输入，不含轴向电流和端点输入 |
| $J_0(t),J_4(t)$ | 两端 CurrentClamp 的绝对电流，向内为正 |
| $s_0(t),s_4(t)$ | 两端电导型突触的瞬时电导 |
| $E_0,E_4$ | 两端突触的反转电位 |

所有电流统一采用向内为正。示例使用欧姆型突触 $I_{\mathrm{syn}}=s(E-V)$，
求边界电压时视该时刻的 s 为已知；s 自身仍可由突触动力学演化。
取有限正轴向电导和非负突触电导，保证下面的端点约束可解。

## 完整的五行方程

沿节点 0 到 4 排列：

$$
\begin{aligned}
0 &= J_0+s_0(E_0-V_0)+a_0(V_1-V_0), &&\text{左边界}\\
C_1\dot V_1 &= F_1+a_0(V_0-V_1)+a_1(V_2-V_1), &&\text{CV1}\\
C_2\dot V_2 &= F_2+a_1(V_1-V_2)+a_2(V_3-V_2), &&\text{CV2}\\
C_3\dot V_3 &= F_3+a_2(V_2-V_3)+a_3(V_4-V_3), &&\text{CV3}\\
0 &= J_4+s_4(E_4-V_4)+a_3(V_3-V_4). &&\text{右边界}
\end{aligned}
$$

第一、第五行要求边界输入与流向相邻 CV 的轴向电流平衡。它们决定 $V_0,V_4$，
再通过第二、第四行的轴向项影响 $V_1,V_3$；CV2 随后通过与两侧 CV 的耦合受到影响。

### Staggered 为什么可以消费边界输入

staggered/DHS 对这五个节点构造时间离散后的方程。以左端为例，边界行可以整理为：

$$
(a_0+s_0)V_0-a_0V_1=J_0+s_0E_0.
$$

$J_0$ 进入右端项，突触同时贡献对角系数 $s_0$ 和右端项 $s_0E_0$。右端边界同理。
当前 point-space 电流装配保留这些边界贡献，DHS 求解边界电压和 CV 电压，
因此端点刺激和突触能通过边界条件实际作用于膜节点。

边界行没有 $C\dot V$，属于代数约束。数值比较还需控制时间离散误差，检查步长收敛。

## 当前显式路径只剩下什么

当前 build_cv_axial_operator 只对纯轴向矩阵做边界消元，相当于在构造这个算子时采用：

$$
a_0(V_1-V_0)=0,\qquad a_3(V_3-V_4)=0.
$$

于是 $V_0=V_1$、$V_4=V_3$，两端轴向项在约化算子中消失。另一方面，
通用膜电流路径只取中点贡献，没有把端点上的 J 和突触电流补回来。
这个例子中的电压导数因此对应以下三行：

$$
\begin{aligned}
C_1\dot V_1 &= F_1+a_1(V_2-V_1),\\
C_2\dot V_2 &= F_2+a_1(V_1-V_2)+a_2(V_3-V_2),\\
C_3\dot V_3 &= F_3+a_2(V_2-V_3).
\end{aligned}
$$

这三行适用于端点没有额外输入的情形。即使端点已经 place 了 CurrentClamp 或 Synapse，
当前通用路径也没有保留上面完整边界条件产生的反馈。所有消费这个导数的显式 Euler/RK
都会继承这一缺项；其他复用通用导数的 solver 也需要核查。

## 正确消元后少不了的两项

从完整的第一、第五行解出端点电压：

$$
V_0=\frac{a_0V_1+J_0+s_0E_0}{a_0+s_0},
\qquad
V_4=\frac{a_3V_3+J_4+s_4E_4}{a_3+s_4}.
$$

代回相邻 CV 的轴向项，得到边界对 CV 的实际输入：

$$
\begin{aligned}
B_L &= a_0(V_0-V_1)
     =\frac{a_0}{a_0+s_0}\big[J_0+s_0(E_0-V_1)\big],\\
B_R &= a_3(V_4-V_3)
     =\frac{a_3}{a_3+s_4}\big[J_4+s_4(E_4-V_3)\big].
\end{aligned}
$$

所以，保留三行 ODE 时，完整形式应当是：

$$
\begin{aligned}
C_1\dot V_1 &= F_1+a_1(V_2-V_1)+B_L,\\
C_2\dot V_2 &= F_2+a_1(V_1-V_2)+a_2(V_3-V_2),\\
C_3\dot V_3 &= F_3+a_2(V_2-V_3)+B_R.
\end{aligned}
$$

**当前缺少的就是第一、第三行中的 $B_L,B_R$，以及与这些输入一致的端点电压。**
在代码计算 dV/dt 时，对应缺少的是 $B_L/C_1$ 和 $B_R/C_3$。
三个动态方程足以表达这个例子，前提是消元同步保留边界输入，而非只保留无输入的轴向算子。

### 三种情形检查

| 边界配置 | 左端正确反馈 $B_L$ | 当前三行遗漏了什么 |
| --- | --- | --- |
| 无刺激、无突触：$J_0=s_0=0$ | $0$ | 本例没有边界输入缺项 |
| 仅电流刺激：$s_0=0$ | $J_0$ | 注入左边界的电流应完整进入 CV1 |
| 仅突触：$J_0=0$ | $\frac{a_0s_0}{a_0+s_0}(E_0-V_1)$ | 突触产生的等效电导作用 |

右端分别对应 $J_4,s_4,a_3,V_3$。只有突触时，令
$s_{\mathrm{eff},L}=a_0s_0/(a_0+s_0)$，遗漏项可展开为：

$$
B_L=s_{\mathrm{eff},L}E_0-s_{\mathrm{eff},L}V_1.
$$

因此缺少的不只是外加电流常数，还包括电压反馈项。突触和刺激同时存在时，二者通过边界
电压共同决定反馈，不能直接把 $J_0+s_0(E_0-V_1)$ 当作正确输入搬到中点。

上述式子中 a 和 s 均为电导，$a/(a+s)$ 无量纲，B 与 J、F 均为绝对电流，
$B/C$ 为电压变化率。非欧姆型或电压依赖电导需要重新求相应边界约束，不能直接套用本例闭式表达式。

## 对应到当前源码

| 位置 | 与本例的对应关系 |
| --- | --- |
| [node_build](../../../../braincell/_discretization/node_build.py) | 构造三个 CV 中点和 branch 两端，共五个节点 |
| [staggered](../../../../braincell/quad/_staggered.py) | DHS 保留边界行；build_cv_axial_operator 则只约化纯轴向矩阵 |
| [currents](../../../../braincell/_multi_compartment/currents.py) | total_membrane_rate_point 保留边界绝对电流；通用路径的 _clamp_density 只填充 midpoint_ids |
| [bridge](../../../../braincell/_compute/bridge.py) | point_to_cv 只取中点；cv_to_point 只散布中点值，不求解 $V_0,V_4$ |
| [Cell](../../../../braincell/_multi_compartment/cell.py) | compute_voltage_derivative 将中点膜项和约化轴向项相加，没有补齐本例的 $B_L,B_R$ |
| [Runge-Kutta](../../../../braincell/quad/_runge_kutta.py) | 各 stage 调用上述通用导数，积分阶数不会补回遗漏的输入 |

## 后续讨论与验收入口

下一步目标是补齐上述边界输入及边界电压，保持合法 placement 的原始物理语义。
可比较在 CV 导数求值时恢复边界约束、提取与 staggered 共用的 point-space 装配，
或将两者组合。

后续需要明确 RK stage 状态、刺激采样与事件投递、非线性边界、边界电压观测及缓存。
验收先覆盖本例的无输入、仅刺激、仅突触和二者同时存在，再扩展到单 CV、多分支汇聚、
电流守恒、population/batch、reset 和分段运行。在稳定步长下与解析或独立参考解比较并检查收敛。
场景入口见 [staggered 端点测试](../../../../braincell/quad/_staggered_test.py)。

[两个 Compartment 类的统一提案](single-multi-compartment-unification.md) 的中点输入限制属于集中参数模型的范围，
不能用于回避普通 cable Cell 在这里缺失的边界输入。
