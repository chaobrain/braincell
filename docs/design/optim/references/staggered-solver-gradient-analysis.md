# Staggered Biophysical Neuron Solver 梯度分析

## Reference 状态

本文说明 BrainCell 当前 `Cell(solver="staggered")` 的离散一步程序，以及该程序如何产生局部
state Jacobian $J_t$ 和 parameter injection $G_t$。它只讨论 solver-specific program
derivative，不重复 BPTT、RTRL、online update 或多步 loss 理论。通用链式法则见
[BPTT/RTRL 理论](./bptt-to-rtrl-neuron-derivation.md)。

本文是非规范性技术分析，不定义 Trainable Parameter API。规范性参数合同见
[API](../api.md) 和 [Architecture](../architecture.md)。

实现入口：

- [`staggered_step`](../../../../braincell/quad/_staggered.py)：DHS voltage step 和
  post-voltage mechanism 调度；
- [`Cell._update_dynamics`](../../../../braincell/_multi_compartment/cell.py)：完整 cell 动力学步；
- [`HHTypedNeuron.get_spike`](../../../../braincell/_base_neuron.py)：hard-forward、
  surrogate-backward threshold-crossing readout。

## 1. Solver 与梯度的关系

给定状态、参数和本步输入，具体 solver 定义离散映射：

$$
s_{n+1}=F_n^{\mathrm{solver}}(s_n,\theta,u_n).
$$

该 solver 的局部导数是：

$$
J_n^{\mathrm{solver}}
=
\frac{\partial F_n^{\mathrm{solver}}}{\partial s_n},
\qquad
G_n^{\mathrm{solver}}
=
\left.
\frac{\partial F_n^{\mathrm{solver}}}{\partial\theta}
\right|_{s_n}.
$$

更换 solver 通常会更换离散前向轨迹、$J_n$ 和 $G_n$。BPTT 与 RTRL 的通用求和公式不变，
但二者计算的是新离散程序的导数，不应假设不同 solver 给出相同参数梯度。

需要区分三类变化：

| 变化 | Forward value | Program gradient | 典型影响 |
| --- | --- | --- | --- |
| 更换数值积分公式 | 通常改变 | 改变 | 截断误差、稳定域和局部 Jacobian 都变化 |
| 更换相同线性系统的消元/backsub 算法 | 理想实数中相同 | 理想实数中相同 | 浮点顺序、编译图、temporary 和墙钟变化 |
| 只使用 checkpoint/rematerialization | 不改变 | 不改变 | 计算/内存权衡变化 |

## 2. 状态、参数与空间表示

第 $n$ 步开始时的连续状态写成：

$$
s_n=
\begin{bmatrix}
v_n\\
w_n
\end{bmatrix}.
$$

- $v_n$：CV midpoint space 中的 membrane voltage，shape 通常为
  `pop_size + (n_cv,)`；
- $w_n$：gates、ion concentrations、continuous synapse states 等 `DiffEqState`；
- $\theta$：保持在 JAX 热路径中的 channel、ion 或 mechanism parameter；
- $u_n$：本步已经准备好的 current、clamp 和 synaptic drive。

Painted density channel/ion state 位于 CV space。Placed continuous synapse state 使用 sparse
point-layout rows；只有需要 point mechanism 时才在 CV 与 point space 之间路由。DHS 的
boundary/sentinel point rows 是 algebraic workspace，不是额外的动态 CV state。

Morphology、CV policy、row ordering、axial resistance/capacitance coefficient 当前在 runtime
build/cache 阶段由 NumPy 生成，因此不在 autodiff parameter path 中。

`spike`、current cache 和 delay ring state 不一定属于连续 $v/w$ 状态。它们只有在被 loss
读取或反馈到未来 dynamics 时，才进入对应目标的 recurrent state contract。

## 3. 当前一个 timestep 的实际顺序

Standalone `Cell.update()` 的一整步是：

1. `_begin_step()` 在旧电压 $v_n$ 上应用已经准备好的离散 synaptic events；
2. 保存 `last_V = v_n`；
3. 对需要 total-current source 的 ion，在 $v_n$ 上缓存电流；
4. `dhs_voltage_step()` 求得 $v_{n+1}$；
5. painted density channel/ion 在 CV voltage 上按 schedule 执行 post-voltage update；
6. placed synapse/point mechanism 按需读取 point-space voltage 并更新连续状态；
7. 清除临时 current cache；
8. 用 $(v_n,v_{n+1})$ 计算可选的 threshold-crossing readout；
9. standalone 路径为下一步准备 synapse event payload。

Post-voltage update 有 `family` 和 `integration` 两种调度。每种调度都定义确定的顺序复合映射；
后 phase 可以读取前 phase 已更新的值，因此不能把所有 mechanism state 默认视为并行的同一个
Euler update。

## 4. DHS voltage step

### 4.1 Membrane current 局部线性化

`compute_membrane_derivative(v)` 计算不含 axial coupling 的 membrane derivative：

$$
f_m(v,w_n,\theta,u_n)
=
\frac{I_{\mathrm{mem}}(v,w_n,\theta,u_n)}{C}.
$$

当前程序在旧电压 $v_n$ 处计算逐 CV slope：

$$
\Lambda_n
=
\operatorname{vgrad}_{v}f_m(v_n,w_n,\theta,u_n),
$$

$$
c_n
=
f_m(v_n,w_n,\theta,u_n)-\Lambda_n\odot v_n.
$$

于是局部近似为：

$$
f_m(v,w_n,\theta,u_n)
\approx
\Lambda_n\odot v+c_n.
$$

Painted membrane current 当前在 CV 之间局部，axial coupling 单独进入 DHS，因此
`vector_grad` 的结果等价于所需对角 slope。若未来加入跨 CV 的非轴向 current，这个等价性
必须重新验证。

### 4.2 概念上的 CV-space 线性系统

令 $K$ 为已经按 membrane capacitance 缩放的 axial operator。忽略 point-tree algebraic row
的表示细节后，当前装配等价于：

$$
\boxed{
M_nv_{n+1}=r_n
},
$$

其中：

$$
M_n
=
I+\Delta tK-\Delta t\operatorname{diag}(\Lambda_n),
$$

$$
r_n
=
v_n+\Delta tc_n
+\Delta tC^{-1}I_{\mathrm{boundary},n}.
$$

因此 voltage phase 是：

$$
v_{n+1}
=
\Phi_v(v_n,w_n,\theta,u_n)
=
M_n^{-1}r_n.
$$

$M_n$ 和 $r_n$ 都通过在 $v_n$ 处计算的 membrane current 与 slope 依赖旧状态和参数。

### 4.3 DHS 表示层

代码不构造 dense $M_n^{-1}$，而是在 point-tree rows 上执行：

1. 将 CV voltage、local slope、constant term 和 boundary clamp 装配到 numeric rows；
2. leaf-to-root forward elimination；
3. recursive-doubling back substitution；
4. 从 dynamic midpoint rows 恢复 CV voltage。

Boundary row 和 sentinel row 改变程序表示与浮点执行顺序，不改变“求解线性系统”的数学目标。

## 5. Post-voltage mechanism solver

Dependent `DiffEqState` 当前不是简单显式 Euler。对一个 selected state $w^{(k)}$，程序冻结
本 phase 的其他 selected states，计算：

$$
f_n^{(k)}
=
f_k(v_{n+1},w_n,\theta,u_n),
$$

$$
\lambda_n^{(k)}
=
\operatorname{vgrad}_{w^{(k)}}f_k(v_{n+1},w_n,\theta,u_n),
$$

再执行 independent exponential Euler：

$$
\boxed{
w_{n+1}^{(k)}
=
w_n^{(k)}
+
\Delta t\,
\operatorname{exprel}(\Delta t\lambda_n^{(k)})
f_n^{(k)}
}.
$$

这里：

$$
\operatorname{exprel}(z)=\frac{e^z-1}{z},
\qquad
\operatorname{exprel}(0)=1.
$$

同一 selected-state phase 的 integrated values 从同一 snapshot 计算后统一写回；不同 family
phase 顺序执行。`IndependentIntegration` submodule 则使用自身配置的 solver。把全部
post-voltage phases 复合为：

$$
w_{n+1}
=
\Psi_w(v_{n+1},w_n,\theta,u_n).
$$

因此更换某个 mechanism 的 state solver 或 schedule，同样会更换完整一步的 $J_n,G_n$。

## 6. 完整一步的局部 Jacobian

定义 voltage phase 导数：

$$
\Phi_{v,n}:=\frac{\partial v_{n+1}}{\partial v_n},
\qquad
\Phi_{w,n}:=\frac{\partial v_{n+1}}{\partial w_n},
\qquad
\Phi_{\theta,n}:=
\left.\frac{\partial v_{n+1}}{\partial\theta}\right|_{s_n}.
$$

定义 post-voltage 复合映射的局部导数：

$$
\Psi_{v^+,n}:=\frac{\partial w_{n+1}}{\partial v_{n+1}},
$$

$$
\Psi_{w,n}:=
\left.\frac{\partial w_{n+1}}{\partial w_n}\right|_{v_{n+1}},
\qquad
\Psi_{\theta,n}:=
\left.\frac{\partial w_{n+1}}{\partial\theta}\right|_{v_{n+1},w_n}.
$$

复合两个 phase 后：

$$
\boxed{
J_n
=
\frac{\partial s_{n+1}}{\partial s_n}
=
\begin{bmatrix}
\Phi_{v,n}&\Phi_{w,n}\\
\Psi_{v^+,n}\Phi_{v,n}&
\Psi_{v^+,n}\Phi_{w,n}+\Psi_{w,n}
\end{bmatrix}
},
$$

$$
\boxed{
G_n
=
\left.\frac{\partial s_{n+1}}{\partial\theta}\right|_{s_n}
=
\begin{bmatrix}
\Phi_{\theta,n}\\
\Psi_{v^+,n}\Phi_{\theta,n}+\Psi_{\theta,n}
\end{bmatrix}
}.
$$

“Voltage phase 固定 $w_n$”只描述前向 phase 的输入快照，并不表示 autodiff 对 $w_n$ 执行
`stop_gradient`。通常 $\Phi_{w,n}\ne0$，post-voltage update 也通常满足
$\Psi_{v^+,n}\ne0$。

当前 JAX 还会继续对 membrane slope $\Lambda_n$ 求导。因此对 nonlinear current，当前
exact program derivative 可能包含 membrane dynamics 的二阶导数；它不是“冻结
linearization coefficient”后的自定义近似梯度。

## 7. Linear solve 的 forward 与 reverse derivative

对抽象线性系统：

$$
Mv=r,
\qquad
v=M^{-1}r,
$$

forward differential 为：

$$
\boxed{
dv=M^{-1}(dr-dM\,v)
}.
$$

每个 parameter tangent direction 可以复用同一消元结构求解新的 tangent right-hand side，
不需要形成 dense inverse；但不同 parameter directions 仍是不同 tangent columns。

给定输出 cotangent $\bar v$，reverse derivative 先解：

$$
\boxed{
M^T\lambda=\bar v
},
$$

再得到：

$$
\boxed{
\bar r=\lambda,
\qquad
\bar M=-\lambda v^T
}.
$$

当前 DHS 没有 custom VJP；JAX 对实际 elimination/backsub array program 求导。在系统
nonsingular 且浮点误差可控时，它与上述隐式微分一致。接近奇异的 $M_n$ 会同时放大 primal
误差、tangent 和 cotangent。

## 8. 更换 solver 时应重新确认什么

### 8.1 Voltage solver

从当前 semi-implicit DHS 换到 explicit、fully implicit 或其他 cable solver 时，需要重新确认：

- 新 $F_n$ 是否读取相同状态和参数；
- membrane linearization coefficient 是否参与 autodiff；
- linear solve 是否有 custom JVP/VJP，以及其导数是否对应实际 forward；
- boundary clamp、axial coefficient 和 morphology 是否仍为静态量；
- 新 solver 的稳定域与 $dt$ 是否改变 forward spike timing。

### 8.2 Mechanism state solver

从 independent exponential Euler 换成 Euler、RK 或 implicit state solver 时，需要重新确认：

- selected states 使用 simultaneous snapshot 还是顺序 updated value；
- independent submodule 是否包含 nondifferentiable branch；
- rate/slope 的高阶导数是否进入 program derivative；
- state reset、clip 或 hard bound 是否改变局部 Jacobian。

### 8.3 相同方程的实现替换

Ordinary Hines 与 recursive-doubling backsub 目标是同一线性系统，理论导数相同，但编译器图、
浮点归约顺序、temporary 和 GPU 并行度不同。实现替换后必须分别验证前向数值、梯度误差、
峰值内存和稳态执行时间。

## 9. 可选 `Cell.spike` readout

完整连续状态更新后，BrainCell 使用旧/新电压计算 threshold crossing：

$$
z_+=\frac{v_{n+1}-v_{\mathrm{th}}}{\Delta v_s},
\qquad
z_-=\frac{v_{\mathrm{th}}-v_n}{\Delta v_s},
$$

$$
o_{n+1}^{\mathrm{spike}}
=
H_{\mathrm{sg}}(z_+)H_{\mathrm{sg}}(z_-).
$$

默认 `ReluGrad(alpha=0.3, width=1)` 在 forward 使用 hard step，在 backward 使用有限支撑
三角 slope：

$$
\rho(z)=\max(0,\alpha(\mathrm{width}-|z|)).
$$

对应 surrogate partial derivative 为：

$$
\frac{\widetilde\partial o_{n+1}^{\mathrm{spike}}}{\partial v_{n+1}}
=
H(z_-)\frac{\rho(z_+)}{\Delta v_s},
$$

$$
\frac{\widetilde\partial o_{n+1}^{\mathrm{spike}}}{\partial v_n}
=
-H(z_+)\frac{\rho(z_-)}{\Delta v_s}.
$$

该 surrogate 只有在 loss 读取 `Cell.spike`、filtered event trace，或 spike feedback 影响未来
state 时才进入目标路径。单纯拟合连续 voltage trace 时，动作电位仍由连续 $v/w$ dynamics
产生，梯度不经过这个 readout。

## 10. Solver-gradient 验证

对每个新增或替换的 solver，至少验证：

1. 一步 JVP 与 central finite-difference directional derivative；
2. 一步 VJP 与 JVP 的 bilinear identity；
3. linear solve residual、NaN/Inf 和接近奇异时的失败行为；
4. x32/x64、多个 $dt$ 和 representative nonlinear channel；
5. state schedule 的 snapshot/顺序写回语义；
6. 若包含 `Cell.spike`，分别验证 hard forward value 与 surrogate local slope。

当前实现可以确认：DHS elimination、backsub 和 post-voltage JAX mechanism update 不会天然
切断 autodiff；但这不自动保证任意自定义 mechanism、independent solver、hard event、
morphology 或静态 topology 都可微。每个实际模型仍需对其真实一步程序做端到端验证。
