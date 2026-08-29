# 从 e-prop 到 HH：路径分解、Forward Sensitivity 与 Online 梯度

## Reference 状态

本文澄清 BPTT、exact forward sensitivity（RTRL）和 e-prop 的关系。单个
multi-compartment Cell 的具体 online 实施见
[单细胞 Exact Online Forward Sensitivity](./single-cell-online-forward-sensitivity.md)，当前
staggered solver 的离散映射见
[Staggered Solver BPTT 分析](./staggered_solver_BPTT_spike_gradient_analysis.md)。

## 1. 离散动力系统

考虑固定参数的离散系统：

$$
x_{t+1}=F_t(x_t,\theta,u_t),
\qquad
L_T=\sum_{t=0}^{T}\ell_t(x_t,\theta).
$$

定义：

$$
J_t=\frac{\partial x_{t+1}}{\partial x_t},
\qquad
G_t=\frac{\partial x_{t+1}}{\partial\theta},
\qquad
q_t=\frac{\partial\ell_t}{\partial x_t}.
$$

这里所有偏导都在实际前向轨迹上求值。若初始状态由参数决定，还要保留：

$$
S_0=\frac{\partial x_0}{\partial\theta}.
$$

## 2. 三步展开与不重复计数

对 $x_{t+1}=F_t(x_t,\theta)$：

$$
S_1=J_0S_0+G_0,
$$

$$
S_2=J_1J_0S_0+J_1G_0+G_1,
$$

$$
S_3=J_2J_1J_0S_0+J_2J_1G_0+J_2G_1+G_2.
$$

完整梯度可以按每个 loss 的落点唯一切分：

$$
\boxed{
\frac{dL_T}{d\theta}
=
\sum_{t=0}^{T}
\left(q_tS_t+\frac{\partial\ell_t}{\partial\theta}\right)
}
$$

其中 $S_t$ 汇总所有从过去参数注入到当前状态的路径。也可以按每个 parameter injection
edge 唯一切分，使用未来完整 adjoint。不能把“完整未来梯度”与“完整过去 sensitivity”在
每个中间状态相乘后再对所有状态求和：

$$
\sum_t
\frac{\partial L}{\partial x_t}
\frac{\partial x_t}{\partial\theta}
$$

一般会重复计算同一条经过多个中间状态的路径。正确规则是二选一：

- local loss derivative $\times$ full past sensitivity；
- local parameter injection $\times$ full future adjoint。

## 3. BPTT：未来 adjoint 向后传播

定义：

$$
a_t=\frac{dL_T}{dx_t}.
$$

则：

$$
a_T=q_T,
\qquad
\boxed{a_t=q_t+a_{t+1}J_t}.
$$

采用 row-vector adjoint 记号时，总梯度是：

$$
\boxed{
\frac{dL_T}{d\theta}
=
a_0S_0
+\sum_{t=0}^{T-1}a_{t+1}G_t
+\sum_{t=0}^{T}\frac{\partial\ell_t}{\partial\theta}
}
$$

若 $x_0$ 与参数无关，$S_0=0$。注意 parameter injection $G_t$ 影响的是
$x_{t+1}$，因此与它配对的是 $a_{t+1}$，不是 $a_t$。

## 4. Exact forward sensitivity：过去路径向前传播

定义完整 sensitivity：

$$
S_t=\frac{\partial x_t}{\partial\theta}.
$$

递推为：

$$
\boxed{S_{t+1}=J_tS_t+G_t.}
$$

每当 $\ell_t$ 可用时，可以立即形成本步 gradient contribution：

$$
\Delta g_t
=q_tS_t+\frac{\partial\ell_t}{\partial\theta},
\qquad
g_t=g_{t-1}+\Delta g_t.
$$

$g_t$ 是前缀目标 $\sum_{k=0}^{t}\ell_k$ 对同一个固定 $\theta$ 的精确梯度。trial 内参数
固定、结束后才更新 optimizer 时，最终 $g_T$ 与 full BPTT 完全相同。二者只是 chain rule
的求值顺序不同：BPTT 保存过去、向后传播一个 cotangent；forward sensitivity 保存当前
$S_t$、向前传播所有 parameter tangent。

## 5. e-prop 不是 forward sensitivity 的同义词

e-prop 把 recurrent-network gradient 写成：

$$
\frac{dE}{dW_{ji}}
=\sum_t L_j^t e_{ji}^t,
$$

其中 eligibility $e_{ji}^t$ 收集 neuron-local、可在 forward 中递推的因素，ideal learning
signal 是 $L_j^t=dE/dz_j^t$。原始推导明确区分：local eligibility 的因子化本身不是近似；
online e-prop 的主要近似通常来自用当前可得的 partial learning signal 代替包含未来、非局部
影响的 ideal learning signal。对 hard spike 还需要 pseudo/surrogate derivative。

因此不能把 e-prop 简化为“eligibility 不让梯度穿过所有 event-mediated path”。更准确的
说法是：

- ideal learning signal 配合精确 local eligibility 时，因子化可以表示完整梯度；
- online e-prop 近似不可在线获得的 future/non-local learning signal；
- 具体 neuron model 还可能对 reset、spike derivative 或 eligibility locality 作额外近似。

参考：[Bellec et al., 2020](https://doi.org/10.1038/s41467-020-17236-y)。

对本文的单 HH Cell，直接保存 $S_t=\partial x_t/\partial\theta$ 是 full-state RTRL/forward
sensitivity。它不要求 neuron-local factorization，也不依赖离散 spike readout，所以不应仅因
形式上出现“trace $\times$ local loss derivative”就称为 e-prop。

## 6. HH、compartment 与 spike 的边界

对 multi-compartment HH/cable，状态至少包括：

$$
x_t=(V_t,Q_t),
$$

其中 $V_t$ 是所有 CV 电压，$Q_t$ 包括 gates、dynamic ion concentration 和连续 synapse
state。Axial coupling 使一个局部参数对其他 compartment 的影响随时间传播，因此完整
$S_t$ 一般不是局部标量 trace，而是：

$$
S_t\in\mathbb{R}^{N_x\times N_\theta}.
$$

需要区分：

- HH 动作电位波形是连续 $V,Q$ 动力学的一部分，voltage loss 对它的梯度不经过
  `Cell.spike`；
- `Cell.spike` 是 hard-forward、surrogate-backward 的离散 crossing readout；
- 只要 loss 不读取 `Cell.spike`，它也不经 synapse/delay 反馈到未来状态，该 readout 不在
  本目标的梯度路径上。

因此连续 HH 即使产生动作电位，也仍可用 exact forward sensitivity；“产生 AP”和“通过离散
spike event 做 credit assignment”不是一回事。

## 7. 固定外源 synapse 与 delay

若外部事件序列和 delay 预先给定，可以把已经对齐的刺激记为 $u_t$：

$$
x_{t+1}=F_t(x_t,\theta,u_t).
$$

delay 只决定哪个 $u_t$ 非零，不引入从 cell state 到未来 input 的可微路径。若事件驱动一个
连续 synapse，则 synapse 的 conductance/kinetic state 属于 $x_t$，其 sensitivity 正常随
$J_tS_t+G_t$ 传播。以下情况不属于这个简化：

- delay 自身可训练；当前固定步长量化 delay 对参数通常是分段常数；
- cell spike 经 ring buffer 反馈到自身或其他 cell；
- loss 显式依赖 hard event 或 event time。

## 8. Memory 与计算复杂度

朴素 reverse BPTT 的时间历史 memory 近似为：

$$
M_{\mathrm{BPTT}}=O(TN_x),
$$

checkpoint/rematerialization 可以用重计算降低它。Exact forward sensitivity 保存：

$$
M_{\mathrm{forward}}=O(N_xN_\theta),
$$

与 $T$ 无关，但不一定比 BPTT 小；只有相对长轨迹、较少参数时更有吸引力。

一般 dense recurrence 显式执行 $J_tS_t$ 的成本是
$O(N_x^2N_\theta)$ 每步。对 staggered cable，tangent 可以复用 tree/DHS linear solve 的
结构，每个参数方向近似需要一个 tangent right-hand side，理想情况下约为
$O(N_xN_\theta)$ 每步。相比之下 reverse BPTT 每步只传播一个 loss cotangent，参数多时通常
明显更便宜。Axial matrix 稀疏不代表 $M^{-1}$ 的影响局部，也不代表 full sensitivity 保持
稀疏。

所以 exact online forward sensitivity 的核心优势是：

- 不保存长度为 $T$ 的历史；
- 每个 prefix 的精确梯度立即可得；

其代价是保存和推进所有 state-parameter sensitivity，不能同时声称它天然具有 e-prop 的局部
内存与计算规模。

## 9. 结论

- BPTT 与 exact forward sensitivity 对固定参数离散系统计算同一个梯度。
- 单 cell、连续 HH、固定外源 input/delay、不使用 event feedback 时，不需要截断任何时间
  梯度。
- 维持的对象是完整 forward sensitivity matrix，而不是默认局部、标量的 eligibility trace。
- trial 内立刻改变参数会改变动力系统和优化目标，不再等价于原 fixed-parameter full BPTT。
- e-prop 的 exact factorization 与其 online learning-signal approximation 必须分开描述。
