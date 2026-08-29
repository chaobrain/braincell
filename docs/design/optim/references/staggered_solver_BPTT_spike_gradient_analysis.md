# Staggered Biophysical Neuron Solver 中的 BPTT 与 Spike 梯度传播分析

## Reference 状态

本文是对当前 solver 和 gradient path 的非规范性技术分析，不定义 Trainable Parameter
API。规范性参数合同见 [API](../api.md) 和 [Architecture](../architecture.md)。

本文说明 BrainCell 当前 multi-compartment `Cell(solver="staggered")` 的实际时间推进、
JAX reverse-mode BPTT 路径，以及动作电位波形和离散 spike event 的不同梯度语义。这里的
公式用于解释当前代码，不把尚未实现的训练策略写成 solver 已有能力。

实现入口：

- [`staggered_step`](../../../../braincell/quad/_staggered.py)：DHS 电压步和 post-voltage
  mechanism 调度；
- [`Cell._update_dynamics`](../../../../braincell/_multi_compartment/cell.py)：完整 cell 动力学步和
  spike 检测；
- [`HHTypedNeuron.get_spike`](../../../../braincell/_base.py)：hard-forward、surrogate-backward
  的阈值上穿读出。

## 1. 与当前仓库的一致性结论

原有描述的核心结论是正确的：前向阶段“固定另一组变量”不等于在 backward 中对它
`stop_gradient`。只要参数和状态保持为 JAX-traceable value，梯度会穿过 voltage solve、
mechanism update 和多步状态递推。

但原有描述只对应一个简化的 operator-splitting 模型，以下细节与当前实现不完全一致：

| 主题 | 简化描述 | 当前实现 |
| --- | --- | --- |
| 电压更新 | 一般的 \(A(S_n,\theta)V_{n+1}=b\) | 先在 \(V_n\) 处线性化膜电流，再用 DHS 求解隐式 Euler；矩阵通常也依赖 \(V_n\) |
| 状态更新 | 显式 Euler 示例 | dependent state 使用逐状态独立指数 Euler；independent state 使用自身配置的 solver |
| 状态调度 | 一个整体 \(S_n\to S_{n+1}\) | runtime synapse、ion self-state、channel state 按配置分阶段推进 |
| spike | 与 spike 波形混合讨论 | 连续动作电位由 HH 动力学产生；`Cell.spike` 是求解后的 surrogate crossing 读出 |
| 可微参数 | 默认所有 \(\theta\) 可训练 | 当前 `ChannelView.trainable()` 只开放 schema 白名单中的连续 channel 参数；静态 topology 仍不可微 |
| 精度 | 容易被理解为二阶 staggered | 当前是 first-order Lie/semi-implicit split，不是 symmetric Strang split |

因此本文采用两层描述：先给出当前实现的精确数据流，再把它折叠成适合 BPTT 推导的复合
状态映射。

## 2. 状态、参数与空间表示

定义第 \(n\) 步开始时的状态：

\[
x_n = \begin{bmatrix}V_n\\Q_n\end{bmatrix}.
\]

其中：

- \(V_n\) 是 CV midpoint space 中的膜电位，shape 通常为
  `pop_size + (n_cv,)`；
- \(Q_n\) 汇总 gating variables、ion concentrations、continuous synapse states 等
  `DiffEqState`。Painted density channel/ion state 位于 CV space，shape 通常为
  `pop_size + (n_cv,)`；placed continuous synapse state 使用 sparse point-layout rows；
- \(\theta\) 是保持在 JAX 热路径中的模型参数，例如 channel conductance、reversal
  potential 或 kinetics parameter；
- \(u_n\) 表示本步已经准备好的外部电流、clamp 和 synaptic drive；
- \(P_{cp}\) 表示 CV-to-point 映射，\(P_{pc}\) 表示 point-to-CV 映射。它们由静态
  morphology/runtime metadata 决定。

`spike`、`_event_previous_V`、current cache 和 network delay ring 等辅助状态不都属于连续
ODE 状态。分析单 cell voltage fitting 时可以把它们留在 \(Q_n\) 之外；分析带事件反馈的
network 时，则必须把会影响未来步骤的 event/delivery state 一起纳入完整状态。

## 3. 当前一个 timestep 的实际顺序

Standalone `Cell.update()` 的一整步是：

1. `_begin_step()` 在旧电压 \(V_n\) 上应用已准备好的离散 synaptic events；
2. 保存 `last_V = V_n`；
3. 对需要 total-current source 的 ion，在 \(V_n\) 上缓存电流；
4. `dhs_voltage_step()` 求得 \(V_{n+1}\)；
5. Painted density channel/ion 直接在 CV voltage 上按 `ion_channel_update_order` 执行
   post-voltage mechanism schedule；只有 placed synapse/point mechanism 按需将
   \(V_{n+1}\) 映射到 point space；
6. 完成 channel/ion/synapse continuous state 更新；
7. 清除本步临时 current cache；
8. 用 \((V_n,V_{n+1})\) 检测上穿阈值并写入 `Cell.spike`；
9. standalone 路径为下一步准备 synapse event payload。

第 6 步存在两种调度：

- `family`：runtime synapse dynamics 之后，依次处理 dependent ion self-state、
  independent ion、dependent channel、independent channel；
- `integration`：把 runtime synapse 也纳入顶层 integration ownership，先处理 dependent
  node，再调用每个 node 的 independent updater。

后续推导把这些 post-voltage 子阶段复合为一个映射 \(\Psi_Q\)。这不会假设它们在代码中
同时更新，也不会丢掉阶段间的依赖。

## 4. DHS 电压步的实际方程

### 4.1 膜电流局部线性化

`compute_membrane_derivative(V)` 计算不含 axial coupling 的 membrane derivative：

\[
f_m(V,Q_n,\theta,u_n)
=
\frac{I_{\mathrm{mem}}(V,Q_n,\theta,u_n)}{C}.
\]

代码在旧电压 \(V_n\) 处计算逐点 voltage slope：

\[
\Lambda_n
=
\operatorname{vgrad}_{V}
f_m(V_n,Q_n,\theta,u_n),
\qquad
c_n
=
f_m(V_n,Q_n,\theta,u_n)-\Lambda_n\odot V_n.
\]

这里 `brainstate.transform.vector_grad` 返回 vector output 总和对输入的梯度。Painted density
current 直接在 CV space 计算；point current 经 midpoint routing 汇入 CV。当前 non-axial
membrane current 在 CV 之间是局部的，axial coupling 又被单独拿到 DHS 中，因此结果等价于
每个 CV 的 self derivative，即所需的对角斜率。若未来引入跨 CV 的非轴向 membrane current，
这个等价关系必须重新检查，不能把 `vector_grad` 自动当作完整 Jacobian。

局部一阶近似为：

\[
f_m(V,Q_n,\theta,u_n)
\approx
\Lambda_n\odot V+c_n.
\]

### 4.2 概念上的 CV-space 线性系统

令 \(K\) 为已经除以对应 membrane capacitance 的 axial operator，符号约定使
\(K\) 的对角为正、相邻 CV 项为负。忽略 node-tree algebraic row 的表示细节后，当前
数值装配等价于：

\[
\boxed{
M_nV_{n+1}=r_n
}
\]

其中：

\[
M_n
=
I+\Delta t K-\Delta t\operatorname{diag}(\Lambda_n),
\]

\[
r_n
=
V_n+\Delta t c_n
+\Delta t C^{-1}I_{\mathrm{boundary},n}.
\]

于是：

\[
V_{n+1}
=
\Phi_V(V_n,Q_n,\theta,u_n)
=
M_n^{-1}r_n.
\]

这比写成 \(A(S_n,\theta)V_{n+1}=b(V_n,S_n,\theta)\) 更精确，因为
\(\Lambda_n\) 和 \(c_n\) 都在 \(V_n\) 处计算；对 nonlinear membrane current，\(M_n\)
本身也依赖 \(V_n\)。

### 4.3 DHS 表示层

代码没有构造 dense \(M_n^{-1}\)。它在 point-tree row 上：

1. 将 CV voltage、local slope、constant term 和 boundary clamp 装配到 numeric rows；
2. leaf-to-root forward elimination；
3. recursive-doubling back substitution；
4. 从 dynamic midpoint rows 恢复 CV voltage。

Boundary point 是 algebraic row，额外 sentinel row 用于统一 back-substitution 索引。它们改变
求解表示，不改变“解一个线性系统”的微分结论。

Axial topology、row ordering、resistance/capacitance-derived static coefficient 通过 NumPy 在
runtime build/cache 阶段生成。因此当前 autodiff 不会给 morphology、CV policy 或这些静态
axial coefficient 产生梯度。channel current 和动态状态仍在 JAX/`brainunit` 热路径中。

## 5. Post-voltage mechanism update

Dependent `DiffEqState` 不是用显式 Euler 更新。对第 \(k\) 个状态，代码冻结其他状态，计算：

\[
f_n^{(k)}
=
f_k(V_{n+1},Q_n,\theta,u_n),
\]

\[
\lambda_n^{(k)}
=
\operatorname{vgrad}_{Q^{(k)}}f_k(V_{n+1},Q_n,\theta,u_n),
\]

然后执行 independent exponential Euler：

\[
\boxed{
Q_{n+1}^{(k)}
=
Q_n^{(k)}
+
\Delta t\,
\operatorname{exprel}(\Delta t\lambda_n^{(k)})
f_n^{(k)}
}
\]

其中：

\[
\operatorname{exprel}(z)=\frac{e^z-1}{z},
\qquad
\operatorname{exprel}(0)=1.
\]

在同一个 selected-state phase 内，各状态的 integrated value 先从同一 snapshot 计算，最后
统一写回；family 的不同 phase 之间则是顺序执行，后 phase 可以读取前 phase 已更新的状态。
`IndependentIntegration` submodule 不受上述公式强制约束，而是调用它自身配置的 solver。

把所有 phase 复合后记为：

\[
Q_{n+1}
=
\Psi_Q(V_{n+1},Q_n,\theta,u_n).
\]

## 6. 一个完整连续动力学步的 Jacobian

定义 voltage phase 的局部 Jacobian：

\[
A_n=\frac{\partial V_{n+1}}{\partial V_n},
\qquad
B_n=\frac{\partial V_{n+1}}{\partial Q_n},
\qquad
C_n=\frac{\partial V_{n+1}}{\partial\theta}.
\]

定义整个 post-voltage 复合更新的局部 Jacobian：

\[
D_n=\frac{\partial Q_{n+1}}{\partial V_{n+1}},
\qquad
E_n=\left.\frac{\partial Q_{n+1}}{\partial Q_n}\right|_{V_{n+1}},
\qquad
F_n=\left.\frac{\partial Q_{n+1}}{\partial\theta}\right|_{V_{n+1}}.
\]

则：

\[
dV_{n+1}
=A_n\,dV_n+B_n\,dQ_n+C_n\,d\theta,
\]

\[
dQ_{n+1}
=D_n\,dV_{n+1}+E_n\,dQ_n+F_n\,d\theta.
\]

完整状态 Jacobian 为：

\[
\boxed{
J_n
=
\frac{\partial x_{n+1}}{\partial x_n}
=
\begin{bmatrix}
A_n & B_n\\
D_nA_n & D_nB_n+E_n
\end{bmatrix}
}
\]

参数注入 Jacobian 为：

\[
\boxed{
G_n
=
\frac{\partial x_{n+1}}{\partial\theta}
=
\begin{bmatrix}
C_n\\
D_nC_n+F_n
\end{bmatrix}
}
\]

“更新 \(V\) 时固定 \(Q_n\)”只描述前向数值阶段。代码没有对 \(Q_n\) 调用
`stop_gradient`，所以通常 \(B_n\ne0\)。同理，post-voltage update 读取
\(V_{n+1}\)，所以通常 \(D_n\ne0\)。

还要注意：\(M_n\) 包含对 membrane derivative 的局部导数 \(\Lambda_n\)。当前 JAX 反向
会继续对 \(\Lambda_n\) 求导，因此 nonlinear current 的 exact program derivative 可能包含
membrane dynamics 的二阶导数。它不是“冻结 linearization coefficient 后”的近似梯度。

## 7. Linear solve 的 forward 与 reverse 梯度

对抽象线性系统：

\[
Mv=r,
\qquad
v=M^{-1}r,
\]

前向微分为：

\[
dv=M^{-1}(dr-dM\,v).
\]

所以 forward sensitivity 的每个 parameter tangent direction 都可以通过同一个消元程序求得
新的 tangent right-hand side，而不需要显式形成 dense $M^{-1}$。但不同 parameter direction
仍对应不同的 tangent column；tree/DHS 稀疏结构降低单列成本，不会让完整
$N_x\times N_\theta$ sensitivity 自动变成局部或稀疏对象。

reverse-mode 不需要显式形成 \(M^{-1}\)。给定输出 cotangent \(\bar v\)，先解伴随系统：

\[
\boxed{M^T\lambda=\bar v}
\]

然后：

\[
\boxed{\bar r=\lambda},
\qquad
\boxed{\bar M=-\lambda v^T}.
\]

当前 DHS 没有 custom VJP；forward elimination 和 back substitution 都由普通 JAX array
运算组成。JAX 对实际消元程序求导，其结果在系统 nonsingular、浮点误差可控的条件下与上述
隐式微分一致。因此 DHS 不会天然阻断 BPTT 或 forward-mode JVP，但病态或接近奇异的
\(M_n\) 仍会同时放大前向误差、tangent 和反向 cotangent。

## 8. Exact forward sensitivity

对完整状态映射：

\[
x_{n+1}=F_n(x_n,\theta,u_n),
\]

定义：

\[
S_n=\frac{\partial x_n}{\partial\theta}.
\]

则：

\[
\boxed{S_{n+1}=J_nS_n+G_n.}
\]

若 local loss 为 $l_n(x_n,\theta)$，可在 forward 中立即累计：

\[
\boxed{
g_n=g_{n-1}
+\frac{\partial l_n}{\partial x_n}S_n
+\frac{\partial l_n}{\partial\theta}
}.
\]

trial 内 $\theta$ 固定时，$g_n$ 是 prefix loss 的精确梯度，最终 $g_T$ 与下一节的 full BPTT
相同。该方法不保存长度为 $T$ 的 trajectory，但必须保存 $S_n$，memory 为
$O(N_xN_\theta)$。当前 BrainState `StatefulFunction.jaxpr_call` 可以把 stateful Cell step
临时暴露为 pure state-value mapping，再用 `jax.linearize`/JVP 推进 tangent；这不要求修改
DHS solver。

若 reset 或 gate initialization 依赖参数，必须使用：

\[
S_0=\frac{\partial x_0}{\partial\theta},
\]

不能一律设为零。完整实施和 online update 边界见
[单细胞 Exact Online Forward Sensitivity](./single-cell-online-forward-sensitivity.md)。

## 9. 多步 MSE 与离散伴随递推

假设观测 mask 选中的预测电压和 target 为 \(\widehat V_n,V_n^*\)。应将所有实际参与平均的
元素计入归一化：

\[
L
=
\frac{1}{N_{\mathrm{obs}}}
\sum_{n,p,c}
m_{n,p,c}
\left(
\frac{\widehat V_{n,p,c}-V^*_{n,p,c}}{V_{\mathrm{scale}}}
\right)^2.
\]

这里 \(p,c\) 可表示 population、probe 或 CV；\(V_{\mathrm{scale}}\) 使 loss 无量纲。只在
“每个时间步恰好一个标量观测”时，`1 / 100` 才等价于 100-step MSE 的完整平均。

令 \(l_n=l(x_n,\theta)\)，终点到起点的 reverse recursion 为：

\[
a_T=\frac{\partial l_T}{\partial x_T},
\]

\[
\boxed{
a_n
=
\frac{\partial l_n}{\partial x_n}
+J_n^Ta_{n+1}
}
\qquad n=T-1,\ldots,0.
\]

总参数梯度为：

\[
\boxed{
\frac{dL}{d\theta}
=
S_0^Ta_0
+
\sum_{n=0}^{T-1}G_n^Ta_{n+1}
+
\sum_{n=0}^{T}\frac{\partial l_n}{\partial\theta}
}
\]

若初始化与参数无关，$S_0=0$，第一项消失。

这同时包含：

- 参数对当前 voltage solve 的 direct effect；
- 参数对 post-voltage state update 的 direct effect；
- 旧状态经 \(J_n^T\) 传播到所有未来 loss 的 indirect effect；
- target preprocessing 或 regularization 显式依赖参数时的 loss direct effect。

在 BrainCell 中，多步 rollout 应放在 `brainstate.transform.for_loop`、`scan` 或它们的
checkpointed variants 内。Checkpoint/rematerialization 只以重计算换内存，不改变上述梯度；
truncated BPTT 或显式 detach 才会删除跨窗口的 \(J_n^T a_{n+1}\) 项。

## 10. 两种 spike 必须分开讨论

### 10.1 连续动作电位波形

HH channel 产生的 upstroke、peak、downstroke 和 AHP 都是连续状态方程的结果。对 voltage
MSE，loss 直接读取 \(V_n\)，梯度沿 DHS 和 channel dynamics 回传，不经过
`Cell.get_spike()`。

动作电位附近仍可能难训练，原因包括：

- channel rate 对 voltage 的斜率快速变化，使 \(D_n\) 很大或快速变号；
- conductance 对 gate 的高次幂使 \(B_n\) 对 gating state 很敏感；
- spike timing 的小偏移会让 prediction peak 对上 target baseline，反之亦然；
- 多步 \(J_n^T\) 连乘可能产生梯度消失、爆炸或强烈 phase sensitivity。

这些现象来自连续动力学，即使完全不计算离散 event spike 也存在。

### 10.2 `Cell.spike` surrogate crossing

完整动力学步之后，BrainCell 计算：

\[
z_+=\frac{V_{n+1}-V_{\mathrm{th}}}{\Delta V_s},
\qquad
z_-=\frac{V_{\mathrm{th}}-V_n}{\Delta V_s},
\qquad
\Delta V_s=20\,\mathrm{mV},
\]

\[
\boxed{
s_{n+1}=H_{\mathrm{sg}}(z_+)H_{\mathrm{sg}}(z_-)
}
\]

默认 `ReluGrad(alpha=0.3, width=1)` 的 forward 是 hard step，backward 使用有限支撑三角形：

\[
\rho(z)=\max\left(0,\alpha(\mathrm{width}-|z|)\right).
\]

因此程序采用的 surrogate partial derivative 为：

\[
\frac{\widetilde\partial s_{n+1}}{\partial V_{n+1}}
=
H(z_-)\frac{\rho(z_+)}{\Delta V_s},
\]

\[
\frac{\widetilde\partial s_{n+1}}{\partial V_n}
=
-H(z_+)\frac{\rho(z_-)}{\Delta V_s}.
\]

例如 \(V_n=-10\,\mathrm{mV}\)、\(V_{n+1}=10\,\mathrm{mV}\)、
\(V_{\mathrm{th}}=0\,\mathrm{mV}\) 时，forward spike 为 1，两个 normalized input 都是
0.5，故对 $V_{n+1}$ 和 $V_n$ 的梯度分别为 $+0.0075/\mathrm{mV}$ 和
$-0.0075/\mathrm{mV}$。当 normalized
input 离零超过 width 时，对应 surrogate gradient 为零。

只有下列路径会让该 surrogate 直接参与目标梯度：

- loss 显式读取 `Cell.spike` 或由它构造 filtered event trace；
- spike 经 network delivery 影响未来 synapse state 和 postsynaptic voltage；
- 其他 observable 或 learning rule 读取这一 event output。

仅拟合连续 voltage trace 时，surrogate crossing 不是必经路径。

## 11. Spike timing mismatch 与 loss cotangent

假设某一采样点 target 位于 \(40\,\mathrm{mV}\) 的 peak，而 prediction 因 timing 偏移仍在
\(0\,\mathrm{mV}\)。未归一化 squared error 的 local cotangent 为：

\[
\frac{\partial (V-V^*)^2}{\partial V}
=2(V-V^*)
\approx -80\,\mathrm{mV}.
\]

这只是注入 BPTT chain 的 local cotangent。最终参数梯度还取决于该 cotangent 经后续
\(J_n^T\) 和各步 \(G_n^T\) 的传播。

把 target peak 从 40 mV 压到 5 mV、降低 spike-window 权重或平滑 loss，都会改变
\(\partial L/\partial V_n\)，但不会改变 solver 的 \(J_n\)：

\[
\boxed{
\text{target/loss smoothing}
\ne
\text{dynamics-Jacobian smoothing}
}
\]

若在 backward 中人为把 \(D_n\) 替换为 \(\alpha D_n\)，则得到的是 biased/custom
gradient，不再是原 forward objective 的精确导数。当前 staggered solver 没有实现这种
gradient damping；若未来引入，必须通过显式 custom VJP、独立开关和有限差分对照记录其
语义。

更完整的 voltage/event composite loss、alignment、curriculum 和 parameter-system 设计见
[`voltage-and-spike-parameter-fitting.md`](./voltage-and-spike-parameter-fitting.md)。

## 12. 建议的梯度诊断

只观察最终 loss 无法区分 loss cotangent、动力学 Jacobian 和参数注入中的问题。针对一个
固定 protocol，至少记录：

1. 每步 \(\|\partial l_n/\partial V_n\|\) 与 cumulative adjoint \(\|a_n\|\)；
2. spike 前、upstroke、peak、downstroke、AHP 各窗口的 adjoint norm；
3. voltage block、mechanism block 对 \(J_n^Ta_{n+1}\) 的相对贡献；
4. 每个物理参数的 direct contribution \(G_n^Ta_{n+1}\) 及其跨时间累计；
5. AD gradient 与 central finite difference 或 directional JVP 的一致性；
6. \(M_n\) 的最小对角安全裕度、solve residual，以及 NaN/Inf 首次出现的时间步；
7. `dt`、precision、rollout length、checkpoint 配置和任何 truncation boundary。

有限差分只适合小模型和少量方向，并且在 hard event 边界附近会跨越不同 forward branch。
因此 continuous voltage objective 应优先做 AD/finite-difference 对照；event surrogate 则应另外
测试 hard forward value、surrogate support 和局部 backward slope。

## 13. 当前能力边界

可以从当前实现得出的结论：

- DHS linear solve 和 post-voltage JAX mechanism update 不会天然切断 reverse-mode BPTT；
- 同一 state transition 可以通过 forward-mode JVP 传播完整 sensitivity；
- 一步 Jacobian 具有 voltage/mechanism 双向 coupling；
- 默认 spike readout 是 hard forward、finite-support surrogate backward；
- `for_loop`/`scan` 中的多步 state transition 可以由 JAX/BrainState 反向传播。

不能从当前实现直接推出：

- 任意声明字段都能通过公共 API 变成 optimizer parameter；当前只支持 channel schema
  白名单中的连续参数；
- morphology、CV discretization、static axial coefficient 可微；
- hard spike time 的梯度是 continuous-time event 的精确梯度；
- target smoothing 会稳定动力学 Jacobian；
- 当前 first-order staggered split 具有一般二阶时间精度；
- 所有自定义 channel、synapse 或 independent solver 都没有 nondifferentiable operation。

因此，在声称某个具体模型“可训练”之前，仍应对该模型实际使用的参数路径、rollout、loss 和
event feedback 做端到端梯度验证。
