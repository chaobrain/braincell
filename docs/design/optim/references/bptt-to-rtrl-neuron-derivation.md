# 从 BPTT 到 RTRL：神经元模型的通用梯度理论

## Reference 状态

本文是 BrainCell 参数训练中 BPTT、exact real-time recurrent learning (RTRL) 和 online
forward sensitivity 的唯一通用理论稿。它只假设神经元可写成可微的离散状态映射，不依赖
Staggered、DHS、Euler 或其他具体 solver。

当前 Staggered 离散程序如何产生一步 Jacobian，见
[Staggered Solver 梯度分析](./staggered-solver-gradient-analysis.md)。

为避免与 Hodgkin-Huxley 的失活门变量 $h_t$ 冲突，完整神经元状态记为 $s_t$。全文采用
行导数约定：scalar 对向量的导数是行向量，链式法则按阅读顺序从左向右相乘，不额外使用
梯度简写或转置。

## 1. 问题定义

### 1.1 状态、参数和 voltage-only loss

任意选定的离散 solver 都定义一个一步映射：

$$
s_t=f_t(s_{t-1},\theta,u_t),
\qquad
s_t=
\begin{bmatrix}
v_t\\
w_t
\end{bmatrix}.
$$

- $v_t$：一个或多个 compartment/CV 的膜电位；
- $w_t$：所有会影响未来动力学的非电压状态，例如 gates、动态离子浓度和连续突触状态；
- $\theta$：一次 rollout 内固定的可训练参数；
- $u_t$：外部电流、clamp 或已经按时间对齐的固定事件输入。

电压是完整状态中的一个坐标块：

$$
v_t=P_vs_t,
\qquad
P_v=\begin{bmatrix}I_{N_v}&0\end{bmatrix}.
$$

对逐时间步 voltage loss：

$$
L=\sum_{t=1}^{T}\ell_t,
\qquad
\ell_t=\ell(v_t,v_t^{\mathrm{target}}),
$$

先定义 voltage learning signal：

$$
\varepsilon_t:=\frac{\partial\ell_t}{\partial v_t}.
$$

再把它放入完整状态的对应行：

$$
\boxed{
\frac{\partial\ell_t}{\partial s_t}
=
\varepsilon_tP_v
=
\begin{bmatrix}
\varepsilon_t&0
\end{bmatrix}
}.
$$

$w_t$ 的直接 local loss 导数为零，但它会改变未来的 $v$，因此不能从时序梯度中删除。

### 1.2 局部 Jacobian 与完整 sensitivity

定义一步 state Jacobian 和一步 parameter injection：

$$
J_t:=\frac{\partial s_t}{\partial s_{t-1}},
\qquad
G_t:=
\left.
\frac{\partial s_t}{\partial\theta}
\right|_{s_{t-1}}.
$$

$G_t$ 固定上一步状态，只描述参数在第 $t$ 步的局部作用。与之不同，完整 forward
sensitivity 为：

$$
S_t:=\frac{ds_t}{d\theta},
$$

它包含参数经过所有历史状态到达 $s_t$ 的路径。电压行块为：

$$
\boxed{
S_t^v:=P_vS_t=\frac{dv_t}{d\theta}
}.
$$

### 1.3 两个核心梯度公式

先假设 $ds_0/d\theta=0$，并且 loss 不在状态之外直接依赖参数。BPTT 按每一步的局部参数
注入分组：

$$
\boxed{
\frac{dL}{d\theta}
=
\sum_{t=1}^{T}
\underbrace{\frac{dL}{ds_t}}_{\text{全部未来 loss 对当前状态的 credit}}
\underbrace{G_t}_{\text{当前一步的局部参数注入}}
}
\qquad\text{BPTT}.
$$

RTRL 按当前 local loss 接收到的完整历史参数路径分组：

$$
\boxed{
\frac{dL}{d\theta}
=
\sum_{t=1}^{T}
\underbrace{\frac{\partial\ell_t}{\partial s_t}}_{\text{当前 local loss}}
\underbrace{S_t}_{\text{完整过去 sensitivity}}
}
\qquad\text{RTRL}.
$$

对 voltage-only loss，RTRL 的当前贡献进一步化为：

$$
\boxed{
\frac{d\ell_t}{d\theta}
=
\frac{\partial\ell_t}{\partial s_t}S_t
=
\varepsilon_tS_t^v
}.
$$

BPTT 把完整未来 credit 送回每次局部参数注入；RTRL 把完整过去 sensitivity 带到当前
local loss。二者只是同一组链式法则路径的不同求和顺序。

### 1.4 Shape 速查

| 符号 | Shape | 定义 |
| --- | --- | --- |
| $s_t$ | $N_s\times1$ | 完整动力学状态 $[v_t;w_t]$ |
| $v_t$ | $N_v\times1$ | 一个或多个 CV 的膜电位 |
| $w_t$ | $N_w\times1$ | 非电压动态状态 |
| $\theta$ | $N_\theta\times1$ | rollout 内固定参数 |
| $P_v$ | $N_v\times N_s$ | voltage row selector |
| $\partial\ell_t/\partial s_t$ | $1\times N_s$ | 当前 local loss 的状态导数 |
| $dL/ds_t$ | $1\times N_s$ | 全部未来 loss 的状态导数 |
| $J_t$ | $N_s\times N_s$ | 一步 state Jacobian |
| $G_t$ | $N_s\times N_\theta$ | 一步 parameter injection |
| $S_t=ds_t/d\theta$ | $N_s\times N_\theta$ | full forward sensitivity |
| $S_t^v=dv_t/d\theta$ | $N_v\times N_\theta$ | voltage sensitivity rows |
| $dL/d\theta$ | $1\times N_\theta$ | objective 的参数总导数 |

## 2. BPTT：向后传播未来 credit

终点为：

$$
\frac{dL}{ds_T}=\frac{\partial\ell_T}{\partial s_T}.
$$

反向递推为：

$$
\boxed{
\frac{dL}{ds_t}
=
\frac{\partial\ell_t}{\partial s_t}
+
\frac{dL}{ds_{t+1}}J_{t+1}
},
\qquad t=T-1,\ldots,1.
$$

第 $t$ 步局部参数注入的贡献为 $(dL/ds_t)G_t$，所以：

$$
\frac{dL}{d\theta}
=
\sum_{t=1}^{T}
\frac{dL}{ds_t}G_t.
$$

BPTT 通常保存或 rematerialize 前向轨迹，再从 $T$ 到 $1$ 传播一个
$1\times N_s$ cotangent。

## 3. RTRL：向前传播过去 sensitivity

对一步状态映射应用链式法则：

$$
\boxed{
S_t=J_tS_{t-1}+G_t
}.
$$

展开后：

$$
S_t
=
J_tJ_{t-1}\cdots J_1S_0
+
\sum_{k=1}^{t}
J_tJ_{t-1}\cdots J_{k+1}G_k.
$$

当 $k=t$ 时，$G_t$ 左侧的 Jacobian 乘积为空，按单位矩阵处理。在线累计 prefix gradient：

$$
\Gamma_0=0,
$$

$$
\boxed{
\Gamma_t
=
\Gamma_{t-1}
+
\frac{\partial\ell_t}{\partial s_t}S_t
}.
$$

在参数不变且 sensitivity 不被截断时，$\Gamma_T=dL/d\theta$，与 full BPTT 相同。

### 3.1 两步等价性

只考虑 $L=\ell_1+\ell_2$，并令 $S_0=0$：

$$
S_1=G_1,
\qquad
S_2=J_2G_1+G_2.
$$

RTRL 为：

$$
\frac{dL}{d\theta}
=
\frac{\partial\ell_1}{\partial s_1}G_1
+
\frac{\partial\ell_2}{\partial s_2}(J_2G_1+G_2).
$$

BPTT 为：

$$
\frac{dL}{ds_2}=\frac{\partial\ell_2}{\partial s_2},
$$

$$
\frac{dL}{ds_1}
=
\frac{\partial\ell_1}{\partial s_1}
+
\frac{\partial\ell_2}{\partial s_2}J_2,
$$

$$
\begin{aligned}
\frac{dL}{d\theta}
&=
\frac{dL}{ds_1}G_1
+
\frac{dL}{ds_2}G_2\\
&=
\frac{\partial\ell_1}{\partial s_1}G_1
+
\frac{\partial\ell_2}{\partial s_2}(J_2G_1+G_2).
\end{aligned}
$$

### 3.2 不能同时使用两边的完整路径

下面的表达一般是错误的：

$$
\sum_t
\frac{dL}{ds_t}
\frac{ds_t}{d\theta}.
$$

$dL/ds_t$ 已包含从 $s_t$ 到未来 loss 的完整路径，$ds_t/d\theta$ 又包含从过去参数注入到
$s_t$ 的完整路径。对所有中间状态求和会重复计算经过多个状态的同一条路径。正确规则只能
选择一种分组：

- full future derivative $dL/ds_t$ 乘 local injection $G_t$；
- local loss derivative $\partial\ell_t/\partial s_t$ 乘 full past sensitivity $S_t$。

## 4. 神经元状态的 v/w 通用分块

不指定 solver 时，一步 Jacobian 仍可按状态坐标分块：

$$
J_t=
\begin{bmatrix}
J_t^{vv}&J_t^{vw}\\
J_t^{wv}&J_t^{ww}
\end{bmatrix},
\qquad
G_t=
\begin{bmatrix}
G_t^v\\
G_t^w
\end{bmatrix}.
$$

这些 block 只表示所选离散映射的局部导数：

$$
J_t^{vv}=\frac{\partial v_t}{\partial v_{t-1}},
\qquad
J_t^{vw}=\frac{\partial v_t}{\partial w_{t-1}},
$$

$$
J_t^{wv}=\frac{\partial w_t}{\partial v_{t-1}},
\qquad
J_t^{ww}=\frac{\partial w_t}{\partial w_{t-1}},
$$

$$
G_t^v=
\left.\frac{\partial v_t}{\partial\theta}\right|_{s_{t-1}},
\qquad
G_t^w=
\left.\frac{\partial w_t}{\partial\theta}\right|_{s_{t-1}}.
$$

RTRL 分块递推为：

$$
\boxed{
S_t^v
=J_t^{vv}S_{t-1}^v
+J_t^{vw}S_{t-1}^w
+G_t^v
},
$$

$$
\boxed{
S_t^w
=J_t^{wv}S_{t-1}^v
+J_t^{ww}S_{t-1}^w
+G_t^w
}.
$$

Voltage-only loss 只在当前乘法中读取 $S_t^v$：

$$
\frac{d\ell_t}{d\theta}=\varepsilon_tS_t^v.
$$

但不能只保存 $S_t^v$。一般情况下，$S_t^w$ 会在下一步通过
$J_{t+1}^{vw}S_t^w$ 返回 voltage path。只有额外结构证明该路径恒为零时，删除 $S_t^w$
才保持 exact。

BPTT 也可以按相同坐标分块。令：

$$
\frac{dL}{ds_t}=\begin{bmatrix}a_t^v&a_t^w\end{bmatrix},
$$

则：

$$
a_t^v
=
\varepsilon_t
+a_{t+1}^vJ_{t+1}^{vv}
+a_{t+1}^wJ_{t+1}^{wv},
$$

$$
a_t^w
=
a_{t+1}^vJ_{t+1}^{vw}
+a_{t+1}^wJ_{t+1}^{ww}.
$$

具体 solver 决定这些 block 的数值和内部求导路径，但不改变上述重排关系。

## 5. Worked example：1 CV、2 个 HH 参数

考虑一个 CV 的 HH cell，只训练 Na/K 最大电导：

$$
s_t=
\begin{bmatrix}
v_t\\m_t\\h_t\\n_t
\end{bmatrix},
\qquad
\theta=
\begin{bmatrix}
\bar g_{\mathrm{Na}}\\
\bar g_{\mathrm{K}}
\end{bmatrix}.
$$

通道电流为：

$$
I_{\mathrm{Na},t}
=\bar g_{\mathrm{Na}}m_t^3h_t(v_t-E_{\mathrm{Na}}),
$$

$$
I_{\mathrm{K},t}
=\bar g_{\mathrm{K}}n_t^4(v_t-E_{\mathrm{K}}).
$$

选定 solver 后，整个 HH timestep 仍统一写成：

$$
s_t=f_t(s_{t-1},\bar g_{\mathrm{Na}},\bar g_{\mathrm{K}},u_t).
$$

矩阵尺寸为：

| 对象 | Shape |
| --- | ---: |
| $J_t$ | $4\times4$ |
| $G_t$ | $4\times2$ |
| $S_t$ | $4\times2$ |
| $S_t^v$ | $1\times2$ |

使用无量纲 voltage MSE：

$$
\ell_t
=\frac12
\left(
\frac{v_t-v_t^{\mathrm{target}}}{v_{\mathrm{scale}}}
\right)^2,
$$

则：

$$
\varepsilon_t
=
\frac{v_t-v_t^{\mathrm{target}}}{v_{\mathrm{scale}}^2},
$$

$$
\frac{\partial\ell_t}{\partial s_t}
=
\begin{bmatrix}
\varepsilon_t&0&0&0
\end{bmatrix},
$$

$$
S_t^v
=
\begin{bmatrix}
\dfrac{dv_t}{d\bar g_{\mathrm{Na}}}&
\dfrac{dv_t}{d\bar g_{\mathrm{K}}}
\end{bmatrix}.
$$

因此当前 local loss 的梯度贡献为：

$$
\boxed{
\frac{d\ell_t}{d\theta}
=
\varepsilon_t
\begin{bmatrix}
\dfrac{dv_t}{d\bar g_{\mathrm{Na}}}&
\dfrac{dv_t}{d\bar g_{\mathrm{K}}}
\end{bmatrix}
}.
$$

Gate sensitivity 没有直接出现在最后一行，但已通过 $S_t^w$ 和后续
$J_{t+1}^{vw}$ 进入 voltage sensitivity。例如两步展开为：

$$
S_2^v
=J_2^{vv}G_1^v
+J_2^{vw}G_1^w
+G_2^v.
$$

第二项正是参数先改变 $w_1=(m_1,h_1,n_1)$，再由 gate-dependent current 改变 $v_2$
的路径。选择不同 solver 会改变 $J$ 和 $G$ 的具体数值，但不会删除这类依赖。

## 6. 初始化、direct term 与 online 边界

### 6.1 参数相关初始化

只有初始状态与参数无关时，才能令 $S_0=0$。一般情况为：

$$
s_0=I(\theta),
\qquad
S_0=\frac{ds_0}{d\theta}.
$$

例如 gate steady-state 初始化可能依赖初始电压和 kinetics parameter。忽略这条路径会让
BPTT 和 RTRL 同时缺少 initialization gradient。

### 6.2 Loss 直接依赖参数

若 loss 在状态之外还直接依赖参数，BPTT 的完整形式为：

$$
\frac{dL}{d\theta}
=
\frac{dL}{ds_0}\frac{ds_0}{d\theta}
+
\sum_{t=1}^{T}
\frac{dL}{ds_t}G_t
+
\sum_{t=1}^{T}
\left.
\frac{\partial\ell_t}{\partial\theta}
\right|_{s_t}.
$$

RTRL 从正确的 $S_0$ 开始，并在每步加入 direct term：

$$
\frac{dL}{d\theta}
=
\sum_{t=1}^{T}
\left(
\frac{\partial\ell_t}{\partial s_t}S_t
+
\left.
\frac{\partial\ell_t}{\partial\theta}
\right|_{s_t}
\right).
$$

### 6.3 “Online” 的不同语义

| 运行方式 | Sensitivity/参数边界 | 与 fixed-parameter full BPTT 的关系 |
| --- | --- | --- |
| rollout 内固定 $\theta$，末尾更新 | 全程 carry $S_t$ | 完全等价 |
| 固定 $\theta$，逐步读取 prefix gradient | 全程 carry $S_t$ | 每个 prefix 都精确 |
| 分 chunk 但不更新参数 | 跨 chunk carry $S_t$ | 完全等价 |
| chunk 后更新并清空 $S_t$ | dynamic state 保留，gradient state detach | truncated、biased |
| 参数更新后继续使用旧 $S_t$ | sensitivity 来自旧参数轨迹 | adaptive heuristic |
| 每时间步更新 $\theta_t$ | 参数序列进入动力系统 | 不是原 fixed-parameter objective |

Optimizer 若逐步消费梯度，应消费本步新增 contribution，而不是反复消费累计 prefix gradient。
一旦 rollout 中间更新参数，问题已经变成新的 online adaptive algorithm。

### 6.4 固定外源 delay 与 event feedback

固定的外源 event 和 delay 可以预先编码进 $u_t$，不会引入从 cell state 到未来 input 的
梯度路径。由外源 event 驱动的连续 synapse state 仍属于 $s_t$。

以下情况必须扩展状态或重新定义导数语义：

- delay 自身可训练；
- cell-generated spike 经 ring buffer 反馈到未来状态；
- loss 直接依赖 hard event 或 continuous event time。

## 7. Exact RTRL 与 e-prop 的边界

Exact full-state RTRL 直接保存：

$$
S_t=\frac{ds_t}{d\theta}.
$$

它不要求 neuron-local factorization，也不自动具有局部内存规模。e-prop 常写成：

$$
\frac{dE}{dW_{ji}}
=
\sum_t L_j^t e_{ji}^t,
$$

其中 eligibility $e_{ji}^t$ 是可正向递推的局部因子，learning signal $L_j^t$ 提供误差信息。
Eligibility 的代数因子化本身不必是近似；online e-prop 的近似通常来自用当前可得的局部
learning signal 代替包含未来或非局部影响的 ideal signal，并可能额外近似 reset、spike
derivative 或 eligibility locality。

因此，出现“trace 乘 local signal”的形式并不足以把 full-state RTRL 称为 e-prop。本文其余
公式讨论的是 exact sensitivity；任何 block-local、low-rank 或 surrogate learning signal 都应
显式标记为近似。

参考：[Bellec et al., 2020](https://doi.org/10.1038/s41467-020-17236-y)。

## 8. 工程代价与 exact 边界

设状态 DOF 为 $N_s$、参数 DOF 为 $N_\theta$、rollout 长度为 $T$：

| 维度 | BPTT | Exact RTRL |
| --- | --- | --- |
| 传播方向 | backward in time | forward in time |
| 主要对象 | $dL/ds_t\in\mathbb R^{1\times N_s}$ | $S_t\in\mathbb R^{N_s\times N_\theta}$ |
| 主要存储 | trajectory/tape，约 $O(TN_s)$ | current sensitivity，约 $O(N_sN_\theta)$ |
| 每步微分 | 一个 VJP/cotangent | $N_\theta$ 个 tangent directions |
| Prefix gradient | 需要 reverse prefix | 前向时立即可得 |

Checkpoint/rematerialization 可以用重算降低 BPTT tape。RTRL memory 与时间长度无关，但
推进全部 parameter directions 的计算和 carry 会随 $N_\theta$ 增长。具体墙钟还由 solver、
硬件并行度、batch、静态 shape 和编译器调度决定，不能仅由大 O 排序。

以下条件下，RTRL 与 full BPTT 对同一个离散 objective 完全等价：

1. rollout 内 $\theta$ 固定；
2. $s_t$ 包含所有会影响未来 loss 的 dynamic state；
3. $S_0=ds_0/d\theta$ 被正确处理；
4. sensitivity 不跨 chunk 清零或 detach；
5. local loss、direct term 和 learning signal 被完整计入；
6. 两者对同一个实际离散 solver program 求导。

连续 HH 动作电位属于 $v/w$ 动力学。只要 loss 读取 voltage trace 而不读取离散 spike
readout，梯度不需要经过 surrogate event。具体 solver 和 readout 的程序导数见独立 solver
文档。

## 9. 建议的 PPT 页面顺序

1. $s_t=(v_t,w_t)$ 与 voltage-only loss；
2. BPTT 和 RTRL 两个核心求和公式；
3. BPTT future-credit 递推；
4. RTRL past-sensitivity 递推；
5. 两步展开与防重复计数；
6. $S_t^v=dv_t/d\theta$ 和不可省略的 gate path；
7. 1 CV、$\bar g_{\mathrm{Na}}/\bar g_{\mathrm K}$ 示例；
8. online/e-prop 边界与 $O(TN_s)$、$O(N_sN_\theta)$ 权衡。
