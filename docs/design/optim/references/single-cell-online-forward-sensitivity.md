# 单细胞 Exact Online Forward Sensitivity

## Reference 状态

本文给出 BrainCell multi-compartment Cell 在以下边界内的 exact online-gradient 设计与验证
方案：

- 一个 Cell，允许任意数量 compartment；
- channel parameter 在一次 rollout 内固定；
- loss 是逐时间步可计算的 $L_T=\sum_t\ell_t$；
- synapse/input 可以有不同固定 delay，但事件流是外源常量；
- 不训练 delay，不以离散 `Cell.spike` 为 loss，也没有 spike event feedback。

本文不定义公共 API。BPTT 和 e-prop 的一般路径关系见
[从 e-prop 到 HH](./eprop_to_hh_summary.md)，staggered 离散映射见
[Staggered Solver BPTT 分析](./staggered_solver_BPTT_spike_gradient_analysis.md)。

## 1. 精确问题定义

将所有会影响未来连续动力学的状态收集为 $x_t$：

$$
x_t=(V_t,Q_t,R_t).
$$

- $V_t$：全部 CV 电压；
- $Q_t$：gates、ion concentration、continuous synapse states；
- $R_t$：目标或 observable 所需的 causal filter state；没有 filter 时为空。

预先准备好的外部 current、clamp、event payload 和固定 delay 后的输入记为 $u_t$。离散
staggered program 定义：

$$
x_{t+1}=F_t(x_t,\theta,u_t),
\qquad
\ell_{t+1}=\ell(x_{t+1},\theta,y_{t+1}^*).
$$

这里追求的是当前离散程序的 exact program derivative，不是 underlying continuous ODE 的
解析梯度，也不是 continuous-time hard event-time derivative。

## 2. Forward carry

参数在 rollout 内固定时，online carry 只需要：

$$
\mathcal{C}_t=(x_t,S_t,g_t),
$$

其中：

$$
S_t=\frac{\partial x_t}{\partial\theta},
\qquad
g_t=\frac{d}{d\theta}\sum_{k=0}^{t}\ell_k.
$$

每步执行：

$$
S_{t+1}=J_tS_t+G_t,
$$

$$
\Delta g_{t+1}
=
\frac{\partial\ell_{t+1}}{\partial x_{t+1}}S_{t+1}
+\frac{\partial\ell_{t+1}}{\partial\theta},
$$

$$
g_{t+1}=g_t+\Delta g_{t+1}.
$$

实现不应显式构造 $J_t$ 和 $G_t$。将 state values 和 parameter roots 作为纯函数输入，对一步
program 调用 `jax.linearize`，再把 linear map 应用于各 parameter tangent direction，即可同时
得到 $S_{t+1}$ 和 $\Delta g_{t+1}$。

## 3. 初始化不是默认零 sensitivity

只有 $x_0$ 与参数无关时才能令 $S_0=0$。BrainCell 的 `reset_state()` 可能执行：

- trainable root 到 runtime physical field 的 materialization；
- 依赖 $V_{init}$ 和 kinetics parameter 的 gate steady-state 初始化；
- dynamic ion 或 mechanism state reset。

因此 exact 实现需要把初始化也函数化：

$$
x_0=I(\theta),
\qquad
S_0=\frac{\partial I}{\partial\theta}.
$$

如果当前实验只训练 `g_max` scale，且 reset gates 不依赖 `g_max`，零初始化恰好成立；这不能
推广到 kinetics shift、parameterized initial state 或未来可训练 concentration。

## 4. Stateful Cell 的纯函数边界

当前 `Cell.update()` 通过 `brainstate.State` 读写模型状态。验证原型采用：

1. 用 `brainstate.transform.StatefulFunction(return_only_write=False)` trace
   `materialize -> update -> local loss`；
2. 取得稳定的 state object 顺序和 state-value PyTree；
3. 通过 `StatefulFunction.jaxpr_call(state_values, step_data)` 暴露纯函数：

   ```text
   (state_values_t, step_data_t)
       -> (state_values_t+1, local_loss_t+1)
   ```

4. 在 parameter `ParamState` 对应的输入 leaf 上放置 basis tangent，其他 leaf 为零；
5. 用 `jax.linearize` 加 batched tangent propagation 推进 sensitivity；
6. 用 `brainstate.transform.scan` carry state、sensitivity 和累计梯度。

`materialize()` 在验证版每步执行，以显式保留 root 到 runtime parameter buffer 的路径。之后若
需要性能优化，可以把 materialized parameter buffer 作为 rollout-constant pure input，但必须用
等价性测试证明没有切断 root mapping。

`StatefulFunction` 会捕获一些不参与目标路径的辅助 state，例如未被 loss/feedback 使用的
`spike`。携带它们不会改变梯度，但会增加原型 sensitivity 的尺寸。只有建立明确的 state-role
schema 后，才能安全删除这些 inert leaves；第一版以完整 capture 保证正确性。

## 5. 不同 delay 的外部输入

固定外源事件可以先变成时间对齐数组：

$$
u_t=\sum_k\operatorname{deliver}(e_k,t-d_k).
$$

$e_k$ 和 $d_k$ 对 $\theta$ 都是常量，因此它们只作为 `scan` 的 `xs` 输入。若 payload 驱动
ExpSyn 等连续机制，synapse state 会被 `StatefulFunction` 捕获并进入 $S_t$。以下对象无需
parameter tangent：

- 固定 event timestamp；
- 固定、量化后的 delay index；
- 只依赖外源事件的预处理 ring buffer。

如果未来 cell-generated spike 写入 ring buffer，则 ring state 会影响未来 cell state，必须加入
完整 recurrent state，并明确使用 hard、surrogate 还是 continuous-time event derivative。那是
另一问题，不应由本原型暗中支持。

## 6. “Online” 的六种不同语义

| 运行方式 | sensitivity/参数边界 | 与原 full BPTT 的关系 |
| --- | --- | --- |
| trial 内固定 $\theta$，逐步累计，末尾更新 | 全程 carry $S_t$ | 完全等价 |
| 固定 $\theta$，每步读取 prefix gradient | 全程 carry $S_t$ | 每个 prefix 都精确 |
| 分 chunk 计算但不更新 | 跨 chunk carry $S_t$ | 完全等价 |
| chunk 更新并清空 $S_t$ | dynamic state carry，gradient state detach | truncated、biased |
| chunk 更新并继续旧 $S_t$ | 不再对应单一固定 $\theta$ 轨迹 | adaptive first-order heuristic |
| 每时间步更新 $\theta_t$ | dynamics 由参数序列驱动 | 不等价于 fixed-parameter BPTT |

如果 optimizer 在每步消费梯度，必须使用本步新增的 $\Delta g_t$；反复使用累计 $g_t$ 会让
过去 loss 被重复计数。即便只使用 $\Delta g_t$，参数更新后实际 state 是旧参数轨迹的结果，
后续学习也已成为 online adaptive algorithm，而不是原 objective 的 exact optimizer step。

若要对参数更新规则本身求 meta-gradient，必须把 optimizer state 和
$\theta_{t+1}=U(\theta_t,\Delta g_t)$ 一起纳入更大的动力系统；这会重新产生跨整个训练过程的
credit assignment，不能由当前一个固定大小的 $S_t=\partial x_t/\partial\theta$ 自动解决。

## 7. 与 BPTT 的工程权衡

设 state DOF 为 $N_x$、parameter DOF 为 $N_\theta$、步数为 $T$：

| 方法 | 主要时间历史 memory | 主要 tangent/cotangent 工作 |
| --- | --- | --- |
| reverse BPTT | $O(TN_x)$，可 checkpoint | 每步一个 loss cotangent/VJP |
| exact forward | $O(N_xN_\theta)$ | 每步 $N_\theta$ 个 tangent direction |

对 DHS cable，单个 tangent direction 可以沿当前 elimination/back-substitution program 求导，
不需要 dense inverse；但所有 parameter columns 的总体成本仍大致随 $N_\theta$ 增长。Axial
coupling 会使局部参数的 sensitivity 沿 morphology 传播，因此按 CV/channel 保存局部标量
eligibility 一般不是 exact 方法。

适合 exact forward 的区域：

- rollout 很长而 trainable DOF 较少；
- 需要实时读取 prefix gradient；
- 无法保存或重算长时间历史；
- 单 Cell 参数经过 `group_by="all"`、`population` 或低维 parameterized source 压缩。

不适合的区域：

- 每个 CV/channel row 都有独立参数；
- 大 network 中每条 connection 都可训练；
- 只在 trial 末尾需要一个 scalar loss，且 checkpointed reverse mode 已满足内存预算。

## 8. Core 函数边界

非公共验证原型位于
[`forward_sensitivity_core.py`](../../../../examples/experimental/online_learning/forward_sensitivity_core.py)，
对应等价性用例位于
[`forward_sensitivity_core_test.py`](../../../../examples/experimental/online_learning/forward_sensitivity_core_test.py)。
核心只需要以下函数：

```text
build_stateful_step(step_fn, example_step_data, parameter_states)
    -> FunctionalStep

seed_scalar_parameter_directions(functional_step, state_values)
    -> batched_state_tangents

build_parameter_coordinates(functional_step, state_values)
    -> ParameterCoordinates

build_active_state_projection(functional_step, selections, state_values)
    -> ActiveStateProjection

forward_sensitivity_step(functional_step, state_values,
                         state_tangents, step_data)
    -> next_state_values, next_state_tangents,
       local_loss, local_gradient_contribution

forward_sensitivity_rollout(functional_step, initial_state_values,
                            initial_state_tangents, step_data)
    -> losses, local_gradients, prefix_gradients

compact_forward_sensitivity_rollout(functional_step, projection,
                                    parameter_coordinates,
                                    initial_state_values,
                                    initial_active_state_tangents,
                                    step_data)
    -> losses, local_gradients, prefix_gradients

initialize_forward_sensitivity(initializer_step, functional_step,
                               initializer_data)
    -> reset_state_values, reset_state_tangents

bptt_reference_loss(functional_step, initial_state_values, step_data)
    -> scalar_loss

initialized_bptt_reference_loss(initializer_step, functional_step,
                                initializer_state_values,
                                initializer_data, step_data)
    -> scalar_loss
```

初始化扩展使用独立的 state transform trace reset/materialization，并按 state object identity
把输出 state 与 step trace 对齐。实验 core 现在会把单 array-leaf optimizer root 按 stable root
顺序展开为 coordinate basis；例如三个 shape `(3,)` 的 row roots 形成 9 个方向。任意多-leaf
root 和 public physical-unit flattening 合同仍不在该 prototype 范围内。

实验 core 接受显式 `ActiveStateSelection`，按 state object identity 和 active flattened indices
在 full runtime tangent 与 compact `(N_\theta,N_x)` sensitivity 之间 gather/scatter。Density
runtime 迁移到 CV space 后，endpoint gate padding 已从 full tangent 消失；projection 主要继续
排除无反馈辅助 state。省略的 state 必须被证明不会影响未来 loss。

## 9. 验证与已有证据

在当前仓库上已经做过两个无文件修改的 feasibility check：

1. 两-compartment Leak、一个 scale parameter、8 步 voltage MSE：forward JVP 累计梯度
   `-25.068812955953636`，reverse BPTT `-25.068812955953646`，绝对误差约
   $1.1\times10^{-14}$；
2. 两-compartment HH、Na/K 两个 scale parameter、21 个 captured runtime states、4 步：
   forward 与 reverse 最大绝对误差约 $1.6\times10^{-15}$。

当前自动化原型测试已经覆盖：

- 每个 prefix gradient 与相同 prefix loss 的 reverse-mode gradient；
- central finite-difference directional derivative；
- reset 对参数有依赖时的 $S_0$；
- 两个固定 delay 的外源输入；
- carry shape 不含 $T$ 轴；
- x64 下 forward/reverse `rtol <= 1e-8`。

x32 的误差范围、真实 morphology 的 memory scaling 和 batched tangent throughput 留给后续性能
实验；它们不改变本轮的数学等价性结论。

### 可配置 Multi-CV、row 参数扩展

[`multicv_hh_rtrl.py`](../../../../examples/experimental/online_learning/multicv_hh_rtrl.py)
按 `dendrite_segments=(left_count, right_count)` 构造一个 soma distal bifurcation 连接两条
segmented dendrite arms 的 Cell。每个 branch 由 `CVPerBranch` 产生一个 CV，每个 CV 都有
Leak、Na_HH1952 和 K_HH1952，三个 `g_max` scale 使用 `group_by="row"`。因此对任意
$C=1+left_count+right_count$：

```text
active state = V[C] + Na.m[C] + Na.h[C] + K.n[C] = 4C
parameter DOF = leak[C] + Na[C] + K[C] = 3C
compact sensitivity shape = (3C, 4C)
```

三 CV `dendrite_segments=(1,1)` 回归为：

```text
active state = V[3] + Na.m[3] + Na.h[3] + K.n[3] = 12
parameter DOF = leak[3] + Na[3] + K[3] = 9
compact sensitivity shape = (9, 12)
```

五 CV `dendrite_segments=(2,2)` 自动扩展为：

```text
active state = 20
parameter DOF = 15
compact sensitivity shape = (15, 20)
x64 sensitivity carry = 2,400 bytes
```

初值相等的 row roots 不能按当前数值降为 `uniform` runtime storage，否则后两个 row 的梯度
会被投影掉。`TrainableManager` 因此使用 binding grouping 与数值可压缩轴的 join：row binding
始终保持 `row`，population/cv binding 至少保持各自声明轴，只有 `all` 且 baseline 允许时才
保持 `uniform`。

x64、CPU-only JAX、2000 steps (`dt=0.025 ms`, 50 ms) 的一次本机测量为：

| Method | Steady median | XLA temporary | Sensitivity carry |
| --- | ---: | ---: | ---: |
| reverse BPTT | 91.4 ms | 6.81 MB | temporal tape |
| full-state-tree RTRL | 50.1 ms | 8.8 KB | 5,040 bytes |
| compact 12x9 RTRL | 51.0 ms | 9.6 KB | 864 bytes |

compact/full/BPTT 最大绝对梯度误差分别约为 `2.7e-12` 和 `7.1e-11`。在该小模型中 compact
projection 主要把 carry 减少 5.83 倍，没有降低实际 whole-cell JVP 工作量，所以速度与 full
RTRL 相同。不同 dendrite 的 row parameter 对彼此 voltage sensitivity 非零，确认 exact
projection 没有按 compartment 截断 cable coupling。XLA memory analysis 是静态 buffer 统计，
不等同于进程峰值 RSS；性能数字也只适用于记录的 CPU backend。

同样配置下，五 CV、15 row 参数的 50 ms 测量为：

| Method | Steady median | XLA temporary | Sensitivity carry |
| --- | ---: | ---: | ---: |
| reverse BPTT | 96.8 ms | 11.19 MB | temporal tape |
| full-state-tree RTRL | 54.0 ms | 20.0 KB | 12,720 bytes |
| compact 20x15 RTRL | 61.3 ms | 22.6 KB | 2,400 bytes |

五 CV compact/full/BPTT 最大绝对梯度误差分别约为 `1.8e-12` 和 `3.9e-12`。三 CV与五 CV
自动化测试共同验证 coordinate basis、central directional finite difference、row runtime axis
和两条 distal dendrite arms 之间的非零 sensitivity。compact projection 仍主要减少 carry；它
每步会嵌回 full runtime tangent，因此没有减少 whole-cell JVP 的实际方向数和 solver 工作。

### Density runtime 迁移到 CV space

Painted `Density` channel/ion runtime 已从 point-tree dense storage 迁移到 CV dense storage。
Point-tree 仍作为 DHS algebraic workspace；placed synapse、clamp 和其他 point mechanism 继续
使用 sparse point layouts。对五 CV bifurcation：

```text
                         before       after
n_cv                        5            5
n_point (DHS workspace)     11           11
Na.m / Na.h / K.n shape     (1,11)       (1,5)
density layout axis         11           5
captured state bytes        848          560
full RTRL tangent carry     12,720       8,400
compact 20x15 carry         2,400        2,400
```

迁移前保存的 x64、16-step target voltage 与 loss 在迁移后位级一致；compact/full/BPTT
gradient 最大绝对变化不超过 `3.5e-18`，compact sensitivity 最大绝对变化约 `6.9e-17`。

同一 CPU-only JAX process protocol、x64、2000 steps、十次稳态中位数：

| Kernel | Before time | After time | Before temporary | After temporary |
| --- | ---: | ---: | ---: | ---: |
| primal terminal rollout | 9.43 ms | 6.89 ms | 3.07 KB | 2.22 KB |
| reverse BPTT | 91.86 ms | 104.16 ms | 11.19 MB | 7.14 MB |
| full-state-tree RTRL | 57.29 ms | 30.90 ms | 20.01 KB | 13.15 KB |
| compact-adapter RTRL | 71.24 ms | 36.55 ms | 22.64 KB | 15.28 KB |

Primal 减少约 27% 时间和 28% temporary；full RTRL 减少约 46% 时间、34% temporary 与
34% carry。BPTT temporal tape 明显下降约 36%，但本机墙钟在两次后测中约 104--106 ms，
比单次迁移前基线更慢；CPU run-to-run 抖动和新的 CV-space reverse graph 都可能贡献，不能由
该数字声称 BPTT 加速。内存和 state-shape 变化是编译器静态分析与结构断言的稳定结果。

Partial channel coverage 本轮保持历史行为：任意正面积覆盖即在整个 CV 上使用声明值，
`coverage_area_fraction` 仍只作为元数据；partial layout 使用 CV mask，尚未压成 `n_active`
dynamic state。下一阶段 active-packed runtime 才会删除未 paint CV 的 gate state。

### Experimental rollout gradient engine

Full RTRL 与 BPTT 共享同一个 stateful step，并在 experimental 层提供统一调用：

```python
from examples.experimental.online_learning.rollout_gradients import (
    build_rollout_value_and_grad,
)

def rollout_step(target_mv):
    cell.update()
    error = cell.V.value.to_decimal(u.mV) - target_mv
    return jnp.mean(error * error)

engine = build_rollout_value_and_grad(
    cell,
    step=rollout_step,
    method="rtrl",  # or "bptt"
)
result = engine(target_voltage_mv)
optimizer.update(result.gradients)
```

参数默认来自 `cell.trainables.parameters().states()`；每步和默认 reset 前由 engine 自动
materialize。正常路径只返回 `losses`、总 `loss` 和命名参数 `gradients`。Full RTRL 的当前
sensitivity 只存在于 scan carry，不输出时间历史，也不显式形成 `A_t`、`P_t` 或 `L_t`。

`engine.diagnose(step_data, at=(...))` 是单独的 RTRL 分析路径。它仅在指定时间点返回 full-state
`S_t`、learning signal `L_t`、direct parameter term、eligibility contraction、local/prefix
gradient 和 decomposition residual。诊断会增加计算和 `O(n_sample N_z N_theta)` 输出内存，
不会进入正常训练 HLO。第一版不物化 `A_t` 或 `P_t`。

五 CV、x64 CPU、2000 steps、十次稳态测量中，统一 API 的 full RTRL 为 `29.35 ms`、
XLA temporary `14,968 bytes`；BPTT 为 `102.33 ms`、`7,125,584 bytes`。两者都返回
`(2000,)` losses、标量总 loss 和三个 `(5,)` 参数梯度，output allocation 为 `16,168 bytes`。
RTRL/BPTT 梯度最大相对差为 `5.4e-12`。相对 terminal-only RTRL，新增内存主要是调用方明确
要求保留的 2000 个标量 loss；normal result 不含 sensitivity、local-gradient 或 prefix-gradient
history。

若 objective 不能写成逐步 scalar loss 的和，例如它包含整条 voltage trace 的均值、相邻时刻
差分或其他跨时间耦合，可显式改用 `build_trajectory_value_and_grad`。其 BPTT 路径对一次完整
observation rollout 做 reverse mode；exact RTRL 路径执行 two-pass：

1. 第一遍生成 observation trace，并只对全局 loss 关于 trace 求导，得到各时刻 learning signal；
2. 第二遍重放相同 transition，递推当前 full sensitivity，并与该时刻 learning signal 立即收缩。

因此无需用户手工拆分 `partial L / partial observation_t`，也不截断时间梯度。代价是比 local
one-pass RTRL 多一次仿真，以及 `O(T * N_observation)` 的 trace/learning-signal 存储；它仍不保存
`O(T * N_state * N_parameter)` 的 sensitivity history。正常返回值只含 scalar loss 和总梯度。
该接口当前要求 global loss 通过 observation trace 依赖参数，不包含参数的显式 direct term。

### Performance Evidence

A100 scaling、seed-block/batch-shared sweep、large-CV 和 ordinary/recursive backsub A/B 的
详细硬件结果不属于本理论 reference 的稳定合同。原始 CSV、NPZ、manifest 和 worker 日志位于
本地 ignored 目录：

```text
examples/experimental/online_learning/artifacts/rtrl_bptt_scaling/
```

运行 `rtrl_bptt_scaling_report.py` 会在该目录生成 `RESULTS.md`；代码结构、suite、复现命令和
notebook 入口见 [Online Learning Experiments](../../../../examples/experimental/online_learning/README.md)。Tracked
notebook `rtrl_bptt_scaling_analysis.ipynb` 保存当前图形输出，但不会启动 benchmark。

这些测量支持三项稳定结论：RTRL logical carry 与 $N_xN_\theta$ 一致；BPTT temporal
workspace 随 rollout 长度增长；GPU wall time 还受并行填充、kernel/span、memory traffic 和
静态 shape regime 影响，不能直接由总 work 的大 O 推出。

## 10. 结论与后续边界

当前 solver 和 BrainState functionization 足以实现 exact single-cell forward sensitivity，不需要
修改 DHS 数值算法，也不需要截断时间梯度。下一阶段真正需要评估的是：

- 真实 morphology 下 $N_xN_\theta$ memory；
- 更大 CV/network 下 batched tangent solve 何时进入 compute-bound；
- 是否值得为 read-only/inert state 建立更小的 pure state contract；
- 何时从 exact full sensitivity 转向 block-local、low-rank 或 e-prop-style approximation。

这些结论不自动扩展到 trainable delay、hard spike-time objective 或 recurrent event network。
