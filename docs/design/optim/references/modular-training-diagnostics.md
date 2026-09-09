# 模块化训练诊断与优化恢复

## 文档定位

本文解释观测、归档与 spike-region 的方法，不定义 BrainCell 公共 API，也不引入 Trainer。
方法建议不代表每项均已实现；当前实验能力以 [实验工作流](../current/experimental-workflows.md) 为准。
当前实验实现位于
[`diagnostics.py`](../../../../examples/optim/parameter_fitting/diagnostics.py)。
参数选择与 runtime 映射仍由 `braincell.trainable` 负责，优化器由 BrainTools 或用户代码
负责。

## 历史证据

32-start 基线及已观察的 spike signatures 见
[拟合与可辨识性结果](../current/results/fitting-and-identifiability.md#诊断基线)。

## 可组合观测合同

训练循环保持四个独立角色：

```text
predictions = rollout(protocols)
loss, components = objective(predictions, targets)
metrics = evaluator(predictions, targets)
state = observer.capture(parameters, loss, components, metrics)
update = observer.capture_update(gradients, learning_rate)
```

角色使用普通函数和 immutable history，不要求继承框架基类。替换 objective 不应迫使用户
改动 rollout、evaluator 或 observer。

| 合同 | Shape / 内容 | 约束 |
| --- | --- | --- |
| prediction | `[time, start, probe]` | protocol 可有不同 time/probe 数 |
| target | `[time, probe]` | 与同名 prediction 对齐 |
| evaluator | spike count、signed count error、RMSE、finite | 不参与梯度 |
| state | optimizer roots、physical values、loss、metrics | optimizer update 前捕获 |
| update | optimizer-space gradients、effective LR | 与一次参数位移对应 |

协议名、probe 的物理含义、单位和采样规则进入 metadata，不编码为隐式数组位置。当前
`voltage_mse_objective()` 对协议分别产生 `[start]` MSE 后等权平均，只是最小基线。

硬 spike count 使用统一的上穿规则 `v[t] < threshold` 且 `v[t+1] >= threshold`。每个
protocol 可指定一个 spike probe；voltage RMSE 仍保留全部 probe。

### 时间对齐

训练结束后再评价 endpoint，并由 `finalize_history()` 形成：

```text
state axis:  N + 1  # initial，N 次 update 对应的状态，endpoint
update axis: N      # gradient、LR、state[t] -> state[t + 1] 位移
```

update 前计算的 loss 对应 `parameter_trajectory[t]`，绝不能与更新后的
`parameter_trajectory[t + 1]` 配对。archive、resume 和 plateau controller 都依赖这一
不变量。

### 派生诊断

`TrainingHistory` 应派生 optimizer-space gradient L2 norm、相邻 gradient cosine、step
norm、bound-normalized physical step、bound position、spike region、finite、initial /
best / final loss、best epoch 和可选 parameter error。分类阈值集中保存于
`DiagnosticConfig`；标签只是观测启发式，不证明局部最优。

## Spike Region

固定 protocol 下，spike count 在参数区域内是整数常量，跨兴奋性边界时跳变。
观测过的 signature 示例保存于结果页；下述记录与评价是方法约定。

每次 forward 至少记录：

```text
signature[p]
signed_count_error[p] = signature[p] - target_signature[p]
count_distance = sum(abs(signed_count_error))
count_feasible = all(signed_count_error == 0)
spike_times[p]
finite
composite_loss
component_losses
```

完整 signature 是 region ID；相同 distance 的不同 signature 不能合并。`dt`、threshold
和 refractory 规则共同定义 region，必须写入实验配置。

| Hard 指标用于 | Hard 指标不用于 |
| --- | --- |
| region、archive、候选接受、landscape、成功判定、held-out | 参数梯度、动态裁剪 trace、改变 JAX 输出结构、替代连续/surrogate loss |

梯度始终来自连续或 surrogate component。region 只允许控制下一阶段的固定形状 loss 配置。

## 恢复动作的边界

诊断观测不自动改变训练。plateau 阈值、LR kick、SGDR、perturb、恢复顺序与验收矩阵
属于 [训练恢复提案](../proposals/training-recovery.md)，尚未实现为自动 controller。

## Archive 与模型选择

以下是 archive 与选择的方法约定；当前已实现的是训练后历史提取。
timing tie-break、在线维护及恢复控制不能由这些建议推断为现成 API，具体边界见
[实验工作流](../current/experimental-workflows.md#诊断与历史) 与 [恢复提案](../proposals/training-recovery.md)。

每条 start 维护两个固定 shape archive：

| Archive | 更新条件 | 用途 |
| --- | --- | --- |
| `continuous_best` | finite 且 Composite loss 更低 | 保存连续目标最优状态 |
| `spike_feasible_best` | 所有 protocol signed count error 为零 | 保存满足离散成功条件的最优状态 |

每项保存 valid、epoch、optimizer roots、physical values、loss components 和 metrics。无
finite 状态或从未进入可行区时使用 `valid=False, epoch=-1`，浮点 payload 为 NaN；调用方
必须先检查 valid。continuous-best 不能冒充 spike-feasible 成功。

多个 feasible 候选依次比较 maximum matched spike-time error、Composite loss、aggregate
voltage RMSE、到 bounds 的参数距离。timing 差异小于 `0.025 ms` 视为并列。第一版由
`extract_best_archives(history)` 在训练后逐 start 提取，不回写当前参数，因此不改变训练
轨迹；长训练才考虑在 JAX loop 内在线维护。

held-out 模型选择依次比较：train/held-out finite、held-out signature、held-out timing 与
voltage、train Composite loss、生理先验或不确定性。synthetic 数据同时报告 trace 和
parameter success；真实数据不能把不可见的真参数作为唯一标准。

## Region-aware loss 与可视化

| Region | 连续目标调整 | 搜索约束 |
| --- | --- | --- |
| 缺失 spike | threshold margin、smooth peak/event、target-fixed window | 不硬编码 `gNa`/`gK` 方向 |
| 额外 spike | unmatched-event、late no-event、AHP/steady/late voltage | 检查 rebound，不永久裁掉 peak |
| count 正确 | 增加 timing、ISI、AP shape、AHP、full trace | LR 约 `0.001`，radius 至多 `0.1` |

landscape 图必须显示真实 grid 或 grid size，强制加入 target、initial、checkpoint 和 candidate
坐标，并把 contour 标为 sampled boundary estimate。边界附近的自适应细化要报告额外
forward 数；endpoint-anchored 二维切片中的真值只能标为 `target projection`。

## Artifact 与事件合同

```text
history.npz    optimizer/physical/loss/metric histories and archives
metadata.json  seed, backend, precision, dt, duration, solver, protocols, probes, units
summary.json   per-start classification and aggregate counts
```

每条轨迹还应记录 signature/count-distance history、首次和最后 feasible epoch、entry/exit
次数、region dwell、restart 与 perturb events、effective LR 和额外 forward 数。高分辨率 trace
不按每个 epoch 保存，避免 `time x start x epoch` 膨胀；只保留 initial/final/best 或另设采样
模块。

## 实现与验证入口

已实现的 history 与训练后 archive 见 [实验工作流](../current/experimental-workflows.md#诊断与历史)。
会改变训练的恢复状态、在线 archive、完整 resume 和候选接受规则见提案，不是本页的现行 API。

## References

1. Loshchilov, I. & Hutter, F. *SGDR: Stochastic Gradient Descent with Warm
   Restarts*. ICLR 2017. [arXiv:1608.03983](https://arxiv.org/abs/1608.03983).
2. Wales, D. J. & Doye, J. P. K. *Global Optimization by Basin-Hopping and the
   Lowest Energy Structures of Lennard-Jones Clusters Containing up to 110
   Atoms*. J. Phys. Chem. A 101, 5111-5116 (1997).
   [arXiv:cond-mat/9803344](https://arxiv.org/abs/cond-mat/9803344).
3. Hansen, N. *The CMA Evolution Strategy: A Tutorial*.
   [arXiv:1604.00772](https://arxiv.org/abs/1604.00772).
4. Bengio, Y., Louradour, J., Collobert, R. & Weston, J. *Curriculum Learning*.
   ICML 2009. [PDF](https://icml.cc/2009/papers/119.pdf).
